from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from http.client import IncompleteRead
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ModelError(RuntimeError):
    """A model-service or protocol error recorded by the active generation stage."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    backend: str = "chat_completions"
    timeout_sec: int = 1800
    max_tokens: int = 32768
    parallelism: int = 4
    retries: int = 2
    json_response_format: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    tokenizer_path: str | None = None
    context_window: int | None = None
    token_budget_margin: int = 1000
    reasoning_effort: str = "max"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.api_key, str)
            or self.api_key.strip() in {"", "YOUR_API_KEY"}
            or "\n" in self.api_key
            or "\r" in self.api_key
        ):
            raise ValueError(
                "Set api_key to your service key in the LLM configuration; line breaks are not allowed"
            )
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be a complete HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "Use api_key for credentials; base_url cannot contain credentials, query, or fragment"
            )
        if not self.model.strip():
            raise ValueError("model is required")
        if self.backend not in {"chat_completions", "raw_dsv4"}:
            raise ValueError("backend must be chat_completions or raw_dsv4")
        for name in ["timeout_sec", "max_tokens", "parallelism"]:
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.retries, int) or not 0 <= self.retries <= 5:
            raise ValueError("retries must be an integer between 0 and 5")
        if self.token_budget_margin < 0:
            raise ValueError("token_budget_margin must be nonnegative")
        if self.context_window is not None and self.context_window <= self.token_budget_margin:
            raise ValueError("context_window must exceed token_budget_margin")
        if self.context_window is not None and not self.tokenizer_path:
            raise ValueError("context_window requires tokenizer_path for exact token counting")
        if self.backend == "raw_dsv4" and not self.tokenizer_path:
            raise ValueError(
                "raw_dsv4 requires tokenizer_path containing encoding/encoding_dsv4.py"
            )
        protected = {
            "model",
            "messages",
            "prompt",
            "temperature",
            "max_tokens",
            "response_format",
            "stream",
            "n",
        }
        if not isinstance(self.extra_body, dict) or protected.intersection(self.extra_body):
            raise ValueError("extra_body must be an object and cannot override core request fields")

    @classmethod
    def from_file(cls, path: str | Path) -> LLMConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("The model configuration must be a JSON object")
        return cls(**payload)

class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._semaphore = threading.BoundedSemaphore(config.parallelism)
        self._tokenizer = None
        self._encoder = None
        if config.backend == "raw_dsv4":
            model_path = Path(config.tokenizer_path).expanduser().resolve()
            path = model_path / "encoding/encoding_dsv4.py"
            if not path.is_file():
                raise FileNotFoundError(f"Required DeepSeek encoder is missing: {path}")
            spec = importlib.util.spec_from_file_location("_scidata_dsv4_encoder", path)
            if spec is None or spec.loader is None:
                raise ModelError(f"Cannot load encoder: {path}")
            # The upstream encoder may import sibling modules from its own bundle.
            sys.path.insert(0, str(path.parent))
            try:
                self._encoder = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self._encoder)
            finally:
                sys.path.remove(str(path.parent))
        if config.context_window is not None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path, local_files_only=True, trust_remote_code=True
            )

    def _request_body(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        body = {"model": self.config.model, "temperature": temperature, **self.config.extra_body}
        if self.config.backend == "raw_dsv4":
            if response_format:
                messages[-1]["content"] += (
                    "\n\nReturn only one valid JSON object. Do not include markdown fences, comments, or explanatory prose outside the JSON object."
                )
            prompt = self._encoder.encode_messages(
                messages, thinking_mode="thinking", reasoning_effort=self.config.reasoning_effort
            )
            body.update(prompt=str(prompt), top_p=1.0)
            endpoint = "completions"
            if self._tokenizer is not None:
                prompt_tokens = len(self._tokenizer.encode(str(prompt), add_special_tokens=False))
        else:
            body["messages"] = messages
            if response_format and self.config.json_response_format:
                body["response_format"] = response_format
            endpoint = "chat/completions"
            if self._tokenizer is not None:
                prompt_tokens = len(
                    self._tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
        budget = min(max_tokens, self.config.max_tokens)
        if self._tokenizer is not None:
            budget = min(
                budget, self.config.context_window - prompt_tokens - self.config.token_budget_margin
            )
        if budget <= 0:
            raise ModelError("The prompt exceeds the configured context window")
        body["max_tokens"] = budget
        return self.config.base_url.rstrip("/") + "/" + endpoint, body

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        api_log_dir: Path,
        call_prefix: str,
    ) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", call_prefix):
            raise ValueError("call_prefix must be a safe filename")
        url, body = self._request_body(
            system_prompt, user_prompt, temperature, max_tokens, response_format
        )
        headers = {"Content-Type": "application/json", "User-Agent": "scidata/0.1"}
        key = self.config.api_key.strip()
        if key:
            headers["Authorization"] = "Bearer " + key
        api_log_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.config.retries + 1):
            record = {
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "backend": self.config.backend,
                "url": url,
                "request": body,
                "attempt": attempt,
                "response": None,
                "error": None,
            }
            start = time.monotonic()
            retry = False
            status = None
            try:
                request = Request(
                    url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
                )
                with self._semaphore:
                    with urlopen(request, timeout=self.config.timeout_sec) as response:
                        status = response.status
                        data = json.loads(response.read().decode("utf-8"))
                record["response"] = data
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ModelError("The model output was truncated at the token limit")
                if self.config.backend == "raw_dsv4":
                    parsed = self._encoder.parse_message_from_completion_text(
                        choice["text"], thinking_mode="thinking"
                    )
                    content = parsed["content"]
                else:
                    content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ModelError("The model response has no nonempty string content")
                record["content"] = content
                return content
            except HTTPError as exc:
                status = exc.code
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except (TimeoutError, ConnectionError, IncompleteRead) as read_error:
                    detail = f"{type(read_error).__name__}: {read_error}"
                if key:
                    detail = detail.replace(key, "[REDACTED]")
                record["error"] = f"HTTP {status}: {detail[:4000]}"
                retry = status in {408, 429, 500, 502, 503, 504}
            except (URLError, TimeoutError, socket.timeout, ConnectionError, IncompleteRead) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                retry = True
            except (KeyError, IndexError, TypeError, ValueError, ModelError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                record["status"] = status
                record["elapsed_seconds"] = round(time.monotonic() - start, 6)
                # Authorization headers and API keys are never recorded.
                (api_log_dir / f"{call_prefix}.http{attempt:02d}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            if not retry or attempt == self.config.retries:
                raise ModelError(str(record["error"]))
            time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


def chat_logged(client: LLMClient, **kwargs: Any) -> str:
    return client.complete(**kwargs)


def parse_json_from_text(text: str) -> Any:
    """Require valid JSON; malformed answers enter the explicit repair stage."""
    return json.loads(text)
