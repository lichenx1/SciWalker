from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
import datetime as dt
import json
from pathlib import Path
import sys
import threading
from typing import Any

from . import stages
from .chain_io import SavedChainSequence
from .context import chain_potential_accepts, empty_diversity_state, update_diversity_state
from .llm import LLMClient, LLMConfig, ModelError
from .paths import catalog_root
from .sampling import LiveSampledChainSequence, load_subdomain_weights, parse_chain_lengths


@dataclass(frozen=True)
class GenerationConfig:
    max_repair_rounds: int = 8
    quality_repair_rounds: int = 1
    quality_min_grade: str = "A"
    validation_timeout: int = 45
    validation_python: str = sys.executable
    temperature: float = 0.4
    background_max_rounds: int = 3
    background_candidate_count: int = 1
    background_soft_quota: int = 0
    chain_potential_min_score: float = 5.0
    workflow_workers: int = 8

    def __post_init__(self) -> None:
        for name in [
            "validation_timeout",
            "background_max_rounds",
            "background_candidate_count",
            "workflow_workers",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ["max_repair_rounds", "quality_repair_rounds", "background_soft_quota"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.quality_min_grade not in {"A", "S"}:
            raise ValueError("quality_min_grade must be A or S")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def generate_one(
    client: LLMClient, chain: dict, config: GenerationConfig, api_log_dir: Path, diversity: dict
) -> tuple[dict | None, dict]:
    """Run generation with the historical stage retries and pre-polish recovery."""
    report = {
        "seed_id": chain["scicode_seed_id"],
        "subdomain_slug": chain["subdomain_slug"],
        "chain_length": chain["length"],
        "source_chain": chain,
        "passed": False,
        "included": False,
        "stage": "chain_potential",
    }
    common = {"client": client, "chain": chain, "api_log_dir": api_log_dir}
    try:
        try:
            potential = stages.judge_chain_potential(**common, temperature=0.0)
            report["chain_potential"] = potential
            report["chain_potential_passed"] = chain_potential_accepts(
                potential, config.chain_potential_min_score
            )
            if not report["chain_potential_passed"]:
                return None, report
        except Exception as exc:
            # The production workflow continued after an unavailable potential review.
            report["chain_potential_passed"] = False
            report["chain_potential"] = {
                "verdict": "uncertain",
                "score": 0,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        report["stage"] = "background_selection"
        background, attempts = stages.select_domain_background_with_judge(
            **common,
            max_rounds=config.background_max_rounds,
            candidate_count=config.background_candidate_count,
            selection_temperature=0.6,
            judge_temperature=0.0,
            diversity_context=diversity,
            background_soft_quota=config.background_soft_quota,
        )
        report["background_attempts"] = attempts
        if background is None:
            return None, report
        report["selected_background"] = background
        candidate_args = {
            **common,
            "selected_background": background,
            "validation_timeout": config.validation_timeout,
            "validation_python": config.validation_python,
        }
        report["stage"] = "generation"
        problem, attempts = stages.generate_with_repairs(
            **candidate_args,
            max_repair_rounds=config.max_repair_rounds,
            temperature=config.temperature,
        )
        report["attempts"] = attempts
        if problem is None:
            return None, report
        report["stage"] = "final_polish"
        polished_problem, attempts = stages.polish_problem_after_validation(
            **candidate_args, problem=problem, temperature=0.2
        )
        report["polish_attempts"] = attempts
        if polished_problem is not None:
            problem = polished_problem
        source_metadata = {
            "seed_id": chain["scicode_seed_id"],
            "length": chain["length"],
            "operator_ids": chain["operator_ids"],
            "subdomain": chain["subdomain"],
            "subdomain_slug": chain["subdomain_slug"],
            "domain_scenario": chain["domain_scenario"],
        }
        # Quality repair receives the same provenance fields as the production prompt.
        problem["_source_chain"] = source_metadata
        problem["_selected_domain_background"] = background
        problem["_selected_physical_background"] = background
        report["stage"] = "quality_gate"
        problem, attempts, _ = stages.quality_gate_problem(
            **candidate_args,
            problem=problem,
            min_grade=config.quality_min_grade,
            repair_rounds=config.quality_repair_rounds,
            judge_temperature=0.0,
            repair_temperature=config.temperature,
        )
        report["quality_attempts"] = attempts
        if problem is None:
            return None, report
        problem["_source_chain"] = source_metadata
        problem["_selected_domain_background"] = background
        problem["_selected_physical_background"] = background
        report.update(passed=True, included=True, stage="accepted")
        return problem, report
    except (ModelError, ValueError, TypeError, KeyError) as exc:
        report["errors"] = [f"{type(exc).__name__}: {exc}"]
        return None, report


def run_generation(
    client: LLMClient,
    chains: LiveSampledChainSequence | SavedChainSequence,
    output_dir: Path,
    config: GenerationConfig,
) -> dict:
    if len(chains) == 0:
        raise ValueError("No chains were scheduled")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    api_logs = output_dir / "api_calls"
    api_logs.mkdir()
    (output_dir / "scicode_style_problems.jsonl").touch()
    manifest = {
        "framework_version": version("scidata-generation"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sampling": chains.sampling_summary(),
        "generation": asdict(config),
        "model": {key: value for key, value in asdict(client.config).items() if key != "api_key"},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    state_lock = threading.Lock()
    stop = threading.Event()
    diversity = empty_diversity_state()
    counts = Counter()
    next_index = 0

    def worker() -> None:
        nonlocal next_index
        while not stop.is_set():
            with state_lock:
                if next_index >= len(chains) or stop.is_set():
                    return
                index = next_index
                next_index += 1
                snapshot = json.loads(json.dumps(diversity))
            try:
                chain = chains[index]
                problem, report = generate_one(client, chain, config, api_logs, snapshot)
            except Exception as exc:
                # Record unexpected workflow failures and stop; never accept a partial result.
                problem = None
                report = {
                    "chain_index": index,
                    "passed": False,
                    "included": False,
                    "stage": "workflow_error",
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                }
            with state_lock:
                counts[report["stage"]] += 1
                if report.get("fatal_error"):
                    counts["fatal_errors"] += 1
                    stop.set()
                report["completion_index"] = sum(
                    v for k, v in counts.items() if k != "fatal_errors"
                )
                append_jsonl(output_dir / "validation_report.jsonl", report)
                if problem is not None:
                    append_jsonl(output_dir / "scicode_style_problems.jsonl", problem)
                    update_diversity_state(diversity, problem)
            print(f"[{index + 1}/{len(chains)}] {report['stage']}", flush=True)

    with ThreadPoolExecutor(max_workers=min(len(chains), config.workflow_workers)) as executor:
        futures = [
            executor.submit(worker) for _ in range(min(len(chains), config.workflow_workers))
        ]
        for future in futures:
            future.result()
    completion = {
        "scheduled_chains": len(chains),
        "started_chains": next_index,
        "accepted_problems": counts["accepted"],
        "stage_counts": dict(counts),
        "fatal_errors": counts["fatal_errors"],
    }
    (output_dir / "run_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate scientific coding problems from operator graphs."
    )
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--graph-root",
        type=Path,
        default=catalog_root() / "operator_graphs_runtime_verified",
    )
    source.add_argument("--chains-file", type=Path, help="Use saved chains without resampling")
    parser.add_argument("--chain-manifest", type=Path)
    parser.add_argument("--subdomain", action="append", default=[])
    parser.add_argument("--chain-lengths", default="3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--chains-per-subdomain-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026071001)
    parser.add_argument("--subdomain-weights", type=Path)
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximum scheduled chains; 0 uses the full schedule."
    )
    parser.add_argument("--workflow-workers", type=int, default=8)
    parser.add_argument("--validation-python", default=sys.executable)
    parser.add_argument("--validation-timeout", type=int, default=45)
    parser.add_argument("--max-repair-rounds", type=int, default=8)
    parser.add_argument("--quality-repair-rounds", type=int, default=1)
    parser.add_argument("--quality-min-grade", choices=["A", "S"], default="A")
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.chain_manifest and not args.chains_file:
        parser.error("--chain-manifest requires --chains-file")
    sampling_flags = {
        "--subdomain",
        "--chain-lengths",
        "--chains-per-subdomain-length",
        "--seed",
        "--subdomain-weights",
    }
    supplied_flags = {arg.split("=", 1)[0] for arg in (argv if argv is not None else sys.argv[1:])}
    if args.chains_file and sampling_flags.intersection(supplied_flags):
        parser.error("Sampling flags apply to --graph-root; saved chains are already sampled")
    config = GenerationConfig(
        **{
            name: getattr(args, name)
            for name in [
                "workflow_workers",
                "validation_python",
                "validation_timeout",
                "max_repair_rounds",
                "quality_repair_rounds",
                "quality_min_grade",
                "temperature",
            ]
        }
    )
    if args.chains_file:
        chains = SavedChainSequence(args.chains_file, args.chain_manifest, args.limit)
    else:
        chains = LiveSampledChainSequence(
            graph_root=args.graph_root,
            subdomains=args.subdomain,
            chain_lengths=parse_chain_lengths(args.chain_lengths),
            chains_per_subdomain_length=args.chains_per_subdomain_length,
            seed=args.seed,
            max_attempts=200,
            generation_order="round_robin",
            subdomain_weights=load_subdomain_weights(args.subdomain_weights),
            limit=args.limit,
        )
    client = LLMClient(LLMConfig.from_file(args.api_config))
    result = run_generation(client, chains, args.output_dir, config)
    print(json.dumps(result, indent=2))
    return 1 if result["fatal_errors"] else (0 if result["accepted_problems"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
