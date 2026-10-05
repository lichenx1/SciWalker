from __future__ import annotations

from pathlib import Path
from typing import Any
from .context import (
    background_judge_accepts,
    background_over_quota,
    canonicalize_background_candidates,
    canonicalize_chain_potential,
    canonicalize_quality_review,
    quality_dimension_gate_errors,
    quality_judge_accepts,
    selected_background_index,
)
from .llm import (
    LLMClient,
    chat_logged,
    parse_json_from_text,
)
from .prompts import (
    build_background_judge_prompt,
    build_background_selection_prompt,
    build_chain_potential_prompt,
    build_final_polish_prompt,
    build_generation_prompt,
    build_quality_judge_prompt,
    build_quality_repair_prompt,
    build_repair_prompt,
)
from .validation import (
    canonicalize_problem,
    validate_problem,
)


def select_domain_background_with_judge(
    *,
    client: LLMClient,
    chain: dict[str, Any],
    max_rounds: int,
    candidate_count: int,
    selection_temperature: float,
    judge_temperature: float,
    diversity_context: dict[str, Any] | None,
    background_soft_quota: int,
    api_log_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seed_id = str(chain.get("scicode_seed_id") or chain.get("chain_id") or "SCICODE-SEED")
    for round_index in range(max(0, max_rounds)):
        try:
            select_system, select_user = build_background_selection_prompt(
                chain,
                rejected,
                candidate_count=max(1, candidate_count),
                diversity_context=diversity_context or {},
            )
            raw_selection = chat_logged(
                client,
                system_prompt=select_system,
                user_prompt=select_user,
                temperature=selection_temperature,
                max_tokens=_full_context_json_max_tokens(client),
                response_format={"type": "json_object"},
                api_log_dir=api_log_dir,
                call_prefix=f"{seed_id}.background_select{round_index}",
            )
            proposal = parse_json_from_text(raw_selection)
            if not isinstance(proposal, dict):
                raise ValueError("background_selection_non_object_json")
            proposals = canonicalize_background_candidates(proposal)
            if not proposals:
                raise ValueError("background_selection_empty_candidates")
            judge_system, judge_user = build_background_judge_prompt(
                chain, proposals, diversity_context or {}
            )
            raw_judge = chat_logged(
                client,
                system_prompt=judge_system,
                user_prompt=judge_user,
                temperature=judge_temperature,
                max_tokens=_full_context_json_max_tokens(client),
                response_format={"type": "json_object"},
                api_log_dir=api_log_dir,
                call_prefix=f"{seed_id}.background_judge{round_index}",
            )
            judge = parse_json_from_text(raw_judge)
            if not isinstance(judge, dict):
                raise ValueError("background_judge_non_object_json")
            selected_index = selected_background_index(judge, proposals)
            proposal = proposals[selected_index]
            quota_rejected = background_over_quota(
                proposal, diversity_context or {}, background_soft_quota
            )
            accepted = background_judge_accepts(judge) and (not quota_rejected)
            attempt = {
                "round": round_index,
                "proposals": proposals,
                "proposal": proposal,
                "judge": judge,
                "quota_rejected": quota_rejected,
                "accepted": accepted,
            }
            attempts.append(attempt)
            if accepted:
                proposal["_background_judge"] = judge
                proposal["_background_round"] = round_index
                return (proposal, attempts)
            rejected.append({"proposal": proposal, "judge": judge})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"round": round_index, "errors": [error], "accepted": False})
            rejected.append({"error": error})
    return (None, attempts)


def judge_chain_potential(
    *, client: LLMClient, chain: dict[str, Any], temperature: float, api_log_dir: Path
) -> dict[str, Any]:
    seed_id = str(chain.get("scicode_seed_id") or chain.get("chain_id") or "SCICODE-SEED")
    system_prompt, user_prompt = build_chain_potential_prompt(chain)
    raw = chat_logged(
        client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=_full_context_json_max_tokens(client),
        response_format={"type": "json_object"},
        api_log_dir=api_log_dir,
        call_prefix=f"{seed_id}.chain_potential",
    )
    parsed = parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        raise ValueError("chain_potential_non_object_json")
    return canonicalize_chain_potential(parsed)


def _full_context_json_max_tokens(client: LLMClient) -> int:
    return client.config.max_tokens


def generate_with_repairs(
    *,
    client: LLMClient,
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    max_repair_rounds: int,
    temperature: float,
    validation_timeout: int,
    validation_python: str,
    api_log_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    previous_problem: dict[str, Any] | None = None
    previous_errors: list[str] = []
    seed_id = str(chain.get("scicode_seed_id") or chain.get("chain_id") or "SCICODE-SEED")
    for round_index in range(max_repair_rounds + 1):
        if round_index == 0:
            system_prompt, user_prompt = build_generation_prompt(chain, selected_background)
        else:
            system_prompt, user_prompt = build_repair_prompt(
                chain, selected_background, previous_problem, previous_errors
            )
        try:
            print(f"  - {seed_id} attempt{round_index}: requesting model", flush=True)
            raw = chat_logged(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=_full_context_json_max_tokens(client),
                response_format={"type": "json_object"},
                api_log_dir=api_log_dir,
                call_prefix=f"{seed_id}.attempt{round_index}",
            )
            print(f"  - {seed_id} attempt{round_index}: parsing and validating", flush=True)
            parsed = parse_json_from_text(raw)
            if not isinstance(parsed, dict):
                raise ValueError("model_returned_non_object_json")
            problem = canonicalize_problem(parsed, chain)
            errors = validate_problem(problem, chain, validation_timeout, validation_python)
            attempts.append({"round": round_index, "errors": errors})
            if not errors:
                print(
                    f"  - {seed_id} attempt{round_index}: accepted by executable checks", flush=True
                )
                return (problem, attempts)
            previous_problem = problem
            previous_errors = errors
            print(
                f"  - {seed_id} attempt{round_index}: {len(errors)} validation errors", flush=True
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"round": round_index, "errors": [error]})
            previous_errors = [error]
            print(f"  - {seed_id} attempt{round_index}: exception {error}", flush=True)
    return (None, attempts)


def polish_problem_after_validation(
    *,
    client: LLMClient,
    problem: dict[str, Any],
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    temperature: float,
    validation_timeout: int,
    validation_python: str,
    api_log_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    seed_id = str(chain.get("scicode_seed_id") or chain.get("chain_id") or "SCICODE-SEED")
    attempts: list[dict[str, Any]] = []
    try:
        system_prompt, user_prompt = build_final_polish_prompt(
            chain=chain, selected_background=selected_background, problem=problem
        )
        raw = chat_logged(
            client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=_full_context_json_max_tokens(client),
            response_format={"type": "json_object"},
            api_log_dir=api_log_dir,
            call_prefix=f"{seed_id}.final_polish0",
        )
        parsed = parse_json_from_text(raw)
        if not isinstance(parsed, dict):
            raise ValueError("final_polish_returned_non_object_json")
        polished = canonicalize_problem(parsed, chain)
        errors = validate_problem(polished, chain, validation_timeout, validation_python)
        attempts.append({"round": 0, "accepted": not errors, "errors": errors})
        if errors:
            return (None, attempts)
        return (polished, attempts)
    except Exception as exc:
        attempts.append({"round": 0, "accepted": False, "errors": [f"{type(exc).__name__}: {exc}"]})
        return (None, attempts)


def quality_gate_problem(
    *,
    client: LLMClient,
    problem: dict[str, Any],
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    min_grade: str,
    repair_rounds: int,
    judge_temperature: float,
    repair_temperature: float,
    validation_timeout: int,
    validation_python: str,
    api_log_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    attempts: list[dict[str, Any]] = []
    current_problem = problem
    last_review: dict[str, Any] | None = None
    seed_id = str(chain.get("scicode_seed_id") or chain.get("chain_id") or "SCICODE-SEED")
    for round_index in range(max(0, repair_rounds) + 1):
        if round_index > 0:
            try:
                system_prompt, user_prompt = build_quality_repair_prompt(
                    chain=chain,
                    selected_background=selected_background,
                    problem=current_problem,
                    review=last_review or {},
                    # The production runs did not enable the separate semantic judge.
                    semantic_review={},
                    min_grade=min_grade,
                )
                raw_repair = chat_logged(
                    client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=repair_temperature,
                    max_tokens=_full_context_json_max_tokens(client),
                    response_format={"type": "json_object"},
                    api_log_dir=api_log_dir,
                    call_prefix=f"{seed_id}.quality_repair{round_index}",
                )
                parsed = parse_json_from_text(raw_repair)
                if not isinstance(parsed, dict):
                    raise ValueError("quality_repair_returned_non_object_json")
                repaired = canonicalize_problem(parsed, chain)
                validation_errors = validate_problem(
                    repaired, chain, validation_timeout, validation_python
                )
                attempts.append(
                    {
                        "round": round_index,
                        "stage": "quality_repair_validation",
                        "errors": validation_errors,
                    }
                )
                if validation_errors:
                    current_problem = repaired
                    continue
                current_problem = repaired
            except Exception as exc:
                attempts.append(
                    {
                        "round": round_index,
                        "stage": "quality_repair_exception",
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    }
                )
                continue
        try:
            review = judge_problem_quality(
                client=client,
                problem=current_problem,
                chain=chain,
                selected_background=selected_background,
                min_grade=min_grade,
                temperature=judge_temperature,
                api_log_dir=api_log_dir,
                call_prefix=f"{seed_id}.quality_judge{round_index}",
            )
            last_review = review
            accepted = quality_judge_accepts(review, min_grade)
            dimension_gate_errors = quality_dimension_gate_errors(review, min_grade)
            attempts.append(
                {
                    "round": round_index,
                    "stage": "quality_judge",
                    "accepted": accepted,
                    "review": review,
                    "dimension_gate_errors": dimension_gate_errors,
                }
            )
            if accepted:
                current_problem["_quality_review"] = review
                current_problem["_quality_min_grade"] = min_grade
                return (current_problem, attempts, current_problem)
        except Exception as exc:
            attempts.append(
                {
                    "round": round_index,
                    "stage": "quality_judge_exception",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            )
    if last_review is not None:
        current_problem["_quality_review"] = last_review
    return (None, attempts, current_problem)


def judge_problem_quality(
    *,
    client: LLMClient,
    problem: dict[str, Any],
    chain: dict[str, Any],
    selected_background: dict[str, Any] | None,
    min_grade: str,
    temperature: float,
    api_log_dir: Path,
    call_prefix: str,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_quality_judge_prompt(
        problem=problem, chain=chain, selected_background=selected_background, min_grade=min_grade
    )
    raw = chat_logged(
        client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=_full_context_json_max_tokens(client),
        response_format={"type": "json_object"},
        api_log_dir=api_log_dir,
        call_prefix=call_prefix,
    )
    parsed = parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        raise ValueError("quality_judge_returned_non_object_json")
    return canonicalize_quality_review(parsed)
