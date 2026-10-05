from __future__ import annotations

from typing import Any
import re
import math
from .rules import (
    ALLOWED_GENERIC_OPERATOR_WORDS,
    ALLOWED_VISIBLE_LIBRARY_REFERENCES,
    DIRECT_LIBRARY_API_VISIBLE_PATTERNS,
    QUALITY_DIMENSION_MIN_SCORES,
    QUALITY_GRADE_ORDER,
    SUBDOMAIN_BACKGROUND_CATALOG,
)


def hidden_operator_term_variants(simple_name: str) -> list[str]:
    simple = simple_name.strip().lower()
    if not simple or simple in ALLOWED_GENERIC_OPERATOR_WORDS:
        return []
    variants = [simple]
    if "_" in simple:
        variants.extend([simple.replace("_", " "), simple.replace("_", "-")])
    return sorted(set(variants), key=lambda item: (len(item), item), reverse=True)


def visible_direct_library_api_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in DIRECT_LIBRARY_API_VISIBLE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.I):
            hit = re.sub(r"\s+", " ", match.group(0)).strip("` \t\r\n.,;:")
            normalized = hit.lower()
            if normalized in ALLOWED_VISIBLE_LIBRARY_REFERENCES:
                continue
            hits.append(hit)
    return sorted(set(hits), key=lambda item: (len(item), item), reverse=True)


def hidden_operator_terms_to_avoid(chain: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for op_id in chain.get("operator_ids", []):
        simple = str(op_id).split(".")[-1].lower()
        terms.extend(hidden_operator_term_variants(simple))
    for node in chain.get("operator_nodes", []) or []:
        if not isinstance(node, dict):
            continue
        terms.extend(split_symbol_terms(node.get("api_symbol")))
        library = str(node.get("library") or "").strip().lower()
        if library and library not in ALLOWED_GENERIC_OPERATOR_WORDS:
            terms.append(library)
    for edge in chain.get("edge_trace", []) or []:
        if not isinstance(edge, dict):
            continue
        for key in ["source_api_symbol", "target_api_symbol", "source_library", "target_library"]:
            if key.endswith("_library"):
                library = str(edge.get(key) or "").strip().lower()
                if library and library not in ALLOWED_GENERIC_OPERATOR_WORDS:
                    terms.append(library)
            else:
                terms.extend(split_symbol_terms(edge.get(key)))
    return sorted(set(terms), key=lambda item: (len(item), item), reverse=True)


def split_symbol_terms(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    raw_parts = re.split(r"[^0-9A-Za-z_]+", text)
    parts: list[str] = []
    for raw in raw_parts:
        if not raw:
            continue
        pieces = [raw]
        if "_" in raw:
            pieces.extend(raw.split("_"))
        for piece in pieces:
            piece = piece.strip().lower()
            if len(piece) < 4 or piece in ALLOWED_GENERIC_OPERATOR_WORDS:
                continue
            if piece.isdigit():
                continue
            parts.extend(hidden_operator_term_variants(piece))
    return parts


def compact_operator_nodes_for_prompt(
    chain: dict[str, Any], *, limit: int = 5
) -> list[dict[str, Any]]:
    nodes = chain.get("operator_nodes", [])
    if not isinstance(nodes, list):
        return []
    compact: list[dict[str, Any]] = []
    for node in nodes[:limit]:
        if not isinstance(node, dict):
            continue
        compact.append(
            {
                "id": node.get("id"),
                "library": node.get("library"),
                "api_symbol": node.get("api_symbol"),
                "source_operator": node.get("source_operator"),
                "operation_mode": node.get("operation_mode"),
                "inputs": node.get("inputs", [])[:4]
                if isinstance(node.get("inputs"), list)
                else node.get("inputs"),
                "outputs": node.get("outputs", [])[:4]
                if isinstance(node.get("outputs"), list)
                else node.get("outputs"),
                "input_types": node.get("input_types", [])[:8]
                if isinstance(node.get("input_types"), list)
                else node.get("input_types"),
                "output_types": node.get("output_types", [])[:8]
                if isinstance(node.get("output_types"), list)
                else node.get("output_types"),
            }
        )
    return compact


def compact_edge_trace_for_prompt(chain: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    edges = chain.get("edge_trace", [])
    if not isinstance(edges, list):
        return []
    compact: list[dict[str, Any]] = []
    for edge in edges[:limit]:
        if not isinstance(edge, dict):
            continue
        compact.append(
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "matched_input": edge.get("matched_input"),
                "matched_output": edge.get("matched_output"),
                "matched_types": edge.get("matched_types"),
                "matched_tokens": edge.get("matched_tokens"),
                "score": edge.get("score"),
                "workflow_reasons": edge.get("workflow_reasons"),
            }
        )
    return compact


def chain_context_for_prompt(chain: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed_id": chain.get("scicode_seed_id"),
        "chain_id": chain.get("chain_id"),
        "domain": chain.get("domain"),
        "subdomain": chain.get("subdomain"),
        "subdomain_slug": chain.get("subdomain_slug"),
        "length": chain.get("length") or len(chain.get("operator_ids", [])),
        "operator_ids": chain.get("operator_ids", []),
        "operator_nodes": compact_operator_nodes_for_prompt(chain),
        "edge_trace": compact_edge_trace_for_prompt(chain),
        "operator_runtime_env_id": chain.get("operator_runtime_env_id") or "",
        "operator_runtime_python": chain.get("operator_runtime_python") or "",
        "operator_runtime_usage_doc": chain.get("operator_runtime_usage_doc") or "",
        "hidden_terms_to_avoid_in_student_problem": hidden_operator_terms_to_avoid(chain),
        "domain_scenario": chain.get("domain_scenario")
        or chain.get("physical_scenario")
        or chain.get("verdict", {}).get("physical_scenario")
        or "",
        "reason": chain.get("reason") or chain.get("verdict", {}).get("reason") or "",
        "verification_hints": chain.get("verification_hints")
        or chain.get("verdict", {}).get("verification_hints")
        or [],
        "implementation_risks": chain.get("implementation_risks")
        or chain.get("verdict", {}).get("implementation_risks")
        or [],
        "chain_potential_review": chain.get("chain_potential_review") or {},
        "generation_strategy": chain.get("generation_strategy") or "",
        "base_chain_seed_id": chain.get("base_scicode_seed_id") or "",
        "generation_variant_index": chain.get("generation_variant_index"),
        "generation_variant_total": chain.get("generation_variant_total"),
        "variant_instruction": chain.get("variant_instruction") or "",
    }


def contains_hidden_operator_term(text: str, term: str) -> bool:
    lowered = text.lower()
    normalized = term.lower().strip()
    if not normalized:
        return False
    return bool(re.search(rf"(?<![0-9a-z]){re.escape(normalized)}(?![0-9a-z])", lowered))


def canonicalize_background_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    item = dict(proposal)
    for key in [
        "background_id",
        "background_title",
        "background_domain",
        "domain_story",
        "observable_to_compute",
        "why_natural",
    ]:
        item[key] = str(item.get(key, "")).strip()
    if not item["domain_story"]:
        item["domain_story"] = str(item.get("physical_story", "")).strip()
    item["physical_story"] = item["domain_story"]
    inputs = item.get("input_quantities", [])
    if not isinstance(inputs, list):
        inputs = [str(inputs)]
    item["input_quantities"] = [str(x).strip() for x in inputs if str(x).strip()]
    mapping = item.get("operator_role_mapping", [])
    if not isinstance(mapping, list):
        mapping = []
    normalized_mapping: list[dict[str, str]] = []
    for entry in mapping:
        if not isinstance(entry, dict):
            continue
        normalized_mapping.append(
            {
                "operator_id": str(entry.get("operator_id", "")).strip(),
                "scientific_role": str(entry.get("scientific_role", "")).strip(),
            }
        )
    item["operator_role_mapping"] = normalized_mapping
    avoid = item.get("student_visible_terms_to_avoid", [])
    if not isinstance(avoid, list):
        avoid = [str(avoid)]
    item["student_visible_terms_to_avoid"] = [str(x).strip() for x in avoid if str(x).strip()]
    return item


def canonicalize_background_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [
            canonicalize_background_proposal(item) for item in candidates if isinstance(item, dict)
        ]
    return [canonicalize_background_proposal(payload)]


def selected_background_index(judge: dict[str, Any], proposals: list[dict[str, Any]]) -> int:
    index = judge.get("selected_index")
    if type(index) is not int or not 0 <= index < len(proposals):
        raise ValueError("background_judge_invalid_selected_index")
    return index


def background_over_quota(
    proposal: dict[str, Any],
    diversity_context: dict[str, Any],
    quota: int,
) -> bool:
    if quota <= 0:
        return False
    background_id = str(proposal.get("background_id", "")).strip()
    counts = diversity_context.get("background_counts", {})
    if not isinstance(counts, dict) or not background_id:
        return False
    return int(counts.get(background_id, 0) or 0) >= quota


def summarize_diversity_context(context: dict[str, Any], *, limit: int = 12) -> dict[str, Any]:
    background_counts = context.get("background_counts", {})
    prefix_counts = context.get("problem_prefix_counts", {})
    observables = context.get("observables", [])
    names = context.get("problem_names", [])
    if not isinstance(background_counts, dict):
        background_counts = {}
    if not isinstance(prefix_counts, dict):
        prefix_counts = {}
    if not isinstance(observables, list):
        observables = []
    if not isinstance(names, list):
        names = []
    top_backgrounds = sorted(
        background_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:limit]
    top_prefixes = sorted(
        prefix_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:limit]
    return {
        "overused_background_ids": [
            {"background_id": str(k), "count": int(v)} for k, v in top_backgrounds
        ],
        "overused_problem_prefixes": [{"prefix": str(k), "count": int(v)} for k, v in top_prefixes],
        "recent_problem_names": [str(x) for x in names[-limit:]],
        "recent_observables": [str(x) for x in observables[-limit:]],
    }


def background_judge_accepts(judge: dict[str, Any]) -> bool:
    score = review_score(judge.get("score"))
    return judge.get("verdict") == "accept" and judge.get("is_forced") is False and score >= 4


def coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("Review flags must be JSON booleans")
    return value


def canonicalize_chain_potential(review: dict[str, Any]) -> dict[str, Any]:
    item = dict(review)
    if item.get("verdict") not in {"accept", "reject", "uncertain"}:
        raise ValueError("Invalid chain-potential verdict")
    item["score"] = review_score(item.get("score"))
    if item.get("likely_grade_ceiling") not in QUALITY_GRADE_ORDER:
        raise ValueError("Invalid chain-potential grade ceiling")
    return item


def chain_potential_accepts(review: dict[str, Any], min_score: float) -> bool:
    verdict = str(review.get("verdict", "")).strip().lower()
    score = float(review.get("score", 0) or 0)
    ceiling = str(review.get("likely_grade_ceiling", "")).strip().upper()[:1]
    if verdict == "reject":
        return False
    if ceiling in {"C", "D"}:
        return False
    if score < float(min_score):
        return False
    return verdict in {"accept", "uncertain"} or ceiling in {"A", "S"}


def canonicalize_quality_review(review: dict[str, Any]) -> dict[str, Any]:
    item = dict(review)
    if item.get("grade") not in QUALITY_GRADE_ORDER:
        raise ValueError("Invalid quality grade")
    item["accept"] = coerce_bool(item.get("accept"), default=False)
    item["score_1_to_10"] = review_score(item.get("score_1_to_10"))
    for key in QUALITY_DIMENSION_MIN_SCORES:
        item[key] = review_score(item.get(key))
    return item


def quality_judge_accepts(review: dict[str, Any], min_grade: str) -> bool:
    grade = str(review.get("grade", "")).strip().upper()[:1]
    min_grade = str(min_grade).strip().upper()[:1]
    if grade not in QUALITY_GRADE_ORDER or min_grade not in QUALITY_GRADE_ORDER:
        return False
    accept = coerce_bool(review.get("accept"), default=False)
    if not accept or QUALITY_GRADE_ORDER[grade] < QUALITY_GRADE_ORDER[min_grade]:
        return False
    if quality_dimension_gate_errors(review, min_grade):
        return False
    return True


def quality_dimension_gate_errors(review: dict[str, Any], min_grade: str) -> list[str]:
    if min_grade not in {"A", "S"}:
        return ["invalid_quality_min_grade"]
    errors = []
    for key, minimum in QUALITY_DIMENSION_MIN_SCORES.items():
        value = review_score(review.get(key))
        if value < minimum:
            errors.append(f"{key}_below_threshold:{value:g}<{minimum:g}")
    return errors


def empty_diversity_state() -> dict[str, Any]:
    return {
        "background_counts": {},
        "problem_prefix_counts": {},
        "problem_names": [],
        "observables": [],
    }


def background_options_for_chain(chain: dict[str, Any]) -> list[dict[str, str]]:
    slug = str(chain.get("subdomain_slug") or "").strip().lower()
    options = SUBDOMAIN_BACKGROUND_CATALOG.get(slug)
    if options:
        return options
    subdomain = str(chain.get("subdomain") or slug or "the selected subdomain").strip()
    safe_slug = re.sub(r"[^a-z0-9]+", "_", subdomain.lower()).strip("_") or "domain"
    return [
        {
            "id": f"{safe_slug}_measurement_diagnostics",
            "title": f"{subdomain} measurement diagnostics",
            "description": f"Choose concrete {subdomain} inputs, intermediate computations, and a measurable output.",
        },
        {
            "id": f"{safe_slug}_model_calibration",
            "title": f"{subdomain} model calibration",
            "description": f"Fit or calibrate a small {subdomain} model with residual, sensitivity, and validation checks.",
        },
        {
            "id": f"{safe_slug}_screening_metric",
            "title": f"{subdomain} screening metric",
            "description": f"Compute a ranked diagnostic or quality metric that is natural for {subdomain}.",
        },
    ]


def update_diversity_state(state: dict[str, Any], problem: dict[str, Any]) -> None:
    bg = problem.get("_selected_domain_background") or problem.get("_selected_physical_background")
    background_id = ""
    observable = ""
    if isinstance(bg, dict):
        background_id = str(bg.get("background_id", "")).strip()
        observable = str(bg.get("observable_to_compute", "")).strip()
    if background_id:
        counts = state.setdefault("background_counts", {})
        counts[background_id] = int(counts.get(background_id, 0) or 0) + 1
    name = str(problem.get("problem_name", "")).strip()
    if name:
        names = state.setdefault("problem_names", [])
        names.append(name)
        del names[:-80]
        prefix = problem_prefix(name)
        if prefix:
            prefix_counts = state.setdefault("problem_prefix_counts", {})
            prefix_counts[prefix] = int(prefix_counts.get(prefix, 0) or 0) + 1
    if observable:
        observables = state.setdefault("observables", [])
        observables.append(observable)
        del observables[:-80]


def problem_prefix(problem_name: str) -> str:
    parts = [part for part in re.split(r"_+", problem_name.lower()) if part]
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0] if parts else ""


def review_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Review scores must be JSON numbers")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 10:
        raise ValueError("Review scores must be finite and between 0 and 10")
    return number
