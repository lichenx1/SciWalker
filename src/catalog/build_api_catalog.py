from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


from ..paths import catalog_root, resource_path
from .input_catalog import read_catalog, slugify

ROOT = catalog_root()
SOURCE = resource_path("relativity_gravitation.jsonl")
OUT_DIR = ROOT / "expanded_api_catalog"


MODE_TEMPLATES = {
    "base_api_symbol": {
        "meaning": "Use this public API symbol as the central operation in a SciCode-style subproblem.",
        "inputs": [
            "api_inputs: library-specific arguments - exact signature should be checked against installed docs"
        ],
        "outputs": [
            "api_outputs: library-specific return value - object, scalar, array, table, model, or diagnostic"
        ],
        "patterns": ["ask for a minimal usage of this API symbol on a deterministic fixture"],
        "oracles": [
            "return type and shape match the fixture",
            "parent operator family oracle is satisfied",
        ],
    },
    "construct_minimal_input": {
        "meaning": "Construct the smallest valid input object or argument set for this API symbol.",
        "inputs": ["raw_params: dict/list/scalars - simple user-facing parameters"],
        "outputs": ["minimal_input: object/array/dict - valid API-ready input"],
        "patterns": [
            "write a helper that converts plain numbers or strings into valid library inputs"
        ],
        "oracles": ["required fields are present", "dimensions or lengths match expected values"],
    },
    "normalize_units_or_scales": {
        "meaning": "Normalize physical units, coordinate scales, rates, probabilities, tolerances, or dimensions before calling the API.",
        "inputs": ["values: scalars/arrays/objects - raw values with possible units or scales"],
        "outputs": [
            "normalized_values: arrays/dict - values in the canonical convention for this task"
        ],
        "patterns": ["prepare stable inputs for scientific code generation"],
        "oracles": ["known conversion fixture matches", "dimensionless values remain unchanged"],
    },
    "validate_input_constraints": {
        "meaning": "Check domain constraints before the API symbol is used.",
        "inputs": [
            "candidate_inputs: dict/object - inputs to validate",
            "constraints: dict - positivity, shape, bounds, conservation, or allowed values",
        ],
        "outputs": [
            "is_valid: bool - validation flag",
            "violations: list - human-readable failed checks",
        ],
        "patterns": ["create hidden-test guards for invalid scientific cases"],
        "oracles": ["valid fixture passes", "invalid fixture fails with expected violation"],
    },
    "configure_options": {
        "meaning": "Configure algorithmic, numerical, model, parser, or solver options for the API symbol.",
        "inputs": [
            "base_object_or_config: object/dict - prior setup",
            "options: dict - method, tolerances, bounds, backend, parser, or analysis options",
        ],
        "outputs": ["configured_object_or_config: object/dict - normalized configuration"],
        "patterns": ["separate configuration from execution in multi-step problems"],
        "oracles": [
            "options are preserved after normalization",
            "defaults are filled deterministically",
        ],
    },
    "build_fixture": {
        "meaning": "Build a small synthetic fixture that exercises this API symbol.",
        "inputs": [
            "fixture_params: dict - small sizes, seed, known coefficients, toy geometry, sequence, molecule, or model"
        ],
        "outputs": ["fixture: dict/object - input data plus expected metadata"],
        "patterns": ["generate tiny examples suitable for hidden tests"],
        "oracles": ["fixture is deterministic", "expected metadata matches generated data"],
    },
    "single_case_execute": {
        "meaning": "Run the API symbol on one small case.",
        "inputs": ["prepared_input: object/dict/array - validated and configured input"],
        "outputs": ["raw_result: object/array/scalar/table - direct API result"],
        "patterns": ["solve, parse, simulate, transform, score, or analyze one fixture"],
        "oracles": [
            "result is finite/nonempty where expected",
            "result shape matches input dimensions",
        ],
    },
    "batch_execute": {
        "meaning": "Run the API symbol over a batch of inputs.",
        "inputs": [
            "input_batch: list/array/table - multiple cases",
            "shared_config: dict - reused options",
        ],
        "outputs": ["batch_results: list/array/table - one result per case"],
        "patterns": ["generate many problem instances from one operator"],
        "oracles": [
            "number of outputs equals number of inputs",
            "batch and looped single-case outputs agree",
        ],
    },
    "parameter_sweep": {
        "meaning": "Sweep one parameter while holding the rest of the API setup fixed.",
        "inputs": [
            "base_config: dict - fixed settings",
            "sweep_values: array/list - values to scan",
        ],
        "outputs": ["sweep_table: table/list - parameter value and result for each run"],
        "patterns": [
            "study convergence, phase behavior, sensitivity, model response, or threshold"
        ],
        "oracles": [
            "row count equals sweep value count",
            "known limiting value or monotonic trend holds if applicable",
        ],
    },
    "extract_primary_output": {
        "meaning": "Extract the main scientific or algorithmic output from the API result.",
        "inputs": ["raw_result: object/dict/table/array - output from the API call"],
        "outputs": ["primary_output: scalar/array/table - value used by downstream steps"],
        "patterns": ["turn rich library results into simple arrays/scalars for answer checking"],
        "oracles": ["expected field/key exists", "plain output shape is stable"],
    },
    "extract_metadata": {
        "meaning": "Extract labels, units, dimensions, topology, coordinates, schema, or provenance metadata.",
        "inputs": ["object_or_result: object - library object or output"],
        "outputs": ["metadata: dict - normalized metadata fields"],
        "patterns": ["make library-specific objects usable in generic generators"],
        "oracles": [
            "required metadata keys exist",
            "metadata values agree with constructed fixture",
        ],
    },
    "to_plain_arrays": {
        "meaning": "Convert library-specific output into plain Python, NumPy-like arrays, or JSON-like data.",
        "inputs": ["object_or_result: object - library-specific output"],
        "outputs": [
            "plain_result: dict/list/array - serializable or NumPy-compatible representation"
        ],
        "patterns": ["prepare hidden tests and parquet/json exports"],
        "oracles": ["conversion is deterministic", "numeric values match source object"],
    },
    "compute_derived_observable": {
        "meaning": "Compute a derived scientific quantity from the API output.",
        "inputs": [
            "raw_or_plain_result: object/array/table - API output",
            "observable_spec: str/dict/callable - derived quantity to compute",
        ],
        "outputs": ["observable: scalar/array/table - derived measurement"],
        "patterns": ["make a later SciCode substep depend on an earlier API result"],
        "oracles": ["observable agrees with direct small-case calculation"],
    },
    "compute_residual": {
        "meaning": "Compute residuals, reconstruction error, balance error, or model mismatch.",
        "inputs": [
            "candidate_result: object/array",
            "reference_or_forward_rule: object/callable/array",
        ],
        "outputs": ["residual: array", "residual_norm: float"],
        "patterns": ["validate solvers, fits, transforms, parsers, and simulators"],
        "oracles": ["residual norm is below tolerance for reference fixture"],
    },
    "compute_error_metric": {
        "meaning": "Compute a scalar error, distance, score, loss, likelihood, or discrepancy metric.",
        "inputs": ["prediction: array/object", "target: array/object", "metric_config: dict"],
        "outputs": ["metric: float", "metric_components: dict"],
        "patterns": ["grade a generated result with a numeric oracle"],
        "oracles": ["metric is zero for identical inputs", "metric is nonnegative when expected"],
    },
    "compare_with_reference": {
        "meaning": "Compare the API result against an analytic, tabulated, brute-force, or smaller-library reference.",
        "inputs": [
            "candidate: object/array/scalar",
            "reference: object/array/scalar",
            "tolerance: float",
        ],
        "outputs": ["matches: bool", "comparison: dict - max error, relative error, pass/fail"],
        "patterns": ["create robust hidden-test comparison logic"],
        "oracles": ["reference fixture passes", "perturbed fixture fails above tolerance"],
    },
    "convergence_diagnostic": {
        "meaning": "Diagnose convergence with respect to iterations, grid size, basis size, sample count, or tolerance.",
        "inputs": ["history_or_results: list/array/table", "convergence_rule: dict"],
        "outputs": ["converged: bool", "diagnostics: dict"],
        "patterns": ["turn iterative scientific workflows into testable substeps"],
        "oracles": ["final diagnostic meets threshold", "iteration count is within budget"],
    },
    "stability_check": {
        "meaning": "Check numerical, statistical, dynamical, or algebraic stability of the API-symbol result.",
        "inputs": [
            "results: array/object",
            "stability_params: dict - perturbations, time step, tolerance, seed, or bounds",
        ],
        "outputs": ["stable: bool", "stability_metrics: dict"],
        "patterns": ["avoid brittle tasks by explicitly testing stability"],
        "oracles": ["stable fixture passes", "known unstable fixture fails or shows larger metric"],
    },
    "sensitivity_analysis": {
        "meaning": "Measure response of the API-symbol result to input perturbations.",
        "inputs": ["base_input: object/dict/array", "perturbations: dict/list"],
        "outputs": ["sensitivity_table: table/array", "summary: dict"],
        "patterns": ["generate parameter-reasoning subtasks"],
        "oracles": [
            "zero perturbation gives zero response",
            "finite-difference sign or scale matches known toy model",
        ],
    },
    "handle_boundary_conditions": {
        "meaning": "Apply or validate boundary, initial, terminal, or domain constraints relevant to the API symbol.",
        "inputs": [
            "state_or_model: object/array",
            "conditions: dict - boundary, initial, terminal, or domain values",
        ],
        "outputs": ["conditioned_state_or_model: object/array", "condition_report: dict"],
        "patterns": ["split modeling tasks into condition application and solve/evaluate"],
        "oracles": [
            "conditioned values match prescribed values",
            "unconstrained values remain unchanged",
        ],
    },
    "seeded_random_run": {
        "meaning": "Run stochastic or randomized usage of the API symbol with explicit seed control.",
        "inputs": ["random_config: dict - seed, sample count, distribution, stochastic method"],
        "outputs": ["random_result: array/table/object", "seed_metadata: dict"],
        "patterns": ["make stochastic SciCode tasks reproducible"],
        "oracles": ["same seed gives same result", "sample moments match expected trend"],
    },
    "serialization_roundtrip": {
        "meaning": "Serialize, export, parse, or round-trip the API input or output.",
        "inputs": [
            "object_or_result: object",
            "format_config: dict - json, csv, text, domain file, or plain dict representation",
        ],
        "outputs": ["serialized: str/bytes/dict", "roundtrip_object: object/dict optional"],
        "patterns": ["support dataset generation, provenance, and reusable fixtures"],
        "oracles": ["round-trip preserves key metadata and values"],
    },
    "schema_or_type_check": {
        "meaning": "Check shape, dtype, schema, object type, units, labels, or allowed categories.",
        "inputs": ["object_or_result: object/array/table", "schema: dict"],
        "outputs": ["schema_valid: bool", "schema_report: dict"],
        "patterns": ["write robust validators before hidden numerical tests"],
        "oracles": ["valid fixture passes", "wrong shape or missing key fails"],
    },
    "mask_or_filter": {
        "meaning": "Apply a mask, query, threshold, region, selector, or filter to API data/results.",
        "inputs": ["data_or_result: object/array/table", "filter_rule: callable/string/dict"],
        "outputs": ["filtered: object/array/table", "filter_summary: dict"],
        "patterns": [
            "select events, molecules, sequences, mesh cells, samples, structures, or trajectories"
        ],
        "oracles": ["filtered count matches expected fixture", "mask shape matches source data"],
    },
    "aggregate_statistics": {
        "meaning": "Aggregate per-item outputs into means, histograms, counts, rates, confidence intervals, or grouped summaries.",
        "inputs": ["values: array/table/list", "aggregation_config: dict"],
        "outputs": ["summary_statistics: dict/table"],
        "patterns": ["create final report-style subtasks"],
        "oracles": ["counts sum to total", "mean of constant data equals constant"],
    },
    "interpolate_or_resample": {
        "meaning": "Interpolate, resample, reindex, remesh, align, or map API outputs onto a target grid or index.",
        "inputs": [
            "source_values: array/object",
            "source_grid_or_index: array/object",
            "target_grid_or_index: array/object",
        ],
        "outputs": ["resampled_values: array/object", "resampling_report: dict"],
        "patterns": ["bridge simulation output and comparison/evaluation grids"],
        "oracles": ["constant data remains constant", "target shape matches target grid"],
    },
    "differentiate_or_jacobian": {
        "meaning": "Compute a derivative, gradient, Jacobian, finite-difference sensitivity, or local linearization.",
        "inputs": [
            "function_or_result: callable/object/array",
            "point_or_state: array/object",
            "step_or_method: dict",
        ],
        "outputs": ["derivative: array/matrix", "derivative_report: dict"],
        "patterns": ["turn optimization, dynamics, kinetics, and fitting into multi-step problems"],
        "oracles": ["finite-difference derivative matches analytic toy model"],
    },
    "optimize_or_fit": {
        "meaning": "Use the API symbol inside a fitting, minimization, calibration, or inverse-design loop.",
        "inputs": [
            "objective_data: array/table/object",
            "initial_params: array/dict",
            "bounds_or_constraints: dict",
        ],
        "outputs": ["best_params: array/dict", "fit_diagnostics: dict"],
        "patterns": ["wrap a domain API in a calibratable objective"],
        "oracles": ["objective improves from initial guess", "synthetic parameters are recovered"],
    },
    "compose_pipeline": {
        "meaning": "Compose this API symbol with upstream and downstream operators into a multi-step pipeline.",
        "inputs": ["upstream_output: object/array/table", "pipeline_config: dict"],
        "outputs": ["pipeline_output: object/array/table", "pipeline_metadata: dict"],
        "patterns": ["make API symbol one stage of a SciCode main problem"],
        "oracles": [
            "pipeline preserves expected dimensions",
            "identity stage leaves output unchanged",
        ],
    },
    "inverse_or_roundtrip": {
        "meaning": "Apply an inverse operation or verify round-trip consistency.",
        "inputs": [
            "forward_input: object/array",
            "forward_output: object/array",
            "inverse_config: dict",
        ],
        "outputs": ["recovered_input: object/array", "roundtrip_error: float"],
        "patterns": [
            "validate transforms, encodings, parsers, decompositions, and coordinate changes"
        ],
        "oracles": ["round-trip error below tolerance"],
    },
    "edge_case_behavior": {
        "meaning": "Evaluate the API symbol on small, degenerate, boundary, empty, or extreme cases.",
        "inputs": ["edge_case_input: object/array/dict", "expected_behavior: dict"],
        "outputs": ["edge_case_result: object/array/scalar", "behavior_report: dict"],
        "patterns": ["improve hidden-test coverage for generated problems"],
        "oracles": ["known edge case result", "error handling matches task specification"],
    },
    "performance_budget": {
        "meaning": "Check that the API-symbol usage stays within problem-size, time, memory, or iteration budgets.",
        "inputs": ["problem_size: dict", "budget: dict", "implementation_result: object optional"],
        "outputs": ["budget_ok: bool", "budget_metrics: dict"],
        "patterns": ["avoid generated tasks that are too heavy for evaluation"],
        "oracles": ["toy fixture under budget", "size scaling report fields present"],
    },
    "documentation_example_adapter": {
        "meaning": "Adapt a documentation-style API example into a small deterministic SciCode subproblem.",
        "inputs": [
            "example_params: dict - simplified example values",
            "task_constraints: dict - what to expose or hide",
        ],
        "outputs": ["adapted_problem_io: dict - input fixture, expected output, and notes"],
        "patterns": ["convert library examples into benchmark-ready substeps"],
        "oracles": [
            "adapted fixture is deterministic",
            "expected output computed from parent oracle",
        ],
    },
    "hidden_test_generator": {
        "meaning": "Generate hidden-test fixtures for this API-symbol operation.",
        "inputs": [
            "test_config: dict - seeds, ranges, sizes, tolerances",
            "reference_rule: callable or text rule",
        ],
        "outputs": ["hidden_tests: list - inputs and expected checks", "test_metadata: dict"],
        "patterns": ["make the operator directly usable for benchmark generation"],
        "oracles": [
            "all generated tests pass reference implementation",
            "test cases cover at least two regimes",
        ],
    },
    "answer_stub_generator": {
        "meaning": "Generate a function signature or answer stub around this API-symbol operation.",
        "inputs": [
            "function_name: str",
            "input_specs: list",
            "output_specs: list",
            "allowed_dependencies: list",
        ],
        "outputs": ["function_header: str", "docstring: str", "dependency_list: list"],
        "patterns": ["turn API operator into a SciCode-style sub-step prompt"],
        "oracles": ["header parses as Python", "declared outputs align with output specs"],
    },
}


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_record(parent: dict, api_symbol: str, mode: str, ordinal: int) -> dict:
    template = MODE_TEMPLATES[mode]
    record_id = f"{slugify(parent['subdomain'])}--{slugify(parent['id'])}--{slugify(api_symbol)}--{mode}--{ordinal:05d}"
    return {
        "id": record_id,
        "record_type": "api_usage_operator",
        "domain": parent["domain"],
        "subdomain": parent["subdomain"],
        "library": parent["library"],
        "source_operator_id": parent["id"],
        "source_operator": parent["operator"],
        "source_operator_kind": parent["operator_kind"],
        "api_symbol": api_symbol,
        "operation_mode": mode,
        "inputs": parent["inputs"] + template["inputs"],
        "outputs": parent["outputs"] + template["outputs"],
        "meaning": f"{template['meaning']} Parent meaning: {parent['meaning']}",
        "scicode_task_patterns": [parent["scicode_use"]] + template["patterns"],
        "test_oracles": parent["test_oracles"] + template["oracles"],
        "signature_scope": "symbol_level_not_full_signature",
        "scope_note": "Generated from curated operator-family API symbols. Validate exact signatures against installed package version/docs before execution.",
    }


def write_schema(path: Path) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Scientific API usage operator record",
        "type": "object",
        "required": [
            "id",
            "record_type",
            "domain",
            "subdomain",
            "library",
            "source_operator_id",
            "source_operator",
            "source_operator_kind",
            "api_symbol",
            "operation_mode",
            "inputs",
            "outputs",
            "meaning",
            "scicode_task_patterns",
            "test_oracles",
            "signature_scope",
            "scope_note",
        ],
        "properties": {
            "id": {"type": "string"},
            "record_type": {"type": "string", "enum": ["api_usage_operator"]},
            "domain": {"type": "string"},
            "subdomain": {"type": "string"},
            "library": {"type": "string"},
            "source_operator_id": {"type": "string"},
            "source_operator": {"type": "string"},
            "source_operator_kind": {"type": "string"},
            "api_symbol": {"type": "string"},
            "operation_mode": {"type": "string"},
            "inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "outputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "meaning": {"type": "string"},
            "scicode_task_patterns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "test_oracles": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "signature_scope": {"type": "string", "enum": ["symbol_level_not_full_signature"]},
            "scope_note": {"type": "string"},
        },
        "additionalProperties": False,
    }
    path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    parents = read_catalog(args.source)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for parent in parents:
        grouped[parent["subdomain"]].append(parent)

    if not grouped:
        parser.error("The source catalog contains no operator families")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Catalog output directory must be empty: {out_dir}")
    write_schema(out_dir / "api_operator_dump.schema.json")

    modes = list(MODE_TEMPLATES)
    index = {
        "description": "API-symbol-level operator dumps expanded from the supplied semantic families.",
        "generation_rule": f"Each curated API symbol is expanded by {len(modes)} operation modes.",
        "signature_scope": "symbol_level_not_full_signature",
        "subdomains": [],
    }
    all_records: list[dict] = []

    for subdomain in sorted(grouped):
        source_records = grouped[subdomain]
        slug = slugify(subdomain)
        expanded = []
        ordinal = 0
        seen_source_symbols = []
        for parent in source_records:
            for api_symbol in parent["related_apis"]:
                seen_source_symbols.append(api_symbol)
                for mode in modes:
                    ordinal += 1
                    expanded.append(build_record(parent, api_symbol, mode, ordinal))

        out_subdir = out_dir / slug
        out_subdir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_subdir / "api_operator_dump.jsonl", expanded)
        write_jsonl(out_subdir / "source_operator_families.jsonl", source_records)
        library_counts = Counter(r["library"] for r in source_records)
        unique_api_symbols = sorted(set(seen_source_symbols))
        (out_subdir / "README.md").write_text(
            "\n".join(
                [
                    f"# {subdomain} Expanded API Operator Dump",
                    "",
                    f"- Domain: {source_records[0]['domain']}",
                    f"- Source operator families: {len(source_records)}",
                    f"- Unique source API symbols: {len(unique_api_symbols)}",
                    f"- Expanded records: {len(expanded)}",
                    f"- Operation modes per API symbol: {len(modes)}",
                    "",
                    "Files:",
                    "",
                    "- `api_operator_dump.jsonl`: expanded API-symbol-level usage operators.",
                    "- `source_operator_families.jsonl`: source semantic operator-family records.",
                    "",
                    "Libraries:",
                    "",
                    *[
                        f"- `{lib}`: {count} source operator families"
                        for lib, count in sorted(library_counts.items())
                    ],
                    "",
                    "Scope: task-generation catalog at symbol-and-usage-mode granularity, not an exact installed-package signature dump.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        item = {
            "slug": slug,
            "domain": source_records[0]["domain"],
            "subdomain": subdomain,
            "source_operator_families": len(source_records),
            "unique_api_symbols": len(unique_api_symbols),
            "operation_modes_per_api_symbol": len(modes),
            "expanded_records": len(expanded),
            "libraries": sorted(library_counts),
            "api_operator_dump": f"{slug}/api_operator_dump.jsonl",
            "source_operator_families_jsonl": f"{slug}/source_operator_families.jsonl",
        }
        index["subdomains"].append(item)
        all_records.extend(expanded)

    write_jsonl(out_dir / "all_api_operator_dump.jsonl", all_records)
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Expanded API Operator Catalog",
                "",
                f"This directory covers {len(grouped)} supplied subdomain(s).",
                "",
                "The records are API-symbol usage operators for SciCode-style task generation. They are not exact call signatures.",
                "",
                "Files:",
                "",
                "- `index.json`: summary and paths.",
                "- `api_operator_dump.schema.json`: JSON schema for expanded records.",
                "- `all_api_operator_dump.jsonl`: concatenated records across supplied subdomains.",
                "- `<subdomain>/api_operator_dump.jsonl`: per-subdomain expanded records.",
                "- `<subdomain>/source_operator_families.jsonl`: source high-level operator families.",
                "",
                "Record counts:",
                "",
                *[
                    f"- `{item['slug']}`: {item['expanded_records']} records from {item['unique_api_symbols']} API symbols"
                    for item in index["subdomains"]
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
