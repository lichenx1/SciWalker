from __future__ import annotations


import argparse


import ast


import csv


import hashlib


import json
from dataclasses import dataclass


from pathlib import Path


from typing import Any, Iterator


import pyarrow as pa


import pyarrow.parquet as pq


PROMPT_PREFIX = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{previous_steps}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task will be to develop the Python code for this next step based on the provided description and function header.

{step_prompt}

{function_header}

{return_line}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""


@dataclass(frozen=True)
class SourceSpec:
    batch: str
    run: str
    namespace: str
    path: Path


def human_title(problem_name: str) -> str:
    return problem_name.replace("_", " ").replace("-", " ").title()


def canonical_problem_id(original_id: str, namespace: str) -> str:
    prefix = "SCIDATA-V2-"
    suffix = original_id[len(prefix) :] if original_id.startswith(prefix) else original_id
    return f"{prefix}{namespace}-{suffix}"


def normalize_test_cases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def derive_function_header(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError("Cannot derive function_header from invalid ground_truth_code") from exc
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args = ast.unparse(node.args)
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            return f"{prefix}def {node.name}({args}){returns}:"
    raise ValueError("Cannot derive function_header: no top-level function definition")


def normalize_sub_step(
    step: dict[str, Any],
    index: int,
    original_problem_id: str,
    canonical_id: str,
) -> dict[str, Any]:
    ground_truth_code = str(step.get("ground_truth_code", ""))
    function_header = str(step.get("function_header") or "").strip()
    if not function_header:
        function_header = derive_function_header(ground_truth_code)

    step_number = str(step.get("step_number") or f"{canonical_id}.{index}")
    if step_number.startswith(original_problem_id):
        step_number = canonical_id + step_number[len(original_problem_id) :]

    return {
        "step_number": step_number,
        "step_description_prompt": str(step.get("step_description_prompt", "")),
        "step_background": str(step.get("step_background", "")),
        "function_header": function_header,
        "return_line": str(step.get("return_line", "    return result")),
        "test_cases": normalize_test_cases(step.get("test_cases")),
        "ground_truth_code": ground_truth_code,
    }


def build_first_step_prompt(problem: dict[str, Any], sub_steps: list[dict[str, Any]]) -> str:
    first = sub_steps[0]
    step_parts = [first["step_description_prompt"].strip()]
    if first["step_background"].strip():
        step_parts.append(first["step_background"].strip())
    return PROMPT_PREFIX.format(
        previous_steps="",
        step_prompt="\n".join(part for part in step_parts if part),
        function_header=first["function_header"].strip(),
        return_line=first["return_line"].rstrip(),
        dependencies=str(problem.get("required_dependencies", "")).strip(),
    )


def content_hash(problem: dict[str, Any]) -> str:
    payload = "\n".join(
        str(problem.get(key, ""))
        for key in (
            "problem_name",
            "problem_description_main",
            "problem_io",
            "general_solution",
            "general_tests",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_problems(spec: SourceSpec) -> Iterator[tuple[int, dict[str, Any]]]:
    with spec.path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def build_parquet_row(
    problem: dict[str, Any],
    spec: SourceSpec,
    source_line: int,
    source_tag: str,
    scicode_repo: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_id = str(problem["problem_id"])
    canonical_id = canonical_problem_id(original_id, spec.namespace)
    raw_steps = problem.get("sub_steps") or []
    if not raw_steps:
        raise ValueError(f"Problem has no sub_steps: {spec.path}:{source_line}")
    sub_steps = [
        normalize_sub_step(step, index, original_id, canonical_id)
        for index, step in enumerate(raw_steps, 1)
    ]

    problem_name = str(problem.get("problem_name") or canonical_id)
    problem_title = str(problem.get("problem_title") or human_title(problem_name))
    required_dependencies = str(problem.get("required_dependencies", ""))
    source_file = str(spec.path.resolve())
    h5py_file = ""

    problem_data = {
        "problem_id": canonical_id,
        "problem_name": problem_name,
        "problem_title": problem_title,
        "required_dependencies": required_dependencies,
        "sub_steps": sub_steps,
        "h5py_file": h5py_file,
        "scicode_repo": scicode_repo,
        "dense_reward": True,
        "with_background": True,
    }
    extra_info = {
        "problem_id": canonical_id,
        "problem_data": problem_data,
        "sub_steps": sub_steps,
        "required_dependencies": required_dependencies,
        "h5py_file": h5py_file,
        "scicode_repo": scicode_repo,
        "dense_reward": True,
        "with_background": True,
        "evaluation_mode": "official_sequential_multiturn",
        "source_file": source_file,
        "source_line": source_line,
        "source_tag": source_tag,
        "original_data_source": "scicode_synthetic_deepseek_v4_flash",
    }
    parquet_row = {
        "prompt": [{"role": "user", "content": build_first_step_prompt(problem, sub_steps)}],
        "label": f"{source_tag}:{canonical_id}",
        "data_source": "scicode_multiturn_problem",
        "reward_model": {"style": "rule", "ground_truth": canonical_id},
        "extra_info": extra_info,
        "agent_name": "scicode_multiturn_agent",
    }

    review = problem.get("_quality_review") or {}
    chain = problem.get("_source_chain") or {}
    manifest_row = {
        "batch": spec.batch,
        "run": spec.run,
        "namespace": spec.namespace,
        "source_file": source_file,
        "source_line": source_line,
        "original_problem_id": original_id,
        "canonical_problem_id": canonical_id,
        "problem_name": problem_name,
        "quality_grade": str(review.get("grade", "")),
        "subdomain_slug": str(chain.get("subdomain_slug") or ""),
        "chain_length": str(chain.get("length") or ""),
    }
    return parquet_row, manifest_row


def write_batch(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def export(args: argparse.Namespace) -> dict[str, Any]:
    reference_schema = (
        pq.ParquetFile(args.reference_parquet).schema_arrow
        if args.reference_parquet
        else scicode_schema()
    )
    expected_columns = [
        "prompt",
        "label",
        "data_source",
        "reward_model",
        "extra_info",
        "agent_name",
    ]
    if reference_schema.names != expected_columns:
        raise ValueError(f"Unexpected reference columns: {reference_schema.names!r}")

    sources, manifest_tag = load_sources(args.sources)
    source_tag = args.source_tag or manifest_tag
    outputs = [args.full_output, args.s_output, args.manifest, args.summary]
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("Output paths must be distinct")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    for spec in sources:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    args.s_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    full_tmp = args.full_output.with_suffix(args.full_output.suffix + ".tmp")
    s_tmp = args.s_output.with_suffix(args.s_output.suffix + ".tmp")
    manifest_tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    for path in (full_tmp, s_tmp, manifest_tmp):
        path.unlink(missing_ok=True)

    full_writer = pq.ParquetWriter(full_tmp, reference_schema, compression="snappy")
    s_writer = pq.ParquetWriter(s_tmp, reference_schema, compression="snappy")
    full_buffer: list[dict[str, Any]] = []
    s_buffer: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    content_counts: dict[str, int] = {}
    grade_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    full_rows = 0
    s_rows = 0

    manifest_fields = [
        "full_parquet_row",
        "s_parquet_row",
        "batch",
        "run",
        "namespace",
        "source_file",
        "source_line",
        "original_problem_id",
        "canonical_problem_id",
        "problem_name",
        "quality_grade",
        "subdomain_slug",
        "chain_length",
    ]

    try:
        with manifest_tmp.open("w", newline="", encoding="utf-8") as handle:
            manifest_writer = csv.DictWriter(handle, fieldnames=manifest_fields, delimiter="\t")
            manifest_writer.writeheader()
            for spec in sources:
                source_key = f"{spec.batch}:{spec.run}"
                for source_line, problem in iter_problems(spec):
                    parquet_row, manifest_row = build_parquet_row(
                        problem=problem,
                        spec=spec,
                        source_line=source_line,
                        source_tag=source_tag,
                        scicode_repo=args.scicode_repo,
                    )
                    grade = manifest_row["quality_grade"]
                    if grade not in {"S", "A"}:
                        raise ValueError(f"Unexpected grade {grade!r}: {spec.path}:{source_line}")
                    canonical_id = manifest_row["canonical_problem_id"]
                    if canonical_id in seen_ids:
                        raise ValueError(f"Duplicate canonical ID: {canonical_id}")
                    seen_ids.add(canonical_id)

                    manifest_row["full_parquet_row"] = full_rows
                    manifest_row["s_parquet_row"] = s_rows if grade == "S" else ""
                    manifest_writer.writerow(manifest_row)

                    full_buffer.append(parquet_row)
                    full_rows += 1
                    if grade == "S":
                        s_buffer.append(parquet_row)
                        s_rows += 1
                    grade_counts[grade] = grade_counts.get(grade, 0) + 1
                    source_counts[source_key] = source_counts.get(source_key, 0) + 1
                    digest = content_hash(problem)
                    content_counts[digest] = content_counts.get(digest, 0) + 1

                    if len(full_buffer) >= args.batch_size:
                        write_batch(full_writer, full_buffer, reference_schema)
                    if len(s_buffer) >= args.batch_size:
                        write_batch(s_writer, s_buffer, reference_schema)

        write_batch(full_writer, full_buffer, reference_schema)
        write_batch(s_writer, s_buffer, reference_schema)
    except Exception:
        full_writer.close()
        s_writer.close()
        full_tmp.unlink(missing_ok=True)
        s_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise
    else:
        full_writer.close()
        s_writer.close()

    full_tmp.replace(args.full_output)
    s_tmp.replace(args.s_output)
    manifest_tmp.replace(args.manifest)

    full_file = pq.ParquetFile(args.full_output)
    s_file = pq.ParquetFile(args.s_output)
    full_schema_equal = full_file.schema_arrow.equals(reference_schema, check_metadata=True)
    s_schema_equal = s_file.schema_arrow.equals(reference_schema, check_metadata=True)
    if not full_schema_equal or not s_schema_equal:
        raise AssertionError("Output parquet schema differs from reference schema")
    if full_file.metadata.num_rows != full_rows:
        raise AssertionError("Full parquet row count mismatch")
    if s_file.metadata.num_rows != s_rows:
        raise AssertionError("S-only parquet row count mismatch")

    if full_rows:
        full_sample = full_file.read_row_group(0).slice(0, 1).to_pylist()[0]
        sample_id = full_sample["extra_info"]["problem_id"]
        if not (
            sample_id
            == full_sample["extra_info"]["problem_data"]["problem_id"]
            == full_sample["reward_model"]["ground_truth"]
        ):
            raise AssertionError("Problem ID contract is inconsistent in output row")

    return {
        "reference_schema": str(args.reference_parquet.resolve())
        if args.reference_parquet
        else "packaged_scicode_schema",
        "full_output": str(args.full_output.resolve()),
        "s_output": str(args.s_output.resolve()),
        "manifest": str(args.manifest.resolve()),
        "full_rows": full_rows,
        "s_rows": s_rows,
        "grade_counts": grade_counts,
        "source_counts": source_counts,
        "unique_canonical_ids": len(seen_ids),
        "exact_content_duplicate_rows": sum(
            count - 1 for count in content_counts.values() if count > 1
        ),
        "full_schema_equal": full_schema_equal,
        "s_schema_equal": s_schema_equal,
        "top_level_columns": reference_schema.names,
    }


def scicode_schema() -> pa.Schema:
    from .paths import resource_path

    spec = json.loads(resource_path("scicode_schema.json").read_text(encoding="utf-8"))

    def field(item):
        kind = item["type"]
        if kind == "struct":
            dtype = pa.struct([field(child) for child in item["children"]])
        elif kind == "list":
            dtype = pa.list_(field(item["element"]))
        else:
            dtype = {"string": pa.string(), "bool": pa.bool_(), "int64": pa.int64()}[kind]
        return pa.field(item["name"], dtype, nullable=item["nullable"])

    return pa.schema([field(item) for item in spec])


def load_sources(path: Path) -> tuple[list[SourceSpec], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_tag = payload["source_tag"]
    if not isinstance(source_tag, str) or not source_tag:
        raise ValueError("source_tag must be a nonempty string")
    sources = []
    for item in payload["sources"]:
        sources.append(
            SourceSpec(
                batch=item["batch"],
                run=item["run"],
                namespace=item["namespace"],
                path=(path.parent / item["path"]).resolve(),
            )
        )
    if not sources:
        raise ValueError("The manifest must contain at least one source")
    if len({s.namespace for s in sources}) != len(sources):
        raise ValueError("Source namespaces must be unique")
    if len({s.path for s in sources}) != len(sources):
        raise ValueError("Source paths must be unique")
    return sources, source_tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-parquet", type=Path)
    parser.add_argument(
        "--scicode-repo",
        default="",
        help="Optional downstream SciCode checkout path stored as metadata",
    )
    parser.add_argument("--source-tag", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    args.full_output = args.output_dir / "full.parquet"
    args.s_output = args.output_dir / "S.parquet"
    args.manifest = args.output_dir / "manifest.tsv"
    args.summary = args.output_dir / "summary.json"
    summary = export(args)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
