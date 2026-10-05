from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from .catalog.input_catalog import read_catalog
from .llm import LLMConfig


STAGES = {
    "00": "prepare_env",
    "01": "expand_apis",
    "02": "build_graph",
    "03": "validate_graph",
    "04": "sample_chains",
    "05": "generate_problems",
    "06": "export_dataset",
}


PHASES = {
    "prepare": ("00", "01", "02", "03", "04"),
    "generate": ("05",),
    "export": ("06",),
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: Path, api_config: Path) -> dict:
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "input_catalog",
        "work_dir",
        "environment",
        "sampling",
        "generation",
        "source_tag",
    }
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Unknown workflow configuration fields")
    missing = {"input_catalog", "work_dir"} - set(data)
    if missing:
        raise ValueError(f"Missing workflow configuration fields: {sorted(missing)}")

    def resolve(value):
        p = Path(value).expanduser()
        return str((path.parent / p).resolve())

    def resolve_python(value):
        # A venv's executable can point to the base Python binary. Dereferencing
        # that symlink changes sys.prefix and silently drops the venv packages.
        return os.path.abspath(path.parent / Path(value).expanduser())

    for key in ["input_catalog", "work_dir"]:
        data[key] = resolve(data[key])
    data["api_config"] = str(api_config.expanduser().resolve())
    environment = dict(data.get("environment", {}))
    if set(environment) - {"mode", "python", "config", "venv_root"}:
        raise ValueError("Unknown environment configuration fields")
    environment.setdefault("mode", "current")
    if environment["mode"] not in {"current", "existing", "install"}:
        raise ValueError("Environment mode must be current, existing, or install")
    environment["python"] = (
        resolve_python(environment["python"]) if environment.get("python") else sys.executable
    )
    for key, default in [("config", "environments.json"), ("venv_root", "../venvs")]:
        environment[key] = resolve(environment.get(key, default))
    data["environment"] = environment
    sampling = dict(data.get("sampling", {}))
    if set(sampling) - {
        "subdomains",
        "chain_lengths",
        "chains_per_subdomain_length",
        "seed",
        "subdomain_weights",
        "limit",
    }:
        raise ValueError("Unknown sampling configuration fields")
    if sampling.get("subdomain_weights"):
        sampling["subdomain_weights"] = resolve(sampling["subdomain_weights"])
    data["sampling"] = sampling
    generation = dict(data.get("generation", {}))
    if set(generation) - {
        "workflow_workers",
        "validation_python",
        "validation_timeout",
        "max_repair_rounds",
        "quality_repair_rounds",
        "quality_min_grade",
        "temperature",
    }:
        raise ValueError("Unknown generation configuration fields")
    if generation.get("validation_python"):
        generation["validation_python"] = resolve_python(generation["validation_python"])
    elif environment["mode"] == "current":
        generation["validation_python"] = environment["python"]
    data["generation"] = generation
    data.setdefault("source_tag", "scidata_generated")
    if not isinstance(data["source_tag"], str) or not data["source_tag"]:
        raise ValueError("source_tag must be nonempty")
    return data


def stage_paths(config: dict) -> dict[str, Path]:
    work = Path(config["work_dir"])
    return {
        "00": work / "environments.json",
        "01": work / "catalog/expanded_api_catalog",
        "02": work / "catalog/operator_graphs",
        "03": work / "catalog/operator_graphs_runtime_verified",
        "04": work / "chains",
        "05": work / "generation",
        "06": work / "dataset",
    }


def prepare_current_environment(config: dict, output: Path) -> None:
    python = Path(config["environment"]["python"])
    if not python.is_file():
        raise FileNotFoundError(python)
    records = read_catalog(Path(config["input_catalog"]))
    probe = (
        "import importlib.metadata as m,json,sys; "
        "names=sorted({d.metadata['Name'] for d in m.distributions() if d.metadata['Name']}); "
        "print(json.dumps({'python_version':sys.version.split()[0],"
        "'packages':{name:m.version(name) for name in names}}))"
    )
    runtime = json.loads(subprocess.check_output([str(python), "-c", probe], text=True, timeout=60))
    environment = {
        "env_id": "current",
        "venv_path": str(python.parent.parent),
        "python": str(python),
    }
    write_json(
        output,
        {
            "environments": {"current": {**environment, **runtime}},
            "subdomain_to_env": {f"{r['domain']}::{r['subdomain']}": environment for r in records},
        },
    )


def options(mapping: dict, allowed: list[str]) -> list[str]:
    result = []
    for key in allowed:
        value = mapping.get(key)
        if value is not None:
            result.extend(["--" + key.replace("_", "-"), str(value)])
    return result


def check_model_configuration(path: str) -> None:
    model = LLMConfig.from_file(path)
    if model.model == "YOUR_MODEL_ID":
        raise ValueError("Set model in your LLM configuration to the model served by your endpoint")
    if model.backend == "raw_dsv4":
        encoder = Path(model.tokenizer_path).expanduser() / "encoding/encoding_dsv4.py"
        if not encoder.is_file():
            raise FileNotFoundError(f"Required DeepSeek encoder is missing: {encoder}")


def execute_stage(stage: str, config: dict, config_path: Path) -> dict:
    paths = stage_paths(config)
    output = paths[stage]
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise FileExistsError(f"Stage {stage} output already exists: {output}")
    work = Path(config["work_dir"])
    manifest = work / "stage_manifests" / f"{stage}_{STAGES[stage]}.json"
    if manifest.exists():
        raise FileExistsError(f"Stage already attempted; use a new run directory: {manifest}")
    work.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "name": STAGES[stage],
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "configuration": config,
        "output": str(output),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "0"),
    }
    write_json(manifest, record)
    env = dict(os.environ, SCIDATA_DATA_DIR=str(work))
    env.setdefault("PYTHONHASHSEED", "0")
    command = [sys.executable, "-m"]
    try:
        inputs = {
            "00": [Path(config["input_catalog"])],
            "01": [Path(config["input_catalog"])],
            "02": [paths["01"] / "index.json"],
            "03": [paths["02"] / "index.json", paths["00"]],
            "04": [paths["03"] / "index.json"],
            "05": [
                paths["04"] / "chains.jsonl",
                paths["04"] / "sampling_manifest.json",
                Path(config["api_config"]),
            ],
            "06": [work / "sources.json", paths["05"] / "scicode_style_problems.jsonl"],
        }[stage]
        if stage == "02":
            inputs.append(paths["01"] / "all_api_operator_dump.jsonl")
        elif stage in {"03", "04"}:
            graph = paths["02"] if stage == "03" else paths["03"]
            graph_index = json.loads((graph / "index.json").read_text())
            for item in graph_index["subdomains"]:
                inputs.extend(graph / item[key] for key in ("nodes_file", "edges_file"))
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
        record["inputs"] = [{"path": str(path)} for path in inputs]
        if stage == "00" and config["environment"]["mode"] == "current":
            prepare_current_environment(config, output)
            command = []
        elif stage == "00":
            environment = config["environment"]
            command += [
                "scidata.environments",
                "--config",
                environment["config"],
                "--venv-root",
                environment["venv_root"],
                "--python",
                environment["python"],
                "--output",
                str(output),
            ]
            if environment["mode"] == "existing":
                command.append("--index-only")
        elif stage == "01":
            command += [
                "scidata.catalog.build_api_catalog",
                "--source",
                config["input_catalog"],
                "--output-dir",
                str(output),
            ]
        elif stage == "02":
            command += [
                "scidata.catalog.build_operator_io_graphs",
                "--catalog-dir",
                str(paths["01"]),
            ]
        elif stage == "03":
            command += [
                "scidata.catalog.validate_and_filter_operator_graphs",
                "--graph-dir",
                str(paths["02"]),
                "--env-index",
                str(paths["00"]),
                "--out-dir",
                str(output),
            ]
        elif stage == "04":
            command += [
                "scidata.chain_io",
                "--graph-root",
                str(paths["03"]),
                "--output-dir",
                str(output),
            ]
            sampling = config["sampling"]
            for subdomain in sampling.get("subdomains", []):
                command += ["--subdomain", subdomain]
            if "chain_lengths" in sampling:
                lengths = sampling["chain_lengths"]
                command += [
                    "--chain-lengths",
                    ",".join(map(str, lengths)) if isinstance(lengths, list) else str(lengths),
                ]
            command += options(
                sampling, ["chains_per_subdomain_length", "seed", "subdomain_weights", "limit"]
            )
        elif stage == "05":
            check_model_configuration(config["api_config"])
            command += [
                "scidata.pipeline",
                "--chains-file",
                str(paths["04"] / "chains.jsonl"),
                "--chain-manifest",
                str(paths["04"] / "sampling_manifest.json"),
                "--api-config",
                config["api_config"],
                "--output-dir",
                str(output),
            ]
            command += options(
                config["generation"],
                [
                    "workflow_workers",
                    "validation_python",
                    "validation_timeout",
                    "max_repair_rounds",
                    "quality_repair_rounds",
                    "quality_min_grade",
                    "temperature",
                ],
            )
        elif stage == "06":
            command += [
                "scidata.export",
                "--sources",
                str(work / "sources.json"),
                "--output-dir",
                str(output),
            ]
        record["command"] = command
        write_json(manifest, record)
        print(f"[{stage}] {STAGES[stage]} -> {output}", flush=True)
        returncode = 0
        if command:
            log = work / "logs" / f"{stage}_{STAGES[stage]}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            record["log"] = str(log)
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        handle.write(line)
                        handle.flush()
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        if stage == "05":
            completion_file = output / "run_completion.json"
            completion = (
                json.loads(completion_file.read_text()) if completion_file.exists() else None
            )
            # CLI code 2 also means argparse failure: only a complete run report can
            # establish that zero accepted problems is a normal finished result.
            successful = completion is not None and completion.get("fatal_errors") == 0
            successful = successful and completion.get("started_chains") == completion.get(
                "scheduled_chains"
            )
            if returncode not in {0, 2} or not successful:
                raise RuntimeError(
                    f"Generation did not complete successfully; see {record.get('log')}"
                )
            record["completion"] = completion
            write_json(
                work / "sources.json",
                {
                    "source_tag": config["source_tag"],
                    "sources": [
                        {
                            "batch": work.name,
                            "run": "generation",
                            "namespace": work.name,
                            "path": "generation/scicode_style_problems.jsonl",
                        }
                    ],
                },
            )
        elif returncode:
            raise RuntimeError(
                f"Stage {stage} exited with code {returncode}; see {record.get('log')}"
            )
        record["returncode"] = returncode
        record["status"] = "complete"
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(manifest, record)
        return record
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest, record)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[*PHASES, "all", "check"],
        metavar="COMMAND",
        help="prepare, generate, export, all, or check",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/llm.local.json"),
        help="LLM service configuration",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.local.json"),
        help="Pipeline configuration for input, output, sampling, and validation",
    )
    parser.add_argument(
        "--from",
        dest="start",
        choices=list(PHASES),
        default="prepare",
        metavar="PHASE",
        help="First phase for all",
    )
    parser.add_argument(
        "--to",
        dest="end",
        choices=list(PHASES),
        default="export",
        metavar="PHASE",
        help="Last phase for all",
    )
    args = parser.parse_args(argv)
    phases = list(PHASES)
    start = phases.index(args.start)
    end = phases.index(args.end)
    if start > end:
        parser.error("--from must not be after --to")
    supplied = {arg.split("=", 1)[0] for arg in (argv if argv is not None else sys.argv[1:])}
    if args.stage != "all" and supplied.intersection({"--from", "--to"}):
        parser.error("--from and --to require the 'all' command")
    try:
        config = load_config(args.pipeline_config, args.config)
        if args.stage == "check":
            records = read_catalog(Path(config["input_catalog"]))
            check_model_configuration(config["api_config"])
            for python in (
                config["environment"]["python"],
                config["generation"].get("validation_python", sys.executable),
            ):
                if not Path(python).is_file():
                    raise FileNotFoundError(f"Missing Python interpreter: {python}")
            print(f"Configuration valid: {len(records)} input families; no model request sent")
            return 0
        selected_phases = phases[start : end + 1] if args.stage == "all" else [args.stage]
        for phase in selected_phases:
            for stage in PHASES[phase]:
                execute_stage(stage, config, args.pipeline_config)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Workflow stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
