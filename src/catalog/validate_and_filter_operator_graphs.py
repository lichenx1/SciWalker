from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


from ..paths import catalog_root, data_root

ROOT = catalog_root()
DEFAULT_GRAPH_DIR = ROOT / "operator_graphs"
DEFAULT_ENV_INDEX = data_root() / "environments.json"
DEFAULT_OUT_DIR = ROOT / "operator_graphs_runtime_verified"
WORKER = Path(__file__).with_name("operator_symbol_call_worker.py")


def slugify(text: str) -> str:
    text = text.lower().replace("/", "_").replace("&", "and")
    out = []
    last_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        else:
            if not last_sep:
                out.append("_")
                last_sep = True
    return "".join(out).strip("_")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def subdomain_env_map(env_index_path: Path) -> dict[str, dict]:
    env_index = json.loads(env_index_path.read_text(encoding="utf-8"))
    mapping = {}
    for key, item in env_index["subdomain_to_env"].items():
        subdomain = key.split("::", 1)[1]
        mapping[subdomain] = item
        mapping[slugify(subdomain)] = item
    return mapping


def worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.setdefault("MPLCONFIGDIR", "/tmp/scicode_operator_mpl")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/scicode_operator_numba")
    env.setdefault("XDG_CACHE_HOME", "/tmp/scicode_operator_xdg")
    env.setdefault("PYTHONWARNINGS", "ignore")
    for key in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_worker(
    *,
    python_bin: str,
    symbols: list[str],
    work_dir: Path,
    env_id: str,
    chunk_id: int,
    timeout_seconds: int,
    process_timeout_seconds: int,
) -> tuple[list[dict], dict]:
    in_path = work_dir / f"{env_id}.chunk{chunk_id:04d}.symbols.json"
    out_path = work_dir / f"{env_id}.chunk{chunk_id:04d}.validation.jsonl"
    in_path.write_text(json.dumps(symbols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cmd = [
        python_bin,
        str(WORKER),
        "--symbols-json",
        str(in_path),
        "--out-jsonl",
        str(out_path),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    start = time.time()
    meta = {
        "env_id": env_id,
        "chunk_id": chunk_id,
        "symbol_count": len(symbols),
        "returncode": None,
        "seconds": None,
        "stdout_tail": [],
    }
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=worker_env(),
            timeout=process_timeout_seconds,
        )
        meta["returncode"] = proc.returncode
        meta["seconds"] = time.time() - start
        meta["stdout_tail"] = proc.stdout.strip().splitlines()[-20:]
    except subprocess.TimeoutExpired as exc:
        meta["returncode"] = "timeout"
        meta["seconds"] = time.time() - start
        meta["stdout_tail"] = (
            (exc.stdout or "").splitlines()[-20:] if isinstance(exc.stdout, str) else []
        )
    rows = read_jsonl(out_path) if out_path.exists() else []
    seen = {row["api_symbol"] for row in rows}
    for symbol in symbols:
        if symbol not in seen:
            rows.append(
                {
                    "api_symbol": symbol,
                    "resolve_ok": False,
                    "module": None,
                    "object_type": None,
                    "callable": False,
                    "call_status": "chunk_failed",
                    "call_ok": False,
                    "call_error": None,
                    "return_type": None,
                    "return_repr": None,
                    "validation_status": "chunk_failed",
                    "error": f"worker chunk returncode={meta['returncode']}",
                }
            )
    return rows, meta


def summarize_validation(rows: list[dict]) -> dict:
    status_counts = Counter(row.get("validation_status") for row in rows)
    call_counts = Counter(row.get("call_status") for row in rows)
    total = len(rows)
    return {
        "total_unique_symbols": total,
        "resolve_ok_symbols": sum(1 for row in rows if row.get("resolve_ok")),
        "call_ok_symbols": sum(1 for row in rows if row.get("call_ok")),
        "status_counts": dict(status_counts),
        "call_status_counts": dict(call_counts),
        "resolve_ok_rate": round(sum(1 for row in rows if row.get("resolve_ok")) / total, 6)
        if total
        else 0,
        "call_ok_rate": round(sum(1 for row in rows if row.get("call_ok")) / total, 6)
        if total
        else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--env-index", default=str(DEFAULT_ENV_INDEX))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=35)
    parser.add_argument("--timeout-seconds", type=int, default=6)
    parser.add_argument("--process-timeout-seconds", type=int, default=420)
    parser.add_argument(
        "--keep-policy",
        choices=["call_ok", "callable_resolved"],
        default="call_ok",
        help="call_ok keeps only actually smoke-called symbols; callable_resolved also keeps resolved callables skipped by fixture policy.",
    )
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    out_dir = Path(args.out_dir)
    work_dir = out_dir / "_validation_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((graph_dir / "index.json").read_text(encoding="utf-8"))
    env_map = subdomain_env_map(Path(args.env_index))

    subdomain_nodes: dict[str, list[dict]] = {}
    subdomain_edges: dict[str, list[dict]] = {}
    symbol_to_contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
    symbol_to_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)
    slug_items = []

    for item in index["subdomains"]:
        slug = item["slug"]
        nodes = read_jsonl(graph_dir / item["nodes_file"])
        edges = read_jsonl(graph_dir / item["edges_file"])
        subdomain_nodes[slug] = nodes
        subdomain_edges[slug] = edges
        env_item = env_map.get(item["subdomain"]) or env_map.get(slug)
        if env_item is None:
            raise SystemExit(f"no venv mapping for subdomain {item['subdomain']} ({slug})")
        env_id = env_item["env_id"]
        slug_items.append((item, env_item))
        for node in nodes:
            symbol = node.get("api_symbol")
            if not symbol:
                continue
            key = (env_id, symbol)
            symbol_to_contexts[key].add(slug)
            symbol_to_nodes[key].append(node["id"])

    env_to_symbols: dict[str, list[str]] = defaultdict(list)
    env_to_python: dict[str, str] = {}
    for (env_id, symbol), _slugs in symbol_to_contexts.items():
        env_to_symbols[env_id].append(symbol)
    for _item, env_item in slug_items:
        env_to_python[env_item["env_id"]] = env_item["python"]
    for env_id in env_to_symbols:
        env_to_symbols[env_id] = sorted(set(env_to_symbols[env_id]))

    all_rows: list[dict] = []
    worker_chunks: list[dict] = []
    start = time.time()
    for env_id, symbols in sorted(env_to_symbols.items()):
        python_bin = env_to_python[env_id]
        for i in range(0, len(symbols), args.chunk_size):
            chunk = symbols[i : i + args.chunk_size]
            chunk_id = i // args.chunk_size
            rows, meta = run_worker(
                python_bin=python_bin,
                symbols=chunk,
                work_dir=work_dir,
                env_id=env_id,
                chunk_id=chunk_id,
                timeout_seconds=args.timeout_seconds,
                process_timeout_seconds=args.process_timeout_seconds,
            )
            for row in rows:
                row["env_id"] = env_id
                row["subdomain_slugs"] = sorted(
                    symbol_to_contexts.get((env_id, row["api_symbol"]), [])
                )
                row["node_count"] = len(symbol_to_nodes.get((env_id, row["api_symbol"]), []))
            all_rows.extend(rows)
            worker_chunks.append(meta)
            print(env_id, chunk_id, len(chunk), meta["returncode"], flush=True)

    symbol_result = {(row["env_id"], row["api_symbol"]): row for row in all_rows}

    def keep_symbol(env_id: str, symbol: str) -> bool:
        row = symbol_result.get((env_id, symbol))
        if not row:
            return False
        if args.keep_policy == "call_ok":
            return bool(row.get("call_ok"))
        return bool(row.get("call_ok") or (row.get("resolve_ok") and row.get("callable")))

    out_index = {
        "description": "Runtime-validated subset of SciData v2 operator IO-compatibility graphs.",
        "source_graph_dir": str(graph_dir),
        "generated_at": datetime.now().isoformat(),
        "keep_policy": args.keep_policy,
        "edge_direction": index.get("edge_direction", "A -> B means A.outputs can feed B.inputs."),
        "subdomains": [],
    }

    report_subdomains = []
    total_original_nodes = total_kept_nodes = 0
    total_original_edges = total_kept_edges = 0
    total_nodes_by_status = Counter()
    unavailable_rows = []

    for item, env_item in slug_items:
        slug = item["slug"]
        env_id = env_item["env_id"]
        nodes = subdomain_nodes[slug]
        edges = subdomain_edges[slug]
        kept_nodes = []
        removed_nodes = []
        for node in nodes:
            symbol = node.get("api_symbol")
            row = symbol_result.get((env_id, symbol))
            if row:
                status = row.get("validation_status")
                total_nodes_by_status[status] += 1
            if symbol and keep_symbol(env_id, symbol):
                enriched = dict(node)
                row = symbol_result[(env_id, symbol)]
                enriched["runtime_validation"] = {
                    "env_id": env_id,
                    "validation_status": row.get("validation_status"),
                    "call_status": row.get("call_status"),
                    "return_type": row.get("return_type"),
                    "object_type": row.get("object_type"),
                    "module": row.get("module"),
                }
                kept_nodes.append(enriched)
            else:
                removed_nodes.append(node)
        kept_ids = {node["id"] for node in kept_nodes}
        kept_edges = [
            edge for edge in edges if edge["source"] in kept_ids and edge["target"] in kept_ids
        ]

        slug_dir = out_dir / slug
        write_jsonl(slug_dir / "nodes.jsonl", kept_nodes)
        write_jsonl(slug_dir / "edges.jsonl", kept_edges)
        summary = {
            "slug": slug,
            "subdomain": item["subdomain"],
            "env_id": env_id,
            "original_nodes": len(nodes),
            "kept_nodes": len(kept_nodes),
            "removed_nodes": len(removed_nodes),
            "original_edges": len(edges),
            "kept_edges": len(kept_edges),
            "removed_edges": len(edges) - len(kept_edges),
            "node_keep_rate": round(len(kept_nodes) / len(nodes), 6) if nodes else 0,
            "node_unavailable_rate": round(len(removed_nodes) / len(nodes), 6) if nodes else 0,
            "edge_keep_rate": round(len(kept_edges) / len(edges), 6) if edges else 0,
        }
        write_json(slug_dir / "graph_summary.json", summary)
        (slug_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {item['subdomain']} Runtime-Validated Operator Graph",
                    "",
                    f"- Environment: `{env_id}`",
                    f"- Original nodes: `{len(nodes)}`",
                    f"- Kept nodes: `{len(kept_nodes)}`",
                    f"- Removed nodes: `{len(removed_nodes)}`",
                    f"- Original edges: `{len(edges)}`",
                    f"- Kept edges: `{len(kept_edges)}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        out_index["subdomains"].append(
            {
                "slug": slug,
                "subdomain": item["subdomain"],
                "env_id": env_id,
                "nodes": len(kept_nodes),
                "edges": len(kept_edges),
                "nodes_file": f"{slug}/nodes.jsonl",
                "edges_file": f"{slug}/edges.jsonl",
                "summary_file": f"{slug}/graph_summary.json",
            }
        )
        report_subdomains.append(summary)
        total_original_nodes += len(nodes)
        total_kept_nodes += len(kept_nodes)
        total_original_edges += len(edges)
        total_kept_edges += len(kept_edges)

    write_jsonl(out_dir / "validation" / "api_symbol_validation.jsonl", all_rows)
    for row in all_rows:
        if not keep_symbol(row["env_id"], row["api_symbol"]):
            unavailable_rows.append(row)
    write_jsonl(out_dir / "validation" / "unavailable_api_symbols.jsonl", unavailable_rows)
    write_json(out_dir / "validation" / "worker_chunks.json", worker_chunks)
    write_json(out_dir / "index.json", out_index)

    overall = {
        "generated_at": out_index["generated_at"],
        "source_graph_dir": str(graph_dir),
        "out_dir": str(out_dir),
        "keep_policy": args.keep_policy,
        "elapsed_seconds": round(time.time() - start, 3),
        "unique_symbol_validation": summarize_validation(all_rows),
        "total_original_nodes": total_original_nodes,
        "total_kept_nodes": total_kept_nodes,
        "total_removed_nodes": total_original_nodes - total_kept_nodes,
        "node_keep_rate": round(total_kept_nodes / total_original_nodes, 6)
        if total_original_nodes
        else 0,
        "node_unavailable_rate": round(
            (total_original_nodes - total_kept_nodes) / total_original_nodes, 6
        )
        if total_original_nodes
        else 0,
        "total_original_edges": total_original_edges,
        "total_kept_edges": total_kept_edges,
        "total_removed_edges": total_original_edges - total_kept_edges,
        "edge_keep_rate": round(total_kept_edges / total_original_edges, 6)
        if total_original_edges
        else 0,
        "node_status_counts": dict(total_nodes_by_status),
        "subdomains": report_subdomains,
    }
    write_json(out_dir / "VALIDATION_SUMMARY.json", overall)

    md_lines = [
        "# Runtime-Validated Operator Graph Report",
        "",
        f"Generated at: `{overall['generated_at']}`",
        f"Source graph dir: `{graph_dir}`",
        f"Output graph dir: `{out_dir}`",
        f"Keep policy: `{args.keep_policy}`",
        "",
        "## Overall",
        "",
        f"- Unique API symbols: `{overall['unique_symbol_validation']['total_unique_symbols']}`",
        f"- Resolved symbols: `{overall['unique_symbol_validation']['resolve_ok_symbols']}` "
        f"({overall['unique_symbol_validation']['resolve_ok_rate']})",
        f"- Smoke-call passed symbols: `{overall['unique_symbol_validation']['call_ok_symbols']}` "
        f"({overall['unique_symbol_validation']['call_ok_rate']})",
        f"- Original nodes: `{total_original_nodes}`",
        f"- Kept nodes: `{total_kept_nodes}`",
        f"- Removed nodes: `{overall['total_removed_nodes']}`",
        f"- Node unavailable rate: `{overall['node_unavailable_rate']}`",
        f"- Original edges: `{total_original_edges}`",
        f"- Kept edges: `{total_kept_edges}`",
        f"- Edge keep rate: `{overall['edge_keep_rate']}`",
        "",
        "## Per Subdomain",
        "",
        "| Subdomain | Env | Original Nodes | Kept Nodes | Removed Nodes | Unavailable Rate | Original Edges | Kept Edges |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report_subdomains:
        md_lines.append(
            f"| {row['subdomain']} | `{row['env_id']}` | {row['original_nodes']} | {row['kept_nodes']} | "
            f"{row['removed_nodes']} | {row['node_unavailable_rate']} | {row['original_edges']} | {row['kept_edges']} |"
        )
    md_lines.extend(
        [
            "",
            "## Validation Status Counts",
            "",
        ]
    )
    for key, value in sorted(overall["unique_symbol_validation"]["status_counts"].items()):
        md_lines.append(f"- `{key}`: `{value}`")
    md_lines.extend(
        [
            "",
            "## Main Artifacts",
            "",
            f"- Filtered graph index: `{out_dir / 'index.json'}`",
            f"- Summary JSON: `{out_dir / 'VALIDATION_SUMMARY.json'}`",
            f"- Symbol validation details: `{out_dir / 'validation' / 'api_symbol_validation.jsonl'}`",
            f"- Unavailable symbol details: `{out_dir / 'validation' / 'unavailable_api_symbols.jsonl'}`",
        ]
    )
    (out_dir / "RUNTIME_VALIDATED_GRAPH_REPORT.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(out_dir)
    print(
        json.dumps(
            {
                k: overall[k]
                for k in [
                    "total_original_nodes",
                    "total_kept_nodes",
                    "node_keep_rate",
                    "total_kept_edges",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
