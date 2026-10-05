from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


from ..paths import catalog_root

ROOT = catalog_root()
SOURCES = [ROOT / "expanded_api_catalog"]
OUT_DIR = ROOT / "operator_graphs"

MIN_EDGE_SCORE = 1.35
MAX_OUT_EDGES_PER_NODE = 45
MAX_CANDIDATES_PER_NODE = 90


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "per",
    "the",
    "this",
    "to",
    "used",
    "using",
    "with",
    "without",
    "optional",
    "object",
    "objects",
    "array",
    "arrays",
    "dict",
    "list",
    "lists",
    "value",
    "values",
    "result",
    "results",
    "data",
    "params",
    "parameters",
    "config",
    "configuration",
    "callable",
    "library",
    "specific",
    "arguments",
    "return",
    "api",
    "symbol",
    "parent",
    "operator",
    "operators",
    "output",
    "outputs",
    "input",
    "inputs",
    "optional",
    "numeric",
    "scalars",
    "scalar",
    "float",
    "int",
    "string",
    "str",
    "bool",
    "table",
    "tables",
    "metadata",
    "diagnostics",
    "report",
    "reports",
    "summary",
    "summaries",
    "plain",
    "raw",
    "candidate",
    "base",
    "source",
    "target",
    "expected",
    "fixture",
    "fixtures",
    "minimal",
    "api_inputs",
    "api_outputs",
    "library-specific",
    "library_specific",
    "semantics",
    "context",
    "family",
    "follow",
    "call",
    "calls",
    "symbol",
}

GENERIC_TYPES = {
    "object",
    "array",
    "dict",
    "list",
    "scalar",
    "float",
    "int",
    "bool",
    "string",
    "table",
    "data",
    "metadata",
    "diagnostic",
    "callable",
    "config",
    "parameter",
    "result",
    "value",
    "function",
}

TYPE_PATTERNS = [
    (
        "matrix",
        r"\b(matrix|matrices|sparse|dense|hamiltonian|jacobian|hessian|operator matrix|superoperator|choi)\b",
    ),
    ("vector", r"\b(vector|vectors|state vector|eigenvector|four-vector|momentum|velocity)\b"),
    ("tensor", r"\b(tensor|tensors|density matrix|rank|einsum)\b"),
    ("mesh", r"\b(mesh|cell|cells|element|elements|dof|facet|node|nodes|grid|stencil)\b"),
    (
        "field",
        r"\b(field|fields|wavefront|intensity|phase|flux|potential|density|pressure|temperature)\b",
    ),
    ("state", r"\b(state|states|ket|density matrix|wavefunction|mps|trajectory|phase-space)\b"),
    ("model", r"\b(model|models|ansatz|hamiltonian|mechanism|network|workspace|pdf|likelihood)\b"),
    (
        "solver",
        r"\b(solver|solution|solve|fit|optimizer|minimizer|integrator|simulator|simulation)\b",
    ),
    (
        "spectrum",
        r"\b(spectrum|spectra|eigenvalue|eigenvalues|band|bands|dos|periodogram|power spectrum)\b",
    ),
    (
        "geometry",
        r"\b(geometry|geometries|lattice|structure|cell|surface|polygon|orientation|coordinate|coordinates)\b",
    ),
    (
        "molecule",
        r"\b(molecule|molecules|mol|smiles|sdf|pdb|atom|atoms|bond|bonds|formula|forcefield)\b",
    ),
    (
        "reaction",
        r"\b(reaction|reactions|species|rates|kinetics|stoichiometry|enzyme|catalytic|mechanism)\b",
    ),
    (
        "sequence",
        r"\b(sequence|sequences|fasta|fastq|bam|vcf|genotype|motif|pwm|alignment|hmm|k-mer|variant)\b",
    ),
    ("tree", r"\b(tree|phylogenetic|newick|branch|clade|ancestry)\b"),
    ("graph", r"\b(graph|network|nodes|edges|connectivity|lattice graph)\b"),
    ("image", r"\b(image|images|ccd|fits|detector|map|pixel|pixels|projection|mask)\b"),
    ("time_series", r"\b(time series|light curve|trace|times|trajectory|history|series)\b"),
    ("histogram", r"\b(histogram|hist|bins|counts|binned|axis|axes)\b"),
    (
        "probability",
        r"\b(probability|probabilities|posterior|likelihood|pvalue|confidence|credible|distribution)\b",
    ),
    ("orbit", r"\b(orbit|orbits|kepler|ephemeris|trajectory|body|bodies|n-body|lambert)\b"),
    ("circuit", r"\b(circuit|circuits|gate|gates|qubit|qubits|pauli|kraus|channel)\b"),
    ("unit", r"\b(unit|units|quantity|quantities|dimensionless|conversion)\b"),
]

MODE_TRANSITIONS = {
    "build_fixture": {
        "construct_minimal_input",
        "validate_input_constraints",
        "single_case_execute",
        "batch_execute",
        "parameter_sweep",
    },
    "construct_minimal_input": {
        "validate_input_constraints",
        "configure_options",
        "single_case_execute",
        "batch_execute",
        "parameter_sweep",
    },
    "construct_or_initialize": {
        "configure_or_parameterize",
        "evaluate_or_run",
        "single_case_execute",
        "metadata_query",
        "extract_metadata",
    },
    "configure_options": {"single_case_execute", "batch_execute", "parameter_sweep"},
    "configure_or_parameterize": {
        "evaluate_or_run",
        "single_case_execute",
        "parameter_sweep",
        "extract_or_reduce_result",
    },
    "validate_input_constraints": {"single_case_execute", "batch_execute", "parameter_sweep"},
    "single_case_execute": {
        "extract_primary_output",
        "compute_derived_observable",
        "compute_residual",
        "compute_error_metric",
        "compare_with_reference",
        "to_plain_arrays",
        "schema_or_type_check",
        "aggregate_statistics",
        "validate_or_compare",
    },
    "evaluate_or_run": {
        "extract_or_reduce_result",
        "compute_derived_observable",
        "compute_residual",
        "compare_with_reference",
        "parameter_sweep",
        "validate_or_compare",
    },
    "batch_execute": {
        "aggregate_statistics",
        "compute_error_metric",
        "compare_with_reference",
        "mask_or_filter",
        "to_plain_arrays",
    },
    "parameter_sweep": {
        "aggregate_statistics",
        "convergence_diagnostic",
        "sensitivity_analysis",
        "compute_error_metric",
        "extract_primary_output",
    },
    "extract_primary_output": {
        "compute_derived_observable",
        "compute_error_metric",
        "compare_with_reference",
        "aggregate_statistics",
        "schema_or_type_check",
    },
    "extract_or_reduce_result": {
        "compute_derived_observable",
        "compare_with_reference",
        "validate_or_compare",
    },
    "compute_derived_observable": {
        "compute_error_metric",
        "compare_with_reference",
        "aggregate_statistics",
        "schema_or_type_check",
    },
    "to_plain_arrays": {
        "compute_error_metric",
        "compare_with_reference",
        "aggregate_statistics",
        "schema_or_type_check",
        "serialization_roundtrip",
    },
    "interpolate_or_resample": {
        "compute_error_metric",
        "compare_with_reference",
        "compute_derived_observable",
    },
    "differentiate_or_jacobian": {"optimize_or_fit", "compute_residual", "compare_with_reference"},
    "optimize_or_fit": {
        "extract_primary_output",
        "compute_error_metric",
        "compare_with_reference",
        "convergence_diagnostic",
    },
    "hidden_test_generator": {"compare_with_reference", "validate_or_compare"},
    "answer_stub_generator": {"single_case_execute", "validate_input_constraints"},
}


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


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def tokens(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]+", text.lower())
    out = set()
    for tok in raw:
        tok = tok.strip("_+-")
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        out.add(tok)
    return out


def types_for(text: str) -> set[str]:
    low = text.lower()
    found = set()
    for typ, pattern in TYPE_PATTERNS:
        if re.search(pattern, low):
            found.add(typ)
    # Lightweight generic tags, kept for node metadata but mostly ignored for edges.
    for typ in GENERIC_TYPES:
        if re.search(rf"\b{re.escape(typ)}s?\b", low):
            found.add(typ)
    return found


def split_name(desc: str) -> tuple[str, str]:
    if ":" in desc:
        name, rest = desc.split(":", 1)
        return name.strip(), rest.strip()
    return "", desc.strip()


def descriptors(descs: list[str]) -> list[dict]:
    result = []
    for desc in descs:
        name, rest = split_name(desc)
        text = f"{name} {rest}"
        token_set = tokens(text)
        type_set = types_for(text)
        result.append(
            {
                "raw": desc,
                "name": name,
                "text": rest,
                "tokens": sorted(token_set),
                "types": sorted(type_set),
                "_token_set": token_set,
                "_type_set": type_set,
                "_specific_type_set": specific_types(type_set),
            }
        )
    return result


def specific_types(type_set: set[str]) -> set[str]:
    return {t for t in type_set if t not in GENERIC_TYPES}


def descriptor_score(out_desc: dict, in_desc: dict) -> tuple[float, dict]:
    out_specific = out_desc["_specific_type_set"]
    in_specific = in_desc["_specific_type_set"]
    type_inter = out_specific & in_specific

    out_tokens = out_desc["_token_set"]
    in_tokens = in_desc["_token_set"]
    token_inter = out_tokens & in_tokens

    score = 0.0
    if type_inter:
        score += 1.2 + min(0.8, 0.25 * len(type_inter))
    # Generic-only type overlap should not create edges by itself.
    if token_inter:
        score += 0.65 + min(0.75, 0.12 * len(token_inter))
    if out_desc["name"] and in_desc["name"]:
        name_out = tokens(out_desc["name"])
        name_in = tokens(in_desc["name"])
        if name_out & name_in:
            score += 0.5

    reason = {
        "matched_types": sorted(type_inter),
        "matched_tokens": sorted(token_inter)[:12],
        "matched_output": out_desc["raw"],
        "matched_input": in_desc["raw"],
    }
    return score, reason


def workflow_score(src: dict, dst: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    smode = src.get("operation_mode", "")
    dmode = dst.get("operation_mode", "")
    if dmode in MODE_TRANSITIONS.get(smode, set()):
        score += 0.9
        reasons.append("mode_transition")
    if src.get("api_symbol") == dst.get("api_symbol"):
        score += 0.25
        reasons.append("same_api_symbol")
    if src.get("source_operator_id") == dst.get("source_operator_id"):
        score += 0.35
        reasons.append("same_source_operator")
    if src.get("library") == dst.get("library"):
        score += 0.12
        reasons.append("same_library")
    return score, reasons


def best_descriptor_match(src: dict, dst: dict) -> tuple[float, dict]:
    best_score = 0.0
    best_reason = {
        "matched_types": [],
        "matched_tokens": [],
        "matched_output": src["_outputs_norm"][0]["raw"] if src["_outputs_norm"] else "",
        "matched_input": dst["_inputs_norm"][0]["raw"] if dst["_inputs_norm"] else "",
    }
    for out_desc in src["_outputs_norm"]:
        for in_desc in dst["_inputs_norm"]:
            score, reason = descriptor_score(out_desc, in_desc)
            if score > best_score:
                best_score = score
                best_reason = reason
    return best_score, best_reason


def best_edge(src: dict, dst: dict) -> dict | None:
    if src["id"] == dst["id"]:
        return None
    type_inter = src["_output_specific_types"] & dst["_input_specific_types"]
    token_inter = src["_output_tokens_set"] & dst["_input_tokens_set"]

    descriptor_score_value, descriptor_reason = best_descriptor_match(src, dst)

    wscore, wreasons = workflow_score(src, dst)
    score = descriptor_score_value + wscore
    # Require either some real descriptor match or a strong natural workflow relation.
    if score < MIN_EDGE_SCORE:
        return None
    if not type_inter and not token_inter and "mode_transition" not in wreasons:
        return None

    return {
        "source": src["id"],
        "target": dst["id"],
        "score": round(score, 3),
        "edge_rule": "source.outputs -> target.inputs",
        "source_api_symbol": src.get("api_symbol"),
        "target_api_symbol": dst.get("api_symbol"),
        "source_operation_mode": src.get("operation_mode"),
        "target_operation_mode": dst.get("operation_mode"),
        "source_library": src.get("library"),
        "target_library": dst.get("library"),
        "source_operator": src.get("source_operator"),
        "target_operator": dst.get("source_operator"),
        "matched_output": descriptor_reason["matched_output"],
        "matched_input": descriptor_reason["matched_input"],
        "matched_types": descriptor_reason["matched_types"],
        "matched_tokens": descriptor_reason["matched_tokens"],
        "workflow_reasons": wreasons,
    }


def load_all_records(sources: list[Path] | None = None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for source in SOURCES if sources is None else sources:
        for path in source.glob("*/api_operator_dump.jsonl"):
            for record in read_jsonl(path):
                record["_catalog_source"] = source.name
                record["_subdomain_slug"] = path.parent.name
                grouped[record["subdomain"]].append(record)
    return grouped


def node_from_record(record: dict) -> dict:
    input_desc = descriptors(record.get("inputs", []))
    output_desc = descriptors(record.get("outputs", []))
    input_types = sorted(
        set().union(*(set(d["types"]) for d in input_desc)) if input_desc else set()
    )
    output_types = sorted(
        set().union(*(set(d["types"]) for d in output_desc)) if output_desc else set()
    )
    input_tokens = sorted(Counter(tok for d in input_desc for tok in d["tokens"]).keys())
    output_tokens = sorted(Counter(tok for d in output_desc for tok in d["tokens"]).keys())
    return {
        "id": record["id"],
        "domain": record["domain"],
        "subdomain": record["subdomain"],
        "library": record.get("library"),
        "api_symbol": record.get("api_symbol"),
        "operation_mode": record.get("operation_mode"),
        "source_operator_id": record.get("source_operator_id"),
        "source_operator": record.get("source_operator"),
        "source_operator_kind": record.get("source_operator_kind"),
        "catalog_source": record.get("_catalog_source"),
        "input_types": input_types,
        "output_types": output_types,
        "input_tokens": input_tokens[:40],
        "output_tokens": output_tokens[:40],
        "inputs": record.get("inputs", []),
        "outputs": record.get("outputs", []),
    }


def build_subdomain_graph(
    subdomain: str, records: list[dict]
) -> tuple[list[dict], list[dict], dict]:
    for record in records:
        record["_inputs_norm"] = descriptors(record.get("inputs", []))
        record["_outputs_norm"] = descriptors(record.get("outputs", []))
        record["_input_specific_types"] = set()
        record["_output_specific_types"] = set()
        record["_input_tokens_set"] = set()
        record["_output_tokens_set"] = set()
        for desc in record["_inputs_norm"]:
            record["_input_specific_types"].update(specific_types(set(desc["types"])))
            record["_input_tokens_set"].update(desc["tokens"])
        for desc in record["_outputs_norm"]:
            record["_output_specific_types"].update(specific_types(set(desc["types"])))
            record["_output_tokens_set"].update(desc["tokens"])

    nodes = [node_from_record(record) for record in records]

    input_index: dict[str, set[int]] = defaultdict(set)
    api_index: dict[str, set[int]] = defaultdict(set)
    source_operator_index: dict[str, set[int]] = defaultdict(set)

    for idx, record in enumerate(records):
        api_index[record.get("api_symbol", "")].add(idx)
        source_operator_index[record.get("source_operator_id", "")].add(idx)
        for in_desc in record["_inputs_norm"]:
            for typ in specific_types(set(in_desc["types"])):
                input_index[f"type:{typ}"].add(idx)
            for tok in in_desc["tokens"]:
                input_index[f"tok:{tok}"].add(idx)

    edges = []
    for src_idx, src in enumerate(records):
        candidate_counter: Counter[int] = Counter()
        for typ in src["_output_specific_types"]:
            for idx in input_index.get(f"type:{typ}", set()):
                candidate_counter[idx] += 3
        for tok in src["_output_tokens_set"]:
            for idx in input_index.get(f"tok:{tok}", set()):
                candidate_counter[idx] += 1

        candidate_indices = {
            idx for idx, _ in candidate_counter.most_common(MAX_CANDIDATES_PER_NODE)
        }
        # Natural workflow transitions are most reliable within the same API
        # symbol or source operator family.
        candidate_indices.update(api_index.get(src.get("api_symbol", ""), set()))
        candidate_indices.update(
            source_operator_index.get(src.get("source_operator_id", ""), set())
        )
        candidate_indices.discard(src_idx)

        candidates = []
        for dst_idx in candidate_indices:
            dst = records[dst_idx]
            edge = best_edge(src, dst)
            if edge is not None:
                candidates.append(edge)
        candidates.sort(key=lambda e: (-e["score"], e["target"]))
        edges.extend(candidates[:MAX_OUT_EDGES_PER_NODE])

    in_degree = Counter(edge["target"] for edge in edges)
    out_degree = Counter(edge["source"] for edge in edges)
    type_counts = Counter(t for node in nodes for t in node["output_types"])
    mode_counts = Counter(node["operation_mode"] for node in nodes)
    summary = {
        "subdomain": subdomain,
        "slug": slugify(subdomain),
        "edge_direction": "A -> B means A.outputs can feed B.inputs",
        "edge_score_threshold": MIN_EDGE_SCORE,
        "max_out_edges_per_node": MAX_OUT_EDGES_PER_NODE,
        "max_candidates_per_node": MAX_CANDIDATES_PER_NODE,
        "nodes": len(nodes),
        "edges": len(edges),
        "avg_out_degree": round(len(edges) / len(nodes), 3) if nodes else 0,
        "nodes_with_out_edges": sum(1 for node in nodes if out_degree[node["id"]] > 0),
        "nodes_with_in_edges": sum(1 for node in nodes if in_degree[node["id"]] > 0),
        "top_output_types": type_counts.most_common(20),
        "top_operation_modes": mode_counts.most_common(20),
        "top_out_degree_nodes": [
            {"id": node_id, "degree": degree} for node_id, degree in out_degree.most_common(10)
        ],
        "top_in_degree_nodes": [
            {"id": node_id, "degree": degree} for node_id, degree in in_degree.most_common(10)
        ],
    }
    return nodes, edges, summary


def write_schema_files(out_dir: Path) -> None:
    node_schema = {
        "title": "SciData v2 operator graph node",
        "type": "object",
        "required": [
            "id",
            "domain",
            "subdomain",
            "library",
            "api_symbol",
            "operation_mode",
            "inputs",
            "outputs",
        ],
        "additionalProperties": True,
    }
    edge_schema = {
        "title": "SciData v2 operator graph edge",
        "type": "object",
        "required": [
            "source",
            "target",
            "score",
            "edge_rule",
            "matched_output",
            "matched_input",
            "matched_types",
            "matched_tokens",
        ],
        "additionalProperties": True,
    }
    (out_dir / "node.schema.json").write_text(
        json.dumps(node_schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "edge.schema.json").write_text(
        json.dumps(edge_schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, action="append")
    args = parser.parse_args(argv)
    grouped = load_all_records(args.catalog_dir)
    if not grouped:
        parser.error("No expanded API catalogs found; run build_api_catalog first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_schema_files(OUT_DIR)

    index = {
        "description": "Per-subdomain operator IO-compatibility graphs.",
        "edge_direction": "A -> B means A.outputs can feed B.inputs.",
        "edge_rule_note": "Edges are heuristic high-confidence compatibility edges, not an exhaustive typed-program proof.",
        "edge_score_threshold": MIN_EDGE_SCORE,
        "max_out_edges_per_node": MAX_OUT_EDGES_PER_NODE,
        "max_candidates_per_node": MAX_CANDIDATES_PER_NODE,
        "subdomains": [],
    }

    total_nodes = 0
    total_edges = 0
    for subdomain in sorted(grouped):
        print(f"building {subdomain} ({len(grouped[subdomain])} nodes)", flush=True)
        records = grouped[subdomain]
        nodes, edges, summary = build_subdomain_graph(subdomain, records)
        slug = slugify(subdomain)
        subdir = OUT_DIR / slug
        subdir.mkdir(parents=True, exist_ok=True)
        write_jsonl(subdir / "nodes.jsonl", nodes)
        write_jsonl(subdir / "edges.jsonl", edges)
        (subdir / "graph_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (subdir / "README.md").write_text(
            "\n".join(
                [
                    f"# {subdomain} Operator IO Graph",
                    "",
                    f"- Nodes: {summary['nodes']}",
                    f"- Edges: {summary['edges']}",
                    f"- Edge direction: `{summary['edge_direction']}`",
                    f"- Edge threshold: `{MIN_EDGE_SCORE}`",
                    f"- Max outgoing edges per node: `{MAX_OUT_EDGES_PER_NODE}`",
                    f"- Candidate targets scored per node: `{MAX_CANDIDATES_PER_NODE}`",
                    "",
                    "Files:",
                    "",
                    "- `nodes.jsonl`: all operators in this subdomain.",
                    "- `edges.jsonl`: high-confidence output-to-input compatibility edges.",
                    "- `graph_summary.json`: counts and graph diagnostics.",
                    "",
                    "Edge records include matched input/output descriptors, matched semantic types, matched tokens, and workflow reasons.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        index["subdomains"].append(
            {
                "slug": slug,
                "subdomain": subdomain,
                "nodes": summary["nodes"],
                "edges": summary["edges"],
                "nodes_with_out_edges": summary["nodes_with_out_edges"],
                "nodes_with_in_edges": summary["nodes_with_in_edges"],
                "nodes_file": f"{slug}/nodes.jsonl",
                "edges_file": f"{slug}/edges.jsonl",
                "summary_file": f"{slug}/graph_summary.json",
            }
        )
        total_nodes += summary["nodes"]
        total_edges += summary["edges"]
        print(
            f"finished {summary['slug']}: {summary['nodes']} nodes, {summary['edges']} edges",
            flush=True,
        )

    index["total_subdomains"] = len(index["subdomains"])
    index["total_nodes"] = total_nodes
    index["total_edges"] = total_edges
    (OUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# SciData v2 Operator IO Graphs",
                "",
                "This directory contains one directed graph per subdomain.",
                "",
                "Edge direction:",
                "",
                "- `A -> B` means at least one output descriptor of operator `A` is compatible with at least one input descriptor of operator `B`.",
                "",
                "The user-facing intent is dataflow composition for SciCode-style task generation. Edges are heuristic high-confidence links based on semantic type overlap, token overlap, and workflow-mode transitions.",
                "",
                "Files:",
                "",
                "- `index.json`: graph inventory.",
                "- `node.schema.json`: node schema.",
                "- `edge.schema.json`: edge schema.",
                "- `<subdomain>/nodes.jsonl`: graph nodes.",
                "- `<subdomain>/edges.jsonl`: graph edges.",
                "- `<subdomain>/graph_summary.json`: graph diagnostics.",
                "",
                f"Total subdomains: `{len(index['subdomains'])}`",
                f"Total nodes: `{total_nodes}`",
                f"Total edges: `{total_edges}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
