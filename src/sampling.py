from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import hashlib
import heapq
import json
import math
import random
import re


def parse_chain_lengths(text: str) -> list[int]:
    lengths: list[int] = []
    for part in re.split(r"[, ]+", str(text or "").strip()):
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"chain length must be positive: {value}")
        lengths.append(value)
    if not lengths:
        raise ValueError("at least one chain length is required")
    return lengths


def stable_chain_seed(base_seed: int, slug: str, length: int, sample_index: int) -> int:
    key = f"{base_seed}|{slug}|{length}|{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def load_subdomain_weights(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"subdomain weight file must contain a JSON object: {path}")
    raw_weights = payload.get("weights", payload)
    if not isinstance(raw_weights, dict):
        raise ValueError(f"subdomain weight file has no object-valued `weights`: {path}")
    weights: dict[str, float] = {}
    for raw_slug, raw_weight in raw_weights.items():
        slug = str(raw_slug).strip()
        if not slug:
            continue
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"subdomain weight must be finite and positive: {slug}={raw_weight}")
        weights[slug] = weight
    return weights


def allocate_weighted_domain_counts(
    slugs: list[str],
    count_per_pair: int,
    weights: dict[str, float],
) -> dict[str, int]:
    target = int(count_per_pair) * len(slugs)
    effective = {slug: float(weights.get(slug, 1.0)) for slug in slugs}
    total_weight = sum(effective.values())
    if total_weight <= 0:
        raise ValueError("sum of effective subdomain weights must be positive")
    ideals = {slug: target * effective[slug] / total_weight for slug in slugs}
    counts = {slug: max(1, int(math.floor(ideals[slug]))) for slug in slugs}
    delta = target - sum(counts.values())
    if delta > 0:
        order = sorted(
            slugs, key=lambda slug: (ideals[slug] - math.floor(ideals[slug]), slug), reverse=True
        )
        for index in range(delta):
            counts[order[index % len(order)]] += 1
    elif delta < 0:
        order = sorted(slugs, key=lambda slug: (ideals[slug] - math.floor(ideals[slug]), slug))
        remaining = -delta
        while remaining:
            changed = False
            for slug in order:
                if counts[slug] <= 1:
                    continue
                counts[slug] -= 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
            if not changed:
                raise ValueError(
                    "unable to allocate at least one chain to every weighted subdomain"
                )
    return counts


def build_weighted_pair_schedule(
    *,
    slugs: list[str],
    lengths: list[int],
    domain_counts: dict[str, int],
    generation_order: str,
) -> list[tuple[str, int, int]]:
    if generation_order == "by_domain":
        return [
            (slug, length, sample_index)
            for slug in slugs
            for sample_index in range(domain_counts[slug])
            for length in lengths
        ]

    pair_order = round_robin_pair_order(slugs, lengths)
    heap: list[tuple[float, int, str, int, int, int]] = []
    for rank, (slug, length) in enumerate(pair_order):
        quota = domain_counts[slug]
        heapq.heappush(heap, (0.0, rank, slug, length, 0, quota))
    schedule: list[tuple[str, int, int]] = []
    while heap:
        _priority, rank, slug, length, sample_index, quota = heapq.heappop(heap)
        schedule.append((slug, length, sample_index))
        next_index = sample_index + 1
        if next_index < quota:
            heapq.heappush(
                heap,
                (next_index / quota, rank, slug, length, next_index, quota),
            )
    return schedule


class LiveSampledChainSequence:
    """List-like on-demand chain sampler used to keep LLM workers saturated.

    The generator's existing executor asks for chains by integer index. This
    class keeps that contract while sampling the actual walk only when the item
    is requested, so no large domain_chains JSONL files are written.
    """

    def __init__(
        self,
        *,
        graph_root: Path,
        subdomains: list[str],
        chain_lengths: list[int],
        chains_per_subdomain_length: int,
        seed: int,
        max_attempts: int,
        generation_order: str,
        subdomain_weights: dict[str, float] | None,
        limit: int,
    ) -> None:
        if chains_per_subdomain_length <= 0:
            raise ValueError("--chains-per-subdomain-length must be positive for live sampling")
        self.graph_root = Path(graph_root)
        self.lengths = list(chain_lengths)
        self.count_per_pair = int(chains_per_subdomain_length)
        self.seed = int(seed)
        self.max_attempts = int(max_attempts)
        self.graphs: dict[str, dict[str, Any]] = {}

        index_items = load_index(self.graph_root)
        requested = {slug.strip() for slug in subdomains if slug and slug.strip()}
        available = {str(item["slug"]) for item in index_items}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown subdomains: {sorted(unknown)}")
        if requested:
            index_items = [item for item in index_items if str(item.get("slug")) in requested]
        if not index_items:
            raise RuntimeError(
                f"no graph index entries matched subdomains={sorted(requested) or 'ALL'}"
            )

        for item in index_items:
            slug = str(item["slug"])
            nodes_path = self.graph_root / str(item["nodes_file"])
            edges_path = self.graph_root / str(item["edges_file"])
            nodes_list = read_jsonl(nodes_path)
            edges_list = read_jsonl(edges_path)
            nodes = {
                str(node["id"]): node
                for node in nodes_list
                if isinstance(node, dict) and node.get("id")
            }
            adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for edge in edges_list:
                if not isinstance(edge, dict):
                    continue
                source = str(edge.get("source", ""))
                target = str(edge.get("target", ""))
                if source in nodes and target in nodes:
                    adjacency[source].append(edge)
            self.graphs[slug] = {
                "item": item,
                "nodes": nodes,
                "adjacency": adjacency,
                "edge_count": len(edges_list),
            }

        slugs = [str(item["slug"]) for item in index_items]
        self.domain_counts = allocate_weighted_domain_counts(
            slugs,
            self.count_per_pair,
            subdomain_weights or {},
        )
        self.schedule = build_weighted_pair_schedule(
            slugs=slugs,
            lengths=self.lengths,
            domain_counts=self.domain_counts,
            generation_order=generation_order,
        )
        self.effective_weights = {
            slug: float((subdomain_weights or {}).get(slug, 1.0)) for slug in slugs
        }
        self.full_count = len(self.schedule)
        self.total_count = (
            min(self.full_count, int(limit)) if limit and limit > 0 else self.full_count
        )

    def __len__(self) -> int:
        return self.total_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slug, length, sample_index = self.schedule[index]
        graph = self.graphs[slug]
        item = graph["item"]
        nodes = graph["nodes"]
        adjacency = graph["adjacency"]
        rng = random.Random(stable_chain_seed(self.seed, slug, length, sample_index))
        operator_ids, edge_trace = choose_walk(
            nodes=nodes,
            adjacency=adjacency,
            rng=rng,
            length=length,
            max_attempts=self.max_attempts,
        )
        operator_nodes = [compact_node(nodes[node_id]) for node_id in operator_ids]
        compact_edges = [compact_edge(edge) for edge in edge_trace]
        subdomain = str(item.get("subdomain") or slug)
        env_id = str(item.get("env_id") or "").strip()
        seed_id = (
            f"SCIDATA-V2-{safe_slug(slug).upper().replace('_', '-')}"
            f"-L{length:02d}-{sample_index + 1:05d}"
        )
        return {
            "scicode_seed_id": seed_id,
            "chain_id": seed_id,
            "length": length,
            "domain": operator_nodes[0].get("domain") if operator_nodes else "",
            "subdomain": subdomain,
            "subdomain_slug": slug,
            "operator_runtime_env_id": env_id,
            "operator_runtime_python": "",
            "operator_runtime_usage_doc": "README.md#install",
            "operator_ids": operator_ids,
            "operator_nodes": operator_nodes,
            "edge_trace": compact_edges,
            "domain_scenario": f"{subdomain} workflow induced by a compatible operator walk.",
            "physical_scenario": f"{subdomain} workflow induced by a compatible operator walk.",
            "reason": summarize_chain_reason(operator_nodes, compact_edges),
            "verification_hints": [
                "Use only this subdomain's operator graph as hidden inspiration.",
                "Translate specialized operators into student-solvable NumPy/SciPy/math computations.",
                "If hidden operator execution is needed for chain/runtime inspection, use the recorded operator runtime environment, not the student solution environment.",
                "Do not expose library names or operator API symbols in student-facing text.",
            ],
            "implementation_risks": [
                "Avoid making the student solution a direct wrapper around the hidden API symbol.",
                "Avoid tests that only compare shapes or types.",
            ],
        }

    def sampling_summary(self) -> dict[str, Any]:
        return {
            "count_per_pair_baseline": self.count_per_pair,
            "chain_lengths": self.lengths,
            "effective_weights": self.effective_weights,
            "chains_per_subdomain_per_length": self.domain_counts,
            "full_count": self.full_count,
            "scheduled_count": self.total_count,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_index(graph_root: Path) -> list[dict[str, Any]]:
    index_path = graph_root / "index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    subdomains = data.get("subdomains", [])
    if not isinstance(subdomains, list):
        raise ValueError(f"invalid graph index: {index_path}")
    return [item for item in subdomains if isinstance(item, dict)]


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "id",
        "domain",
        "subdomain",
        "library",
        "api_symbol",
        "source_operator",
        "source_operator_id",
        "source_operator_kind",
        "operation_mode",
        "inputs",
        "outputs",
        "input_types",
        "output_types",
        "input_tokens",
        "output_tokens",
        "catalog_source",
    ]
    compact = {key: node.get(key) for key in keep if key in node}
    runtime = node.get("runtime_validation")
    if isinstance(runtime, dict) and runtime.get("env_id"):
        compact["runtime_env_id"] = runtime.get("env_id")
        compact["runtime_validation_status"] = runtime.get("validation_status") or runtime.get(
            "call_status"
        )
    return compact


def compact_edge(edge: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "source",
        "target",
        "source_api_symbol",
        "target_api_symbol",
        "source_library",
        "target_library",
        "source_operation_mode",
        "target_operation_mode",
        "matched_input",
        "matched_output",
        "matched_types",
        "matched_tokens",
        "workflow_reasons",
        "score",
    ]
    return {key: edge.get(key) for key in keep if key in edge}


def safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def choose_walk(
    *,
    nodes: dict[str, dict[str, Any]],
    adjacency: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    length: int,
    max_attempts: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    starts = [node_id for node_id in sorted(adjacency) if node_id in nodes]
    if not starts:
        raise ValueError("graph_has_no_out_edges")
    for _ in range(max_attempts):
        current = rng.choice(starts)
        chain = [current]
        edges: list[dict[str, Any]] = []
        seen = {current}
        for _step in range(length - 1):
            candidates = [
                edge
                for edge in adjacency.get(current, [])
                if edge.get("target") in nodes and edge.get("target") not in seen
            ]
            if not candidates:
                candidates = [
                    edge for edge in adjacency.get(current, []) if edge.get("target") in nodes
                ]
            if not candidates:
                break
            candidates = sorted(
                candidates,
                key=lambda edge: (float(edge.get("score", 0) or 0), str(edge.get("target", ""))),
                reverse=True,
            )
            top = candidates[: min(12, len(candidates))]
            edge = rng.choice(top)
            target = str(edge["target"])
            edges.append(edge)
            chain.append(target)
            seen.add(target)
            current = target
        if len(chain) == length:
            return chain, edges
    raise ValueError(f"could_not_sample_length_{length}_walk")


def summarize_chain_reason(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, node in enumerate(nodes, start=1):
        symbol = node.get("api_symbol") or node.get("source_operator") or node.get("id")
        mode = node.get("operation_mode") or "operator"
        library = node.get("library") or "scientific library"
        parts.append(f"{idx}. {library} {symbol} as {mode}")
    edge_bits = []
    for edge in edges:
        matched = edge.get("matched_output") or edge.get("matched_tokens") or ""
        need = edge.get("matched_input") or ""
        if matched or need:
            edge_bits.append(f"{matched} -> {need}".strip())
    if edge_bits:
        return "Operator roles: " + "; ".join(parts) + ". IO links: " + "; ".join(edge_bits[:4])
    return "Operator roles: " + "; ".join(parts)


def round_robin_pair_order(slugs: list[str], lengths: list[int]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    if not slugs or not lengths:
        return pairs
    for offset in range(len(lengths)):
        for domain_index, slug in enumerate(slugs):
            pair = (slug, lengths[(domain_index + offset) % len(lengths)])
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    for slug in slugs:
        for length in lengths:
            pair = (slug, length)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs
