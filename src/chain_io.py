from __future__ import annotations

import argparse
from array import array
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .sampling import LiveSampledChainSequence, load_subdomain_weights, parse_chain_lengths


FORMAT = "scidata.chains.v1"


def save_chains(
    chains: LiveSampledChainSequence, output_dir: Path, parameters: dict[str, Any]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty chain output directory: {output_dir}")
    temporary = output_dir / "chains.jsonl.tmp"
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            for index in range(len(chains)):
                # Preserve dictionary/list order as well as every sampler field.
                line = (json.dumps(chains[index], ensure_ascii=False) + "\n").encode("utf-8")
                handle.write(line)
                digest.update(line)
        manifest = {
            "format": FORMAT,
            "chains_file": "chains.jsonl",
            "count": len(chains),
            "sha256": digest.hexdigest(),
            "parameters": parameters,
            "sampling": chains.sampling_summary(),
        }
        temporary.rename(output_dir / "chains.jsonl")
        (output_dir / "sampling_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


class SavedChainSequence:
    """Checksum-verified JSONL sequence with independent reads for concurrent workers."""

    def __init__(self, path: Path, manifest_path: Path | None = None, limit: int = 0):
        if limit < 0:
            raise ValueError("Chain limit must be nonnegative")
        self.path = path.resolve()
        initial_stat = self.path.stat()
        self.file_state = (initial_stat.st_size, initial_stat.st_mtime_ns)
        manifest_path = manifest_path or self.path.with_name("sampling_manifest.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError("Unsupported saved-chain format")
        self.offsets = array("Q")
        digest = hashlib.sha256()
        identities: set[str] = set()
        required = {
            "scicode_seed_id",
            "subdomain_slug",
            "subdomain",
            "length",
            "operator_ids",
            "operator_nodes",
            "edge_trace",
            "domain_scenario",
        }
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                digest.update(line)
                if not line.strip():
                    raise ValueError(f"Blank saved-chain record at line {line_number}")
                record = json.loads(line)
                if not isinstance(record, dict) or not required.issubset(record):
                    raise ValueError(f"Invalid saved chain at line {line_number}")
                identity = record["scicode_seed_id"]
                if not isinstance(identity, str) or not identity or identity in identities:
                    raise ValueError(f"Missing or duplicate chain ID at line {line_number}")
                if record["length"] != len(record["operator_ids"]):
                    raise ValueError(f"Chain length mismatch at line {line_number}")
                identities.add(identity)
                self.offsets.append(offset)
        if digest.hexdigest() != self.manifest.get("sha256"):
            raise ValueError("Saved-chain checksum mismatch")
        if len(self.offsets) != self.manifest.get("count") or not self.offsets:
            raise ValueError("Saved-chain count mismatch or empty chain file")
        final_stat = self.path.stat()
        if self.file_state != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError("Saved-chain file changed while indexing")
        self.count = min(limit, len(self.offsets)) if limit else len(self.offsets)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        current_stat = self.path.stat()
        if self.file_state != (current_stat.st_size, current_stat.st_mtime_ns):
            raise ValueError("Saved-chain file changed after validation")
        # No shared seek position across threads and no full-chain cache in memory.
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def sampling_summary(self) -> dict[str, Any]:
        summary = copy.deepcopy(self.manifest["sampling"])
        summary["scheduled_count"] = len(self)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subdomain", action="append", default=[])
    parser.add_argument("--chain-lengths", default="3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--chains-per-subdomain-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026071001)
    parser.add_argument("--subdomain-weights", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
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
    parameters = {
        "graph_root": str(args.graph_root.resolve()),
        "subdomains": args.subdomain,
        "chain_lengths": chains.lengths,
        "seed": args.seed,
        "max_attempts": 200,
        "generation_order": "round_robin",
        "chains_per_subdomain_length": args.chains_per_subdomain_length,
        "subdomain_weights": chains.effective_weights,
        "limit": args.limit,
    }
    manifest = save_chains(chains, args.output_dir, parameters)
    print(f"Saved {manifest['count']} chains to {args.output_dir / 'chains.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
