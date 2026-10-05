from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ..paths import resource_path


def slugify(text: str) -> str:
    text = text.lower().replace("/", "_").replace("&", "and")
    out = []
    last_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_")


@lru_cache(maxsize=1)
def catalog_validator() -> Draft202012Validator:
    schema = json.loads(resource_path("operator_catalog.schema.json").read_text())
    return Draft202012Validator(schema)


def read_catalog(path: Path) -> list[dict]:
    records = []
    identifiers = {}
    subdomains = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        label = f"{path}:{line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}: invalid JSON: {exc.msg}") from exc
        errors = sorted(catalog_validator().iter_errors(record), key=lambda e: str(e.json_path))
        if errors:
            error = errors[0]
            raise ValueError(f"{label}: {error.json_path}: {error.message}")
        if record["id"] in identifiers:
            raise ValueError(
                f"{label}: duplicate id {record['id']!r}; first seen on line "
                f"{identifiers[record['id']]}"
            )
        slug = slugify(record["subdomain"])
        if not slug:
            raise ValueError(f"{label}: subdomain must produce a nonempty directory name")
        identity = (record["domain"], record["subdomain"])
        if slug in subdomains and subdomains[slug] != identity:
            raise ValueError(f"{label}: subdomain directory collision for {slug!r}")
        identifiers[record["id"]] = line_number
        subdomains[slug] = identity
        records.append(record)
    if not records:
        raise ValueError(f"{path}: the input catalog contains no operator families")
    return records
