from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/environments.json"))
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--env", action="append", help="Environment name; repeat to select multiple"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/environments.json"))
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Index already installed environments without installing packages",
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    envs = config["environments"]
    selected = args.env or list(envs)
    unknown = set(selected) - envs.keys()
    if unknown:
        parser.error(f"Unknown environments: {sorted(unknown)}")
    project_root = args.config.resolve().parent.parent
    index = {"environments": {}, "subdomain_to_env": {}}
    for env_id in selected:
        info = envs[env_id]
        venv = (args.venv_root / env_id).resolve()
        python = venv / "bin/python"
        if not args.index_only:
            if not python.is_file():
                subprocess.run([args.python, "-m", "venv", str(venv)], check=True)
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(project_root / info["requirements"]),
                ],
                check=True,
            )
        if not python.is_file():
            raise FileNotFoundError(f"Missing interpreter: {python}")
        record = {"env_id": env_id, "venv_path": str(venv), "python": str(python)}
        index["environments"][env_id] = record
        for subdomain in info["subdomains"]:
            index["subdomain_to_env"][subdomain] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(index['environments'])} environments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
