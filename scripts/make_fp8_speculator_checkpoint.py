#!/usr/bin/env python3
"""Copy a speculator checkpoint and point its verifier config at the FP8 model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Trained speculator checkpoint")
    parser.add_argument("--verifier", required=True, help="FP8 verifier model directory")
    parser.add_argument("--output", required=True, help="Patched checkpoint directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    verifier = Path(args.verifier).resolve()
    output = Path(args.output)

    if not source.exists():
        raise SystemExit(f"Source checkpoint does not exist: {source}")
    if not verifier.exists():
        raise SystemExit(f"Verifier model does not exist: {verifier}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}")
        shutil.rmtree(output)

    shutil.copytree(source, output, symlinks=False)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    try:
        config["speculators_config"]["verifier"]["name_or_path"] = str(verifier)
    except KeyError as exc:
        raise SystemExit(
            f"{config_path} is not a Speculators checkpoint config; missing {exc}"
        ) from exc

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"Patched verifier path in {config_path} -> {verifier}")


if __name__ == "__main__":
    main()
