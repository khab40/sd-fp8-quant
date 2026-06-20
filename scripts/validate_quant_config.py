#!/usr/bin/env python3
"""Validate that a saved model contains llm-compressor FP8 metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    config_path = Path(args.model_dir) / "config.json"
    config = json.loads(config_path.read_text())
    qconfig = config.get("quantization_config")
    if not qconfig:
        raise SystemExit(f"Missing quantization_config in {config_path}")

    text = json.dumps(qconfig, indent=2, sort_keys=True)
    required_fragments = ["compressed", "fp8"]
    missing = [fragment for fragment in required_fragments if fragment not in text.lower()]
    if missing:
        raise SystemExit(
            f"Quantization config exists, but expected fragments are missing: {missing}\n{text}"
        )

    print(text)


if __name__ == "__main__":
    main()
