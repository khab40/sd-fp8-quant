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

    group = next(iter(qconfig.get("config_groups", {}).values()), None)
    if not group:
        raise SystemExit(f"Missing quantization config groups in {config_path}")

    weights = group.get("weights") or {}
    input_activations = group.get("input_activations") or {}
    checks = {
        "quant_method is compressed-tensors": qconfig.get("quant_method")
        == "compressed-tensors",
        "quantization_status is compressed": qconfig.get("quantization_status")
        == "compressed",
        "lm_head ignored": "lm_head" in qconfig.get("ignore", []),
        "weights are 8-bit float": weights.get("type") == "float"
        and weights.get("num_bits") == 8,
        "input activations are dynamic 8-bit float": input_activations.get("type")
        == "float"
        and input_activations.get("num_bits") == 8
        and input_activations.get("dynamic") is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    text = json.dumps(qconfig, indent=2, sort_keys=True)
    if failed:
        raise SystemExit(
            f"Quantization config exists, but semantic checks failed: {failed}\n{text}"
        )

    print(text)


if __name__ == "__main__":
    main()
