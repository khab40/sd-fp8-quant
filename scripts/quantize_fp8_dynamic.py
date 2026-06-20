#!/usr/bin/env python3
"""Apply llmcompressor FP8 dynamic quantization to Qwen/Qwen3-8B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", default="models/Qwen3-8B-FP8-Dynamic")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    if "quantization_config" not in config:
        raise RuntimeError(f"{config_path} does not contain quantization_config")

    print(f"Saved FP8 dynamic model to {output}")


if __name__ == "__main__":
    main()
