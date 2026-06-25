#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

MODE="${1:-baseline}"

set_speculative_tokens() {
  local checkpoint_dir="$1"
  local speculative_tokens="$2"

  "${ROOT_DIR}/vllm_venv/bin/python" - "$checkpoint_dir" "$speculative_tokens" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]) / "config.json"
speculative_tokens = int(sys.argv[2])

config = json.loads(config_path.read_text())
proposal_methods = config["speculators_config"]["proposal_methods"]
for proposal_method in proposal_methods:
    proposal_method["speculative_tokens"] = speculative_tokens
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
print(f"Set speculative_tokens={speculative_tokens} in {config_path}")
PY
}

case "${MODE}" in
  baseline)
    SERVE_MODEL="${MODEL_ID}"
    ;;
  fp8)
    SERVE_MODEL="${FP8_MODEL_DIR}"
    ;;
  spec)
    SERVE_MODEL="${CHECKPOINT_DIR}/checkpoint_best"
    set_speculative_tokens "${SERVE_MODEL}" "${SPEC_TOKENS}"
    ;;
  fp8-spec)
    SERVE_MODEL="${FP8_SPECULATOR_DIR}"
    set_speculative_tokens "${SERVE_MODEL}" "${FP8_SPEC_TOKENS}"
    ;;
  *)
    echo "Usage: $0 {baseline|spec|fp8|fp8-spec}" >&2
    exit 2
    ;;
esac

SERVE_ARGS=(
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --generation-config "${VLLM_GENERATION_CONFIG}"
)

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  SERVE_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  SERVE_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  SERVE_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${VLLM_COMPILATION_CONFIG}" ]]; then
  SERVE_ARGS+=(--compilation-config "${VLLM_COMPILATION_CONFIG}")
fi

EXTRA_ARGS=()
if [[ -n "${VLLM_SERVE_EXTRA_ARGS:-}" ]]; then
  # Intentionally preserve the old shell-style override for simple flag lists.
  # Do not use it for JSON arguments with spaces; use the dedicated env vars above.
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${VLLM_SERVE_EXTRA_ARGS})
fi

exec "${ROOT_DIR}/vllm_venv/bin/vllm" serve "${SERVE_MODEL}" \
  "${SERVE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
