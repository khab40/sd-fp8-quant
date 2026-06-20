#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat <<EOF
This workflow has a manual server step because hidden-state extraction needs a
long-running vLLM process on the H100.

1. ./scripts/bootstrap_envs.sh
2. ./scripts/prepare_data.sh
3. In terminal A: ./scripts/launch_hidden_state_server.sh
4. In terminal B: ./scripts/generate_hidden_states.sh
5. Stop terminal A.
6. ./scripts/train_eagle3.sh
7. ./scripts/quantize_fp8_dynamic.sh
8. ./scripts/make_fp8_speculator_checkpoint.sh
9. ./scripts/benchmark_commands.sh
EOF
