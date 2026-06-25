#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}"

if [[ ! -d "${SPECULATORS_REPO}/.git" ]]; then
  git clone https://github.com/vllm-project/speculators.git "${SPECULATORS_REPO}"
fi

git -C "${SPECULATORS_REPO}" fetch --tags
git -C "${SPECULATORS_REPO}" checkout v0.5.0

"${PYTHON_BIN}" -m venv "${ROOT_DIR}/speculators_venv"
"${ROOT_DIR}/speculators_venv/bin/python" -m pip install --upgrade pip
"${ROOT_DIR}/speculators_venv/bin/python" -m pip install -e "${SPECULATORS_REPO}"
"${ROOT_DIR}/speculators_venv/bin/python" -m pip install tensorboard datasets transformers accelerate safetensors

"${PYTHON_BIN}" -m venv "${ROOT_DIR}/vllm_venv"
"${ROOT_DIR}/vllm_venv/bin/python" -m pip install --upgrade pip
"${ROOT_DIR}/vllm_venv/bin/python" -m pip install "vllm[bench]==0.20.0" "fastapi<0.137"

"${PYTHON_BIN}" -m venv "${ROOT_DIR}/comp_venv"
"${ROOT_DIR}/comp_venv/bin/python" -m pip install --upgrade pip
"${ROOT_DIR}/comp_venv/bin/python" -m pip install "llmcompressor==0.12.0" transformers accelerate safetensors

cat <<EOF
Environment setup complete.

Created:
  ${ROOT_DIR}/speculators_venv
  ${ROOT_DIR}/vllm_venv
  ${ROOT_DIR}/comp_venv

Speculators source:
  ${SPECULATORS_REPO}
EOF
