#!/usr/bin/env bash
# Start PI0 LoRA training debug server (debugpy) inside the container.
# Attach from Cursor/VS Code with launch config "PI0: Attach LoRA Train".
set -euo pipefail

DEBUG_PORT="${DEBUG_PORT:-5679}"
CONFIG="${CONFIG:-robotic_ultrasound_lora}"
EXP_NAME="${EXP_NAME:-debug_lora}"
REPO_ID="${REPO_ID:-i4h/debug_lora_mock}"
MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export JAX_LOG_LEVEL="${JAX_LOG_LEVEL:-WARNING}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
python3 -m pip install -q debugpy

echo "Listening on 0.0.0.0:${DEBUG_PORT} — attach debugger, then training starts."
echo "Module: training.pi_zero.train"
echo "Suggested breakpoints:"
echo "  training/pi_zero/train.py:61"
echo "  third_party/openpi/src/openpi/train.py (main)"
echo "  third_party/openpi/src/openpi/training/ (train step loop)"

exec python3 -Xfrozen_modules=off -m debugpy \
  --listen "0.0.0.0:${DEBUG_PORT}" \
  --wait-for-client \
  -m training.pi_zero.train \
  --config "${CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --repo_id "${REPO_ID}"
