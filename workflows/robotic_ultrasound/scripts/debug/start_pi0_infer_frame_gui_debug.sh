#!/usr/bin/env bash
# Frame-step debug: Isaac Sim viewport + inline DearPyGui (single process, true unified freeze).
set -euo pipefail

DEBUG_PORT="${DEBUG_PORT:-5678}"
CKPT_PATH="${CKPT_PATH:-nvidia/Liver_Scan_Pi0_Cosmos_Rel}"
REPO_ID="${REPO_ID:-i4h/sim_liver_scan}"

# I4H_DEBUG_STOP: infer | step | both | all | off
export I4H_DEBUG_STOP="${I4H_DEBUG_STOP:-both}"
export I4H_GUI="${I4H_GUI:-1}"
export I4H_INLINE_PANEL="${I4H_INLINE_PANEL:-1}"
export I4H_USE_DEBUGPY="${I4H_USE_DEBUGPY:-1}"
# Silence JAX/TF compile spam in terminal (first infer still ~20s, logs hidden)
export JAX_LOG_LEVEL="${JAX_LOG_LEVEL:-WARNING}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
source "${PWD}/debug/env.sh"

if [[ -d "/root/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel" ]]; then
  export HF_HUB_OFFLINE=1
fi

python3 -m pip install -q debugpy 2>/dev/null || true

echo "=== infer-frame-gui ==="
echo "DISPLAY=${DISPLAY:-unset}  I4H_GUI=${I4H_GUI}  I4H_DEBUG_STOP=${I4H_DEBUG_STOP}"
echo "VNC: connect TigerVNC :5901 to see Isaac Sim viewport + inline debug panel"
echo "I4H_DEBUG_RESET=1 to also pause during 40 reset steps (default: skip)"
echo "Listening on 0.0.0.0:${DEBUG_PORT}"

exec python3 -Xfrozen_modules=off -m debugpy \
  --listen "0.0.0.0:${DEBUG_PORT}" \
  --wait-for-client \
  debug/debug_infer_frame_gui.py \
  --enable_cameras \
  --ckpt_path "${CKPT_PATH}" \
  --repo_id "${REPO_ID}"
