#!/usr/bin/env bash
# Full-pipeline visuals (DDS): sim + DearPyGUI vis + ultrasound + policy with SIGSTOP sync.
set -euo pipefail

DEBUG_PORT="${DEBUG_PORT:-5678}"
CKPT_PATH="${CKPT_PATH:-nvidia/Liver_Scan_Pi0_Cosmos_Rel}"
SIM_WAIT_SEC="${SIM_WAIT_SEC:-120}"
export I4H_DEBUG_STOP="${I4H_DEBUG_STOP:-both}"
export I4H_USE_DEBUGPY="${I4H_USE_DEBUGPY:-1}"
export JAX_LOG_LEVEL="${JAX_LOG_LEVEL:-WARNING}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
source "${PWD}/debug/env.sh"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [[ -d "/root/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel" ]]; then
  export HF_HUB_OFFLINE=1
fi

python3 -m pip install -q debugpy 2>/dev/null || true

echo "=== infer-full-frame (DDS multi-process) ==="
echo "DISPLAY=${DISPLAY:-unset}"
echo "Starting visualization + ultrasound + sim_with_dds ..."
echo "Wait ${SIM_WAIT_SEC}s for Isaac Sim, then attach debugger on port ${DEBUG_PORT}"

python3 -m utils.visualization &
VIS_PID=$!
python3 -m simulation.examples.ultrasound_raytracing &
US_PID=$!
python3 -m simulation.environments.sim_with_dds --enable_cameras &
SIM_PID=$!

export I4H_CHILD_PIDS="${VIS_PID} ${US_PID} ${SIM_PID}"
echo "Child PIDs: ${I4H_CHILD_PIDS}"

echo "Waiting ${SIM_WAIT_SEC}s for sim startup..."
sleep "${SIM_WAIT_SEC}"

cleanup() {
  kill ${VIS_PID} ${US_PID} ${SIM_PID} 2>/dev/null || true
}
trap cleanup EXIT

echo "VNC: TigerVNC :5901 — expect Isaac Sim + Robotic Ultrasound Visualization windows"
echo "Listening on 0.0.0.0:${DEBUG_PORT} (policy DDS frame stepper)"

exec python3 -Xfrozen_modules=off -m debugpy \
  --listen "0.0.0.0:${DEBUG_PORT}" \
  --wait-for-client \
  debug/debug_policy_dds_frame.py \
  --policy pi0 \
  --ckpt_path "${CKPT_PATH}"
