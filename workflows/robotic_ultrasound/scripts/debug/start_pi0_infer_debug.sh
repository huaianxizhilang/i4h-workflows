#!/usr/bin/env bash
# Start PI0 inference debug server (debugpy) inside the container.
# Attach from Cursor/VS Code with launch config "PI0: Attach Inference".
set -euo pipefail

DEBUG_PORT="${DEBUG_PORT:-5678}"
CKPT_PATH="${CKPT_PATH:-nvidia/Liver_Scan_Pi0_Cosmos_Rel}"
REPO_ID="${REPO_ID:-i4h/sim_liver_scan}"
HEADLESS="${HEADLESS:-1}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"
# Prefer offline cache (populated by sync-hf-cache on host)
if [[ -d "/root/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel" ]]; then
  export HF_HUB_OFFLINE=1
fi
python3 -m pip install -q debugpy

EXTRA_ARGS=()
if [[ "${HEADLESS}" == "1" ]]; then
  EXTRA_ARGS+=(--headless)
fi

echo "Listening on 0.0.0.0:${DEBUG_PORT} — attach debugger, then execution continues."
echo "Module: simulation.imitation_learning.pi0_policy.eval"
echo "Suggested breakpoints:"
echo "  policy/pi0/runners.py:56"
echo "  third_party/openpi/src/openpi/policies/policy.py (infer)"

exec python3 -Xfrozen_modules=off -m debugpy \
  --listen "0.0.0.0:${DEBUG_PORT}" \
  --wait-for-client \
  -m simulation.imitation_learning.pi0_policy.eval \
  --enable_cameras \
  "${EXTRA_ARGS[@]}" \
  --ckpt_path "${CKPT_PATH}" \
  --repo_id "${REPO_ID}"
