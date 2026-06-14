#!/usr/bin/env bash
# Debug PI0 infer() only — no Isaac Sim. Breakpoints hit in ~30s after attach.
set -euo pipefail

DEBUG_PORT="${DEBUG_PORT:-5678}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"

if [[ -d "/root/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel" ]]; then
  export HF_HUB_OFFLINE=1
fi

python3 -m pip install -q debugpy 2>/dev/null || true

echo "Listening on 0.0.0.0:${DEBUG_PORT} — attach debugger, then policy-only infer runs."
echo "Breakpoints:"
echo "  policy/pi0/runners.py:46  infer()"
echo "  third_party/openpi/src/openpi/policies/policy.py  infer()"
echo "(No Isaac Sim — skips 5-15 min sim startup)"

exec python3 -Xfrozen_modules=off -m debugpy \
  --listen "0.0.0.0:${DEBUG_PORT}" \
  --wait-for-client \
  debug/debug_infer_policy_only.py
