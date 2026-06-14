#!/usr/bin/env bash
# Launch PI0 policy DDS runner only (terminal 2; requires sim_env in another terminal).
set -euo pipefail

CKPT_PATH="${CKPT_PATH:-nvidia/Liver_Scan_Pi0_Cosmos_Rel}"
REPO_ID="${REPO_ID:-i4h/sim_liver_scan}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"

exec python3 -m policy.run_policy \
  --policy pi0 \
  --ckpt_path "${CKPT_PATH}" \
  --repo_id "${REPO_ID}" \
  "$@"
