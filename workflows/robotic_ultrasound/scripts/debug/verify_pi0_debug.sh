#!/usr/bin/env bash
# Smoke-test PI0 inference + LoRA training debug readiness inside the container.
# Fast by default: no HuggingFace download. Model load uses local cache only.
set -euo pipefail

CKPT_ID="${CKPT_ID:-nvidia/Liver_Scan_Pi0_Cosmos_Rel}"
LOAD_MODEL="${LOAD_MODEL:-auto}"  # auto | 1 | 0

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"

hf_cache_dir() {
  python3 -c "import os; from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)"
}

ckpt_cached() {
  local cache
  cache="$(hf_cache_dir)"
  local slug="models--${CKPT_ID//\//--}"
  [[ -d "${cache}/${slug}/snapshots" ]] && [[ -n "$(ls -A "${cache}/${slug}/snapshots" 2>/dev/null)" ]]
}

echo "=== 1. openpi source ==="
ls /workspace/i4h-workflows/third_party/openpi/src/openpi/models/pi0.py
python3 -c "from openpi.policies import policy_config; from openpi import train; print('policy_config:', policy_config.__file__); print('train:', train.__file__)"

echo "=== 2. debugpy ==="
python3 -m pip install -q debugpy
python3 -c "import debugpy; print('debugpy', debugpy.__version__)"

echo "=== 3. LoRA config ==="
python3 -c "
from policy.pi0.config import get_config
cfg = get_config('robotic_ultrasound_lora', 'i4h/robotic_ultrasound', 'debug_smoke')
print('TrainConfig OK, name:', cfg.name)
"

echo "=== 4. train import ==="
python3 -c "from training.pi_zero.train import parse_args; print('train.py OK')"

echo "=== 5. HF cache status ==="
if ckpt_cached; then
  echo "Checkpoint cache HIT: ${CKPT_ID}"
  du -sh "$(hf_cache_dir)/models--${CKPT_ID//\//--}" 2>/dev/null || true
else
  echo "Checkpoint cache MISS: ${CKPT_ID}"
  echo "Run on host: bash deploy/debug/pi0-debug.sh sync-hf-cache"
fi

should_load="0"
case "${LOAD_MODEL}" in
  1|true|yes) should_load="1" ;;
  0|false|no) should_load="0" ;;
  auto) ckpt_cached && should_load="1" ;;
esac

if [[ "${should_load}" == "1" ]]; then
  echo "=== 6. PI0PolicyRunner load (offline, local cache) ==="
  export HF_HUB_OFFLINE=1
  python3 -c "
from policy.pi0.runners import PI0PolicyRunner
from simulation.utils.common import resolve_checkpoint_path
ckpt = resolve_checkpoint_path('${CKPT_ID}')
print('ckpt:', ckpt)
r = PI0PolicyRunner(ckpt_path=ckpt, repo_id='i4h/sim_liver_scan')
print('PI0PolicyRunner OK, model:', type(r.model).__name__)
"
else
  echo "=== 6. PI0PolicyRunner load SKIPPED (no local cache or LOAD_MODEL=0) ==="
  python3 -c "
from policy.pi0.runners import PI0PolicyRunner
from policy.pi0.config import get_config
print('PI0PolicyRunner import OK')
cfg = get_config('robotic_ultrasound', 'i4h/sim_liver_scan')
print('inference config OK:', cfg.name)
"
fi

echo "=== ALL CHECKS PASSED ==="
