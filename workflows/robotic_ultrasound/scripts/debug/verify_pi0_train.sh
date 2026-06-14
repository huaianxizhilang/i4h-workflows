#!/usr/bin/env bash
# Smoke-test LoRA training: mock data → norm stats → 2 train steps (no debugpy).
set -euo pipefail

REPO_ID="${REPO_ID:-i4h/debug_lora_mock}"
CONFIG="${CONFIG:-robotic_ultrasound_lora}"
EXP_NAME="${EXP_NAME:-debug_lora_smoke}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
MOCK_DATA_DIR="${MOCK_DATA_DIR:-/data/cache/i4h-mock-train/hdf5}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export JAX_LOG_LEVEL="${JAX_LOG_LEVEL:-WARNING}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

echo "=== 1. Prepare mock LeRobot dataset (if missing) ==="
python3 debug/prepare_mock_train_data.py \
  --repo_id "${REPO_ID}" \
  --data_dir "${MOCK_DATA_DIR}"

echo "=== 2. LoRA config + norm stats ==="
python3 -c "
import dataclasses
from policy.pi0.config import get_config
from policy.pi0.utils import compute_normalization_stats
from training.pi_zero.train import ensure_norm_stats_exist

cfg = get_config('${CONFIG}', '${REPO_ID}', '${EXP_NAME}')
ensure_norm_stats_exist(cfg)
print('TrainConfig OK:', cfg.name, 'repo_id:', cfg.data.repo_id)
"

echo "=== 3. Run ${TRAIN_STEPS} training steps (downloads pi0_base on first run) ==="
python3 -c "
import dataclasses
from openpi import train
from policy.pi0.config import get_config
from training.pi_zero.train import ensure_norm_stats_exist

cfg = get_config('${CONFIG}', '${REPO_ID}', '${EXP_NAME}')
cfg = dataclasses.replace(cfg, num_train_steps=${TRAIN_STEPS})
ensure_norm_stats_exist(cfg)
print('[train-smoke] starting', cfg.num_train_steps, 'steps...')
train.main(cfg)
print('[train-smoke] done')
"

echo "=== TRAIN SMOKE PASSED ==="
