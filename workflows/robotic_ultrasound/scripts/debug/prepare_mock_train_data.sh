#!/usr/bin/env bash
# Host/container helper: create mock HDF5 + LeRobot dataset for LoRA debug.
set -euo pipefail

REPO_ID="${REPO_ID:-i4h/debug_lora_mock}"
MOCK_DATA_DIR="${MOCK_DATA_DIR:-/data/cache/i4h-mock-train/hdf5}"

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

exec python3 debug/prepare_mock_train_data.py \
  --repo_id "${REPO_ID}" \
  --data_dir "${MOCK_DATA_DIR}" \
  "$@"
