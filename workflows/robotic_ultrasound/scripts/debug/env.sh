#!/usr/bin/env bash
# Source Isaac Sim / Isaac Lab / robotic_ultrasound conda env inside the debug container.
# Usage: source workflows/robotic_ultrasound/scripts/debug/env.sh
set -euo pipefail

export I4H_ROOT="${I4H_ROOT:-/workspace/i4h-workflows}"
export SCRIPTS_DIR="${I4H_ROOT}/workflows/robotic_ultrasound/scripts"

# Conda env (same as Dockerfile)
if [[ -f /opt/miniconda3/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /opt/miniconda3/bin/activate
  conda activate robotic_ultrasound
fi

export PYTHONPATH="${SCRIPTS_DIR}:${I4H_ROOT}/install/lib/clarius_solum:${I4H_ROOT}/install/lib/clarius_cast:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${I4H_ROOT}/install/lib/clarius_solum:${I4H_ROOT}/install/lib/clarius_cast"

# Isaac Sim / Omniverse (required by AppLauncher)
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

# Sanity check — fail fast with a clear message
python3 -c "import isaaclab; print('isaaclab:', isaaclab.__file__)" 2>/dev/null || {
  echo "ERROR: isaaclab not found. Host mount hid image layers."
  echo "Run on GPU host: bash deploy/debug/pi0-debug.sh sync-deps"
  exit 1
}
