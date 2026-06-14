#!/usr/bin/env bash
# Launch Isaac Sim + DDS only (terminal 1 for split-process debugging).
set -euo pipefail

cd /workspace/i4h-workflows/workflows/robotic_ultrasound/scripts
# shellcheck disable=SC1091
source "${PWD}/debug/env.sh"

exec python3 -m simulation.environments.sim_with_dds --enable_cameras "$@"
