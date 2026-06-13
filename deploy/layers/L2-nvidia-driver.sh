#!/usr/bin/env bash
# L2 — NVIDIA driver (reusable; skips if nvidia-smi already works)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"
load_config

info "L2 NVIDIA driver"

if nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER=$(nvidia_smi_query driver_version)
    info "NVIDIA driver already installed: ${DRIVER_VER}"
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
    exit 0
fi

warn "nvidia-smi unavailable — installing recommended driver via ubuntu-drivers"
require_root_or_sudo
apt_install ubuntu-drivers-common
run_root ubuntu-drivers autoinstall

cat <<'EOF'

L2 installed drivers. A REBOOT is required before continuing.

After reboot, re-run deploy from L3 or:
  bash deploy/modes/deploy-on-gpu-server.sh --from L3

EOF
exit 2
