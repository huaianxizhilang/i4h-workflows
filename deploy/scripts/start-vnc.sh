#!/usr/bin/env bash
# Start TigerVNC + XFCE (idempotent). Used by L4 and systemd i4h-vnc.service.

set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${DEPLOY_ROOT}/scripts/common.sh"
load_config

# shellcheck source=/dev/null
[[ -f "${HOME}/.i4h-deploy.env" ]] && source "${HOME}/.i4h-deploy.env"

export I4H_GUI_MODE=vnc
start_vnc_server
