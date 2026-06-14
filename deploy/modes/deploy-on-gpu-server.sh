#!/usr/bin/env bash
# Run all deploy layers on the current machine (Mode A).

set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../scripts/common.sh
source "${DEPLOY_ROOT}/scripts/common.sh"
load_config

FROM_LAYER="L0"
TO_LAYER="L6"
SKIP_REBOOT_CHECK=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Mode A — deploy on the GPU server you are SSH'd into.

Options:
  --from LAYER   Start at layer (L0|L1|L2|L3|L4|L5|L6), default L0
  --to LAYER     Stop after layer, default L6
  --verify-only  Run verify suite only (no install)
  --help         Show this help

Environment / config:
  deploy/config/defaults.env
  deploy/config/local.env   (optional overrides)

Examples:
  bash deploy/modes/deploy-on-gpu-server.sh
  bash deploy/modes/deploy-on-gpu-server.sh --from L3
  I4H_GUI_MODE=vnc bash deploy/modes/deploy-on-gpu-server.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_LAYER="$2"; shift 2 ;;
        --to)   TO_LAYER="$2"; shift 2 ;;
        --verify-only)
            bash "${DEPLOY_ROOT}/verify/verify-all.sh"
            exit 0
            ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

LAYERS=(
    L0-preflight.sh
    L1-system.sh
    L2-nvidia-driver.sh
    L3-docker.sh
    L4-gui.sh
    L5-i4h-project.sh
    L6-robotic-ultrasound.sh
)

should_run() {
    local id="$1"
    local from_num="${FROM_LAYER#L}"
    local to_num="${TO_LAYER#L}"
    local cur_num="${id#L}"
    (( 10#${cur_num} >= 10#${from_num} && 10#${cur_num} <= 10#${to_num} ))
}

info "Mode A: deploy on local GPU server (${USER}@$(hostname))"
info "Layers ${FROM_LAYER} → ${TO_LAYER}"

for layer in "${LAYERS[@]}"; do
    id="${layer%%-*}"
    if should_run "${id}"; then
        set +e
        bash "${DEPLOY_ROOT}/layers/${layer}"
        rc=$?
        set -e
        if [[ $rc -eq 2 && "${layer}" == "L2-nvidia-driver.sh" ]]; then
            die "Reboot required after L2. Reconnect and run: $0 --from L3"
        fi
        [[ $rc -eq 0 ]] || exit $rc
    fi
done

if [[ "${I4H_RUN_VERIFY_AFTER_DEPLOY}" == "1" && "${TO_LAYER}" == "L6" ]]; then
    bash "${DEPLOY_ROOT}/verify/verify-all.sh"
fi

if [[ "${TO_LAYER}" == "L6" ]]; then
    chmod +x "${DEPLOY_ROOT}/scripts/"*.sh 2>/dev/null || true
    bash "${DEPLOY_ROOT}/scripts/finalize-deploy.sh"
fi

cat <<EOF

================================================================================
 Mode A deploy finished.

 Next steps:
   source ~/.i4h-deploy.env
   bash deploy/run/run-robotic-ultrasound.sh

 Or manually:
   cd ${I4H_INSTALL_DIR}
   xhost +local:docker
   ./i4h run robotic_ultrasound full_pipeline --as-root --no-docker-build
================================================================================

EOF
