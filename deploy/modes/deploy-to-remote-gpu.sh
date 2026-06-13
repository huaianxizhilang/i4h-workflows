#!/usr/bin/env bash
# Push deploy to a remote GPU server from a jump host (Mode B).

set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../scripts/common.sh
source "${DEPLOY_ROOT}/scripts/common.sh"
load_config

REMOTE_HOST="${I4H_REMOTE_HOST:-}"
REMOTE_USER="${I4H_REMOTE_USER:-root}"
REMOTE_PORT="${I4H_REMOTE_PORT:-22}"
REMOTE_PASSWORD="${I4H_REMOTE_PASSWORD:-}"
FROM_LAYER="L0"
TO_LAYER="L6"
USE_REPO_ON_JUMP=0
VERIFY_ONLY=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Mode B — from your jump host, deploy to a new GPU cloud server.

Required (env or deploy/config/local.env):
  I4H_REMOTE_HOST       GPU server IP or hostname
  I4H_REMOTE_USER       SSH user (default: root)
  I4H_REMOTE_PASSWORD   Root password (optional if SSH key works)

Options:
  --host IP             GPU server address
  --user USER           SSH user (default root)
  --password PASS       Root password (prefer SSH key when possible)
  --port PORT           SSH port (default 22)
  --from LAYER          Start layer on remote (default L0)
  --to LAYER            End layer on remote (default L6)
  --use-jump-repo       Rsync this repo to remote instead of git clone on remote
  --verify-only         Run remote verify only
  --help

Examples:
  # Using config file (recommended — do not commit passwords)
  cp deploy/config/local.env.example deploy/config/local.env
  # edit I4H_REMOTE_HOST, I4H_REMOTE_PASSWORD
  bash deploy/modes/deploy-to-remote-gpu.sh

  # One-liner
  I4H_REMOTE_HOST=203.0.113.10 I4H_REMOTE_PASSWORD='secret' \\
    bash deploy/modes/deploy-to-remote-gpu.sh

Security:
  - Prefer SSH key: ssh-copy-id root@GPU_IP
  - If using password, install sshpass: sudo apt install sshpass
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)     REMOTE_HOST="$2"; shift 2 ;;
        --user)     REMOTE_USER="$2"; shift 2 ;;
        --password) REMOTE_PASSWORD="$2"; shift 2 ;;
        --port)     REMOTE_PORT="$2"; shift 2 ;;
        --from)     FROM_LAYER="$2"; shift 2 ;;
        --to)       TO_LAYER="$2"; shift 2 ;;
        --use-jump-repo) USE_REPO_ON_JUMP=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "${REMOTE_HOST}" ]] || die "Set I4H_REMOTE_HOST or --host"

SSH_BASE=(
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=30
    -p "${REMOTE_PORT}"
)

remote_exec() {
    if [[ -n "${REMOTE_PASSWORD}" ]] && command -v sshpass >/dev/null; then
        sshpass -p "${REMOTE_PASSWORD}" ssh "${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "$@"
    else
        ssh "${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "$@"
    fi
}

remote_rsync() {
    local src="$1" dest="$2"
    local rsync_ssh="ssh ${SSH_BASE[*]}"
    if [[ -n "${REMOTE_PASSWORD}" ]] && command -v sshpass >/dev/null; then
        rsync_ssh="sshpass -p ${REMOTE_PASSWORD} ${rsync_ssh}"
    fi
    rsync -az --delete -e "${rsync_ssh}" "${src}" "${REMOTE_USER}@${REMOTE_HOST}:${dest}"
}

info "Mode B: jump host → ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"

# Test SSH
remote_exec "echo OK: connected as \$(whoami)@\$(hostname)"

if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
    remote_exec "bash /tmp/i4h-deploy/verify/verify-all.sh" || \
        die "Remote verify failed. Run full deploy first or re-upload with rsync."
    exit 0
fi

# Upload deploy bundle
REMOTE_DEPLOY="/tmp/i4h-deploy"
info "Uploading deploy scripts to ${REMOTE_DEPLOY}..."

if [[ "${USE_REPO_ON_JUMP}" -eq 1 ]]; then
    REPO_ROOT="$(cd "${DEPLOY_ROOT}/.." && pwd)"
    remote_exec "mkdir -p ${REMOTE_DEPLOY}"
    remote_rsync "${REPO_ROOT}/" "/tmp/i4h-workflows/"
    remote_rsync "${DEPLOY_ROOT}/" "${REMOTE_DEPLOY}/"
    remote_exec "ln -sfn /tmp/i4h-workflows ${REMOTE_DEPLOY}/repo"
else
    remote_exec "mkdir -p ${REMOTE_DEPLOY}"
    remote_rsync "${DEPLOY_ROOT}/" "${REMOTE_DEPLOY}/"
fi

# Push local.env if present (may contain remote credentials — only on wire via SSH)
if [[ -f "${DEPLOY_ROOT}/config/local.env" ]]; then
    remote_rsync "${DEPLOY_ROOT}/config/local.env" "${REMOTE_DEPLOY}/config/local.env"
fi

# Run deploy on remote
REMOTE_CMD="cd ${REMOTE_DEPLOY} && chmod +x layers/*.sh modes/*.sh verify/*.sh run/*.sh scripts/common.sh"

if [[ "${USE_REPO_ON_JUMP}" -eq 1 ]]; then
    REMOTE_CMD+=" && export I4H_INSTALL_DIR=/tmp/i4h-workflows"
fi

REMOTE_CMD+=" && bash modes/deploy-on-gpu-server.sh --from ${FROM_LAYER} --to ${TO_LAYER}"

info "Starting remote deploy (${FROM_LAYER} → ${TO_LAYER})..."
set +e
remote_exec "${REMOTE_CMD}"
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
    warn "Remote deploy exited with code ${rc}"
    if remote_exec "test -f /var/run/reboot-required" 2>/dev/null; then
        cat <<EOF

Remote host may need reboot after driver install.
After reboot, re-run:
  bash deploy/modes/deploy-to-remote-gpu.sh --from L3

EOF
    fi
    exit $rc
fi

cat <<EOF

================================================================================
 Mode B deploy finished on ${REMOTE_HOST}.

 SSH to GPU server:
   ssh ${REMOTE_USER}@${REMOTE_HOST}

 Run workflow on GPU server:
   source ~/.i4h-deploy.env
   cd \$(grep I4H_INSTALL_DIR ~/.i4h-deploy.env | cut -d= -f2 | tr -d '"')
   bash /tmp/i4h-deploy/run/run-robotic-ultrasound.sh

 GUI options on GPU server:
   - ssh -X ${REMOTE_USER}@${REMOTE_HOST}   then run workflow
   - or I4H_GUI_MODE=vnc on server + VNC client to port 5901
================================================================================

EOF
