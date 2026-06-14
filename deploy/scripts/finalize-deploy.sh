#!/usr/bin/env bash
# 部署收尾：修改登录密码等（L6 之后、verify 之前/之后调用）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

finalize_deploy() {
    if [[ -n "${I4H_SET_LOGIN_PASSWORD:-}" ]]; then
        set_login_password "${I4H_SET_LOGIN_PASSWORD}"
    fi
    print_data_disk_usage_summary
}

set_login_password() {
    local new_pass="$1" user="${I4H_LOGIN_USER:-ubuntu}"
    require_root_or_sudo
    id "${user}" >/dev/null 2>&1 || die "User ${user} not found for password change"
    info "Setting login password for ${user} (SSH/VNC unlock — not committing password to disk)"
    echo "${user}:${new_pass}" | run_root chpasswd
    info "Login password updated for ${user}"
}

print_data_disk_usage_summary() {
    [[ -n "${I4H_DATA_ROOT:-}" && -d "${I4H_DATA_ROOT}" ]] || return 0
    info "======== /data 目录占用概览 ========"
    du -sh "${I4H_DATA_ROOT}"/* 2>/dev/null | sort -hr | head -15 || true
    info "  containerd → ${I4H_CONTAINERD_ROOT:-/var/lib/containerd}"
    info "  docker-engine → ${I4H_DOCKER_DATA_ROOT:-/var/lib/docker}"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" 2>/dev/null | head -10 || true
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    load_config
    finalize_deploy
fi
