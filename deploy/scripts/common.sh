#!/usr/bin/env bash
# Shared helpers for i4h deploy scripts.

set -euo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_config() {
    # shellcheck source=/dev/null
    source "${DEPLOY_ROOT}/config/defaults.env"
    if [[ -f "${DEPLOY_ROOT}/config/local.env" ]]; then
        # shellcheck source=/dev/null
        source "${DEPLOY_ROOT}/config/local.env"
    fi
}

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { log "INFO  $*"; }
warn() { log "WARN  $*" >&2; }
die()  { log "ERROR $*" >&2; exit 1; }

require_root_or_sudo() {
    if [[ "${EUID}" -eq 0 ]]; then
        SUDO=""
        return 0
    fi
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
        return 0
    fi
    die "Need root or sudo to run: $*"
}

run_root() {
    require_root_or_sudo
    if [[ -n "${SUDO}" ]]; then
        ${SUDO} "$@"
    else
        "$@"
    fi
}

apt_install() {
    require_root_or_sudo
    run_root apt-get update -qq
    DEBIAN_FRONTEND=noninteractive run_root apt-get install -y "$@"
}

ensure_dirs() {
    local paths=(
        "${I4H_DOCKER_ROOT}/isaac-sim/cache/kit"
        "${I4H_DOCKER_ROOT}/isaac-sim/cache/ov"
        "${I4H_DOCKER_ROOT}/isaac-sim/cache/pip"
        "${I4H_DOCKER_ROOT}/isaac-sim/cache/glcache"
        "${I4H_DOCKER_ROOT}/isaac-sim/cache/computecache"
        "${I4H_DOCKER_ROOT}/isaac-sim/logs"
        "${I4H_DOCKER_ROOT}/isaac-sim/data"
        "${I4H_DOCKER_ROOT}/isaac-sim/documents"
        "${I4H_DOCKER_ROOT}/rti"
        "${I4H_CACHE_ROOT}/i4h-assets"
        "${I4H_CACHE_ROOT}/huggingface"
    )
    for p in "${paths[@]}"; do
        mkdir -p "$p"
    done
    info "Cache directories ready under ${I4H_DOCKER_ROOT} and ${I4H_CACHE_ROOT}"
}

version_ge() {
    # Compare dotted versions: version_ge 555.58 535
    local IFS=.
    local i ver1=($1) ver2=($2)
    for ((i = 0; i < ${#ver1[@]} || i < ${#ver2[@]}; i++)); do
        local a="${ver1[i]:-0}" b="${ver2[i]:-0}"
        if ((10#$a > 10#$b)); then return 0; fi
        if ((10#$a < 10#$b)); then return 1; fi
    done
    return 0
}

compute_cap_ge() {
    # compute_cap_ge 8.6 8.6
    awk -v a="$1" -v b="$2" 'BEGIN { exit (a+0 >= b+0) ? 0 : 1 }'
}

nvidia_smi_query() {
    nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1
}

docker_gpu_test() {
    docker run --rm --gpus all "${I4H_CUDA_TEST_IMAGE}" nvidia-smi >/dev/null
}

setup_display_for_docker() {
    case "${I4H_GUI_MODE}" in
        vnc)
            export DISPLAY="${I4H_VNC_DISPLAY}"
            ;;
        x11-ssh|auto)
            if [[ -z "${DISPLAY:-}" ]]; then
                if [[ "${I4H_GUI_MODE}" == "x11-ssh" ]]; then
                    die "DISPLAY is empty. SSH with: ssh -X user@host"
                fi
                warn "DISPLAY not set; GUI workflows may fail. Use ssh -X or I4H_GUI_MODE=vnc."
                return 0
            fi
            ;;
        headless)
            export DISPLAY="${DISPLAY:-:99}"
            return 0
            ;;
        *)
            die "Unknown I4H_GUI_MODE=${I4H_GUI_MODE}"
            ;;
    esac

    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null 2>&1 || warn "xhost +local:docker failed (GUI may not work)"
    fi
}

download_rti_license() {
    if [[ -f "${RTI_LICENSE_FILE}" ]]; then
        info "RTI license already present: ${RTI_LICENSE_FILE}"
        return 0
    fi
    if [[ -z "${RTI_LICENSE_URL:-}" ]]; then
        warn "RTI_LICENSE_URL not set; skip auto-download"
        return 1
    fi
    info "Downloading RTI evaluation license..."
    mkdir -p "$(dirname "${RTI_LICENSE_FILE}")"
    if command -v curl >/dev/null; then
        curl -fsSL "${RTI_LICENSE_URL}" -o "${RTI_LICENSE_FILE}"
    elif command -v wget >/dev/null; then
        wget -q "${RTI_LICENSE_URL}" -O "${RTI_LICENSE_FILE}"
    else
        die "curl or wget required to download RTI license"
    fi
    chmod 644 "${RTI_LICENSE_FILE}"
    info "RTI license saved to ${RTI_LICENSE_FILE}"
}

clone_or_update_repo() {
    if [[ -d "${I4H_INSTALL_DIR}/.git" ]]; then
        info "Updating existing repo at ${I4H_INSTALL_DIR}"
        git -C "${I4H_INSTALL_DIR}" fetch --depth 1 origin "${I4H_REPO_BRANCH}"
        git -C "${I4H_INSTALL_DIR}" checkout "${I4H_REPO_BRANCH}"
        git -C "${I4H_INSTALL_DIR}" pull --ff-only origin "${I4H_REPO_BRANCH}" || true
    else
        info "Cloning ${I4H_REPO_URL} → ${I4H_INSTALL_DIR}"
        git clone --depth 1 --branch "${I4H_REPO_BRANCH}" "${I4H_REPO_URL}" "${I4H_INSTALL_DIR}"
    fi
}

open_dds_firewall() {
    if command -v ufw >/dev/null 2>&1; then
        run_root ufw allow in proto udp to 239.255.0.1 port 7400:7401 2>/dev/null || true
        run_root ufw allow out proto udp to 239.255.0.1 port 7400:7401 2>/dev/null || true
        info "UFW rules for RTI DDS multicast applied (if ufw active)"
    fi
}

run_layer() {
    local script="$1"
    local name
    name="$(basename "${script}")"
    info "======== ${name} ========"
    bash "${script}"
}
