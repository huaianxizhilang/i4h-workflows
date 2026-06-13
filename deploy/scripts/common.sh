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
    resolve_storage_paths
}

resolve_storage_paths() {
    # Master switch: put heavy data on a mounted data disk (e.g. /data).
    if [[ -n "${I4H_DATA_ROOT:-}" ]]; then
        if [[ ! -d "${I4H_DATA_ROOT}" ]]; then
            die "I4H_DATA_ROOT=${I4H_DATA_ROOT} does not exist. Mount the data disk first."
        fi
        I4H_INSTALL_DIR="${I4H_INSTALL_DIR:-${I4H_DATA_ROOT}/i4h-workflows}"
        I4H_DOCKER_ROOT="${I4H_DOCKER_ROOT:-${I4H_DATA_ROOT}/docker}"
        I4H_CACHE_ROOT="${I4H_CACHE_ROOT:-${I4H_DATA_ROOT}/cache}"
        I4H_DOCKER_DATA_ROOT="${I4H_DOCKER_DATA_ROOT:-${I4H_DATA_ROOT}/docker-engine}"
        I4H_CONTAINERD_ROOT="${I4H_CONTAINERD_ROOT:-${I4H_DATA_ROOT}/containerd}"
    else
        I4H_INSTALL_DIR="${I4H_INSTALL_DIR:-$HOME/i4h-workflows}"
        I4H_DOCKER_ROOT="${I4H_DOCKER_ROOT:-$HOME/docker}"
        I4H_CACHE_ROOT="${I4H_CACHE_ROOT:-$HOME/.cache}"
        I4H_DOCKER_DATA_ROOT="${I4H_DOCKER_DATA_ROOT:-}"
    fi
    RTI_LICENSE_FILE="${RTI_LICENSE_FILE:-${I4H_DOCKER_ROOT}/rti/rti_license.dat}"
    export I4H_INSTALL_DIR I4H_DOCKER_ROOT I4H_CACHE_ROOT I4H_DOCKER_DATA_ROOT I4H_CONTAINERD_ROOT RTI_LICENSE_FILE
}

link_into_home() {
    local target="$1" link="$2"
    mkdir -p "$(dirname "${target}")" "$(dirname "${link}")"
    if [[ -L "${link}" ]]; then
        local current
        current="$(readlink -f "${link}")"
        if [[ "${current}" == "$(readlink -f "${target}")" ]]; then
            return 0
        fi
        rm -f "${link}"
    elif [[ -e "${link}" ]]; then
        warn "${link} exists and is not a symlink — leaving as-is (may use system disk)"
        return 0
    fi
    ln -sfn "${target}" "${link}"
    info "Symlink ${link} → ${target}"
}

setup_home_symlinks() {
    # ./i4h and HoloHub expect ~/docker and ~/.cache/* — link to data disk paths.
    [[ -n "${I4H_DATA_ROOT:-}" ]] || return 0
    link_into_home "${I4H_DOCKER_ROOT}" "${HOME}/docker"
    link_into_home "${I4H_CACHE_ROOT}/i4h-assets" "${HOME}/.cache/i4h-assets"
    link_into_home "${I4H_CACHE_ROOT}/huggingface" "${HOME}/.cache/huggingface"
    if [[ "$(readlink -f "${I4H_INSTALL_DIR}" 2>/dev/null || echo "${I4H_INSTALL_DIR}")" != \
          "$(readlink -f "${HOME}/i4h-workflows" 2>/dev/null || echo "__missing__")" ]]; then
        link_into_home "${I4H_INSTALL_DIR}" "${HOME}/i4h-workflows"
    fi
}

configure_docker_data_root() {
    [[ -n "${I4H_DOCKER_DATA_ROOT:-}" ]] || return 0
    require_root_or_sudo
    mkdir -p "${I4H_DOCKER_DATA_ROOT}"

    local daemon_json="/etc/docker/daemon.json"
    local current_root=""
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        current_root="$(docker info 2>/dev/null | awk -F': ' '/Docker Root Dir/ {print $2}')"
    fi

    if [[ -n "${current_root}" && "${current_root}" == "${I4H_DOCKER_DATA_ROOT}" ]]; then
        info "Docker data-root already ${I4H_DOCKER_DATA_ROOT}"
        return 0
    fi

    if [[ -n "${current_root}" && -d "${current_root}" ]] && \
        [[ "$(find "${current_root}" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)" -gt 0 ]]; then
        warn "Docker already stores data in ${current_root}; not moving automatically."
        warn "Fresh machine: set I4H_DATA_ROOT before L3. To migrate manually:"
        warn "  systemctl stop docker && rsync -a ${current_root}/ ${I4H_DOCKER_DATA_ROOT}/ && configure data-root"
        return 0
    fi

    info "Configuring Docker data-root → ${I4H_DOCKER_DATA_ROOT}"
    run_root mkdir -p /etc/docker
    if [[ -f "${daemon_json}" ]]; then
        python3 - "${daemon_json}" "${I4H_DOCKER_DATA_ROOT}" <<'PY'
import json, sys
path, data_root = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
cfg["data-root"] = data_root
with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
    else
        printf '{\n  "data-root": "%s"\n}\n' "${I4H_DOCKER_DATA_ROOT}" | run_root tee "${daemon_json}" >/dev/null
    fi

    if command -v docker >/dev/null 2>&1; then
        run_root systemctl restart docker
        sleep 3
        local tries=0
        while [[ $tries -lt 5 ]]; do
            current_root="$(docker_cli info 2>/dev/null | awk -F': ' '/Docker Root Dir/ {print $2}')"
            [[ -n "${current_root}" ]] && break
            sleep 2
            tries=$((tries + 1))
        done
        [[ "${current_root}" == "${I4H_DOCKER_DATA_ROOT}" ]] || die "Failed to set Docker data-root (got: ${current_root:-<empty>})"
        info "Docker data-root active: ${current_root}"
    fi
}

# buildkit stores overlay layers under containerd (default /var/lib/containerd on system disk).
configure_containerd_data_root() {
    [[ -n "${I4H_CONTAINERD_ROOT:-}" ]] || return 0
    require_root_or_sudo
    mkdir -p "${I4H_CONTAINERD_ROOT}"

    local link="/var/lib/containerd"
    if [[ -L "${link}" ]]; then
        local target
        target="$(readlink -f "${link}")"
        if [[ "${target}" == "$(readlink -f "${I4H_CONTAINERD_ROOT}")" ]]; then
            info "containerd already on data disk: ${I4H_CONTAINERD_ROOT}"
            return 0
        fi
    fi

    if [[ -d "${link}" && ! -L "${link}" ]] && \
        [[ "$(find "${link}" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)" -gt 0 ]]; then
        info "Migrating containerd → ${I4H_CONTAINERD_ROOT} (frees system disk for docker builds)"
        run_root systemctl stop docker 2>/dev/null || true
        run_root systemctl stop containerd 2>/dev/null || true
        if [[ ! -d "${I4H_CONTAINERD_ROOT}" ]] || \
            [[ "$(find "${I4H_CONTAINERD_ROOT}" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)" -eq 0 ]]; then
            run_root rsync -a "${link}/" "${I4H_CONTAINERD_ROOT}/"
        fi
        run_root mv "${link}" "${link}.bak.$(date +%s)" 2>/dev/null || run_root rm -rf "${link}"
        run_root ln -s "${I4H_CONTAINERD_ROOT}" "${link}"
        run_root systemctl start containerd 2>/dev/null || true
        run_root systemctl start docker
        info "containerd migrated; symlink ${link} → ${I4H_CONTAINERD_ROOT}"
    elif [[ ! -e "${link}" ]]; then
        run_root ln -s "${I4H_CONTAINERD_ROOT}" "${link}"
        info "containerd symlink ${link} → ${I4H_CONTAINERD_ROOT}"
    fi
}

disk_avail_gb() {
    local path="$1"
    df -BG "${path}" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}'
}

print_storage_layout() {
    info "Storage layout:"
    info "  I4H_DATA_ROOT=${I4H_DATA_ROOT:-<system disk>}"
    info "  I4H_INSTALL_DIR=${I4H_INSTALL_DIR}"
    info "  I4H_DOCKER_ROOT=${I4H_DOCKER_ROOT}"
    info "  I4H_CACHE_ROOT=${I4H_CACHE_ROOT}"
    info "  I4H_DOCKER_DATA_ROOT=${I4H_DOCKER_DATA_ROOT:-/var/lib/docker (default)}"
    info "  I4H_CONTAINERD_ROOT=${I4H_CONTAINERD_ROOT:-/var/lib/containerd (default)}"
    if [[ -n "${I4H_DATA_ROOT:-}" ]]; then
        info "  data disk free: ~$(disk_avail_gb "${I4H_DATA_ROOT}") GB under ${I4H_DATA_ROOT}"
    fi
    info "  system disk free: ~$(disk_avail_gb /) GB under /"
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

ensure_vnc_password() {
    mkdir -p "${HOME}/.vnc"
    if [[ -f "${HOME}/.vnc/passwd" ]]; then
        return 0
    fi
    if [[ -n "${I4H_VNC_PASSWORD:-}" ]]; then
        info "Creating VNC password from I4H_VNC_PASSWORD (non-interactive)"
        install -m 700 -d "${HOME}/.vnc"
        printf '%s\n' "${I4H_VNC_PASSWORD}" | vncpasswd -f > "${HOME}/.vnc/passwd"
        chmod 600 "${HOME}/.vnc/passwd"
        return 0
    fi
    if [[ -t 0 ]]; then
        warn "Set VNC password interactively (or set I4H_VNC_PASSWORD for one-click deploy):"
        vncpasswd || die "vncpasswd required for VNC mode"
        return 0
    fi
    die "No ~/.vnc/passwd and I4H_VNC_PASSWORD unset. Set I4H_VNC_PASSWORD in deploy/config/local.env for headless one-click deploy."
}

ensure_vnc_xstartup() {
    install -m 700 -d "${HOME}/.vnc"
    if [[ ! -f "${HOME}/.vnc/xstartup" ]]; then
        cat > "${HOME}/.vnc/xstartup" <<'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
        chmod +x "${HOME}/.vnc/xstartup"
        info "Created ~/.vnc/xstartup for XFCE"
    fi
}

ensure_vnc_config() {
    install -m 700 -d "${HOME}/.vnc"
    local allow_remote="${I4H_VNC_ALLOW_REMOTE:-1}"
    if [[ ! -f "${HOME}/.vnc/config" ]]; then
        if [[ "${allow_remote}" == "1" ]]; then
            cat > "${HOME}/.vnc/config" <<'EOF'
localhost=no
alwaysshared
EOF
            info "Created ~/.vnc/config (remote VNC allowed)"
        else
            touch "${HOME}/.vnc/config"
        fi
    fi
}

vnc_display_number() {
    echo "${I4H_VNC_DISPLAY#:}"
}

vnc_is_running() {
    vncserver -list 2>/dev/null | grep -q "$(vnc_display_number)"
}

start_vnc_server() {
    ensure_vnc_password
    ensure_vnc_xstartup
    ensure_vnc_config
    export DISPLAY="${I4H_VNC_DISPLAY}"

    if vnc_is_running; then
        info "VNC already running on ${DISPLAY}"
        return 0
    fi

    local vnc_args=(
        "${I4H_VNC_DISPLAY}"
        -geometry "${I4H_VNC_GEOMETRY}"
        -depth 24
    )
    if [[ "${I4H_VNC_ALLOW_REMOTE:-1}" == "1" ]]; then
        vnc_args+=(-localhost no)
    fi

    info "Starting TigerVNC + XFCE on ${DISPLAY}..."
    vncserver "${vnc_args[@]}"
    info "VNC listening on port $(( 5900 + $(vnc_display_number) )) (DISPLAY=${DISPLAY})"
}

open_vnc_firewall() {
    local port=$(( 5900 + $(vnc_display_number) ))
    if command -v ufw >/dev/null 2>&1; then
        run_root ufw allow "${port}/tcp" 2>/dev/null || true
        info "UFW allow ${port}/tcp (if ufw active)"
    fi
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
