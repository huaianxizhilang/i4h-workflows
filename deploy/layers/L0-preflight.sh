#!/usr/bin/env bash
# L0 — Hardware & OS preflight (reusable across i4h GPU workflows)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"
load_config

info "L0 preflight checks"

# OS
if [[ "$(uname -m)" != "x86_64" ]]; then
    die "Requires x86_64; got $(uname -m)"
fi

OS_ID=""
OS_VER=""
if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    OS_ID="${ID:-unknown}"
    OS_VER="${VERSION_ID:-unknown}"
fi
info "OS: ${OS_ID} ${OS_VER}"
if [[ "${OS_ID}" != "ubuntu" ]]; then
    warn "Official support is Ubuntu 22.04/24.04; continuing on ${OS_ID}"
fi

# RAM
MEM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
info "System RAM: ~${MEM_GB} GB"
if (( MEM_GB < 32 )); then
    warn "Recommended ≥64 GB RAM for Isaac Sim; you have ~${MEM_GB} GB"
fi

# Disk
DISK_AVAIL_GB=$(df -BG "${HOME}" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
info "Free disk under ${HOME}: ~${DISK_AVAIL_GB} GB"
if (( DISK_AVAIL_GB < 80 )); then
    warn "Recommended ≥100 GB free; you have ~${DISK_AVAIL_GB} GB"
fi

# GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found — L2 will attempt driver install"
    exit 0
fi

GPU_NAME=$(nvidia_smi_query name)
GPU_CAP=$(nvidia_smi_query compute_cap)
GPU_VRAM=$(nvidia_smi_query memory.total)
DRIVER_VER=$(nvidia_smi_query driver_version)

info "GPU: ${GPU_NAME}"
info "Compute capability: ${GPU_CAP}"
info "VRAM: ${GPU_VRAM} MB"
info "Driver: ${DRIVER_VER}"

if ! compute_cap_ge "${GPU_CAP}" "${I4H_MIN_COMPUTE_CAP}"; then
    die "GPU compute capability ${GPU_CAP} < required ${I4H_MIN_COMPUTE_CAP}"
fi

if (( GPU_VRAM < I4H_MIN_VRAM_MB )); then
    warn "VRAM ${GPU_VRAM} MB < recommended ${I4H_MIN_VRAM_MB} MB"
fi

if ! version_ge "${DRIVER_VER}" "${I4H_MIN_DRIVER_VERSION}"; then
    warn "Driver ${DRIVER_VER} < recommended ${I4H_MIN_DRIVER_VERSION}"
fi

# RT Core warning for A100/H100
if echo "${GPU_NAME}" | grep -qiE 'A100|H100'; then
    die "GPU ${GPU_NAME} lacks RT Cores — robotic_ultrasound raytracing will NOT work. Use RTX 3090/4090/A6000 etc."
fi

info "L0 preflight passed"
