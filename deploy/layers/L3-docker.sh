#!/usr/bin/env bash
# L3 — Docker Engine + NVIDIA Container Toolkit (reusable)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"
load_config

info "L3 Docker + NVIDIA Container Toolkit"

if ! nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi failed. Complete L2 and reboot first."
fi

# Docker Engine
if ! command -v docker >/dev/null 2>&1; then
    info "Installing Docker Engine..."
    curl -fsSL https://get.docker.com | run_root sh
    if [[ "${EUID}" -ne 0 ]] && getent group docker >/dev/null; then
        run_root usermod -aG docker "${USER}" || true
        warn "Added ${USER} to docker group — log out/in or run: newgrp docker"
    fi
else
    info "Docker already installed: $(docker --version)"
fi

# NVIDIA Container Toolkit
if ! dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q ^ii; then
    info "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | run_root gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | run_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    apt_install nvidia-container-toolkit
    run_root nvidia-ctk runtime configure --runtime=docker
    run_root systemctl restart docker
else
    info "nvidia-container-toolkit already installed"
fi

info "Testing GPU inside Docker..."
docker_gpu_test
docker run --rm --gpus all "${I4H_CUDA_TEST_IMAGE}" nvidia-smi

info "L3 complete"
