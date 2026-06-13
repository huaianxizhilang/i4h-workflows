#!/usr/bin/env bash
# L1 — OS baseline packages & cache directories (reusable)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"
load_config

info "L1 system baseline"

apt_install \
    git curl wget jq vim htop \
    build-essential ca-certificates gnupg lsb-release \
    xauth x11-xserver-utils \
    software-properties-common \
    openssh-client rsync

ensure_dirs
setup_home_symlinks
print_storage_layout
open_dds_firewall

info "L1 complete"
