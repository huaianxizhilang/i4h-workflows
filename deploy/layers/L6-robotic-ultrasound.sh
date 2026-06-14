#!/usr/bin/env bash
# L6 — robotic_ultrasound build & optional smoke run

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"
load_config

# shellcheck source=/dev/null
[[ -f "${HOME}/.i4h-deploy.env" ]] && source "${HOME}/.i4h-deploy.env"

info "L6 workflow: ${I4H_WORKFLOW}"

if [[ ! -d "${I4H_INSTALL_DIR}" ]]; then
    die "I4H_INSTALL_DIR missing. Run L5 first."
fi

cd "${I4H_INSTALL_DIR}"
export RTI_LICENSE_FILE

setup_display_for_docker

# 自动选择当前最快的 apt 镜像传入 Docker build
if [[ "${I4H_CONFIGURE_HOST_APT_MIRROR:-1}" == "1" && -z "${I4H_APT_MIRROR:-}" ]]; then
    I4H_APT_MIRROR="$(pick_apt_mirror)"
    export I4H_APT_MIRROR
fi
[[ -n "${I4H_APT_MIRROR:-}" ]] && info "Docker build apt mirror: ${I4H_APT_MIRROR}"

if [[ "${I4H_SKIP_BUILD}" != "1" ]]; then
    info "Building container (first run may take 30-60+ minutes)..."
    chmod +x "${SCRIPT_DIR}/../scripts/docker-build-with-progress.sh"
    bash "${SCRIPT_DIR}/../scripts/docker-build-with-progress.sh" "${I4H_WORKFLOW}"
else
    info "Skipping build (I4H_SKIP_BUILD=1)"
fi

if [[ "${I4H_PREFETCH_PI0:-1}" == "1" ]]; then
    chmod +x "${SCRIPT_DIR}/../scripts/prefetch-pi0-model.sh"
    bash "${SCRIPT_DIR}/../scripts/prefetch-pi0-model.sh"
fi

if [[ "${I4H_RUN_SMOKE_TEST}" == "1" ]]; then
    info "Smoke test: dry-run full_pipeline"
    ./i4h run "${I4H_WORKFLOW}" full_pipeline --as-root --dryrun
fi

info "L6 complete"
info "Run workflow: bash ${SCRIPT_DIR}/../run/run-robotic-ultrasound.sh"
