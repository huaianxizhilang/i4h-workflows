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

BUILD_ARGS=()
if [[ "${I4H_BUILD_NO_CACHE}" == "1" ]]; then
    BUILD_ARGS+=(--no-cache)
fi

if [[ "${I4H_SKIP_BUILD}" != "1" ]]; then
    info "Building container (first run may take 30-60+ minutes)..."
    ./i4h build-container "${I4H_WORKFLOW}" "${BUILD_ARGS[@]}"
else
    info "Skipping build (I4H_SKIP_BUILD=1)"
fi

if [[ "${I4H_RUN_SMOKE_TEST}" == "1" ]]; then
    info "Smoke test: dry-run full_pipeline"
    ./i4h run "${I4H_WORKFLOW}" full_pipeline --as-root --dryrun
fi

info "L6 complete"
info "Run workflow: bash ${DEPLOY_ROOT}/run/run-robotic-ultrasound.sh"
