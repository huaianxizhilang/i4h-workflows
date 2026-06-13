#!/usr/bin/env bash
# Run robotic_ultrasound full_pipeline after deploy.

set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../scripts/common.sh
source "${DEPLOY_ROOT}/scripts/common.sh"
load_config

# shellcheck source=/dev/null
[[ -f "${HOME}/.i4h-deploy.env" ]] && source "${HOME}/.i4h-deploy.env"

MODE="${1:-full_pipeline}"
EXTRA_ARGS=("${@:2}")

cd "${I4H_INSTALL_DIR}"
export RTI_LICENSE_FILE

setup_display_for_docker

export HOLOHUB_CMD_NAME="./i4h"

info "Running: ./i4h run ${I4H_WORKFLOW} ${MODE} --as-root --no-docker-build"
exec ./i4h run "${I4H_WORKFLOW}" "${MODE}" --as-root --no-docker-build "${EXTRA_ARGS[@]}"
