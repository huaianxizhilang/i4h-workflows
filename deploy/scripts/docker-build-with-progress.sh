#!/usr/bin/env bash
# L6 Docker 构建包装：plain 进度 + 磁盘用量轮询（弥补无 buildctl 时的可见性）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

WORKFLOW="${1:?workflow}"
shift
EXTRA_ARGS=("$@")

load_config
[[ -f "${HOME}/.i4h-deploy.env" ]] && source "${HOME}/.i4h-deploy.env"

cd "${I4H_INSTALL_DIR}"

BUILD_ARGS=()
[[ "${I4H_BUILD_NO_CACHE}" == "1" ]] && BUILD_ARGS+=(--no-cache)

# 收集 build-args
BUILD_ARG_STR=""
if [[ -n "${I4H_APT_MIRROR:-}" ]]; then
    BUILD_ARG_STR+="--build-arg APT_MIRROR=${I4H_APT_MIRROR} "
fi
if [[ -n "${I4H_PIP_INDEX_URL:-}" ]]; then
    BUILD_ARG_STR+="--build-arg PIP_INDEX_URL=${I4H_PIP_INDEX_URL} "
fi
if [[ -n "${I4H_PIP_TRUSTED_HOST:-}" ]]; then
    BUILD_ARG_STR+="--build-arg PIP_TRUSTED_HOST=${I4H_PIP_TRUSTED_HOST} "
fi
if [[ -n "${I4H_CONDA_MIRROR:-}" ]]; then
    BUILD_ARG_STR+="--build-arg CONDA_MIRROR=${I4H_CONDA_MIRROR} "
fi

watch_build_disk_usage() {
    local pid="$1" interval="${I4H_BUILD_PROGRESS_INTERVAL:-60}"
    while kill -0 "${pid}" 2>/dev/null; do
        sleep "${interval}"
        info "--- Build progress snapshot (every ${interval}s) ---"
        if command -v buildctl >/dev/null 2>&1; then
            BUILDKIT_HOST="${BUILDKIT_HOST:-unix:///run/buildkit/buildkit.sock}" \
                buildctl du 2>/dev/null | head -20 || true
        else
            info "buildctl not installed — showing disk usage instead (apt: docker-buildx-plugin / buildkit)"
        fi
        [[ -n "${I4H_CONTAINERD_ROOT:-}" && -d "${I4H_CONTAINERD_ROOT}" ]] && \
            info "  containerd: $(du -sh "${I4H_CONTAINERD_ROOT}" 2>/dev/null | cut -f1) ${I4H_CONTAINERD_ROOT}"
        [[ -n "${I4H_DOCKER_DATA_ROOT:-}" && -d "${I4H_DOCKER_DATA_ROOT}" ]] && \
            info "  docker-engine: $(du -sh "${I4H_DOCKER_DATA_ROOT}" 2>/dev/null | cut -f1) ${I4H_DOCKER_DATA_ROOT}"
        df -h / "${I4H_DATA_ROOT:-/data}" 2>/dev/null | tail -n +2 || true
        docker_cli system df 2>/dev/null || true
    done
}

info "Starting docker build with progress monitoring..."
info "Build-args: ${BUILD_ARG_STR:-<default>}"

run_build() {
    if [[ -n "${BUILD_ARG_STR}" ]]; then
        # shellcheck disable=SC2086
        with_docker_access ./i4h build-container "${WORKFLOW}" "${BUILD_ARGS[@]}" --build-args "${BUILD_ARG_STR}" "${EXTRA_ARGS[@]}"
    else
        with_docker_access ./i4h build-container "${WORKFLOW}" "${BUILD_ARGS[@]}" "${EXTRA_ARGS[@]}"
    fi
}

if [[ "${I4H_BUILD_SHOW_PROGRESS:-1}" == "1" ]]; then
    (
        run_build
    ) &
    build_pid=$!
    watch_build_disk_usage "${build_pid}" &
    watcher_pid=$!
    wait "${build_pid}"
    rc=$?
    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true
    exit "${rc}"
else
    run_build
fi
