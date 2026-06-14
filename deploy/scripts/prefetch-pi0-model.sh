#!/usr/bin/env bash
# 预拉 PI0 模型到 ${I4H_CACHE_ROOT}/huggingface（L6 之后、首次跑 full_pipeline 之前）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

# shellcheck source=/dev/null
[[ -f "${HOME}/.i4h-deploy.env" ]] && source "${HOME}/.i4h-deploy.env"

if [[ "${I4H_PREFETCH_PI0:-1}" != "1" ]]; then
    info "I4H_PREFETCH_PI0=0 — skip PI0 prefetch"
    exit 0
fi

if [[ "${I4H_WORKFLOW}" != "robotic_ultrasound" ]]; then
    info "PI0 prefetch applies to robotic_ultrasound only (I4H_WORKFLOW=${I4H_WORKFLOW})"
    exit 0
fi

ensure_dirs

# 预拉 PI0 模型到 /data/cache/huggingface
pi0_hf_cache="${I4H_CACHE_ROOT}/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel"
if [[ -d "${pi0_hf_cache}/snapshots" ]] && [[ -n "$(ls -A "${pi0_hf_cache}/snapshots" 2>/dev/null)" ]]; then
    info "PI0 model already cached under ${I4H_CACHE_ROOT}/huggingface — skip prefetch"
    exit 0
fi

image="${I4H_DOCKER_IMAGE:-}"
if [[ -z "${image}" ]]; then
    image="$(docker_cli images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep -E '^i4h_build:robotic_ultrasound$' | head -1 || true)"
fi
if [[ -z "${image}" ]]; then
    image="$(docker_cli images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep 'robotic_ultrasound' | head -1 || true)"
fi
[[ -n "${image}" ]] || die "No i4h docker image found — complete L6 build first"

info "Prefetch PI0 weights → ${I4H_CACHE_ROOT}/huggingface (via i4h-asset-retrieve)"
info "Docker image: ${image}"

docker_env=()
[[ -n "${I4H_HF_ENDPOINT:-}" ]] && docker_env+=(-e "HF_ENDPOINT=${I4H_HF_ENDPOINT}")
docker_env+=(-e "HF_HUB_ENABLE_HF_TRANSFER=0")

with_docker_access docker run --rm --gpus all \
    "${docker_env[@]}" \
    -v "${I4H_CACHE_ROOT}/huggingface:/root/.cache/huggingface" \
    -v "${I4H_CACHE_ROOT}/i4h-assets:/root/.cache/i4h-assets" \
    "${image}" \
    bash -lc 'source /opt/miniconda3/bin/activate && conda activate robotic_ultrasound && yes Yes | i4h-asset-retrieve --sub-path Policies/LiverScan/Pi0'

info "PI0 prefetch complete"
du -sh "${I4H_CACHE_ROOT}/huggingface" 2>/dev/null || true
