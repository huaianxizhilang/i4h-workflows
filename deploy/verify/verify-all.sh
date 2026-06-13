#!/usr/bin/env bash
# Verification suite — run after deploy or on demand.

set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../scripts/common.sh
source "${DEPLOY_ROOT}/scripts/common.sh"
load_config

# shellcheck source=/dev/null
[[ -f "${HOME}/.i4h-deploy.env" ]] && source "${HOME}/.i4h-deploy.env"

PASS=0
FAIL=0
WARN_COUNT=0

check() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "[PASS] ${name}"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] ${name}"
        FAIL=$((FAIL + 1))
    fi
}

warn_check() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "[PASS] ${name}"
        PASS=$((PASS + 1))
    else
        echo "[WARN] ${name}"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi
}

info "=== i4h deploy verification ==="

check "x86_64 architecture" test "$(uname -m)" = "x86_64"
check "nvidia-smi available" command -v nvidia-smi
check "GPU visible" nvidia-smi
check "docker installed" command -v docker
check "docker GPU passthrough" docker_gpu_test
check "git installed" command -v git
check "i4h repo present" test -d "${I4H_INSTALL_DIR}"
check "i4h CLI executable" test -x "${I4H_INSTALL_DIR}/i4h"
check "RTI license file" test -f "${RTI_LICENSE_FILE}"
check "cache dirs" test -d "${I4H_DOCKER_ROOT}/isaac-sim/cache/kit"
check "deploy env file" test -f "${HOME}/.i4h-deploy.env"

if [[ -n "${DISPLAY:-}" ]]; then
    warn_check "X11 DISPLAY (${DISPLAY})" xdpyinfo
else
    echo "[WARN] DISPLAY not set (GUI workflows need ssh -X or VNC)"
    WARN_COUNT=$((WARN_COUNT + 1))
fi

if command -v docker >/dev/null && docker images 2>/dev/null | grep -qE 'i4h|robotic'; then
    echo "[PASS] workflow docker image present"
    PASS=$((PASS + 1))
else
    echo "[WARN] workflow docker image not built yet (run L6)"
    WARN_COUNT=$((WARN_COUNT + 1))
fi

# GPU detail
if nvidia-smi >/dev/null 2>&1; then
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
    GPU_NAME=$(nvidia_smi_query name)
    GPU_CAP=$(nvidia_smi_query compute_cap)
    if echo "${GPU_NAME}" | grep -qiE 'A100|H100'; then
        echo "[FAIL] GPU ${GPU_NAME} — no RT Cores for ultrasound raytracing"
        FAIL=$((FAIL + 1))
    fi
    if ! compute_cap_ge "${GPU_CAP}" "${I4H_MIN_COMPUTE_CAP}"; then
        echo "[FAIL] compute capability ${GPU_CAP} < ${I4H_MIN_COMPUTE_CAP}"
        FAIL=$((FAIL + 1))
    fi
fi

echo "=== Summary: ${PASS} passed, ${FAIL} failed, ${WARN_COUNT} warnings ==="
[[ "${FAIL}" -eq 0 ]]
