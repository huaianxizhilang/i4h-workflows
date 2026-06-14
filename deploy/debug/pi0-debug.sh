#!/usr/bin/env bash
# PI0 debug helper for remote GPU host (106.75.237.179).
# Usage on the GPU server:
#   bash deploy/debug/pi0-debug.sh sync-deps       # copy openpi+lerobot from image (once)
#   bash deploy/debug/pi0-debug.sh sync-hf-cache  # copy HF weights from image (once, ~9GB)
#   bash deploy/debug/pi0-debug.sh verify          # smoke-test inference + LoRA config
#   bash deploy/debug/pi0-debug.sh shell           # interactive debug container
#   bash deploy/debug/pi0-debug.sh infer           # start inference debugpy (port 5678)
#   bash deploy/debug/pi0-debug.sh train           # start LoRA train debugpy (port 5679)
#   bash deploy/debug/pi0-debug.sh sim             # sim_env only (split debug terminal 1)
#   bash deploy/debug/pi0-debug.sh policy          # pi0_policy DDS only (split debug terminal 2)
#
# From your laptop (after SSH -L 5678:localhost:5678 ubuntu@106.75.237.179):
#   Cursor Remote-SSH → open /data/i4h-workflows → F5 → "PI0: Attach Inference"

set -euo pipefail

ROOT="${I4H_INSTALL_DIR:-/data/i4h-workflows}"
CACHE_ROOT="${I4H_CACHE_ROOT:-/data/cache}"
DOCKER_ROOT="${I4H_DOCKER_ROOT:-/data/docker}"
IMAGE="${I4H_DEBUG_IMAGE:-i4h_build-robotic_ultrasound:dev-xzl}"
CONTAINER="${I4H_DEBUG_CONTAINER:-i4h-pi0-debug}"
DEBUG_DIR="${ROOT}/workflows/robotic_ultrasound/scripts/debug"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Match full_pipeline: use VNC display so Isaac Sim Vulkan/GPU works in container
[[ -f "${HOME}/.i4h-deploy.env" ]] && source "${HOME}/.i4h-deploy.env"
if [[ -z "${DISPLAY:-}" ]]; then
  if [[ -n "${I4H_VNC_DISPLAY:-}" ]]; then
    export DISPLAY="${I4H_VNC_DISPLAY}"
  elif [[ -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=:1
  fi
fi
if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

docker_cli() {
  if groups | grep -q docker 2>/dev/null; then
    docker "$@"
  else
    sudo docker "$@"
  fi
}

ensure_deps_on_host() {
  local missing=0
  for req in \
    "${ROOT}/third_party/openpi/src/openpi/models/pi0.py" \
    "${ROOT}/third_party/IsaacLab/source/isaaclab/isaaclab/__init__.py" \
    "${ROOT}/third_party/lerobot" \
    "${ROOT}/install/lib/clarius_solum"; do
    if [[ ! -e "${req}" ]]; then
      echo "Missing: ${req}"
      missing=1
    fi
  done
  if [[ "${missing}" -eq 1 ]]; then
    echo "Run on host: $0 sync-deps"
    return 1
  fi
}

# Backward-compatible alias
ensure_openpi_on_host() { ensure_deps_on_host; }

run_in_container() {
  local cmd="$1"
  local tty_flag=()
  if [[ -t 0 ]]; then
    tty_flag=(-it)
  fi
  docker_cli exec "${tty_flag[@]}" "${CONTAINER}" bash -lc "${cmd}"
}

start_debug_container() {
  local extra_ports="${1:-}"
  docker_cli rm -f "${CONTAINER}" 2>/dev/null || true

  local port_args=()
  if [[ -n "${extra_ports}" ]]; then
    IFS=',' read -ra ports <<< "${extra_ports}"
    for p in "${ports[@]}"; do
      port_args+=(-p "127.0.0.1:${p}:${p}")
    done
  fi

  local display_args=()
  if [[ -n "${DISPLAY:-}" ]]; then
    display_args=(
      -e DISPLAY="${DISPLAY}"
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
    xhost +local:docker 2>/dev/null || true
  fi

  mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/i4h-assets"
  mkdir -p "${DOCKER_ROOT}/isaac-sim/cache/kit" \
           "${DOCKER_ROOT}/isaac-sim/cache/ov" \
           "${DOCKER_ROOT}/isaac-sim/cache/pip" \
           "${DOCKER_ROOT}/isaac-sim/cache/glcache" \
           "${DOCKER_ROOT}/isaac-sim/cache/computecache" \
           "${DOCKER_ROOT}/isaac-sim/logs" \
           "${DOCKER_ROOT}/isaac-sim/data" \
           "${DOCKER_ROOT}/isaac-sim/documents"

  local rti_args=()
  if [[ -n "${RTI_LICENSE_FILE:-}" && -f "${RTI_LICENSE_FILE}" ]]; then
    rti_args=(-v "${RTI_LICENSE_FILE}:/opt/rti/rti_license.dat:ro" -e RTI_LICENSE_FILE=/opt/rti/rti_license.dat)
  elif [[ -f "${DOCKER_ROOT}/rti/rti_license.dat" ]]; then
    rti_args=(-v "${DOCKER_ROOT}/rti/rti_license.dat:/opt/rti/rti_license.dat:ro" -e RTI_LICENSE_FILE=/opt/rti/rti_license.dat)
  fi

  docker_cli run -d --name "${CONTAINER}" \
    --gpus all \
    --runtime=nvidia \
    --entrypoint sleep \
    --ipc=host \
    --network host \
    "${port_args[@]}" \
    "${display_args[@]}" \
    "${rti_args[@]}" \
    -e HF_ENDPOINT="${HF_ENDPOINT}" \
    -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    -e OMNI_KIT_ACCEPT_EULA=Y \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -v "${ROOT}:/workspace/i4h-workflows" \
    -v "${CACHE_ROOT}/huggingface:/root/.cache/huggingface" \
    -v "${CACHE_ROOT}/i4h-assets:/root/.cache/i4h-assets" \
    -v "${DOCKER_ROOT}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "${DOCKER_ROOT}/isaac-sim/cache/ov:/root/.cache/ov:rw" \
    -v "${DOCKER_ROOT}/isaac-sim/cache/pip:/root/.cache/pip:rw" \
    -v "${DOCKER_ROOT}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "${DOCKER_ROOT}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw" \
    -v "${DOCKER_ROOT}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "${DOCKER_ROOT}/isaac-sim/data:/root/.local/share/ov/data:rw" \
    -v "${DOCKER_ROOT}/isaac-sim/documents:/root/Documents:rw" \
    -w /workspace/i4h-workflows \
    "${IMAGE}" infinity

  echo "Debug container started: ${CONTAINER}"
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  sync-openpi|sync-deps)
    mkdir -p "${ROOT}/third_party" "${ROOT}/install"
    cid=$(docker_cli create "${IMAGE}")
    for pkg in openpi lerobot IsaacLab Isaac-GR00T i4h-sensor-simulation; do
      if docker_cli cp "${cid}:/workspace/i4h-workflows/third_party/${pkg}" "${ROOT}/third_party/" 2>/dev/null; then
        echo "Synced ${pkg} → ${ROOT}/third_party/${pkg}"
      else
        echo "Warning: third_party/${pkg} not found in image"
      fi
    done
    if docker_cli cp "${cid}:/workspace/i4h-workflows/install/." "${ROOT}/install/" 2>/dev/null; then
      echo "Synced install → ${ROOT}/install"
    fi
    docker_cli rm "${cid}" >/dev/null
    ls "${ROOT}/third_party/openpi/src/openpi/models/pi0.py"
    ls "${ROOT}/third_party/IsaacLab/source/isaaclab/isaaclab/__init__.py"
    ;;
  sync-hf-cache)
    mkdir -p "${CACHE_ROOT}/huggingface"
    if [[ -d "${CACHE_ROOT}/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel/snapshots" ]]; then
      echo "HF cache already present:"
      du -sh "${CACHE_ROOT}/huggingface"
      exit 0
    fi
    # Try copying from any existing workflow/debug container (runtime cache, not in image layer)
    local src_cid=""
    for c in $(docker_cli ps -aq --filter ancestor="${IMAGE}" 2>/dev/null); do
      if docker_cli exec "${c}" test -d /root/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel 2>/dev/null; then
        src_cid="${c}"
        break
      fi
    done
    if [[ -n "${src_cid}" ]]; then
      echo "Copying HF cache from container ${src_cid:0:12}..."
      docker_cli cp "${src_cid}:/root/.cache/huggingface/." "${CACHE_ROOT}/huggingface/"
      du -sh "${CACHE_ROOT}/huggingface"
      exit 0
    fi
    echo "No local HF cache found in image or containers."
    echo "Run: $0 download-model   # uses hf-mirror.com, saves to ${CACHE_ROOT}/huggingface"
    exit 1
    ;;
  download-model)
    ensure_openpi_on_host
    mkdir -p "${CACHE_ROOT}/huggingface"
    start_debug_container
    echo "Downloading nvidia/Liver_Scan_Pi0_Cosmos_Rel via ${HF_ENDPOINT} ..."
    echo "Cache: ${CACHE_ROOT}/huggingface (resumable)"
    run_in_container "
      export HF_ENDPOINT='${HF_ENDPOINT}'
      export HF_HUB_ENABLE_HF_TRANSFER=0
      huggingface-cli download nvidia/Liver_Scan_Pi0_Cosmos_Rel --resume-download
    "
    du -sh "${CACHE_ROOT}/huggingface"
    ;;
  verify)
    ensure_openpi_on_host
    start_debug_container
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/verify_pi0_debug.sh"
    ;;
  shell)
    ensure_openpi_on_host
    start_debug_container
    docker_cli exec -it "${CONTAINER}" bash -l
    ;;
  exec)
    ensure_openpi_on_host
    run_in_container "$*"
    ;;
  infer)
    ensure_deps_on_host
    if [[ -z "${DISPLAY:-}" ]]; then
      echo "WARNING: DISPLAY not set — Isaac Sim may hang at 'Starting the simulation'."
      echo "  Fix: source ~/.i4h-deploy.env && xhost +local:docker"
      echo "  Or:  export DISPLAY=:1  (with TigerVNC running)"
    else
      echo "DISPLAY=${DISPLAY} (passed into debug container)"
    fi
    start_debug_container "5678"
    echo "Mode: eval.py (full Isaac Sim — slow, 5-15 min before infer() breakpoint)"
    echo "For fast policy breakpoints use: $0 infer-policy"
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/start_pi0_infer_debug.sh"
    ;;
  infer-policy|infer_policy)
    ensure_deps_on_host
    start_debug_container "5678"
    echo "Mode: policy-only (no Isaac Sim — breakpoints hit in ~30s)"
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/start_pi0_policy_only_debug.sh"
    ;;
  infer-frame-gui|infer_frame_gui)
    ensure_deps_on_host
    if [[ -z "${DISPLAY:-}" ]]; then
      echo "ERROR: DISPLAY required for infer-frame-gui. Run: export DISPLAY=:1 && xhost +local:docker"
      exit 1
    fi
    start_debug_container "5678"
    echo "Mode: single-process frame GUI (Isaac viewport + inline panel, unified freeze)"
    echo "TigerVNC :5901 to watch. Cursor F5 -> PI0: Attach Frame GUI"
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/start_pi0_infer_frame_gui_debug.sh"
    ;;
  infer-full-frame|infer_full_frame)
    ensure_deps_on_host
    if [[ -z "${DISPLAY:-}" ]]; then
      echo "ERROR: DISPLAY required. Run: export DISPLAY=:1 && xhost +local:docker"
      exit 1
    fi
    start_debug_container "5678"
    echo "Mode: full DDS pipeline (sim + vis + ultrasound + policy, SIGSTOP sync)"
    echo "TigerVNC :5901. Wait ~2min for sim. Cursor F5 -> PI0: Attach Full Frame"
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/start_pi0_full_frame_debug.sh"
    ;;
  train)
    ensure_openpi_on_host
    start_debug_container "5679"
    echo "SSH tunnel from laptop: ssh -L 5679:localhost:5679 ubuntu@<host>"
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/start_pi0_train_debug.sh"
    ;;
  sim)
    ensure_openpi_on_host
    start_debug_container
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/start_sim_env.sh"
    ;;
  policy)
    ensure_openpi_on_host
    start_debug_container "5680"
    echo "Optional DDS policy debugpy — run sim_env first in another terminal."
    run_in_container "bash workflows/robotic_ultrasound/scripts/debug/start_pi0_policy_dds.sh"
    ;;
  stop)
    docker_cli rm -f "${CONTAINER}" 2>/dev/null || true
    docker_cli stop laughing_gates 2>/dev/null || true
    echo "Stopped debug + any leftover workflow containers."
    ;;
  help|*)
    sed -n '2,12p' "$0"
    ;;
esac
