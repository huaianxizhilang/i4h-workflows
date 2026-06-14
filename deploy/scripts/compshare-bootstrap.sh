#!/usr/bin/env bash
# 优云智算（CompShare）新购机器初始化：DNS 加速、数据盘、SSH 公钥。
# 由 L0 在 preflight 检查之前调用（需 I4H_COMPSHARE_BOOTSTRAP=1）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

run_compshare_bootstrap() {
    [[ "${I4H_COMPSHARE_BOOTSTRAP:-0}" == "1" ]] || return 0

    info "======== CompShare bootstrap (优云智算新机初始化) ========"

    if [[ "${I4H_COMPSHARE_DNS:-1}" == "1" ]]; then
        configure_compshare_dns
    fi

    if [[ "${I4H_AUTO_SETUP_DATA_DISK:-1}" == "1" ]]; then
        setup_data_disk
        # 确保后续层能解析 /data
        export I4H_DATA_ROOT="${I4H_DATA_ROOT:-${I4H_DATA_DISK_MOUNT:-/data}}"
        resolve_storage_paths
    fi

    if [[ "${I4H_INSTALL_SSH_KEYS:-1}" == "1" ]]; then
        install_ssh_authorized_keys
    fi

    if [[ "${I4H_COMPSHARE_DNS:-1}" == "1" && "${I4H_VERIFY_DNS_ACCELERATION:-1}" == "1" ]]; then
        verify_compshare_dns_acceleration
    fi

    print_storage_layout
    info "CompShare bootstrap complete"
}

configure_compshare_dns() {
    require_root_or_sudo
    local servers=(${I4H_COMPSHARE_DNS_SERVERS:-100.90.90.90 100.90.90.100})
    local netplan_file=""

    for candidate in /etc/netplan/50-cloud-init.yaml /etc/netplan/01-netcfg.yaml; do
        [[ -f "${candidate}" ]] && netplan_file="${candidate}" && break
    done
    [[ -n "${netplan_file}" ]] || { warn "No netplan file found — skip DNS config"; return 0; }

    info "Configuring CompShare DNS in ${netplan_file}: ${servers[*]}"

    run_root python3 - "${netplan_file}" "${servers[@]}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
servers = sys.argv[2:]
text = path.read_text(encoding="utf-8")

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

if yaml is not None:
    data = yaml.safe_load(text) or {}
    net = data.setdefault("network", {})
    ethernets = net.setdefault("ethernets", {})
    if not ethernets:
        ethernets["all-en"] = {"match": {"name": "e*"}, "dhcp4": True}
    for name, cfg in ethernets.items():
        if not isinstance(cfg, dict):
            continue
        cfg["nameservers"] = {"addresses": servers}
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
else:
    # Fallback: append or replace nameservers block under first ethernet
    lines = text.splitlines()
    out, in_eth, inserted = [], False, False
    for line in lines:
        if line.strip().startswith("ethernets:"):
            in_eth = True
        if in_eth and line.strip().startswith("nameservers:"):
            out.append("      nameservers:")
            out.append("        addresses:")
            for s in servers:
                out.append(f"          - {s}")
            inserted = True
            continue
        if inserted and line.startswith("        addresses:"):
            continue
        if inserted and line.strip().startswith("- "):
            continue
        if inserted and line.strip() and not line.startswith(" "):
            inserted = False
        out.append(line)
    if not any("nameservers:" in l for l in out):
        # append minimal block
        out.append("      nameservers:")
        out.append("        addresses:")
        for s in servers:
            out.append(f"          - {s}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")

print("DNS servers configured:", ", ".join(servers))
PY

    run_root netplan apply
    info "netplan apply done (DNS persists across reboot)"
}

verify_compshare_dns_acceleration() {
    local domains=(
        github.com
        nvidia.com
        nvcr.io
        docker.com
        golang.org
        googlesource.com
        pythonhosted.org
        pytorch.org
        huggingface.co
        anaconda.org
        conda.io
        anaconda.com
    )
    info "Verifying DNS resolution (CompShare UAAA)..."
    local ok=0 fail=0
    for d in "${domains[@]}"; do
        if timeout 5 getent hosts "${d}" >/dev/null 2>&1; then
            info "  DNS OK: ${d}"
            ok=$((ok + 1))
        else
            warn "  DNS FAIL: ${d}"
            fail=$((fail + 1))
        fi
    done
    info "DNS check: ${ok} ok, ${fail} fail (see https://www.compshare.cn/docs/operation/gpu/uaaa)"
    (( fail == 0 )) || warn "Some domains failed — downloads may be slow until DNS propagates"
}

detect_data_disk_device() {
    # 找容量在 [MIN, MAX] GB 之间、非系统根盘、未挂载的块设备（常见 /dev/vdb）
    local min_gb="${I4H_DATA_DISK_MIN_GB:-200}"
    local max_gb="${I4H_DATA_DISK_MAX_GB:-400}"
    local root_disk=""
    root_disk="$(findmnt -n -o SOURCE / | sed 's/p[0-9]*$//' | sed 's/[0-9]*$//' | xargs basename 2>/dev/null || echo vda)"

    local name size_gb type mount
    while read -r name size_gb type mount; do
        [[ "${type}" == "disk" ]] || continue
        [[ "${name}" == "${root_disk}" ]] && continue
        (( size_gb >= min_gb && size_gb <= max_gb )) || continue
        if [[ -z "${mount}" ]]; then
            echo "/dev/${name}"
            return 0
        fi
    done < <(lsblk -bdn -o NAME,SIZE,TYPE,MOUNTPOINT | awk '{
        gsub(/[^0-9]/,"",$2); sz=int($2/1024/1024/1024);
        print $1, sz, $3, $4
    }')

    # 已挂载到目标路径
    local mount="${I4H_DATA_DISK_MOUNT:-/data}"
    if mountpoint -q "${mount}" 2>/dev/null; then
        findmnt -n -o SOURCE "${mount}"
        return 0
    fi
    return 1
}

setup_data_disk() {
    require_root_or_sudo
    local mount="${I4H_DATA_DISK_MOUNT:-/data}"
    local owner="${I4H_DATA_DISK_OWNER:-ubuntu}"
    local dev=""

    if mountpoint -q "${mount}" 2>/dev/null; then
        info "Data disk already mounted at ${mount}"
        run_root resize2fs "$(findmnt -n -o SOURCE "${mount}")" 2>/dev/null || true
    else
        dev="$(detect_data_disk_device)" || die "No data disk (~${I4H_DATA_DISK_MIN_GB}-${I4H_DATA_DISK_MAX_GB}GB) found. Attach disk or mount ${mount} manually."
        info "Using data disk device: ${dev}"

        local fstype
        fstype="$(lsblk -no FSTYPE "${dev}" 2>/dev/null | head -1)"
        if [[ -z "${fstype}" ]]; then
            info "Formatting ${dev} as ext4..."
            run_root mkfs.ext4 -F "${dev}"
        else
            info "Data disk ${dev} already formatted (${fstype})"
        fi

        run_root mkdir -p "${mount}"
        if ! mountpoint -q "${mount}"; then
            run_root mount "${dev}" "${mount}" || die "Failed to mount ${dev} on ${mount}"
        fi
        run_root resize2fs "${dev}" 2>/dev/null || true
    fi

    dev="$(findmnt -n -o SOURCE "${mount}")"
    local uuid
    uuid="$(blkid -s UUID -o value "${dev}")"
    [[ -n "${uuid}" ]] || die "Could not read UUID for ${dev}"

    if ! grep -q "UUID=${uuid}" /etc/fstab 2>/dev/null; then
        info "Adding ${mount} to /etc/fstab (UUID=${uuid})"
        echo "UUID=${uuid} ${mount} ext4 defaults,nofail 0 2" | run_root tee -a /etc/fstab >/dev/null
    fi

    if id "${owner}" >/dev/null 2>&1; then
        run_root chown -R "${owner}:${owner}" "${mount}"
    fi

    export I4H_DATA_ROOT="${mount}"
    info "Data disk ready: ${mount} ($(df -h "${mount}" | awk 'NR==2 {print $2" total, "$4" free"}'))"
}

install_ssh_authorized_keys() {
    require_root_or_sudo
    local keys_file="${I4H_SSH_AUTHORIZED_KEYS_FILE:-${DEPLOY_ROOT}/config/ssh-authorized-keys.pub}"
    [[ -f "${keys_file}" ]] || { warn "SSH keys file missing: ${keys_file}"; return 0; }

    local users=(${I4H_SSH_AUTHORIZED_USERS:-${USER}})
    [[ -n "${I4H_LOGIN_USER:-}" ]] && users+=("${I4H_LOGIN_USER}")
    local u added=0 key home_dir
    for u in $(printf '%s\n' "${users[@]}" | awk '!seen[$0]++'); do
        home_dir="$(getent passwd "${u}" 2>/dev/null | cut -d: -f6 || true)"
        [[ -n "${home_dir}" && -d "${home_dir}" ]] || continue
        run_root mkdir -p "${home_dir}/.ssh"
        run_root chmod 700 "${home_dir}/.ssh"
        run_root touch "${home_dir}/.ssh/authorized_keys"
        run_root chmod 600 "${home_dir}/.ssh/authorized_keys"
        while IFS= read -r key || [[ -n "${key}" ]]; do
            [[ -z "${key}" || "${key}" =~ ^# ]] && continue
            if run_root grep -qF "${key}" "${home_dir}/.ssh/authorized_keys" 2>/dev/null; then
                continue
            fi
            echo "${key}" | run_root tee -a "${home_dir}/.ssh/authorized_keys" >/dev/null
            added=$((added + 1))
            info "Added SSH key for ${u}: ${key##* }"
        done < "${keys_file}"
    done
    info "SSH authorized_keys: ${added} new key(s)"
}

# 供 L0 直接 source 后调用
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    load_config
    run_compshare_bootstrap
fi
