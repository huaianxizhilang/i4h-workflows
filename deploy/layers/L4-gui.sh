#!/usr/bin/env bash
# L4 — Remote GUI / X11 for Docker (reusable)
# Headless Ubuntu (no desktop): use I4H_GUI_MODE=vnc or default auto (falls back to VNC).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"
load_config

info "L4 GUI setup (mode=${I4H_GUI_MODE})"

case "${I4H_GUI_MODE}" in
    headless)
        info "Headless mode — Xvfb virtual display (no remote desktop)"
        if ! pgrep -x Xvfb >/dev/null 2>&1; then
            apt_install xvfb
            Xvfb "${DISPLAY:-:99}" -screen 0 1920x1080x24 &
            sleep 1
            export DISPLAY="${DISPLAY:-:99}"
        fi
        ;;
    vnc)
        info "Headless/server mode — installing TigerVNC + XFCE desktop"
        apt_install tigervnc-standalone-server xfce4 xfce4-goodies dbus-x11 x11-utils
        start_vnc_server
        open_vnc_firewall
        enable_vnc_autostart
        ;;
    x11-ssh|auto)
        if [[ -z "${DISPLAY:-}" ]]; then
            if [[ "${I4H_GUI_MODE}" == "auto" ]]; then
                warn "No DISPLAY (typical on headless Ubuntu) — auto-fallback to TigerVNC + XFCE"
                I4H_GUI_MODE=vnc
                exec bash "${SCRIPT_DIR}/L4-gui.sh"
            fi
            die "DISPLAY empty. Connect with: ssh -X user@host"
        fi
        info "Using SSH X11 forwarding DISPLAY=${DISPLAY}"
        apt_install x11-utils
        ;;
esac

setup_display_for_docker

# Quick X11 smoke test (non-fatal)
if [[ "${I4H_GUI_MODE}" != "headless" ]]; then
    if command -v xdpyinfo >/dev/null 2>&1; then
        xdpyinfo >/dev/null 2>&1 && info "X11 display OK (${DISPLAY})" || warn "xdpyinfo failed — GUI may not work"
    else
        apt_install x11-utils
        xdpyinfo >/dev/null 2>&1 && info "X11 display OK (${DISPLAY})" || warn "xdpyinfo failed"
    fi
fi

if [[ "${I4H_GUI_MODE}" == "vnc" ]]; then
    cat <<EOF

TigerVNC ready for headless Ubuntu:
  - Connect from your PC: <server-ip>:$(( 5900 + $(vnc_display_number) ))
  - DISPLAY on server: ${I4H_VNC_DISPLAY}
  - Ensure cloud security group / firewall allows TCP $(( 5900 + $(vnc_display_number) ))

EOF
fi

info "L4 complete"
