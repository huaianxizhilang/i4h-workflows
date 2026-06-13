#!/usr/bin/env bash
# L4 — Remote GUI / X11 for Docker (reusable)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"
load_config

info "L4 GUI setup (mode=${I4H_GUI_MODE})"

case "${I4H_GUI_MODE}" in
    headless)
        info "Headless mode — skipping GUI setup"
        if ! pgrep -x Xvfb >/dev/null 2>&1; then
            apt_install xvfb
            Xvfb "${DISPLAY:-:99}" -screen 0 1920x1080x24 &
            sleep 1
            export DISPLAY="${DISPLAY:-:99}"
        fi
        ;;
    vnc)
        apt_install tigervnc-standalone-server xfce4 xfce4-goodies dbus-x11
        export DISPLAY="${I4H_VNC_DISPLAY}"
        if ! vncserver -list 2>/dev/null | grep -q "${I4H_VNC_DISPLAY#:}"; then
            if [[ ! -f "${HOME}/.vnc/passwd" ]]; then
                warn "Run vncpasswd once before first VNC start, or set VNC password now:"
                vncpasswd || die "vncpasswd required for VNC mode"
            fi
            vncserver "${I4H_VNC_DISPLAY}" -geometry "${I4H_VNC_GEOMETRY}" -depth 24
        fi
        info "VNC listening on port $(( 5900 + ${I4H_VNC_DISPLAY#:} )) (DISPLAY=${DISPLAY})"
        ;;
    x11-ssh|auto)
        if [[ -z "${DISPLAY:-}" ]]; then
            if [[ "${I4H_GUI_MODE}" == "auto" ]]; then
                warn "DISPLAY unset — attempting VNC fallback"
                I4H_GUI_MODE=vnc
                exec bash "${SCRIPT_DIR}/L4-gui.sh"
            fi
            die "DISPLAY empty. Connect with: ssh -X user@host"
        fi
        info "Using SSH X11 forwarding DISPLAY=${DISPLAY}"
        ;;
esac

setup_display_for_docker

# Quick X11 smoke test (non-fatal)
if [[ "${I4H_GUI_MODE}" != "headless" ]] && command -v xdpyinfo >/dev/null 2>&1; then
    xdpyinfo >/dev/null 2>&1 && info "X11 display OK" || warn "xdpyinfo failed — GUI may not work"
elif [[ "${I4H_GUI_MODE}" != "headless" ]]; then
    apt_install x11-utils
    xdpyinfo >/dev/null 2>&1 && info "X11 display OK" || warn "xdpyinfo failed"
fi

info "L4 complete"
