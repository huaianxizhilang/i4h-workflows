# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process PI0 eval loop with Isaac Sim viewport + inline debug panel + frame breakpoints.

Everything runs in one Python process: one breakpoint freezes sim, infer, and GUI together.
"""
from __future__ import annotations

import argparse
import collections
import os
import signal
import subprocess
import sys
import time
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher
from simulation.environments.state_machine.utils import (
    RobotPositions,
    RobotQuaternions,
    capture_camera_images,
    compute_relative_action,
    get_joint_states,
    get_robot_obs,
)
from simulation.utils.common import resolve_checkpoint_path

# --- debug configuration (env vars) ---
# I4H_DEBUG_STOP: off | infer | step | both | all
# I4H_GUI: 1 = Isaac Sim viewport (no --headless)
# I4H_INLINE_PANEL: 1 = DearPyGui room/wrist/joint panel (same process)
# I4H_SPAWN_ULTRASOUND: 1 = spawn ultrasound_raytracing child (DDS; frozen via SIGSTOP with sim)
# I4H_SPAWN_VIS: 1 = spawn utils.visualization child (DDS; frozen via SIGSTOP)

_CHILD_PIDS: list[int] = []


def _child_pids() -> Iterable[int]:
    for pid in _CHILD_PIDS:
        if pid > 0:
            try:
                os.kill(pid, 0)
                yield pid
            except OSError:
                pass


def _freeze_children() -> None:
    for pid in _child_pids():
        try:
            os.kill(pid, signal.SIGSTOP)
        except OSError:
            pass


def _thaw_children() -> None:
    for pid in _child_pids():
        try:
            os.kill(pid, signal.SIGCONT)
        except OSError:
            pass


def _debug_pause(tag: str, frame: int, extra: str = "") -> None:
    mode = os.environ.get("I4H_DEBUG_STOP", "both").lower()
    if mode in ("off", "0", "false", "no"):
        return
    # Skip noisy reset-phase pauses unless explicitly enabled
    if frame < 0 and os.environ.get("I4H_DEBUG_RESET", "0") not in ("1", "true", "yes"):
        return
    if mode not in (tag, "both", "all"):
        return
    label = f"[frame-debug] frame={frame} stage={tag}"
    if extra:
        label += f" {extra}"
    print(label, flush=True)
    _freeze_children()
    try:
        if os.environ.get("I4H_USE_DEBUGPY", "1") in ("1", "true", "yes"):
            if sys.gettrace() is not None or __import__("debugpy").is_client_connected():
                __import__("debugpy").breakpoint()
            elif os.environ.get("I4H_FRAME_MANUAL", "0") in ("1", "true"):
                input(f"{label} — press Enter to continue...")
        else:
            input(f"{label} — press Enter to continue...")
    finally:
        _thaw_children()


class InlineDebugPanel:
    """Lightweight DearPyGui panel (same process as sim — freezes with breakpoints)."""

    def __init__(self) -> None:
        self._enabled = os.environ.get("I4H_INLINE_PANEL", "1") in ("1", "true", "yes")
        self._room_tag = None
        self._wrist_tag = None
        self._status_tag = None
        self._h = 224
        self._w = 224

    def setup(self) -> None:
        if not self._enabled:
            return
        try:
            import dearpygui.dearpygui as dpg
        except Exception as exc:
            print(f"[frame-debug] DearPyGui unavailable: {exc}", flush=True)
            self._enabled = False
            return
        try:
            dpg.create_context()
            with dpg.window(label="PI0 Frame Debug (inline)", tag="main", width=760, height=520):
                with dpg.group(horizontal=True):
                    dpg.add_text("Room")
                    dpg.add_text("Wrist")
                with dpg.group(horizontal=True):
                    with dpg.texture_registry(show=False):
                        blank = [0] * (self._w * self._h * 4)
                        dpg.add_raw_texture(self._w, self._h, blank, format=dpg.mvFormat_RGBA8, tag="tex_room")
                        dpg.add_raw_texture(self._w, self._h, blank, format=dpg.mvFormat_RGBA8, tag="tex_wrist")
                    dpg.add_image("tex_room", tag="img_room")
                    dpg.add_image("tex_wrist", tag="img_wrist")
                dpg.add_separator()
                dpg.add_text("frame=0 step=0", tag="status_line")
                dpg.add_text("joints: (waiting)", tag="joint_line")
            dpg.create_viewport(title="PI0 Frame Debug Panel", width=780, height=560, x_pos=820, y_pos=50)
            dpg.setup_dearpygui()
            dpg.show_viewport()
            self._status_tag = "status_line"
            self._room_tag = "tex_room"
            self._wrist_tag = "tex_wrist"
            print("[frame-debug] DearPyGui panel started (may appear beside Isaac Sim window)", flush=True)
        except Exception as exc:
            print(f"[frame-debug] DearPyGui setup failed: {exc}", flush=True)
            self._enabled = False

    def update(self, room: np.ndarray, wrist: np.ndarray, joints: np.ndarray, frame: int, stage: str) -> None:
        if not self._enabled:
            return
        import dearpygui.dearpygui as dpg

        def _to_rgba(img: np.ndarray) -> list[float]:
            if img.shape[:2] != (self._h, self._w):
                from PIL import Image

                img = np.array(Image.fromarray(img).resize((self._w, self._h)))
            rgba = np.zeros((self._h, self._w, 4), dtype=np.float32)
            rgba[..., :3] = img[..., :3] / 255.0
            rgba[..., 3] = 1.0
            return (rgba * 255).astype(np.uint8).flatten().tolist()

        dpg.set_value(self._room_tag, _to_rgba(room))
        dpg.set_value(self._wrist_tag, _to_rgba(wrist))
        dpg.set_value(self._status_tag, f"frame={frame} stage={stage}  (paused=breakpoint freezes all)")
        dpg.set_value("joint_line", f"joints: {np.array2string(joints, precision=4, separator=', ')}")
        dpg.render_dearpygui_frame()


def _spawn_optional_children() -> None:
    global _CHILD_PIDS
    scripts = "/workspace/i4h-workflows/workflows/robotic_ultrasound/scripts"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{scripts}:{env.get('PYTHONPATH', '')}"

    if os.environ.get("I4H_SPAWN_VIS", "0") in ("1", "true", "yes"):
        p = subprocess.Popen(
            [sys.executable, "-m", "utils.visualization"],
            cwd=scripts,
            env=env,
            preexec_fn=os.setsid,
        )
        _CHILD_PIDS.append(p.pid)
        print(f"[frame-debug] spawned visualization pid={p.pid}")

    if os.environ.get("I4H_SPAWN_ULTRASOUND", "0") in ("1", "true", "yes"):
        p = subprocess.Popen(
            [sys.executable, "-m", "simulation.examples.ultrasound_raytracing"],
            cwd=scripts,
            env=env,
            preexec_fn=os.setsid,
        )
        _CHILD_PIDS.append(p.pid)
        print(f"[frame-debug] spawned ultrasound_raytracing pid={p.pid}")

    if _CHILD_PIDS:
        print("[frame-debug] DDS child processes started — require sim_with_dds for live DDS feeds.")
        print("[frame-debug] eval single-process sim does NOT publish DDS; use infer-full-frame for DDS panels.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PI0 frame-step debug with optional GUI.")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--task", type=str, default="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0")
    parser.add_argument("--ckpt_path", type=str, default="nvidia/Liver_Scan_Pi0_Cosmos_Rel")
    parser.add_argument("--repo_id", type=str, default="i4h/sim_liver_scan")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def get_reset_action(env, device: str, use_rel: bool = True):
    reset_pos = torch.tensor(RobotPositions.SETUP, device=device)
    reset_quat = torch.tensor(RobotQuaternions.DOWN, device=device)
    reset_tensor = torch.cat([reset_pos, reset_quat], dim=-1)
    reset_tensor = reset_tensor.repeat(env.unwrapped.num_envs, 1)
    if not use_rel:
        return reset_tensor
    robot_obs = get_robot_obs(env)
    return compute_relative_action(reset_tensor, robot_obs)


def main() -> None:
    if os.path.isdir("/root/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    parser = _build_parser()
    # GUI mode: do not pass --headless
    if os.environ.get("I4H_GUI", "1") in ("1", "true", "yes"):
        if "--headless" in sys.argv:
            sys.argv.remove("--headless")
    else:
        if "--headless" not in sys.argv:
            sys.argv.append("--headless")

    args_cli = parser.parse_args()
    args_cli.ckpt_path = resolve_checkpoint_path(args_cli.ckpt_path)

    # Load PI0 *before* AppLauncher — Isaac Sim shadows pydantic and breaks openpi import.
    print("[frame-debug] Loading PI0 policy (before Isaac Sim)...", flush=True)
    from policy.pi0.runners import PI0PolicyRunner

    policy_runner = PI0PolicyRunner(
        ckpt_path=args_cli.ckpt_path,
        repo_id=args_cli.repo_id,
        task_description="Perform a liver ultrasound.",
    )
    print("[frame-debug] PI0 policy loaded.", flush=True)

    _spawn_optional_children()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from robotic_us_ext import tasks  # noqa: F401

    panel = InlineDebugPanel()
    panel.setup()

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()
    reset_steps = 40
    max_timesteps = 250

    for i in range(reset_steps):
        reset_tensor = get_reset_action(env, args_cli.device)
        env.step(reset_tensor)
        _debug_pause("step", frame=-1, extra=f"reset {i + 1}/{reset_steps}")

    replan_steps = 5
    frame_idx = 0

    print("[frame-debug] Main loop ready. I4H_DEBUG_STOP=", os.environ.get("I4H_DEBUG_STOP", "both"))
    print("[frame-debug] Isaac Sim viewport on VNC when I4H_GUI=1 and DISPLAY set.")

    while simulation_app.is_running():
        with torch.inference_mode():
            action_plan = collections.deque()
            for t in range(max_timesteps):
                if not action_plan:
                    rgb_images, _, _ = capture_camera_images(
                        env, ["room_camera", "wrist_camera"], device=env.unwrapped.device
                    )
                    room_img = rgb_images[0, 0, ...].cpu().numpy()
                    wrist_img = rgb_images[0, 1, ...].cpu().numpy()
                    joints = get_joint_states(env)[0]
                    panel.update(room_img, wrist_img, joints, frame_idx, "pre-infer")
                    _debug_pause("infer", frame=frame_idx, extra=f"timestep={t} before infer()")
                    t0 = time.perf_counter()
                    action_chunk = policy_runner.infer(
                        room_img=room_img, wrist_img=wrist_img, current_state=joints
                    )
                    infer_ms = (time.perf_counter() - t0) * 1000
                    print(f"[frame-debug] infer took {infer_ms:.1f} ms", flush=True)
                    action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft().astype(np.float32)
                action_t = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                _debug_pause("step", frame=frame_idx, extra=f"timestep={t} before env.step()")
                env.step(action_t)
                frame_idx += 1
                simulation_app.update()

            env.reset()
            frame_idx = 0
            for _ in range(reset_steps):
                env.step(get_reset_action(env, args_cli.device))

    env.close()
    for pid in list(_child_pids()):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    simulation_app.close()


if __name__ == "__main__":
    main()
