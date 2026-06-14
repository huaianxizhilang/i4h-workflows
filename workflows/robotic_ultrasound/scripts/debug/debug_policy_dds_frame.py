#!/usr/bin/env python3
"""DDS policy loop with frame breakpoints + SIGSTOP sync for sim/vis/ultrasound children."""
from __future__ import annotations

import os
import signal
import time

import numpy as np
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.subscriber import SubscriberWithCallback
from PIL import Image
from policy.pi0.runners import PI0PolicyRunner
from simulation.utils.common import resolve_checkpoint_path

_FRAME = 0
current_state = {"room_cam": None, "wrist_cam": None, "joint_pos": None}


def _pids() -> list[int]:
    out = []
    for x in os.environ.get("I4H_CHILD_PIDS", "").split():
        try:
            pid = int(x)
            os.kill(pid, 0)
            out.append(pid)
        except (ValueError, OSError):
            pass
    return out


def _freeze() -> None:
    for pid in _pids():
        try:
            os.kill(pid, signal.SIGSTOP)
        except OSError:
            pass


def _thaw() -> None:
    for pid in _pids():
        try:
            os.kill(pid, signal.SIGCONT)
        except OSError:
            pass


def _pause(tag: str) -> None:
    global _FRAME
    mode = os.environ.get("I4H_DEBUG_STOP", "both")
    if mode in ("off", "0"):
        return
    if mode not in (tag, "both", "all"):
        return
    print(f"[full-frame] pause frame={_FRAME} stage={tag}", flush=True)
    _freeze()
    try:
        if os.environ.get("I4H_USE_DEBUGPY", "1") in ("1", "true"):
            import debugpy

            if debugpy.is_client_connected():
                debugpy.breakpoint()
    finally:
        _thaw()
    _FRAME += 1


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="pi0")
    parser.add_argument("--ckpt_path", default="nvidia/Liver_Scan_Pi0_Cosmos_Rel")
    parser.add_argument("--repo_id", default="i4h/sim_liver_scan")
    parser.add_argument("--task_description", default="Perform a liver ultrasound.")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--chunk_length", type=int, default=50)
    parser.add_argument("--topic_in_room_camera", default="topic_room_camera_data_rgb")
    parser.add_argument("--topic_in_wrist_camera", default="topic_wrist_camera_data_rgb")
    parser.add_argument("--topic_in_franka_pos", default="topic_franka_info")
    parser.add_argument("--topic_out", default="topic_franka_ctrl")
    args = parser.parse_args()

    ckpt = resolve_checkpoint_path(args.ckpt_path)
    policy = PI0PolicyRunner(ckpt_path=ckpt, repo_id=args.repo_id, task_description=args.task_description)
    hz = 30

    class PolicyPublisher(Publisher):
        def __init__(self, topic: str, domain_id: int):
            super().__init__(topic, FrankaCtrlInput, 1 / hz, domain_id)

        def produce(self, dt: float, sim_time: float):
            room_img = Image.fromarray(
                np.frombuffer(current_state["room_cam"], dtype=np.uint8).reshape(args.height, args.width, 3), "RGB"
            )
            wrist_img = Image.fromarray(
                np.frombuffer(current_state["wrist_cam"], dtype=np.uint8).reshape(args.height, args.width, 3), "RGB"
            )
            joint_pos = current_state["joint_pos"]
            _pause("infer")
            t0 = time.perf_counter()
            actions = policy.infer(
                room_img=np.array(room_img),
                wrist_img=np.array(wrist_img),
                current_state=np.array(joint_pos[:7]),
            )
            print(f"[full-frame] infer {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
            _pause("step")
            msg = FrankaCtrlInput()
            msg.joint_positions = np.array(actions).astype(np.float32).reshape(args.chunk_length * 6).tolist()
            return msg

    writer = PolicyPublisher(args.topic_out, args.domain_id)

    def dds_callback(topic, data):
        if topic == args.topic_in_room_camera:
            current_state["room_cam"] = data.data
        elif topic == args.topic_in_wrist_camera:
            current_state["wrist_cam"] = data.data
        elif topic == args.topic_in_franka_pos:
            current_state["joint_pos"] = data.joints_state_positions
        if all(current_state[k] is not None for k in current_state):
            writer.write(0.1, 1.0)
            for k in current_state:
                current_state[k] = None

    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_room_camera, CameraInfo, 1 / hz).start()
    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_wrist_camera, CameraInfo, 1 / hz).start()
    SubscriberWithCallback(dds_callback, args.domain_id, args.topic_in_franka_pos, FrankaInfo, 1 / hz).start()


if __name__ == "__main__":
    main()
