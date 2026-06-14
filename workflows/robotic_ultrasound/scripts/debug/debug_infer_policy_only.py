#!/usr/bin/env python3
"""Debug PI0 policy inference without Isaac Sim (fast path for breakpoints).

Hits runners.py:infer() and openpi policy.py:infer() within seconds.
"""
import os

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if os.path.isdir("/root/.cache/huggingface/hub/models--nvidia--Liver_Scan_Pi0_Cosmos_Rel"):
    os.environ["HF_HUB_OFFLINE"] = "1"

from policy.pi0.runners import PI0PolicyRunner
from simulation.utils.common import resolve_checkpoint_path


def main():
    ckpt_path = os.environ.get("CKPT_PATH", "nvidia/Liver_Scan_Pi0_Cosmos_Rel")
    repo_id = os.environ.get("REPO_ID", "i4h/sim_liver_scan")
    ckpt_path = resolve_checkpoint_path(ckpt_path)

    print(f"[policy-only] Loading checkpoint: {ckpt_path}")
    runner = PI0PolicyRunner(
        ckpt_path=ckpt_path,
        repo_id=repo_id,
        task_description="Perform a liver ultrasound.",
    )

    # Synthetic observations (same shape as eval.py)
    room_img = np.zeros((480, 640, 3), dtype=np.uint8)
    wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)
    current_state = np.zeros(7, dtype=np.float32)

    print("[policy-only] Calling infer() — set breakpoints in runners.py / policy.py")
    actions = runner.infer(room_img=room_img, wrist_img=wrist_img, current_state=current_state)
    print(f"[policy-only] actions shape: {actions.shape}, dtype: {actions.dtype}")
    print("[policy-only] Done.")


if __name__ == "__main__":
    main()
