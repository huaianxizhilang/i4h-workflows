#!/usr/bin/env python3
"""Create minimal HDF5 + LeRobot dataset for LoRA training smoke/debug (no sim required)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from training.convert_hdf5_to_lerobot import Pi0FeatureDict, main as convert_hdf5_to_lerobot


def create_dummy_hdf5(out_dir: Path, num_steps: int = 50, num_episodes: int = 2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = 32, 32  # small images; convert resizes to 224x224
    for ep in range(num_episodes):
        hdf5_path = out_dir / f"data_{ep}.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            # convert_hdf5_to_lerobot expects data/demo_0 in every file (one episode per file)
            root = f.create_group("data/demo_0")
            root.create_dataset("action", data=np.random.rand(num_steps, 6).astype(np.float32))
            root.create_dataset("abs_joint_pos", data=np.random.rand(num_steps, 7).astype(np.float32))
            obs = root.create_group("observations")
            rgb = np.random.randint(0, 256, size=(num_steps, 2, height, width, 3), dtype=np.uint8)
            obs.create_dataset("rgb_images", data=rgb)
            obs.create_dataset("depth_images", data=rgb)
            obs.create_dataset("seg_images", data=rgb)
        print(f"[mock-data] wrote {hdf5_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="i4h/debug_lora_mock")
    parser.add_argument("--task_prompt", default="Perform a liver ultrasound.")
    parser.add_argument("--data_dir", default="/data/cache/i4h-mock-train/hdf5")
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    create_dummy_hdf5(data_dir, num_steps=args.num_steps, num_episodes=args.num_episodes)

    lerobot_home = os.environ.get("LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
    target = Path(lerobot_home) / args.repo_id.replace("/", os.sep)
    if target.exists():
        print(f"[mock-data] LeRobot dataset already exists: {target}")
        print("[mock-data] delete it first to regenerate.")
        return

    convert_hdf5_to_lerobot(
        str(data_dir),
        args.repo_id,
        args.task_prompt,
        Pi0FeatureDict(),
    )
    print(f"[mock-data] LeRobot dataset ready: {target}")


if __name__ == "__main__":
    main()
