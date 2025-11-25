#!/usr/bin/env python3
"""
eb_to_cliport_exact.py

Convert your EB-Manipulation JSON (format as provided) into a CLIPort RavensDataset.

Usage example:

  conda activate cliport
  cd ~/cliport
  export CLIPORT_ROOT=~/cliport

  python eb_to_cliport_exact.py \
    --eb_json /home/ubuntu/cliport/data/embodiedbench/EB-Man_trajectory_dataset/eb-man_dataset_single_step.json \
    --image_root /home/ubuntu/cliport/data/embodiedbench/EB-Man_trajectory_dataset \
    --out_root $CLIPORT_ROOT/data \
    --task_name eb-manip-single-step \
    --train_fraction 0.9 \
    --only_success 0
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from cliport.dataset import RavensDataset


# ---------- helpers ----------

def parse_action_str(action_str: str) -> List[int]:
    """Convert '[33, 43, 27, 0, 60, 90, 1]' -> [33, 43, 27, 0, 60, 90, 1]."""
    s = action_str.strip()
    if "[" in s and "]" in s:
        s = s[s.find("[") + 1 : s.rfind("]")]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def find_pick_and_place(actions: List[List[int]]) -> Tuple[int, int]:
    """
    Given list of 7D actions [x,y,z,r,p,yaw,g], return (pick_idx, place_idx)
    using:

      pick  = first gripper transition 1 -> 0
      place = first gripper transition 0 -> 1 after pick

    Indices are 0-based into the 'actions' list.
    """
    if not actions:
        return None, None

    prev_g = actions[0][6]
    pick_idx = None
    place_idx = None

    for i in range(1, len(actions)):
        g = actions[i][6]
        if pick_idx is None and prev_g == 1 and g == 0:
            pick_idx = i
        elif pick_idx is not None and place_idx is None and prev_g == 0 and g == 1:
            place_idx = i
            break
        prev_g = g

    if pick_idx is None or place_idx is None:
        return None, None

    return pick_idx, place_idx


def euler_xyz_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler XYZ -> quaternion [x,y,z,w]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def eb_grid_to_world_xyz(x: float, y: float, z: float) -> np.ndarray:
    """
    Map EB [0..100] grid to CLIPort Ravens workspace.

    Typical CLIPort bounds:
      x ∈ [0.25, 0.75]
      y ∈ [-0.5, 0.5]
      z ∈ [0, 0.28]
    """
    xn = np.clip(x / 100.0, 0.0, 1.0)
    yn = np.clip(y / 100.0, 0.0, 1.0)
    zn = np.clip(z / 100.0, 0.0, 1.0)

    x_min, x_max = 0.25, 0.75
    y_min, y_max = -0.5, 0.5
    z_min, z_max = 0.0, 0.28

    x_world = x_min + (x_max - x_min) * xn
    y_world = y_min + (y_max - y_min) * yn
    z_world = z_min + (z_max - z_min) * zn

    return np.array([x_world, y_world, z_world], dtype=np.float32)


def load_and_resize_image(path: Path, out_h: int = 320, out_w: int = 160) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


# ---------- main conversion ----------

def convert_eb_to_cliport(
    eb_json: Path,
    image_root: Path,
    out_root: Path,
    task_name: str,
    train_fraction: float,
    img_h: int,
    img_w: int,
    only_success: bool,
):
    # Load JSON
    with open(eb_json, "r") as f:
        episodes = json.load(f)

    print(f"Loaded {len(episodes)} EB episodes")

    # First pass: determine which episodes have valid pick/place
    valid_entries = []  # each: (ep_idx, pick_idx, place_idx)
    for ep_idx, ep in enumerate(episodes):
        success = float(ep.get("success", 0.0))
        if only_success and success <= 0.0:
            continue

        traj = ep.get("trajectory", [])
        if not traj:
            continue

        # Collect actions across steps
        actions: List[List[int]] = []
        for step in traj:
            plan = step.get("executable_plan", None)
            if not plan:
                continue
            a_str = plan.get("action", None)
            if not a_str:
                continue
            try:
                a = parse_action_str(a_str)
            except Exception:
                continue
            if len(a) != 7:
                continue
            actions.append(a)

        if len(actions) < 2:
            continue

        pick_idx, place_idx = find_pick_and_place(actions)
        if pick_idx is None or place_idx is None:
            continue

        valid_entries.append((ep_idx, pick_idx, place_idx))

    print(f"Episodes with valid pick/place: {len(valid_entries)}")

    if not valid_entries:
        print("No valid episodes found. Check success filter or gripper patterns.")
        return

    # Train/val split
    n_total = len(valid_entries)
    n_train = int(train_fraction * n_total)
    train_entries = valid_entries[:n_train]
    val_entries = valid_entries[n_train:]

    print(f"Train episodes: {len(train_entries)}, Val episodes: {len(val_entries)}")

    cfg = {
        "dataset": {
            "images": True,
            "cache": False,
            "augment": {"theta_sigma": 60},
        }
    }

    train_path = out_root / f"{task_name}-train"
    val_path   = out_root / f"{task_name}-val"

    train_ds = RavensDataset(str(train_path), cfg, n_demos=0, augment=False)
    val_ds   = RavensDataset(str(val_path),   cfg, n_demos=0, augment=False)

    def build_episode(ep: Dict[str, Any], pick_idx: int, place_idx: int):
        """Return (obs, act, reward, info) for a single CLIPort demo."""
        instr = ep.get("instruction", "")
        success = float(ep.get("success", 0.0))
        episode_id = ep.get("episode_id", None)
        traj = ep["trajectory"]

        # Rebuild the full actions list indexed by time
        actions: List[List[int]] = []
        for step in traj:
            plan = step.get("executable_plan", None)
            if not plan:
                actions.append(None)
                continue
            a_str = plan.get("action", None)
            if not a_str:
                actions.append(None)
                continue
            try:
                a = parse_action_str(a_str)
            except Exception:
                actions.append(None)
                continue
            actions.append(a)

        if pick_idx >= len(actions) or place_idx >= len(actions):
            raise RuntimeError("pick_idx/place_idx out of range")

        pick_action = actions[pick_idx]
        place_action = actions[place_idx]
        if pick_action is None or place_action is None:
            raise RuntimeError("pick_action/place_action is None")

        def eb_action_to_pose(action):
            x, y, z, r, p, yw, g = action

            xyz = eb_grid_to_world_xyz(x, y, z)

            roll_deg  = r   * 3.0
            pitch_deg = p   * 3.0
            yaw_deg   = yw  * 3.0

            roll  = np.deg2rad(roll_deg)
            pitch = np.deg2rad(pitch_deg)
            yaw   = np.deg2rad(yaw_deg)

            quat = euler_xyz_to_quat_xyzw(roll, pitch, yaw)
            return xyz, quat

        pose0 = eb_action_to_pose(pick_action)
        pose1 = eb_action_to_pose(place_action)

        act = {
            "pose0": pose0,
            "pose1": pose1,
        }

        # Observation image: BEFORE the pick action
        obs_step = traj[pick_idx]   # same index into trajectory
        img_rel = obs_step["input_image_path"]  # e.g. "images/.../step_k.png"
        img_path = (image_root / img_rel).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        color = load_and_resize_image(img_path, out_h=img_h, out_w=img_w)
        depth = np.zeros((img_h, img_w), dtype=np.float32)

        obs = {
            "color": color,
            "depth": depth,
        }

        reward = success  # not really used by CLIPort BC

        info = {
            "lang_goal": instr,
            "eb_episode_id": episode_id,
            "pick_idx": pick_idx,
            "place_idx": place_idx,
        }

        return obs, act, reward, info

    # Actually write train demos
    seed = 0
    added_train = 0
    for ep_idx, pick_idx, place_idx in train_entries:
        ep = episodes[ep_idx]
        try:
            obs, act, reward, info = build_episode(ep, pick_idx, place_idx)
        except Exception as e:
            print(f"[Train] Skipping ep_idx={ep_idx}: {e}")
            continue
        train_ds.add(seed, [(obs, act, reward, info)])
        seed += 1
        added_train += 1

    # Val demos
    seed = 0
    added_val = 0
    for ep_idx, pick_idx, place_idx in val_entries:
        ep = episodes[ep_idx]
        try:
            obs, act, reward, info = build_episode(ep, pick_idx, place_idx)
        except Exception as e:
            print(f"[Val] Skipping ep_idx={ep_idx}: {e}")
            continue
        val_ds.add(seed, [(obs, act, reward, info)])
        seed += 1
        added_val += 1

    print(f"Added train demos: {added_train}")
    print(f"Added val demos:   {added_val}")
    print(f"Train dataset path: {train_path}")
    print(f"Val dataset path:   {val_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb_json", type=str, required=True,
                        help="Path to EB JSON file (list of episodes).")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root folder that contains the 'images' directory.")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Output root, usually $CLIPORT_ROOT/data.")
    parser.add_argument("--task_name", type=str, default="eb-manip-single-step",
                        help="Task name used for <task_name>-train / -val dirs.")
    parser.add_argument("--train_fraction", type=float, default=0.9,
                        help="Train / val split fraction.")
    parser.add_argument("--img_h", type=int, default=320)
    parser.add_argument("--img_w", type=int, default=160)
    parser.add_argument("--only_success", type=int, default=0,
                        help="1 => only episodes with success>0, 0 => all episodes.")
    args = parser.parse_args()

    convert_eb_to_cliport(
        eb_json=Path(args.eb_json),
        image_root=Path(args.image_root),
        out_root=Path(args.out_root),
        task_name=args.task_name,
        train_fraction=args.train_fraction,
        img_h=args.img_h,
        img_w=args.img_w,
        only_success=bool(args.only_success),
    )


if __name__ == "__main__":
    main()


# python cliport/train.py \
#   train.task=eb-manip-single-step \
#   train.agent=cliport \
#   dataset.type=single \
#   train.data_dir=./data \
#   train.n_demos=500 \
#   train.n_steps=40000 \
#   train.exp_folder=eb_runs \
#   dataset.cache=False

# python cliport/train.py \
#   train.task=eb-manip-single-step \
#   train.agent=cliport \
#   dataset.type=single \
#   train.data_dir=/home/ubuntu/cliport/data \
#   train.n_demos=500 \
#   train.n_steps=40000 \
#   train.exp_folder=eb_runs \
#   dataset.cache=False