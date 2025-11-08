
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EB-Man (Embodied Benchmark Manipulation) -> CLIPort (RavensDataset) converter.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Please `pip install pillow` first.")


# ---- Constants to mimic CLIPort defaults ----
BOUNDS = np.array([[0.25, 0.75],   # x-range (m)
                   [-0.5, 0.5],    # y-range (m)
                   [0.0, 0.28]])   # z-range (m)
N_CAMS = 3        # CLIPort's standard RealSense D415 3-view setup
RGB_SIZE = (640, 480)  # (W,H)
SWAP_XY = False
FLIP_X = False
FLIP_Y = False
FLIP_Z = False
NEGATE_YAW = False

@dataclass
class EbStep:
    img_path: str
    action: List[int]  # 7 numbers
    action_success: float


@dataclass
class EbSegment:
    input_image_path: str
    steps: List[EbStep]


@dataclass
class EbEpisode:
    model_name: str
    eval_set: str
    episode_id: int
    instruction: str
    success: float
    segments: List[EbSegment]


# ---------------- Utility functions ----------------

def _safe_join(root: str, rel: str) -> Optional[str]:
    """
    Try a few common patterns to resolve an EB relative image path.
    """
    candidates = [
        os.path.join(root, rel),
        os.path.join(root, 'images', rel),
        os.path.join(root, 'images', 'images', rel),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _resize_to_rgb(img_path: str) -> np.ndarray:
    """Load image, convert to RGB, resize to 640x480 (W,H)."""
    with Image.open(img_path) as im:
        im = im.convert('RGB')
        im = im.resize(RGB_SIZE, resample=Image.BILINEAR)
        rgb = np.array(im, dtype=np.uint8)  # (H,W,3)? PIL is (W,H) on size; numpy array is (H,W,3)
    return rgb


def _synth_depth_like(rgb: np.ndarray, z_hint_m: float = 0.6) -> np.ndarray:
    """
    Create a simple synthetic depth map (H,W) in meters.
    We use a constant plane by default; z_hint_m can be varied per sample.
    """
    H, W = rgb.shape[0], rgb.shape[1]
    depth = np.ones((H, W), dtype=np.float32) * z_hint_m
    return depth


def _xyznorm_to_meters(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Map EB normalized [0..100] XYZ to CLIPort bounds in meters,
    with optional swap/flip remaps applied before mapping.
    """
    # 1) optional remaps in 0..100 space
    if SWAP_XY:
        x, y = y, x
    if FLIP_X:
        x = 100.0 - x
    if FLIP_Y:
        y = 100.0 - y
    if FLIP_Z:
        z = 100.0 - z

    # 2) scale to meters using BOUNDS
    xr = BOUNDS[0, 1] - BOUNDS[0, 0]
    yr = BOUNDS[1, 1] - BOUNDS[1, 0]
    zr = BOUNDS[2, 1] - BOUNDS[2, 0]
    xm = BOUNDS[0, 0] + (x / 100.0) * xr
    ym = BOUNDS[1, 0] + (y / 100.0) * yr
    zm = BOUNDS[2, 0] + (z / 100.0) * zr
    return float(xm), float(ym), float(zm)


def _euler_xyz_deg_to_quat_xyzw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float,float,float,float]:
    # If coordinate frames disagree, flipping yaw is a common fix
    if NEGATE_YAW:
        yaw_deg = -yaw_deg

    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    cr = np.cos(r/2.0); sr = np.sin(r/2.0)
    cp = np.cos(p/2.0); sp = np.sin(p/2.0)
    cy = np.cos(y/2.0); sy = np.sin(y/2.0)

    qw = cr*cp*cy - sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy + sr*sp*cy
    return float(qx), float(qy), float(qz), float(qw)


def _parse_action_str(a_str: str) -> List[int]:
    # Actions are stored as strings like "[35, 57, 26, 6, 61, 36, 1]"
    try:
        # ast.literal_eval would be safest, but we avoid import; simple parse:
        s = a_str.strip().lstrip('[').rstrip(']')
        parts = [int(p.strip()) for p in s.split(',')]
        if len(parts) != 7:
            raise ValueError("Expected 7 numbers in action")
        return parts
    except Exception as e:
        raise ValueError(f"Could not parse action string: {a_str}")


def _derive_pick_place(actions_7d: List[List[int]]) -> Optional[Tuple[List[int], List[int]]]:
    """
    Given a list of EB 7D actions for a segment, return the first (pick, place) pair
    using 1->0 (open->close) as pick, and next 0->1 as place. If not found, return None.
    """
    if not actions_7d:
        return None
    g_prev = actions_7d[0][6]
    pick = None
    place = None
    for a in actions_7d[1:]:  # start from 2nd since we compare transitions
        g = a[6]
        if pick is None and (g_prev == 1 and g == 0):
            pick = a
        elif pick is not None and place is None and (g_prev == 0 and g == 1):
            place = a
            break
        g_prev = g
    if pick is not None and place is not None:
        return pick, place
    return None


def _episode_to_cliport_entries(ep: EbEpisode,
                                eb_root: str,
                                keep_unlabeled: bool = False) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], float, Dict[str, Any]]]:
    """
    Convert one EB episode into a list of CLIPort records (obs, act, reward, info).

    Returns a list of tuples suitable for RavensDataset.add():
      obs: {'color': [3x(H,W,3)], 'depth': [3x(H,W)]}
      act: {'pose0': (xyz, quat), 'pose1': (xyz, quat)} or None
      reward: float
      info: {'lang_goal': instruction, 'episode_id': int, 'model': str, 'eval_set': str}
    """
    results = []
    for seg_idx, seg in enumerate(ep.segments):
        # load the segment's input image as the observation
        in_path = _safe_join(eb_root, seg.input_image_path)
        if in_path is None:
            # Skip segment if input image missing
            continue
        rgb = _resize_to_rgb(in_path)
        # Simple synthetic depth plane; z_hint from mid of bounds (0.14m)
        depth = _synth_depth_like(rgb, z_hint_m=float(BOUNDS[2].mean()))

        # Duplicate to N_CAMS views (placeholder to satisfy CLIPort's get_fused_heightmap)
        obs_color = [rgb.copy() for _ in range(N_CAMS)]
        obs_depth = [depth.copy() for _ in range(N_CAMS)]

        # pick/place from segment's low-level actions
        actions_raw = [st.action for st in seg.steps]
        pick_place = _derive_pick_place(actions_raw)

        if pick_place is None and not keep_unlabeled:
            # Skip unlabeled segment
            continue

        act = None
        if pick_place is not None:
            pick7, place7 = pick_place

            # Map normalized positions to meters
            p0_xyz = _xyznorm_to_meters(pick7[0], pick7[1], pick7[2])
            p1_xyz = _xyznorm_to_meters(place7[0], place7[1], place7[2])

            # Convert Euler units to degrees (1 unit = 3 degrees), then to quat
            def _to_deg(u): return float(u) * 3.0
            p0_quat = _euler_xyz_deg_to_quat_xyzw(_to_deg(pick7[3]), _to_deg(pick7[4]), _to_deg(pick7[5]))
            p1_quat = _euler_xyz_deg_to_quat_xyzw(_to_deg(place7[3]), _to_deg(place7[4]), _to_deg(place7[5]))

            act = {'pose0': (p0_xyz, p0_quat),
                   'pose1': (p1_xyz, p1_quat)}

        # reward: use segment-level success if any step indicates success; else episode success
        # (EB provides action_success per step; we consider a logical OR across steps)
        reward = 1.0 if any(s.action_success >= 1.0 for s in seg.steps) else float(ep.success)

        info = {
            'lang_goal': ep.instruction,
            'episode_id': ep.episode_id,
            'model': ep.model_name,
            'eval_set': ep.eval_set,
            'segment_index': seg_idx
        }

        obs = {'color': np.uint8(np.stack(obs_color, axis=0)),  # (N_CAMS,H,W,3)
               'depth': np.float32(np.stack(obs_depth, axis=0))} # (N_CAMS,H,W)

        # RavensDataset.add() expects obs['color'] / obs['depth'] "per step" (we will save arrays later)
        # Here we just return per-step obs in a format we can stack further.
        results.append((obs, act, reward, info))

    return results


# ---------------- JSON parsing ----------------

def _read_eb_json(json_path: str) -> List[EbEpisode]:
    """
    Parse eb-man_dataset_multi_step.json into structured objects.
    The file groups episodes under different model_name/eval_set buckets.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    episodes: List[EbEpisode] = []

    # The observed structure is a list of buckets, each having keys like:
    # { "model_name": "...", "eval_set": "...", "episodes": [...] }  OR
    # { "model_name": "...", "eval_set": "...", "episode_id": ..., "trajectory": [...] } etc.
    # We'll support both "episodes" and a flat list fallback.
    def _coerce_episode(d: Dict[str, Any]) -> Optional[EbEpisode]:
        try:
            model_name = d.get('model_name', 'unknown')
            eval_set = d.get('eval_set', 'unknown')
            instruction = d.get('instruction') or d.get('task') or ""
            episode_id = int(d.get('episode_id'))
            success = float(d.get('success', 0.0))

            segments: List[EbSegment] = []
            # Source can be either "trajectory": [ ... segments ... ]
            # or a list of dicts each with 'executable_plan' + 'input_image_path'
            traj = d.get('trajectory', [])
            for seg in traj:
                steps = []
                exec_plan = seg.get('executable_plan', [])
                for s in exec_plan:
                    a = s.get('action')
                    if isinstance(a, str):
                        a7 = _parse_action_str(a)
                    elif isinstance(a, list) and len(a) == 7 and all(isinstance(x, (int, float)) for x in a):
                        a7 = [int(x) for x in a]
                    else:
                        # Skip malformed
                        continue
                    steps.append(EbStep(
                        img_path=s.get('img_path', ''),
                        action=a7,
                        action_success=float(s.get('action_success', 0.0))
                    ))
                ipath = seg.get('input_image_path')
                if ipath and steps:
                    segments.append(EbSegment(input_image_path=ipath, steps=steps))

            if not segments:
                return None

            return EbEpisode(
                model_name=model_name,
                eval_set=eval_set,
                episode_id=episode_id,
                instruction=instruction,
                success=success,
                segments=segments
            )
        except Exception:
            return None

    # If the top-level is {'episodes':[...]} under multiple buckets
    if isinstance(data, list):
        # Could be a list of buckets
        for item in data:
            if isinstance(item, dict) and 'episodes' in item:
                for ep in item['episodes']:
                    epd = _coerce_episode(ep)
                    if epd:
                        episodes.append(epd)
            elif isinstance(item, dict) and 'trajectory' in item:
                epd = _coerce_episode(item)
                if epd:
                    episodes.append(epd)
            else:
                # Sometimes buckets nest more keys; try deeper
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, list):
                            for ep in v:
                                if isinstance(ep, dict) and 'trajectory' in ep:
                                    epd = _coerce_episode(ep)
                                    if epd:
                                        episodes.append(epd)
    elif isinstance(data, dict):
        # Single bucket with 'episodes' or a single episode
        if 'episodes' in data and isinstance(data['episodes'], list):
            for ep in data['episodes']:
                epd = _coerce_episode(ep)
                if epd:
                    episodes.append(epd)
        elif 'trajectory' in data:
            epd = _coerce_episode(data)
            if epd:
                episodes.append(epd)

    return episodes


# ---------------- Writer ----------------

def _write_cliport_pickles(out_path: str,
                           entries: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], float, Dict[str, Any]]],
                           start_index: int,
                           seed: int) -> int:
    """
    Write CLIPort pickles for a list of entries (one EB segment -> one CLIPort sample).
    Returns how many entries were written.
    """
    import pickle

    if not entries:
        return 0

    # Build per-episode arrays
    color_steps = []
    depth_steps = []
    action_steps = []
    reward_steps = []
    info_steps = []

    for obs, act, rew, info in entries:
        color_steps.append(obs['color'])   # (N_CAMS,H,W,3)
        depth_steps.append(obs['depth'])   # (N_CAMS,H,W)
        action_steps.append(act)           # dict or None
        reward_steps.append(float(rew))
        info_steps.append(info)

    color_arr = np.uint8(np.stack(color_steps, axis=0))   # (T,N_CAMS,H,W,3)
    depth_arr = np.float32(np.stack(depth_steps, axis=0)) # (T,N_CAMS,H,W)

    # Prepare directories
    for field in ('color', 'depth', 'action', 'reward', 'info'):
        d = os.path.join(out_path, field)
        os.makedirs(d, exist_ok=True)

    # Filename pattern: '{episode:06d}-{seed}.pkl'
    fname = f'{start_index:06d}-{seed}.pkl'
    def _dump(obj, field):
        with open(os.path.join(out_path, field, fname), 'wb') as f:
            pickle.dump(obj, f)

    _dump(color_arr, 'color')
    _dump(depth_arr, 'depth')
    _dump(action_steps, 'action')
    _dump(reward_steps, 'reward')
    _dump(info_steps, 'info')

    return 1


def _split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, rng: random.Random):
    # Normalize ratios
    s = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / s, val_ratio / s, test_ratio / s
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_tr = int(round(n * train_ratio))
    n_va = int(round(n * val_ratio))
    tr = idxs[:n_tr]
    va = idxs[n_tr:n_tr+n_va]
    te = idxs[n_tr+n_va:]
    return tr, va, te


def main():
    import gc

    ap = argparse.ArgumentParser()
    ap.add_argument('--eb_root', required=True, help='Root folder of EB-Man_trajectory_dataset (contains images/...)')
    ap.add_argument('--json', required=True, help='Path to eb-man_dataset_multi_step.json')
    ap.add_argument('--out_root', required=True, help='Where to write CLIPort dataset tree')
    ap.add_argument('--task_name', default='eb-manipulation-seq', help='Task name prefix for CLIPort dataset')
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--keep_unlabeled', action='store_true', help='Keep segments without pick/place pair (act=None)')
    ap.add_argument('--swap_xy', action='store_true', help='Swap x and y before mapping')
    ap.add_argument('--flip_x', action='store_true', help='Flip x (x -> 100-x) before mapping')
    ap.add_argument('--flip_y', action='store_true', help='Flip y (y -> 100-y) before mapping')
    ap.add_argument('--flip_z', action='store_true', help='Flip z (z -> 100-z) before mapping')
    ap.add_argument('--negate_yaw', action='store_true', help='Flip sign of yaw before quaternion')
    args = ap.parse_args()
    global SWAP_XY, FLIP_X, FLIP_Y, FLIP_Z, NEGATE_YAW
    SWAP_XY   = args.swap_xy
    FLIP_X    = args.flip_x
    FLIP_Y    = args.flip_y
    FLIP_Z    = args.flip_z
    NEGATE_YAW= args.negate_yaw
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
   

    # 1) Parse JSON into lightweight Python objects (no images yet)
    episodes = _read_eb_json(args.json)
    if not episodes:
        print("No episodes parsed from JSON. Check the file structure.", file=sys.stderr)
        return 2

    # 2) Split by episode indices (so we never need all image arrays in RAM)
    n_eps = len(episodes)
    tr_idx, va_idx, te_idx = _split_indices(n_eps, args.train_ratio, args.val_ratio, args.test_ratio, rng)
    splits = [('train', tr_idx), ('val', va_idx), ('test', te_idx)]

    # 3) Convert + write ONE episode at a time (streaming)
    total_written = 0
    for split_name, idxs in splits:
        split_dir = os.path.join(args.out_root, f'{args.task_name}-{split_name}')
        os.makedirs(split_dir, exist_ok=True)

        ep_counter = 0
        for i, k in enumerate(idxs):
            # Build obs/labels only for this episode
            entries = _episode_to_cliport_entries(
                episodes[k],
                args.eb_root,
                keep_unlabeled=args.keep_unlabeled
            )
            if not entries:
                continue

            # Deterministic seed per episode
            seed = (args.seed * 100003 + k) % 1000000

            # Write immediately, then free memory
            n_written = _write_cliport_pickles(
                split_dir,
                entries,
                start_index=ep_counter,
                seed=seed
            )
            total_written += n_written
            ep_counter += n_written

            del entries
            gc.collect()

            # Optional progress: print every 10 episodes
            if (i + 1) % 10 == 0:
                print(f"[{split_name}] processed {i + 1}/{len(idxs)} episodes...", flush=True)

        print(f"Wrote {ep_counter} episode(s) to {split_name}")

    print(f"Done. Total CLIPort episodes written: {total_written}")
    print(f"Dataset root: {args.out_root}")
    print(f"Try training with: train.task={args.task_name}, train.data_dir={args.out_root}")
    return 0


if __name__ == '__main__':
    sys.exit(main())


# python eb_to_cliport.py \
#   --eb_root /home/ubuntu/cliport/data/embodiedbench/EB-Man_trajectory_dataset \
#   --json /home/ubuntu/cliport/data/embodiedbench/EB-Man_trajectory_dataset/eb-man_dataset_multi_step.json \
#   --out_root ./cliport_data \
#   --task_name eb-manipulation-seq \
#   --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
#   --seed 14
#   --swap_xy

# python ./cliport/train.py \
#   train.agent=cliport \
#   train.task=eb-manipulation-seq \
#   train.data_dir=/home/ubuntu/cliport/cliport_data \
#   dataset.type=single \
#   train.n_demos=500 \
#   train.n_val=32 \
#   train.n_steps=20000 \
#   train.gpu=[0] \
#   train.train_dir=/home/ubuntu/cliport/cliport_quickstart/eb-manipulation-seq-cliport-n1000-train \
#   train.log=false


# export CLIPORT_ROOT=/home/ubuntu/cliport
# export PYTHONPATH="$CLIPORT_ROOT:$PYTHONPATH"
# ln -sfn /home/ubuntu/cliport/cliport_data/eb-manipulation-seq-test \
#        /home/ubuntu/cliport/cliport_data/stack-block-pyramid-seq-seen-colors-test

# python cliport/eval.py \
#   mode=test \
#   eval_task=stack-block-pyramid-seq-seen-colors \
#   type=single \
#   data_dir=/home/ubuntu/cliport/cliport_data \
#   n_demos=138 \
#   agent=cliport \
#   model_path=/home/ubuntu/cliport/cliport_quickstart/eb-manipulation-seq-cliport-n1000-train/checkpoints \
#   train_config=/home/ubuntu/cliport/cliport_quickstart/eb-manipulation-seq-cliport-n1000-train/.hydra/config.yaml \
#   checkpoint_type=last \
#   save_results=true update_results=true \
#   save_path=/home/ubuntu/cliport/cliport_quickstart/eb-manipulation-seq-cliport-n1000-train/eval \
#   results_path=/home/ubuntu/cliport/cliport_quickstart/eb-manipulation-seq-cliport-n1000-train/eval
