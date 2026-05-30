# -*- coding: utf-8 -*-
"""
Per-user x/y normalization for chunk-wise SRP.

SRP_chunk.py: per-sequence local normalization (min/max x,y inside each window).
This script (like XYPlot_chunk.py):
1) Scan training_root for each user's max x / max y.
2) x_norm = x / max_x, y_norm = y / max_y (per user, not per sequence).
3) Pair-wise distance on normalized coordinates, clip to epsilon, then draw SRP.
"""

import os
import argparse

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from XYPlot_chunk import (
    ROOT,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    DEFAULT_TRAINING_ROOT,
    build_user_max_xy_from_training,
    _clean_df,
)


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []

    if "train" in data_root.lower():
        stride = chunk_size
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


def compute_srp_pair(seq, epsilon, norm_x, norm_y):
    coords = seq[:, :2].astype(np.float32)

    scale_x = max(float(norm_x), 1e-8)
    scale_y = max(float(norm_y), 1e-8)

    x_norm = coords[:, 0] / scale_x
    y_norm = coords[:, 1] / scale_y
    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp


_resize_tfms = {}


def _resize_transform(side: int):
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def draw_srp(seq, save_path, epsilon, norm_x, norm_y, output_size=0):
    if len(seq) < 2:
        return

    rp = compute_srp_pair(seq, epsilon, norm_x, norm_y)

    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)

    img = ((rp - rp_min) / denom * 255).astype(np.uint8)

    if output_size and int(output_size) > 0:
        s = int(output_size)
        pil = Image.fromarray(img, mode="L")
        out_pil = _resize_transform(s)(pil)
        img = np.asarray(out_pil, dtype=np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def process_dataset(
    dataset,
    data_root,
    out_dir,
    sizes,
    epsilon,
    user_max_xy,
    output_size=0,
):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("\n[Phase] Generating pair-wise SRP (per-user x/y normalization)...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        if user in user_max_xy:
            norm_x, norm_y = user_max_xy[user]
        else:
            norm_x, norm_y = GLOBAL_MAX_X, GLOBAL_MAX_Y
            print(
                "\n[WARN] User", user, "not in training scan; using GLOBAL_MAX_X/Y:",
                norm_x, norm_y,
            )

        print("\n------------------------------")
        print("User:", user, "| norm max_x/max_y (from training):", norm_x, norm_y)

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]
            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:
                windows = generate_windows(events, chunk_size, data_root)
                print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")

                for i, seq in enumerate(windows):
                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png",
                    )
                    draw_srp(seq, save_path, epsilon, norm_x, norm_y, output_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset training_files.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[120])
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，用 transforms.Resize 将每张 SRP 存为 output_size×output_size PNG；0 表示保持原始 N×N。",
    )
    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = os.path.join(ROOT, training_rel)
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("[ROOT]", ROOT)
    print("[training_root]", training_rel)
    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)

    user_max_xy = build_user_max_xy_from_training(args.dataset, training_root)

    print("\nUSER_MAX_XY (from training_root):")
    for u in sorted(user_max_xy.keys()):
        print("  ", u, "->", user_max_xy[u])

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon,
        user_max_xy=user_max_xy,
        output_size=args.output_size,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
