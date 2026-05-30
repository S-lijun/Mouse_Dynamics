# -*- coding: utf-8 -*-
"""
Chunk-wise SRP with trajectory position on canvas.

1) XYPlot_chunk mapping: per-user max x/y -> trajectory pixel positions on white canvas.
2) Trajectory bbox (min/max x,y) padded to a square in final TARGET_SIZE image.
3) SRP_chunk local normalization for N×N recurrence matrix, resized into that square.
4) Final image: white background with an inner square containing SRP.
"""

import os
import argparse

import pandas as pd
import numpy as np
import cv2

from XYPlot_chunk import (
    ROOT,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    TARGET_SIZE,
    INNER_PADDING,
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

    stride = chunk_size
    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


def compute_srp_pair(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    scale = max(x_max - x_min, y_max - y_min)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - x_min) / scale
    y_norm = (y - y_min) / scale
    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp


def srp_to_uint8(rp):
    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)
    return ((rp - rp_min) / denom * 255).astype(np.uint8)


def trajectory_points_final(
    xs,
    ys,
    norm_width,
    norm_height,
    target_size=TARGET_SIZE,
    inner_padding=INNER_PADDING,
):
    """Map raw trajectory coords to final canvas pixel positions (same as XYPlot_chunk)."""
    W = max(float(norm_width), 1.0)
    H = max(float(norm_height), 1.0)
    a = H / W
    canvas_w = int(W) + 1
    span = float(canvas_w - 1)
    canvas_h = int(np.ceil(a * span)) + 1

    x_pix = np.clip(np.rint(xs / W * span), 0, canvas_w - 1).astype(np.float64)
    y_pix = np.clip(np.rint(ys / W * span), 0, canvas_h - 1).astype(np.float64)

    effective_size = max(1, target_size - 2 * inner_padding)
    scale = effective_size / max(canvas_w, canvas_h)

    new_w = int(canvas_w * scale)
    new_h = int(canvas_h * scale)

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    x_final = x_pix * scale + pad_left
    y_final = y_pix * scale + pad_top
    return x_final, y_final


def pad_rect_to_square(min_x, max_x, min_y, max_y):
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    side = max(w, h)

    pad_w = side - w
    pad_h = side - h

    sq_min_x = min_x - pad_w // 2
    sq_min_y = min_y - pad_h // 2
    sq_max_x = sq_min_x + side - 1
    sq_max_y = sq_min_y + side - 1
    return sq_min_x, sq_max_x, sq_min_y, sq_max_y, side


def clip_square_to_canvas(sq_min_x, sq_max_x, sq_min_y, sq_max_y, canvas_size):
    side = sq_max_x - sq_min_x + 1

    if sq_min_x < 0:
        sq_max_x -= sq_min_x
        sq_min_x = 0
    if sq_min_y < 0:
        sq_max_y -= sq_min_y
        sq_min_y = 0
    if sq_max_x >= canvas_size:
        shift = sq_max_x - canvas_size + 1
        sq_min_x -= shift
        sq_max_x = canvas_size - 1
    if sq_max_y >= canvas_size:
        shift = sq_max_y - canvas_size + 1
        sq_min_y -= shift
        sq_max_y = canvas_size - 1

    sq_min_x = max(0, sq_min_x)
    sq_min_y = max(0, sq_min_y)
    sq_max_x = min(canvas_size - 1, sq_max_x)
    sq_max_y = min(canvas_size - 1, sq_max_y)

    side_x = sq_max_x - sq_min_x + 1
    side_y = sq_max_y - sq_min_y + 1
    side = min(side_x, side_y, side)
    return sq_min_x, sq_min_y, side


def compute_trajectory_square_bbox(
    seq,
    norm_width,
    norm_height,
    target_size=TARGET_SIZE,
    inner_padding=INNER_PADDING,
):
    xs = seq[:, 0].astype(np.float64)
    ys = seq[:, 1].astype(np.float64)

    x_final, y_final = trajectory_points_final(
        xs, ys, norm_width, norm_height, target_size, inner_padding
    )

    min_x = int(np.floor(np.min(x_final)))
    max_x = int(np.ceil(np.max(x_final)))
    min_y = int(np.floor(np.min(y_final)))
    max_y = int(np.ceil(np.max(y_final)))

    min_x = max(0, min(min_x, target_size - 1))
    max_x = max(0, min(max_x, target_size - 1))
    min_y = max(0, min(min_y, target_size - 1))
    max_y = max(0, min(max_y, target_size - 1))

    sq_min_x, sq_max_x, sq_min_y, sq_max_y, _ = pad_rect_to_square(
        min_x, max_x, min_y, max_y
    )
    x0, y0, side = clip_square_to_canvas(
        sq_min_x, sq_max_x, sq_min_y, sq_max_y, target_size
    )
    return x0, y0, side


def render_srp_with_position(seq, epsilon, norm_x, norm_y, target_size=TARGET_SIZE):
    rp = compute_srp_pair(seq, epsilon)
    srp = srp_to_uint8(rp)

    x0, y0, side = compute_trajectory_square_bbox(
        seq, norm_x, norm_y, target_size=target_size
    )
    side = max(1, int(side))

    srp_square = cv2.resize(srp, (side, side), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
    x1 = min(target_size, x0 + side)
    y1 = min(target_size, y0 + side)
    paste_w = x1 - x0
    paste_h = y1 - y0
    canvas[y0:y1, x0:x1] = srp_square[:paste_h, :paste_w]
    return canvas


def draw_srp_pos(seq, save_path, epsilon, norm_x, norm_y, output_size=TARGET_SIZE):
    if len(seq) < 2:
        return

    out_size = int(output_size) if output_size else TARGET_SIZE
    img = render_srp_with_position(seq, epsilon, norm_x, norm_y, target_size=out_size)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def process_dataset(
    dataset,
    data_root,
    out_dir,
    sizes,
    epsilon,
    user_max_xy,
    output_size=TARGET_SIZE,
):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("\n[Phase] Generating positioned SRP (XYPlot bbox + SRP_chunk local norm)...")

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
                    draw_srp_pos(seq, save_path, epsilon, norm_x, norm_y, output_size)


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
        default=TARGET_SIZE,
        help="Final canvas size (white background with inner SRP square).",
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
