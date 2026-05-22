# -*- coding: utf-8 -*-
"""
Chunk-wise speed-magnitude temporal encoding (vertical stripes only).

Per chunk (same windowing as SRP_chunk_velocity.py):
  R = G = B = speed magnitude |v| via global CDF + vertical stripe (np.tile along rows)

Velocity pipeline follows RecurrencePlot/SRP_velocity.py.
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.stats import rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

GLOBAL_V_CDF = None


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


def load_raw_velocity_distribution(path):
    data = np.load(path)
    velocities = data["values"]

    print("\n[Velocity Distribution]")
    print("Samples:", len(velocities))
    print("Min:", velocities.min())
    print("Max:", velocities.max())

    return velocities


def build_runtime_cdf(raw_v, clip_pct):
    print("\nBuilding velocity runtime CDF")

    v_upper = np.percentile(raw_v, clip_pct)
    v_clipped = raw_v[raw_v <= v_upper]

    ranks = rankdata(v_clipped, method="average")
    cdf = (ranks - 1) / (len(v_clipped) - 1 + 1e-8)

    order = np.argsort(v_clipped)
    v_sorted = v_clipped[order]
    cdf_sorted = cdf[order]

    print("Runtime samples:", len(v_sorted))
    print("Runtime max:", v_sorted.max())

    return v_sorted, cdf_sorted


def compute_velocity(xs, ys, ts):
    dt = np.maximum(np.diff(ts), 1e-5)

    v = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
    v = np.concatenate([[v[0]], v])

    return v


def clean_balabit(df):
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state",
    })

    df = df[df["state"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


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


def compute_velocity_image(seq):
    """
    R = G = B = speed magnitude vertical stripes.
    Returns float32 H×W×3 in [0, 1], channels RGB (R, G, B).
    """
    T = len(seq)
    if T < 2:
        return None

    xs = seq[:, 0]
    ys = seq[:, 1]
    ts = seq[:, 2]

    v = compute_velocity(xs, ys, ts)

    v_norm = np.interp(
        v,
        GLOBAL_V_CDF[0],
        GLOBAL_V_CDF[1],
        left=0,
        right=1,
    )

    stripe = np.tile(v_norm[None, :], (T, 1)).astype(np.float32)

    img = np.stack([stripe, stripe, stripe], axis=-1)
    return np.clip(img, 0, 1)


_resize_tfms = {}


def _resize_transform(side: int):
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def draw_velocity(seq, save_path, output_size=0):
    if len(seq) < 2:
        return

    img = compute_velocity_image(seq)
    if img is None:
        return

    img_rgb = (img * 255).astype(np.uint8)

    if output_size and int(output_size) > 0:
        s = int(output_size)
        pil = Image.fromarray(img_rgb, mode="RGB")
        out_pil = _resize_transform(s)(pil)
        img_rgb = np.asarray(out_pil, dtype=np.uint8)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def process_dataset(dataset, data_root, out_dir, sizes, output_size=0):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("\n[Phase] Generating velocity vertical stripes (R=G=B)...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]
            df = pd.read_csv(path)

            if dataset.lower() == "balabit":
                df = clean_balabit(df)
            else:
                for c in ["x", "y", "time"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["x", "y", "time"])

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
                    draw_velocity(seq, save_path, output_size)


def main():
    global GLOBAL_V_CDF

    parser = argparse.ArgumentParser(
        description="Chunk velocity vertical stripes (R=G=B).",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--velocity_dist",
        required=True,
        help="npz with values array (e.g. velocity_distribution_raw.npz)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[120])
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，用 transforms.Resize 将每张图存为 output_size×output_size PNG；0 表示保持 N×N。",
    )
    parser.add_argument(
        "--v_percentile",
        type=float,
        default=100,
        help="Upper clip percentile for speed CDF (same as SRP_velocity).",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    dist_path = resolve_path(args.velocity_dist)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Resolved velocity_dist:", dist_path)

    raw_v = load_raw_velocity_distribution(dist_path)
    GLOBAL_V_CDF = build_runtime_cdf(raw_v, args.v_percentile)

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        output_size=args.output_size,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
