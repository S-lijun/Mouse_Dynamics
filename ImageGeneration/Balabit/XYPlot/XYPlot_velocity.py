# -*- coding: utf-8 -*-
"""
XYPlot — Base + Velocity (Grayscale Encoding, OpenCV Fast)
-----------------------------------------------------------
- Uses RAW velocity distribution (training-only)
- Percentile clipping applied during encoding
- Velocity mapped to [0,1]
- Brightness = velocity
- RGB = (v, v, v)
- OpenCV rendering for high speed
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
from scipy.stats import rankdata

# ============================================================
# ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Base Config
# ============================================================
BASE_EVENT = 60
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 0.5
BASE_MARKERSIZE = 1.0

GLOBAL_V_RAW = None
GLOBAL_V_CDF = None

# ============================================================
# Scaling Utils
# ============================================================
def get_img_size(chunk_size):

    scale = chunk_size / BASE_EVENT
    side = np.sqrt(scale * BASE_IMG_SIZE * BASE_IMG_SIZE)

    return int(round(side))

def get_stroke_params(chunk_size):

    scale = np.sqrt(chunk_size / BASE_EVENT)

    lw = max(1, int(round(BASE_LINEWIDTH * scale)))
    ms = max(1, int(round(BASE_MARKERSIZE * scale)))

    return lw, ms

def _scaled(val, min_val, scale, offset):

    return (val - min_val) * scale + offset

# ============================================================
# Data Cleaning
# ============================================================
def clean_and_rename_cols(df):

    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state",
    })

    df = df[df["state"] == "Move"].copy()

    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask_valid = (df["x"] < 65535) & (df["y"] < 65535)

    df = df[mask_valid].dropna(subset=["x", "y", "time"]).reset_index(drop=True)

    return df

# ============================================================
# Load RAW Distribution
# ============================================================
def load_raw_velocity_distribution(path):

    data = np.load(path)

    velocities = data["velocities"]

    print(f"[Velocity] Loaded RAW distribution from {path}")
    print(f"[Velocity] Total samples: {len(velocities)}")
    print(f"[Velocity] Min: {velocities.min():.6f}")
    print(f"[Velocity] Max: {velocities.max():.6f}")

    return velocities

# ============================================================
# Build Runtime CDF
# ============================================================
def build_runtime_cdf(raw_velocities, clip_pct):

    print(f"\n[Velocity] Building runtime CDF (clip={clip_pct}%)")

    v_upper = np.percentile(raw_velocities, clip_pct)

    v_clipped = raw_velocities[raw_velocities <= v_upper]

    ranks = rankdata(v_clipped, method="average")

    cdf = (ranks - 1) / (len(v_clipped) - 1 + 1e-8)

    order = np.argsort(v_clipped)

    v_sorted = v_clipped[order]
    cdf_sorted = cdf[order]

    print(f"[Velocity] Runtime Min: {v_sorted.min():.6f}")
    print(f"[Velocity] Runtime Max: {v_sorted.max():.6f}")
    print(f"[Velocity] Runtime Samples: {len(v_sorted)}")

    return v_sorted, cdf_sorted

# ============================================================
# Drawing (OpenCV)
# ============================================================
def draw_mouse_chunk(chunk, save_path, chunk_size):

    if len(chunk) < 2:
        return

    IMG_SIZE = get_img_size(chunk_size)

    linewidth, markersize = get_stroke_params(chunk_size)

    # create white canvas
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    dx = np.diff(xs)
    dy = np.diff(ys)

    dt = np.maximum(np.diff(ts), 1e-5)

    velocity = np.sqrt(dx**2 + dy**2) / dt

    # --------------------------------------------------------
    # Map velocity → [0,1]
    # --------------------------------------------------------
    v_norm = np.interp(
        velocity,
        GLOBAL_V_CDF[0],
        GLOBAL_V_CDF[1],
        left=0.0,
        right=1.0
    )

    brightness = (v_norm * 255).astype(np.uint8)

    # --------------------------------------------------------
    # Spatial scaling
    # --------------------------------------------------------
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    pad_x = range_x * 0.05
    pad_y = range_y * 0.05

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    range_x = max_x - min_x
    range_y = max_y - min_y

    scale = min(IMG_SIZE / range_x, IMG_SIZE / range_y)

    offset_x = (IMG_SIZE - range_x * scale) / 2.0
    offset_y = (IMG_SIZE - range_y * scale) / 2.0

    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------
    for i in range(len(xs) - 1):

        x1 = int(_scaled(xs[i], min_x, scale, offset_x))
        y1 = int(_scaled(ys[i], min_y, scale, offset_y))

        x2 = int(_scaled(xs[i+1], min_x, scale, offset_x))
        y2 = int(_scaled(ys[i+1], min_y, scale, offset_y))

        val = int(brightness[i])

        color = (val, val, val)

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            color,
            thickness=linewidth,
            lineType=cv2.LINE_AA
        )

        cv2.circle(
            img,
            (x2, y2),
            markersize,
            color,
            -1,
            lineType=cv2.LINE_AA
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)

# ============================================================
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size):

    n_chunks = len(events) // chunk_size

    print(f"      [ChunkSize={chunk_size}] Total Chunks = {n_chunks}")

    for i in range(n_chunks):

        chunk = events[i * chunk_size:(i + 1) * chunk_size]

        save_path = os.path.join(
            out_dir,
            f"event{chunk_size}",
            user,
            f"{session_name}-{i}.png"
        )

        draw_mouse_chunk(chunk, save_path, chunk_size)

        if (i + 1) % 50 == 0 or (i + 1) == n_chunks:
            print(f"         -> Chunk {i+1}/{n_chunks} done")

# ============================================================
# Dataset Processing
# ============================================================
def process_dataset(data_root, out_dir, sizes):

    users = sorted(os.listdir(data_root))
    total_users = len(users)

    print(f"\n[Dataset] Total Users = {total_users}")

    for u_idx, user in enumerate(users, 1):

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\n====================================================")
        print(f"[User {u_idx}/{total_users}] Processing: {user}")
        print("====================================================")

        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.startswith("session_")
        ])

        total_sessions = len(session_files)

        for s_idx, file in enumerate(session_files, 1):

            session = os.path.splitext(file)[0]

            print(f"\n   [Session {s_idx}/{total_sessions}] {session}")

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

            events = df.to_dict(orient="records")

            print(f"      Total Events = {len(events)}")

            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz)

# ============================================================
# Main
# ============================================================
def main():

    global GLOBAL_V_RAW, GLOBAL_V_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--velocity_dist", type=str, default="velocity_distribution_raw.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--clip", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    dist_path = os.path.join(ROOT, args.velocity_dist)

    GLOBAL_V_RAW = load_raw_velocity_distribution(dist_path)

    GLOBAL_V_CDF = build_runtime_cdf(GLOBAL_V_RAW, args.clip)

    print("\n[Step] Generating XYPlot Base+V Images (OpenCV)")

    process_dataset(data_root, out_dir, sorted(set(args.sizes)))

    print("\n[Done] XYPlot Base+V Generation Complete.")

if __name__ == "__main__":
    main()