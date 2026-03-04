# -*- coding: utf-8 -*-
"""
XYPlot — Vx + Vy + DeltaT Encoding (OpenCV)
-------------------------------------------
R = vx
G = vy
B = DeltaT

OpenCV uses BGR ordering:
(B, G, R) = (DeltaT, vy, vx)
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

GLOBAL_VX_RAW = None
GLOBAL_VY_RAW = None
GLOBAL_VX_CDF = None
GLOBAL_VY_CDF = None

GLOBAL_DT_RAW = None
GLOBAL_DT_CDF = None

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
# Cleaning
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
# Load Distributions
# ============================================================
def load_raw_directional_velocity_distribution(path):

    data = np.load(path)

    vx = data["vx"]
    vy = data["vy"]

    print(f"[Directional Velocity] Loaded RAW distribution from {path}")

    print(f"[vx] samples: {len(vx)}")
    print(f"Min: {vx.min():.6f}")
    print(f"Max: {vx.max():.6f}")

    print(f"\n[vy] samples: {len(vy)}")
    print(f"Min: {vy.min():.6f}")
    print(f"Max: {vy.max():.6f}")

    return vx, vy


def load_raw_dt_distribution(path):

    data = np.load(path)

    dt = data["time_differences"]

    print(f"[DeltaT] Loaded RAW distribution from {path}")
    print(f"Samples: {len(dt)}")

    return dt


# ============================================================
# Runtime CDF
# ============================================================
def build_runtime_cdf_signed(raw_values, clip_pct):

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[
        (raw_values >= lower) & (raw_values <= upper)
    ]

    ranks = rankdata(clipped, method="average")

    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    values_sorted = clipped[order]
    cdf_sorted = cdf[order]

    return values_sorted, cdf_sorted


def build_runtime_cdf(raw_values, clip_pct):

    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[raw_values <= upper]

    ranks = rankdata(clipped, method="average")

    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    values_sorted = clipped[order]
    cdf_sorted = cdf[order]

    return values_sorted, cdf_sorted


# ============================================================
# Drawing
# ============================================================
def draw_mouse_chunk(chunk, save_path, chunk_size):

    if len(chunk) < 2:
        return

    IMG_SIZE = get_img_size(chunk_size)

    linewidth, markersize = get_stroke_params(chunk_size)

    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    dx = np.diff(xs)
    dy = np.diff(ys)

    dt = np.maximum(np.diff(ts), 1e-5)

    vx = dx / dt
    vy = dy / dt

    # --------------------------------------------------------
    # Normalize vx
    # --------------------------------------------------------
    vx_norm = np.interp(
        vx,
        GLOBAL_VX_CDF[0],
        GLOBAL_VX_CDF[1],
        left=0.0,
        right=1.0
    )

    vx_val = (vx_norm * 255).astype(np.uint8)

    # --------------------------------------------------------
    # Normalize vy
    # --------------------------------------------------------
    vy_norm = np.interp(
        vy,
        GLOBAL_VY_CDF[0],
        GLOBAL_VY_CDF[1],
        left=0.0,
        right=1.0
    )

    vy_val = (vy_norm * 255).astype(np.uint8)

    # --------------------------------------------------------
    # Normalize Δt
    # --------------------------------------------------------
    dt_norm = np.interp(
        dt,
        GLOBAL_DT_CDF[0],
        GLOBAL_DT_CDF[1],
        left=0.0,
        right=1.0
    )

    dt_val = (dt_norm * 255).astype(np.uint8)

    # --------------------------------------------------------
    # Spatial scaling (完全照抄你的)
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
    # Draw
    # --------------------------------------------------------
    for i in range(len(xs) - 1):

        x1 = int(_scaled(xs[i], min_x, scale, offset_x))
        y1 = int(_scaled(ys[i], min_y, scale, offset_y))

        x2 = int(_scaled(xs[i+1], min_x, scale, offset_x))
        y2 = int(_scaled(ys[i+1], min_y, scale, offset_y))

        vx_c = int(vx_val[i])
        vy_c = int(vy_val[i])
        t_c = int(dt_val[i])

        # OpenCV BGR
        color = (t_c, vy_c, vx_c)

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
# Chunking / Dataset
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


def process_dataset(data_root, out_dir, sizes):

    users = sorted(os.listdir(data_root))

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print(f"\n[User] {user}")

        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.startswith("session_")
        ])

        for file in session_files:

            session = os.path.splitext(file)[0]

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

            events = df.to_dict(orient="records")

            print(f"   {session} events = {len(events)}")

            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz)


# ============================================================
# Main
# ============================================================
def main():

    global GLOBAL_VX_RAW, GLOBAL_VY_RAW
    global GLOBAL_VX_CDF, GLOBAL_VY_CDF
    global GLOBAL_DT_RAW, GLOBAL_DT_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--velocity_dist", type=str, default="vx_vy_distribution_raw.npz")
    parser.add_argument("--dt_dist", type=str, default="time_difference_space_distribution_raw.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--v_clip", type=float, default=95.0)
    parser.add_argument("--t_clip", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    v_path = os.path.join(ROOT, args.velocity_dist)
    t_path = os.path.join(ROOT, args.dt_dist)

    GLOBAL_VX_RAW, GLOBAL_VY_RAW = load_raw_directional_velocity_distribution(v_path)
    GLOBAL_DT_RAW = load_raw_dt_distribution(t_path)

    GLOBAL_VX_CDF = build_runtime_cdf_signed(GLOBAL_VX_RAW, args.v_clip)
    GLOBAL_VY_CDF = build_runtime_cdf_signed(GLOBAL_VY_RAW, args.v_clip)

    GLOBAL_DT_CDF = build_runtime_cdf(GLOBAL_DT_RAW, args.t_clip)

    print("\n[Step] Generating XYPlot vx vy T Images")

    process_dataset(data_root, out_dir, sorted(set(args.sizes)))

    print("\n[Done] Generation Complete.")


if __name__ == "__main__":
    main()