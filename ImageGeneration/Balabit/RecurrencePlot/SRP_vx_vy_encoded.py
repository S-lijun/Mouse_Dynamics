# -*- coding: utf-8 -*-
"""
SRP — Directional Velocity Hybrid Encoding (Protocol-Safe)
-----------------------------------------------------------
- R Channel: Position Recurrence Matrix
- G Channel: v_x Vertical Stripes
- B Channel: v_y Vertical Stripes
- Uses RAW training vx/vy distributions
- Percentile clipping applied at runtime
- No data leakage
- Unified CLI + unified printing style
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import rankdata

# ============================================================
# ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Base Config
# ============================================================
BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200

GLOBAL_VX_RAW = None
GLOBAL_VY_RAW = None
GLOBAL_VX_CDF = None
GLOBAL_VY_CDF = None

# ============================================================
# Load RAW vx / vy Distribution
# ============================================================
def load_raw_directional_velocity_distribution(path):

    data = np.load(path)
    vx = data["vx"]
    vy = data["vy"]

    print(f"[Directional Velocity] Loaded RAW distribution from {path}")

    print("\n[vx]")
    print(f"Total samples: {len(vx)}")
    print(f"Min: {vx.min():.6f}")
    print(f"Max: {vx.max():.6f}")

    print("\n[vy]")
    print(f"Total samples: {len(vy)}")
    print(f"Min: {vy.min():.6f}")
    print(f"Max: {vy.max():.6f}")

    return vx, vy

# ============================================================
# Build Runtime CDF (Signed, Percentile Clipping)
# ============================================================
def build_runtime_cdf_signed(raw_values, clip_pct):

    print(f"\n[Directional Velocity] Building runtime CDF (clip={clip_pct}%)")

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[
        (raw_values >= lower) & (raw_values <= upper)
    ]

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    v_sorted = clipped[order]
    cdf_sorted = cdf[order]

    print(f"Runtime Min: {v_sorted.min():.6f}")
    print(f"Runtime Max: {v_sorted.max():.6f}")
    print(f"Runtime Samples: {len(v_sorted)}")

    return v_sorted, cdf_sorted

# ============================================================
# Dynamic Image Size
# ============================================================
def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Core SRP Logic
# ============================================================
def compute_hybrid_rp(seq, p_percentile=95):

    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # -------- R Channel (Distance Matrix) --------
    coords = seq[:, :2]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, p_percentile)
    r_channel = 1.0 - np.clip(dist / (eps + 1e-6), 0, 1)

    # -------- Directional Velocity --------
    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    # Normalize via global signed CDF
    vx_norm = np.interp(
        vx,
        GLOBAL_VX_CDF[0],
        GLOBAL_VX_CDF[1],
        left=0.0,
        right=1.0
    )

    vy_norm = np.interp(
        vy,
        GLOBAL_VY_CDF[0],
        GLOBAL_VY_CDF[1],
        left=0.0,
        right=1.0
    )

    stripe_x = np.tile(vx_norm[None, :], (T, 1))
    stripe_y = np.tile(vy_norm[None, :], (T, 1))

    rgb = np.stack([r_channel, stripe_x, stripe_y], axis=-1)

    return np.clip(rgb, 0.0, 1.0)

# ============================================================
# Drawing
# ============================================================
def draw_rp_image(seq, save_path, p_perc, chunk_size):

    rgb_rp = compute_hybrid_rp(seq, p_percentile=p_perc)
    img_size = get_dynamic_image_size(chunk_size)

    fig, ax = plt.subplots(
        figsize=(img_size / DPI, img_size / DPI),
        dpi=DPI
    )

    ax.imshow(rgb_rp, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ============================================================
# Cleaning
# ============================================================
def clean_and_rename_cols(df):

    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })

    df = df[df["state"] == "Move"].copy()

    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size, p_perc):

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

        draw_rp_image(chunk, save_path, p_perc, chunk_size)

        if (i + 1) % 50 == 0 or (i + 1) == n_chunks:
            print(f"         -> Chunk {i+1}/{n_chunks} done")

# ============================================================
# Dataset Processing
# ============================================================
def process_dataset(data_root, out_dir, sizes, p_perc):

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

            events = df[["x", "y", "time"]].values
            print(f"      Total Events = {len(events)}")

            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz, p_perc)

# ============================================================
# Main
# ============================================================
def main():

    global GLOBAL_VX_RAW, GLOBAL_VY_RAW
    global GLOBAL_VX_CDF, GLOBAL_VY_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--velocity_dist", type=str,
                        default="vx_vy_distribution_raw.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--v_percentile", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    dist_path = os.path.join(ROOT, args.velocity_dist)

    GLOBAL_VX_RAW, GLOBAL_VY_RAW = load_raw_directional_velocity_distribution(dist_path)

    GLOBAL_VX_CDF = build_runtime_cdf_signed(GLOBAL_VX_RAW, args.v_percentile)
    GLOBAL_VY_CDF = build_runtime_cdf_signed(GLOBAL_VY_RAW, args.v_percentile)

    print("\n[Step] Generating SRP Directional Velocity Images")
    process_dataset(data_root, out_dir, sorted(set(args.sizes)), args.p_percentile)

    print("\n[Done] SRP Directional Velocity Generation Complete.")

if __name__ == "__main__":
    main()