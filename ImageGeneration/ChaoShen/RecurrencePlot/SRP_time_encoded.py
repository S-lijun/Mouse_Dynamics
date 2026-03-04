# -*- coding: utf-8 -*-
"""
SRP — Time Difference Hybrid Encoding (Protocol-Safe, Raw Distribution)
------------------------------------------------------------------------
ChaoShen FAST Version

R Channel: Position Recurrence (Distance Matrix)
G/B Channel: Pairwise |Δt| Matrix (Global Runtime CDF)

- Uses RAW training time-difference distribution
- Percentile clipping applied at runtime
- No scanning of data_root
- No data leakage
- Fully aligned style with velocity & acceleration encoding
- FAST rendering via OpenCV (INTER_LINEAR)
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
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

GLOBAL_TD_RAW = None
GLOBAL_TD_CDF = None

# ============================================================
# Load RAW Distribution
# ============================================================
def load_raw_time_distribution(path):

    data = np.load(path)
    td = data["time_differences"]

    print(f"[TimeDiff] Loaded RAW distribution from {path}")
    print(f"[TimeDiff] Total samples: {len(td)}")
    print(f"[TimeDiff] Min: {td.min():.6f}")
    print(f"[TimeDiff] Max: {td.max():.6f}")

    return td

# ============================================================
# Build Runtime CDF
# ============================================================
def build_runtime_cdf(raw_td, clip_pct):

    print(f"\n[TimeDiff] Building runtime CDF (clip={clip_pct}%)")

    td_upper = np.percentile(raw_td, clip_pct)
    td_clipped = raw_td[raw_td <= td_upper]

    ranks = rankdata(td_clipped, method="average")
    cdf = (ranks - 1) / (len(td_clipped) - 1 + 1e-8)

    order = np.argsort(td_clipped)

    td_sorted = td_clipped[order]
    cdf_sorted = cdf[order]

    print(f"[TimeDiff] Runtime Min: {td_sorted.min():.6f}")
    print(f"[TimeDiff] Runtime Max: {td_sorted.max():.6f}")
    print(f"[TimeDiff] Runtime Samples: {len(td_sorted)}")

    return td_sorted, cdf_sorted

# ============================================================
# Dynamic Image Size
# ============================================================
def get_dynamic_image_size(chunk_size):

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Core SRP Logic
# ============================================================
def compute_dt_rp(seq, p_percentile=95):

    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # -------- R Channel --------
    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, p_percentile)

    r_channel = 1.0 - np.clip(dist / (eps + 1e-6), 0, 1)

    # -------- Time Difference --------
    dt_matrix = np.abs(ts[:, None] - ts[None, :])

    dt_norm = np.interp(
        dt_matrix,
        GLOBAL_TD_CDF[0],
        GLOBAL_TD_CDF[1],
        left=0.0,
        right=1.0
    )

    g_channel = dt_norm
    b_channel = dt_norm

    rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)

    return np.clip(rgb, 0.0, 1.0)

# ============================================================
# Drawing (OpenCV FAST)
# ============================================================
def draw_rp_image(seq, save_path, p_perc, chunk_size):

    rgb_rp = compute_dt_rp(seq, p_percentile=p_perc)

    img_size = get_dynamic_image_size(chunk_size)

    img = (rgb_rp * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(
            img,
            (img_size, img_size),
            interpolation=cv2.INTER_LINEAR
        )

    # match matplotlib origin="lower"
    img = np.flipud(img)

    # OpenCV BGR fix
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

# ============================================================
# Cleaning (ChaoShen)
# ============================================================
def clean_and_rename_cols(df):

    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "X": "x",
        "Y": "y",
        "Timestamp": "time",
        "EventName": "state"
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
                pd.read_csv(
                    os.path.join(user_dir, file),
                    usecols=["X", "Y", "Timestamp", "EventName"]
                )
            )

            events = df[["x", "y", "time"]].values

            print(f"      Total Events = {len(events)}")

            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz, p_perc)

# ============================================================
# Main
# ============================================================
def main():

    global GLOBAL_TD_RAW, GLOBAL_TD_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--td_dist", type=str,
                        default="ChaoShen_time_difference_distribution_raw.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--t_percentile", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    dist_path = os.path.join(ROOT, args.td_dist)

    GLOBAL_TD_RAW = load_raw_time_distribution(dist_path)
    GLOBAL_TD_CDF = build_runtime_cdf(GLOBAL_TD_RAW, args.t_percentile)

    print("\n[Step] Generating SRP Time-Difference Images")

    process_dataset(
        data_root,
        out_dir,
        sorted(set(args.sizes)),
        args.p_percentile
    )

    print("\n[Done] SRP Time-Difference Generation Complete.")

if __name__ == "__main__":
    main()