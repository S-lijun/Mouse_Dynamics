# -*- coding: utf-8 -*-
"""
SRP — DAT Hybrid Encoding (Distance + Acceleration + Time Difference)
----------------------------------------------------------------------
R Channel: Position Recurrence Matrix
G Channel: Acceleration Horizontal Stripes (Signed Symmetric CDF)
B Channel: Pairwise |Δt| Matrix (Magnitude CDF)

- Acceleration = signed symmetric runtime CDF
- Time Difference = magnitude runtime CDF
- Runtime percentile clipping
- No leakage
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

BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200

GLOBAL_A_CDF = None
GLOBAL_TD_CDF = None


# ============================================================
# -------- Distribution Loading --------
# ============================================================
def load_raw_distribution(path, key):

    data = np.load(path)
    arr = data[key]

    print(f"[Load] {key} distribution from {path}")
    print(f"       Samples: {len(arr)}")
    print(f"       Min: {arr.min():.6f}")
    print(f"       Max: {arr.max():.6f}")

    return arr


# ============================================================
# -------- Signed Symmetric CDF (Acceleration) --------
# ============================================================
def build_signed_cdf(raw_values, clip_pct):

    print(f"\n[Acceleration] Building signed symmetric CDF (|clip|={clip_pct}%)")

    max_abs = np.percentile(np.abs(raw_values), clip_pct)
    print(f"[Acceleration] Symmetric bound = ±{max_abs:.6f}")

    clipped = np.clip(raw_values, -max_abs, max_abs)

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)
    return clipped[order], cdf[order]


# ============================================================
# -------- Magnitude CDF (Time Difference) --------
# ============================================================
def build_magnitude_cdf(raw_values, clip_pct):

    print(f"\n[TimeDiff] Building magnitude CDF (clip={clip_pct}%)")

    upper = np.percentile(raw_values, clip_pct)
    print(f"[TimeDiff] Upper bound = {upper:.6f}")

    clipped = np.clip(raw_values, 0, upper)

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)
    return clipped[order], cdf[order]


# ============================================================
# -------- Dynamic Image Size --------
# ============================================================
def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# -------- Core DAT RP --------
# ============================================================
def compute_dat_rp(seq, p_percentile=95):

    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # -------- R Channel --------
    coords = seq[:, :2]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, p_percentile)
    r_channel = 1.0 - np.clip(dist / (eps + 1e-6), 0, 1)

    # -------- Acceleration (SIGNED) --------
    dt = np.maximum(np.diff(ts), 1e-5)

    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    v_full = np.concatenate([[v[0]], v])

    dv = np.diff(v_full)
    acc = dv / np.maximum(dt, 1e-5)
    acc_full = np.concatenate([[acc[0]], acc])

    acc_norm = np.interp(
        acc_full,   # SIGNED
        GLOBAL_A_CDF[0],
        GLOBAL_A_CDF[1],
        left=0.0,
        right=1.0
    )

    g_channel = np.tile(acc_norm[:, None], (1, T))

    # -------- Time Difference (Magnitude) --------
    dt_matrix = np.abs(ts[:, None] - ts[None, :])

    td_norm = np.interp(
        dt_matrix,
        GLOBAL_TD_CDF[0],
        GLOBAL_TD_CDF[1],
        left=0.0,
        right=1.0
    )

    b_channel = td_norm

    rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


# ============================================================
# -------- Drawing --------
# ============================================================
def draw_rp_image(seq, save_path, p_perc, chunk_size):

    rgb = compute_dat_rp(seq, p_percentile=p_perc)

    img_size = get_dynamic_image_size(chunk_size)

    fig, ax = plt.subplots(
        figsize=(img_size / DPI, img_size / DPI),
        dpi=DPI
    )

    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# -------- Cleaning --------
# ============================================================
def clean_df(df):

    df = df.rename(columns={"client timestamp": "time"})
    df = df[df["state"] == "Move"].copy()

    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)


# ============================================================
# -------- Chunking --------
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


# ============================================================
# -------- Dataset Processing --------
# ============================================================
def process_dataset(data_root, out_dir, sizes, p_perc):

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

            df = clean_df(
                pd.read_csv(os.path.join(user_dir, file))
            )

            events = df[["x", "y", "time"]].values
            print(f"   [Session] {session} | Events={len(events)}")

            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz, p_perc)


# ============================================================
# -------- Main --------
# ============================================================
def main():

    global GLOBAL_A_CDF, GLOBAL_TD_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--acc_dist", type=str, default="acceleration_distribution_raw.npz")
    parser.add_argument("--td_dist", type=str, default="time_difference_distribution_raw.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--a_percentile", type=float, default=95.0)
    parser.add_argument("--t_percentile", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    # Acceleration (SIGNED)
    raw_a = load_raw_distribution(
        os.path.join(ROOT, args.acc_dist),
        "accelerations"
    )
    GLOBAL_A_CDF = build_signed_cdf(raw_a, args.a_percentile)

    # Time Difference (Magnitude)
    raw_td = load_raw_distribution(
        os.path.join(ROOT, args.td_dist),
        "time_differences"
    )
    GLOBAL_TD_CDF = build_magnitude_cdf(raw_td, args.t_percentile)

    print("\n[Step] Generating DAT Hybrid Images")
    process_dataset(data_root, out_dir, sorted(set(args.sizes)), args.p_percentile)

    print("\n[Done] DAT Hybrid Generation Complete.")


if __name__ == "__main__":
    main()