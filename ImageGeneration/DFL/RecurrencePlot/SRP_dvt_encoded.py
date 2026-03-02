# -*- coding: utf-8 -*-
"""
SRP — DVT Hybrid Encoding (DFL Version, Protocol-Safe)
------------------------------------------------------
R Channel: Position Recurrence Matrix
G Channel: Velocity Vertical Stripes (Global Runtime CDF)
B Channel: Pairwise |Δt| Matrix (Global Runtime CDF)

- Uses RAW DFL velocity distribution
- Uses RAW DFL time-difference distribution
- Runtime percentile clipping
- No data leakage
- Fully aligned with previous DFL encoding scripts
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

GLOBAL_V_CDF = None
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


def build_runtime_cdf(raw_values, clip_pct, tag):

    print(f"\n[{tag}] Building runtime CDF (clip={clip_pct}%)")

    upper = np.percentile(raw_values, clip_pct)
    clipped = raw_values[raw_values <= upper]

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    sorted_val = clipped[order]
    sorted_cdf = cdf[order]

    print(f"[{tag}] Runtime Min: {sorted_val.min():.6f}")
    print(f"[{tag}] Runtime Max: {sorted_val.max():.6f}")
    print(f"[{tag}] Runtime Samples: {len(sorted_val)}")

    return sorted_val, sorted_cdf


# ============================================================
# -------- Dynamic Image Size --------
# ============================================================
def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# -------- Core DVT RP --------
# ============================================================
def compute_dvt_rp(seq, p_percentile=95):

    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # -------- R: Distance RP --------
    coords = seq[:, :2]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, p_percentile)
    r_channel = 1.0 - np.clip(dist / (eps + 1e-6), 0, 1)

    # -------- G: Velocity --------
    dt = np.maximum(np.diff(ts), 1e-5)
    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    v_full = np.concatenate([[v[0]], v])

    v_norm = np.interp(
        v_full,
        GLOBAL_V_CDF[0],
        GLOBAL_V_CDF[1],
        left=0.0,
        right=1.0
    )

    g_channel = np.tile(v_norm[None, :], (T, 1))

    # -------- B: Time Difference --------
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

    rgb = compute_dvt_rp(seq, p_percentile=p_perc)

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
# -------- Cleaning (DFL Robust) --------
# ============================================================
def clean_and_rename_cols(df):

    df.columns = [c.strip().lower() for c in df.columns]

    # timestamp
    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp": "time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    elif "time" not in df.columns:
        raise RuntimeError(f"Cannot find timestamp column. Columns = {df.columns}")

    # state optional
    if "state" in df.columns:
        df = df[df["state"].str.lower() == "move"].copy()
    else:
        print("      [Warning] No 'state' column found — skipping Move filtering.")

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

        if (i + 1) % 50 == 0 or (i + 1) == n_chunks:
            print(f"         -> Chunk {i+1}/{n_chunks} done")


# ============================================================
# -------- Dataset Processing --------
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

        # DFL: all CSV files
        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.lower().endswith(".csv")
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
# -------- Main --------
# ============================================================
def main():

    global GLOBAL_V_CDF, GLOBAL_TD_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str,
                        default="Data/DFL-dataset_raw/training_files")
    parser.add_argument("--velocity_dist", type=str,
                        default="DFL_velocity_distribution_raw.npz")
    parser.add_argument("--td_dist", type=str,
                        default="DFL_time_difference_distribution_raw.npz")
    parser.add_argument("--out_dir", type=str,
                        default="Images/DFL/SRP_dvt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--v_percentile", type=float, default=95.0)
    parser.add_argument("--t_percentile", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    # ---- Velocity ----
    raw_v = load_raw_distribution(
        os.path.join(ROOT, args.velocity_dist),
        "velocities"
    )
    GLOBAL_V_CDF = build_runtime_cdf(raw_v, args.v_percentile, "Velocity")

    # ---- Time Difference ----
    raw_td = load_raw_distribution(
        os.path.join(ROOT, args.td_dist),
        "time_differences"
    )
    GLOBAL_TD_CDF = build_runtime_cdf(raw_td, args.t_percentile, "TimeDiff")

    print("\n[Step] Generating DFL DVT Hybrid Images")
    process_dataset(data_root, out_dir, sorted(set(args.sizes)), args.p_percentile)

    print("\n[Done] DFL DVT Hybrid Generation Complete.")


if __name__ == "__main__":
    main()