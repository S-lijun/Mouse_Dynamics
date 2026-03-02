# -*- coding: utf-8 -*-
"""
Build Raw Time-Difference Distribution (Training-Only, Chunk-Based)
--------------------------------------------------------------------
- Uses fixed chunk size (default 60)
- Computes pairwise |t_i - t_j| inside each chunk
- Stores ONLY upper triangle (unique pairs)
- No clipping
- No percentile
- No CDF
- No sorting
"""

import os
import argparse
import pandas as pd
import numpy as np

# ============================================================
# ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Cleaning
# ============================================================
def clean_and_rename_cols(df):

    df.columns = [c.strip().lower() for c in df.columns]

    # ---- timestamp detection ----
    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp": "time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    elif "time" not in df.columns:
        raise RuntimeError(f"Cannot find timestamp column. Columns = {df.columns}")

    # ---- state filtering (if exists) ----
    if "state" in df.columns:
        df = df[df["state"].str.lower() == "move"].copy()
    else:
        print("      [Warning] No 'state' column found — skipping Move filtering.")

    # ---- ensure numeric ----
    for col in ["x", "y", "time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Build Distribution (Upper Triangle Only)
# ============================================================
def build_raw_time_difference_distribution(training_root, chunk_size):

    print("\n[Step 1] Scanning training set for time-difference values...")
    print(f"[ChunkSize] = {chunk_size}")
    print(f"[Pairs per chunk] = {chunk_size*(chunk_size-1)//2}")

    all_td = []

    users = sorted(os.listdir(training_root))
    total_users = len(users)

    print(f"[Dataset] Total Users = {total_users}")

    for u_idx, user in enumerate(users, 1):

        user_dir = os.path.join(training_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n====================================================")
        print(f"[User {u_idx}/{total_users}] Processing: {user}")
        print("====================================================")

        # ---- DFL: scan all CSV files ----
        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.lower().endswith(".csv")
        ])

        total_sessions = len(session_files)

        for s_idx, file in enumerate(session_files, 1):

            print(f"\n   [Session {s_idx}/{total_sessions}] {file}")

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

            if len(df) < chunk_size:
                continue

            times = df["time"].values

            n_chunks = len(times) // chunk_size
            print(f"      Total Chunks = {n_chunks}")

            for i in range(n_chunks):

                chunk_t = times[i*chunk_size:(i+1)*chunk_size]

                # pairwise |t_i - t_j|
                diff = np.abs(chunk_t[:, None] - chunk_t[None, :])

                # Only upper triangle (exclude diagonal)
                upper = diff[np.triu_indices(chunk_size, k=1)]

                all_td.append(upper)

    if not all_td:
        raise RuntimeError("No time-difference values found.")

    # cast to float32 to avoid 300MB+ explosion
    all_td = np.concatenate(all_td).astype(np.float32)

    print("\n[Summary]")
    print(f"Total samples: {len(all_td)}")
    print(f"Min: {all_td.min():.6f}")
    print(f"Max: {all_td.max():.6f}")
    print(f"Mean: {all_td.mean():.6f}")

    return all_td

# ============================================================
# MAIN
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_root", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--out_file", type=str,
                        default="DFL_time_difference_distribution_raw.npz")

    args = parser.parse_args()

    training_root = os.path.join(ROOT, args.training_root)
    out_path = os.path.join(ROOT, args.out_file)

    td_values = build_raw_time_difference_distribution(
        training_root,
        args.chunk_size
    )

    np.savez_compressed(out_path, time_differences=td_values)

    print(f"\n[Saved] Raw time-difference distribution saved to: {out_path}")
    print("[Done]")

if __name__ == "__main__":
    main()