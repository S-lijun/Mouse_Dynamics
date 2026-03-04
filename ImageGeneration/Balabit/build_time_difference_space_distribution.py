# -*- coding: utf-8 -*-
"""
Build Raw Time-Difference Distribution (Training-Only, Next-Event)
-------------------------------------------------------------------
- Scans training set only
- Computes Δt = t_{i+1} - t_i
- No chunking
- No pairwise differences
- No trajectory segmentation
- No clipping
- No percentile
- No CDF
- Pure statistical ground truth
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
# Build Distribution
# ============================================================
def build_raw_time_difference_distribution(training_root):

    print("\n[Step 1] Scanning training set for Δt values (t_{i+1} - t_i)...")

    all_dt = []

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

        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.startswith("session_")
        ])

        total_sessions = len(session_files)

        for s_idx, file in enumerate(session_files, 1):

            print(f"\n   [Session {s_idx}/{total_sessions}] {file}")

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

            if len(df) < 2:
                print("      Too few events, skipping.")
                continue

            times = df["time"].values

            # compute Δt = t_{i+1} - t_i
            dt = times[1:] - times[:-1]

            # remove invalid values
            dt = dt[dt > 0]

            print(f"      Valid Δt samples = {len(dt)}")

            if len(dt) > 0:
                all_dt.append(dt)

    if not all_dt:
        raise RuntimeError("No time-difference values found.")

    all_dt = np.concatenate(all_dt)

    print("\n[Summary]")
    print(f"Total samples: {len(all_dt)}")
    print(f"Min: {all_dt.min():.6f}")
    print(f"Max: {all_dt.max():.6f}")
    print(f"Mean: {all_dt.mean():.6f}")

    return all_dt

# ============================================================
# MAIN
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_root", type=str, default="Data/Balabit-dataset/training_files")

    parser.add_argument(
        "--out_file",
        type=str,
        default="time_difference_space_distribution_raw.npz"
    )

    args = parser.parse_args()

    training_root = os.path.join(ROOT, args.training_root)
    out_path = os.path.join(ROOT, args.out_file)

    dt_values = build_raw_time_difference_distribution(training_root)

    np.savez_compressed(out_path, time_differences=dt_values)

    print(f"\n[Saved] Raw Δt distribution saved to: {out_path}")
    print("[Done]")

if __name__ == "__main__":
    main()