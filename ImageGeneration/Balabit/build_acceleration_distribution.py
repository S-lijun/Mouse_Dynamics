# -*- coding: utf-8 -*-
"""
Build Raw Acceleration Distribution (Training-Only, Node-Level)
----------------------------------------------------------------
- Scans training set only
- Velocity computed using backward difference
- Acceleration computed as difference of node-level velocity
- First velocity = 0
- First acceleration = 0
- No clipping
- No truncation
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
# Data Cleaning
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
# Build Raw Acceleration Distribution
# ============================================================
def build_raw_acceleration_distribution(training_root):

    print("\n[Step 1] Scanning training set for raw acceleration values...")

    all_accelerations = []

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

            if len(df) < 3:
                continue

            xs = df["x"].values
            ys = df["y"].values
            ts = df["time"].values

            # ------------------------------------------------
            # Step 1: Node-level velocity (backward difference)
            # v[i] = distance(i, i-1) / dt(i, i-1)
            # v[0] = 0
            # ------------------------------------------------
            dx = xs[1:] - xs[:-1]
            dy = ys[1:] - ys[:-1]
            dt = ts[1:] - ts[:-1]
            dt = np.maximum(dt, 1e-5)

            v_segment = np.sqrt(dx**2 + dy**2) / dt

            v = np.zeros(len(xs))
            v[1:] = v_segment

            # ------------------------------------------------
            # Step 2: Node-level acceleration
            # a[i] = v[i] - v[i-1]
            # a[0] = 0
            # ------------------------------------------------
            a = np.zeros(len(xs))
            a[1:] = v[1:] - v[:-1]

            a = a[np.isfinite(a)]

            print(f"      Extracted accelerations (node-level): {len(a)}")

            all_accelerations.append(a)

    if not all_accelerations:
        raise RuntimeError("No acceleration values found.")

    all_accelerations = np.concatenate(all_accelerations)

    print("\n[Summary]")
    print(f"Total acceleration samples: {len(all_accelerations)}")
    print(f"Min acceleration: {all_accelerations.min():.6f}")
    print(f"Max acceleration: {all_accelerations.max():.6f}")
    print(f"Mean acceleration: {all_accelerations.mean():.6f}")

    return all_accelerations

# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_root", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="acceleration_distribution_raw.npz")

    args = parser.parse_args()

    training_root = os.path.join(ROOT, args.training_root)
    out_path = os.path.join(ROOT, args.out_file)

    accelerations = build_raw_acceleration_distribution(training_root)

    np.savez_compressed(out_path, accelerations=accelerations)

    print(f"\n[Saved] Raw acceleration distribution saved to: {out_path}")
    print("[Done]")

if __name__ == "__main__":
    main()