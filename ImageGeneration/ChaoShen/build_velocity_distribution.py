# -*- coding: utf-8 -*-
"""
Build Raw Velocity Distribution (Training-Only, Node-Level, Backward Difference)
ChaoShen Version
--------------------------------------------------------------------------------
- Scans training set only
- Each point velocity computed using previous point
- First point velocity = 0
- No percentile clipping
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
# Data Cleaning (ChaoShen format)
# ============================================================
def clean_and_rename_cols(df):

    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "X": "x",
        "Y": "y",
        "Timestamp": "time",
        "EventName": "event"
    })

    # Only keep Move events
    df = df[df["event"] == "Move"].copy()

    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Build Raw Velocity (Node-Level, Backward Difference)
# ============================================================
def build_raw_velocity_distribution(training_root):

    print("\n[Step 1] Scanning training set for raw velocity values...")

    all_velocities = []

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
            if f.endswith(".csv")
        ])

        total_sessions = len(session_files)

        for s_idx, file in enumerate(session_files, 1):

            print(f"   [Session {s_idx}/{total_sessions}] {file}")

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

            if len(df) < 2:
                continue

            xs = df["x"].values
            ys = df["y"].values
            ts = df["time"].values

            # ------------------------------------------------
            # Backward difference (node-level velocity)
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

            v = v[np.isfinite(v)]

            print(f"      Extracted velocities: {len(v)}")

            all_velocities.append(v)

    if not all_velocities:
        raise RuntimeError("No velocity values found.")

    all_velocities = np.concatenate(all_velocities)

    print("\n[Summary]")
    print(f"Total velocity samples: {len(all_velocities)}")
    print(f"Min velocity: {all_velocities.min():.6f}")
    print(f"Max velocity: {all_velocities.max():.6f}")
    print(f"Mean velocity: {all_velocities.mean():.6f}")
    print(f"Std velocity: {all_velocities.std():.6f}")

    return all_velocities

# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_root", type=str, default="Data/ChaoShen/training_files")
    parser.add_argument("--out_file", type=str, default="ChaoShen_velocity_distribution_raw.npz")

    args = parser.parse_args()

    training_root = os.path.join(ROOT, args.training_root)
    out_path = os.path.join(ROOT, args.out_file)

    velocities = build_raw_velocity_distribution(training_root)

    np.savez_compressed(out_path, velocities=velocities)

    print(f"\n[Saved] Raw velocity distribution saved to: {out_path}")
    print("[Done]")

if __name__ == "__main__":
    main()