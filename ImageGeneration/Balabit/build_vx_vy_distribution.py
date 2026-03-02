# -*- coding: utf-8 -*-
"""
Build Raw Directional Velocity Distributions (Training-Only, Node-Level)
------------------------------------------------------------------------
- Scans training set only
- Backward difference
- Computes vx and vy separately
- First point velocity = 0
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
# Build Raw vx / vy Distribution
# ============================================================
def build_raw_directional_velocity_distributions(training_root):

    print("\n[Step 1] Scanning training set for raw vx and vy values...")

    all_vx = []
    all_vy = []

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
                continue

            xs = df["x"].values
            ys = df["y"].values
            ts = df["time"].values

            # ------------------------------------------------
            # Backward difference
            # vx[i] = (x_i - x_{i-1}) / dt
            # vy[i] = (y_i - y_{i-1}) / dt
            # vx[0] = 0
            # vy[0] = 0
            # ------------------------------------------------
            dx = xs[1:] - xs[:-1]
            dy = ys[1:] - ys[:-1]
            dt = ts[1:] - ts[:-1]

            dt = np.maximum(dt, 1e-5)

            vx_segment = dx / dt
            vy_segment = dy / dt

            vx = np.zeros(len(xs))
            vy = np.zeros(len(xs))

            vx[1:] = vx_segment
            vy[1:] = vy_segment

            vx = vx[np.isfinite(vx)]
            vy = vy[np.isfinite(vy)]

            print(f"      Extracted vx samples: {len(vx)}")
            print(f"      Extracted vy samples: {len(vy)}")

            all_vx.append(vx)
            all_vy.append(vy)

    if not all_vx or not all_vy:
        raise RuntimeError("No directional velocity values found.")

    all_vx = np.concatenate(all_vx)
    all_vy = np.concatenate(all_vy)

    print("\n[Summary - vx]")
    print(f"Total vx samples: {len(all_vx)}")
    print(f"Min vx: {all_vx.min():.6f}")
    print(f"Max vx: {all_vx.max():.6f}")
    print(f"Mean vx: {all_vx.mean():.6f}")

    print("\n[Summary - vy]")
    print(f"Total vy samples: {len(all_vy)}")
    print(f"Min vy: {all_vy.min():.6f}")
    print(f"Max vy: {all_vy.max():.6f}")
    print(f"Mean vy: {all_vy.mean():.6f}")

    return all_vx, all_vy

# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_root", type=str, default="Data/Balabit-dataset/training_files")
    parser.add_argument("--out_file", type=str,
                        default="vx_vy_distribution_raw.npz")

    args = parser.parse_args()

    training_root = os.path.join(ROOT, args.training_root)
    out_path = os.path.join(ROOT, args.out_file)

    vx, vy = build_raw_directional_velocity_distributions(training_root)

    np.savez_compressed(out_path, vx=vx, vy=vy)

    print(f"\n[Saved] Raw directional velocity distribution saved to: {out_path}")
    print("[Done]")

if __name__ == "__main__":
    main()