# -*- coding: utf-8 -*-
"""
Analyze Global Δt Distribution (Chunk-Aware)
-------------------------------------------
- chunk size fixed (default=60)
- only intra-chunk Δt
- visualize histogram + CDF
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# ============================================================
# ROOT & DATA PATH
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_DATA_ROOT = os.path.join(
    ROOT, "Data", "Balabit-dataset", "training_files"
)


# ============================================================
# Collect Δt
# ============================================================
def collect_global_deltat(data_root, chunk_size=60):
    all_dt = []

    file_paths = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.startswith("session_"):
                file_paths.append(os.path.join(root, f))

    print(f"Total session files: {len(file_paths)}")

    for idx, file_path in enumerate(file_paths, 1):
        if idx % 200 == 0:
            print(f"[Progress] {idx}/{len(file_paths)}")

        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip() for c in df.columns]
            df = df[df["state"] == "Move"].copy()

            events = df[['x', 'y', 'client timestamp']].dropna().values

            n_chunks = len(events) // chunk_size

            for i in range(n_chunks):
                chunk = events[i*chunk_size:(i+1)*chunk_size]
                ts = chunk[:, 2]

                dt_matrix = np.abs(ts[:, None] - ts[None, :])

                # only upper triangle (avoid duplicates)
                triu = np.triu_indices(chunk_size, k=1)
                dt_values = dt_matrix[triu]

                all_dt.extend(dt_values)

        except Exception as e:
            continue

    return np.array(all_dt)


# ============================================================
# Visualization
# ============================================================
def visualize_distribution(all_dt, bins=200, percentile=95):
    print("\n===== Δt Global Statistics =====")
    print(f"Total pairs: {len(all_dt)}")
    print(f"Min: {np.min(all_dt):.6f}")
    print(f"Max: {np.max(all_dt):.6f}")
    print(f"Mean: {np.mean(all_dt):.6f}")
    print(f"Median: {np.median(all_dt):.6f}")
    print(f"Std: {np.std(all_dt):.6f}")

    perc_value = np.percentile(all_dt, percentile)
    print(f"{percentile} percentile: {perc_value:.6f}")

    # ===============================
    # 1️⃣ 全数据 Histogram（但限制显示范围）
    # ===============================
    plt.figure(figsize=(10,5))
    plt.hist(all_dt, bins=bins, range=(0, perc_value))
    plt.axvline(perc_value, linestyle="--", linewidth=2)
    plt.title(f"Global Δt Histogram (Display ≤ {percentile}%)")
    plt.xlabel("Δt")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # ===============================
    # 2️⃣ CDF（完整数据）
    # ===============================
    sorted_dt = np.sort(all_dt)
    cdf = np.linspace(0, 1, len(sorted_dt))

    plt.figure(figsize=(10,5))
    plt.plot(sorted_dt, cdf)
    plt.axvline(perc_value, linestyle="--", linewidth=2)
    plt.title(f"Global Δt CDF")
    plt.xlabel("Δt")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--bins", type=int, default=200)

    args = parser.parse_args()

    print(f"\nUsing data root: {args.data_root}")
    print(f"Chunk size: {args.chunk_size}")

    all_dt = collect_global_deltat(args.data_root, args.chunk_size)

    if len(all_dt) == 0:
        print("No Δt collected. Check path.")
    else:
        visualize_distribution(all_dt, args.bins)