# -*- coding: utf-8 -*-
"""
Plot Protocol2 Δt Distribution
--------------------------------
- Compare Genuine vs Imposter
- Chunk size = 60
- Upper triangle only
- Two parallel histograms
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

PROTO2_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol2")
GENUINE_ROOT = os.path.join(PROTO2_ROOT, "genuine")
IMPOSTER_ROOT = os.path.join(PROTO2_ROOT, "imposter")

CHUNK_SIZE = 60


# ============================================================
# Collect Global Δt
# ============================================================

def collect_global_dt(data_dir):

    all_dt = []

    file_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.startswith("session_"):
                file_paths.append(os.path.join(root, f))

    total = len(file_paths)
    print(f"[INFO] Found {total} session files in {data_dir}")

    for idx, file_path in enumerate(file_paths, 1):

        if idx % 200 == 0 or idx == total:
            print(f"[Progress] {idx}/{total}")

        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip() for c in df.columns]
            df = df[df["state"] == "Move"].copy()

            events = df[['x', 'y', 'client timestamp']].dropna().values
            n_chunks = len(events) // CHUNK_SIZE

            for i in range(n_chunks):

                chunk = events[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
                ts = chunk[:, 2]

                dt_matrix = np.abs(ts[:, None] - ts[None, :])
                triu_indices = np.triu_indices(CHUNK_SIZE, k=1)
                dt_values = dt_matrix[triu_indices]

                all_dt.extend(dt_values)

        except:
            continue

    return np.array(all_dt)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\nCollecting Genuine Δt...")
    genuine_dt = collect_global_dt(GENUINE_ROOT)

    print("\nCollecting Imposter Δt...")
    imposter_dt = collect_global_dt(IMPOSTER_ROOT)

    # 为了可视化稳定，我们 clip 到 combined 95%
    combined = np.concatenate([genuine_dt, imposter_dt])
    clip_val = np.percentile(combined, 95)

    print(f"\nCombined 95% Δt: {clip_val:.4f} ms")

    genuine_clip = genuine_dt[genuine_dt <= clip_val]
    imposter_clip = imposter_dt[imposter_dt <= clip_val]

    # ============================================================
    # Plot
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].hist(genuine_clip, bins=200, density=True)
    axes[0].set_xlim(0, clip_val)
    axes[0].set_title("Genuine")
    axes[0].set_xlabel("Δt (ms)")
    axes[0].set_ylabel("Density")

    axes[1].hist(imposter_clip, bins=200, density=True)
    axes[1].set_xlim(0, clip_val)
    axes[1].set_title("Imposter")
    axes[1].set_xlabel("Δt (ms)")

    fig.suptitle("Protocol2 Global Δt Distribution (Chunk=60)")
    plt.tight_layout()

    plt.savefig("protocol2_dt_genuine_vs_imposter.png", dpi=300)
    plt.show()

    print("\nSaved as: protocol2_dt_genuine_vs_imposter.png")