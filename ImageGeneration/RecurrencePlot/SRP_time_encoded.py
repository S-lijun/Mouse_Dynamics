# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Symmetrical Δt Encoding
---------------------------------------------------------
- R Channel: Distance Matrix (p_percentile threshold)
- G/B Channels: Time Difference Matrix (dt_percentile threshold)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

# ============================================================
# Paths Setup
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol2/genuine")
#DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200 


# ============================================================
# Global Δt Scaler (Chunk-aware)
# ============================================================
class GlobalDeltaTScaler:
    def __init__(self, data_dir, chunk_size=60, dt_percentile=95):
        self.dt_max = 0
        self.chunk_size = chunk_size
        self.lookup_func = self._build_scaler(data_dir, dt_percentile)

    def _build_scaler(self, data_dir, dt_percentile):
        print(f"--- Step 1: Scanning global Δt distribution (chunk-aware) ---")

        all_dt = []

        file_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.startswith("session_"):
                    file_paths.append(os.path.join(root, file))

        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths, 1):
            if idx % 500 == 0:
                print(f"  [Scan Progress] {idx}/{total_files} files scanned...")

            try:
                df = pd.read_csv(file_path)
                df.columns = [c.strip() for c in df.columns]
                df = df[df["state"] == "Move"].copy()

                events = df[['x', 'y', 'client timestamp']].dropna().values

                n_chunks = len(events) // self.chunk_size
                for i in range(n_chunks):
                    chunk = events[i*self.chunk_size : (i+1)*self.chunk_size]
                    ts = chunk[:, 2]

                    # Compute full pairwise Δt matrix
                    dt_matrix = np.abs(ts[:, None] - ts[None, :])

                    # Only take upper triangle to avoid duplication
                    triu_indices = np.triu_indices(self.chunk_size, k=1)
                    dt_values = dt_matrix[triu_indices]

                    all_dt.extend(dt_values)

            except:
                continue

        all_dt = np.array(all_dt)

        self.dt_max = np.percentile(all_dt, dt_percentile)
        print(f"--- Scan complete. Δt_max ({dt_percentile}%): {self.dt_max:.4f} ---")

        sorted_dt = np.sort(np.clip(all_dt, 0, self.dt_max))
        y = np.linspace(0, 1, len(sorted_dt))

        return interp1d(sorted_dt, y, bounds_error=False, fill_value=(0, 1))

    def transform(self, dt_matrix):
        return self.lookup_func(np.clip(dt_matrix, 0, self.dt_max))


# ============================================================
# Core RP Logic
# ============================================================
def compute_dt_rp(seq, dt_scaler, p_percentile=100):
    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # ---------------- R Channel: Spatial Distance ----------------
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    eps_p = np.percentile(dist_p, p_percentile)

    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # ---------------- G/B Channels: Δt Matrix ----------------
    dt_matrix = np.abs(ts[:, None] - ts[None, :])
    dt_norm = dt_scaler.transform(dt_matrix)

    # Symmetrical: G = B = dt_norm
    g_channel = dt_norm
    b_channel = dt_norm

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)


# ============================================================
# Draw Image
# ============================================================
def draw_rp_image(seq, save_path, dt_scaler, p_perc, chunk_size):
    rgb_rp = compute_dt_rp(seq, dt_scaler, p_percentile=p_perc)

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    img_size_px = int(round(BASE_IMG_SIZE * scale))

    fig, ax = plt.subplots(figsize=(img_size_px/DPI, img_size_px/DPI), dpi=DPI)
    ax.imshow(rgb_rp, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# Dataset Processing
# ============================================================
def process_dataset(data_dir, out_dir, dt_scaler, sizes, p_perc):
    users = sorted([u for u in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, u))])
    total_users = len(users)

    print(f"\n[Step 2] Generating SRP Δt Images (P_perc={p_perc})...")

    for u_idx, user in enumerate(users, 1):
        user_dir = os.path.join(data_dir, user)
        sessions = []

        for root, _, files in os.walk(user_dir):
            for f in files:
                if f.startswith("session_"):
                    sessions.append(os.path.join(root, f))

        sessions.sort()

        print(f"\n{'='*20} User [{u_idx}/{total_users}]: {user} {'='*20}")

        for file_path in sessions:
            file_name = os.path.basename(file_path)
            print(f"[Process] {user}/{file_name}")

            try:
                df = pd.read_csv(file_path)
                df.columns = [c.strip() for c in df.columns]
                df = df[df["state"] == "Move"].copy()

                events = df[['x', 'y', 'client timestamp']].dropna().values

                for sz in sizes:
                    n_chunks = len(events) // sz
                    for i in range(n_chunks):
                        chunk = events[i*sz : (i+1)*sz]
                        rel_path = os.path.relpath(os.path.dirname(file_path), data_dir)
                        save_path = os.path.join(out_dir, f"event{sz}", rel_path, f"{file_name}-{i}.png")
                        draw_rp_image(chunk, save_path, dt_scaler, p_perc, sz)

            except Exception as e:
                print(f"    !!! Error processing {file_name}: {e}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_time_protocol2/genuine")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--dt_percentile", type=float, default=95)

    args = parser.parse_args()

    dt_scaler = GlobalDeltaTScaler(
        args.data_root,
        chunk_size=args.sizes[0],
        dt_percentile=args.dt_percentile
    )

    process_dataset(
        args.data_root,
        os.path.join(ROOT, args.out_dir),
        dt_scaler,
        args.sizes,
        args.p_percentile
    )