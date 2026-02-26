# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Position + Velocity + Δt Hybrid
---------------------------------------------------------
R Channel: Position Recurrence (Distance Matrix)
G Channel: Velocity Vertical Stripes (Global CDF)
B Channel: Δt Matrix (Chunk-aware Global CDF)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d


# ============================================================
# ROOT & PATH
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol2/genuine")

BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200


# ============================================================
# GLOBAL VELOCITY SCALER
# ============================================================
class GlobalVelocityScaler:
    def __init__(self, data_dir, v_percentile=95):
        self.v_max = 0
        self.lookup_func = self._build_scaler(data_dir, v_percentile)

    def _build_scaler(self, data_dir, v_percentile):
        print("\n--- Scanning Global Velocity Distribution ---")
        all_v = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if not file.startswith("session_"):
                    continue

                try:
                    df = pd.read_csv(os.path.join(root, file))
                    df.columns = [c.strip() for c in df.columns]
                    df = df[df["state"] == "Move"]

                    if len(df) < 2:
                        continue

                    dx = df["x"].diff()
                    dy = df["y"].diff()
                    dt = df["client timestamp"].diff() + 1e-6
                    v = np.sqrt(dx**2 + dy**2) / dt

                    all_v.extend(v.dropna().values)

                except:
                    continue

        all_v = np.array(all_v)
        self.v_max = np.percentile(all_v, v_percentile)

        print(f"Velocity V_max ({v_percentile}%): {self.v_max:.4f}")

        sorted_v = np.sort(np.clip(all_v, 0, self.v_max))
        y = np.linspace(0, 1, len(sorted_v))

        return interp1d(sorted_v, y, bounds_error=False, fill_value=(0, 1))

    def transform(self, v_array):
        return self.lookup_func(np.clip(v_array, 0, self.v_max))


# ============================================================
# GLOBAL Δt SCALER (Chunk-aware)
# ============================================================
class GlobalDeltaTScaler:
    def __init__(self, data_dir, chunk_size=60, dt_percentile=95):
        self.dt_max = 0
        self.chunk_size = chunk_size
        self.lookup_func = self._build_scaler(data_dir, dt_percentile)

    def _build_scaler(self, data_dir, dt_percentile):
        print("\n--- Scanning Global Δt Distribution (Chunk-aware) ---")

        all_dt = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if not file.startswith("session_"):
                    continue

                try:
                    df = pd.read_csv(os.path.join(root, file))
                    df.columns = [c.strip() for c in df.columns]
                    df = df[df["state"] == "Move"]

                    events = df[['x', 'y', 'client timestamp']].dropna().values
                    n_chunks = len(events) // self.chunk_size

                    for i in range(n_chunks):
                        chunk = events[i*self.chunk_size:(i+1)*self.chunk_size]
                        ts = chunk[:, 2]

                        dt_matrix = np.abs(ts[:, None] - ts[None, :])
                        triu = np.triu_indices(self.chunk_size, k=1)
                        dt_values = dt_matrix[triu]

                        all_dt.extend(dt_values)

                except:
                    continue

        all_dt = np.array(all_dt)
        self.dt_max = np.percentile(all_dt, dt_percentile)

        print(f"Δt_max ({dt_percentile}%): {self.dt_max:.4f}")

        sorted_dt = np.sort(np.clip(all_dt, 0, self.dt_max))
        y = np.linspace(0, 1, len(sorted_dt))

        return interp1d(sorted_dt, y, bounds_error=False, fill_value=(0, 1))

    def transform(self, dt_matrix):
        return self.lookup_func(np.clip(dt_matrix, 0, self.dt_max))


# ============================================================
# CORE HYBRID RP
# ============================================================
def compute_hybrid_rp(seq, v_scaler, dt_scaler, p_percentile=95):

    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # ---------------- R Channel (Position) ----------------
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    eps_p = np.percentile(dist_p, p_percentile)

    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # ---------------- G Channel (Velocity Vertical Stripes) ----------------
    dt_local = np.diff(ts) + 1e-6
    v_scalar = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt_local
    v_scalar = np.concatenate([[v_scalar[0]], v_scalar])

    v_norm = v_scaler.transform(v_scalar)

    g_channel = np.tile(v_norm[None, :], (T, 1))  # vertical stripes

    # ---------------- B Channel (Δt Matrix) ----------------
    dt_matrix = np.abs(ts[:, None] - ts[None, :])
    b_channel = dt_scaler.transform(dt_matrix)

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)


# ============================================================
# DRAW IMAGE
# ============================================================
def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))


def draw_rp_image(seq, save_path, v_scaler, dt_scaler, p_perc, chunk_size):

    rgb = compute_hybrid_rp(seq, v_scaler, dt_scaler, p_perc)

    img_size_px = get_dynamic_image_size(chunk_size)

    fig, ax = plt.subplots(figsize=(img_size_px/DPI, img_size_px/DPI), dpi=DPI)
    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# DATA PROCESSING
# ============================================================
def clean_df(df):
    df.columns = [c.strip() for c in df.columns]
    df = df[df["state"] == "Move"].copy()

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["client timestamp"] = pd.to_numeric(df["client timestamp"], errors="coerce")

    return df.dropna(subset=["x", "y", "client timestamp"])


def process_dataset(data_dir, out_dir, v_scaler, dt_scaler, sizes, p_perc):

    for user in sorted(os.listdir(data_dir)):

        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue

        for file in sorted(os.listdir(user_dir)):

            if not file.startswith("session_"):
                continue

            print(f"[Process] {user}/{file}")

            df = clean_df(pd.read_csv(os.path.join(user_dir, file)))
            events = df[['x', 'y', 'client timestamp']].values

            for sz in sizes:
                n_chunks = len(events) // sz

                for i in range(n_chunks):
                    chunk = events[i*sz:(i+1)*sz]

                    save_path = os.path.join(
                        out_dir,
                        f"event{sz}",
                        user,
                        f"{file}-{i}.png"
                    )

                    draw_rp_image(chunk, save_path, v_scaler, dt_scaler, p_perc, sz)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_dvt_protocol2/genuine")
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--v_percentile", type=float, default=95)
    parser.add_argument("--dt_percentile", type=float, default=95)

    args = parser.parse_args()

    # Build global scalers
    v_scaler = GlobalVelocityScaler(args.data_root, args.v_percentile)
    dt_scaler = GlobalDeltaTScaler(args.data_root, args.sizes[0], args.dt_percentile)

    out_path = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.data_root,
        out_path,
        v_scaler,
        dt_scaler,
        args.sizes,
        args.p_percentile
    )