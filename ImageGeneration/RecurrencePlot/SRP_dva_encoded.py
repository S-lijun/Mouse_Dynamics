# -*- coding: utf-8 -*-
"""
SRP — Position + Velocity + Acceleration Hybrid
---------------------------------------------------------
R Channel: Position Recurrence (Distance Matrix)
G Channel: Velocity Vertical Stripes (Global CDF)
B Channel: Acceleration Horizontal Stripes (Global CDF)
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

BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200


# ============================================================
# GLOBAL SCALER (Velocity + Acceleration)
# ============================================================
class GlobalDynamicScaler:
    def __init__(self, data_dir, v_percentile=95, a_percentile=95):
        self.v_max = 0
        self.a_max = 0
        self.v_lookup = None
        self.a_lookup = None
        self._build_scalers(data_dir, v_percentile, a_percentile)

    def _build_scalers(self, data_dir, v_percentile, a_percentile):
        print("\n--- Scanning Global Velocity & Acceleration ---")

        all_v = []
        all_a = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if not file.startswith("session_"):
                    continue

                try:
                    df = pd.read_csv(os.path.join(root, file))
                    df.columns = [c.strip() for c in df.columns]
                    df = df[df["state"] == "Move"]

                    if len(df) < 3:
                        continue

                    xs = df["x"].values
                    ys = df["y"].values
                    ts = df["client timestamp"].values

                    dt = np.diff(ts) + 1e-6

                    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
                    v_full = np.concatenate([[v[0]], v])

                    acc = np.diff(v_full) / dt
                    acc_full = np.concatenate([[acc[0]], acc])

                    all_v.extend(v_full)
                    all_a.extend(np.abs(acc_full))

                except:
                    continue

        all_v = np.array(all_v)
        all_a = np.array(all_a)

        self.v_max = np.percentile(all_v, v_percentile)
        self.a_max = np.percentile(all_a, a_percentile)

        print(f"Velocity V_max ({v_percentile}%): {self.v_max:.4f}")
        print(f"Acceleration A_max ({a_percentile}%): {self.a_max:.4f}")

        # Velocity CDF
        sorted_v = np.sort(np.clip(all_v, 0, self.v_max))
        y_v = np.linspace(0, 1, len(sorted_v))
        self.v_lookup = interp1d(sorted_v, y_v, bounds_error=False, fill_value=(0, 1))

        # Acceleration CDF
        sorted_a = np.sort(np.clip(all_a, 0, self.a_max))
        y_a = np.linspace(0, 1, len(sorted_a))
        self.a_lookup = interp1d(sorted_a, y_a, bounds_error=False, fill_value=(0, 1))

    def transform_velocity(self, v_array):
        return self.v_lookup(np.clip(v_array, 0, self.v_max))

    def transform_acceleration(self, a_array):
        return self.a_lookup(np.clip(np.abs(a_array), 0, self.a_max))


# ============================================================
# CORE RP
# ============================================================
def compute_rp(seq, scaler, p_percentile=95):

    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # ---------- R Channel ----------
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    eps_p = np.percentile(dist_p, p_percentile)
    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # ---------- Velocity ----------
    dt = np.diff(ts) + 1e-6
    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    v_full = np.concatenate([[v[0]], v])  # 长度 T

    v_norm = scaler.transform_velocity(v_full)
    g_channel = np.tile(v_norm[None, :], (T, 1))  # vertical stripes

    # ---------- Acceleration ----------
    acc = np.diff(v_full) / dt
    acc_full = np.concatenate([[acc[0]], acc])  # 长度 T

    a_norm = scaler.transform_acceleration(acc_full)
    b_channel = np.tile(a_norm[:, None], (1, T))  # horizontal stripes

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)


# ============================================================
# DRAW
# ============================================================
def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))


def draw_rp_image(seq, save_path, scaler, p_perc, chunk_size):

    rgb = compute_rp(seq, scaler, p_perc)

    img_size_px = get_dynamic_image_size(chunk_size)

    fig, ax = plt.subplots(figsize=(img_size_px/DPI, img_size_px/DPI), dpi=DPI)
    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# DATA
# ============================================================
def clean_df(df):
    df.columns = [c.strip() for c in df.columns]
    df = df[df["state"] == "Move"].copy()

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["client timestamp"] = pd.to_numeric(df["client timestamp"], errors="coerce")

    return df.dropna(subset=["x", "y", "client timestamp"])


def process_dataset(data_dir, out_dir, scaler, sizes, p_perc):

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

                    draw_rp_image(chunk, save_path, scaler, p_perc, sz)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_dva_protocol1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--v_percentile", type=float, default=95)
    parser.add_argument("--a_percentile", type=float, default=95)

    args = parser.parse_args()

    scaler = GlobalDynamicScaler(
        args.data_root,
        args.v_percentile,
        args.a_percentile
    )

    out_path = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.data_root,
        out_path,
        scaler,
        args.sizes,
        args.p_percentile
    )