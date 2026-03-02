# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Symmetrical Δt Encoding
---------------------------------------------------------
- R Channel: Distance Matrix (p_percentile threshold)
- G/B Channels: Time Difference Matrix (dt_percentile threshold)
"""
'''
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
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol2/imposter")
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
    parser.add_argument("--out_dir", type=str, default="Images/SRP_time_protocol2/imposter")
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
'''

# -*- coding: utf-8 -*-
"""
SRP — Time Difference Hybrid Encoding (Protocol-Safe, Raw Distribution)
------------------------------------------------------------------------
R Channel: Position Recurrence (Distance Matrix)
G/B Channel: Pairwise |Δt| Matrix (Global Runtime CDF)

- Uses RAW training time-difference distribution
- Percentile clipping applied at runtime
- No scanning of data_root
- No data leakage
- Fully aligned style with velocity & acceleration encoding
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import rankdata

# ============================================================
# ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Base Config
# ============================================================
BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200

GLOBAL_TD_RAW = None
GLOBAL_TD_CDF = None

# ============================================================
# Load RAW Distribution
# ============================================================
def load_raw_time_distribution(path):

    data = np.load(path)
    td = data["time_differences"]

    print(f"[TimeDiff] Loaded RAW distribution from {path}")
    print(f"[TimeDiff] Total samples: {len(td)}")
    print(f"[TimeDiff] Min: {td.min():.6f}")
    print(f"[TimeDiff] Max: {td.max():.6f}")

    return td

# ============================================================
# Build Runtime CDF
# ============================================================
def build_runtime_cdf(raw_td, clip_pct):

    print(f"\n[TimeDiff] Building runtime CDF (clip={clip_pct}%)")

    td_upper = np.percentile(raw_td, clip_pct)
    td_clipped = raw_td[raw_td <= td_upper]

    ranks = rankdata(td_clipped, method="average")
    cdf = (ranks - 1) / (len(td_clipped) - 1 + 1e-8)

    order = np.argsort(td_clipped)

    td_sorted = td_clipped[order]
    cdf_sorted = cdf[order]

    print(f"[TimeDiff] Runtime Min: {td_sorted.min():.6f}")
    print(f"[TimeDiff] Runtime Max: {td_sorted.max():.6f}")
    print(f"[TimeDiff] Runtime Samples: {len(td_sorted)}")

    return td_sorted, cdf_sorted

# ============================================================
# Dynamic Image Size
# ============================================================
def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Core SRP Logic
# ============================================================
def compute_dt_rp(seq, p_percentile=95):

    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]

    # -------- R Channel --------
    coords = seq[:, :2]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, p_percentile)
    r_channel = 1.0 - np.clip(dist / (eps + 1e-6), 0, 1)

    # -------- Time Difference --------
    dt_matrix = np.abs(ts[:, None] - ts[None, :])

    dt_norm = np.interp(
        dt_matrix,
        GLOBAL_TD_CDF[0],
        GLOBAL_TD_CDF[1],
        left=0.0,
        right=1.0
    )

    g_channel = dt_norm
    b_channel = dt_norm

    rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

# ============================================================
# Drawing
# ============================================================
def draw_rp_image(seq, save_path, p_perc, chunk_size):

    rgb_rp = compute_dt_rp(seq, p_percentile=p_perc)

    img_size = get_dynamic_image_size(chunk_size)

    fig, ax = plt.subplots(
        figsize=(img_size / DPI, img_size / DPI),
        dpi=DPI
    )

    ax.imshow(rgb_rp, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ============================================================
# Cleaning
# ============================================================
def clean_and_rename_cols(df):

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
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size, p_perc):

    n_chunks = len(events) // chunk_size
    print(f"      [ChunkSize={chunk_size}] Total Chunks = {n_chunks}")

    for i in range(n_chunks):

        chunk = events[i * chunk_size:(i + 1) * chunk_size]

        save_path = os.path.join(
            out_dir,
            f"event{chunk_size}",
            user,
            f"{session_name}-{i}.png"
        )

        draw_rp_image(chunk, save_path, p_perc, chunk_size)

        if (i + 1) % 50 == 0 or (i + 1) == n_chunks:
            print(f"         -> Chunk {i+1}/{n_chunks} done")

# ============================================================
# Dataset Processing
# ============================================================
def process_dataset(data_root, out_dir, sizes, p_perc):

    users = sorted(os.listdir(data_root))
    total_users = len(users)

    print(f"\n[Dataset] Total Users = {total_users}")

    for u_idx, user in enumerate(users, 1):

        user_dir = os.path.join(data_root, user)
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

            session = os.path.splitext(file)[0]
            print(f"\n   [Session {s_idx}/{total_sessions}] {session}")

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

            events = df[["x", "y", "time"]].values
            print(f"      Total Events = {len(events)}")

            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz, p_perc)

# ============================================================
# Main
# ============================================================
def main():

    global GLOBAL_TD_RAW, GLOBAL_TD_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--td_dist", type=str, default="time_difference_distribution_raw.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--t_percentile", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    dist_path = os.path.join(ROOT, args.td_dist)

    GLOBAL_TD_RAW = load_raw_time_distribution(dist_path)
    GLOBAL_TD_CDF = build_runtime_cdf(GLOBAL_TD_RAW, args.t_percentile)

    print("\n[Step] Generating SRP Time-Difference Images")
    process_dataset(data_root, out_dir, sorted(set(args.sizes)), args.p_percentile)

    print("\n[Done] SRP Time-Difference Generation Complete.")

if __name__ == "__main__":
    main()