# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Hybrid Encoding with Global Velocity Stripes
---------------------------------------------------------
- R Channel: Position Recurrence (Distance Matrix)
- G/B Channels: Velocity Stripes (Vertical bars based on Global CDF)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

# ============================================================
# Automatically detect project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# paths
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")

# Base Configuration
BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200 

# ============================================================
# Global Scaler Logic: Scan global distribution and generate CDF mapping
# ============================================================
class GlobalVelocityScaler:
    def __init__(self, data_dir, v_percentile=95):
        self.v_max = 0
        self.lookup_func = self._build_scaler(data_dir, v_percentile)

    def _build_scaler(self, data_dir, v_percentile):
        print(f"--- Step 1: Scanning global velocity distribution in {data_dir} ---")
        all_v = []
        
        # Recursively traverse all session files without extensions
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.startswith("session_"):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        df.columns = [c.strip() for c in df.columns]
                        if 'state' not in df.columns: continue
                        
                        df_move = df[df['state'] == 'Move']
                        if len(df_move) < 2: continue
                        
                        dx = df_move['x'].diff()
                        dy = df_move['y'].diff()
                        dt = df_move['client timestamp'].diff() + 1e-6
                        v = np.sqrt(dx**2 + dy**2) / dt
                        all_v.extend(v.dropna().values)
                    except:
                        continue
        
        if not all_v:
            raise ValueError(f"No velocity data found! Please check path: {data_dir}")
            
        all_v = np.array(all_v)
        self.v_max = np.percentile(all_v, v_percentile)
        print(f"Global V_max ({v_percentile}%): {self.v_max:.2f}")

        # Build CDF mapping
        sorted_v = np.sort(np.clip(all_v, 0, self.v_max))
        y = np.linspace(0, 1, len(sorted_v))
        return interp1d(sorted_v, y, bounds_error=False, fill_value=(0, 1))

    def transform(self, v_array):
        return self.lookup_func(np.clip(v_array, 0, self.v_max))

def get_dynamic_image_size(chunk_size):
    """Calculate image size proportionally based on chunk size"""
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    dynamic_size = int(round(BASE_IMG_SIZE * scale))
    return dynamic_size

# ============================================================
# Core RP Logic: R (Distance Matrix) + GB (Global Velocity Stripes)
# ============================================================
def compute_hybrid_rp(seq, v_scaler, p_percentile=100):
    """
    seq: (T, 3) -> [x, y, t]
    """
    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]
    
    # --- R Channel: Position Distance (Symmetric N x N) ---
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    
    eps_p = np.percentile(dist_p, p_percentile)
    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # --- G & B Channels: Velocity Stripes (Vertical) ---
    dt = np.diff(ts) + 1e-6
    v_scalar = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    v_scalar = np.concatenate([[v_scalar[0]], v_scalar]) # Pad to match T
    
    # Map to pixel values [0, 1] using global distribution
    v_norm = v_scaler.transform(v_scalar)
    
    # Broadcast to generate vertical stripes: value of each column j equals v_norm[j]
    stripe_matrix = np.tile(v_norm[None, :], (T, 1))
    
    g_channel = stripe_matrix
    b_channel = stripe_matrix

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)

# ============================================================
# Drawing & IO
# ============================================================
def draw_rp_image(seq, save_path, v_scaler, p_perc, chunk_size):
    rgb_rp = compute_hybrid_rp(seq, v_scaler, p_percentile=p_perc)
    if rgb_rp is None: return

    img_size_px = get_dynamic_image_size(chunk_size)
    fig, ax = plt.subplots(figsize=(img_size_px/DPI, img_size_px/DPI), dpi=DPI)
    
    ax.imshow(rgb_rp, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"client timestamp": "time", "x": "x", "y": "y", "state": "state"})
    df = df[df["state"] == "Move"].copy()
    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])

def process_dataset(data_dir, out_dir, v_scaler, sizes, p_perc, target_users=None):
    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users: continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            
            df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
            events = df[['x', 'y', 'time']].values
            
            print(f"[Process] {user}/{file}")
            for sz in sizes:
                n_chunks = len(events) // sz
                for i in range(n_chunks):
                    chunk = events[i*sz : (i+1)*sz]
                    save_path = os.path.join(out_dir, f"event{sz}", user, f"{file}-{i}.png")
                    draw_rp_image(chunk, save_path, v_scaler, p_perc, sz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_224_velocity_protocol1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--v_percentile", type=float, default=95)
    args = parser.parse_args()

    # 1. Scan global distribution (processing files without extensions)
    v_scaler = GlobalVelocityScaler(args.data_root, v_percentile=args.v_percentile)

    # 2. Generate images
    out_path = os.path.join(ROOT, args.out_dir)
    process_dataset(args.data_root, out_path, v_scaler, args.sizes, args.p_percentile)