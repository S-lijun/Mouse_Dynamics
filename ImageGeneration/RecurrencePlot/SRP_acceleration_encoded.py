# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Hybrid Encoding with Global Acceleration Strips
---------------------------------------------------------
- R Channel: Position Recurrence (Distance Matrix)
- G/B Channels: Acceleration Strips (Vertical bars based on Global CDF)
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
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol2/genuine")

# Base Configuration
BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200 

# ============================================================
# Global Scaler Logic: Scan global distribution and generate Acceleration CDF mapping
# ============================================================
class GlobalAccelerationScaler:
    def __init__(self, data_dir, acc_percentile=95):
        self.acc_max = 0
        self.lookup_func = self._build_scaler(data_dir, acc_percentile)

    def _build_scaler(self, data_dir, acc_percentile):
        print(f"--- Step 1: Scanning global acceleration distribution in {data_dir} ---")
        all_acc = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.startswith("session_"):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        df.columns = [c.strip() for c in df.columns]
                        df_move = df[df['state'] == 'Move']
                        if len(df_move) < 3: continue
                        
                        dt = df_move['client timestamp'].diff() + 1e-6
                        v = np.sqrt(df_move['x'].diff()**2 + df_move['y'].diff()**2) / dt
                        acc = v.diff() / dt
                        all_acc.extend(np.abs(acc.dropna().values))
                    except:
                        continue
        
        if not all_acc:
            raise ValueError(f"No acceleration data found! Path: {data_dir}")
            
        all_acc = np.array(all_acc)
        self.acc_max = np.percentile(all_acc, acc_percentile)
        print(f"Global Acc_max ({acc_percentile}%): {self.acc_max:.2f}")

        sorted_acc = np.sort(np.clip(all_acc, 0, self.acc_max))
        y = np.linspace(0, 1, len(sorted_acc))
        return interp1d(sorted_acc, y, bounds_error=False, fill_value=(0, 1))

    def transform(self, acc_array):
        return self.lookup_func(np.clip(np.abs(acc_array), 0, self.acc_max))

def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Core RP Logic: R (Distance Matrix) + GB (Global Acceleration Vertical Strips)
# ============================================================
def compute_hybrid_rp(seq, acc_scaler, p_percentile=100):
    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]
    
    # --- R Channel: Position Distance ---
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    eps_p = np.percentile(dist_p, p_percentile)
    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # --- G & B Channels: Acceleration Strips (Vertical) ---
    dt = np.diff(ts) + 1e-6
    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    
    # Calculate Acceleration: dv/dt
    acc = np.diff(v) / (dt[1:] + 1e-6)
    # Pad to T points
    acc_full = np.concatenate([[acc[0]], acc, [acc[-1]]]) 
    
    # Use global distribution to map to pixel values [0, 1]
    acc_norm = acc_scaler.transform(acc_full)
    
    # 【Modified】 Broadcast to generate vertical strips: value of each column j equals acc_norm[j]
    # shape: (1, T) -> (T, T)
    stripe_matrix = np.tile(acc_norm[None, :], (T, 1))
    
    g_channel = stripe_matrix
    b_channel = stripe_matrix

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)

# ============================================================
# Drawing & IO
# ============================================================
def draw_rp_image(seq, save_path, acc_scaler, p_perc, chunk_size):
    rgb_rp = compute_hybrid_rp(seq, acc_scaler, p_percentile=p_perc)
    if rgb_rp is None: return

    img_size_px = get_dynamic_image_size(chunk_size)
    fig, ax = plt.subplots(figsize=(img_size_px/DPI, img_size_px/DPI), dpi=DPI)
    
    # origin="lower" ensures time increases from bottom to top and left to right
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

def process_dataset(data_dir, out_dir, acc_scaler, sizes, p_perc, target_users=None):
    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users: continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            
            try:
                df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
                events = df[['x', 'y', 'time']].values
                if len(events) < max(sizes): continue
                
                print(f"[Process] {user}/{file}")
                for sz in sizes:
                    n_chunks = len(events) // sz
                    for i in range(n_chunks):
                        chunk = events[i*sz : (i+1)*sz]
                        save_path = os.path.join(out_dir, f"event{sz}", user, f"{file}-{i}.png")
                        draw_rp_image(chunk, save_path, acc_scaler, p_perc, sz)
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_acceleration_protocol2/genuine")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=100)
    parser.add_argument("--acc_percentile", type=float, default=95)
    args = parser.parse_args()

    acc_scaler = GlobalAccelerationScaler(args.data_root, acc_percentile=args.acc_percentile)

    out_path = os.path.join(ROOT, args.out_dir)
    process_dataset(args.data_root, out_path, acc_scaler, args.sizes, args.p_percentile)