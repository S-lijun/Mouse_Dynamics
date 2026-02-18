# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Symmetrical Hybrid Encoding (Full Threshold Control)
---------------------------------------------------------
- R Channel: Distance Matrix (p_percentile threshold)
- G/B Channels: Velocity Stripes (v_percentile threshold)
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
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")

BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200 

# ============================================================
# Global Scaler Logic: Adaptive Velocity CDF Mapping
# ============================================================
class GlobalVelocityScaler:
    def __init__(self, data_dir, v_percentile=95):
        self.v_max = 0
        # 传入 v_percentile，决定速度映射的上限
        self.lookup_func = self._build_scaler(data_dir, v_percentile)

    def _build_scaler(self, data_dir, v_percentile):
        print(f"--- Step 1: Scanning global velocity distribution in {data_dir} ---")
        all_v = []
        
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
                df_move = df[df['state'] == 'Move']
                if len(df_move) < 2: continue
                
                dx = df_move['x'].diff()
                dy = df_move['y'].diff()
                dt = df_move['client timestamp'].diff() + 1e-6
                v = np.sqrt(dx**2 + dy**2) / dt
                all_v.extend(v.dropna().values)
            except:
                continue
        
        all_v = np.array(all_v)
        # 这里的 v_max 决定了速度条纹中最亮(值为1)的物理含义
        self.v_max = np.percentile(all_v, v_percentile)
        print(f"--- Scan complete. V_max ({v_percentile}%): {self.v_max:.2f} ---")

        # 构建 CDF，确保高于 v_max 的部分截断为 1
        sorted_v = np.sort(np.clip(all_v, 0, self.v_max))
        y = np.linspace(0, 1, len(sorted_v))
        return interp1d(sorted_v, y, bounds_error=False, fill_value=(0, 1))

    def transform(self, v_array):
        # 将速度数组根据 CDF 映射到 [0, 1]
        return self.lookup_func(np.clip(v_array, 0, self.v_max))

# ============================================================
# Core RP Logic
# ============================================================
def compute_hybrid_rp(seq, v_scaler, p_percentile=100):
    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]
    
    # R Channel: Position Distance (p_percentile 截断)
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    eps_p = np.percentile(dist_p, p_percentile)
    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # Velocity Preparation
    dt = np.diff(ts) + 1e-6
    v_scalar = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    v_scalar = np.concatenate([[v_scalar[0]], v_scalar])
    v_norm = v_scaler.transform(v_scalar)
    
    # G & B Channels: Symmetrical Stripes
    # G=竖条(代表j), B=横条(代表i)
    g_channel = np.tile(v_norm[None, :], (T, 1))
    b_channel = np.tile(v_norm[:, None], (1, T))

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)

def draw_rp_image(seq, save_path, v_scaler, p_perc, chunk_size):
    rgb_rp = compute_hybrid_rp(seq, v_scaler, p_percentile=p_perc)
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
# Processing Engine
# ============================================================
def process_dataset(data_dir, out_dir, v_scaler, sizes, p_perc):
    users = sorted([u for u in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, u))])
    total_users = len(users)

    print(f"\n[Step 2] Generating SRP Images (P_perc={p_perc}, V_perc={v_scaler.v_max})...")
    
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
                df = df.rename(columns={"client timestamp": "time", "x": "x", "y": "y", "state": "state"})
                df = df[df["state"] == "Move"].copy()
                events = df[['x', 'y', 'time']].dropna().values
                
                for sz in sizes:
                    n_chunks = len(events) // sz
                    for i in range(n_chunks):
                        chunk = events[i*sz : (i+1)*sz]
                        rel_path = os.path.relpath(os.path.dirname(file_path), data_dir)
                        save_path = os.path.join(out_dir, f"event{sz}", rel_path, f"{file_name}-{i}.png")
                        draw_rp_image(chunk, save_path, v_scaler, p_perc, sz)
            except Exception as e:
                print(f"    !!! Error processing {file_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_Velocity_Sym_protocol1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)

    parser.add_argument("--v_percentile", type=float, default=95)
    args = parser.parse_args()

    # 初始化时传入 v_percentile
    v_scaler = GlobalVelocityScaler(args.data_root, v_percentile=args.v_percentile)
    
    process_dataset(args.data_root, os.path.join(ROOT, args.out_dir), v_scaler, args.sizes, args.p_percentile)