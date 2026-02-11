# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Triple-Channel Hybrid (D-R, V-Vertical-G, A-Horizontal-B)
---------------------------------------------------------
- R Channel: Position Recurrence (Distance Matrix, local or global)
- G Channel: Velocity Stripes (Vertical, Global CDF Mapped)
- B Channel: Acceleration Stripes (Horizontal, Global CDF Mapped)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

# ============================================================
# 路径与基础配置
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

BASE_CHUNK_SIZE = 15
BASE_IMG_SIZE = 224
DPI = 100 

# ============================================================
# Global Scaler Logic: 分别处理 V 和 A 的全局分布
# ============================================================
class GlobalFeatureScaler:
    def __init__(self, data_dir, v_perc=95, a_perc=95):
        self.v_max = 0
        self.acc_max = 0
        self.v_lookup = None
        self.acc_lookup = None
        self._build_scalers(data_dir, v_perc, a_perc)

    def _build_scalers(self, data_dir, v_perc, a_perc):
        print(f"--- Step 1: Scanning global distributions (V_perc={v_perc}, A_perc={a_perc}) ---")
        all_v = []
        all_acc = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.startswith("session_"):
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        df.columns = [c.strip() for c in df.columns]
                        df_move = df[df['state'] == 'Move']
                        if len(df_move) < 3: continue
                        
                        dt = df_move['client timestamp'].diff() + 1e-6
                        v = np.sqrt(df_move['x'].diff()**2 + df_move['y'].diff()**2) / dt
                        acc = v.diff() / (dt[1:] + 1e-6)
                        
                        all_v.extend(v.dropna().values)
                        all_acc.extend(np.abs(acc.dropna().values))
                    except:
                        continue
        
        if not all_v or not all_acc:
            raise ValueError("No valid movement data found!")

        # 为 V 和 A 分别计算独立的阈值
        self.v_max = np.percentile(all_v, v_perc)
        self.acc_max = np.percentile(all_acc, a_perc)
        print(f"Global Stats: V_max({v_perc}%): {self.v_max:.2f} | Acc_max({a_perc}%): {self.acc_max:.2f}")

        # 构建各自的 CDF 映射函数
        self.v_lookup = self._create_interp(all_v, self.v_max)
        self.acc_lookup = self._create_interp(all_acc, self.acc_max)

    def _create_interp(self, data, val_max):
        sorted_data = np.sort(np.clip(data, 0, val_max))
        y = np.linspace(0, 1, len(sorted_data))
        return interp1d(sorted_data, y, bounds_error=False, fill_value=(0, 1))

    def transform_v(self, v_array):
        return self.v_lookup(np.clip(v_array, 0, self.v_max))

    def transform_acc(self, acc_array):
        return self.acc_lookup(np.clip(np.abs(acc_array), 0, self.acc_max))

# ============================================================
# Core RP Logic
# ============================================================
def compute_triple_channel_rp(seq, scaler, p_perc=100):
    T = len(seq)
    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]
    coords = seq[:, :2]

    # --- R Channel: Distance (Local Recurrence) ---
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    # 使用 chunk 内的分位数作为 epsilon
    eps_p = np.percentile(dist_p, p_perc)
    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # --- 计算动力学序列 ---
    dt = np.diff(ts) + 1e-6
    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    acc = np.diff(v) / (dt[1:] + 1e-6)

    # 维度对齐 T
    v_full = np.concatenate([[v[0]], v])
    acc_full = np.concatenate([[acc[0]], acc, [acc[-1]]])

    # 映射到 [0, 1]
    v_norm = scaler.transform_v(v_full)
    acc_norm = scaler.transform_acc(acc_full)

    # --- G Channel: Velocity (竖条 Vertical) ---
    g_channel = np.tile(v_norm[None, :], (T, 1))

    # --- B Channel: Acceleration (横条 Horizontal) ---
    b_channel = np.tile(acc_norm[:, None], (1, T))

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)

# ============================================================
# Drawing & IO Logic
# ============================================================
def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

def draw_rp_image(seq, save_path, scaler, p_perc, chunk_size):
    rgb_rp = compute_triple_channel_rp(seq, scaler, p_perc=p_perc)
    img_size = get_dynamic_image_size(chunk_size)
    
    fig, ax = plt.subplots(figsize=(img_size/DPI, img_size/DPI), dpi=DPI)
    ax.imshow(rgb_rp, origin="lower", aspect='auto')
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

def process_dataset(data_dir, out_dir, scaler, sizes, p_perc):
    user_list = sorted([u for u in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, u))])
    for user in user_list:
        user_dir = os.path.join(data_dir, user)
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            try:
                df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
                events = df[['x', 'y', 'time']].values
                if len(events) < max(sizes): continue
                
                print(f"[Process] {user}/{file}")
                for sz in sizes:
                    for i in range(len(events) // sz):
                        chunk = events[i*sz : (i+1)*sz]
                        save_path = os.path.join(out_dir, f"event{sz}", user, f"{file}-{i}.png")
                        draw_rp_image(chunk, save_path, scaler, p_perc, sz)
            except Exception as e:
                print(f"Skip {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_dva")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 30, 60, 120])
    # 三个通道独立的阈值设置
    parser.add_argument("--p_perc", type=float, default=100, help="Distance percentile (local)")
    parser.add_argument("--v_perc", type=float, default=95, help="Global velocity percentile")
    parser.add_argument("--a_perc", type=float, default=95, help="Global acceleration percentile")
    args = parser.parse_args()

    # 1. 扫描全局并应用独立分位数
    scaler = GlobalFeatureScaler(args.data_root, v_perc=args.v_perc, a_perc=args.a_perc)

    # 2. 生成图像
    out_path = os.path.join(ROOT, args.out_dir)
    process_dataset(args.data_root, out_path, scaler, args.sizes, args.p_perc)