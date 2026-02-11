# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Multi-Channel Directional Encoding
---------------------------------------------------------
- R Channel: Position Recurrence (Distance Matrix)
- G/B Channels: Directional Similarity (Cosine Similarity Matrix)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# Automatically detect project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# 默认数据路径
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")


# Base Configuration
BASE_CHUNK_SIZE = 15
BASE_IMG_SIZE = 224
DPI = 100 

def get_dynamic_image_size(chunk_size):
    """根据比例计算图片尺寸"""
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    dynamic_size = int(round(BASE_IMG_SIZE * scale))
    return dynamic_size

# ============================================================
# Core RP Logic: R(距离) + GB(方向余弦相似度)
# ============================================================
def compute_directional_rp(seq, p_percentile=100):
    """
    seq: (T, 3) -> [x, y, t]
    """
    T = len(seq)
    if T < 3: return None
    
    xs, ys = seq[:, 0], seq[:, 1]
    
    # --- R Channel: Position Distance (N x N) ---
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    
    eps_p = np.percentile(dist_p, p_percentile)
    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # --- G & B Channels: Directional Cosine Similarity (N x N) ---
    # 1. 计算位移向量 (dx, dy)
    dx = np.diff(xs)
    dy = np.diff(ys)
    dir_vecs = np.stack([dx, dy], axis=1) # shape: (T-1, 2)
    
    # 2. 归一化为单位向量
    norms = np.linalg.norm(dir_vecs, axis=1, keepdims=True) + 1e-9
    unit_vecs = dir_vecs / norms
    
    # 3. 为了对齐 T 个点，我们在首位复制一个向量 (或末尾，保持维度一致)
    # 这样矩阵就是 T x T
    unit_vecs_full = np.vstack([unit_vecs[0:1], unit_vecs]) 
    
    # 4. 计算 Cosine Similarity 矩阵: (T, 2) dot (2, T) -> (T, T)
    cos_sim = np.dot(unit_vecs_full, unit_vecs_full.T)
    
    # 5. 线性映射从 [-1, 1] 到 [0, 1]
    # 靠近1表示方向一致，靠近0表示方向相反
    dir_matrix = (cos_sim + 1.0) / 2.0
    
    g_channel = dir_matrix
    b_channel = dir_matrix

    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)

# ============================================================
# Drawing & IO
# ============================================================
def draw_rp_image(seq, save_path, p_perc, chunk_size):
    rgb_rp = compute_directional_rp(seq, p_percentile=p_perc)
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
    # 清理 Balabit 常见的表头空格
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"client timestamp": "time", "x": "x", "y": "y", "state": "state"})
    # 只处理 Move 状态
    df = df[df["state"] == "Move"].copy()
    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])

def process_dataset(data_dir, out_dir, sizes, p_perc, target_users=None):
    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users: continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        
        # 遍历无后缀文件
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
                        draw_rp_image(chunk, save_path, p_perc, sz)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str, default="Images/SRP_angle_protocol1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 30, 60, 120])
    parser.add_argument("--p_percentile", type=float, default=100, help="Position distance percentile")
    args = parser.parse_args()

    # 执行处理
    out_path = os.path.join(ROOT, args.out_dir)
    process_dataset(args.data_root, out_path, args.sizes, args.p_percentile)