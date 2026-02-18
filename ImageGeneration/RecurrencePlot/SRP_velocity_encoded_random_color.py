# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Ablation Version (STRICT G=B)
---------------------------------------------------------
- R Channel: Position Recurrence (STAYS SAME)
- G/B Channels: RANDOM Velocity Stripes (G and B are IDENTICAL)
- Purpose: Ensure identical color space as the original, but destroy velocity semantics.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# Root & Paths
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")
#DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200 

def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Core Ablation Logic: R (Real) + GB (Strictly Same Random)
# ============================================================
def compute_hybrid_rp_ablation_strict(seq, p_percentile=95):
    """
    seq: (T, 3) -> [x, y, t]
    """
    T = len(seq)
    
    # --- R Channel: Position Distance (Symmetric Distance Matrix) ---
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    
    eps_p = np.percentile(dist_p, p_percentile)
    r_channel = 1.0 - np.clip(dist_p / (eps_p + 1e-6), 0, 1)

    # --- G & B Channels: Random Stripes (G must equal B) ---
    # 1. 产生唯一的随机向量，长度为 T
    random_vector = np.random.rand(T) 
    
    # 2. 广播成 T x T 的垂直条纹矩阵
    # 每一列 j 的所有像素值都等于 random_vector[j]
    stripe_matrix = np.tile(random_vector[None, :], (T, 1))
    
    # 3. 严格同步赋值，确保 G 通道和 B 通道完全相同
    g_channel = stripe_matrix
    b_channel = stripe_matrix

    # 堆叠成 RGB 图像
    return np.clip(np.stack([r_channel, g_channel, b_channel], axis=-1), 0.0, 1.0)

# ============================================================
# Processing & Logging
# ============================================================
def draw_rp_image_ablation(seq, save_path, p_perc, chunk_size):
    rgb_rp = compute_hybrid_rp_ablation_strict(seq, p_percentile=p_perc)
    if rgb_rp is None: return

    img_size_px = get_dynamic_image_size(chunk_size)
    fig, ax = plt.subplots(figsize=(img_size_px/DPI, img_size_px/DPI), dpi=DPI)
    
    ax.imshow(rgb_rp, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def clean_df(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"client timestamp": "time", "x": "x", "y": "y", "state": "state"})
    df = df[df["state"] == "Move"].copy()
    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])

def process_dataset(data_dir, out_dir, sizes, p_perc):
    users = sorted(os.listdir(data_dir))
    total_users = len(users)

    for u_idx, user in enumerate(users, 1):
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        
        print(f"\n[{u_idx}/{total_users}] Ablation(Strict G=B) Processing: {user}")
        
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            print(f"  -> Session: {file}")
            
            try:
                df = clean_df(pd.read_csv(os.path.join(user_dir, file)))
                events = df[['x', 'y', 'time']].values
                
                for sz in sizes:
                    n_chunks = len(events) // sz
                    for i in range(n_chunks):
                        chunk = events[i*sz : (i+1)*sz]
                        save_path = os.path.join(out_dir, f"event{sz}", user, f"{file}-{i}.png")
                        draw_rp_image_ablation(chunk, save_path, p_perc, sz)
            except Exception as e:
                print(f"    [Error] {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="Images/SRP_random_color_protocol1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    args = parser.parse_args()

    out_path = os.path.join(ROOT, args.out_dir)
    print(f"[Ablation] Output: {out_path} (G and B channels are SYNCED)")
    process_dataset(DATA_ROOT, out_path, args.sizes, args.p_percentile)