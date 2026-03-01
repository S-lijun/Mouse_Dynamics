# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) — Multi-Channel Encoding with Dual Percentile Control
---------------------------------------------------------
- R Channel: Position Recurrence (controlled by p_percentile)
- G Channel: Velocity Recurrence (controlled by v_percentile)
- B Channel: Direction Recurrence (Cosine Similarity, No Percentile needed)
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
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")

# Base Configuration
BASE_CHUNK_SIZE = 15
BASE_IMG_SIZE = 224
DPI = 100 

def get_dynamic_image_size(chunk_size):
    """Calculate image dimensions proportionally based on chunk size"""
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    dynamic_size = int(round(BASE_IMG_SIZE * scale))
    return dynamic_size

# ============================================================
# Core RP Logic
# ============================================================
def compute_multichannel_rp(seq, p_percentile=95, v_percentile=95):
    """
    seq: (T, 3) -> [x, y, t]
    p_percentile: Cutoff percentile for position distance
    v_percentile: Cutoff percentile for velocity difference
    """
    non_zero_mask = ~np.all(seq == 0, axis=1)
    seq = seq[non_zero_mask]
    if len(seq) < 3: return None

    xs, ys, ts = seq[:, 0], seq[:, 1], seq[:, 2]
    
    # --- R Channel: Position ---
    coords = seq[:, :2]
    diff_p = coords[:, None, :] - coords[None, :, :]
    dist_p = np.sqrt(np.sum(diff_p ** 2, axis=2))
    
    # Control position truncation using p_percentile
    eps_p = np.percentile(dist_p, p_percentile)
    rec_p = np.clip(dist_p, 0, eps_p)
    r_channel = 1.0 - (rec_p / (eps_p + 1e-6))

    # --- G Channel: Velocity ---
    dt = np.diff(ts) + 1e-6
    dx = np.diff(xs)
    dy = np.diff(ys)
    v_scalar = np.sqrt(dx**2 + dy**2) / dt
    v_scalar = np.concatenate([[v_scalar[0]], v_scalar]) # Align to T (padding the first element)
    
    # Calculate velocity difference matrix between all pairs
    dist_v = np.abs(v_scalar[:, None] - v_scalar[None, :])
    
    # Use v_percentile for velocity truncation to handle skewed data distributions
    eps_v = np.percentile(dist_v, v_percentile)
    rec_v = np.clip(dist_v, 0, eps_v)
    g_channel = 1.0 - (rec_v / (eps_v + 1e-6))

    # --- B Channel: Angle ---
    # Calculate directional unit vectors
    dir_vecs = np.stack([dx, dy], axis=1)
    norms = np.linalg.norm(dir_vecs, axis=1, keepdims=True) + 1e-6
    unit_vecs = dir_vecs / norms
    unit_vecs = np.vstack([unit_vecs[0:1], unit_vecs]) # Align to T
    
    # Cosine Similarity: Range [-1, 1]
    cos_sim = np.dot(unit_vecs, unit_vecs.T)
    # Linearly map to [0, 1]
    b_channel = (cos_sim + 1.0) / 2.0

    # Final Defense: Force clamp all channels to [0.0, 1.0] to avoid matplotlib warnings
    r_channel = np.clip(r_channel, 0.0, 1.0)
    g_channel = np.clip(g_channel, 0.0, 1.0)
    b_channel = np.clip(b_channel, 0.0, 1.0)

    return np.stack([r_channel, g_channel, b_channel], axis=-1)

# ============================================================
# Drawing & IO
# ============================================================
def draw_rp_image(seq, save_path, p_perc, v_perc, chunk_size):
    rgb_rp = compute_multichannel_rp(seq, p_percentile=p_perc, v_percentile=v_perc)
    if rgb_rp is None: return

    img_size_px = get_dynamic_image_size(chunk_size)
    
    # Plot configuration
    fig, ax = plt.subplots(figsize=(img_size_px/DPI, img_size_px/DPI), dpi=DPI)
    
    # origin="lower" ensures matrix (0,0) is at bottom-left, matching the time axis growth
    ax.imshow(rgb_rp, origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"client timestamp": "time", "x": "x", "y": "y", "state": "state"})
    df = df[df["state"] == "Move"].copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])

def process_dataset(data_dir, out_dir, sizes, p_perc, v_perc, target_users=None):
    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users: continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            session_name = os.path.splitext(file)[0]
            
            df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
            events = df.to_dict(orient="records")
            
            print(f"[Process] {user}/{file} (P-Perc: {p_perc}, V-Perc: {v_perc})")
            for sz in sizes:
                n_chunks = len(events) // sz
                for i in range(n_chunks):
                    chunk = events[i*sz : (i+1)*sz]
                    seq = np.array([[e["x"], e["y"], e["time"]] for e in chunk], dtype=np.float32)
                    save_path = os.path.join(out_dir, f"event{sz}", user, f"{session_name}-{i}.png")
                    draw_rp_image(seq, save_path, p_perc, v_perc, sz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="Images/SRP_MultiChannel_protocol1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 30, 60, 120])
    parser.add_argument("--p_percentile", type=float, default=100, help="Percentile for Position distance")
    parser.add_argument("--v_percentile", type=float, default=100, help="Percentile for Velocity difference")
    args = parser.parse_args()

    out_path = os.path.join(ROOT, args.out_dir)
    process_dataset(DATA_ROOT, out_path, args.sizes, args.p_percentile, args.v_percentile)