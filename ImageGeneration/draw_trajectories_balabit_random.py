# -*- coding: utf-8 -*-
"""
Ablation Study Version — RANDOM Color Encoding
---------------------------------------------------------
- Goal: Test if velocity semantics matter.
- Encoding: Uniform Random [0, 1] color assignment.
- Geometry: Same as Velocity version.
- LOGGING: Full session-level progress printing.
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
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

print(f"[Ablation] Project root detected = {ROOT}")
print(f"[Ablation] Target Data Root = {DATA_ROOT}")

# ============================================================
# Global Anchors
# ============================================================
BASE_EVENT = 60
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 0.5
BASE_MARKERSIZE = 1.0
DPI = 200

# ============================================================
# Scaling & Utils
# ============================================================
def get_img_size(chunk_size):
    scale = chunk_size / BASE_EVENT
    side = int(round(np.sqrt(scale * BASE_IMG_SIZE * BASE_IMG_SIZE)))
    return (side, side)

def get_stroke_params(chunk_size):
    scale = np.sqrt(chunk_size / BASE_EVENT)
    return BASE_LINEWIDTH * scale, BASE_MARKERSIZE * scale

def _scaled(val, min_val, scale, offset):
    return (val - min_val) * scale + offset

def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"client timestamp": "time", "x": "x", "y": "y", "state": "state"})
    df = df[df["state"] == "Move"].copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    mask_valid = (df["x"] < 65535) & (df["y"] < 65535)
    return df[mask_valid].dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Drawing (Random Color via Continuous Uniform Distribution)
# ============================================================
def draw_mouse_chunk_random_color(chunk, save_path, chunk_size):
    if len(chunk) < 2: return

    IMG_SIZE = get_img_size(chunk_size)
    linewidth, markersize = get_stroke_params(chunk_size)

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])

    # ABLATION: Sample colors from Uniform(0, 1)
    num_segments = len(xs) - 1
    random_color_indices = np.random.rand(num_segments) # Continuous Uniform Distribution
    
    cmap = plt.get_cmap("plasma")
    seg_colors = cmap(random_color_indices)

    # Geometry & Scaling
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    range_x, range_y = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
    pad_x, pad_y = range_x * 0.05, range_y * 0.05
    min_x, max_x = min_x - pad_x, max_x + pad_x
    min_y, max_y = min_y - pad_y, max_y + pad_y
    range_x, range_y = max_x - min_x, max_y - min_y

    scale = min(IMG_SIZE[0] / range_x, IMG_SIZE[1] / range_y)
    offset_x = (IMG_SIZE[0] - range_x * scale) / 2.0
    offset_y = (IMG_SIZE[1] - range_y * scale) / 2.0

    fig, ax = plt.subplots(figsize=(IMG_SIZE[0]/DPI, IMG_SIZE[1]/DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, IMG_SIZE[0]); ax.set_ylim(IMG_SIZE[1], 0)
    ax.axis("off")

    for i in range(num_segments):
        x1_s = _scaled(xs[i], min_x, scale, offset_x)
        y1_s = _scaled(ys[i], min_y, scale, offset_y)
        x2_s = _scaled(xs[i+1], min_x, scale, offset_x)
        y2_s = _scaled(ys[i+1], min_y, scale, offset_y)
        
        color = seg_colors[i]
        ax.plot(
            [x1_s, x2_s], [y1_s, y2_s],
            color=color, linewidth=linewidth,
            marker="o", markersize=markersize,
            markerfacecolor=color, markeredgewidth=0,
            solid_capstyle='round'
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)

# ============================================================
# Dataset Processing with Full Logging
# ============================================================
def process_dataset(data_dir, out_dir, sizes, target_users=None):
    if not os.path.exists(data_dir):
        print(f"[Error] Data directory not found: {data_dir}")
        return

    users = sorted(os.listdir(data_dir))
    total_users = len(users)

    for u_idx, user in enumerate(users, 1):
        if target_users and user not in target_users:
            continue
            
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue
            
        print(f"\n[{u_idx}/{total_users}] Processing User: {user}")
        
        session_files = [f for f in sorted(os.listdir(user_dir)) if f.startswith("session_")]
        
        for file in session_files:
            session_name = os.path.splitext(file)[0]
            # 这一行就是你要的打印
            print(f"  -> Session: {session_name}")
            
            try:
                df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
                events = df.to_dict(orient="records")
                
                for sz in sizes:
                    n_chunks = len(events) // sz
                    # 可以选择在这里再加一个 chunk 数量的打印，如果需要更细的话
                    for i in range(n_chunks):
                        chunk = events[i*sz : (i+1)*sz]
                        save_path = os.path.join(out_dir, f"event{sz}", user, f"{session_name}-{i}.png")
                        draw_mouse_chunk_random_color(chunk, save_path, sz)
            except Exception as e:
                print(f"    [Error] Failed to process {file}: {e}")

# ============================================================
# Main Execution
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation: Random Color (Uniform Distribution)")
    parser.add_argument("--out_dir", type=str, default="Images/XYPlot_random_color")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    args = parser.parse_args()

    full_out_path = os.path.join(ROOT, args.out_dir)
    print(f"[Ablation] Output directory: {full_out_path}")
    
    process_dataset(DATA_ROOT, full_out_path, sorted(set(args.sizes)))
    print("\n[Done] Random Color Ablation complete.")

if __name__ == "__main__":
    main()