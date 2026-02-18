# -*- coding: utf-8 -*-
"""
Ablation Study Version — RANDOM Color Encoding
---------------------------------------------------------
- Goal: Test if velocity semantics matter or if color alone provides the boost.
- Encoding: Randomly assign colors from the 'plasma' colormap to each point.
- Geometry: Identical to the Velocity CDF version.
- Pure chunking (no sliding window)
- White background, random color information
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
# Ensure this matches your data path
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

print(f"[Ablation] Project root detected = {ROOT}")

# ============================================================
# Global Anchors
# ============================================================
BASE_EVENT = 60
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 0.5
BASE_MARKERSIZE = 1.0
DPI = 200

# ============================================================
# Scaling & Cleaning Utils
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
# Drawing (RANDOM Color Assignment)
# ============================================================
def draw_mouse_chunk_random_color(chunk, save_path, chunk_size):
    """Renders a chunk with RANDOM colors from plasma map."""
    if len(chunk) < 2: return

    IMG_SIZE = get_img_size(chunk_size)
    linewidth, markersize = get_stroke_params(chunk_size)

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])

    # --- ABLATION CORE: Randomly assign colors ---
    # Instead of looking up Velocity CDF, we pick random values in [0, 1]
    # for each segment. This preserves color richness but removes semantic meaning.
    num_segments = len(xs) - 1
    random_color_indices = np.random.rand(num_segments)
    
    cmap = plt.get_cmap("plasma")
    seg_colors = cmap(random_color_indices)

    # Geometry & Padding
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
# Processing Loop
# ============================================================
def process_dataset(data_dir, out_dir, sizes, target_users=None):
    users = sorted(os.listdir(data_dir))
    for user in users:
        if target_users and user not in target_users: continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        
        print(f"[Ablation-Random] Processing User: {user}")
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            
            df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
            events = df.to_dict(orient="records")
            session_name = os.path.splitext(file)[0]

            for sz in sizes:
                n_chunks = len(events) // sz
                for i in range(n_chunks):
                    chunk = events[i*sz : (i+1)*sz]
                    save_path = os.path.join(out_dir, f"event{sz}", user, f"{session_name}-{i}.png")
                    draw_mouse_chunk_random_color(chunk, save_path, sz)

# ============================================================
# CLI Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Random Color vs Velocity")
    parser.add_argument("--out_dir", type=str, default="Images/XYPlot_random_color")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    process_dataset(DATA_ROOT, out_dir, sorted(set(args.sizes)))

if __name__ == "__main__":
    main()