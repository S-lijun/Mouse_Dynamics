# -*- coding: utf-8 -*-
"""
Fixed Canvas Version (448x448) with GLOBAL Velocity CDF - RED TO BLACK
--------------------------------------------------------------------------
- Color encoded by velocity: Slow = RED, Fast = BLACK
- Background: WHITE (Red and Black both stand out)
- Maintains physical aspect ratio
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

# ============================================================
# Path & Fixed Config
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

FIXED_IMG_SIZE = 448
FIXED_LINEWIDTH = 1.0
FIXED_MARKERSIZE = 1.0
DPI = 100 

GLOBAL_V_ALL = None
GLOBAL_CDF_ALL = None

# 创建自定义 Colormap: 0.0(慢) -> 纯红 (1,0,0); 1.0(快) -> 纯黑 (0,0,0)
cdict = {
    'red':   [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
    'green': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    'blue':  [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
}
RED_TO_BLACK_CMP = LinearSegmentedColormap('RedToBlack', cdict)

# ============================================================
# Data Cleaning
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"client timestamp": "time", "x": "x", "y": "y", "state": "state"})
    df = df[df["state"] == "Move"].copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Build GLOBAL velocity CDF
# ============================================================
def build_global_velocity_cdf(data_dir, clip_pct=95.0):
    velocities = []
    if not os.path.exists(data_dir): return np.array([0, 1000]), np.array([0, 1])

    for user in sorted(os.listdir(data_dir)):
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            try:
                df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
                if len(df) < 2: continue
                v = np.sqrt(np.diff(df["x"])**2 + np.diff(df["y"])**2) / np.maximum(np.diff(df["time"]), 1e-5)
                v = v[np.isfinite(v)]
                if len(v) > 0: velocities.append(v)
            except: continue

    if not velocities: return np.array([0, 1000]), np.array([0, 1])
    velocities = np.concatenate(velocities)
    v_upper_bound = np.percentile(velocities, clip_pct)
    velocities = velocities[velocities <= v_upper_bound]
    ranks = rankdata(velocities, method="average")
    cdf = (ranks - 1) / (len(velocities) - 1 + 1e-8)
    order = np.argsort(velocities)
    return velocities[order], cdf[order]

# ============================================================
# Drawing Function (Red to Black)
# ============================================================
def draw_mouse_chunk(chunk, save_path, global_bounds):
    if len(chunk) < 2: return
    SIDE = FIXED_IMG_SIZE

    # 1. Coordinate Mapping
    g_min_x, g_max_x, g_min_y, g_max_y = global_bounds
    range_x, range_y = max(g_max_x - g_min_x, 1.0), max(g_max_y - g_min_y, 1.0)
    y_target_height = (range_y / range_x) * SIDE

    xs, ys, ts = np.array([e["x"] for e in chunk]), np.array([e["y"] for e in chunk]), np.array([e["time"] for e in chunk])
    x_pix = ((xs - g_min_x) / range_x) * SIDE
    y_pix = ((ys - g_min_y) / range_y) * y_target_height 

    # 2. Velocity & Red-Black Encoding
    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / np.maximum(np.diff(ts), 1e-5)
    v_norm = np.interp(v, GLOBAL_V_ALL, GLOBAL_CDF_ALL, left=0.0, right=1.0)
    seg_colors = RED_TO_BLACK_CMP(v_norm) # 0.0 -> Red, 1.0 -> Black

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(SIDE/DPI, SIDE/DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    ax.set_xlim(0, SIDE)
    ax.set_ylim(SIDE, 0)
    ax.axis("off")

    for i in range(len(xs) - 1):
        ax.plot(
            [x_pix[i], x_pix[i+1]], [y_pix[i], y_pix[i+1]],
            color=seg_colors[i], linewidth=FIXED_LINEWIDTH,
            marker="o", markersize=FIXED_MARKERSIZE,
            markerfacecolor=seg_colors[i], markeredgewidth=0,
            solid_capstyle='round'
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)

# ============================================================
# Main Processing Logic
# ============================================================
def main():
    global GLOBAL_V_ALL, GLOBAL_CDF_ALL
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=f"Images/fixed_{FIXED_IMG_SIZE}padding_rb_cdf")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 30, 60, 120, 300])
    parser.add_argument("--clip", type=float, default=95.0)
    args = parser.parse_args()

    GLOBAL_V_ALL, GLOBAL_CDF_ALL = build_global_velocity_cdf(DATA_ROOT, clip_pct=args.clip)
    full_out_dir = os.path.join(ROOT, args.out_dir)

    for user in sorted(os.listdir(DATA_ROOT)):
        user_dir = os.path.join(DATA_ROOT, user)
        if not os.path.isdir(user_dir): continue
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
            if df.empty: continue
            
            g_bounds = (df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max())
            events = df.to_dict(orient="records")
            
            print(f"[Processing] {user}/{file} (Red-Black)")
            for sz in args.sizes:
                for i in range(len(events) // sz):
                    chunk = events[i*sz : (i+1)*sz]
                    save_path = os.path.join(full_out_dir, f"event{sz}", user, f"{os.path.splitext(file)[0]}-{i}.png")
                    draw_mouse_chunk(chunk, save_path, g_bounds)

if __name__ == "__main__":
    main()