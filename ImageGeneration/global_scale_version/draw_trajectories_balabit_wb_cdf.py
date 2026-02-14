# -*- coding: utf-8 -*-
"""
Fixed Canvas Version (448x448) with GLOBAL Velocity CDF - GRAYSCALE
--------------------------------------------------------------------------
- Fixed 448x448 canvas size
- Color encoded by velocity using GLOBAL CDF (Grayscale: Fast=Black, Slow=White)
- Percentile Clipping for outliers
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# ============================================================
# Path & Fixed Config
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# 默认路径根据你的目录结构调整
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")

# Fixed canvas and stroke parameters
FIXED_IMG_SIZE = 448
FIXED_LINEWIDTH = 1.0
FIXED_MARKERSIZE = 1.0
DPI = 100 # 调整DPI以匹配 448x448

# Global holders for Velocity CDF
GLOBAL_V_ALL = None
GLOBAL_CDF_ALL = None

# ============================================================
# Data Cleaning (Discard 65535)
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x", 
        "y": "y", 
        "state": "state"
    })
    # Keep only movement events
    df = df[df["state"] == "Move"].copy()
    
    # Ensure numeric types
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    
    # Discard 65535 outliers and NaNs
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Build GLOBAL velocity CDF
# ============================================================
def build_global_velocity_cdf(data_dir, clip_pct=95.0):
    velocities = []
    print(f"[GlobalCDF] Scanning {data_dir} for CDF building...")
    
    if not os.path.exists(data_dir):
        print(f"Error: Path {data_dir} does not exist.")
        return np.array([0, 1000]), np.array([0, 1])

    for user in sorted(os.listdir(data_dir)):
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            
            try:
                df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
                if len(df) < 2: continue

                xs, ys, ts = df["x"].values, df["y"].values, df["time"].values
                dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(ts)
                
                dt[dt <= 0] = 1e-5
                v = np.sqrt(dx**2 + dy**2) / dt
                v = v[np.isfinite(v)]
                if len(v) > 0: velocities.append(v)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not velocities:
        return np.array([0, 1000]), np.array([0, 1])

    velocities = np.concatenate(velocities)
    v_upper_bound = np.percentile(velocities, clip_pct)
    velocities = velocities[velocities <= v_upper_bound]
    
    ranks = rankdata(velocities, method="average")
    cdf = (ranks - 1) / (len(velocities) - 1 + 1e-8)

    order = np.argsort(velocities)
    return velocities[order], cdf[order]

# ============================================================
# Drawing Function (Grayscale Encoding)
# ============================================================
def draw_mouse_chunk(chunk, save_path, global_bounds):
    if len(chunk) < 2: return

    SIDE = FIXED_IMG_SIZE
    lw = FIXED_LINEWIDTH
    ms = FIXED_MARKERSIZE
    
    # 1. Coordinate Mapping (Physical Ratio)
    g_min_x, g_max_x, g_min_y, g_max_y = global_bounds
    range_x = max(g_max_x - g_min_x, 1.0)
    range_y = max(g_max_y - g_min_y, 1.0)
    y_target_height = (range_y / range_x) * SIDE

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    norm_x = (xs - g_min_x) / range_x
    norm_y = (ys - g_min_y) / range_y
    x_pix = norm_x * SIDE
    y_pix = norm_y * y_target_height 

    # 2. Velocity & Color Calculation (Grayscale)
    dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(ts)
    dt[dt <= 0] = 1e-5
    velocity = np.sqrt(dx**2 + dy**2) / dt

    # Map to CDF [0, 1]
    v_norm = np.interp(velocity, GLOBAL_V_ALL, GLOBAL_CDF_ALL, left=0.0, right=1.0)
    
    # Grayscale mapping: High speed (v_norm -> 1) is Black (0), Low speed is White (1)
    # We use (1.0 - v_norm) to invert it
    gray_values = 1.0 - v_norm
    cmap = plt.get_cmap("gray")
    seg_colors = cmap(gray_values)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(SIDE/DPI, SIDE/DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    ax.set_xlim(0, SIDE)
    ax.set_ylim(SIDE, 0) # 0 at top (Screen coordinates style)
    ax.axis("off")

    # Draw segments
    for i in range(len(xs) - 1):
        ax.plot(
            [x_pix[i], x_pix[i+1]], [y_pix[i], y_pix[i+1]],
            color=seg_colors[i], linewidth=lw,
            marker="o", markersize=ms,
            markerfacecolor=seg_colors[i], markeredgewidth=0,
            solid_capstyle='round'
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)

# ============================================================
# Processing Logic
# ============================================================
def process_one_session(path, user, session_name, out_dir, sizes):
    try:
        df = pd.read_csv(path)
        df = clean_and_rename_cols(df)
        if df.empty: return

        # Use local session bounds to maintain internal relative aspect ratio
        g_bounds = (df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max())
        events = df.to_dict(orient="records")
        
        for sz in sizes:
            n_chunks = len(events) // sz
            for i in range(n_chunks):
                chunk = events[i*sz : (i+1)*sz]
                save_path = os.path.join(out_dir, f"event{sz}", user, f"{session_name}-{i}.png")
                draw_mouse_chunk(chunk, save_path, g_bounds)
    except Exception as e:
        print(f"Skip {session_name} due to error: {e}")

def main():
    global GLOBAL_V_ALL, GLOBAL_CDF_ALL
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=f"Images/fixed_{FIXED_IMG_SIZE}padding_wb_cdf_protocol1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 30, 60, 120, 300])
    parser.add_argument("--clip", type=float, default=95.0, help="Percentile to clip global velocity")
    parser.add_argument("--users", type=str, nargs="+", default=[])
    args = parser.parse_args()

    # 1. Build Global CDF first
    GLOBAL_V_ALL, GLOBAL_CDF_ALL = build_global_velocity_cdf(DATA_ROOT, clip_pct=args.clip)
    print(f"[GlobalCDF] Stats: Min=0.00, Max_Clipped={GLOBAL_V_ALL[-1]:.2f}")

    full_out_dir = os.path.join(ROOT, args.out_dir)
    target_users = set(args.users) if args.users else None

    # 2. Iterate and Render
    current_data_path = DATA_ROOT 
    
    if not os.path.exists(current_data_path):
        print(f"Data root not found: {current_data_path}")
        return

    for user in sorted(os.listdir(current_data_path)):
        if target_users and user not in target_users: continue
        user_dir = os.path.join(current_data_path, user)
        if not os.path.isdir(user_dir): continue
            
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            session_name = os.path.splitext(file)[0]

            print(f"[Processing] {user}/{file} (Grayscale & Fixed {FIXED_IMG_SIZE})")
            process_one_session(os.path.join(user_dir, file), user, 
                               session_name, full_out_dir, args.sizes)

if __name__ == "__main__":
    main()