# -*- coding: utf-8 -*-
"""
Pure Chunking Version (Auto Root Detection) - GLOBAL Velocity CDF Encoding
--------------------------------------------------------------------------
- No sliding window
- Each chunk is chunk_size consecutive events
- Velocity representation: Movement only
- WHITE background, Velocity via GLOBAL CDF (plasma colormap)
- CLIPPING: Velocity distribution clipped at a specified percentile (e.g., 95th) 
  to ensure high color contrast and remove sensor noise/outliers.
"""
'''
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# ============================================================
# Automatically detect project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Default to training files for CDF building; can be modified via DATA_ROOT logic
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol2/imposter")

print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Global Anchors (DO NOT CHANGE SEMANTICS)
# ============================================================
BASE_EVENT = 60
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 0.5
BASE_MARKERSIZE = 1.0
DPI = 200

# ============================================================
# Global CDF holders
# ============================================================
GLOBAL_V_ALL = None
GLOBAL_CDF_ALL = None

# ============================================================
# Scaling Utils
# ============================================================
def get_img_size(chunk_size, base_event=BASE_EVENT, base_size=BASE_IMG_SIZE):
    """Scales image canvas size based on the number of events in a chunk."""
    scale = chunk_size / base_event
    side = np.sqrt(scale * base_size * base_size)
    side = int(round(side))
    return (side, side)

def get_stroke_params(chunk_size, base_event=BASE_EVENT):
    """Scales line width and marker size based on chunk size."""
    scale = np.sqrt(chunk_size / base_event)
    lw = BASE_LINEWIDTH * scale
    ms = BASE_MARKERSIZE * scale
    return lw, ms

def _scaled(val, min_val, scale, offset):
    """Utility for min-max scaling with offset."""
    return (val - min_val) * scale + offset

# ============================================================
# Data Cleaning
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns, filters 'Move' states, and removes 65535 error values."""
    df = df.rename(
        columns={
            "client timestamp": "time",
            "x": "x",
            "y": "y",
            "state": "state",
        }
    )

    # 1. Keep only movement events
    df = df[df["state"] == "Move"].copy()
    
    # 2. Filter outlier coordinates (65535 is a common sensor error in Balabit)
    # This prevents massive jumps in velocity calculations
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    
    mask_valid = (df["x"] < 65535) & (df["y"] < 65535)
    df = df[mask_valid].dropna(subset=["x", "y", "time"]).reset_index(drop=True)
    
    df["state"] = "movement"
    return df

# ============================================================
# Build GLOBAL velocity CDF
# ============================================================
def build_global_velocity_cdf(data_dir, target_users=None, target_sessions=None, clip_pct=95.0):
    """Iterates dataset to build a cumulative distribution of velocities with clipping."""
    velocities = []

    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users:
            continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue

        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            session = os.path.splitext(file)[0]
            if target_sessions and session not in target_sessions: continue

            df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
            if len(df) < 2: continue

            xs, ys, ts = df["x"].values, df["y"].values, df["time"].values
            dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(ts)
            
            # Prevent division by zero
            dt[dt <= 0] = 1e-5
            v = np.sqrt(dx**2 + dy**2) / dt
            v = v[np.isfinite(v)]
            if len(v) > 0: velocities.append(v)

    if not velocities:
        return np.array([0, 1000]), np.array([0, 1])

    velocities = np.concatenate(velocities)

    # --- Clipping Logic ---
    # Determine the threshold for the specified percentile (e.g., 95%)
    # This ensures the 'plasma' color range is used for human movement, not outliers.
    v_upper_bound = np.percentile(velocities, clip_pct)
    velocities = velocities[velocities <= v_upper_bound]
    
    # Calculate ranks and map to 0-1 range
    ranks = rankdata(velocities, method="average")
    cdf = (ranks - 1) / (len(velocities) - 1 + 1e-8)

    order = np.argsort(velocities)
    return velocities[order], cdf[order]

# ============================================================
# Drawing (Velocity CDF + scaled geometry)
# ============================================================
def draw_mouse_chunk(chunk, save_path, chunk_size):
    """Renders a single chunk of mouse movement into a PNG image."""
    if len(chunk) < 2: return

    IMG_SIZE = get_img_size(chunk_size)
    linewidth, markersize = get_stroke_params(chunk_size)

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(ts)
    dt[dt <= 0] = 1e-5
    velocity = np.sqrt(dx**2 + dy**2) / dt

    # Map velocity to 0-1 using Global CDF
    # Values outside the global range are clipped to 0.0 or 1.0
    v_norm = np.interp(
        velocity,
        GLOBAL_V_ALL,
        GLOBAL_CDF_ALL,
        left=0.0,
        right=1.0
    )

    cmap = plt.get_cmap("plasma")
    seg_colors = cmap(v_norm)

    # Calculate padding and aspect-ratio scaling
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

    # Segmented plotting to apply color per velocity point
    for i in range(len(xs) - 1):
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
# Chunking & Dataset Processing
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size):
    """Splits session into chunks and calls the drawing function."""
    n_chunks = len(events) // chunk_size
    for i in range(n_chunks):
        chunk = events[i * chunk_size:(i + 1) * chunk_size]
        save_path = os.path.join(out_dir, f"event{chunk_size}", user, f"{session_name}-{i}.png")
        draw_mouse_chunk(chunk, save_path, chunk_size)

def process_dataset(data_dir, out_dir, sizes, target_users=None, target_sessions=None):
    """Walks through the dataset directory to process files."""
    users = sorted(os.listdir(data_dir))
    for u_idx, user in enumerate(users, 1):
        if target_users and user not in target_users: continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir): continue
        print(f"\n[User {u_idx}/{len(users)}] {user}")
        
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): continue
            session = os.path.splitext(file)[0]
            if target_sessions and session not in target_sessions: continue
            
            print(f"  -> Session: {session}")
            df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
            events = df.to_dict(orient="records")
            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz)

# ============================================================
# Main Execution
# ============================================================
def main():
    global GLOBAL_V_ALL, GLOBAL_CDF_ALL
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="Images/XYPLot_cdf_protocol2/imposter")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--clip", type=float, default=100.0, help="Percentile to clip global velocity")
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    
    # 1. Build Global CDF with Clipping
    print(f"[GlobalCDF] Building CDF (Clipping at {args.clip}%)...")
    GLOBAL_V_ALL, GLOBAL_CDF_ALL = build_global_velocity_cdf(DATA_ROOT, clip_pct=args.clip)

    print(f"[GlobalCDF] Stats: Min={GLOBAL_V_ALL[0]:.2f}, Max_Clipped={GLOBAL_V_ALL[-1]:.2f}")

    # 2. Start Rendering Images
    process_dataset(DATA_ROOT, out_dir, sorted(set(args.sizes)))

if __name__ == "__main__":
    main()
'''

# -*- coding: utf-8 -*-
"""
XYPlot — Velocity Encoding (Protocol-Safe, Raw Distribution)
--------------------------------------------------------------
- Uses RAW velocity distribution (training-only)
- Percentile clipping applied during encoding
- CDF built at runtime from clipped training distribution
- No data leakage
- Unified CLI + printing style
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# ============================================================
# ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Base Config
# ============================================================
BASE_EVENT = 60
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 0.5
BASE_MARKERSIZE = 1.0
DPI = 200

GLOBAL_V_RAW = None
GLOBAL_V_CDF = None

# ============================================================
# Scaling Utils
# ============================================================
def get_img_size(chunk_size):
    scale = chunk_size / BASE_EVENT
    side = np.sqrt(scale * BASE_IMG_SIZE * BASE_IMG_SIZE)
    return (int(round(side)), int(round(side)))

def get_stroke_params(chunk_size):
    scale = np.sqrt(chunk_size / BASE_EVENT)
    return BASE_LINEWIDTH * scale, BASE_MARKERSIZE * scale

def _scaled(val, min_val, scale, offset):
    return (val - min_val) * scale + offset

# ============================================================
# Data Cleaning
# ============================================================
def clean_and_rename_cols(df):

    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state",
    })

    df = df[df["state"] == "Move"].copy()

    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask_valid = (df["x"] < 65535) & (df["y"] < 65535)
    df = df[mask_valid].dropna(subset=["x", "y", "time"]).reset_index(drop=True)

    return df

# ============================================================
# Load RAW Distribution
# ============================================================
def load_raw_velocity_distribution(path):

    data = np.load(path)
    velocities = data["velocities"]

    print(f"[Velocity] Loaded RAW distribution from {path}")
    print(f"[Velocity] Total samples: {len(velocities)}")
    print(f"[Velocity] Min: {velocities.min():.6f}")
    print(f"[Velocity] Max: {velocities.max():.6f}")

    return velocities

# ============================================================
# Build CDF from RAW distribution with clip
# ============================================================
def build_runtime_cdf(raw_velocities, clip_pct):

    print(f"\n[Velocity] Building runtime CDF (clip={clip_pct}%)")

    v_upper = np.percentile(raw_velocities, clip_pct)
    v_clipped = raw_velocities[raw_velocities <= v_upper]

    ranks = rankdata(v_clipped, method="average")
    cdf = (ranks - 1) / (len(v_clipped) - 1 + 1e-8)

    order = np.argsort(v_clipped)

    v_sorted = v_clipped[order]
    cdf_sorted = cdf[order]

    print(f"[Velocity] Runtime Min: {v_sorted.min():.6f}")
    print(f"[Velocity] Runtime Max: {v_sorted.max():.6f}")
    print(f"[Velocity] Runtime Samples: {len(v_sorted)}")

    return v_sorted, cdf_sorted

# ============================================================
# Drawing
# ============================================================
def draw_mouse_chunk(chunk, save_path, chunk_size):

    if len(chunk) < 2:
        return

    IMG_SIZE = get_img_size(chunk_size)
    linewidth, markersize = get_stroke_params(chunk_size)

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.maximum(np.diff(ts), 1e-5)

    velocity = np.sqrt(dx**2 + dy**2) / dt

    v_norm = np.interp(
        velocity,
        GLOBAL_V_CDF[0],
        GLOBAL_V_CDF[1],
        left=0.0,
        right=1.0
    )

    cmap = plt.get_cmap("plasma")
    seg_colors = cmap(v_norm)

    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    pad_x = range_x * 0.05
    pad_y = range_y * 0.05

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    range_x = max_x - min_x
    range_y = max_y - min_y

    scale = min(IMG_SIZE[0] / range_x, IMG_SIZE[1] / range_y)
    offset_x = (IMG_SIZE[0] - range_x * scale) / 2.0
    offset_y = (IMG_SIZE[1] - range_y * scale) / 2.0

    fig, ax = plt.subplots(
        figsize=(IMG_SIZE[0] / DPI, IMG_SIZE[1] / DPI),
        dpi=DPI
    )

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.set_xlim(0, IMG_SIZE[0])
    ax.set_ylim(IMG_SIZE[1], 0)
    ax.axis("off")

    for i in range(len(xs) - 1):

        x1 = _scaled(xs[i], min_x, scale, offset_x)
        y1 = _scaled(ys[i], min_y, scale, offset_y)
        x2 = _scaled(xs[i+1], min_x, scale, offset_x)
        y2 = _scaled(ys[i+1], min_y, scale, offset_y)

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=seg_colors[i],
            linewidth=linewidth,
            marker="o",
            markersize=markersize,
            markerfacecolor=seg_colors[i],
            markeredgewidth=0,
            solid_capstyle="round"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ============================================================
# Dataset Processing
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size):

    n_chunks = len(events) // chunk_size
    print(f"      [ChunkSize={chunk_size}] Total Chunks = {n_chunks}")

    for i in range(n_chunks):

        chunk = events[i * chunk_size:(i + 1) * chunk_size]

        save_path = os.path.join(
            out_dir,
            f"event{chunk_size}",
            user,
            f"{session_name}-{i}.png"
        )

        draw_mouse_chunk(chunk, save_path, chunk_size)

        if (i + 1) % 50 == 0 or (i + 1) == n_chunks:
            print(f"         -> Chunk {i+1}/{n_chunks} done")

def process_dataset(data_root, out_dir, sizes):

    users = sorted(os.listdir(data_root))
    total_users = len(users)

    print(f"\n[Dataset] Total Users = {total_users}")

    for u_idx, user in enumerate(users, 1):

        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n====================================================")
        print(f"[User {u_idx}/{total_users}] Processing: {user}")
        print("====================================================")

        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.startswith("session_")
        ])

        total_sessions = len(session_files)

        for s_idx, file in enumerate(session_files, 1):

            session = os.path.splitext(file)[0]
            print(f"\n   [Session {s_idx}/{total_sessions}] {session}")

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

            events = df.to_dict(orient="records")
            print(f"      Total Events = {len(events)}")

            for sz in sizes:
                chunk_and_draw(events, out_dir, user, session, sz)

# ============================================================
# Main
# ============================================================
def main():

    global GLOBAL_V_RAW, GLOBAL_V_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--velocity_dist", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--clip", type=float, default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    dist_path = os.path.join(ROOT, args.velocity_dist)

    GLOBAL_V_RAW = load_raw_velocity_distribution(dist_path)
    GLOBAL_V_CDF = build_runtime_cdf(GLOBAL_V_RAW, args.clip)

    print("\n[Step] Generating XYPlot Velocity Images")
    process_dataset(data_root, out_dir, sorted(set(args.sizes)))

    print("\n[Done] XYPlot Velocity Generation Complete.")

if __name__ == "__main__":
    main()