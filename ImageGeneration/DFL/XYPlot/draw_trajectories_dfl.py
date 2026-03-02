# -*- coding: utf-8 -*-
"""
Pure Chunking XYPlot Baseline (DFL Version, Unified Style)
----------------------------------------------------------
- Pure spatial geometry
- No timing, no velocity, no encoding
- White background, black trajectory
- Pixel-mass normalized across chunk sizes
- Robust DFL cleaning
- Fully aligned with SRP scripts
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Global Drawing Config
# ============================================================
BASE_EVENT = 60
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 0.5
BASE_MARKERSIZE = 1.0
DPI = 200

# ============================================================
# Scaling Utilities
# ============================================================
def get_img_size(chunk_size, base_event=BASE_EVENT, base_size=BASE_IMG_SIZE):
    scale = chunk_size / base_event
    side = np.sqrt(scale * base_size * base_size)
    return (int(round(side)), int(round(side)))

def get_stroke_params(chunk_size, base_event=BASE_EVENT):
    scale = np.sqrt(chunk_size / base_event)
    linewidth = BASE_LINEWIDTH * scale
    markersize = BASE_MARKERSIZE * scale
    return linewidth, markersize

def _scaled(val, min_val, scale, offset):
    return (val - min_val) * scale + offset

# ============================================================
# Cleaning (DFL Robust)
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame):

    df.columns = [c.strip().lower() for c in df.columns]

    # timestamp detection
    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp": "time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    elif "time" not in df.columns:
        raise RuntimeError(f"Cannot find timestamp column. Columns = {df.columns}")

    # state optional
    if "state" in df.columns:
        df = df[df["state"].str.lower() == "move"].copy()
    else:
        print("      [Warning] No 'state' column found — skipping Move filtering.")

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    df = df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)
    return df

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

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
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
            color="black",
            linewidth=linewidth,
            marker="o",
            markersize=markersize,
            markerfacecolor="black",
            markeredgewidth=0,
            solid_capstyle="round"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)

# ============================================================
# Chunking
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

# ============================================================
# Dataset Processing
# ============================================================
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

        # DFL: all CSV files
        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.lower().endswith(".csv")
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

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str,
                        default="Data/DFL-dataset_raw/training_files")
    parser.add_argument("--out_dir", type=str,
                        default="Images/DFL/XYPlot")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    print("\n[Step] Generating DFL XYPlot Baseline Images")
    process_dataset(data_root, out_dir, sorted(set(args.sizes)))

    print("\n[Done] DFL XYPlot Baseline Generation Complete.")


if __name__ == "__main__":
    main()