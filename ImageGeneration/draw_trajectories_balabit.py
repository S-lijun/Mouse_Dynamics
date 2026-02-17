# -*- coding: utf-8 -*-
"""
Pure Chunking Version (Auto Root Detection)
-------------------------------------------
- No sliding window
- Each chunk is chunk_size consecutive events
- Auto-detect project root using __file__
- Data folder assumed at: <root>/Data/Balabit-dataset/training_files

Baseline definition:
- Only mouse movement events
- Pure spatial geometry
- No timing, no order, no click semantics
- White background, black trajectory
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Automatically detect project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")

print(f"[AutoRoot] Project root detected = {ROOT}")
print(f"[AutoRoot] Using data_dir = {DATA_ROOT}")

# ============================================================
# Global Drawing Config (ANCHORS)
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
    """
    event15 -> 224 x 224
    eventK  -> sqrt((K / 15) * 224 * 224)
    """
    scale = chunk_size / base_event
    side = np.sqrt(scale * base_size * base_size)
    side = int(round(side))
    return (side, side)


def get_stroke_params(chunk_size, base_event=BASE_EVENT):
    """
    Ensure each event contributes the same pixel mass
    """
    scale = np.sqrt(chunk_size / base_event)
    linewidth = BASE_LINEWIDTH * scale
    markersize = BASE_MARKERSIZE * scale
    return linewidth, markersize


def _scaled(val, min_val, scale, offset):
    return (val - min_val) * scale + offset


# ============================================================
# Drawing Function
# ============================================================
def draw_mouse_chunk(chunk, save_path, chunk_size):
    """
    Draw pure spatial mouse trajectory:
    - only movement
    - white background
    - black line
    - pixel-mass normalized across chunk sizes
    """
    if len(chunk) == 0:
        return

    IMG_SIZE = get_img_size(chunk_size)
    linewidth, markersize = get_stroke_params(chunk_size)
    

    x_coords = np.array([float(e["x"]) for e in chunk])
    y_coords = np.array([float(e["y"]) for e in chunk])

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    padding_ratio = 0.05
    range_x = max_x - min_x
    range_y = max_y - min_y
    pad_x = range_x * padding_ratio if range_x > 0 else 1.0
    pad_y = range_y * padding_ratio if range_y > 0 else 1.0

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

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

    prev_x, prev_y = None, None

    for event in chunk:
        x = float(event["x"])
        y = float(event["y"])

        if x > 10000 or y > 10000:
            continue

        x_s = _scaled(x, min_x, scale, offset_x)
        y_s = _scaled(y, min_y, scale, offset_y)

        if prev_x is not None:
            ax.plot(
                [prev_x, x_s],
                [prev_y, y_s],
                color="black",
                linewidth=linewidth,
                marker="o",
                markersize=markersize,
                markerfacecolor="black",
                markeredgewidth=0,
            )

        prev_x, prev_y = x_s, y_s

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


# ============================================================
# Data Cleaning
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "client timestamp": "time",
            "x": "x",
            "y": "y",
            "state": "state",
        }
    )

    df = df[df["state"] == "Move"].copy()
    df["state"] = "movement"

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    df = df.dropna(subset=["x", "y", "time"])
    return df


# ============================================================
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size):
    L = len(events)
    n_chunks = L // chunk_size
    count = 0

    for i in range(n_chunks):
        chunk = events[i * chunk_size : (i + 1) * chunk_size]
        save_dir = os.path.join(out_dir, f"event{chunk_size}", user)
        save_path = os.path.join(save_dir, f"{session_name}-{i}.png")

        draw_mouse_chunk(chunk, save_path, chunk_size)
        count += 1

    return count


def process_one_session(path, user, session_name, out_dir, sizes):
    df = pd.read_csv(path)
    df = clean_and_rename_cols(df)
    events = df.to_dict(orient="records")

    produced = {}
    for chunk_size in sizes:
        produced[chunk_size] = chunk_and_draw(
            events, out_dir, user, session_name, chunk_size
        )
    return produced


def process_dataset(data_dir, out_dir, sizes, target_users=None, target_sessions=None):
    produced_total = {k: 0 for k in sizes}

    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users:
            continue

        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue

        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"):
                continue

            session_name = os.path.splitext(file)[0]
            if target_sessions and session_name not in target_sessions:
                continue

            path = os.path.join(user_dir, file)

            
            print(f"[Process] {user}/{file} for chunk sizes {sizes}")

            produced = process_one_session(
                path, user, session_name, out_dir, sizes
            )
            for k, v in produced.items():
                produced_total[k] += v



# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Pixel-mass normalized mouse trajectory chunking"
    )
    p.add_argument("--out_dir", type=str, default="Images/XYPlot_protocol1")
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[60]
    )
    p.add_argument("--users", type=str, nargs="+", default=[])
    p.add_argument("--sessions", type=str, nargs="+", default=[])
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    sizes = sorted(set(args.sizes))
    target_users = set(args.users) if args.users else None
    target_sessions = set(args.sessions) if args.sessions else None

    process_dataset(DATA_ROOT, out_dir, sizes, target_users, target_sessions)


if __name__ == "__main__":
    main()
