# -*- coding: utf-8 -*-
"""
Pure Chunking Version (Auto Root Detection) - Velocity Encoding
---------------------------------------------------------------
- No sliding window
- Each chunk is chunk_size consecutive events
- Auto-detect project root using __file__
- Data folder assumed at: <root>/Data/Balabit-dataset/training_files

Velocity representation:
- movement only
- no click semantics
- black background
- velocity encoded via color (CDF-normalized)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# ============================================================
# Automatically detect project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

print(f"[AutoRoot] Project root detected = {ROOT}")
print(f"[AutoRoot] Using data_dir = {DATA_ROOT}")

# ----------------------------
# Drawing Utils
# ----------------------------
IMG_SIZE = (224, 224)
DPI = 100


def _scaled(val, min_val, scale, offset):
    return (val - min_val) * scale + offset


def draw_mouse_chunk(chunk, save_path):
    """
    Velocity-encoded mouse trajectory:
    - movement only
    - black background
    - velocity encoded by color (CDF-normalized)
    """
    if len(chunk) < 2:
        return

    # -----------------------------
    # Extract x, y, time
    # -----------------------------
    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    # -----------------------------
    # Compute velocity
    # -----------------------------
    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.diff(ts)
    dt[dt <= 0] = 1e-5  # numerical safety

    velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
    velocity += 1e-5

    # -----------------------------
    # CDF normalization → [0, 1]
    # -----------------------------
    ranks = rankdata(velocity, method="average")
    v_norm = (ranks - 1) / (len(velocity) - 1 + 1e-8)

    cmap = plt.get_cmap("plasma")
    seg_colors = cmap(v_norm)

    # -----------------------------
    # ORIGINAL spatial scaling (unchanged)
    # -----------------------------
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

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

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(
        figsize=(IMG_SIZE[0] / DPI, IMG_SIZE[1] / DPI), dpi=DPI
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, IMG_SIZE[0])
    ax.set_ylim(IMG_SIZE[1], 0)
    ax.axis("off")

    prev_x_s, prev_y_s = None, None
    color_idx = 0

    for i in range(len(xs)):
        x, y = xs[i], ys[i]

        if x > 10000 or y > 10000:
            continue

        x_s = _scaled(x, min_x, scale, offset_x)
        y_s = _scaled(y, min_y, scale, offset_y)

        if prev_x_s is not None:
            color = seg_colors[color_idx]
            ax.plot(
                [prev_x_s, x_s],
                [prev_y_s, y_s],
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                markerfacecolor=color,
                markeredgewidth=0,
            )
            color_idx += 1

        prev_x_s, prev_y_s = x_s, y_s

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


# ----------------------------
# Data Cleaning
# ----------------------------
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ONLY mouse movement events.
    Discard all click semantics.
    """
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


# ----------------------------
# Pure chunking (No overlap)
# ----------------------------
def chunk_and_draw(events, out_dir, user, session_name, chunk_size):
    L = len(events)
    n_chunks = L // chunk_size
    count = 0

    for i in range(n_chunks):
        start = i * chunk_size
        chunk = events[start : start + chunk_size]

        save_dir = os.path.join(out_dir, f"event{chunk_size}", user)
        save_path = os.path.join(save_dir, f"{session_name}-{i}.png")

        draw_mouse_chunk(chunk, save_path)
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

    print("=" * 50)
    for k in sizes:
        print(f"event{k}: {produced_total[k]} images")
    print(f"TOTAL images: {sum(produced_total.values())}")

    return produced_total


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Velocity-encoded mouse trajectory chunking (movement only)"
    )
    p.add_argument("--out_dir", type=str, default="Images/Chunk/Balabit_chunks_cdf_")
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[10, 15, 30, 60, 120, 300]
    )
    p.add_argument("--users", type=str, nargs="+", default=[])
    p.add_argument("--sessions", type=str, nargs="+", default=[])
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = DATA_ROOT
    out_dir = os.path.join(ROOT, args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    sizes = sorted(set(args.sizes))
    target_users = set(args.users) if args.users else None
    target_sessions = set(args.sessions) if args.sessions else None

    process_dataset(data_dir, out_dir, sizes, target_users, target_sessions)


if __name__ == "__main__":
    main()
