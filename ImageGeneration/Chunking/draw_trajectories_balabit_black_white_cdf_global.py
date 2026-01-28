# -*- coding: utf-8 -*-
"""
Pure Chunking Version (Auto Root Detection) - GLOBAL Velocity CDF Encoding
--------------------------------------------------------------------------
- No sliding window
- Each chunk is chunk_size consecutive events
- Auto-detect project root
- Data folder assumed at: <root>/Data/Balabit-dataset/training_files

Velocity representation:
- movement only
- no click semantics
- WHITE background
- velocity encoded via GLOBAL CDF (plasma)
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

IMG_SIZE = (224, 224)
DPI = 100

# ============================================================
# Global CDF holders
# ============================================================
GLOBAL_V_ALL = None
GLOBAL_CDF_ALL = None


# ============================================================
# Utils
# ============================================================
def _scaled(val, min_val, scale, offset):
    return (val - min_val) * scale + offset


# ============================================================
# Data Cleaning (UNCHANGED)
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
# Build GLOBAL velocity CDF (respect user/session filters)
# ============================================================
def build_global_velocity_cdf(data_dir, target_users=None, target_sessions=None):
    velocities = []

    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users:
            continue

        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue

        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"):
                continue

            session = os.path.splitext(file)[0]
            if target_sessions and session not in target_sessions:
                continue

            df = pd.read_csv(os.path.join(user_dir, file))
            df = clean_and_rename_cols(df)

            xs = df["x"].values
            ys = df["y"].values
            ts = df["time"].values

            if len(xs) < 2:
                continue

            dx = np.diff(xs)
            dy = np.diff(ys)
            dt = np.diff(ts)
            dt[dt <= 0] = 1e-5

            v = np.sqrt(dx ** 2 + dy ** 2) / dt
            v = v[np.isfinite(v)]

            if len(v) > 0:
                velocities.append(v)

    velocities = np.concatenate(velocities)

    ranks = rankdata(velocities, method="average")
    cdf = (ranks - 1) / (len(velocities) - 1 + 1e-8)

    order = np.argsort(velocities)
    return velocities[order], cdf[order]


# ============================================================
# Drawing (UNCHANGED except color source)
# ============================================================
def draw_mouse_chunk(chunk, save_path):
    if len(chunk) < 2:
        return

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.diff(ts)
    dt[dt <= 0] = 1e-5

    velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
    velocity += 1e-5

    # ---- GLOBAL CDF COLOR ----
    v_norm = np.interp(
        velocity,
        GLOBAL_V_ALL,
        GLOBAL_CDF_ALL,
        left=0.0,
        right=1.0
    )

    cmap = plt.get_cmap("plasma")
    seg_colors = cmap(v_norm)

    # ---- spatial scaling (UNCHANGED) ----
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
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


# ============================================================
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size):
    L = len(events)
    n_chunks = L // chunk_size

    for i in range(n_chunks):
        chunk = events[i * chunk_size:(i + 1) * chunk_size]
        save_dir = os.path.join(out_dir, f"event{chunk_size}", user)
        save_path = os.path.join(save_dir, f"{session_name}-{i}.png")
        draw_mouse_chunk(chunk, save_path)


# ============================================================
# Dataset processing with progress printing
# ============================================================
def process_dataset(data_dir, out_dir, sizes, target_users=None, target_sessions=None):
    users = sorted(os.listdir(data_dir))
    total_users = len(users)

    for u_idx, user in enumerate(users, 1):
        if target_users and user not in target_users:
            continue

        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue

        print(f"\n[User {u_idx}/{total_users}] {user}")

        sessions = sorted(os.listdir(user_dir))
        total_sessions = len(sessions)

        for s_idx, file in enumerate(sessions, 1):
            if not file.startswith("session_"):
                continue

            session = os.path.splitext(file)[0]
            if target_sessions and session not in target_sessions:
                continue

            print(f"  -> Session {s_idx}/{total_sessions}: {session}")

            df = clean_and_rename_cols(pd.read_csv(os.path.join(user_dir, file)))
            events = df.to_dict(orient="records")

            for sz in sizes:
                print(f"     [chunk={sz}] rendering...")
                chunk_and_draw(events, out_dir, user, session, sz)


# ============================================================
# Main
# ============================================================
def main():
    global GLOBAL_V_ALL, GLOBAL_CDF_ALL

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Images/Chunk/Balabit_chunks_XY_global_cdf/training"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 15, 30, 60, 120, 300]
    )
    parser.add_argument(
        "--users",
        type=str,
        nargs="+",
        default=[]
    )
    parser.add_argument(
        "--sessions",
        type=str,
        nargs="+",
        default=[]
    )
    args = parser.parse_args()

    target_users = set(args.users) if args.users else None
    target_sessions = set(args.sessions) if args.sessions else None

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[GlobalCDF] Building global velocity CDF...")
    GLOBAL_V_ALL, GLOBAL_CDF_ALL = build_global_velocity_cdf(
        DATA_ROOT,
        target_users,
        target_sessions
    )

    print("[GlobalCDF] Sanity check:")
    print(f"  min     = {GLOBAL_V_ALL[0]:.6f}")
    print(f"  median  = {np.median(GLOBAL_V_ALL):.6f}")
    print(f"  99%ile  = {np.percentile(GLOBAL_V_ALL, 99):.6f}")

    process_dataset(
        DATA_ROOT,
        out_dir,
        sorted(set(args.sizes)),
        target_users,
        target_sessions
    )


if __name__ == "__main__":
    main()
