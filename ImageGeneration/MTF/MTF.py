# -*- coding: utf-8 -*-
"""
Pure Chunking Version - Markov Transition Field (MTF)
----------------------------------------------------
- No sliding window
- Each chunk is chunk_size consecutive events
- Auto-detect project root
- Velocity-based MTF representation
- Movement only
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Automatically detect project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

print(f"[AutoRoot] Project root detected = {ROOT}")
print(f"[AutoRoot] Using data_dir = {DATA_ROOT}")

# ============================================================
# Image settings
# ============================================================
IMG_SIZE = (448, 448)
DPI = 200

# ============================================================
# MTF parameters
# ============================================================
N_BINS = 8        # quantile bins (reviewer-safe default)
EPS = 1e-8

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
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["x", "y", "time"])

    return df


# ============================================================
# Velocity computation
# ============================================================
def compute_velocity(events):
    xs = np.array([e["x"] for e in events])
    ys = np.array([e["y"] for e in events])
    ts = np.array([e["time"] for e in events])

    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.diff(ts)

    valid = dt > 0
    v = np.zeros(len(dx))
    v[valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) / (dt[valid] + EPS)

    return v


# ============================================================
# MTF computation
# ============================================================
def compute_mtf(v, n_bins=N_BINS):
    """
    v: velocity sequence, shape (T,)
    returns: MTF matrix shape (T, T)
    """
    T = len(v)
    if T < 2:
        return None

    # ---- quantile binning ----
    bins = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    states = np.digitize(v, bins[1:-1], right=True)

    # ---- transition matrix ----
    P = np.zeros((n_bins, n_bins), dtype=np.float32)
    for i in range(len(states) - 1):
        P[states[i], states[i + 1]] += 1

    row_sum = P.sum(axis=1, keepdims=True)
    P = P / (row_sum + EPS)

    # ---- MTF field ----
    mtf = np.zeros((T, T), dtype=np.float32)
    for i in range(T):
        for j in range(T):
            mtf[i, j] = P[states[i], states[j]]

    return mtf


# ============================================================
# Draw MTF image
# ============================================================
def draw_mtf(chunk, save_path):
    v = compute_velocity(chunk)
    mtf = compute_mtf(v)

    if mtf is None:
        return

    fig, ax = plt.subplots(
        figsize=(IMG_SIZE[0] / DPI, IMG_SIZE[1] / DPI),
        dpi=DPI
    )

    ax.imshow(mtf, cmap="jet", interpolation="nearest")
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)


# ============================================================
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size):
    L = len(events)
    n_chunks = L // chunk_size
    count = 0

    for i in range(n_chunks):
        chunk = events[i * chunk_size:(i + 1) * chunk_size]

        save_dir = os.path.join(out_dir, f"event{chunk_size}", user)
        save_path = os.path.join(save_dir, f"{session_name}-{i}.png")

        draw_mtf(chunk, save_path)
        count += 1

    return count


def process_one_session(path, user, session_name, out_dir, sizes):
    df = pd.read_csv(path)
    df = clean_and_rename_cols(df)
    events = df.to_dict(orient="records")

    produced = {}
    for size in sizes:
        produced[size] = chunk_and_draw(
            events, out_dir, user, session_name, size
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
            print(f"[Process] {user}/{file}")

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


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="MTF image generation")
    p.add_argument(
        "--out_dir",
        type=str,
        default="Images/Chunk/Balabit_chunks_MTF"
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[15,30,60,120,300]
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
