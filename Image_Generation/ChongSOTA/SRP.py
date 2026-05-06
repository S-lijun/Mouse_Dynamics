# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# Parameters
# ============================================================

TIME_THRESHOLD = 1.0   # seconds（与 XYPlot 一致）
TARGET_SIZE = 224      # 输出方形边长（与 SOTA/SRP_pair 常用尺寸一致时可改）
DEFAULT_EPSILON = 0.3  # 与 SOTA/SRP_pair.py 默认一致


# ============================================================
# Trajectory Length
# ============================================================

def compute_length(seq):

    if len(seq) < 2:
        return 0.0

    length = 0.0

    for i in range(len(seq)-1):

        dx = seq[i+1]["x"] - seq[i]["x"]
        dy = seq[i+1]["y"] - seq[i]["y"]

        length += math.sqrt(dx*dx + dy*dy)

    return length


# ============================================================
# Time Difference Split
# ============================================================

def split_by_time(events):

    sequences = []
    current = []

    for i in range(len(events)):

        if i == 0:
            current.append(events[i])
            continue

        dt = events[i]["time"] - events[i-1]["time"]

        if dt > TIME_THRESHOLD:
            sequences.append(current)
            current = []

        current.append(events[i])

    if current:
        sequences.append(current)

    return sequences


# ============================================================
# Merge Short Sequences
# ============================================================

def merge_sequences(sequences, min_length):

    merged = []
    i = 0

    while i < len(sequences):

        current = sequences[i]

        while compute_length(current) < min_length and i+1 < len(sequences):

            current = current + sequences[i+1]
            i += 1

        merged.append(current)
        i += 1

    return merged


# ============================================================
# Pair-wise SRP（与 SOTA/SRP_pair.py 相同：逐轴局部归一化 + epsilon 截断）
# ============================================================

def _seq_to_xy_array(seq):
    """与 XYPlot.draw_sequence 一致：seq 为 {'x','y',...} 字典列表；也支持 (n,2+) 的 ndarray。"""
    if isinstance(seq, np.ndarray):
        return np.asarray(seq[:, :2], dtype=np.float64)
    return np.array([[float(e["x"]), float(e["y"])] for e in seq], dtype=np.float64)


def compute_srp_pair(seq, epsilon):
    coords = _seq_to_xy_array(seq).astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_norm = (x - x_min) / max(x_max - x_min, 1e-8)
    y_norm = (y - y_min) / max(y_max - y_min, 1e-8)

    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp


def draw_srp(seq, save_path, epsilon):
    """与 SOTA/SRP_pair.draw_srp 一致：RP 最小值映射 0、最大映射 255；再缩放到 TARGET_SIZE。"""
    if len(seq) < 2:
        return

    rp = compute_srp_pair(seq, epsilon)

    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)

    img = ((rp - rp_min) / denom * 255).astype(np.uint8)

    if img.shape[0] != TARGET_SIZE or img.shape[1] != TARGET_SIZE:
        img = cv2.resize(
            img,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


# ============================================================
# Cleaning (unchanged)
# ============================================================

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp":"time",
        "x":"x",
        "y":"y",
        "state":"state"
    })

    df = df[df["state"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_chaoshen(df):

    df = df.rename(columns={
        "X":"x",
        "Y":"y",
        "Timestamp":"time",
        "EventName":"event"
    })

    df = df[df["event"] == "Move"]

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_dfl(df):

    df.columns = [c.strip().lower() for c in df.columns]

    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp":"time"})

    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"time"})

    if "state" in df.columns:
        df = df[df["state"].str.lower() == "move"]

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


# ============================================================
# Dataset Processing (MODIFIED)
# ============================================================

def process_dataset(dataset, data_root, out_dir, epsilon):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        session_files = sorted(os.listdir(user_dir))

        for file in session_files:

            path = os.path.join(user_dir, file)

            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]

            print("   Session:", session)

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df.to_dict("records")

            print("      Events:", len(events))
            if len(events) < 2:
                continue

            # ========================================================
            # 1. Time Split
            # ========================================================

            sequences = split_by_time(events)

            print("      After split:", len(sequences))

            # ========================================================
            # 2. Compute min_length (per session)
            # ========================================================

            xs = np.array([float(e["x"]) for e in events], dtype=np.float64)
            ys = np.array([float(e["y"]) for e in events], dtype=np.float64)
            session_width = float(np.max(xs))
            session_height = float(np.max(ys))
            print("      Session max x/y:", session_width, session_height)

            min_length = session_width

            # ========================================================
            # 3. Merge
            # ========================================================

            sequences = merge_sequences(sequences, min_length)

            print("      After merge:", len(sequences))

            # ========================================================
            # 4. Draw
            # ========================================================

            for i, seq in enumerate(sequences):

                save_path = os.path.join(
                    out_dir,
                    "Chong",
                    user,
                    f"{session}-{i}.png"
                )

                draw_srp(seq, save_path, epsilon)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit","chaoshen","dfl"])

    parser.add_argument("--data_root",
                        required=True)

    parser.add_argument("--out_dir",
                        required=True)

    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="pair-wise",
    )

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.epsilon,
    )

    print("\nTimeDiff image generation finished.")


if __name__ == "__main__":
    main()