# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# Config
# ============================================================

BASE_CHUNK_SIZE = 300
BASE_IMG_SIZE = 300

# ⭐ 全局收集 ratio
all_ratios = []

# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# SRP
# ============================================================

def compute_srp(seq, epsilon=0.3):

    coords = seq[:, :2].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    x_range = max_x - min_x
    y_range = max_y - min_y

    scale = max(x_range, y_range)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - min_x) / scale
    y_norm = (y - min_y) / scale

    coords_norm = np.stack([x_norm, y_norm], axis=1)

    # ---------------- distance ----------------
    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    dist = dist / np.sqrt(2)

    M = dist.shape[0]

    # ---------------- avg ----------------
    avg = np.sum(dist, axis=1) / (M - 1 + 1e-8)

    # ---------------- recurrent ----------------
    recurrent = avg < epsilon

    # ⭐ 计算 ratio
    ratio = recurrent.mean()

    # ---------------- clip ----------------
    dist_clipped = np.minimum(dist, epsilon)

    rp = np.where(
        recurrent[:, None] & recurrent[None, :],
        dist_clipped,
        epsilon
    ).astype(np.float32)

    return rp, ratio


def draw_srp(seq, save_path, epsilon):

    global all_ratios

    rp, ratio = compute_srp(seq, epsilon)

    # ⭐ 收集 ratio
    all_ratios.append(ratio)

    img = (rp / epsilon * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


# ============================================================
# Sliding Window
# ============================================================

def generate_windows(events, chunk_size, stride):

    windows = []

    if len(events) < chunk_size:
        return windows

    for start in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[start:start + chunk_size])

    return windows


# ============================================================
# Cleaning
# ============================================================

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["x", "y", "time"])

    return df


# ============================================================
# Process Dataset
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes, epsilon):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\nUser:", user)

        for file in sorted(os.listdir(user_dir)):

            session = os.path.splitext(file)[0]
            path = os.path.join(user_dir, file)

            print("   Session:", session)

            df = pd.read_csv(path)
            df = clean_balabit(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:

                if "train" in data_root.lower():
                    stride = chunk_size // 4
                else:
                    stride = chunk_size

                windows = generate_windows(events, chunk_size, stride)

                print(f"      chunk={chunk_size}, stride={stride}, windows={len(windows)}")

                for i, seq in enumerate(windows):

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_srp(seq, save_path, epsilon)


# ============================================================
# Plot ratio distribution
# ============================================================

def analyze_ratios(out_dir):

    ratios = np.array(all_ratios)

    print("\n==============================")
    print("[Recurrence Ratio Stats]")
    print("Count:", len(ratios))
    print("Mean :", ratios.mean())
    print("Std  :", ratios.std())
    print("Min  :", ratios.min())
    print("Max  :", ratios.max())
    print("==============================")

    # 保存 raw
    np.save(os.path.join(out_dir, "recurrence_ratios.npy"), ratios)

    # ⭐ 画 histogram
    plt.figure()
    plt.hist(ratios, bins=50)
    plt.title("Recurrence Ratio Distribution")
    plt.xlabel("Ratio")
    plt.ylabel("Count")
    plt.grid()

    save_path = os.path.join(out_dir, "recurrence_ratio_distribution.png")
    plt.savefig(save_path)
    plt.show()

    print("\nSaved plot to:", save_path)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--epsilon", type=float, default=0.3)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        args.epsilon
    )

    # ⭐ 最关键：分析 ratio
    analyze_ratios(out_dir)

    print("\nSRP generation finished.")


if __name__ == "__main__":
    main()