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
# Config
# ============================================================

BASE_CHUNK_SIZE = 300
BASE_IMG_SIZE = 300

# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# SRP (PAPER VERSION)
# ============================================================

def compute_srp(seq, epsilon=0.3):

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    dist_norm = dist / (dist.max() + 1e-8)

    # --------------------------------------------------
    # LOCAL average per point 
    # --------------------------------------------------
    avg_per_point = np.sum(dist_norm, axis=1) / (dist_norm.shape[1] - 1)

    # reshape for broadcasting
    avg_matrix = avg_per_point[:, None]

    # --------------------------------------------------
    # recurrence (row-wise threshold)
    # --------------------------------------------------
    rp = np.where(dist_norm > avg_matrix, epsilon, dist_norm).astype(np.float32)

    return rp


# ============================================================
# Draw
# ============================================================

def draw_srp(seq, save_path, epsilon, chunk_size):

    rp = compute_srp(seq, epsilon)

    img_size = get_dynamic_image_size(chunk_size)

    img = (rp * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_NEAREST)

    # 保持 (0,0) 在左下
    img = np.flipud(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)


# ============================================================
# Sliding Window
# ============================================================

def generate_windows(events, chunk_size, stride):

    windows = []

    for start in range(0, len(events) - chunk_size + 1, stride):
        end = start + chunk_size
        windows.append(events[start:end])

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

    df = df[df["state"] == "Move"].copy()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["x", "y", "time"])

    df = df[(df["x"] < 1e4) & (df["y"] < 1e4)]

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

                # ------------------------------------------
                # Sliding Strategy 
                # ------------------------------------------
                if "train" in data_root.lower():
                    stride = chunk_size // 4
                else:
                    stride = chunk_size

                windows = generate_windows(events, chunk_size, stride)

                print(f"      chunk_size={chunk_size}, stride={stride}, windows={len(windows)}")

                for i, seq in enumerate(windows):

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_srp(seq, save_path, epsilon, chunk_size)


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

    print("\nSRP generation finished.")


if __name__ == "__main__":
    main()