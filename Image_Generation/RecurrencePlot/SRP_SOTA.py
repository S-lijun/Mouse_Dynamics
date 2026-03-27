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
# SRP (SOTA VERSION)
# ============================================================

def draw_distance_matrix(seq, save_path):

    coords = seq[:, :2].astype(np.float32)

    # --------------------------------------------------
    # Step 1: normalize x,y（分别）
    # --------------------------------------------------
    x = coords[:, 0]
    y = coords[:, 1]

    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    coords = np.stack([x, y], axis=1)

    # --------------------------------------------------
    # Step 2: distance matrix（raw）
    # --------------------------------------------------
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))   # ∈ [0, √2]

    # --------------------------------------------------
    # Step 3: 用固定尺度 √2 映射（关键）
    # --------------------------------------------------
    img = (dist / np.sqrt(2) * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

    # Debug print
    print("dist min:", dist.min())
    print("dist max:", dist.max())


# ============================================================
# Sliding Window（核心）
# ============================================================

def generate_windows(events, chunk_size, stride):

    windows = []

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

                # --------------------------------------------------
                # Sliding strategy（关键）
                # --------------------------------------------------
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

                    draw_distance_matrix(seq, save_path.replace(".png", "_dist.png"))


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