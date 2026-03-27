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
# SRP (FINAL VERSION)
# ============================================================

def compute_srp(seq, epsilon=0.3):

    coords = seq[:, :2].astype(np.float32)

    # --------------------------------------------------
    # Step 1: RAW distance
    # --------------------------------------------------
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # --------------------------------------------------
    # Step 2: normalize distance
    # --------------------------------------------------
    max_val = dist.max()
    if max_val < 1e-8:
        max_val = 1e-8

    dist = dist / max_val   # ∈ [0,1]

    M = dist.shape[0]

    # --------------------------------------------------
    # Step 3: avg distance
    # --------------------------------------------------
    avg = np.sum(dist, axis=1) / (M - 1 + 1e-8)

    # --------------------------------------------------
    # Step 4: recurrent points
    # --------------------------------------------------
    recurrent = avg < epsilon

    # --------------------------------------------------
    # Step 5: clip distance
    # --------------------------------------------------
    dist_clipped = np.minimum(dist, epsilon)

    # --------------------------------------------------
    # Step 6: SRP
    # --------------------------------------------------
    rp = np.where(
        recurrent[:, None] & recurrent[None, :],
        dist_clipped,
        epsilon
    ).astype(np.float32)

    return rp


def draw_srp(seq, save_path, epsilon):

    rp = compute_srp(seq, epsilon)

    # --------------------------------------------------
    # 映射到灰度
    # --------------------------------------------------
    img = (rp / (epsilon + 1e-8) * 255).astype(np.uint8)

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
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--epsilon", type=float, default=0.5)

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