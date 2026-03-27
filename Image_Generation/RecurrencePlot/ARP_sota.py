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

BASE_CHUNK_SIZE = 600
BASE_IMG_SIZE = 600

# ============================================================
# ARP (SOTA VERSION - SAME BRIGHTNESS AS SRP)
# ============================================================

def compute_arp(seq, epsilon=0.3):

    coords = seq[:, :2]

    N = coords.shape[0]


    if N % 2 != 0:
        coords = coords[:-1]
        N -= 1

    half = N // 2

    # --------------------------------------------------
    # Step 1: FULL distance matrix
    # --------------------------------------------------
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # --------------------------------------------------
    # Step 2: normalize（和SRP完全一致）
    # --------------------------------------------------
    dist_norm = dist / (dist.max() + 1e-8)

    # --------------------------------------------------
    # Step 3: global average（和SRP一致）
    # --------------------------------------------------
    avg_dist = np.mean(dist_norm)

    # --------------------------------------------------
    # Step 4: threshold（关键！！！）
    # --------------------------------------------------
    srp_matrix = np.where(dist_norm > avg_dist, epsilon, dist_norm)

    # --------------------------------------------------
    # Step 5: build ARP（只改排版）
    # --------------------------------------------------
    arp = np.zeros((half, half), dtype=np.float32)

    for i in range(half):
        for j in range(half):

            if i < j:
                # 上三角 → 前半段
                arp[i, j] = srp_matrix[i, j]

            elif i > j:
                # 下三角 → 后半段
                arp[i, j] = srp_matrix[i + half, j + half]

            else:
                arp[i, j] = 0.0

    return arp


# ============================================================
# Draw
# ============================================================

def draw_arp(seq, save_path, epsilon, chunk_size):

    arp = compute_arp(seq, epsilon)

    img = (arp * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)


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

                n_chunks = len(events) // chunk_size

                for i in range(n_chunks):

                    seq = events[i*chunk_size:(i+1)*chunk_size]

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_arp(seq, save_path, epsilon, chunk_size)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[BASE_CHUNK_SIZE])
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

    print("\nARP generation finished.")


if __name__ == "__main__":
    main()