# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


'''
Clean up Balabit dataset
'''
def clean_balabit(df):
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })

    # Disregard extreme outliers (logger bug, 16 bit INTMAX)
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()
    df = df[df["state"] == "Move"].copy()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


'''
Generate train/test windows
'''
def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []

    # If the folder contains train, stride = M // 4
    if "train" in data_root.lower():
        stride = chunk_size // 4
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])

    return windows


'''
Generate SRP
Normalization follows the author reply:
for each chunk (length M), x and y are min-max normalized to [0, 1] independently.
'''
def compute_srp(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    x_range = max(max_x - min_x, 1e-8)
    y_range = max(max_y - min_y, 1e-8)

    scale = max(x_range, y_range)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - min_x) / scale
    y_norm = (y - min_y) / scale

    coords_norm = np.stack([x_norm, y_norm], axis=1)

    # pairwise Euclidean distances
    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    m = dist.shape[0]

    # average distance of each point to all other points
    avg_dist = (np.sum(dist, axis=1) - np.diag(dist)) / (m - 1)

    # recurrent points mask
    recurrent = avg_dist < epsilon

    # keep distances only if BOTH points are recurrent
    mask = recurrent[:, None] & recurrent[None, :]

    # recurrence matrix
    rp = np.where(mask, dist, epsilon)

    return rp


'''
Save SRP (largest values to 255, smallest to 0)
'''
def draw_srp(seq, save_path, epsilon):
    rp = compute_srp(seq, epsilon)

    # normalize based on actual values
    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)

    img = ((rp - rp_min) / denom * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


'''
Convert dataset into windows and SRP images
'''
def process_dataset(dataset, data_root, out_dir, sizes, epsilon):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("\n[Phase] Generating SRP...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("\nUser:", user)

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]
            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:
                windows = generate_windows(events, chunk_size, data_root)

                print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")

                for i, seq in enumerate(windows):
                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_srp(seq, save_path, epsilon)


'''Main'''
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

    print("\nDone.")


if __name__ == "__main__":
    main()
