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



# ============================================================
# Dynamic Image Size
# ============================================================


# ============================================================
# SRP (FINAL VERSION)
# ============================================================
'''
def compute_srp(seq, epsilon=0.3):

    coords = seq[:, :2].astype(np.float32)

    # --------------------------------------------------
    # x,y normalization
    # --------------------------------------------------
    x = coords[:, 0]
    y = coords[:, 1]


    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    x_range = max_x - min_x
    y_range = max_y - min_y

    # scale
    scale = max(x_range, y_range)
    if scale < 1e-8:
        scale = 1e-8

    # find x_norm y_norm
    x_norm = (x - min_x) / scale
    y_norm = (y - min_y) / scale
    
    coords_norm = np.stack([x_norm, y_norm], axis=1)

    # --------------------------------------------------
    # Construct distance matrix
    # --------------------------------------------------
    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))   # ∈ [0, √2]

    dist = (dist - dist.min()) / (dist.max() - dist.min())

    M = dist.shape[0]
    
    
    # --------------------------------------------------
    # avg distance
    # --------------------------------------------------
    avg = np.sum(dist, axis=1) / (M - 1)

    # --------------------------------------------------
    # recurrent points
    # --------------------------------------------------
    recurrent = avg < epsilon

    # --------------------------------------------------
    # clip
    # --------------------------------------------------
    #dist_clipped = dist
    dist_clipped = np.minimum(dist, epsilon)

    # --------------------------------------------------
    # SRP
    # --------------------------------------------------
    rp = np.where(
        recurrent[:, None] & recurrent[None, :],
        dist_clipped,
        epsilon
    ).astype(np.float32)
    
    #print("recurrent ratio:", recurrent.mean())
    

    #rp = dist

    return rp
'''
def compute_srp(seq, epsilon=0.3):

    coords = seq[:, :2].astype(np.float32)

    # --------------------------------------------------
    # x,y normalization
    # --------------------------------------------------
    x = coords[:, 0]
    y = coords[:, 1]

    
    coords_norm = np.stack([x, y], axis=1)

    # --------------------------------------------------
    # Construct distance matrix
    # --------------------------------------------------
    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))  

    dist = (dist - dist.min()) / (dist.max() - dist.min())


    M = dist.shape[0]
    #print(dist)
    
    
    # --------------------------------------------------
    # avg distance
    # --------------------------------------------------

    sum_dist = np.sum(dist, axis=1) - np.diag(dist)
    avg = sum_dist / (M - 1)


    # --------------------------------------------------
    # recurrent points
    # --------------------------------------------------
    recurrent = avg < epsilon

    # --------------------------------------------------
    # clip
    # --------------------------------------------------
    #dist_clipped = dist
    dist_clipped = np.minimum(dist, epsilon)

    # --------------------------------------------------
    # SRP
    # --------------------------------------------------
    rp = np.where(
        recurrent[:, None] & recurrent[None, :],
        dist_clipped,
        epsilon
    ).astype(np.float32)
    
    #print("recurrent ratio:", recurrent.mean())
    

    #rp = dist

    return rp


def draw_srp(seq, save_path, epsilon):

    rp = compute_srp(seq, epsilon)

    # --------------------------------------------------
    # --------------------------------------------------
    #img = (rp * 255).astype(np.uint8)
    #print(img.shape)

    img = (rp / epsilon * 255).astype(np.uint8)

    img = img.astype(np.uint8)

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

    return df


def clean_chaoshen(df):

    df = df.rename(columns={
        "X":"x",
        "Y":"y",
        "Timestamp":"time",
        "EventName":"event"
    })

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
# Session Detection
# ============================================================

def get_session_files(dataset, user_dir):

    files = os.listdir(user_dir)

    sessions = []

    for f in files:

        path = os.path.join(user_dir, f)

        if os.path.isfile(path):
            sessions.append(f)

    return sorted(sessions)

# ============================================================
# Process Dataset
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes, epsilon):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue
    
        print("\n------------------------------")
        print("\nUser:", user)

        session_files = get_session_files(dataset, user_dir)
        print("Sessions found:", len(session_files))

        for file in session_files:

            session = os.path.splitext(file)[0]
            path = os.path.join(user_dir, file)

            print("   Session:", session)

            df = pd.read_csv(
                path,
                sep=",",
                engine="python",
                header=0
            )

            if dataset == "balabit":
                df = clean_balabit(df)

            elif dataset == "chaoshen":
                df = clean_chaoshen(df)

            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:

                if "train" in data_root.lower():
                    stride = chunk_size // 4
                    #stride = chunk_size 
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