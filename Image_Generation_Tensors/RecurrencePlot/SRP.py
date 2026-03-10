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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
print("[ROOT]", ROOT)

# ============================================================
# Config
# ============================================================

BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224

# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# RP computation
# ============================================================

def compute_rp(seq, percentile=95):

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, percentile)

    rec = np.where(dist <= eps, dist, eps).astype(np.float32)

    if rec.max() > rec.min():
        rec = (rec - rec.min()) / (rec.max() - rec.min())

    return rec


def rp_to_uint8(seq, percentile, chunk_size):

    rp = compute_rp(seq, percentile)

    img_size = get_dynamic_image_size(chunk_size)

    img = (rp * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_NEAREST)

    img = 255 - img
    img = np.flipud(img)

    return img

# ============================================================
# Dataset Cleaning
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

    return df.dropna(subset=["x", "y", "time"])


def clean_chaoshen(df):

    df = df.rename(columns={
        "X": "x",
        "Y": "y",
        "Timestamp": "time",
        "EventName": "event"
    })

    df = df[df["event"] == "Move"].copy()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


def clean_dfl(df):

    df.columns = [c.strip().lower() for c in df.columns]

    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp": "time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})

    if "state" in df.columns:
        df = df[df["state"].str.lower() == "move"]

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])

# ============================================================
# Count samples
# ============================================================

def count_samples(dataset, data_root, chunk_size):

    users = sorted(os.listdir(data_root))

    total = 0

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        for f in os.listdir(user_dir):

            path = os.path.join(user_dir, f)

            if not os.path.isfile(path):
                continue

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            n_chunks = len(df) // chunk_size
            total += n_chunks

    return total, users

# ============================================================
# Main generation
# ============================================================

def generate_tensor_dataset(dataset, data_root, out_dir, chunk_size, percentile):

    print("\nCounting dataset size...")

    total_samples, users = count_samples(dataset, data_root, chunk_size)

    num_users = len(users)

    print("Total samples:", total_samples)
    print("Users:", num_users)

    os.makedirs(out_dir, exist_ok=True)

    images = np.memmap(
        os.path.join(out_dir, "images.npy"),
        dtype="uint8",
        mode="w+",
        shape=(total_samples, BASE_IMG_SIZE, BASE_IMG_SIZE)
    )

    labels = np.memmap(
        os.path.join(out_dir, "labels.npy"),
        dtype="uint8",
        mode="w+",
        shape=(total_samples, num_users)
    )

    sessions = []

    user2index = {u: i for i, u in enumerate(users)}

    idx = 0

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\nUser:", user)

        files = sorted(os.listdir(user_dir))

        for file in files:

            session = os.path.splitext(file)[0]

            path = os.path.join(user_dir, file)

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            n_chunks = len(events) // chunk_size

            print("   ", session, "chunks:", n_chunks)

            for i in range(n_chunks):

                seq = events[i*chunk_size:(i+1)*chunk_size]

                img = rp_to_uint8(seq, percentile, chunk_size)

                images[idx] = img

                y = np.zeros(num_users, dtype=np.uint8)
                y[user2index[user]] = 1

                labels[idx] = y

                sessions.append(session)

                idx += 1

    np.save(os.path.join(out_dir, "sessions.npy"), np.array(sessions))

    print("\nTensor dataset generation finished.")
    print("Saved to:", out_dir)

# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit", "chaoshen", "dfl"])

    parser.add_argument("--data_root",
                        required=True)

    parser.add_argument("--out_dir",
                        required=True)

    parser.add_argument("--chunk_size",
                        type=int,
                        default=60)

    parser.add_argument("--percentile",
                        type=float,
                        default=95)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    generate_tensor_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.chunk_size,
        args.percentile
    )

if __name__ == "__main__":
    main()