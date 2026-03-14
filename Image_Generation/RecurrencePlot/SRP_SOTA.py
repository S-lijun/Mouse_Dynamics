# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

BASE_CHUNK_SIZE = 150
BASE_IMG_SIZE = 224


def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# Mazumdar Recurrence Plot
# ============================================================

def compute_rp(seq, epsilon=0.3):

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # per-trace distance normalization
    dist = dist / (dist.max() + 1e-8)

    avg_dist = np.mean(dist, axis=1)

    recurrent = avg_dist < epsilon

    rp = np.where(
        np.outer(recurrent, recurrent),
        dist,
        epsilon
    ).astype(np.float32)

    if rp.max() > rp.min():
        rp = (rp - rp.min()) / (rp.max() - rp.min())

    return rp


# ============================================================
# Draw RP
# ============================================================

def draw_rp(seq, save_path, epsilon, chunk_size):

    rp = compute_rp(seq, epsilon)

    img_size = get_dynamic_image_size(chunk_size)

    img = (rp * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(
            img,
            (img_size, img_size),
            interpolation=cv2.INTER_NEAREST
        )

    img = np.stack([img, img, img], axis=-1)

    img = 255 - img
    img = np.flipud(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)


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
# Session Files
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
# Dataset Processing
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
        print("User:", user)

        session_files = get_session_files(dataset, user_dir)

        print("Sessions found:", len(session_files))

        for file in session_files:

            session = os.path.splitext(file)[0]
            path = os.path.join(user_dir, file)

            print("   Session:", session)

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)

            elif dataset == "chaoshen":
                df = clean_chaoshen(df)

            elif dataset == "dfl":
                df = clean_dfl(df)

            # ==================================================
            # session-level screen normalization
            # ==================================================

            min_x = df["x"].min()
            max_x = df["x"].max()
            min_y = df["y"].min()
            max_y = df["y"].max()

            df["x"] = (df["x"] - min_x) / (max_x - min_x + 1e-8)
            df["y"] = (df["y"] - min_y) / (max_y - min_y + 1e-8)

            events = df.to_dict("records")

            events_np = np.array(
                [[e["x"], e["y"], e["time"]] for e in events],
                dtype=np.float32
            )

            print("      Events:", len(events_np))

            for chunk_size in sizes:

                n_chunks = len(events_np) // chunk_size

                print("      chunk", chunk_size, "->", n_chunks)

                for i in range(n_chunks):

                    seq = events_np[i*chunk_size:(i+1)*chunk_size]

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_rp(seq, save_path, epsilon, chunk_size)


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

    parser.add_argument("--sizes",
                        type=int,
                        nargs="+",
                        default=[150])

    parser.add_argument("--epsilon",
                        type=float,
                        default=0.8)

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

    print("\nRP generation finished.")


if __name__ == "__main__":
    main()