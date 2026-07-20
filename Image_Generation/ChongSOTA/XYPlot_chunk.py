# -*- coding: utf-8 -*-
"""
Chunked centered XYPlot: fixed-size event windows + per-sequence bbox fit/center.

Drawing matches XYPlot_centered (no screen-width / per-user normalization).
Segmentation is fixed chunk_size (default 125), not time-diff split + merge.
"""

import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd

from XYPlot import (
    ROOT,
    TARGET_SIZE,
    INNER_PADDING,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

TENSOR_SUBDIR = "Chong"


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", string)]


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    raise ValueError(dataset)


def list_users(data_root):
    return sorted(
        [u for u in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, u))],
        key=natural_key,
    )


def list_session_files(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )


# ============================================================
# Chunking
# ============================================================

def split_by_chunk_size(events, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]


# ============================================================
# Centered draw (same as XYPlot_centered)
# ============================================================

def render_sequence_centered(seq):
    """
    Fit trajectory bbox into TARGET_SIZE×TARGET_SIZE with uniform scale, centered.
    """
    if len(seq) < 2:
        return None

    img_size = int(TARGET_SIZE)
    effective_size = max(1, img_size - 2 * INNER_PADDING)

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    pad_x = range_x * 0.05
    pad_y = range_y * 0.05

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    range_x = max_x - min_x
    range_y = max_y - min_y

    scale = min(effective_size / range_x, effective_size / range_y)
    offset_x = (img_size - range_x * scale) / 2
    offset_y = (img_size - range_y * scale) / 2

    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    prev = None
    for x, y in zip(xs, ys):
        x_s = int(np.clip((x - min_x) * scale + offset_x, 0, img_size - 1))
        y_s = int(np.clip((y - min_y) * scale + offset_y, 0, img_size - 1))

        if prev is not None:
            cv2.line(
                canvas,
                prev,
                (x_s, y_s),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        prev = (x_s, y_s)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_sequence_centered(seq, save_path):
    final = render_sequence_centered(seq)
    if final is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final)


def _session_sequences(dataset, path, chunk_size):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)

    events = df.to_dict("records")
    if len(events) < 2:
        return []

    sequences = split_by_chunk_size(events, chunk_size)
    return [seq for seq in sequences if len(seq) >= 2]


def count_samples(dataset, data_root, chunk_size):
    total = 0
    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            total += len(_session_sequences(dataset, path, chunk_size))
    return total


def bgr_to_tensor_chw(img):
    return img.transpose(2, 0, 1)


# ============================================================
# Dataset Processing
# ============================================================

def process_dataset_tensors(dataset, data_root, out_dir, chunk_size):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Chunk size:", chunk_size)
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)
    print("\n[Phase] Generating centered chunk XYPlot tensors...")

    total_samples = count_samples(dataset, data_root, chunk_size)
    tensor_root = os.path.join(out_dir, TENSOR_SUBDIR)
    os.makedirs(tensor_root, exist_ok=True)

    H = W = int(TARGET_SIZE)
    print(f"\n[{TENSOR_SUBDIR}] Total samples: {total_samples} | Tensor size: {H}x{W}")

    images = np.memmap(
        os.path.join(tensor_root, "images.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total_samples, 3, H, W),
    )
    labels = np.memmap(
        os.path.join(tensor_root, "labels.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total_samples, num_users),
    )

    sessions = []
    idx = 0

    for user in users:
        user_dir = os.path.join(data_root, user)

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path, chunk_size)
            print(f"   Session: {session} -> {len(sequences)} chunks")

            for seq in sequences:
                img = render_sequence_centered(seq)
                if img is None:
                    continue

                images[idx] = bgr_to_tensor_chw(img)
                y = np.zeros(num_users, dtype=np.uint8)
                y[user_to_idx[user]] = 1
                labels[idx] = y
                sessions.append(session)
                idx += 1

    images.flush()
    labels.flush()
    np.save(
        os.path.join(tensor_root, "sessions.npy"),
        np.array(sessions, dtype=object),
    )
    print(f"\nTensor dataset saved to: {tensor_root}")


def process_dataset(dataset, data_root, out_dir, chunk_size, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, chunk_size)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Chunk size:", chunk_size)
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)

    for user in users:
        user_dir = os.path.join(data_root, user)

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            print("   Session:", session)

            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            events = df.to_dict("records")

            print("      Events:", len(events))
            if len(events) < 2:
                continue

            sequences = split_by_chunk_size(events, chunk_size)
            print("      Chunks:", len(sequences))

            for i, seq in enumerate(sequences):
                if len(seq) < 2:
                    continue
                save_path = os.path.join(
                    out_dir,
                    TENSOR_SUBDIR,
                    user,
                    f"{session}-{i}.png",
                )
                draw_sequence_centered(seq, save_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        default=125,
        help="Number of events per chunk.",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="Output images.npy / labels.npy / sessions.npy instead of PNG.",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("[data_root]", data_root)
    print("[out_dir]", out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        tensors=args.tensors,
    )
    print("\nCentered chunk XYPlot generation finished.")


if __name__ == "__main__":
    main()
