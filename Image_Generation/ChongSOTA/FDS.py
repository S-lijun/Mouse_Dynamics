# -*- coding: utf-8 -*-

"""
FAST FDS (Frequency Domain Spectrogram)

Single-channel grayscale version.

Representation:
velocity sequence spectrogram

IMPORTANT:
- NO matplotlib
- NO fake RGB colormap
- single signal -> single channel
- suitable for future multi-signal stacking
"""

import os
import argparse

import pandas as pd
import numpy as np
import cv2

from scipy.signal import spectrogram

from SRP import (
    ROOT,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

GLOBAL_MAX_X = 1919.0
GLOBAL_MAX_Y = 1079.0

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}

# ============================================================
# Clean wrapper
# ============================================================

def _clean_df(dataset, df):

    if dataset == "balabit":
        return clean_balabit(df)

    if dataset == "chaoshen":
        return clean_chaoshen(df)

    if dataset == "dfl":
        return clean_dfl(df)

    raise ValueError(dataset)

# ============================================================
# Build user max x/y
# ============================================================

def build_user_max_xy_from_training(dataset, training_root):

    user_max_xy = {}

    users = sorted(os.listdir(training_root))

    for user in users:

        user_dir = os.path.join(training_root, user)

        if not os.path.isdir(user_dir):
            continue

        max_x = 0.0
        max_y = 0.0
        saw_points = False

        for name in sorted(os.listdir(user_dir)):

            path = os.path.join(user_dir, name)

            if not os.path.isfile(path):
                continue

            df = pd.read_csv(path)

            df = _clean_df(dataset, df)

            if len(df) == 0:
                continue

            max_x = max(max_x, float(df["x"].max()))
            max_y = max(max_y, float(df["y"].max()))

            saw_points = True

        if saw_points:
            user_max_xy[user] = (max_x, max_y)

        else:
            user_max_xy[user] = (GLOBAL_MAX_X, GLOBAL_MAX_Y)

    return user_max_xy

# ============================================================
# Velocity sequence
# ============================================================

def compute_velocity(xs, ys, ts):

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]

    dt = np.maximum(dt, 1e-5)

    v = np.sqrt(dx * dx + dy * dy) / dt

    return v

# ============================================================
# Draw FDS
# ============================================================

def draw_fds(seq_array, save_path, output_size=448):

    xs = seq_array[:, 0]
    ys = seq_array[:, 1]
    ts = seq_array[:, 2]

    if len(xs) < 8:
        return

    # ========================================================
    # Velocity sequence
    # ========================================================

    v = compute_velocity(xs, ys, ts)

    # log compression
    v = np.log1p(v)

    # ========================================================
    # Spectrogram
    # ========================================================

    _, _, Sxx = spectrogram(
        v,
        fs=1.0,
        nperseg=min(16, len(v)),
        noverlap=min(8, len(v) // 2),
        scaling="spectrum",
        mode="magnitude"
    )

    # ========================================================
    # Log compression
    # ========================================================

    Sxx = np.log1p(Sxx)

    # ========================================================
    # Normalize
    # ========================================================

    if Sxx.max() > 0:
        Sxx = Sxx / Sxx.max()

    # ========================================================
    # Convert to grayscale image
    # ========================================================

    img = (Sxx * 255).astype(np.uint8)

    # low frequency at bottom
    img = np.flipud(img)

    # ========================================================
    # Resize
    # ========================================================

    img = cv2.resize(
        img,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)

# ============================================================
# Process dataset
# ============================================================

def process_dataset(
    dataset,
    data_root,
    out_dir,
    user_max_xy,
    output_size=448
):

    users = sorted(os.listdir(data_root))

    sequence_lengths = []

    print("\nDataset:", dataset)

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        if user in user_max_xy:
            norm_x, norm_y = user_max_xy[user]

        else:
            norm_x, norm_y = GLOBAL_MAX_X, GLOBAL_MAX_Y

        print("\n------------------------------")
        print("User:", user)

        session_files = sorted(os.listdir(user_dir))

        for file in session_files:

            path = os.path.join(user_dir, file)

            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]

            print("   Session:", session)

            df = pd.read_csv(path)

            df = _clean_df(dataset, df)

            events = df.to_dict("records")

            if len(events) < 2:
                continue

            # ====================================================
            # Time split
            # ====================================================

            sequences = split_by_time(events)

            # ====================================================
            # Merge short sequences
            # ====================================================

            min_length = norm_x

            sequences = merge_sequences(
                sequences,
                min_length
            )

            sequence_lengths.extend(
                [len(seq) for seq in sequences]
            )

            # ====================================================
            # Draw FDS
            # ====================================================

            for i, seq in enumerate(sequences):

                save_path = os.path.join(
                    out_dir,
                    "FDS_velocity_single_channel",
                    user,
                    f"{session}-{i}.png",
                )

                seq_array = np.array(
                    [
                        [
                            float(e["x"]),
                            float(e["y"]),
                            float(e["time"]),
                        ]
                        for e in seq
                    ],
                    dtype=np.float32,
                )

                draw_fds(
                    seq_array,
                    save_path,
                    output_size
                )

    # ========================================================
    # Stats
    # ========================================================

    print("\n========== Sequence Length Stats ==========")

    if len(sequence_lengths) == 0:

        print("No valid sequence generated.")

    else:

        lengths = np.array(
            sequence_lengths,
            dtype=np.float64
        )

        print("Total sequences:", len(lengths))
        print("min:", int(np.min(lengths)))
        print("median:", float(np.median(lengths)))
        print("average:", float(np.mean(lengths)))
        print("max:", int(np.max(lengths)))

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["balabit", "chaoshen", "dfl"]
    )

    parser.add_argument(
        "--training_root",
        default=None
    )

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        required=True
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=448
    )

    args = parser.parse_args()

    training_rel = (
        args.training_root
        or DEFAULT_TRAINING_ROOT[args.dataset]
    )

    training_root = os.path.join(ROOT, training_rel)

    data_root = os.path.join(ROOT, args.data_root)

    out_dir = os.path.join(ROOT, args.out_dir)

    user_max_xy = build_user_max_xy_from_training(
        args.dataset,
        training_root
    )

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        user_max_xy,
        args.output_size
    )

    print("\nFAST single-channel FDS generation finished.")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()