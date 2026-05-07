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
# Parameters
# ============================================================

TIME_THRESHOLD = 1.0   # seconds
TARGET_SIZE = 448      # final image size
INNER_PADDING = 5      # pixels, keep small white margins around trajectory

# Balabit 1920×1080：全局画布与 merge 阈值（与 XYPlot_global 一致）
GLOBAL_MAX_X = 1919.0
GLOBAL_MAX_Y = 1079.0


# ============================================================
# Trajectory Length
# ============================================================

def compute_length(seq):

    if len(seq) < 2:
        return 0.0

    length = 0.0

    for i in range(len(seq)-1):

        dx = seq[i+1]["x"] - seq[i]["x"]
        dy = seq[i+1]["y"] - seq[i]["y"]

        length += math.sqrt(dx*dx + dy*dy)

    return length


# ============================================================
# Time Difference Split
# ============================================================

def split_by_time(events):

    sequences = []
    current = []

    for i in range(len(events)):

        if i == 0:
            current.append(events[i])
            continue

        dt = events[i]["time"] - events[i-1]["time"]

        if dt > TIME_THRESHOLD:
            sequences.append(current)
            current = []

        current.append(events[i])

    if current:
        sequences.append(current)

    return sequences


# ============================================================
# Merge Short Sequences
# ============================================================

def merge_sequences(sequences, min_length):

    merged = []
    i = 0

    while i < len(sequences):

        current = sequences[i]

        while compute_length(current) < min_length and i+1 < len(sequences):

            current = current + sequences[i+1]
            i += 1

        merged.append(current)
        i += 1

    return merged


# ============================================================
# Draw Sequence (NEW)
# ============================================================
'''
def draw_sequence(seq, save_path, norm_width, norm_height):

    if len(seq) < 2:
        return

    Wg = max(float(norm_width), 1.0)
    Hg = max(float(norm_height), 1.0)
    Wc = int(Wg) + 1
    Hc = int(Hg) + 1

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    x_range = max(xmax - xmin, 1e-6)
    y_range = max(ymax - ymin, 1e-6)

    if x_range >= y_range:
        xn = (xs - xmin) / x_range
        yn = (ys - ymin) / x_range
        S = Wc * (x_range / Wg)
        px = xn * S
        py = yn * S
    else:
        yn = (ys - ymin) / y_range
        xn = (xs - xmin) / y_range
        S = Hc * (y_range / Hg)
        px = xn * S
        py = yn * S

    x_pix = np.clip(np.rint(px), 0, Wc - 1).astype(np.int32)
    y_pix = np.clip(np.rint(py), 0, Hc - 1).astype(np.int32)

    canvas = np.ones((Hc, Wc, 3), dtype=np.uint8) * 255

    prev = None

    for x_i, y_i in zip(x_pix, y_pix):

        if prev is not None:

            cv2.line(
                canvas,
                prev,
                (int(x_i), int(y_i)),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        prev = (int(x_i), int(y_i))

    # ========================================================
    # Resize with aspect ratio → 448 正方形（padding 最后做）
    # ========================================================

    h, w = canvas.shape[:2]
    effective_size = max(1, TARGET_SIZE - 2 * INNER_PADDING)
    scale = effective_size / max(w, h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (TARGET_SIZE - new_h) // 2
    pad_bottom = TARGET_SIZE - new_h - pad_top

    pad_left = (TARGET_SIZE - new_w) // 2
    pad_right = TARGET_SIZE - new_w - pad_left

    final = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, final)

'''


def draw_sequence(seq, save_path, norm_width, norm_height):

    if len(seq) < 2:
        return

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    W = max(float(norm_width), 1.0)
    H = max(float(norm_height), 1.0)
    canvas_w = int(W) + 1
    canvas_h = int(H) + 1

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    xn = xs / W
    yn = ys / H
    x_pix = np.clip(np.rint(xn * W), 0, canvas_w - 1).astype(np.int32)
    y_pix = np.clip(np.rint(yn * H), 0, canvas_h - 1).astype(np.int32)

    prev = None

    for x_i, y_i in zip(x_pix, y_pix):

        if prev is not None:

            cv2.line(
                canvas,
                prev,
                (int(x_i), int(y_i)),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        prev = (int(x_i), int(y_i))

    # ========================================================
    # Resize with aspect ratio
    # ========================================================

    h, w = canvas.shape[:2]
    effective_size = max(1, TARGET_SIZE - 2 * INNER_PADDING)
    scale = effective_size / max(w, h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #bw = np.where(gray == 255, 255, 0).astype(np.uint8)
    #resized = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    #resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    darken = 100
    gray = np.where(gray < 255, np.clip(gray - darken, 0, 255), 255).astype(np.uint8)
    resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ========================================================
    # Center Padding
    # ========================================================

    pad_top = (TARGET_SIZE - new_h) // 2
    pad_bottom = TARGET_SIZE - new_h - pad_top

    pad_left = (TARGET_SIZE - new_w) // 2
    pad_right = TARGET_SIZE - new_w - pad_left

    final = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255,255,255)
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, final)

# ============================================================
# Cleaning (unchanged)
# ============================================================

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp":"time",
        "x":"x",
        "y":"y",
        "state":"state"
    })

    df = df[df["state"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_chaoshen(df):

    df = df.rename(columns={
        "X":"x",
        "Y":"y",
        "Timestamp":"time",
        "EventName":"event"
    })

    df = df[df["event"] == "Move"]

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
# Dataset Processing (MODIFIED)
# ============================================================

def process_dataset(dataset, data_root, out_dir):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Global norm W×H (min_length & draw):", GLOBAL_MAX_X, GLOBAL_MAX_Y)

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

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

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df.to_dict("records")

            print("      Events:", len(events))
            if len(events) < 2:
                continue

            # ========================================================
            # 1. Time Split
            # ========================================================

            sequences = split_by_time(events)

            print("      After split:", len(sequences))

            # ========================================================
            # 2. Compute min_length (per session)
            # ========================================================

            xs = np.array([float(e["x"]) for e in events], dtype=np.float64)
            ys = np.array([float(e["y"]) for e in events], dtype=np.float64)
            session_width = float(np.max(xs))
            session_height = float(np.max(ys))
            print("      Session max x/y:", session_width, session_height)

            min_length = session_width

            # ========================================================
            # 3. Merge
            # ========================================================

            sequences = merge_sequences(sequences, min_length)

            print("      After merge:", len(sequences))

            # ========================================================
            # 4. Draw
            # ========================================================

            for i, seq in enumerate(sequences):

                save_path = os.path.join(
                    out_dir,
                    "Chong",
                    user,
                    f"{session}-{i}.png"
                )

                draw_sequence(seq, save_path, GLOBAL_MAX_X, GLOBAL_MAX_Y)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit","chaoshen","dfl"])

    parser.add_argument("--data_root",
                        required=True)

    parser.add_argument("--out_dir",
                        required=True)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir
    )

    print("\nTimeDiff image generation finished.")


if __name__ == "__main__":
    main()