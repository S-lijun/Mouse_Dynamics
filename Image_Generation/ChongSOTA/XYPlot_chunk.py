# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# Parameters
# ============================================================

TARGET_SIZE = 448      # final image size
INNER_PADDING = 5      # pixels, keep small white margins around trajectory

# Balabit 1920×1080
GLOBAL_MAX_X = 1919.0
GLOBAL_MAX_Y = 1079.0

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}


# ============================================================
# Chunking
# ============================================================

def split_by_chunk_size(events, chunk_size):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    return [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]


# ============================================================
# Draw Sequence (NEW)
# ============================================================


def draw_sequence(seq, save_path, norm_width, norm_height):

    if len(seq) < 2:
        return

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    W = max(float(norm_width), 1.0)
    H = max(float(norm_height), 1.0)
    a = H / W
    canvas_w = int(W) + 1
    span = float(canvas_w - 1)
    canvas_h = int(np.ceil(a * span)) + 1

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    x_pix = np.clip(np.rint(xs / W * span), 0, canvas_w - 1).astype(np.int32)
    y_pix = np.clip(np.rint(ys / W * span), 0, canvas_h - 1).astype(np.int32)

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
# Per-user bounds from training split
# ============================================================

def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    raise ValueError(dataset)


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
# Dataset Processing
# ============================================================

def process_dataset(dataset, data_root, out_dir, chunk_size, user_max_xy):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Chunk size:", chunk_size)

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        if user in user_max_xy:
            norm_x, norm_y = user_max_xy[user]
        else:
            norm_x, norm_y = GLOBAL_MAX_X, GLOBAL_MAX_Y
            print(
                "\n[WARN] User", user, "not in training scan; using GLOBAL_MAX_X/Y:",
                norm_x, norm_y,
            )

        print("\n------------------------------")
        print("User:", user, "| norm W×H (from training):", norm_x, norm_y)

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

            print("      Events:", len(events))
            if len(events) < 2:
                continue

            # ========================================================
            # Chunk only (no segmentation, no merge)
            # ========================================================

            sequences = split_by_chunk_size(events, chunk_size)
            print("      Chunks:", len(sequences))

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

                draw_sequence(seq, save_path, norm_x, norm_y)


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

    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset training_files.",
    )

    parser.add_argument("--out_dir",
                        required=True)

    parser.add_argument("--sizes",
                        type=int,
                        required=True,
                        help="Number of events per chunk.")

    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = os.path.join(ROOT, training_rel)
    print("[training_root]", training_rel)
    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    user_max_xy = build_user_max_xy_from_training(args.dataset, training_root)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        user_max_xy,
    )

    print("\nChunk image generation finished.")


if __name__ == "__main__":
    main()