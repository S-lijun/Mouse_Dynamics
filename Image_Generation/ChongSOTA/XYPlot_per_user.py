# -*- coding: utf-8 -*-
"""
Per-user normalization: max x / max y are taken from all sessions of that user
under training_files only (default path per --dataset). The same bounds are
used when drawing sessions from any data_root (e.g. testing_files).
"""

import os
import argparse

import pandas as pd

from XYPlot import (
    ROOT,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    draw_sequence,
)

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    raise ValueError(dataset)


def build_user_max_xy_from_training(dataset, training_root):
    """
    Scan every session under training_root/<user>/ and record each user's
    global max x and max y (after the same cleaning as drawing).
    """
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


def process_dataset(dataset, data_root, out_dir, user_max_xy):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users in data_root:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")

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

            sequences = split_by_time(events)
            print("      After split:", len(sequences))

            min_length = norm_x
            sequences = merge_sequences(sequences, min_length)
            print("      After merge:", len(sequences))

            for i, seq in enumerate(sequences):
                save_path = os.path.join(
                    out_dir,
                    "Chong_per_user",
                    user,
                    f"{session}-{i}.png",
                )
                draw_sequence(seq, save_path, norm_x, norm_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset (Balabit/ChaoShen/DFL training_files).",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (train or test), relative to ROOT.",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = os.path.join(ROOT, training_rel)
    print("[training_root]", training_rel)
    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    user_max_xy = build_user_max_xy_from_training(args.dataset, training_root)

    print("\nUSER_MAX_XY (from training_root):")
    for u in sorted(user_max_xy.keys()):
        print("  ", u, "->", user_max_xy[u])

    process_dataset(args.dataset, data_root, out_dir, user_max_xy)
    print("\nPer-user XYPlot image generation finished.")


if __name__ == "__main__":
    main()
