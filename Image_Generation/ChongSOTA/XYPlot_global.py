# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np

from XYPlot import (
    ROOT,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    draw_sequence,
)

GLOBAL_MAX_X = 1919.0
GLOBAL_MAX_Y = 1079.0


def process_dataset(dataset, data_root, out_dir):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Using global max x/y:", GLOBAL_MAX_X, GLOBAL_MAX_Y)

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

            sequences = split_by_time(events)
            print("      After split:", len(sequences))

            min_length = GLOBAL_MAX_X
            sequences = merge_sequences(sequences, min_length)
            print("      After merge:", len(sequences))

            for i, seq in enumerate(sequences):
                save_path = os.path.join(
                    out_dir,
                    "Chong_global",
                    user,
                    f"{session}-{i}.png",
                )
                draw_sequence(seq, save_path, GLOBAL_MAX_X, GLOBAL_MAX_Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(args.dataset, data_root, out_dir)
    print("\nGlobal XYPlot image generation finished.")


if __name__ == "__main__":
    main()
