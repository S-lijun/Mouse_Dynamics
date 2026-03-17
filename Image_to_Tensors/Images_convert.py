# -*- coding: utf-8 -*-

import os
import re
import cv2
import argparse
import numpy as np
from tqdm import tqdm


# ============================================================
# 自动找到 project root
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("[ROOT]", ROOT)


# ============================================================
# Natural Sort 
# ============================================================

def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r'(\d+)', string)]


# ============================================================
# Count dataset size
# ============================================================

def count_images(root):

    users = sorted(os.listdir(root), key=natural_key)

    total = 0

    for u in users:

        user_dir = os.path.join(root, u)

        if not os.path.isdir(user_dir):
            continue

        files = sorted(os.listdir(user_dir), key=natural_key)

        for f in files:
            if f.endswith(".png"):
                total += 1

    return total, users


# ============================================================
# Convert dataset
# ============================================================

def convert_dataset(image_root, out_root):

    print("\nCounting dataset size...")

    total_samples, users = count_images(image_root)

    num_users = len(users)

    print("Total samples:", total_samples)
    print("Users:", num_users)

    os.makedirs(out_root, exist_ok=True)

    H = 150
    W = 150

    images = np.memmap(
        os.path.join(out_root, "images.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total_samples, 3, H, W)
    )

    labels = np.memmap(
        os.path.join(out_root, "labels.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total_samples, num_users)
    )

    sessions = []

    user_to_idx = {u: i for i, u in enumerate(users)}

    idx = 0

    for user in users:

        user_dir = os.path.join(image_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\nUser:", user)

        files = sorted(os.listdir(user_dir), key=natural_key)

        for f in tqdm(files):

            if not f.endswith(".png"):
                continue

            path = os.path.join(user_dir, f)

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            # resize
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

            # ===============================
            # convert to 3 channel
            # ===============================

            if img.ndim == 2:
                # grayscale
                img = np.stack([img, img, img], axis=0)

            elif img.ndim == 3:
                # RGB / BGR
                img = img.transpose(2, 0, 1)

                # remove alpha channel if exists
                if img.shape[0] > 3:
                    img = img[:3]

            else:
                raise ValueError("Unsupported image format")

            images[idx] = img

            y = np.zeros(num_users, dtype=np.uint8)
            y[user_to_idx[user]] = 1

            labels[idx] = y

            session = f.split("-")[0]
            sessions.append(session)

            idx += 1

    np.save(
        os.path.join(out_root, "sessions.npy"),
        np.array(sessions, dtype=object)
    )

    print("\nTensor dataset saved to:", out_root)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        required=True,
        help="Image folder relative to project root"
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output tensor folder relative to project root"
    )

    args = parser.parse_args()

    image_root = os.path.join(ROOT, args.data_root)
    out_root = os.path.join(ROOT, args.out_dir)

    convert_dataset(image_root, out_root)


if __name__ == "__main__":
    main()