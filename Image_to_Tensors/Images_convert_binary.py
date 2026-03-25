# -*- coding: utf-8 -*-

import os
import re
import cv2
import argparse
import numpy as np
from tqdm import tqdm

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("[ROOT]", ROOT)

# ============================================================
# Natural sort
# ============================================================

def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r'(\d+)', string)]

# ============================================================
# Convert Binary Dataset
# ============================================================

def convert_binary_dataset(image_root, out_root, target_user):

    users = sorted(os.listdir(image_root), key=natural_key)

    print("\nUsers:", users)

    total = 0
    for u in users:
        files = os.listdir(os.path.join(image_root, u))
        total += len([f for f in files if f.endswith(".png")])

    print("Total samples:", total)

    os.makedirs(out_root, exist_ok=True)

    H, W = 224, 224

    images = np.memmap(
        os.path.join(out_root, "images.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total, 3, H, W)
    )

    labels = np.memmap(
        os.path.join(out_root, "labels.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total,)
    )

    sessions = []

    idx = 0

    for user in users:

        user_dir = os.path.join(image_root, user)

        files = sorted(os.listdir(user_dir), key=natural_key)

        print("\nProcessing:", user)

        for f in tqdm(files):

            if not f.endswith(".png"):
                continue

            path = os.path.join(user_dir, f)

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

            # 3-channel
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=0)
            else:
                img = img.transpose(2, 0, 1)
                if img.shape[0] > 3:
                    img = img[:3]

            images[idx] = img

            # binary label
            labels[idx] = 1 if user == target_user else 0

            session = f.split("-")[0]
            sessions.append(session)

            idx += 1

    np.save(os.path.join(out_root, "sessions.npy"), np.array(sessions, dtype=object))

    print("\nSaved to:", out_root)

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_user", required=True)

    args = parser.parse_args()

    image_root = os.path.join(ROOT, args.data_root)
    out_root = os.path.join(ROOT, args.out_dir)

    convert_binary_dataset(image_root, out_root, args.target_user)