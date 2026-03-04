# -*- coding: utf-8 -*-
"""
SRP Baseline — Pure Spatial Recurrence Plot (FAST)
--------------------------------------------------
ChaoShen Dataset Version

Same logic as original version.
Only optimized for speed.

Optimizations
-------------
1. Replace matplotlib with OpenCV image writing
2. Faster numpy operations
3. Faster CSV loading
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math

# ============================================================
# ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(f"[AutoRoot] Project root detected = {ROOT}")

# ============================================================
# Base Config
# ============================================================
BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200

# ============================================================
# Dynamic Image Size
# ============================================================
def get_dynamic_image_size(chunk_size):

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Core RP Logic (UNCHANGED)
# ============================================================
def compute_rp_xy(seq, percentile=95):

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, percentile)

    rec = np.where(dist <= eps, dist, eps).astype(np.float32)

    if rec.max() > rec.min():
        rec = (rec - rec.min()) / (rec.max() - rec.min())

    return rec

# ============================================================
# FAST Drawing (OpenCV instead of matplotlib)
# ============================================================
def draw_rp_image(seq, save_path, percentile, chunk_size):

    rp = compute_rp_xy(seq, percentile)

    img_size = get_dynamic_image_size(chunk_size)

    img = (rp * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_NEAREST)

    # gray_r
    img = 255 - img

    # fix orientation (match origin="lower")
    img = np.flipud(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)

# ============================================================
# Cleaning (ChaoShen specific)
# ============================================================
def clean_and_rename_cols(df):

    df = df.rename(columns={
        "X": "x",
        "Y": "y",
        "Timestamp": "time",
        "EventName": "event"
    })

    df = df[df["event"] == "Move"].copy()

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size, percentile):

    n_chunks = len(events) // chunk_size

    print(f"      [ChunkSize={chunk_size}] Total Chunks = {n_chunks}")

    events_np = np.array(
        [[e["x"], e["y"], e["time"]] for e in events],
        dtype=np.float32
    )

    for i in range(n_chunks):

        seq = events_np[i * chunk_size:(i + 1) * chunk_size]

        save_path = os.path.join(
            out_dir,
            f"event{chunk_size}",
            user,
            f"{session_name}-{i}.png"
        )

        draw_rp_image(seq, save_path, percentile, chunk_size)

        if (i + 1) % 50 == 0 or (i + 1) == n_chunks:
            print(f"         -> Chunk {i+1}/{n_chunks} done")

# ============================================================
# Dataset Processing
# ============================================================
def process_dataset(data_root, out_dir, sizes, percentile,
                    target_users=None, target_sessions=None):

    users = sorted(os.listdir(data_root))
    total_users = len(users)

    print(f"\n[Dataset] Total Users = {total_users}")

    produced_total = {k: 0 for k in sizes}

    for u_idx, user in enumerate(users, 1):

        if target_users and user not in target_users:
            continue

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\n====================================================")
        print(f"[User {u_idx}/{total_users}] Processing: {user}")
        print("====================================================")

        session_files = sorted([
            f for f in os.listdir(user_dir)
            if f.startswith("session_") and f.endswith(".csv")
        ])

        total_sessions = len(session_files)

        for s_idx, file in enumerate(session_files, 1):

            session = os.path.splitext(file)[0]

            if target_sessions and session not in target_sessions:
                continue

            print(f"\n   [Session {s_idx}/{total_sessions}] {session}")

            df = pd.read_csv(
                os.path.join(user_dir, file),
                usecols=["X", "Y", "EventName", "Timestamp"]
            )

            df = clean_and_rename_cols(df)

            events = df.to_dict(orient="records")

            print(f"      Total Events = {len(events)}")

            for sz in sizes:

                before = len(events) // sz

                chunk_and_draw(events, out_dir, user, session, sz, percentile)

                produced_total[sz] += before

    print("\n====================================================")
    print("[Summary]")

    for k in sizes:

        resolution = get_dynamic_image_size(k)

        print(
            f"event{k}: {produced_total[k]} images "
            f"(Resolution: {resolution}x{resolution})"
        )

    print(f"TOTAL images: {sum(produced_total.values())}")
    print("====================================================")

# ============================================================
# CLI
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description="SRP Baseline (Pure Spatial Recurrence Plot)"
    )

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    parser.add_argument("--p_percentile", type=float, default=95)
    parser.add_argument("--users", type=str, nargs="+", default=[])
    parser.add_argument("--sessions", type=str, nargs="+", default=[])

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    sizes = sorted(set(args.sizes))

    target_users = set(args.users) if args.users else None
    target_sessions = set(args.sessions) if args.sessions else None

    process_dataset(
        data_root,
        out_dir,
        sizes,
        args.p_percentile,
        target_users,
        target_sessions,
    )

    print("\n[Done] SRP Baseline Generation Complete.")


if __name__ == "__main__":
    main()