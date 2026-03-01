# -*- coding: utf-8 -*-
"""
SRP (Recurrence Plot) Version — Dynamic Image Sizing
---------------------------------------------------------
- Pure chunking (no sliding window)
- Auto root detection
- Image size scales with chunk size: base 15 -> 224
- Scale formula: new_size = 224 * sqrt(chunk_size / 15)
- White background, black information
"""
'''
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# Automatically detect project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")

print(f"[AutoRoot] Project root detected = {ROOT}")
print(f"[AutoRoot] Using data_dir = {DATA_ROOT}")

# Base Configuration
BASE_CHUNK_SIZE = 60
BASE_IMG_SIZE = 224
DPI = 200

def get_dynamic_image_size(chunk_size):
    """
    size = 224 * sqrt(chunk_size / 15)
    """
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    dynamic_size = int(round(BASE_IMG_SIZE * scale))
    return dynamic_size

# ============================================================
# RP core
# ============================================================
def compute_rp_xy(seq, percentile=95):
    """
    Compute RP directly from raw (x, y) coordinates.
    seq: shape (T, 3) = [x, y, t]
    """
    # 1. remove padding
    non_zero_mask = ~np.all(seq == 0, axis=1)
    seq = seq[non_zero_mask]
    if len(seq) == 0:
        return None

    # 2. take (x, y)
    coords = seq[:, :2]  # (T, 2)

    # 3. pairwise distance
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (T, T)

    # 4. epsilon by percentile
    epsilon = np.percentile(dist, percentile)

    # 5. clipped RP
    rec = np.where(dist <= epsilon, dist, epsilon).astype(np.float32)

    # 6. normalize to [0, 1]
    if rec.max() > rec.min():
        rec = (rec - rec.min()) / (rec.max() - rec.min())

    return rec


def draw_rp_image(seq, save_path, percentile, chunk_size):
    rp = compute_rp_xy(seq, percentile)
    if rp is None:
        return

    # calculate the image size for specific chunk size
    img_size_px = get_dynamic_image_size(chunk_size)
    
    fig, ax = plt.subplots(
        figsize=(img_size_px / DPI, img_size_px / DPI), dpi=DPI
    )

    # ---- white background ----
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(rp, cmap="gray_r", origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)


# ============================================================
# Data cleaning
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "client timestamp": "time",
            "x": "x",
            "y": "y",
            "state": "state",
        }
    )

    df = df[df["state"] == "Move"].copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["x", "y", "time"])

    return df


# ============================================================
# Pure chunking
# ============================================================
def chunk_and_draw_rp(events, out_dir, user, session_name, chunk_size, percentile):
    L = len(events)
    n_chunks = L // chunk_size
    count = 0

    for i in range(n_chunks):
        start = i * chunk_size
        chunk = events[start : start + chunk_size]

        seq = np.array([[e["x"], e["y"], e["time"]] for e in chunk], dtype=np.float32)

        save_dir = os.path.join(out_dir, f"event{chunk_size}", user)
        save_path = os.path.join(save_dir, f"{session_name}-{i}.png")

        draw_rp_image(seq, save_path, percentile, chunk_size)
        count += 1

    return count


def process_one_session(path, user, session_name, out_dir, sizes, percentile):
    df = pd.read_csv(path)
    df = clean_and_rename_cols(df)
    events = df.to_dict(orient="records")

    produced = {}
    for chunk_size in sizes:
        produced[chunk_size] = chunk_and_draw_rp(
            events, out_dir, user, session_name, chunk_size, percentile
        )

    return produced


def process_dataset(data_dir, out_dir, sizes, percentile, target_users=None, target_sessions=None):
    produced_total = {k: 0 for k in sizes}

    for user in sorted(os.listdir(data_dir)):
        if target_users and user not in target_users:
            continue

        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue

        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"):
                continue

            session_name = os.path.splitext(file)[0]
            if target_sessions and session_name not in target_sessions:
                continue

            path = os.path.join(user_dir, file)
            print(f"[Process] {user}/{file} for chunk sizes {sizes}")

            produced = process_one_session(
                path, user, session_name, out_dir, sizes, percentile
            )

            for k, v in produced.items():
                produced_total[k] += v

    print("=" * 50)
    for k in sizes:
        actual_size = get_dynamic_image_size(k)
        print(f"event{k}: {produced_total[k]} images (Resolution: {actual_size}x{actual_size})")
    print(f"TOTAL images: {sum(produced_total.values())}")

    return produced_total


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="SRP generation with dynamic scaling (Balabit)"
    )
   
    p.add_argument("--out_dir", type=str, default=f"Images/SRP_protocol1")
    p.add_argument("--sizes", type=int, nargs="+", default=[60])
    p.add_argument("--users", type=str, nargs="+", default=[])
    p.add_argument("--sessions", type=str, nargs="+", default=[])
    p.add_argument("--percentile", type=float, default=95)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    sizes = sorted(set(args.sizes))
    target_users = set(args.users) if args.users else None
    target_sessions = set(args.sessions) if args.sessions else None

    process_dataset(
        DATA_ROOT,
        out_dir,
        sizes,
        args.percentile,
        target_users,
        target_sessions,
    )


if __name__ == "__main__":
    main()
'''

# -*- coding: utf-8 -*-
"""
SRP Baseline — Pure Spatial Recurrence Plot (Protocol-Safe)
------------------------------------------------------------
- R Channel only (Position Recurrence Matrix)
- No velocity
- No temporal encoding
- Dynamic image sizing
- Unified CLI + unified printing style
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Core RP Logic
# ============================================================
def compute_rp_xy(seq, percentile=95):

    T = len(seq)
    if T == 0:
        return None

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, percentile)
    rec = np.where(dist <= eps, dist, eps).astype(np.float32)

    if rec.max() > rec.min():
        rec = (rec - rec.min()) / (rec.max() - rec.min())

    return rec

# ============================================================
# Drawing
# ============================================================
def draw_rp_image(seq, save_path, percentile, chunk_size):

    rp = compute_rp_xy(seq, percentile)
    if rp is None:
        return

    img_size = get_dynamic_image_size(chunk_size)

    fig, ax = plt.subplots(
        figsize=(img_size / DPI, img_size / DPI),
        dpi=DPI
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(rp, cmap="gray_r", origin="lower")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)

# ============================================================
# Cleaning
# ============================================================
def clean_and_rename_cols(df):

    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state",
    })

    df = df[df["state"] == "Move"].copy()

    for col in ["x", "y", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["x", "y", "time"]).reset_index(drop=True)

# ============================================================
# Chunking
# ============================================================
def chunk_and_draw(events, out_dir, user, session_name, chunk_size, percentile):

    n_chunks = len(events) // chunk_size
    print(f"      [ChunkSize={chunk_size}] Total Chunks = {n_chunks}")

    for i in range(n_chunks):

        chunk = events[i * chunk_size:(i + 1) * chunk_size]
        seq = np.array([[e["x"], e["y"], e["time"]] for e in chunk], dtype=np.float32)

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
def process_dataset(data_root, out_dir, sizes, percentile, target_users=None, target_sessions=None):

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
            if f.startswith("session_")
        ])

        total_sessions = len(session_files)

        for s_idx, file in enumerate(session_files, 1):

            session = os.path.splitext(file)[0]

            if target_sessions and session not in target_sessions:
                continue

            print(f"\n   [Session {s_idx}/{total_sessions}] {session}")

            df = clean_and_rename_cols(
                pd.read_csv(os.path.join(user_dir, file))
            )

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
        print(f"event{k}: {produced_total[k]} images "
              f"(Resolution: {resolution}x{resolution})")

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