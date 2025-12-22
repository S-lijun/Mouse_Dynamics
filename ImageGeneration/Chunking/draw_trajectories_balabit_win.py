# -*- coding: utf-8 -*-
"""
Chunk Balabit mouse logs and render images
(movement = red line, mousedown = blue circle, mouseup = green cross)

- Generate images directly to Images/Chunck/event{chunk_size}/...
- Support multiple chunk sizes via CLI parser (default: 60 80 100 120 130)
- Sliding window (stride) per chunk size
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Drawing
# ----------------------------
IMG_SIZE = (224, 224)  # pixels
DPI = 100

def _scaled(val, min_val, scale, offset):
    return (val - min_val) * scale + offset

def draw_mouse_chunk(chunk, save_path):
    if len(chunk) == 0:
        return

    x_coords = np.array([float(e['x']) for e in chunk])
    y_coords = np.array([float(e['y']) for e in chunk])

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # padding
    padding_ratio = 0.05
    range_x = max_x - min_x
    range_y = max_y - min_y
    pad_x = range_x * padding_ratio if range_x > 0 else 1.0
    pad_y = range_y * padding_ratio if range_y > 0 else 1.0
    min_x -= pad_x; max_x += pad_x
    min_y -= pad_y; max_y += pad_y

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    scale = min(IMG_SIZE[0] / range_x, IMG_SIZE[1] / range_y)
    offset_x = (IMG_SIZE[0] - range_x * scale) / 2.0
    offset_y = (IMG_SIZE[1] - range_y * scale) / 2.0

    fig, ax = plt.subplots(figsize=(IMG_SIZE[0]/DPI, IMG_SIZE[1]/DPI), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, IMG_SIZE[0]); ax.set_ylim(IMG_SIZE[1], 0); ax.axis("off")

    prev_x, prev_y = None, None
    for event in chunk:
        x = float(event['x']); y = float(event['y'])
        state = str(event['state']).lower()

        if x > 10000 or y > 10000:
            continue

        x_s = _scaled(x, min_x, scale, offset_x)
        y_s = _scaled(y, min_y, scale, offset_y)

        if state == "movement" and prev_x is not None:
            ax.plot([prev_x, x_s], [prev_y, y_s],
                    color="red", linewidth=2,
                    marker="o", markersize=4, markerfacecolor="red", markeredgewidth=0)

        elif state == "mousedown":
            ax.add_patch(plt.Circle((x_s, y_s), 7, color="blue", fill=True))
        elif state == "mouseup":
            ax.plot([x_s - 5, x_s + 5], [y_s - 5, y_s + 5], color="green", linewidth=2)
            ax.plot([x_s - 5, x_s + 5], [y_s + 5, y_s - 5], color="green", linewidth=2)

        prev_x, prev_y = x_s, y_s

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ----------------------------
# Processing (Balabit)
# ----------------------------
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Balabit 原始列: RecordTimeStamp, ClientTimeStamp, button, state, x, y
    df = df.rename(columns={
        'client timestamp': 'time',
        'x': 'x',
        'y': 'y',
        'state': 'state'
    })

    # 只保留三类 state
    df = df[df['state'].isin(['Pressed', 'Released', 'Move'])].copy()

    # 映射到 Gmail 格式
    state_map = {
        'Pressed': 'mousedown',
        'Released': 'mouseup',
        'Move': 'movement'
    }
    df['state'] = df['state'].map(state_map)

    # 数值化
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x', 'y', 'state', 'time'])

    return df

def windows_count(length, size, stride):
    if length < size:
        return 0
    return 1 + (length - size) // stride

def chunk_and_draw_sliding(events, out_dir, user, session_name, chunk_size, stride):
    L = len(events)
    if L < chunk_size:
        return 0
    count = 0
    for start in range(0, L - chunk_size + 1, stride):
        chunk = events[start:start + chunk_size]
        save_dir = os.path.join(out_dir, f"event{chunk_size}", user)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{session_name}-{start}.png")
        draw_mouse_chunk(chunk, save_path)
        count += 1
    return count

def process_one_session(path, user, session_name, out_dir, size_to_stride):
    df = pd.read_csv(path)
    df = clean_and_rename_cols(df)
    events = df.to_dict(orient='records')
    produced = {k: 0 for k in size_to_stride}
    for sz, st in size_to_stride.items():
        produced[sz] += chunk_and_draw_sliding(events, out_dir, user, session_name, sz, st)
    return produced

def process_dataset(data_dir, out_dir, size_to_stride, target_sessions = None):
    produced_total = {k: 0 for k in size_to_stride}
    target_users = {"user7"}   # 只跑这两个用户
    for user in sorted(os.listdir(data_dir)):
        if user not in target_users: #
            continue #
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"):  # Balabit 文件
                continue
            #session_path = os.path.join(user_dir, file)

            session_name = os.path.splitext(file)[0]

            if target_sessions and session_name not in target_sessions: #
                continue

            session_path = os.path.join(user_dir, file)
            print(f"[{user}/{file}] -> sizes {list(size_to_stride.keys())} with strides {list(size_to_stride.values())}")
            produced = process_one_session(session_path, user, session_name, out_dir, size_to_stride)
            for k, v in produced.items():
                produced_total[k] += v
    total_imgs = sum(produced_total.values())
    print("="*50)
    for k in sorted(produced_total):
        print(f"event{k}: {produced_total[k]} images (stride={size_to_stride[k]})")
    print(f"TOTAL images: {total_imgs}")
    return produced_total, total_imgs

# ----------------------------
# Utility
# ----------------------------
def count_total_events(data_dir):
    total = 0
    for user in sorted(os.listdir(data_dir)):
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"):
                continue
            path = os.path.join(user_dir, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                n = sum(1 for _ in f) - 1
            total += max(n, 0)
    return total

def parse_stride_map(s):
    m = {}
    if not s:
        return m
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split(":")
        m[int(k)] = max(1, int(v))
    return m

def plan_strides_per_size(total_events, sizes, target_images):
    size_to_stride = {}
    for k in sizes:
        base = max(1, total_events // k)
        best_stride = 1
        best_diff = float("inf")
        for stride in range(1, k + 1):
            est = int(base * (k / stride))
            diff = abs(est - target_images)
            if diff < best_diff:
                best_diff = diff
                best_stride = stride
        size_to_stride[k] = best_stride
    return size_to_stride

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Chunk Balabit mouse logs and render images with sliding window."
    )
    p.add_argument("--data_dir", type=str, default="Data/Balabit-dataset/training_files",
                   help="Root of Balabit dataset (users/session files).")
    p.add_argument("--out_dir", type=str, default="Images/Chunck/Balabit_chunks_baseline",
                   help="Output directory.")
    p.add_argument("--sizes", type=int, nargs="+", default=[60, 80, 100, 120, 130],
                   help="Chunk sizes to generate, e.g., --sizes 60 80 100 120 130")
    p.add_argument("--sessions", type=str, nargs="+", default=[], #
               help="List of session names to process (without .csv extension). Example: --sessions session_0041905381 session_1060325796")

    p.add_argument("--strides", type=str, default="",
                   help='Explicit stride map like "60:12,80:16,100:20".')
    p.add_argument("--target_images", type=int, default=54418,
                   help="Target minimum total images when auto-planning strides.")
    p.add_argument("--print_plan_only", action="store_true",
                   help="Only print planned strides & estimated totals, do not render images.")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    explicit = parse_stride_map(args.strides)
    sizes = sorted(set(int(s) for s in args.sizes))

    if explicit:
        size_to_stride = {k: explicit.get(k, k) for k in sizes}
        print("[Stride] Using explicit:", size_to_stride)
    else:
        total_events = count_total_events(args.data_dir)
        print(f"[Info] Total events counted: {total_events}")
        size_to_stride = plan_strides_per_size(total_events, sizes, args.target_images)
        print("[Stride] Auto-planned:", size_to_stride)

    if args.print_plan_only:
        total_events = count_total_events(args.data_dir)
        est_total = 0
        for k in sizes:
            base = max(1, total_events // k)
            est_total += int(base * (k / size_to_stride[k]))
        print(f"[Estimate] total images ≈ {est_total} (target >= {args.target_images})")
        return

    process_dataset(args.data_dir, args.out_dir, size_to_stride, target_sessions=args.sessions)

if __name__ == "__main__":
    main()
