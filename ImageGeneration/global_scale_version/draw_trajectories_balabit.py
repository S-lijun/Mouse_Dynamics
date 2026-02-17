# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Path & Fixed Config
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# 默认指向训练集
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")
#DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "testing_files_protocol1")

# 固定图像尺寸为 448x448
FIXED_IMG_SIZE = 224
# 固定线宽和点径
FIXED_LINEWIDTH = 0.5
FIXED_MARKERSIZE = 1.0
DPI = 200

# ============================================================
# Drawing Function (Fixed Canvas Size)
# ============================================================
def draw_mouse_chunk(chunk, save_path, global_bounds):
    if len(chunk) == 0:
        return

    SIDE = FIXED_IMG_SIZE
    lw = FIXED_LINEWIDTH
    ms = FIXED_MARKERSIZE
    
    g_min_x, g_max_x, g_min_y, g_max_y = global_bounds
    range_x = max(g_max_x - g_min_x, 1.0)
    range_y = max(g_max_y - g_min_y, 1.0)

    # 创建固定大小的正方形画布
    fig, ax = plt.subplots(figsize=(SIDE/DPI, SIDE/DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 设置固定坐标轴范围 [0, SIDE]
    ax.set_xlim(0, SIDE)
    ax.set_ylim(SIDE, 0) # 0在顶端，实现向上对齐
    ax.axis("off")

    # 映射计算：y 轴按物理长宽比占据对应高度
    y_target_height = (range_y / range_x) * SIDE
    
    coords = np.array([[float(e["x"]), float(e["y"])] for e in chunk])
    
    # 归一化映射逻辑
    norm_x = (coords[:, 0] - g_min_x) / range_x
    norm_y = (coords[:, 1] - g_min_y) / range_y
    
    x_pix = norm_x * SIDE
    y_pix = norm_y * y_target_height 

    # 绘图
    ax.plot(x_pix, y_pix, color="black", linewidth=lw, 
            marker="o", markersize=ms, markerfacecolor="black", markeredgewidth=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ============================================================
# Data Cleaning (Discard 65535)
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"x": "x", "y": "y", "state": "state"})
    df = df[df["state"] == "Move"].copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    
    # 丢弃 65535 异常值行
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    return df.dropna(subset=["x", "y"])

# ============================================================
# Processing Logic
# ============================================================
def process_one_session(path, user, session_name, out_dir, sizes):
    df = pd.read_csv(path)
    df = clean_and_rename_cols(df)
    if df.empty: return {s: 0 for s in sizes}

    g_bounds = (df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max())
    events = df.to_dict(orient="records")
    
    results = {}
    for sz in sizes:
        n_chunks = len(events) // sz
        for i in range(n_chunks):
            chunk = events[i*sz : (i+1)*sz]
            save_path = os.path.join(out_dir, f"event{sz}", user, f"{session_name}-{i}.png")
            draw_mouse_chunk(chunk, save_path, g_bounds)
        results[sz] = n_chunks
    return results

# ============================================================
# CLI with User/Session filtering
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=f"Images/fixed_{FIXED_IMG_SIZE}_padding")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60])
    # 添加可选的过滤参数
    parser.add_argument("--users", type=str, nargs="+", default=[])
    parser.add_argument("--sessions", type=str, nargs="+", default=[])
    args = parser.parse_args()

    full_out_dir = os.path.join(ROOT, args.out_dir)
    target_users = set(args.users) if args.users else None
    target_sessions = set(args.sessions) if args.sessions else None

    # 遍历数据集
    for user in sorted(os.listdir(DATA_ROOT)):
        if target_users and user not in target_users:
            continue
            
        user_dir = os.path.join(DATA_ROOT, user)
        if not os.path.isdir(user_dir): 
            continue
            
        for file in sorted(os.listdir(user_dir)):
            if not file.startswith("session_"): 
                continue
                
            session_name = os.path.splitext(file)[0]
            if target_sessions and session_name not in target_sessions:
                continue

            print(f"[Processing] {user}/{file} (Fixed {FIXED_IMG_SIZE} Output)")
            process_one_session(os.path.join(user_dir, file), user, 
                               session_name, full_out_dir, args.sizes)

if __name__ == "__main__":
    main()