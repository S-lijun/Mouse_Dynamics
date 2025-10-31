# preprocessing/Chunking/make_fixed_event_chunks.py
import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Make fixed-length event chunks per user")
    p.add_argument("--data_root", default="Data/Gmail_Dataset",
                   help="Root folder that contains user* folders with session_*.csv")
    p.add_argument("--out_root", default="Data/Chunck",
                   help="Output root folder. Script will create event{size}/user{idx}.npy")
    p.add_argument("--sizes", default="60,80,100,120,130",
                   help="Comma-separated chunk sizes, e.g. 60,80,100")
    p.add_argument("--include_types", default="movement",
                   help='Comma-separated event types to include (default: "movement")')
    p.add_argument("--stride", type=int, default=0,
                   help="Stride for sliding window. 0 means non-overlapping (stride = chunk_size).")
    p.add_argument("--sort_by_ts", action="store_true",
                   help="Sort session rows by TimeStamp before chunking (recommended if order not guaranteed).")
    return p.parse_args()

def read_session_csv(path, allowed_types, sort_by_ts=False):
    """
    Return ndarray of shape (N, 3) with [x, y, t].
    Robust to header variants like 'Mouse X' vs 'Mouse_X', 'Timestamp' vs 'TimeStamp'.
    """
    df = pd.read_csv(path, encoding="utf-8")

    # --- robust header normalization ---
    def canon(s: str) -> str:
        # lower + remove spaces/underscores and non-alnum
        s = s.lower()
        s = "".join(ch for ch in s if ch.isalnum())   # keep [a-z0-9]
        return s

    colmap = {c: canon(c) for c in df.columns}

    # find columns by canonical key
    def find_col(target_keys):
        for orig, can in colmap.items():
            if can in target_keys:
                return orig
        return None

    # possible keys for each required field
    x_col  = find_col({"mousex", "x", "cursorx"})
    y_col  = find_col({"mousey", "y", "cursory"})
    t_col  = find_col({"timestamp", "time", "unix", "unixtime", "timeepoch"})
    typ_col = find_col({"type", "eventtype"})

    missing = []
    if x_col is None: missing.append("Mouse_X")
    if y_col is None: missing.append("Mouse_Y")
    if t_col is None: missing.append("TimeStamp")
    if missing:
        raise ValueError(f"{path} missing column(s): {', '.join(missing)}")

    # optional type filter
    if typ_col is not None and allowed_types:
        df = df[df[typ_col].astype(str).str.lower().isin([t.lower() for t in allowed_types])]
        # 如果全被过滤掉，直接返回空
        if df.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)

    if sort_by_ts:
        df = df.sort_values(t_col, kind="mergesort")

    # 只保留所需三列，转成数值
    out = df[[x_col, y_col, t_col]].copy()
    for c in [x_col, y_col, t_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    return out.to_numpy()


def make_chunks(arr, chunk_size, stride=0):
    """arr: (N,3). Return chunks (M, chunk, 3)."""
    n = arr.shape[0]
    if stride <= 0:
        stride = chunk_size  # non-overlapping
    if n < chunk_size:
        return np.empty((0, chunk_size, 3), dtype=arr.dtype)
    # number of starting indices
    idxs = range(0, n - chunk_size + 1, stride)
    chunks = [arr[i:i+chunk_size] for i in idxs]
    return np.stack(chunks, axis=0) if chunks else np.empty((0, chunk_size, 3), dtype=arr.dtype)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def process_user(user_dir, out_user_path, chunk_sizes, allowed_types, sort_by_ts, stride_arg):
    # gather all sessions
    session_paths = sorted(glob.glob(os.path.join(user_dir, "session_*.csv")))
    if not session_paths:
        return {size: 0 for size in chunk_sizes}

    # read all sessions, concatenate
    all_arrs = []
    for sp in session_paths:
        try:
            arr = read_session_csv(sp, allowed_types, sort_by_ts=sort_by_ts)
            if arr.size:
                all_arrs.append(arr)
        except Exception as e:
            print(f"Skip session {sp}: {e}")
    if not all_arrs:
        return {size: 0 for size in chunk_sizes}

    data = np.vstack(all_arrs)  # (N,3)

    stats = {}
    for size in chunk_sizes:
        stride = stride_arg if stride_arg > 0 else size
        chunks = make_chunks(data, size, stride=stride)  # (M, size, 3)
        # save
        ensure_dir(out_user_path[size])
        np.save(os.path.join(out_user_path[size], f"{os.path.basename(user_dir).replace('user','')}.npy"), chunks)
        stats[size] = int(chunks.shape[0])
    return stats

def main():
    args = parse_args()
    chunk_sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    allowed_types = [t.strip() for t in args.include_types.split(",") if t.strip()]

    # find users
    user_dirs = sorted([p for p in glob.glob(os.path.join(args.data_root, "user*")) if os.path.isdir(p)],
                       key=lambda p: int(os.path.basename(p).replace("user", "")))
    if not user_dirs:
        raise SystemExit(f"No users found under {args.data_root}")

    # prepare out dirs per chunk size
    out_dirs_per_size = {size: os.path.join(args.out_root, f"event{size}") for size in chunk_sizes}
    for d in out_dirs_per_size.values():
        ensure_dir(d)

    print(f"Found {len(user_dirs)} users. Chunk sizes: {chunk_sizes}. Types: {allowed_types}")
    grand_stats = {size: 0 for size in chunk_sizes}

    for ud in tqdm(user_dirs, desc="Users"):
        out_user_path = {size: out_dirs_per_size[size] for size in chunk_sizes}
        stats = process_user(
            user_dir=ud,
            out_user_path=out_user_path,
            chunk_sizes=chunk_sizes,
            allowed_types=allowed_types,
            sort_by_ts=args.sort_by_ts,
            stride_arg=args.stride
        )
        for size, cnt in stats.items():
            grand_stats[size] += cnt

    print("Done. Total sequences per chunk size:")
    for size in chunk_sizes:
        print(f"  event{size}: {grand_stats[size]} sequences")

if __name__ == "__main__":
    main()
