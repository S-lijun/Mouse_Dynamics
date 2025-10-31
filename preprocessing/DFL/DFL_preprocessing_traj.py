import os
import csv
import numpy as np
from tqdm import tqdm

# ==== directory ====
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DFL_DIR = os.path.join(DATA_ROOT, 'Data', 'DFL Dataset')
OUT_DIR = os.path.join(DATA_ROOT, 'Data', 'DFL_processed')

os.makedirs(OUT_DIR, exist_ok=True)

# ==== load CSV with [x, y, t] ====
def extract_xyz(file_path):
    xyt = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row['client timestamp'])
                x = float(row['x'])
                y = float(row['y'])
                xyt.append([x, y, t])
            except:
                continue
    return np.array(xyt, dtype=np.float32)

# ==== velocity based cutting ====
def segment_by_velocity(xyt, v_thresh=1e-6):
    segments = []
    current = []
    for i in range(1, len(xyt)):
        x0, y0, t0 = xyt[i - 1]
        x1, y1, t1 = xyt[i]
        dt = max(t1 - t0, 1e-6)
        dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        velocity = dist / dt

        if velocity < v_thresh:
            if current:
                segments.append(current)
                current = []
        else:
            current.append([x1, y1, t1])
    if current:
        segments.append(current)
    return segments

# ==== Padding (with truncation) ====
def pad_segments(segments, max_len, pad_val=0.0):
    padded = []
    for seg in segments:
        seg_arr = np.array(seg, dtype=np.float32)
        if len(seg_arr) > max_len:
            seg_arr = seg_arr[:max_len]  # cut
        pad_len = max_len - len(seg_arr)
        if pad_len > 0:
            pad = np.full((pad_len, 3), pad_val, dtype=np.float32)
            seg_arr = np.vstack([seg_arr, pad])
        padded.append(seg_arr)
    return np.stack(padded).astype(np.float32)

# ==== find segment length distribution ====
def collect_segment_lengths(in_dir):
    lengths = []
    for user in os.listdir(in_dir):
        user_path = os.path.join(in_dir, user)
        if not os.path.isdir(user_path):
            continue
        for file in os.listdir(user_path):
            if not file.endswith('.CSV'):
                continue
            xyt = extract_xyz(os.path.join(user_path, file))
            segments = segment_by_velocity(xyt)
            for seg in segments:
                lengths.append(len(seg))
    return lengths

# ==== save as .npy ====
def process_folder(in_dir, out_dir, global_max_len):
    for user in tqdm(sorted(os.listdir(in_dir)), desc=f'Processing {in_dir}'):
        user_path = os.path.join(in_dir, user)
        if not os.path.isdir(user_path):
            continue
        all_segments = []
        for file in os.listdir(user_path):
            if not file.endswith('.CSV'):
                continue
            session_path = os.path.join(user_path, file)
            xyt = extract_xyz(session_path)
            segments = segment_by_velocity(xyt)
            all_segments.extend(segments)
        if all_segments:
            padded = pad_segments(all_segments, max_len=global_max_len)
            np.save(os.path.join(out_dir, f'{user}.npy'), padded)

# ==== main ====
if __name__ == "__main__":
    print("Collecting segment lengths...")
    lengths = collect_segment_lengths(DFL_DIR)

    # 95% 分位数作为 max_len（你可以改成 99）
    global_max_len = int(np.percentile(lengths, 95))
    print(f"[INFO] Global max segment length (95% quantile) = {global_max_len}")

    process_folder(DFL_DIR, OUT_DIR, global_max_len)
