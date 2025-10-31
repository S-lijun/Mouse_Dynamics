import os
import csv
import numpy as np
from tqdm import tqdm

'''Trajectory Segmentation for Balabit Dataset'''

# ==== directory ====
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TRAIN_DIR = os.path.join(DATA_ROOT, 'Data', 'Balabit-dataset', 'training_files')
TEST_DIR = os.path.join(DATA_ROOT, 'Data', 'Balabit-dataset', 'test_files')
OUT_TRAIN = os.path.join(DATA_ROOT, 'Data', 'Balabit', 'training')
OUT_TEST = os.path.join(DATA_ROOT, 'Data', 'Balabit', 'testing')

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

# ==== load CSV with [x, y, t] ====
def extract_xyz(file_path):
    xyt = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['x'])
                y = float(row['y'])
                t = float(row['client timestamp'])
                xyt.append([x, y, t])
            except:
                continue
    return xyt

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

# ==== Padding to global_max_len ====
def pad_segments(segments, max_len, pad_val=0.0):
    padded = []
    for seg in segments:
        seg_arr = np.array(seg)
        pad_len = max_len - len(seg)
        if pad_len > 0:
            pad = np.full((pad_len, 3), pad_val)
            seg_arr = np.vstack([seg_arr, pad])
        padded.append(seg_arr)
    return np.stack(padded)

# ==== find segment global max ====
def collect_all_segment_lengths(in_dirs):
    max_len = 0
    for in_dir in in_dirs:
        for user in os.listdir(in_dir):
            user_path = os.path.join(in_dir, user)
            for session in os.listdir(user_path):
                xyt = extract_xyz(os.path.join(user_path, session))
                segments = segment_by_velocity(xyt)
                for seg in segments:
                    max_len = max(max_len, len(seg))
    return max_len

# ==== save as .npy ====
def process_folder(in_dir, out_dir, global_max_len):
    for user in tqdm(sorted(os.listdir(in_dir)), desc=f'Processing {in_dir}'):
        user_path = os.path.join(in_dir, user)
        all_segments = []
        for session in os.listdir(user_path):
            session_path = os.path.join(user_path, session)
            xyt = extract_xyz(session_path)
            segments = segment_by_velocity(xyt)
            all_segments.extend(segments)
        if all_segments:
            padded = pad_segments(all_segments, max_len=global_max_len)
            np.save(os.path.join(out_dir, f'{user}.npy'), padded)

# ==== main ====
if __name__ == "__main__":
    print("Collecting global max segment length...")
    global_max_len = collect_all_segment_lengths([TRAIN_DIR, TEST_DIR])
    print(f"[INFO] Global max segment length = {global_max_len}")

    process_folder(TRAIN_DIR, OUT_TRAIN, global_max_len)
    process_folder(TEST_DIR, OUT_TEST, global_max_len)
