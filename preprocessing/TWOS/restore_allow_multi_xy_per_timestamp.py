"""
Restore TWOS to: cleaned + exact-(t,x,y) dedupe ONLY
(still allow multiple x,y under the same timestamp).

Rebuilds Data/TWOS/User* from Data/TWOS-dataset, then:
  1) copy only User*.log*.csv session files (skip UserN.csv dumps)
  2) clean missing timestamp/x/y
  3) drop empty sessions, then rename -> session1.csv, ... by start time
  4) drop exact duplicate (timestamp, x, y)
  5) regenerate training_files / testing_files_protocol1
"""
import os
import re
import csv
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "TWOS"))
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "TWOS-dataset"))


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", string)]


def is_user_dir(path, name):
    return name.startswith("User") and os.path.isdir(os.path.join(path, name))


def wipe_user_dirs(root):
    for name in os.listdir(root):
        if is_user_dir(root, name):
            shutil.rmtree(os.path.join(root, name))


def is_session_log(filename):
    # User1.log.csv, User1.log.2017-03-21_18.csv — not User1.csv
    return filename.endswith(".csv") and ".log" in filename


def copy_users_from_dataset():
    if not os.path.isdir(SRC):
        raise SystemExit(f"[Error] Source not found: {SRC}")
    wipe_user_dirs(ROOT)
    users = sorted([u for u in os.listdir(SRC) if is_user_dir(SRC, u)], key=natural_key)
    for user in users:
        src_u = os.path.join(SRC, user)
        dst_u = os.path.join(ROOT, user)
        os.makedirs(dst_u, exist_ok=True)
        n = 0
        for fn in os.listdir(src_u):
            if not is_session_log(fn):
                if fn.endswith(".csv"):
                    print(f"  skip {user}/{fn}")
                continue
            shutil.copy2(os.path.join(src_u, fn), os.path.join(dst_u, fn))
            n += 1
        print(f"  copied {user}: {n} sessions")
    return users


def remove_empty_sessions():
    removed = 0
    for user in sorted(
        [u for u in os.listdir(ROOT) if is_user_dir(ROOT, u)], key=natural_key
    ):
        user_dir = os.path.join(ROOT, user)
        for fn in list(os.listdir(user_dir)):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(user_dir, fn)
            ts = pd.read_csv(fp, usecols=["timestamp"], low_memory=False)["timestamp"]
            ts = pd.to_numeric(ts, errors="coerce").dropna()
            if len(ts) == 0:
                os.remove(fp)
                removed += 1
                print(f"  remove empty {user}/{fn}")
    print(f"  removed empty files: {removed}")


def rename_sessions_by_time():
    users = sorted([u for u in os.listdir(ROOT) if is_user_dir(ROOT, u)], key=natural_key)
    for user in users:
        user_dir = os.path.join(ROOT, user)
        files = [f for f in os.listdir(user_dir) if f.endswith(".csv")]
        items = []
        for fn in files:
            fp = os.path.join(user_dir, fn)
            ts = pd.read_csv(fp, usecols=["timestamp"], low_memory=False)["timestamp"]
            ts = pd.to_numeric(ts, errors="coerce").dropna()
            start = int(ts.iloc[0]) if len(ts) else 0
            items.append((start, fp))
        items.sort(key=lambda x: (x[0], x[1]))

        temps = []
        for i, (_, fp) in enumerate(items, 1):
            tmp = os.path.join(user_dir, f"__tmp_session{i}.csv")
            os.rename(fp, tmp)
            temps.append((tmp, os.path.join(user_dir, f"session{i}.csv")))
        for tmp, final in temps:
            os.rename(tmp, final)
        print(f"  renamed {user}: {len(items)} sessions")


def main():
    os.makedirs(ROOT, exist_ok=True)
    print("[1/5] Copy TWOS-dataset log sessions -> TWOS/User*")
    copy_users_from_dataset()

    clean_script = os.path.join(os.path.dirname(__file__), "clean_twos_rows.py")
    print(f"\n[2/5] Clean missing rows via {clean_script}")
    subprocess.check_call([sys.executable, clean_script, ROOT])

    print("\n[3/5] Drop empty sessions")
    remove_empty_sessions()

    print("\n[4/5] Rename to sessionN.csv by start timestamp")
    rename_sessions_by_time()

    exact_script = os.path.join(os.path.dirname(__file__), "dedupe_exact_rows.py")
    print(f"\n[5/5] Exact (t,x,y) dedupe + train/test split via {exact_script}")
    subprocess.check_call([sys.executable, exact_script])

    # quick verify: same timestamp can still have multiple rows
    sample = os.path.join(ROOT, "User1", "session1.csv")
    df = pd.read_csv(sample, usecols=["timestamp", "x", "y"])
    n = len(df)
    n_ts = df["timestamp"].nunique()
    print("\n[verify] User1/session1.csv")
    print(f"  rows={n:,}  unique_ts={n_ts:,}  multi-ts-extra={n - n_ts:,}")
    print("Done. Same timestamp may still have multiple x,y; exact (t,x,y) dups removed.")


if __name__ == "__main__":
    main()
