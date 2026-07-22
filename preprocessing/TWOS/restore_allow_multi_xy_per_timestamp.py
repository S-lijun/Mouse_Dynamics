"""
Restore TWOS to: cleaned + exact-(t,x,y) dedupe ONLY
(still allow multiple x,y under the same timestamp).

Rebuilds Data/TWOS/User* from Data/TWOS-dataset, then:
  1) clean missing timestamp/x/y
  2) rename sessions -> session1.csv, session2.csv, ... by start time
  3) drop exact duplicate (timestamp, x, y)
  4) regenerate training_files / testing_files_protocol1
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


def copy_users_from_dataset():
    if not os.path.isdir(SRC):
        raise SystemExit(f"[Error] Source not found: {SRC}")
    wipe_user_dirs(ROOT)
    users = sorted([u for u in os.listdir(SRC) if is_user_dir(SRC, u)], key=natural_key)
    for user in users:
        src_u = os.path.join(SRC, user)
        dst_u = os.path.join(ROOT, user)
        shutil.copytree(src_u, dst_u)
        print(f"  copied {user}")
    return users


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
    print("[1/4] Copy TWOS-dataset -> TWOS/User*")
    copy_users_from_dataset()

    clean_script = os.path.join(os.path.dirname(__file__), "clean_twos_rows.py")
    print(f"\n[2/4] Clean missing rows via {clean_script}")
    subprocess.check_call([sys.executable, clean_script, ROOT])

    print("\n[3/4] Rename to sessionN.csv by start timestamp")
    rename_sessions_by_time()

    exact_script = os.path.join(os.path.dirname(__file__), "dedupe_exact_rows.py")
    print(f"\n[4/4] Exact (t,x,y) dedupe + train/test split via {exact_script}")
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
