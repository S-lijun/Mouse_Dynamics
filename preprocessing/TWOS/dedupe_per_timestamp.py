"""
TWOS preprocess step 2: one row per timestamp.

For cases where the same timestamp has multiple (x, y) events,
keep the first occurrence only.

Operates in-place on Data/TWOS/User*/session*.csv, then regenerates
training_files / testing_files_protocol1 (2/3-1/3).
"""
import os
import re
import csv
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "TWOS"))


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", string)]


def is_user_dir(name):
    return name.startswith("User") and os.path.isdir(os.path.join(ROOT, name))


def keep_first_per_timestamp(path):
    with open(path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            return 0, 0
        fieldnames = reader.fieldnames
        rows = list(reader)

    before = len(rows)
    seen = set()
    kept = []
    for row in rows:
        ts = (row.get("timestamp") or "").strip()
        if ts in seen:
            continue
        seen.add(ts)
        kept.append(row)

    after = len(kept)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    shutil.move(tmp, path)
    return before, after


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"[Error] Not found: {ROOT}")

    users = sorted([u for u in os.listdir(ROOT) if is_user_dir(u)], key=natural_key)
    total_before = total_after = 0
    n_files = 0

    print(f"Keeping first row per timestamp under: {ROOT}")
    print(f"Users: {len(users)}\n")

    for user in users:
        user_dir = os.path.join(ROOT, user)
        sessions = sorted(
            [f for f in os.listdir(user_dir) if f.endswith(".csv")],
            key=natural_key,
        )
        for fn in sessions:
            path = os.path.join(user_dir, fn)
            before, after = keep_first_per_timestamp(path)
            removed = before - after
            total_before += before
            total_after += after
            n_files += 1
            if removed:
                print(f"  {user}/{fn}: {before:,} -> {after:,}  (-{removed:,})")

    print("-" * 60)
    print(f"files: {n_files}")
    print(f"before: {total_before:,}")
    print(f"after : {total_after:,}")
    if total_before:
        print(
            f"removed: {total_before - total_after:,} "
            f"({100 * (total_before - total_after) / total_before:.2f}%)"
        )

    split_script = os.path.join(os.path.dirname(__file__), "split_twos_protocol1.py")
    print(f"\nRegenerating train/test via {split_script} ...")
    subprocess.check_call([sys.executable, split_script])


if __name__ == "__main__":
    main()
