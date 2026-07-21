"""
Split TWOS sessions into protocol1 train/test (same as DFL):
  training_files          = first 2/3 of each session
  testing_files_protocol1 = last 1/3 of each session

Source: Data/TWOS/User*/session*.csv
Output:
  Data/TWOS/training_files/User*/session*.csv
  Data/TWOS/testing_files_protocol1/User*/session*.csv
"""
import os
import re
import csv
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "TWOS"))
TRAIN_DIR = os.path.join(ROOT, "training_files")
TEST_DIR = os.path.join(ROOT, "testing_files_protocol1")


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", string)]


def is_user_dir(name):
    return name.startswith("User") and os.path.isdir(os.path.join(ROOT, name))


def split_file(src_path, train_path, test_path):
    with open(src_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        rows = list(reader)

    n = len(rows)
    if n == 0:
        return 0, 0

    split = (n * 2) // 3
    # ensure both sides nonempty when possible
    if n >= 2:
        split = max(1, min(split, n - 1))

    train_rows = rows[:split]
    test_rows = rows[split:]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(train_rows)

    with open(test_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(test_rows)

    return len(train_rows), len(test_rows)


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"[Error] Not found: {ROOT}")

    # clean previous split outputs if present
    for d in (TRAIN_DIR, TEST_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)

    users = sorted([u for u in os.listdir(ROOT) if is_user_dir(u)], key=natural_key)

    total_src = 0
    total_train = 0
    total_test = 0
    n_files = 0

    print(f"Source: {ROOT}")
    print(f"Train : {TRAIN_DIR}")
    print(f"Test  : {TEST_DIR}")
    print(f"Users : {len(users)}")
    print()

    for user in users:
        user_dir = os.path.join(ROOT, user)
        sessions = sorted(
            [f for f in os.listdir(user_dir) if f.endswith(".csv")],
            key=natural_key,
        )
        print(f"[{user}] {len(sessions)} sessions")

        for fn in sessions:
            src = os.path.join(user_dir, fn)
            train_path = os.path.join(TRAIN_DIR, user, fn)
            test_path = os.path.join(TEST_DIR, user, fn)
            n_train, n_test = split_file(src, train_path, test_path)
            n = n_train + n_test
            total_src += n
            total_train += n_train
            total_test += n_test
            n_files += 1
            ratio = (n_train / n) if n else 0.0
            print(f"  {fn}: {n:,} -> train {n_train:,} ({ratio:.1%}) | test {n_test:,}")

    print("-" * 60)
    print(f"files: {n_files}")
    print(f"rows total : {total_src:,}")
    print(f"rows train : {total_train:,} ({100 * total_train / total_src:.2f}%)")
    print(f"rows test  : {total_test:,} ({100 * total_test / total_src:.2f}%)")


if __name__ == "__main__":
    main()
