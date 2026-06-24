"""Remove TWOS CSV rows missing timestamp, x, or y."""
import csv
import os
import shutil
import sys
import tempfile

DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Data", "TWOS")
)


def row_is_valid(row):
    ts = (row.get("timestamp") or "").strip()
    x = (row.get("x") or "").strip()
    y = (row.get("y") or "").strip()
    if not ts or not x or not y:
        return False
    try:
        float(ts)
        float(x)
        float(y)
    except ValueError:
        return False
    return True


def clean_file(path):
    removed = 0
    kept = 0
    fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=os.path.dirname(path))
    os.close(fd)
    try:
        with open(path, "r", encoding="utf-8", newline="") as fin, open(
            tmp_path, "w", encoding="utf-8", newline=""
        ) as fout:
            reader = csv.DictReader(fin)
            if reader.fieldnames is None:
                return 0, 0
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row_is_valid(row):
                    writer.writerow(row)
                    kept += 1
                else:
                    removed += 1
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return kept, removed


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    if not os.path.isdir(root):
        print(f"[Error] Not found: {root}")
        sys.exit(1)

    total_kept = 0
    total_removed = 0
    empty_files = []

    csv_files = []
    for user_dir in sorted(os.listdir(root)):
        up = os.path.join(root, user_dir)
        if not os.path.isdir(up) or user_dir.startswith("."):
            continue
        for fn in sorted(os.listdir(up)):
            if fn.endswith(".csv"):
                csv_files.append(os.path.join(up, fn))

    print(f"Cleaning {len(csv_files)} files under {root}")
    for fp in csv_files:
        rel = os.path.relpath(fp, root)
        kept, removed = clean_file(fp)
        total_kept += kept
        total_removed += removed
        if kept == 0:
            empty_files.append(rel)
        if removed:
            print(f"  {rel}: removed {removed:,}, kept {kept:,}")

    print("-" * 60)
    print(f"Total removed: {total_removed:,}")
    print(f"Total kept:    {total_kept:,}")
    if empty_files:
        print(f"Empty sessions after clean ({len(empty_files)}):")
        for f in empty_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
