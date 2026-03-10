import os
import re
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def parse_session_and_index(filename):
    m = re.match(r"(session_\d+)-(\d+)\.png", filename)
    if m is None:
        raise RuntimeError(f"Bad filename: {filename}")
    return m.group(1), int(m.group(2))


def count_images(split_root):

    total = 0
    user_list = []

    for u in os.listdir(split_root):

        user_path = split_root / u

        if not os.path.isdir(user_path):
            continue

        user_list.append(u)

        total += len([f for f in os.listdir(user_path) if f.endswith(".png")])

    return sorted(user_list), total


def build_dataset(split_root, out_dir, img_size=224):

    split_root = Path(split_root)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nScanning dataset...")

    user_list, total_images = count_images(split_root)

    num_users = len(user_list)

    print("Users:", num_users)
    print("Total images:", total_images)

    user2index = {u: i for i, u in enumerate(user_list)}

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    img_file = np.memmap(
        out_dir / "images.npy",
        dtype="float32",
        mode="w+",
        shape=(total_images, 3, img_size, img_size)
    )

    label_file = np.memmap(
        out_dir / "labels.npy",
        dtype="float32",
        mode="w+",
        shape=(total_images, num_users)
    )

    session_file = []

    idx = 0

    for user in user_list:

        user_path = split_root / user

        files = []

        for f in os.listdir(user_path):

            if not f.endswith(".png"):
                continue

            try:
                sess, i = parse_session_and_index(f)
                files.append((sess, i, f))
            except:
                continue

        files.sort(key=lambda x: (x[0], x[1]))

        print(f"{user} : {len(files)} images")

        for sess, _, fname in tqdm(files):

            img_path = user_path / fname

            img = Image.open(img_path).convert("RGB")
            img = transform(img)

            img_file[idx] = img.numpy()

            y = np.zeros(num_users, dtype=np.float32)
            y[user2index[user]] = 1.0

            label_file[idx] = y

            session_file.append(sess)

            idx += 1

    np.save(out_dir / "sessions.npy", np.array(session_file))

    print("\nDataset built successfully.")
    print("Saved to:", out_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    build_dataset(args.input, args.output)