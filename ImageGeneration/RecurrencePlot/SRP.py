# save as: RP_x_y.py

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def compute_rp_xy(seq, percentile=80):
    """
    Compute RP directly from raw (x, y) coordinates.
    seq: shape (T, 3) = [x, y, t]
    percentile: epsilon threshold (% of points kept)
    """

    # 1. 去掉 padding
    non_zero_mask = ~np.all(seq == 0, axis=1)
    seq = seq[non_zero_mask]
    if len(seq) == 0:
        return None

    # 2. 直接取 (x, y)
    coords = seq[:, :2]       # shape (T, 2)

    # 3. pairwise distance on x,y directly
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # shape (T, T)

    # 4. epsilon by percentile
    epsilon = np.percentile(dist, percentile)

    # 5. RP matrix
    rec = np.where(dist <= epsilon, dist, epsilon).astype(np.float32)

    # 6. normalize [0,1]
    if rec.max() > rec.min():
        rec = (rec - rec.min()) / (rec.max() - rec.min())

    return rec


def draw_rp_for_folder(script_root, data_path, out_path, img_size=224, percentile=95, sparse_users=None):

    # === Convert to absolute paths relative to project root ===
    data_path = os.path.join(script_root, data_path)
    out_path = os.path.join(script_root, out_path)

    os.makedirs(out_path, exist_ok=True)

    print(f"Input: {data_path}")
    print(f"Output: {out_path}")

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".npy"):
            continue

        user_id = filename.replace(".npy", "")

        if sparse_users is not None and user_id not in sparse_users:
            continue

        user_npy = os.path.join(data_path, filename)
        save_dir = os.path.join(out_path, user_id)
        os.makedirs(save_dir, exist_ok=True)

        print(f"Processing {user_id} ...")

        X = np.load(user_npy)      # shape (N, T, 3)

        for i, seq in enumerate(X):
            rp = compute_rp_xy(seq, percentile)
            if rp is None:
                continue

            # save image
            fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
            ax.imshow(rp, cmap="gray_r", origin="lower")
            ax.axis("off")
            ax.set_position([0, 0, 1, 1])  # remove white borders

            save_path = os.path.join(save_dir, f"{user_id}_rp{i}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig)

    print("RP generation finished!")


# ======================================================================
#                               MAIN
# ======================================================================
if __name__ == "__main__":

    # ==== get root dir (two levels up from this script) ====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_root = os.path.dirname(os.path.dirname(script_dir))
    # script_root now == "Sasha-clean/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Relative path from project root to data folder")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Relative path from project root to output folder")
    parser.add_argument("--percentile", type=float, default=95)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--user", nargs="+")
    args = parser.parse_args()

    draw_rp_for_folder(
        script_root,
        args.data_path,
        args.out_path,
        img_size=args.size,
        percentile=args.percentile,
        sparse_users=args.user
    )
