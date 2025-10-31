import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

'''
Generate Symmetric Recurrence Plots (SRP) from binary dataset using 2D features (dx, dy).
Each segment (sequence) is transformed into its own SRP image.
All output images are fixed 224x224 with no white borders.
'''


def compute_srp(seq, percentile=80):
    """
    Compute SRP for one sequence with percentile-based epsilon.
    seq: shape (T, 3), dx, dy, time
    percentile: 保留百分比 (e.g. 80 表示保留80%的点对)
    """
    # 1. 去掉 padding
    non_zero_mask = ~np.all(seq == 0, axis=1)
    seq = seq[non_zero_mask]
    if len(seq) == 0:
        return None

    # 2. (dx,dy) → (x,y)
    seq_2d = seq[:, :2]
    coords = np.cumsum(seq_2d, axis=0)

    # 3. pairwise distance
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # 4. 动态 epsilon by percentile
    epsilon = np.percentile(dist, percentile)

    # 5. 构造 SRP
    rec_matrix = np.where(dist <= epsilon, dist, epsilon).astype(np.float32)

    # 6. normalize to [0,1]
    if rec_matrix.max() > rec_matrix.min():
        rec_matrix = (rec_matrix - rec_matrix.min()) / (rec_matrix.max() - rec_matrix.min())

    return rec_matrix


def draw_recurrence_plots(data_path=None, out_path=None, img_size=224, ratio=95, sparse_users=None):
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))

    if data_path is None:
        data_path = os.path.join(root_dir, 'Data', 'Balabit_PeakClick/training')
    if out_path is None:
        out_path = os.path.join(root_dir, 'Images', 'Balabit_srp/training')

    print(f"Data path: {data_path}")
    print(f"Output path: {out_path}")
    os.makedirs(out_path, exist_ok=True)

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".npy"):
            continue

        user_id = filename.replace(".npy", "")

        # 如果 sparse_users 不是 None，就只处理指定的 user
        if sparse_users is not None and user_id not in sparse_users:
            continue

        npy_path = os.path.join(data_path, filename)

        save_dir = os.path.join(out_path, f"{user_id}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Processing {user_id}...")

        X = np.load(npy_path)  # shape: (N, T, 3)

        for i, seq in enumerate(X):
            srp = compute_srp(seq, percentile=ratio)
            if srp is None:
                continue

            # Plot and save
            fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
            ax.imshow(srp, cmap='gray_r', origin='lower')
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])  # remove borders

            img_path = os.path.join(save_dir, f"{user_id}_srp{i}.png")
            plt.savefig(img_path, dpi=100)
            plt.close(fig)

    print("✅ SRP generation done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SRP images for users")
    parser.add_argument("--user", nargs="+", help="指定要处理的用户ID, 例如: --user user7 user9")
    parser.add_argument("--data_path", type=str, default=None, help="输入数据路径 (默认=Data/Balabit_PeakClick/training)")
    parser.add_argument("--out_path", type=str, default=None, help="输出图像路径 (默认=Images/Balabit_srp/training)")
    parser.add_argument("--ratio", type=float, default=95, help="percentile ratio (默认=95)")
    args = parser.parse_args()

    draw_recurrence_plots(data_path=args.data_path, out_path=args.out_path,
                          ratio=args.ratio, sparse_users=args.user)
