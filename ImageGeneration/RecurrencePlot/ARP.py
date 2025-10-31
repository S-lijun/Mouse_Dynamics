import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

'''
Generate Asymmetric Recurrence Plots (ARP) from binary dataset using 2D features (dx, dy).
Each segment is divided into two halves. Each half generates its own SRP,
then the ARP is constructed by combining the upper triangle of SRP1 and
the lower triangle of SRP2. All output images are fixed 224x224 with no borders.

'''


def compute_arp(seq, percentile=80):
    """
    Compute ARP for one sequence with percentile-based epsilon.
    seq: shape (T, 3), dx, dy, time
    percentile: 保留百分比 (e.g. 80 表示保留80%的点对)
    """
    # 1. 去掉 padding
    non_zero_mask = ~np.all(seq == 0, axis=1)
    seq = seq[non_zero_mask]
    if len(seq) < 2:
        return None

    # 如果长度为奇数，丢弃最后一个点，保证能平分
    if len(seq) % 2 != 0:
        seq = seq[:-1]

    # 2. 分成两半
    mid = len(seq) // 2
    seq1, seq2 = seq[:mid], seq[mid:]
    if len(seq1) < 2 or len(seq2) < 2:
        return None

    # 3. 分别生成 SRP1 和 SRP2
    def compute_srp_half(s):
        seq_2d = s[:, :2]
        coords = np.cumsum(seq_2d, axis=0)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        epsilon = np.percentile(dist, percentile)
        rec_matrix = np.where(dist <= epsilon, dist, epsilon).astype(np.float32)
        if rec_matrix.max() > rec_matrix.min():
            rec_matrix = (rec_matrix - rec_matrix.min()) / (rec_matrix.max() - rec_matrix.min())
        return rec_matrix

    srp1 = compute_srp_half(seq1)
    srp2 = compute_srp_half(seq2)
    if srp1 is None or srp2 is None:
        return None

    # 4. 取上三角和下三角拼接
    upper = np.triu(srp1)
    lower = np.tril(srp2)
    arp = upper + lower

    # 5. normalize to [0,1]
    if arp.max() > arp.min():
        arp = (arp - arp.min()) / (arp.max() - arp.min())

    return arp


def draw_recurrence_plots(data_path=None, out_path=None, img_size=224, ratio=95, sparse_users=None):
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))

    if data_path is None:
        data_path = os.path.join(root_dir, 'Data', 'Balabit_PeakClick/training')
    if out_path is None:
        out_path = os.path.join(root_dir, 'Images', 'Balabit_arp/training')

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
            arp = compute_arp(seq, percentile=ratio)
            if arp is None:
                continue

            # Plot and save
            fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
            ax.imshow(arp, cmap='gray_r', origin='lower')
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])  # remove borders

            img_path = os.path.join(save_dir, f"{user_id}_arp{i}.png")
            plt.savefig(img_path, dpi=100)
            plt.close(fig)

    print("ARP generation done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ARP images for users")
    parser.add_argument("--user", nargs="+", help="指定要处理的用户ID, 例如: --user user7 user9")
    parser.add_argument("--data_path", type=str, default=None, help="输入数据路径 (默认=Data/Balabit_PeakClick/training)")
    parser.add_argument("--out_path", type=str, default=None, help="输出图像路径 (默认=Images/Balabit_arp/training)")
    parser.add_argument("--ratio", type=float, default=95, help="percentile ratio (默认=95)")
    args = parser.parse_args()

    draw_recurrence_plots(data_path=args.data_path, out_path=args.out_path,
                          ratio=args.ratio, sparse_users=args.user)
