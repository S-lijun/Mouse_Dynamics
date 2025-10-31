import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 强制无GUI后端，避免HPC环境重复savefig
import matplotlib.pyplot as plt
import argparse

def draw_balabit_baseline(data_root='Data/Balabit_PeakClick', 
                          out_root='Images/Balabit_PeakClick',
                          target_user=None):
    for split in ['training', 'testing']:
        data_path = os.path.join(data_root, split)
        out_path = os.path.join(out_root, split)
        os.makedirs(out_path, exist_ok=True)

        for filename in sorted(os.listdir(data_path)):
            if not filename.endswith('.npy'):
                continue

            user_id = filename.replace('.npy', '')

            # 如果指定了 target_user，就跳过其他 user
            if target_user is not None and user_id != target_user:
                continue

            npy_path = os.path.join(data_path, filename)
            save_dir = os.path.join(out_path, user_id)
            os.makedirs(save_dir, exist_ok=True)

            print(f"[{split}] Drawing user {user_id} ...", flush=True)
            X = np.load(npy_path)  # shape: (N, T, 3)

            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            for i, seg in enumerate(X):
                nonzero_mask = ~np.all(seg == 0, axis=1)
                if not np.any(nonzero_mask):
                    continue

                seg = seg[nonzero_mask]
                dx = seg[:, 0]
                dy = seg[:, 1]
                xs = np.cumsum(dx)
                ys = np.cumsum(dy)

                ax.clear()
                ax.plot(xs, ys, color='red', linewidth=2, marker='o', markersize=5)

                ax.set_axis_off()
                ax.set_aspect('equal')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                img_path = os.path.join(save_dir, f"{user_id}_seg{i}.png")
                if not os.path.exists(img_path):  # 防止重复写
                    plt.savefig(img_path, dpi=100)

            plt.close(fig)
            print(f"[{split}] Finished user {user_id}, total {len(X)} segments", flush=True)

    print("Done. All trajectory images saved under:", out_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default=None,
                        help="Specify user ID (e.g. 'user12'), otherwise process all users")
    args = parser.parse_args()

    draw_balabit_baseline(target_user=args.user)
