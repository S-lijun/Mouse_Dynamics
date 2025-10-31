import os
import numpy as np
import matplotlib.pyplot as plt

def draw_balabit_baseline(data_root='Data/Balabit_PeakClick', out_root='Images/Balabit_PeakClick'):
    for split in ['training', 'testing']:
        data_path = os.path.join(data_root, split)
        out_path = os.path.join(out_root, split)
        os.makedirs(out_path, exist_ok=True)

        for filename in sorted(os.listdir(data_path)):
            if not filename.endswith('.npy'):
                continue

            user_id = filename.replace('.npy', '')
            npy_path = os.path.join(data_path, filename)
            save_dir = os.path.join(out_path, user_id)
            os.makedirs(save_dir, exist_ok=True)

            print(f"[{split}] Drawing user {user_id} ...")
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
                #xs = seg[:, 0]
                #ys = seg[:, 1]


                ax.clear()
                ax.plot(xs, ys, color='red', linewidth=2, marker='o', markersize=5)

                ax.set_axis_off()
                ax.set_aspect('equal')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                img_path = os.path.join(save_dir, f"{user_id}_seg{i}.png")
                plt.savefig(img_path, dpi=100)

            plt.close(fig)
    print("All trajectory images saved under:", out_root)

if __name__ == "__main__":
    draw_balabit_baseline()
