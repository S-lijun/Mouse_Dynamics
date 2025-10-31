import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.stats import rankdata

'''
Uses CDF transformation to encode the velocity and direction of the trajectories.
Takes in binary data from Data/binary folder and saves the trajectories as images in Images/binary_v_cdf
'''

def draw_trajectories_with_velocity_direction_cdf(data_path=None, out_path=None):
    # Get the Sasha root directory -- parent of ImageGeneration folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    if data_path is None:
        data_path = os.path.join(root_dir, 'Data', 'Balabit/training')
    if out_path is None:
        out_path = os.path.join(root_dir, 'Images', 'Balabit_v_cdf/training')
    
    print(f"Data path: {data_path}")
    print(f"Output path: {out_path}")
    
    cmap = plt.get_cmap('plasma')
    os.makedirs(out_path, exist_ok=True)

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith('.npy'):
            continue

        user_id = filename.replace('.npy', '')
        file_path = os.path.join(data_path, filename)
        save_dir = os.path.join(out_path, f"{user_id}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Processing {user_id}...")

        data = np.load(file_path)

        for i, seq in enumerate(data):
            seq = seq[~np.all(seq == 0, axis=1)]
            if seq.shape[0] < 2:
                continue

            xs = seq[:, 0]
            ys = seq[:, 1]
            ts = seq[:, 2]

            dx = np.diff(xs)
            dy = np.diff(ys)
            dt = np.diff(ts)
            dt[dt == 0] = 1e-5  # Avoid divide-by-zero

            velocity = np.sqrt(dx**2 + dy**2) / dt
            velocity += 1e-5

            if len(velocity) <= 1:
                continue

            try:
                ranks = rankdata(velocity, method='average')
                trans_velocity = (ranks - 1) / (len(velocity) - 1)
            except Exception as e:
                print(f"Skipping trajectory {i} in user{user_id} due to CDF error: {e}")
                continue

            theta = np.arctan2(dy, dx)
            direction_score = 1.0 - np.abs(np.cos(theta))
            min_width, max_width = 2.0, 4.0
            linewidths = min_width + direction_score * (max_width - min_width)

            segment_colors = cmap(trans_velocity[:-1])  # 旧版风格
            #point_colors = cmap(trans_velocity)         # 旧版风格
            point_colors = cmap(np.concatenate([trans_velocity, [trans_velocity[-1]]]))


            points = np.stack([xs, ys], axis=1)
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=segment_colors, linewidths=linewidths)

            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.add_collection(lc)
            ax.scatter(xs, ys, c=point_colors, marker='s', s=15, zorder=7)
            ax.plot(xs[0], ys[0], marker='o', color='red', markersize=7, zorder=8)
            ax.plot(xs[-1], ys[-1], marker='x', color='red', markersize=7, zorder=8)

            ax.autoscale()
            ax.margins(x=0.1, y=0.1)
            ax.set_axis_off()
            ax.set_aspect('equal')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            out_path_img = os.path.join(save_dir, f"{user_id}_img{i}.png")
            plt.savefig(out_path_img, dpi=100, pad_inches=0.3)
            plt.close(fig)

    print("All trajectory images saved with CDF velocity transformation + direction encoding.")


if __name__ == "__main__":
    draw_trajectories_with_velocity_direction_cdf()
