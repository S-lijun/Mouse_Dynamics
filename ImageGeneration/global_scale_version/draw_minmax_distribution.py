import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 配置路径 - 请确保 DATA_ROOT 指向你完整的数据集
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

def get_all_session_bounds(data_dir):
    all_bounds = []
    
    # 获取所有用户文件夹
    users = [u for u in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, u))]
    print(f"Found {len(users)} users. Starting full scan...")

    for user in sorted(users):
        user_dir = os.path.join(data_dir, user)
        files = [f for f in os.listdir(user_dir) if f.startswith("session_")]
        
        for file in files:
            path = os.path.join(user_dir, file)
            try:
                # 只读取 x, y 列以加快速度
                df = pd.read_csv(path, usecols=['x', 'y'])
                # 转换为数值，报错的变 NaN 然后删掉
                df['x'] = pd.to_numeric(df['x'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                df = df.dropna()
                
                if not df.empty:
                    all_bounds.append({
                        'min_x': df['x'].min(),
                        'max_x': df['x'].max(),
                        'min_y': df['y'].min(),
                        'max_y': df['y'].max()
                    })
            except Exception as e:
                print(f"Error reading {user}/{file}: {e}")
        
        print(f"Finished User: {user}")
                
    return pd.DataFrame(all_bounds)

def plot_all_distributions(df):
    # 创建 2x2 的画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Global Distribution of Session Boundaries (All Users & Sessions)', fontsize=20)
    
    metrics = [
        ('min_x', 'blue'), 
        ('max_x', 'red'), 
        ('min_y', 'green'), 
        ('max_y', 'orange')
    ]
    
    for i, (col, color) in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # 使用 log=True 很有必要，因为 65535 和 1000 差距太大，不开启 log 看不到小分布
        ax.hist(df[col], bins=100, color=color, edgecolor='black', alpha=0.7, log=True)
        
        ax.set_title(f'Distribution of {col}', fontsize=14)
        ax.set_xlabel('Coordinate Value')
        ax.set_ylabel('Frequency (Log Scale)')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 计算该指标的全局统计
        stats_text = (f"Min: {df[col].min()}\n"
                      f"Max: {df[col].max()}\n"
                      f"Mean: {df[col].mean():.1f}\n"
                      f"Median: {df[col].median():.1f}")
        
        # 将统计数据写在图上
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = 'all_sessions_distribution.png'
    plt.savefig(save_path, dpi=200)
    print(f"\n[Success] Full distribution plot saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Path {DATA_ROOT} does not exist.")
    else:
        df_all = get_all_session_bounds(DATA_ROOT)
        if not df_all.empty:
            plot_all_distributions(df_all)
            
            # 辅助排查：打印出那些跨度极其离谱的数据量
            outliers_count = len(df_all[df_all['max_x'] > 5000])
            print(f"\nTotal sessions: {len(df_all)}")
            print(f"Sessions with Max_X > 5000 (potential noise): {outliers_count}")
        else:
            print("No valid session data found.")