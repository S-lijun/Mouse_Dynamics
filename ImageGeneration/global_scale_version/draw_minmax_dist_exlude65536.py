import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 配置路径
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

def get_clean_session_bounds(data_dir):
    all_bounds = []
    users = [u for u in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, u))]
    
    print(f"Scanning {len(users)} users... Skipping coordinate >= 65535")

    for user in sorted(users):
        user_dir = os.path.join(data_dir, user)
        files = [f for f in os.listdir(user_dir) if f.startswith("session_")]
        
        for file in files:
            path = os.path.join(user_dir, file)
            try:
                # 只读 x, y 加快速度
                df = pd.read_csv(path, usecols=['x', 'y'])
                df['x'] = pd.to_numeric(df['x'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                
                # --- 核心逻辑：跳过异常值 ---
                # 只要 x 或 y 有一个是 65535 就扔掉该行
                df_clean = df[(df['x'] < 65535) & (df['y'] < 65535)].dropna()
                
                if not df_clean.empty:
                    all_bounds.append({
                        'max_x': df_clean['x'].max(),
                        'max_y': df_clean['y'].max(),
                        'user': user,
                        'session': file
                    })
            except Exception as e:
                print(f"Error: {user}/{file}: {e}")
                
    return pd.DataFrame(all_bounds)

def plot_refined_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Session Max Boundaries Distribution (Excluding 65535)', fontsize=16)
    
    # 绘制 Max X
    axes[0].hist(df['max_x'], bins=50, color='red', edgecolor='black', alpha=0.7)
    axes[0].set_title('Cleaned Max X Distribution')
    axes[0].set_xlabel('Coordinate Value')
    
    # 绘制 Max Y
    axes[1].hist(df['max_y'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    axes[1].set_title('Cleaned Max Y Distribution')
    axes[1].set_xlabel('Coordinate Value')

    for ax in axes:
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('cleaned_bounds_distribution.png')
    print("\n[Success] Distribution plot saved as 'cleaned_bounds_distribution.png'")
    plt.show()

if __name__ == "__main__":
    df_results = get_clean_session_bounds(DATA_ROOT)
    if not df_results.empty:
        plot_refined_distribution(df_results)
        print("\nCleaned Stats:")
        print(df_results[['max_x', 'max_y']].describe())