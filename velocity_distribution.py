# -*- coding: utf-8 -*-
"""
Unified Velocity Distribution Analysis
--------------------------------------
Synchronized with test_draw_trajectories.py logic:
- Same data cleaning (65535 filtering)
- Same velocity calculation (dt epsilon handling)
- Same Global CDF clipping approach
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Project Root and Data Path
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
DATA_ROOT = os.path.join(ROOT, "Data", "Balabit-dataset", "training_files")

# ============================================================
# Data Cleaning (Unified with test_draw_trajectories.py)
# ============================================================
def clean_and_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns and filters out 65535 coordinate errors."""
    df = df.rename(
        columns={
            "client timestamp": "time",
            "x": "x",
            "y": "y",
            "state": "state",
        }
    )
    # Filter 'Move' states
    df = df[df["state"] == "Move"].copy()
    
    # Standardize numeric types and filter Balabit-specific sensor errors (65535)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    
    mask_valid = (df["x"] < 65535) & (df["y"] < 65535)
    df = df[mask_valid].dropna(subset=["x", "y", "time"]).reset_index(drop=True)
    
    return df

# ============================================================
# Velocity Extraction Logic
# ============================================================
def get_velocities_from_file(file_path):
    """Calculates velocities using the same diff logic as the drawing script."""
    try:
        raw_df = pd.read_csv(file_path)
        df = clean_and_rename_cols(raw_df)
        
        if len(df) < 2:
            return []

        xs, ys, ts = df["x"].values, df["y"].values, df["time"].values
        dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(ts)
        
        # Handle dt <= 0 using the same epsilon as drawing script
        dt[dt <= 0] = 1e-5
        
        velocity = np.sqrt(dx**2 + dy**2) / dt
        # Keep only finite values
        velocity = velocity[np.isfinite(velocity)]
        
        return velocity.tolist()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# ============================================================
# Main Analysis and Visualization
# ============================================================
def main():
    all_v = []
    clip_pct = 95.0  # Set to match your --clip argument in drawing script
    
    print(f"Project Root: {ROOT}")
    print(f"Data Path: {DATA_ROOT}")
    print("Processing sessions to synchronize statistics...")

    # Traverse dataset
    for root, dirs, files in os.walk(DATA_ROOT):
        for name in files:
            if name.startswith('session_'):
                file_path = os.path.join(root, name)
                all_v.extend(get_velocities_from_file(file_path))

    if not all_v:
        print("No valid data found. Please check your data paths.")
        return

    all_v = np.array(all_v)
    
    # Calculate Clipping Threshold (The Max_Clipped value)
    v_upper_bound = np.percentile(all_v, clip_pct)
    
    print("\n" + "="*30)
    print(f"STATISTICS (Clip at {clip_pct}%)")
    print("-" * 30)
    print(f"Total Records : {len(all_v)}")
    print(f"Min Velocity  : {all_v.min():.2f}")
    print(f"Mean Velocity : {all_v.mean():.2f}")
    print(f"Max (Raw)     : {all_v.max():.2f}")
    print(f"Max (Clipped) : {v_upper_bound:.2f}  <-- THIS should match drawing script output")
    print("="*30 + "\n")

    # Plotting
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Filter data for visualization to focus on the clipped range
    visible_v = all_v[all_v <= v_upper_bound]
    
    sns.histplot(visible_v, bins=100, kde=True, color='royalblue', alpha=0.6)
    
    plt.axvline(v_upper_bound, color='red', linestyle='--', label=f'{clip_pct}th Percentile')
    
    plt.title(f'Unified Velocity Distribution (Clipped at {clip_pct}%)', fontsize=14)
    plt.xlabel('Velocity (pixels / second)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Save statistics figure for reference
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()