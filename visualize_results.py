import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ----------------------------------------------------------------------------- #
# Configuration and Constants                                                   #
# ----------------------------------------------------------------------------- #
STATE_NAMES = ["delta_x", "delta_y", "yaw", "steering", 
               "velocity", "side_slip", "yaw_rate"]
STATE_UNITS = ["m", "m", "rad", "rad", "m/s", "rad", "rad/s"]
STATE_LABELS = [f"{name} [{unit}]" for name, unit in zip(STATE_NAMES, STATE_UNITS)]

RESULTS_DIR = Path("test_results")
OUTPUT_DIR = Path("visualizations")

# ----------------------------------------------------------------------------- #
# Data Loading Functions                                                        #
# ----------------------------------------------------------------------------- #
def load_all_batch_data(results_dir: Path = RESULTS_DIR) -> Tuple[pd.DataFrame, Dict]:
    """Load all batch CSV files and combine into a single DataFrame."""
    csv_files = sorted(results_dir.glob("batch_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No batch CSV files found in {results_dir}")
    
    print(f"Found {len(csv_files)} batch files to process...")
    
    # Load and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Load overall statistics if available
    stats_file = results_dir / "overall_statistics.json"
    overall_stats = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            overall_stats = json.load(f)
    
    print(f"Loaded {len(combined_df)} trajectory points from {len(csv_files)} batches")
    return combined_df, overall_stats

def sample_trajectories(df: pd.DataFrame, max_trajs: int = 20) -> pd.DataFrame:
    """Sample a subset of trajectories for cleaner visualization."""
    unique_trajs = df[['batch', 'traj']].drop_duplicates()
    if len(unique_trajs) <= max_trajs:
        return df
    
    sampled_trajs = unique_trajs.sample(n=max_trajs, random_state=42)
    return df.merge(sampled_trajs, on=['batch', 'traj'])


def plot_xy_trajectories_separate(df: pd.DataFrame, max_trajs: int = 9, save_path: Optional[Path] = None):
    """Plot XY trajectories for each trajectory in separate subplots (no error inset)."""
    df_sample = sample_trajectories(df, max_trajs)
    trajs = list(df_sample.groupby(['batch', 'traj']).groups.keys())
    n_trajs = len(trajs)
    ncols = 3
    nrows = int(np.ceil(n_trajs / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    axes = axes.flatten()
    
    for idx, (batch, traj) in enumerate(trajs):
        group = df_sample[(df_sample['batch'] == batch) & (df_sample['traj'] == traj)]
        ax = axes[idx]
        ax.plot(group['true_delta_x'], group['true_delta_y'], 'k-', label='True')
        ax.plot(group['pred_delta_x'], group['pred_delta_y'], 'r--', label='Predicted')
        ax.set_xlabel('Δx [m]')
        ax.set_ylabel('Δy [m]')
        ax.set_title(f'Traj: batch={batch}, traj={traj}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for ax in axes[n_trajs:]:
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved XY trajectories (separate) plot to {save_path}")
    plt.show()

def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("Hybrid Neural ODE - Comprehensive Visualization Generator")  
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        # Load all data
        print("Loading test results...")
        df, overall_stats = load_all_batch_data()
        
        print(f"Dataset overview:")
        print(f"  - {df['batch'].nunique()} batches")
        print(f"  - {len(df.groupby(['batch', 'traj']))} trajectories") 
        print(f"  - {len(df):,} total data points")
        print()
        
        # Generate all visualizations
        print("Generating visualizations...")
        
   
        print("2. Creating separate XY trajectory and error plots...")
        plot_xy_trajectories_separate(df, max_trajs=9, 
                                    save_path=OUTPUT_DIR / "02_xy_trajectories_separate.png")
        
        print()
        print("=" * 60)
        print("All visualizations completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR.resolve()}")
        print("=" * 60)
        
        # Print quick statistics
        position_error = np.sqrt(df['err_delta_x']**2 + df['err_delta_y']**2)
        print("\nQuick Performance Summary:")
        print(f"  • Mean position error: {np.mean(position_error):.4f} m")
        print(f"  • Position RMSE: {np.sqrt(np.mean(position_error**2)):.4f} m")
        print(f"  • Velocity RMSE: {np.sqrt(np.mean(df['err_velocity']**2)):.4f} m/s")
        print(f"  • Yaw RMSE: {np.sqrt(np.mean(df['err_yaw']**2)):.4f} rad")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please ensure test_model.py has been run and CSV files exist in test_results/")

if __name__ == "__main__":
    main()
