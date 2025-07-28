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

# ----------------------------------------------------------------------------- #
# Plotting Functions                                                            #
# ----------------------------------------------------------------------------- #
def plot_xy_trajectories(df: pd.DataFrame, max_trajs: int = 20, save_path: Optional[Path] = None):
    """Plot 2D trajectory paths (x-y plane)."""
    df_sample = sample_trajectories(df, max_trajs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot individual trajectories
    for (batch, traj), group in df_sample.groupby(['batch', 'traj']):
        alpha = min(0.7, 20 / max_trajs)  # Adjust transparency based on number of trajectories
        
        ax1.plot(group['true_delta_x'], group['true_delta_y'], 
                'k-', alpha=alpha, linewidth=1, label='True' if (batch, traj) == (0, 0) else None)
        ax1.plot(group['pred_delta_x'], group['pred_delta_y'], 
                'r--', alpha=alpha, linewidth=1, label='Predicted' if (batch, traj) == (0, 0) else None)
    
    ax1.set_xlabel('Δx [m]')
    ax1.set_ylabel('Δy [m]')
    ax1.set_title(f'Vehicle Trajectories (Sample of {len(df_sample.groupby(["batch", "traj"]))} trajectories)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot error magnitude over trajectories
    df_sample['position_error'] = np.sqrt(df_sample['err_delta_x']**2 + df_sample['err_delta_y']**2)
    
    for (batch, traj), group in df_sample.groupby(['batch', 'traj']):
        ax2.plot(group['step'], group['position_error'], alpha=0.6, linewidth=1)
    
    # Add mean error trend
    mean_error = df_sample.groupby('step')['position_error'].mean()
    ax2.plot(mean_error.index, mean_error.values, 'r-', linewidth=3, 
             label=f'Mean Error (Final: {mean_error.iloc[-1]:.3f}m)')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error [m]')
    ax2.set_title('Position Error Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved XY trajectories plot to {save_path}")
    
    plt.show()

def plot_state_time_series(df: pd.DataFrame, max_trajs: int = 10, save_path: Optional[Path] = None):
    """Plot all state variables over time."""
    df_sample = sample_trajectories(df, max_trajs)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for i, (state, unit, label) in enumerate(zip(STATE_NAMES, STATE_UNITS, STATE_LABELS)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot individual trajectories
        for (batch, traj), group in df_sample.groupby(['batch', 'traj']):
            alpha = 0.5
            ax.plot(group['time'], group[f'true_{state}'], 'k-', alpha=alpha, linewidth=1,
                   label='True' if (batch, traj) == (0, 0) else None)
            ax.plot(group['time'], group[f'pred_{state}'], 'r--', alpha=alpha, linewidth=1,
                   label='Predicted' if (batch, traj) == (0, 0) else None)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(label)
        ax.set_title(f'{state.replace("_", " ").title()} Prediction')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Add legend to first subplot
            ax.legend()
    
    # Remove empty subplot if odd number of states
    if len(STATE_NAMES) < len(axes):
        axes[-1].remove()
    
    plt.suptitle(f'State Variables Over Time (Sample of {len(df_sample.groupby(["batch", "traj"]))} trajectories)', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved state time series plot to {save_path}")
    
    plt.show()

def plot_error_analysis(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Comprehensive error analysis plots."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    # 1. Error distribution for each state
    ax1 = fig.add_subplot(gs[0, :2])
    error_data = [df[f'err_{state}'].values for state in STATE_NAMES]
    bp = ax1.boxplot(error_data, labels=[s.replace('_', '\n') for s in STATE_NAMES], patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(STATE_NAMES)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Error Distribution by State Variable')
    ax1.set_ylabel('Error Magnitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Error evolution over time steps
    ax2 = fig.add_subplot(gs[0, 2:])
    for state in STATE_NAMES:
        mean_abs_error = df.groupby('step')[f'err_{state}'].apply(lambda x: np.mean(np.abs(x)))
        ax2.plot(mean_abs_error.index, mean_abs_error.values, label=state, marker='o', markersize=3)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Error Evolution Over Time')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Position error 2D histogram
    ax3 = fig.add_subplot(gs[1, :2])
    h = ax3.hist2d(df['err_delta_x'], df['err_delta_y'], bins=50, cmap='Blues')
    ax3.set_xlabel('X Position Error [m]')
    ax3.set_ylabel('Y Position Error [m]')
    ax3.set_title('2D Position Error Distribution')
    plt.colorbar(h[3], ax=ax3, label='Count')
    ax3.set_aspect('equal', adjustable='box')
    
    # 4. Error vs time step scatter
    ax4 = fig.add_subplot(gs[1, 2:])
    position_error = np.sqrt(df['err_delta_x']**2 + df['err_delta_y']**2)
    scatter = ax4.scatter(df['step'], position_error, c=df['batch'], alpha=0.5, s=1, cmap='viridis')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Position Error [m]')
    ax4.set_title('Position Error vs Time Step (colored by batch)')
    plt.colorbar(scatter, ax=ax4, label='Batch ID')
    ax4.grid(True, alpha=0.3)
    
    # 5. Final vs initial error comparison
    ax5 = fig.add_subplot(gs[2, :2])
    initial_errors = df[df['step'] == 0].groupby(['batch', 'traj']).apply(
        lambda x: np.sqrt(x['err_delta_x'].iloc[0]**2 + x['err_delta_y'].iloc[0]**2)).values
    final_errors = df.groupby(['batch', 'traj']).apply(
        lambda x: np.sqrt(x['err_delta_x'].iloc[-1]**2 + x['err_delta_y'].iloc[-1]**2)).values
    
    ax5.scatter(initial_errors, final_errors, alpha=0.6, s=20)
    ax5.plot([0, max(initial_errors.max(), final_errors.max())], 
             [0, max(initial_errors.max(), final_errors.max())], 'r--', alpha=0.5)
    ax5.set_xlabel('Initial Position Error [m]')
    ax5.set_ylabel('Final Position Error [m]')
    ax5.set_title('Final vs Initial Position Error')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')
    
    # 6. Error statistics table
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    
    # Calculate statistics
    stats_data = []
    for state in STATE_NAMES:
        errors = df[f'err_{state}'].values
        stats_data.append([
            state.replace('_', ' ').title(),
            f"{np.mean(np.abs(errors)):.4f}",
            f"{np.std(errors):.4f}",
            f"{np.sqrt(np.mean(errors**2)):.4f}",
            f"{np.percentile(np.abs(errors), 95):.4f}"
        ])
    
    table = ax6.table(cellText=stats_data,
                     colLabels=['State', 'MAE', 'Std', 'RMSE', '95th %ile'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax6.set_title('Error Statistics Summary', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error analysis plot to {save_path}")
    
    plt.show()

def plot_velocity_analysis(df: pd.DataFrame, max_trajs: int = 15, save_path: Optional[Path] = None):
    """Specialized analysis for velocity predictions."""
    df_sample = sample_trajectories(df, max_trajs)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Velocity trajectories
    ax1 = axes[0, 0]
    for (batch, traj), group in df_sample.groupby(['batch', 'traj']):
        ax1.plot(group['time'], group['true_velocity'], 'k-', alpha=0.5, linewidth=1,
                label='True' if (batch, traj) == (0, 0) else None)
        ax1.plot(group['time'], group['pred_velocity'], 'r--', alpha=0.5, linewidth=1,
                label='Predicted' if (batch, traj) == (0, 0) else None)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.set_title('Velocity Predictions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity prediction scatter
    ax2 = axes[0, 1]
    ax2.scatter(df['true_velocity'], df['pred_velocity'], alpha=0.3, s=1)
    
    v_min, v_max = df[['true_velocity', 'pred_velocity']].min().min(), df[['true_velocity', 'pred_velocity']].max().max()
    ax2.plot([v_min, v_max], [v_min, v_max], 'r--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('True Velocity [m/s]')
    ax2.set_ylabel('Predicted Velocity [m/s]')
    ax2.set_title('Velocity Prediction Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 3. Velocity error over time
    ax3 = axes[1, 0]
    velocity_error_mean = df.groupby('step')['err_velocity'].apply(lambda x: np.mean(np.abs(x)))
    velocity_error_std = df.groupby('step')['err_velocity'].apply(lambda x: np.std(x))
    
    ax3.plot(velocity_error_mean.index, velocity_error_mean.values, 'b-', linewidth=2, label='Mean |Error|')
    ax3.fill_between(velocity_error_mean.index, 
                     velocity_error_mean.values - velocity_error_std.values,
                     velocity_error_mean.values + velocity_error_std.values,
                     alpha=0.3, label='±1 std')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Velocity Error [m/s]')
    ax3.set_title('Velocity Error Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Velocity error distribution
    ax4 = axes[1, 1]
    ax4.hist(df['err_velocity'], bins=50, alpha=0.7, density=True, edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(df['err_velocity'].mean(), color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {df["err_velocity"].mean():.4f}')
    ax4.set_xlabel('Velocity Error [m/s]')
    ax4.set_ylabel('Density')
    ax4.set_title('Velocity Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved velocity analysis plot to {save_path}")
    
    plt.show()

def plot_yaw_analysis(df: pd.DataFrame, max_trajs: int = 15, save_path: Optional[Path] = None):
    """Specialized analysis for yaw angle predictions."""
    df_sample = sample_trajectories(df, max_trajs)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Yaw trajectories
    ax1 = axes[0, 0]
    for (batch, traj), group in df_sample.groupby(['batch', 'traj']):
        ax1.plot(group['time'], group['true_yaw'], 'k-', alpha=0.5, linewidth=1,
                label='True' if (batch, traj) == (0, 0) else None)
        ax1.plot(group['time'], group['pred_yaw'], 'r--', alpha=0.5, linewidth=1,
                label='Predicted' if (batch, traj) == (0, 0) else None)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Yaw [rad]')
    ax1.set_title('Yaw Angle Predictions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Yaw error (wrapped) histogram
    ax2 = axes[0, 1]
    yaw_errors = df['err_yaw'].values
    ax2.hist(yaw_errors, bins=50, alpha=0.7, density=True, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(yaw_errors.mean(), color='green', linestyle='-', linewidth=2,
               label=f'Mean: {yaw_errors.mean():.4f} rad')
    ax2.set_xlabel('Yaw Error [rad]')
    ax2.set_ylabel('Density')
    ax2.set_title('Yaw Error Distribution (Wrapped)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Yaw error over time
    ax3 = axes[1, 0]
    yaw_error_mean = df.groupby('step')['err_yaw'].apply(lambda x: np.mean(np.abs(x)))
    yaw_error_std = df.groupby('step')['err_yaw'].apply(lambda x: np.std(x))
    
    ax3.plot(yaw_error_mean.index, yaw_error_mean.values, 'b-', linewidth=2, label='Mean |Error|')
    ax3.fill_between(yaw_error_mean.index,
                     yaw_error_mean.values - yaw_error_std.values,
                     yaw_error_mean.values + yaw_error_std.values,
                     alpha=0.3, label='±1 std')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Yaw Error [rad]')
    ax3.set_title('Yaw Error Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Circular error plot
    ax4 = axes[1, 1]
    # Convert to polar coordinates for circular plot
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Plot unit circle
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='±π boundary')
    
    # Plot errors as points on the circle
    error_angles = yaw_errors
    error_radii = np.ones_like(error_angles)  # All errors at unit distance
    
    x_errors = error_radii * np.cos(error_angles)
    y_errors = error_radii * np.sin(error_angles)
    
    ax4.scatter(x_errors, y_errors, alpha=0.5, s=20, c=df['step'], cmap='viridis')
    ax4.set_xlabel('cos(yaw_error)')
    ax4.set_ylabel('sin(yaw_error)')
    ax4.set_title('Yaw Errors on Unit Circle')
    ax4.set_aspect('equal', adjustable='box')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved yaw analysis plot to {save_path}")
    
    plt.show()

def create_summary_report(df: pd.DataFrame, overall_stats: Dict, save_path: Optional[Path] = None):
    """Create a comprehensive summary report."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Hybrid Neural ODE - Test Results Summary Report', fontsize=20, fontweight='bold')
    
    # 1. Overall metrics table
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    if overall_stats:
        metrics_data = [
            ['Total MSE', f"{overall_stats.get('mse_total', 'N/A'):.6f}"],
            ['Position RMSE', f"{overall_stats.get('position_rmse', 'N/A'):.4f} m"],
            ['Position MAE', f"{overall_stats.get('position_mae', 'N/A'):.4f} m"],
            ['Final Position RMSE', f"{overall_stats.get('final_position_rmse', 'N/A'):.4f} m"],
            ['Velocity RMSE', f"{overall_stats.get('velocity_rmse', 'N/A'):.4f} m/s"],
            ['Velocity MAE', f"{overall_stats.get('velocity_mae', 'N/A'):.4f} m/s"],
        ]
    else:
        # Calculate from data
        position_error = np.sqrt(df['err_delta_x']**2 + df['err_delta_y']**2)
        final_position_error = df.groupby(['batch', 'traj']).apply(
            lambda x: np.sqrt(x['err_delta_x'].iloc[-1]**2 + x['err_delta_y'].iloc[-1]**2))
        
        metrics_data = [
            ['Position RMSE', f"{np.sqrt(np.mean(position_error**2)):.4f} m"],
            ['Position MAE', f"{np.mean(position_error):.4f} m"],
            ['Final Position RMSE', f"{np.sqrt(np.mean(final_position_error**2)):.4f} m"],
            ['Velocity RMSE', f"{np.sqrt(np.mean(df['err_velocity']**2)):.4f} m/s"],
            ['Velocity MAE', f"{np.mean(np.abs(df['err_velocity'])):.4f} m/s"],
            ['Yaw RMSE', f"{np.sqrt(np.mean(df['err_yaw']**2)):.4f} rad"],
        ]
    
    table = ax1.table(cellText=metrics_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.2, 0.3, 0.6, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax1.set_title('Overall Performance Metrics', fontsize=16, pad=20)
    
    # 2. Error evolution plot
    ax2 = fig.add_subplot(gs[1, :2])
    position_error = np.sqrt(df['err_delta_x']**2 + df['err_delta_y']**2)
    df_with_pos_error = df.copy()
    df_with_pos_error['position_error'] = position_error
    
    error_evolution = df_with_pos_error.groupby('step')['position_error'].agg(['mean', 'std'])
    ax2.plot(error_evolution.index, error_evolution['mean'], 'b-', linewidth=2, label='Mean Position Error')
    ax2.fill_between(error_evolution.index,
                     error_evolution['mean'] - error_evolution['std'],
                     error_evolution['mean'] + error_evolution['std'],
                     alpha=0.3, label='±1 std')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error [m]')
    ax2.set_title('Position Error Evolution Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. State-wise error comparison
    ax3 = fig.add_subplot(gs[1, 2])
    state_rmses = [np.sqrt(np.mean(df[f'err_{state}']**2)) for state in STATE_NAMES]
    colors = plt.cm.Set3(np.linspace(0, 1, len(STATE_NAMES)))
    
    bars = ax3.bar(range(len(STATE_NAMES)), state_rmses, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(STATE_NAMES)))
    ax3.set_xticklabels([s.replace('_', '\n') for s in STATE_NAMES], rotation=45, ha='right')
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE by State Variable')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, state_rmses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(state_rmses),
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Data summary statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate summary statistics
    n_batches = df['batch'].nunique()
    n_trajectories = len(df.groupby(['batch', 'traj']))
    n_total_points = len(df)
    time_span = df['time'].max() - df['time'].min()
    
    summary_text = f"""
    Dataset Summary:
    • Number of batches: {n_batches}
    • Number of trajectories: {n_trajectories}
    • Total data points: {n_total_points:,}
    • Time span: {time_span:.2f} seconds
    • Average trajectory length: {n_total_points/n_trajectories:.1f} points
    
    Model Performance Highlights:
    • Best performing state: {STATE_NAMES[np.argmin(state_rmses)]} (RMSE: {min(state_rmses):.4f})
    • Most challenging state: {STATE_NAMES[np.argmax(state_rmses)]} (RMSE: {max(state_rmses):.4f})
    • Position tracking accuracy: {np.sqrt(np.mean(position_error**2)):.3f} m RMSE
    • Final position drift: {np.sqrt(np.mean(final_position_error**2)):.3f} m RMSE
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary report to {save_path}")
    
    plt.show()

# ----------------------------------------------------------------------------- #
# Main Function                                                                 #
# ----------------------------------------------------------------------------- #
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
        
        print("1. Creating XY trajectory plots...")  
        plot_xy_trajectories(df, max_trajs=25, 
                           save_path=OUTPUT_DIR / "01_xy_trajectories.png")
        
        print("2. Creating state time series plots...")
        plot_state_time_series(df, max_trajs=15,
                             save_path=OUTPUT_DIR / "02_state_time_series.png")
        
        print("3. Creating comprehensive error analysis...")
        plot_error_analysis(df, save_path=OUTPUT_DIR / "03_error_analysis.png")
        
        print("4. Creating velocity analysis...")
        plot_velocity_analysis(df, max_trajs=20,
                             save_path=OUTPUT_DIR / "04_velocity_analysis.png")
        
        print("5. Creating yaw analysis...")
        plot_yaw_analysis(df, max_trajs=20,
                        save_path=OUTPUT_DIR / "05_yaw_analysis.png")
        
        print("6. Creating summary report...")
        create_summary_report(df, overall_stats,
                            save_path=OUTPUT_DIR / "06_summary_report.png")
        
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
