#!/usr/bin/env python3
"""
Visualization Script for Hybrid Neural ODE Results

Creates plots from saved evaluation data in the evaluation_results directory.
Focuses on trajectory visualization and performance metrics analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# Set matplotlib backend and style
matplotlib.use('Agg')  # Use non-interactive backend for better compatibility
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_trajectory_data(csv_path: str) -> pd.DataFrame:
    """
    Load trajectory comparison data from CSV file.
    
    Args:
        csv_path: Path to trajectory CSV file
        
    Returns:
        DataFrame containing trajectory data
    """
    df = pd.read_csv(csv_path)
    # Add robot_idx and sample_idx from filename if missing
    filename = os.path.basename(csv_path)
    parts = filename.split('_')
    if 'robot_idx' not in df.columns:
        df['robot_idx'] = int(parts[1].replace('robot', '')) if len(parts) > 2 else -1
    if 'sample_idx' not in df.columns:
        df['sample_idx'] = int(parts[2].replace('sample', '')) if len(parts) > 2 else -1
    # Add timestep if missing
    if 'timestep' not in df.columns:
        df['timestep'] = np.arange(len(df))
    return df


def load_metrics(json_path: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        json_path: Path to metrics JSON file
        
    Returns:
        Dictionary of metrics
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def load_summary_metrics(csv_path: str) -> pd.DataFrame:
    """
    Load summary metrics from CSV file.
    
    Args:
        csv_path: Path to summary metrics CSV
        
    Returns:
        DataFrame with metrics
    """
    return pd.read_csv(csv_path)


def find_evaluation_files(eval_dir: str = "evaluation_results", 
                         dataset: str = "test",
                         max_files: int = None) -> Tuple[List[str], List[str]]:
    """
    Find trajectory and metrics files for a dataset.
    
    Args:
        eval_dir: Evaluation results directory
        dataset: Dataset name ('train', 'val', or 'test')
        max_files: Maximum number of files to return
        
    Returns:
        Tuple of (trajectory_files, metrics_files)
    """
    eval_path = Path(eval_dir)
    
    # Find all trajectory files for the dataset
    traj_files = list(eval_path.glob(f"{dataset}_*_trajectory.csv"))
    metrics_files = list(eval_path.glob(f"{dataset}_*_metrics.json"))
    
    # Sort by sample index
    traj_files.sort()
    metrics_files.sort()
    
    # Limit number of files if specified
    if max_files is not None:
        traj_files = traj_files[:max_files]
        metrics_files = metrics_files[:max_files]
    
    return traj_files, metrics_files


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_trajectory_comparison(trajectory_data: pd.DataFrame, 
                              title: Optional[str] = None,
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
    """
    Create plot comparing true vs. predicted trajectory.
    
    Args:
        trajectory_data: DataFrame with trajectory data
        title: Optional title override
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    true_x = trajectory_data['true_delta_x']
    true_y = trajectory_data['true_delta_y']
    pred_x = trajectory_data['pred_delta_x']
    pred_y = trajectory_data['pred_delta_y']
    pos_error = trajectory_data['pos_error']
    
    # Extract robot and sample info from column names
    robot_idx = trajectory_data['robot_idx'].iloc[0] if 'robot_idx' in trajectory_data.columns else '?'
    sample_idx = trajectory_data['sample_idx'].iloc[0] if 'sample_idx' in trajectory_data.columns else '?'
    
    # Create default title if not provided
    if title is None:
        title = f"Robot {robot_idx} - Trajectory Comparison (Sample {sample_idx})"
    
    # Plot trajectories
    ax.plot(true_x, true_y, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(pred_x, pred_y, 'r--', linewidth=2, label='Predicted')
    
    # Mark start and endpoints
    ax.plot(true_x.iloc[0], true_y.iloc[0], 'go', markersize=10, label='Start')
    ax.plot(true_x.iloc[-1], true_y.iloc[-1], 'bs', markersize=10, label='End (True)')
    ax.plot(pred_x.iloc[-1], pred_y.iloc[-1], 'rs', markersize=10, label='End (Pred)')
    
    # Add error connectors at regular intervals
    step = max(1, len(true_x) // 10)
    for i in range(0, len(true_x), step):
        ax.plot([true_x.iloc[i], pred_x.iloc[i]], 
               [true_y.iloc[i], pred_y.iloc[i]], 
               'k-', alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Δx (m)', fontsize=14)
    ax.set_ylabel('Δy (m)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=12)
    
    # Add statistics
    mean_error = pos_error.mean()
    max_error = pos_error.max()
    final_error = pos_error.iloc[-1]
    
    stats_text = f"Mean Error: {mean_error:.3f}m\nMax Error: {max_error:.3f}m\nFinal Error: {final_error:.3f}m"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_state_comparison(trajectory_data: pd.DataFrame,
                         state_name: str,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """
    Plot comparison of true vs. predicted values for a specific state variable.
    
    Args:
        trajectory_data: DataFrame with trajectory data
        state_name: Name of state variable to plot ('delta_x', 'delta_y', etc.)
        title: Optional title override
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract timesteps
    timesteps = trajectory_data['timestep']
    
    # Extract true and predicted values
    true_col = f'true_{state_name}'
    pred_col = f'pred_{state_name}'
    
    if true_col not in trajectory_data or pred_col not in trajectory_data:
        print(f"Warning: State {state_name} not found in trajectory data")
        return
    
    true_values = trajectory_data[true_col]
    pred_values = trajectory_data[pred_col]
    
    # Extract robot and sample info
    robot_idx = trajectory_data['robot_idx'].iloc[0] if 'robot_idx' in trajectory_data.columns else '?'
    sample_idx = trajectory_data['sample_idx'].iloc[0] if 'sample_idx' in trajectory_data.columns else '?'
    
    # Create default title if not provided
    if title is None:
        title = f"Robot {robot_idx} - {state_name} Comparison (Sample {sample_idx})"
    
    # Plot state values
    ax.plot(timesteps, true_values, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(timesteps, pred_values, 'r--', linewidth=2, label='Predicted')
    
    # Calculate error statistics
    error = np.abs(true_values - pred_values)
    rmse = np.sqrt(np.mean(error**2))
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(state_name, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add statistics
    stats_text = f"RMSE: {rmse:.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"State comparison saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_comprehensive_states(trajectory_data: pd.DataFrame,
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
    """
    Create a comprehensive plot showing all state variables.
    
    Args:
        trajectory_data: DataFrame with trajectory data
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    # Extract timesteps
    timesteps = trajectory_data['timestep']
    
    # State variables to plot
    state_names = ['delta_x', 'delta_y', 'yaw', 'steering', 
                  'velocity', 'side_slip', 'yaw_rate']
    
    # Plot trajectory in last subplot
    ax_traj = axes[7]
    ax_traj.plot(trajectory_data['true_delta_x'], trajectory_data['true_delta_y'], 
                'b-', linewidth=2, label='Ground Truth')
    ax_traj.plot(trajectory_data['pred_delta_x'], trajectory_data['pred_delta_y'], 
                'r--', linewidth=2, label='Predicted')
    ax_traj.plot(trajectory_data['true_delta_x'].iloc[0], trajectory_data['true_delta_y'].iloc[0], 
                'go', markersize=8, label='Start')
    ax_traj.plot(trajectory_data['true_delta_x'].iloc[-1], trajectory_data['true_delta_y'].iloc[-1], 
                'bs', markersize=8, label='End (True)')
    ax_traj.plot(trajectory_data['pred_delta_x'].iloc[-1], trajectory_data['pred_delta_y'].iloc[-1], 
                'rs', markersize=8, label='End (Pred)')
    ax_traj.set_xlabel('Δx (m)')
    ax_traj.set_ylabel('Δy (m)')
    ax_traj.set_title('X-Y Trajectory')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')
    ax_traj.legend(loc='upper left')
    
    # Plot individual state variables
    for i, state_name in enumerate(state_names):
        ax = axes[i]
        true_col = f'true_{state_name}'
        pred_col = f'pred_{state_name}'
        
        if true_col in trajectory_data and pred_col in trajectory_data:
            ax.plot(timesteps, trajectory_data[true_col], 'b-', linewidth=2, label='Ground Truth')
            ax.plot(timesteps, trajectory_data[pred_col], 'r--', linewidth=2, label='Predicted')
            
            # Calculate error statistics
            error = np.abs(trajectory_data[true_col] - trajectory_data[pred_col])
            rmse = np.sqrt(np.mean(error**2))
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(state_name)
            ax.set_title(f'{state_name} (RMSE: {rmse:.4f})')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
    
    # Extract robot and sample info
    robot_idx = trajectory_data['robot_idx'].iloc[0] if 'robot_idx' in trajectory_data.columns else '?'
    sample_idx = trajectory_data['sample_idx'].iloc[0] if 'sample_idx' in trajectory_data.columns else '?'
    
    fig.suptitle(f'Robot {robot_idx} - Comprehensive State Comparison (Sample {sample_idx})', 
                fontsize=18, y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive state plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_position_error(trajectory_data: pd.DataFrame,
                       save_path: Optional[str] = None,
                       show_plot: bool = True) -> None:
    """
    Plot position error over time.
    
    Args:
        trajectory_data: DataFrame with trajectory data
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract timesteps and position error
    timesteps = trajectory_data['timestep']
    pos_error = trajectory_data['pos_error']
    
    # Extract robot and sample info
    robot_idx = trajectory_data['robot_idx'].iloc[0] if 'robot_idx' in trajectory_data.columns else '?'
    sample_idx = trajectory_data['sample_idx'].iloc[0] if 'sample_idx' in trajectory_data.columns else '?'
    
    # Plot error
    ax.plot(timesteps, pos_error, 'r-', linewidth=2)
    ax.fill_between(timesteps, 0, pos_error, alpha=0.2, color='red')
    
    # Calculate statistics
    mean_error = pos_error.mean()
    max_error = pos_error.max()
    final_error = pos_error.iloc[-1]
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Position Error (m)', fontsize=14)
    ax.set_title(f'Robot {robot_idx} - Position Error Over Time (Sample {sample_idx})', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"Mean Error: {mean_error:.3f}m\nMax Error: {max_error:.3f}m\nFinal Error: {final_error:.3f}m"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position error plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# SUMMARY AND AGGREGATE PLOTTING FUNCTIONS
# ============================================================================

def plot_metrics_summary(metrics_files: List[str],
                        save_path: Optional[str] = None,
                        show_plot: bool = True) -> None:
    """
    Create summary plot of metrics across multiple samples.
    
    Args:
        metrics_files: List of metrics JSON files
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    # Load metrics from all files
    all_metrics = []
    for file_path in metrics_files:
        # Convert to string if it's a Path object
        file_str = str(file_path)
        
        # Skip summary metrics files
        if "summary" in file_str:
            continue
            
        try:
            metrics = load_metrics(file_str)
            # Extract robot and sample IDs from filename
            parts = os.path.basename(file_str).split('_')
            dataset = parts[0]
            
            # Make sure this is a regular metrics file with robot and sample info
            if len(parts) >= 3 and "robot" in parts[1] and "sample" in parts[2]:
                robot_idx = int(parts[1].replace('robot', ''))
                sample_idx = int(parts[2].replace('sample', ''))
                
                metrics['robot_idx'] = robot_idx
                metrics['sample_idx'] = sample_idx
                metrics['dataset'] = dataset
                all_metrics.append(metrics)
            else:
                print(f"Skipping non-standard metrics file: {file_str}")
        except (ValueError, IndexError) as e:
            print(f"Error processing metrics file {file_str}: {e}")
            continue
    
    # Check if we have any valid metrics
    if not all_metrics:
        print("No valid metrics files found for summary plot.")
        return
        
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(all_metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Position errors by robot
    ax = axes[0, 0]
    robot_position_errors = metrics_df.groupby('robot_idx')['mean_pos_error'].agg(['mean', 'std'])
    robot_position_errors.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False)
    ax.set_xlabel('Robot ID')
    ax.set_ylabel('Mean Position Error (m)')
    ax.set_title('Position Error by Robot')
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution of position errors
    ax = axes[0, 1]
    sns.histplot(metrics_df['mean_pos_error'], kde=True, ax=ax)
    ax.set_xlabel('Mean Position Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Position Errors')
    ax.grid(True, alpha=0.3)
    
    # 3. MSE by state variable
    ax = axes[1, 0]
    state_names = ['delta_x', 'delta_y', 'yaw', 'steering', 'velocity', 'side_slip', 'yaw_rate']
    state_rmse = [metrics_df[f'rmse_{state}'].mean() for state in state_names]
    
    ax.bar(state_names, state_rmse)
    ax.set_xlabel('State Variable')
    ax.set_ylabel('RMSE')
    ax.set_title('Average RMSE by State Variable')
    ax.set_xticklabels(state_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 4. Position error vs. final error
    ax = axes[1, 1]
    ax.scatter(metrics_df['mean_pos_error'], metrics_df['final_pos_error'], alpha=0.5)
    ax.set_xlabel('Mean Position Error (m)')
    ax.set_ylabel('Final Position Error (m)')
    ax.set_title('Mean vs. Final Position Error')
    ax.grid(True, alpha=0.3)
    
    # Add line of equality
    min_val = min(metrics_df['mean_pos_error'].min(), metrics_df['final_pos_error'].min())
    max_val = max(metrics_df['mean_pos_error'].max(), metrics_df['final_pos_error'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add overall title
    plt.suptitle(f'Metrics Summary ({len(all_metrics)} Valid Samples)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics summary plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_error_time_analysis(trajectory_files: List[str], 
                           num_samples: int = 5,
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
    """
    Analyze how position error changes over time across multiple trajectories.
    
    Args:
        trajectory_files: List of trajectory CSV files
        num_samples: Number of individual samples to plot
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # List to store average error over time
    all_errors = []
    
    # Plot individual trajectories
    for i, file_path in enumerate(trajectory_files[:num_samples]):
        traj_data = load_trajectory_data(file_path)
        
        # Extract robot index from filename
        parts = os.path.basename(file_path).split('_')
        robot_idx = parts[1].replace('robot', '')
        
        # Plot position error
        if i == 0:
            ax.plot(traj_data['timestep'], traj_data['pos_error'], 'k-', alpha=0.3, label='Individual Samples')
        else:
            ax.plot(traj_data['timestep'], traj_data['pos_error'], 'k-', alpha=0.3)
        
        # Store for computing average
        all_errors.append(traj_data['pos_error'].values)
    
    # Compute average error over time
    if len(all_errors) > 0:
        # Find the minimum length to ensure alignment
        min_length = min(len(err) for err in all_errors)
        # Truncate all errors to the minimum length
        all_errors = [err[:min_length] for err in all_errors]
        # Stack and compute mean
        mean_error = np.mean(all_errors, axis=0)
        std_error = np.std(all_errors, axis=0)
        
        # Create time array for the mean error
        time = traj_data['timestep'].values[:min_length]
        
        # Plot average error
        ax.plot(time, mean_error, 'r-', linewidth=3, label='Mean Error')
        ax.fill_between(time, mean_error - std_error, mean_error + std_error, 
                       alpha=0.2, color='red', label='±1 Std Dev')
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Position Error (m)', fontsize=14)
    ax.set_title('Position Error vs. Time Analysis', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error time analysis saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_dataset_comparison(test_metrics: List[str], val_metrics: List[str], 
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """
    Compare metrics between test and validation datasets.
    
    Args:
        test_metrics: List of test dataset metric files
        val_metrics: List of validation dataset metric files
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    # Load test metrics
    test_data = []
    for file_path in test_metrics:
        metrics = load_metrics(file_path)
        metrics['dataset'] = 'test'
        test_data.append(metrics)
    
    # Load validation metrics
    val_data = []
    for file_path in val_metrics:
        metrics = load_metrics(file_path)
        metrics['dataset'] = 'validation'
        val_data.append(metrics)
    
    # Combine datasets
    all_data = pd.DataFrame(test_data + val_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. MSE comparison
    ax = axes[0, 0]
    sns.boxplot(x='dataset', y='mse', data=all_data, ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('MSE')
    ax.set_title('MSE Comparison')
    ax.grid(True, alpha=0.3)
    
    # 2. Position error comparison
    ax = axes[0, 1]
    sns.boxplot(x='dataset', y='mean_pos_error', data=all_data, ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean Position Error (m)')
    ax.set_title('Position Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # 3. Final position error comparison
    ax = axes[1, 0]
    sns.boxplot(x='dataset', y='final_pos_error', data=all_data, ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Final Position Error (m)')
    ax.set_title('Final Position Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # 4. Max position error comparison
    ax = axes[1, 1]
    sns.boxplot(x='dataset', y='max_pos_error', data=all_data, ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Max Position Error (m)')
    ax.set_title('Max Position Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle('Test vs. Validation Dataset Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset comparison saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def main():
    """Main visualization script."""
    print("=" * 60)
    print("Hybrid Neural ODE Evaluation Results Visualization")
    print("=" * 60)
    
    # Create output directory for plots
    plots_dir = "evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if evaluation_results directory exists
    if not os.path.exists("evaluation_results"):
        print("Error: evaluation_results directory not found.")
        print("Please run test_model.py first to generate evaluation data.")
        return
    
    # Find evaluation files
    print("Finding evaluation files...")
    test_traj_files, test_metrics_files = find_evaluation_files("evaluation_results", "test")
    val_traj_files, val_metrics_files = find_evaluation_files("evaluation_results", "val")
    
    print(f"Found {len(test_traj_files)} test trajectory files and {len(test_metrics_files)} test metrics files")
    print(f"Found {len(val_traj_files)} validation trajectory files and {len(val_metrics_files)} validation metrics files")
    
    # Check if we have summary metrics
    test_summary_path = "evaluation_results/test_summary_metrics.csv"
    val_summary_path = "evaluation_results/val_summary_metrics.csv"
    
    has_test_summary = os.path.exists(test_summary_path)
    has_val_summary = os.path.exists(val_summary_path)
    
    print(f"Test summary metrics available: {has_test_summary}")
    print(f"Validation summary metrics available: {has_val_summary}")
    
    # Create sample trajectory visualizations
    if test_traj_files:
        print("\nCreating sample trajectory visualizations...")
        
        # Limit to 5 samples for visualization
        max_samples = min(5, len(test_traj_files))
        for i, traj_file in enumerate(test_traj_files[:max_samples]):
            print(f"  Processing {traj_file}...")
            traj_data = load_trajectory_data(traj_file)
            
            # Add sample and robot info to dataframe if not present
            if 'sample_idx' not in traj_data.columns:
                filename = os.path.basename(traj_file)
                parts = filename.split('_')
                traj_data['robot_idx'] = int(parts[1].replace('robot', ''))
                traj_data['sample_idx'] = int(parts[2].replace('sample', ''))
            
            # Create trajectory comparison plot
            save_path = os.path.join(plots_dir, f"trajectory_{i}.png")
            plot_trajectory_comparison(traj_data, save_path=save_path, show_plot=False)
            
            # Create comprehensive state plot
            save_path = os.path.join(plots_dir, f"comprehensive_states_{i}.png")
            plot_comprehensive_states(traj_data, save_path=save_path, show_plot=False)
            
            # Create position error plot
            save_path = os.path.join(plots_dir, f"position_error_{i}.png")
            plot_position_error(traj_data, save_path=save_path, show_plot=False)
    
    # Create metrics summaries
    if test_metrics_files:
        print("\nCreating metrics summaries...")
        
        # Filter out summary metrics files and convert to strings
        test_metrics_files_filtered = [str(f) for f in test_metrics_files if "summary" not in str(f)]
    
        # Test metrics summary
        if test_metrics_files_filtered:
            save_path = os.path.join(plots_dir, "test_metrics_summary.png")
            plot_metrics_summary(test_metrics_files_filtered, save_path=save_path, show_plot=False)
        else:
            print("No valid metrics files found after filtering.")
        
        # Error time analysis
        save_path = os.path.join(plots_dir, "error_time_analysis.png")
        plot_error_time_analysis(test_traj_files, save_path=save_path, show_plot=False)
    
    # Create dataset comparison if we have both test and validation data
    if test_metrics_files and val_metrics_files:
        print("\nCreating dataset comparison...")
        
        # Filter and convert both lists
        test_metrics_filtered = [str(f) for f in test_metrics_files if "summary" not in str(f)]
        val_metrics_filtered = [str(f) for f in val_metrics_files if "summary" not in str(f)]
        
        if test_metrics_filtered and val_metrics_filtered:
            save_path = os.path.join(plots_dir, "dataset_comparison.png")
            plot_dataset_comparison(test_metrics_filtered, val_metrics_filtered, 
                                   save_path=save_path, show_plot=False)
        else:
            print("Insufficient filtered metrics files for dataset comparison.")
    
    print(f"\nVisualization complete! Plots saved to {plots_dir}/")
    print("Key visualizations:")
    print("  - trajectory_*.png: XY path comparisons")
    print("  - comprehensive_states_*.png: All state variables")
    print("  - position_error_*.png: Position error over time")
    print("  - test_metrics_summary.png: Summary of test metrics")
    print("  - error_time_analysis.png: How errors evolve over time")
    print("  - dataset_comparison.png: Test vs. validation comparison")


if __name__ == "__main__":
    main()
