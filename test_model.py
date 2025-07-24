#!/usr/bin/env python3
"""
Model Evaluation for Hybrid Neural ODE

Evaluates trained model performance on test data.
Computes prediction errors and saves trajectory comparison data as CSV files.
No direct plotting to avoid matplotlib issues in the JAX environment.
"""

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from models import HybridODE, create_model_from_config
from data_processing import process_data


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def compute_metrics(true_states: np.ndarray, pred_states: np.ndarray, 
                   state_names: List[str]) -> Dict[str, float]:
    """
    Compute evaluation metrics between true and predicted states.
    
    Args:
        true_states: Ground truth states (T, state_dim)
        pred_states: Predicted states (T, state_dim)
        state_names: Names of state variables
        
    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    mse = np.mean((true_states - pred_states) ** 2)
    rmse = np.sqrt(mse)
    
    # Per-state metrics
    state_metrics = {}
    for i, name in enumerate(state_names):
        state_mse = np.mean((true_states[:, i] - pred_states[:, i]) ** 2)
        state_rmse = np.sqrt(state_mse)
        state_metrics[f"rmse_{name}"] = float(state_rmse)
    
    # Position error metrics (focus on delta_x and delta_y)
    pos_error = np.sqrt(
        (true_states[:, 0] - pred_states[:, 0])**2 + 
        (true_states[:, 1] - pred_states[:, 1])**2
    )
    mean_pos_error = float(np.mean(pos_error))
    max_pos_error = float(np.max(pos_error))
    final_pos_error = float(pos_error[-1])
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mean_pos_error': mean_pos_error,
        'max_pos_error': max_pos_error,
        'final_pos_error': final_pos_error,
        **state_metrics
    }
    
    return metrics


def evaluate_sample(model: HybridODE, params: Dict, sample: Dict, 
                  state_scaler: Dict, input_scaler: Dict) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate model on a single multi-step sample.
    
    Args:
        model: Hybrid ODE model
        params: Model parameters
        sample: Dictionary with sample data
        state_scaler: State normalization parameters
        input_scaler: Input normalization parameters
        
    Returns:
        Tuple of (metrics, predicted_states)
    """
    # Convert sample to JAX arrays
    current_state = jnp.array(sample['current_state'])
    current_input = jnp.array(sample['current_input'])
    future_inputs = jnp.array(sample['future_inputs'])
    timesteps = jnp.array(sample['timesteps'])
    true_future_states = jnp.array(sample['future_states'])
    
    # Create input sequence for prediction (including current input)
    inputs_sequence = jnp.concatenate([current_input[None, :], future_inputs])
    
    # Initialize state array
    initial_state = current_state
    
    # Make prediction
    predicted_trajectory = model.predict_trajectory(
        params, initial_state, inputs_sequence, timesteps,
        state_scaler, input_scaler
    )
    
    # Get only future states (exclude initial state which is at index 0)
    predicted_future = predicted_trajectory[1:]
    
    # Convert to numpy for metrics calculation
    true_future_np = np.array(true_future_states)
    pred_future_np = np.array(predicted_future)
    
    # Compute metrics using normalized values
    metrics = compute_metrics(true_future_np, pred_future_np, 
                             ["delta_x", "delta_y", "yaw", "steering", "velocity", "side_slip", "yaw_rate"])
    
    return metrics, pred_future_np


# ============================================================================
# DATA EXPORT
# ============================================================================

def save_trajectory_data(sample_idx: int, robot_idx: int, 
                        true_trajectory: np.ndarray, pred_trajectory: np.ndarray,
                        timesteps: np.ndarray, metrics: Dict, dataset: str,
                        output_dir: str = "evaluation_results"):
    """
    Save trajectory comparison data as CSV.
    
    Args:
        sample_idx: Sample index
        robot_idx: Robot ID
        true_trajectory: True trajectory (T, state_dim)
        pred_trajectory: Predicted trajectory (T, state_dim)
        timesteps: Time values
        metrics: Evaluation metrics
        dataset: Dataset name ('train', 'val', or 'test')
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trajectory comparison as CSV
    csv_path = f"{output_dir}/{dataset}_robot{robot_idx}_sample{sample_idx}_trajectory.csv"
    
    with open(csv_path, 'w') as f:
        # Header
        f.write("timestep,")
        f.write("true_delta_x,true_delta_y,true_yaw,true_steering,true_velocity,true_side_slip,true_yaw_rate,")
        f.write("pred_delta_x,pred_delta_y,pred_yaw,pred_steering,pred_velocity,pred_side_slip,pred_yaw_rate,")
        f.write("pos_error\n")
        
        # Data
        for i in range(len(timesteps) - 1):  # Skip initial state in timesteps
            # Calculate position error
            pos_error = np.sqrt(
                (true_trajectory[i, 0] - pred_trajectory[i, 0])**2 + 
                (true_trajectory[i, 1] - pred_trajectory[i, 1])**2
            )
            
            # Write row
            f.write(f"{timesteps[i+1]:.6f},")
            
            # True states
            for j in range(7):
                f.write(f"{true_trajectory[i, j]:.6f},")
            
            # Predicted states
            for j in range(7):
                f.write(f"{pred_trajectory[i, j]:.6f},")
            
            # Position error
            f.write(f"{pos_error:.6f}\n")
    
    # Save metrics as JSON
    metrics_path = f"{output_dir}/{dataset}_robot{robot_idx}_sample{sample_idx}_metrics.json"
    
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_summary_metrics(metrics_list: List[Dict], dataset: str, 
                        output_dir: str = "evaluation_results"):
    """
    Save summary metrics for a dataset.
    
    Args:
        metrics_list: List of metrics dictionaries
        dataset: Dataset name
        output_dir: Output directory
    """
    # Calculate average metrics
    avg_metrics = {}
    
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        avg_metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    
    # Save as JSON
    import json
    with open(f"{output_dir}/{dataset}_summary_metrics.json", 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    # Save as CSV for easy import to other tools
    with open(f"{output_dir}/{dataset}_summary_metrics.csv", 'w') as f:
        f.write("metric,mean,std,min,max\n")
        for key, values in avg_metrics.items():
            f.write(f"{key},{values['mean']:.6f},{values['std']:.6f},{values['min']:.6f},{values['max']:.6f}\n")


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_dataset(model: HybridODE, params: Dict, samples: List[Dict],
                   state_scaler: Dict, input_scaler: Dict,
                   dataset_name: str, max_samples: int = None) -> List[Dict]:
    """
    Evaluate model on a dataset of samples.
    
    Args:
        model: Hybrid ODE model
        params: Model parameters
        samples: List of sample dictionaries
        state_scaler: State normalization parameters
        input_scaler: Input normalization parameters
        dataset_name: Name of dataset for logging
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        List of metrics dictionaries
    """
    print(f"Evaluating {dataset_name} dataset...")
    
    # Limit number of samples if specified
    if max_samples is not None:
        samples = samples[:max_samples]
    
    # Track metrics
    all_metrics = []
    
    # Evaluate each sample
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
        robot_idx = sample['robot_idx']
        
        # Skip if already evaluated
        output_path = f"evaluation_results/{dataset_name}_robot{robot_idx}_sample{i}_trajectory.csv"
        if os.path.exists(output_path):
            print(f"  Sample {i} (Robot {robot_idx}) already evaluated, skipping...")
            continue
        
        # Evaluate sample
        metrics, pred_trajectory = evaluate_sample(
            model, params, sample, state_scaler, input_scaler
        )
        
        # Add sample info to metrics
        metrics['sample_idx'] = i
        metrics['robot_idx'] = robot_idx
        
        # Add to metrics list
        all_metrics.append(metrics)
        
        # Save trajectory data
        save_trajectory_data(
            i, robot_idx,
            sample['future_states'], pred_trajectory,
            sample['timesteps'], metrics, dataset_name
        )
        
        # Print metrics for a few samples
        if i < 5 or i % 100 == 0:
            print(f"  Sample {i} (Robot {robot_idx}): MSE = {metrics['mse']:.6f}, "
                  f"Position Error = {metrics['mean_pos_error']:.6f}m")
    
    # Save summary metrics
    save_summary_metrics(all_metrics, dataset_name)
    
    return all_metrics


def load_model(model_path: str, config: Dict) -> Tuple[HybridODE, Dict, Dict, Dict]:
    """
    Load trained model and parameters.
    
    Args:
        model_path: Path to model file
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, params, state_scaler, input_scaler)
    """
    print(f"Loading model from {model_path}...")
    
    # Create model
    model = create_model_from_config(config)
    
    # Load saved parameters and scalers
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    params = saved_data['params']
    state_scaler = saved_data['state_scaler']
    input_scaler = saved_data['input_scaler']
    
    print("Model loaded successfully")
    
    return model, params, state_scaler, input_scaler


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("Hybrid Neural ODE Model Evaluation")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_path = "results/model_best.pkl"  # Use the best model from training
    model, params, state_scaler, input_scaler = load_model(model_path, config)
    
    # Load data samples
    print("Loading data samples...")
    train_samples, val_samples, test_samples, _ = load_multistep_samples()
    print(f"Loaded {len(train_samples)} training samples")
    print(f"Loaded {len(val_samples)} validation samples")
    print(f"Loaded {len(test_samples)} test samples")
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_metrics = evaluate_dataset(
        model, params, test_samples, state_scaler, input_scaler, 
        "test", max_samples=None
    )
    
    # Optional: Evaluate on a small subset of validation data for comparison
    print("\nEvaluating on validation data (subset)...")
    val_metrics = evaluate_dataset(
        model, params, val_samples, state_scaler, input_scaler, 
        "val", max_samples=500  # Limit to 500 samples for efficiency
    )
    
    # Print summary results
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    # Calculate average metrics
    test_mse = np.mean([m['mse'] for m in test_metrics])
    test_pos_error = np.mean([m['mean_pos_error'] for m in test_metrics])
    val_mse = np.mean([m['mse'] for m in val_metrics])
    val_pos_error = np.mean([m['mean_pos_error'] for m in val_metrics])
    
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test Position Error: {test_pos_error:.6f}m")
    print(f"Validation MSE: {val_mse:.6f}")
    print(f"Validation Position Error: {val_pos_error:.6f}m")
    
    print("\nDetailed metrics saved to evaluation_results/ directory")
    print("Run visualize_results.py to generate plots from the saved data")


# Helper function to load multistep samples
def load_multistep_samples(data_path: str = "processed_data"):
    """Load multi-step prediction samples."""
    from train import load_multistep_samples as load_samples
    return load_samples(data_path)


if __name__ == "__main__":
    main()
