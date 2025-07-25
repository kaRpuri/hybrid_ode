#!/usr/bin/env python3
"""
Data processing for Hybrid Neural ODE Vehicle Dynamics Model

This module loads multi-robot trajectory data, splits it by robot (not time),
applies normalization, and prepares data for hybrid ODE training.

IMPORTANT: Each robot has a separate trajectory. We do NOT concatenate 
different robot trajectories as they represent different driving scenarios.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import pickle

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def load_npz_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and concatenate data from all NPZ files in the given directory.

    Returns:
        states: (500, total_robots, 7)
        inputs: (500, total_robots, 2)
        timestamps: (500,)
    """
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("hybrid_neural_ode_data*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    all_states = []
    all_inputs = []
    timestamps_ref = None

    for idx, file_path in enumerate(npz_files):
        data = np.load(file_path)
        states = data['states']  # (500, n, 7)
        inputs = data['inputs']  # (500, n, 2)
        timestamps = data['timestamps']  # (500,)

        all_states.append(states)
        all_inputs.append(inputs)

        if idx == 0:
            timestamps_ref = timestamps
        else:
            if not np.allclose(timestamps, timestamps_ref):
                raise ValueError(f"Timestamps in {file_path} do not match the reference.")

        print(f"Loaded {file_path.name}: states {states.shape}, inputs {inputs.shape}")

    # Concatenate along the robot axis (axis=1)
    states_concat = np.concatenate(all_states, axis=1)
    inputs_concat = np.concatenate(all_inputs, axis=1)

    print(f"Concatenated states shape: {states_concat.shape}")
    print(f"Concatenated inputs shape: {inputs_concat.shape}")
    print(f"Timestamps shape: {timestamps_ref.shape}")

    return states_concat, inputs_concat, timestamps_ref

def validate_data(states: np.ndarray, inputs: np.ndarray, config: Dict) -> None:
    """Validate data quality and ranges."""
    if np.any(np.isnan(states)) or np.any(np.isnan(inputs)):
        raise ValueError("Data contains NaN values")
    
    if np.any(np.isinf(states)) or np.any(np.isinf(inputs)):
        raise ValueError("Data contains infinite values")
    
    if config.get('verbose', True):
        print("Data validation passed")

def convert_to_relative_positions(states: np.ndarray) -> np.ndarray:
    """
    Convert absolute positions (x_pos, y_pos) to relative positions (Δx, Δy).
    
    Args:
        states: (T, N, 7) where states are [x_pos, y_pos, yaw, steering, velocity, side_slip, yaw_rate]
        
    Returns:
        states with [Δx, Δy, yaw, steering, velocity, side_slip, yaw_rate]
    """
    states_relative = states.copy()
    
    # For each robot, convert to relative positions
    for robot_idx in range(states.shape[1]):
        # Get initial positions for this robot
        x_initial = states[0, robot_idx, 0]  # Initial x_pos
        y_initial = states[0, robot_idx, 1]  # Initial y_pos
        
        # Convert to relative positions: Δx = x_pos - x_initial
        states_relative[:, robot_idx, 0] = states[:, robot_idx, 0] - x_initial  # Δx
        states_relative[:, robot_idx, 1] = states[:, robot_idx, 1] - y_initial  # Δy
        
        # Other states remain unchanged: [yaw, steering, velocity, side_slip, yaw_rate]
    
    return states_relative

def split_data_by_robots(states: np.ndarray, inputs: np.ndarray, timestamps: np.ndarray, 
                        config: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Split data by robot trajectories using ratio-based or count-based splitting.
    
    Each robot has a complete independent trajectory that should not be mixed.
    """
    total_robots = states.shape[1]
    
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio'] 
    test_ratio = config['data']['test_ratio']
    
    # Validate ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
    
    # Calculate counts
    train_count = int(total_robots * train_ratio)
    val_count = int(total_robots * val_ratio)
    test_count = total_robots - train_count - val_count  # Ensure all robots are used
        
    
    # Create robot splits
   
    np.random.seed(config.get('random_seed', 42))  # Ensure reproducible splits
    all_robot_ids = np.arange(total_robots)
    np.random.shuffle(all_robot_ids)
    
    train_robots = all_robot_ids[:train_count].tolist()
    val_robots = all_robot_ids[train_count:train_count + val_count].tolist()
    test_robots = all_robot_ids[train_count + val_count:train_count + val_count + test_count].tolist()
    
    # Convert absolute positions to relative positions (Δx, Δy)
    print("Converting absolute positions to relative positions (Δx, Δy)...")
    states_relative = convert_to_relative_positions(states)
    
    # Extract data for each split (keeping robot dimension)
    train_data = {
        'states': states_relative[:, train_robots, :],    # (500, train_count, 7) with Δx, Δy
        'inputs': inputs[:, train_robots, :],             # (500, train_count, 2) 
        'timestamps': timestamps,                         # (500,)
        'robot_ids': train_robots
    }
    
    val_data = {
        'states': states_relative[:, val_robots, :],      # (500, val_count, 7) with Δx, Δy
        'inputs': inputs[:, val_robots, :],               # (500, val_count, 2)
        'timestamps': timestamps,                         # (500,)
        'robot_ids': val_robots  
    }
    
    test_data = {
        'states': states_relative[:, test_robots, :],     # (500, test_count, 7) with Δx, Δy
        'inputs': inputs[:, test_robots, :],              # (500, test_count, 2)
        'timestamps': timestamps,                         # (500,)
        'robot_ids': test_robots
    }
    
    if config.get('verbose', True):
        print(f"Data split (from {total_robots} total robots):")
        print(f"  Train: {len(train_robots)} robots (IDs: {train_robots[:5]}{'...' if len(train_robots) > 5 else ''}), shape {train_data['states'].shape}")
        print(f"  Val:   {len(val_robots)} robots (IDs: {val_robots[:5]}{'...' if len(val_robots) > 5 else ''}), shape {val_data['states'].shape}")
        print(f"  Test:  {len(test_robots)} robots (IDs: {test_robots[:5]}{'...' if len(test_robots) > 5 else ''}), shape {test_data['states'].shape}")
        print(f"  Note: Positions converted to relative coordinates (Δx, Δy)")
    
    return train_data, val_data, test_data

def compute_normalization_params(train_states: np.ndarray, train_inputs: np.ndarray) -> Dict:
    """
    Compute z-score normalization parameters from training data ONLY.
    
    Args:
        train_states: (500, train_count, 7) - training robot states
        train_inputs: (500, train_count, 2) - training robot inputs

    Returns:
        Dictionary with normalization parameters
    """
    # Flatten across time and robots for statistics: (1500, 7) and (1500, 2)
    all_train_states = train_states.reshape(-1, train_states.shape[-1])
    all_train_inputs = train_inputs.reshape(-1, train_inputs.shape[-1])
    
    normalization_params = {
        'state_mean': np.mean(all_train_states, axis=0),  # (7,)
        'state_std': np.std(all_train_states, axis=0),    # (7,)
        'input_mean': np.mean(all_train_inputs, axis=0),  # (2,)
        'input_std': np.std(all_train_inputs, axis=0),    # (2,)
    }
    
    print(f"Normalization parameters computed from {all_train_states.shape[0]} training samples")
    print(f"State means: {normalization_params['state_mean']}")
    print(f"State stds:  {normalization_params['state_std']}")
    print(f"Input means: {normalization_params['input_mean']}")
    print(f"Input stds:  {normalization_params['input_std']}")
    
    return normalization_params

def apply_normalization(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization: (data - mean) / std"""
    return (data - mean) / std

def normalize_datasets(train_data: Dict, val_data: Dict, test_data: Dict, 
                      normalization_params: Dict) -> Tuple[Dict, Dict, Dict]:
    """Apply normalization to all datasets using training parameters."""
    
    def normalize_split(data_dict: Dict) -> Dict:
        normalized = data_dict.copy()
        normalized['states_normalized'] = apply_normalization(
            data_dict['states'], 
            normalization_params['state_mean'], 
            normalization_params['state_std']
        )
        normalized['inputs_normalized'] = apply_normalization(
            data_dict['inputs'],
            normalization_params['input_mean'],
            normalization_params['input_std'] 
        )
        # Keep original data as well
        normalized['states_original'] = data_dict['states']
        normalized['inputs_original'] = data_dict['inputs']
        return normalized
    
    train_normalized = normalize_split(train_data)
    val_normalized = normalize_split(val_data)
    test_normalized = normalize_split(test_data)
    
    # Verify normalization worked for training data
    train_states_flat = train_normalized['states_normalized'].reshape(-1, 7)
    train_mean_check = np.mean(train_states_flat, axis=0)
    train_std_check = np.std(train_states_flat, axis=0)
    
    print(f"Normalization verification (training data):")
    print(f"  Normalized mean: {train_mean_check} (should be ~0)")
    print(f"  Normalized std:  {train_std_check} (should be ~1)")
    
    return train_normalized, val_normalized, test_normalized

def add_noise(data: np.ndarray, noise_std: float, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to data for training robustness."""
    np.random.seed(seed)
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise



def process_data(config_path: str = "config.yaml") -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Main data processing pipeline with simplified output.
    
    Returns:
        train_samples, val_samples, test_samples, normalization_params
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load raw data
    print("Loading data...")
    states, inputs, timestamps = load_npz_data("data")
    
    # Validate data
    validate_data(states, inputs, config)
    
    # Split by robot trajectories
    print("Splitting data by robot trajectories...")
    train_data, val_data, test_data = split_data_by_robots(states, inputs, timestamps, config)
    
    # Compute normalization parameters from training data only
    print("Computing normalization parameters...")
    normalization_params = compute_normalization_params(
        train_data['states'], train_data['inputs']
    )
    
    # Apply normalization to all datasets
    print("Applying normalization...")
    train_data, val_data, test_data = normalize_datasets(
        train_data, val_data, test_data, normalization_params
    )
    
    # Add noise to training data if enabled
    if config['training'].get('noise_std', 0) > 0:
        print(f"Adding noise (std={config['training']['noise_std']}) to training data...")
        train_data['states_normalized'] = add_noise(
            train_data['states_normalized'], 
            config['training']['noise_std'],
            config.get('random_seed', 42)
        )
    
    # Create multi-step datasets
    n_steps = config['data'].get('num_multi_step_predictions', 10)
    print(f"Creating multi-step datasets with {n_steps} step prediction horizon...")
    
    # Create multi-step samples (direct access to samples list)
    train_samples = create_multistep_dataset(train_data, n_steps)['samples']
    val_samples = create_multistep_dataset(val_data, n_steps)['samples']
    test_samples = create_multistep_dataset(test_data, n_steps)['samples']
    
    # Save processed data in simplified format
    print("Saving processed data...")
    save_processed_data(train_samples, val_samples, test_samples, 
                         normalization_params, config)
    
    print("Data processing complete!")
    return train_samples, val_samples, test_samples, normalization_params

def create_multistep_dataset(data_dict: Dict, n_steps: int) -> Dict:
    """
    Create multi-step prediction dataset from normalized data.
    
    Args:
        data_dict: Dictionary with states_normalized, inputs_normalized, timestamps, robot_ids
        n_steps: Number of future timesteps to include in each sample (prediction horizon)
        
    Returns:
        Dictionary containing multi-step dataset
    
    Each sample in the dataset is a dictionary with the following structure:
    {
        'current_state': ndarray(7,),           # State at time t
        'current_input': ndarray(2,),           # Action at time t
        'future_states': ndarray(n_steps, 7),   # Ground truth states from t+1 to t+n
        'future_inputs': ndarray(n_steps-1, 2), # Control inputs from t+1 to t+n-1
        'timesteps': ndarray(n_steps+1,),       # Time values from t to t+n
        'robot_idx': int                        # Which robot this sample is from
    }
    
    Note: Samples are created per robot to maintain trajectory coherence. 
    We never create samples that span across different robots to preserve 
    the physical consistency of each vehicle's dynamics.
    """
    # Extract necessary data
    states = data_dict['states_normalized']  # (timesteps, num_robots, 7)
    inputs = data_dict['inputs_normalized']  # (timesteps, num_robots, 2)
    timestamps = data_dict['timestamps']     # (timesteps,)
    robot_ids = data_dict['robot_ids']       # List of robot IDs
    
    num_timesteps, num_robots, _ = states.shape
    valid_samples = []
    
    # For each robot trajectory
    for r in range(num_robots):
        # For each valid timestep (excluding the last n-1 steps that don't have enough future data)
        for t in range(num_timesteps - n_steps):
            # Current state and input
            current_state = states[t, r].copy()
            current_input = inputs[t, r].copy()
            
            # Future states for prediction targets
            future_states = states[t+1:t+n_steps+1, r].copy()
            
            # Future inputs needed for multi-step prediction
            future_inputs = inputs[t+1:t+n_steps, r].copy()
            
            # Timestep information (current + future)
            sample_timestamps = timestamps[t:t+n_steps+1].copy()
            
            # Create sample dictionary
            sample = {
                'current_state': current_state,
                'current_input': current_input,
                'future_states': future_states,
                'future_inputs': future_inputs,
                'timesteps': sample_timestamps,
                'robot_idx': robot_ids[r]  # Store the actual robot ID
            }
            valid_samples.append(sample)
    
    return {
        'samples': valid_samples,
        'num_samples': len(valid_samples),
        'n_steps': n_steps,
        'num_robots': num_robots
    }

def save_processed_data(train_samples: List, val_samples: List, test_samples: List,
                       normalization_params: Dict, config: Dict) -> None:
    """Save only the essential data needed for training."""
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save samples directly using pickle
    with open(output_dir / 'train_samples.pkl', 'wb') as f:
        pickle.dump(train_samples, f)
    with open(output_dir / 'val_samples.pkl', 'wb') as f:
        pickle.dump(val_samples, f)
    with open(output_dir / 'test_samples.pkl', 'wb') as f:
        pickle.dump(test_samples, f)
    
    # Save normalization parameters
    np.save(output_dir / 'normalization_params.npy', normalization_params)
    
    # Save simplified metadata
    metadata = {
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'num_test_samples': len(test_samples),
        'state_names': config['model']['state_names'],
        'input_names': config['model']['input_names'],
        'noise_std': config['training'].get('noise_std', 0),
        'num_multi_step_predictions': config['data'].get('num_multi_step_predictions', 10)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Simplified processed data saved to {output_dir}")

if __name__ == "__main__":
    """Test the data processing pipeline."""
    print("=" * 60)
    print("Testing Data Processing Pipeline")
    print("=" * 60)
    
    try:
        # Run the complete pipeline
        train_samples, val_samples, test_samples, norm_params = process_data()
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        print(f"Test samples: {len(test_samples)}")
        
        print(f"\nNormalization parameters:")
        print(f"  State means: {norm_params['state_mean']}")
        print(f"  State stds:  {norm_params['state_std']}")
        
        # Print example sample structure
        if len(train_samples) > 0:
            sample = train_samples[0]
            print(f"\nExample multi-step sample structure:")
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {value}")
        
        # Count robot IDs in samples
        train_robot_ids = set(sample['robot_idx'] for sample in train_samples)
        val_robot_ids = set(sample['robot_idx'] for sample in val_samples)
        test_robot_ids = set(sample['robot_idx'] for sample in test_samples)
        
        print(f"\nUnique robot IDs:")
        print(f"  Training: {train_robot_ids}")
        print(f"  Validation: {val_robot_ids}")
        print(f"  Testing: {test_robot_ids}")
        
        print("\nData processing test completed successfully!")
        
    except Exception as e:
        print(f"\nError during data processing: {e}")
        raise
