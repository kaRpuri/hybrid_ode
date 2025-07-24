#!/usr/bin/env python3
"""
Training Pipeline for Hybrid Neural ODE Vehicle Dynamics

Implements mini-batch training with multi-step trajectory prediction:
- Uses samples with future state/input sequences
- Trains on batches across all robots simultaneously
- Supports time-weighted loss for emphasizing short-term accuracy
- Uses RK4 integration for improved numerical stability
"""

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import pickle
import wandb
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import partial
import optax
from flax.training import train_state
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import HybridODE, create_model_from_config, create_train_state
from data_processing import process_data


# ============================================================================
# DATA LOADING
# ============================================================================

def load_multistep_samples(data_path: str = "processed_data") -> Tuple[List, List, List, Dict]:
    """Load multi-step prediction samples from processed data."""
    data_dir = Path(data_path)
    
    # Load samples
    with open(data_dir / 'train_samples.pkl', 'rb') as f:
        train_samples = pickle.load(f)
        
    with open(data_dir / 'val_samples.pkl', 'rb') as f:
        val_samples = pickle.load(f)
        
    with open(data_dir / 'test_samples.pkl', 'rb') as f:
        test_samples = pickle.load(f)
    
    # Load normalization parameters
    norm_params = np.load(data_dir / 'normalization_params.npy', allow_pickle=True).item()
    
    # Convert to format expected by model
    state_scaler = {
        'mean': norm_params['state_mean'],
        'std': norm_params['state_std']
    }
    input_scaler = {
        'mean': norm_params['input_mean'],
        'std': norm_params['input_std']
    }
    
    return train_samples, val_samples, test_samples, (state_scaler, input_scaler)


# ============================================================================
# BATCH HANDLING
# ============================================================================

def create_batches(samples: List[Dict], batch_size: int, shuffle: bool = True) -> List[Dict]:
    """Create mini-batches from samples for efficient training."""
    # Convert samples to arrays for batching
    arrays = {}
    for key in samples[0].keys():
        if key == 'robot_idx':
            arrays[key] = np.array([s[key] for s in samples])
        else:
            arrays[key] = np.stack([s[key] for s in samples])
    
    # Shuffle indices if requested
    num_samples = len(samples)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    # Create batches
    batches = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch = {}
        for key, array in arrays.items():
            batch[key] = array[batch_indices]
            
        batches.append(batch)
    
    return batches


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def multistep_prediction_loss(params: Dict, model: HybridODE, batch: Dict, 
                            state_scaler: Dict, input_scaler: Dict,) -> jnp.ndarray:
    """
    Compute loss for multi-step prediction on a batch of samples.
    
    Args:
        params: Model parameters
        model: Hybrid ODE model
        batch: Dictionary with batch data
        state_scaler: State normalization parameters
        input_scaler: Input normalization parameters
     
        
    Returns:
        Mean squared error loss
    """
    # Extract batch data
    initial_states = batch['current_state']  # (batch_size, state_dim)
    
    # Create input sequence for prediction, including current input
    future_inputs = batch['future_inputs']  # (batch_size, n_steps-1, input_dim)
    current_inputs = batch['current_input'][:, None, :]  # Add time dimension
    inputs_sequence = jnp.concatenate([current_inputs, future_inputs], axis=1)
    
    timesteps = batch['timesteps']  # (batch_size, n_steps+1)
    true_future_states = batch['future_states']  # (batch_size, n_steps, state_dim)
    
    # Predict future states using multi-step prediction
    # Returns: (batch_size, n_steps+1, state_dim), including initial state
    predicted_trajectories = model.predict_batch_trajectories(
        params, initial_states, inputs_sequence, timesteps,
        state_scaler, input_scaler
    )
    
    # Extract future states, excluding initial state
    predicted_future_states = predicted_trajectories[:, 1:, :]  # (batch_size, n_steps, state_dim)
    
    # Check that shapes match for proper loss calculation
    assert predicted_future_states.shape == true_future_states.shape, \
        f"Shape mismatch: predictions {predicted_future_states.shape} vs ground truth {true_future_states.shape}"
    
    # Compute mean squared error
    squared_errors = jnp.mean((true_future_states - predicted_future_states) ** 2)
    
    return squared_errors


def time_weighted_loss(params: Dict, model: HybridODE, batch: Dict, 
                     state_scaler: Dict, input_scaler: Dict,
                     decay_rate: float = 0.9) -> jnp.ndarray:
    """
    Time-weighted loss that emphasizes near-future prediction accuracy.
    
    Args:
        params: Model parameters
        model: Hybrid ODE model
        batch: Dictionary with batch data
        state_scaler: State normalization parameters
        input_scaler: Input normalization parameters
        decay_rate: Weight decay rate (higher values weight near-future more)
  
        
    Returns:
        Weighted mean squared error loss
    """
    # Extract batch data
    initial_states = batch['current_state']
    
    # Create input sequence for prediction
    future_inputs = batch['future_inputs']
    current_inputs = batch['current_input'][:, None, :]
    inputs_sequence = jnp.concatenate([current_inputs, future_inputs], axis=1)
    
    timesteps = batch['timesteps']
    true_future_states = batch['future_states']
    
    # Predict future trajectories
    predicted_trajectories = model.predict_batch_trajectories(
        params, initial_states, inputs_sequence, timesteps,
        state_scaler, input_scaler
    )
    
    # Extract future states, excluding initial state
    predicted_future_states = predicted_trajectories[:, 1:, :]
    
    # Check that shapes match for proper loss calculation
    assert predicted_future_states.shape == true_future_states.shape, \
        f"Shape mismatch: predictions {predicted_future_states.shape} vs ground truth {true_future_states.shape}"
        
    # Compute squared errors
    squared_errors = (true_future_states - predicted_future_states) ** 2  # (batch, n_steps, state_dim)
    
    # Create time-based weights (higher weight for earlier timesteps)
    n_steps = true_future_states.shape[1]
    weights = decay_rate ** jnp.arange(n_steps)
    
    # Normalize weights to preserve loss scale
    weights = weights * (n_steps / jnp.sum(weights))
    
    # Apply weights along the time dimension
    weighted_squared_errors = squared_errors * weights[None, :, None]  # (batch, n_steps, state_dim)
    
    # Average across batch, time, and state dimensions
    weighted_loss = jnp.mean(weighted_squared_errors)
    
    return weighted_loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

@partial(jax.jit, static_argnums=(0,), static_argnames=('decay_rate',))
def train_step(model: HybridODE, state: train_state.TrainState, batch: Dict, 
              state_scaler: Dict, input_scaler: Dict, 
              decay_rate: float = 0.0) -> Tuple[train_state.TrainState, Dict]:
    """
    Single training step with loss computation and gradient update.
    """
    # Define loss function based on decay rate - using JAX's conditional
    def loss_fn(params):
        return jax.lax.cond(
            decay_rate > 0,
            lambda _: time_weighted_loss(params, model, batch, state_scaler, input_scaler, decay_rate),
            lambda _: multistep_prediction_loss(params, model, batch, state_scaler, input_scaler),
            operand=None
        )
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Update parameters
    new_state = state.apply_gradients(grads=grads)
    
    # Compute gradient norm
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    
    metrics = {
        'loss': loss,
        'grad_norm': grad_norm,
    }
    
    return new_state, metrics


def evaluate(model: HybridODE, params: Dict, val_batches: List[Dict],
           state_scaler: Dict, input_scaler: Dict) -> Dict:
    """Evaluate model on validation data."""
    losses = []
    
    for batch in val_batches:
        # Convert numpy batch to jax arrays
        jax_batch = {k: jnp.array(v) for k, v in batch.items()}
        
        # Compute validation loss
        loss = multistep_prediction_loss(
            params, model, jax_batch, state_scaler, input_scaler
        )
        losses.append(loss)
    
    avg_loss = jnp.mean(jnp.array(losses))
    
    return {
        'val_loss': avg_loss,
    }



# ============================================================================
# TRAINING CLASS
# ============================================================================

class HybridODETrainer:
    """Training manager for multi-step hybrid ODE model."""
    
    def __init__(self, config: Dict, use_wandb: bool = True):
        self.config = config
        self.model = create_model_from_config(config)
        self.use_wandb = use_wandb and wandb is not None
        
        # Training configuration
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training'].get('weight_decay', 0.0)
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.decay_rate = config['training'].get('prediction_decay_rate', 0.0)
        self.use_weighted_loss = config['training'].get('use_weighted_loss', False)
        self.val_interval = config['training'].get('validation_interval', 10)
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 20)
        
        # Get model parameters
        self.state_names = config['model']['state_names']
        self.input_names = config['model']['input_names']
        
        # Setup output directory
        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        key = jax.random.PRNGKey(config['random_seed'])
        self.train_state = create_train_state(
            self.model, self.learning_rate, key, self.weight_decay
        )
        
        if self.use_wandb:
            self._init_wandb()
            
    def _init_wandb(self):
        """Initialize wandb logging."""
        wandb.init(
            project="hybrid-neural-ode-vehicle",
            name=f"multistep_h{self.config['model']['hidden_size']}_lr{self.learning_rate}",
            config={
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "hidden_size": self.config['model']['hidden_size'],
                "decay_rate": self.decay_rate if self.use_weighted_loss else 0.0,
                "random_seed": self.config['random_seed']
            },
            tags=["vehicle-dynamics", "neural-ode", "multi-step"]
        )
    
    def train(self, train_samples: List[Dict], val_samples: List[Dict], 
             state_scaler: Dict, input_scaler: Dict) -> Dict:
        """
        Train model on multi-step prediction data.
        """
        # Print training setup
        print(f"Starting multi-step trajectory training:")
        print(f"  Model: Hybrid ODE with hidden size {self.model.hidden_size}")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Using {'weighted' if self.use_weighted_loss else 'unweighted'} loss")
        if self.use_weighted_loss:
            print(f"  Time decay rate: {self.decay_rate}")
        
        # Create validation batches (don't need to recreate these)
        val_batches = create_batches(val_samples, self.batch_size, shuffle=False)
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = self.train_state.params
        
        # Training loop
        for epoch in range(self.epochs):
            # Create fresh training batches each epoch (with shuffling)
            train_batches = create_batches(train_samples, self.batch_size, shuffle=True)
            
            # Train on batches
            epoch_losses = []
            epoch_grad_norms = []
            
            for batch_idx, batch in enumerate(tqdm(train_batches, desc=f"Epoch {epoch+1}/{self.epochs}")):
                # Convert numpy batch to jax arrays
                jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                
                # Training step
                effective_decay = self.decay_rate if self.use_weighted_loss else 0.0
                self.train_state, metrics = train_step(
                    self.model, self.train_state, jax_batch,
                    state_scaler, input_scaler, effective_decay
                )
                
                # Log metrics
                epoch_losses.append(float(metrics['loss']))
                epoch_grad_norms.append(float(metrics['grad_norm']))
                
                # Log batch metrics to wandb
                if self.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'batch_loss': float(metrics['loss']),
                        'batch_grad_norm': float(metrics['grad_norm']),
                        'epoch': epoch,
                        'batch': batch_idx,
                    })
            
            # Compute average epoch metrics
            avg_train_loss = np.mean(epoch_losses)
            avg_grad_norm = np.mean(epoch_grad_norms)
            
            # Log epoch training metrics
            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss = {avg_train_loss:.6f}, Grad Norm = {avg_grad_norm:.6f}")
            
            # Periodic validation
            if epoch % self.val_interval == 0 or epoch == self.epochs - 1:
                # Evaluate on validation data
                val_metrics = evaluate(
                    self.model, self.train_state.params, val_batches,
                    state_scaler, input_scaler
                )
                val_loss = float(val_metrics['val_loss'])
                
                print(f"  Validation Loss = {val_loss:.6f}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'grad_norm': avg_grad_norm,
                    })
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = self.train_state.params
                    patience_counter = 0
                    
                    # Save best model
                    self.save_model(best_params, state_scaler, input_scaler, suffix='best')
                else:
                    patience_counter += 1
                
                # Check for early stopping
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # REMOVED: Visualization code that was causing errors
            # No longer calling visualize_predictions during training
    
        # Final evaluation with best params
        self.train_state = self.train_state.replace(params=best_params)
        final_val_metrics = evaluate(
            self.model, best_params, val_batches,
            state_scaler, input_scaler
        )
        
        # Save final model
        self.save_model(best_params, state_scaler, input_scaler, suffix='final')
        
        # REMOVED: Final visualization
        
        # Training results
        results = {
            'final_train_loss': avg_train_loss,
            'final_val_loss': float(final_val_metrics['val_loss']),
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'best_params': best_params
        }
        
        # Log final results
        print("\nTraining complete!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Final validation loss: {float(final_val_metrics['val_loss']):.6f}")
        print(f"  Total epochs: {epoch+1}")
        
        if self.use_wandb:
            wandb.log({
                'final_train_loss': avg_train_loss,
                'final_val_loss': float(final_val_metrics['val_loss']),
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            })
            wandb.finish()
            
        return results
    
    def save_model(self, params: Dict, state_scaler: Dict, input_scaler: Dict, suffix: str = ''):
        """Save model parameters and metadata."""
        save_dict = {
            'params': params,
            'state_scaler': state_scaler,
            'input_scaler': input_scaler,
            'config': self.config,
        }
        
        if suffix:
            save_path = self.output_dir / f"model_{suffix}.pkl"
        else:
            save_path = self.output_dir / "model.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to {save_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Hybrid Neural ODE Multi-Step Training Pipeline")
    print("=" * 60)
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    np.random.seed(config['random_seed'])
    key = jax.random.PRNGKey(config['random_seed'])
    
    # Load and process data (still load test samples for API compatibility, but don't use them)
    print("Loading multi-step samples...")
    try:
        train_samples, val_samples, test_samples, (state_scaler, input_scaler) = load_multistep_samples()
        print(f"Loaded {len(train_samples)} training samples")
        print(f"Loaded {len(val_samples)} validation samples")
    except FileNotFoundError:
        print("Pre-processed data not found. Processing raw data...")
        train_samples, val_samples, test_samples, norm_params = process_data("config.yaml")
        state_scaler = {'mean': norm_params['state_mean'], 'std': norm_params['state_std']}
        input_scaler = {'mean': norm_params['input_mean'], 'std': norm_params['input_std']}
    
    # Initialize trainer
    trainer = HybridODETrainer(config, use_wandb=True)
    
    # Train model
    results = trainer.train(train_samples, val_samples, state_scaler, input_scaler)
    
    # Save training results
    with open(Path(config['data']['output_dir']) / "training_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\nTraining complete!")
    print(f"Results saved to {config['data']['output_dir']}/training_results.pkl")
    print(f"Model saved to {config['data']['output_dir']}/model_best.pkl")


if __name__ == "__main__":
    main()
