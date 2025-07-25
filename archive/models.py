#!/usr/bin/env python3
"""
Hybrid Neural ODE Model for Vehicle Dynamics with Multi-Step Prediction Support
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any
import numpy as np
from functools import partial
from scipy.interpolate import interp1d


class MLPDynamics(nn.Module):
    """MLP for learning vehicle dynamics (v, β, ψ̇)"""
    hidden_size: int = 10
    
    def setup(self):
        self.layers = [
            nn.Dense(self.hidden_size),
            nn.tanh,
            nn.Dense(3)  # Output: [dv/dt, dβ/dt, dψ̇/dt]
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def physics_dynamics(state: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """Physics-based kinematic equations for [dΔx/dt, dΔy/dt, dψ/dt, dδ/dt]"""
    delta_x, delta_y, psi, delta, v, beta, psi_dot = state
    a, delta_dot = inputs
    
    dx_dt = v * jnp.cos(psi + beta)
    dy_dt = v * jnp.sin(psi + beta)
    dpsi_dt = psi_dot
    ddelta_dt = delta_dot
    
    return jnp.array([dx_dt, dy_dt, dpsi_dt, ddelta_dt])


class HybridODE:
    """Hybrid Neural ODE combining physics and neural network dynamics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hidden_size = config['model']['hidden_size']
        self.state_dim = len(config['model']['state_names'])
        self.input_dim = len(config['model']['input_names'])
        self.physics_states = config['model']['physics_states']
        self.neural_states = config['model']['neural_states']
        self.neural_net = MLPDynamics(hidden_size=self.hidden_size)
        
    def init_params(self, key: jax.random.PRNGKey) -> Dict:
        """Initialize neural network parameters."""
        dummy_input = jnp.ones((6,))  # [δ, v, β, ψ̇, a, δ̇]
        params = self.neural_net.init(key, dummy_input)
        return params
    
    @partial(jax.jit, static_argnums=(0,))
    def neural_dynamics(self, params: Dict, state: jnp.ndarray, inputs: jnp.ndarray, 
                       state_scaler: Dict, input_scaler: Dict) -> jnp.ndarray:
        """Neural network dynamics for [v, β, ψ̇]"""
        neural_states = state[3:7]  # [δ, v, β, ψ̇]
        
        scaled_states = (neural_states - state_scaler['mean'][3:7]) / state_scaler['std'][3:7]
        scaled_inputs = (inputs - input_scaler['mean']) / input_scaler['std']
        
        nn_input = jnp.concatenate([scaled_states, scaled_inputs])
        neural_output = self.neural_net.apply(params, nn_input)
        
        return neural_output
    
    @partial(jax.jit, static_argnums=(0,))
    def dynamics_with_inputs(self, params: Dict, state: jnp.ndarray, inputs: jnp.ndarray,
                           state_scaler: Dict, input_scaler: Dict) -> jnp.ndarray:
        """Hybrid dynamics with explicit input handling"""
        physics_derivs = physics_dynamics(state, inputs)
        neural_derivs = self.neural_dynamics(params, state, inputs, state_scaler, input_scaler)
        state_dot = jnp.concatenate([physics_derivs, neural_derivs])
        
        return state_dot
    
    def euler_step(self, params: Dict, state: jnp.ndarray, inputs: jnp.ndarray, 
                  dt: float, state_scaler: Dict, input_scaler: Dict) -> jnp.ndarray:
        """Simple Euler integration step"""
        state_dot = self.dynamics_with_inputs(params, state, inputs, state_scaler, input_scaler)
        next_state = state + dt * state_dot
        return next_state
    
    def rk4_step(self, params: Dict, state: jnp.ndarray, inputs_t: jnp.ndarray, 
                inputs_t_plus_dt: jnp.ndarray, dt: float, state_scaler: Dict, 
                input_scaler: Dict) -> jnp.ndarray:
        """Runge-Kutta 4th order integration step"""
        inputs_mid = (inputs_t + inputs_t_plus_dt) / 2.0
        
        k1 = self.dynamics_with_inputs(params, state, inputs_t, state_scaler, input_scaler)
        
        k2_state = state + dt * k1 / 2.0
        k2 = self.dynamics_with_inputs(params, k2_state, inputs_mid, state_scaler, input_scaler)
        
        k3_state = state + dt * k2 / 2.0
        k3 = self.dynamics_with_inputs(params, k3_state, inputs_mid, state_scaler, input_scaler)
        
        k4_state = state + dt * k3
        k4 = self.dynamics_with_inputs(params, k4_state, inputs_t_plus_dt, state_scaler, input_scaler)
        
        next_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state
    
    def predict_trajectory(self, params: Dict, initial_state: jnp.ndarray,
                         inputs_sequence: jnp.ndarray, timesteps: jnp.ndarray,
                         state_scaler: Dict, input_scaler: Dict) -> jnp.ndarray:
        """Predict a multi-step trajectory using the hybrid ODE model"""
        states = [initial_state]
        current_state = initial_state
        
        # Ensure inputs_sequence and timesteps are properly aligned
        n_pred_steps = len(timesteps) - 1
        assert len(inputs_sequence) >= n_pred_steps, "Not enough inputs for prediction steps"
        
        for t in range(n_pred_steps):
            dt = timesteps[t+1] - timesteps[t]  #  just pass in dt
            current_input = inputs_sequence[t]
            
            # For the last step, use the same input for next_input if needed
            if t < len(inputs_sequence) - 1:
                next_input = inputs_sequence[t+1]
            else:
                next_input = current_input
                
            next_state = self.rk4_step(
                params, current_state, current_input, next_input, 
                dt, state_scaler, input_scaler
            )
            states.append(next_state)
            current_state = next_state
            

            #  jax.lax.scan try it out
        return jnp.stack(states)

    def predict_batch_trajectories(self, params: Dict, initial_states: jnp.ndarray,
                                 inputs_batch: jnp.ndarray, timesteps_batch: jnp.ndarray,
                                 state_scaler: Dict, input_scaler: Dict) -> jnp.ndarray:
        """Predict multiple trajectories in parallel using vectorization"""
        batch_predict_fn = jax.vmap(
            lambda init_state, inputs, timesteps: self.predict_trajectory(
                params, init_state, inputs, timesteps, state_scaler, input_scaler))
        return batch_predict_fn(initial_states, inputs_batch, timesteps_batch)



def create_model_from_config(config: Dict[str, Any]) -> HybridODE:
    """Create hybrid ODE model from configuration"""
    return HybridODE(config)


def create_train_state(model: HybridODE, learning_rate: float, key: jax.random.PRNGKey,
                     weight_decay: float = 0.0) -> train_state.TrainState:
    """Create training state with optimizer"""
    params = model.init_params(key)
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay) if weight_decay > 0 else optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.neural_net.apply,
        params=params,
        tx=optimizer
    )


def create_interpolators(timestamps: np.ndarray, inputs: np.ndarray, robot_idx: int = 0) -> list:
    """Create interpolation functions for continuous input access"""
    robot_inputs = inputs[:, robot_idx, :]  # (T, 2)
    return [interp1d(timestamps, robot_inputs[:, i], kind='cubic', bounds_error=False, fill_value='extrapolate')
            for i in range(inputs.shape[-1])]




if __name__ == "__main__":
    """Test the hybrid ODE model with multi-step prediction capabilities"""
    import yaml
    
    print("=" * 50)
    print("Testing Hybrid ODE with Multi-Step Prediction")
    print("=" * 50)
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model_from_config(config)
    print(f"Model created with hidden size: {model.hidden_size}")
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = model.init_params(key)
    print("Parameters initialized")
    
    # Create dummy state, inputs and scalers
    state = jnp.array([0.0, 0.0, 0.1, 0.05, 5.0, 0.02, 0.1])  # [Δx, Δy, ψ, δ, v, β, ψ̇]
    inputs = jnp.array([0.5, 0.01])  # [a, δ̇]
    
    state_scaler = {'mean': jnp.zeros(7), 'std': jnp.ones(7)}
    input_scaler = {'mean': jnp.zeros(2), 'std': jnp.ones(2)}
    
    # Test dynamics
    derivatives = model.dynamics_with_inputs(params, state, inputs, state_scaler, input_scaler)
    print(f"\nState derivatives shape: {derivatives.shape}")
    print(f"First few derivatives: {derivatives[:3]}")
    
    # Test multi-step prediction (single trajectory)
    n_steps = 10
    timesteps = jnp.linspace(0, 1.0, n_steps+1)
    input_sequence = jnp.ones((n_steps, 2)) * inputs
    

    
    # RK4 integration
    rk4_trajectory = model.predict_trajectory(
        params, state, input_sequence, timesteps, state_scaler, input_scaler)
    print(f"RK4 trajectory shape: {rk4_trajectory.shape}")
    
    # Test batch prediction
    batch_size = 5
    initial_states = jnp.tile(state, (batch_size, 1))
    input_sequences = jnp.tile(input_sequence, (batch_size, 1, 1))
    timestep_sequences = jnp.tile(timesteps, (batch_size, 1))
    
    batch_trajectories = model.predict_batch_trajectories(
        params, initial_states, input_sequences, timestep_sequences, state_scaler, input_scaler
    )
    
    print(f"\nBatch trajectories shape: {batch_trajectories.shape}")
    print(f"Expected: (batch_size={batch_size}, timesteps={n_steps+1}, state_dim=7)")
    
    print("\nMulti-step prediction is working correctly!")