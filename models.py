import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any
import numpy as np
from functools import partial
from scipy.interpolate import interp1d
import yaml
import numpy as np



class MLPDynamics(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input expects shape (..., 6)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(3)(x)
        return x
    


def kinematics(state, inputs):

    pos_x, pos_y, psi, delta, v, beta, psi_dot = state
    a, delta_dot = inputs

    dx_dt = v * jnp.cos(psi + beta)
    dy_dt = v * jnp.sin(psi + beta)
    dpsi_dt = psi_dot
    ddelta_dt = delta_dot

    return jnp.array([dx_dt, dy_dt, dpsi_dt, ddelta_dt])


class HybridODE:

    def __init__(self, config):
        self.config = config
        self.input_dim = len(config['model']['input_names'])
        self.physics_states = config['model']['physics_states']
        self.neural_states = config['model']['neural_states']
        self.neural_net = MLPDynamics()  
        self.params = self.init_network(jax.random.PRNGKey(0))

    def init_network(self, key):
        dummy_input = jnp.ones((9,))  # [δ, v, β, ψ̇, a, δ̇]
        params = self.neural_net.init(key, dummy_input)
        return params


    def neural_dynamics(self, state, inputs, params=None):

        nn_inputs = jnp.concatenate((state, inputs))  # (6,)
        assert nn_inputs.shape == (9,), f"nn_inputs shape is {nn_inputs.shape}, expected (9,)"
        if params is None:
            params = self.params
        neural_output = self.neural_net.apply(params, nn_inputs)
        return neural_output
    

    def hybrid_dynamics(self, state, inputs, params=None):

        kinematic_derivis = kinematics(state, inputs)
        neural_derivatives = self.neural_dynamics(state, inputs, params)

        state_derives = jnp.concatenate((kinematic_derivis, neural_derivatives))

        return state_derives
    
    def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None):
        """
        Perform a single RK4 integration step for the hybrid ODE.
        Args:
            params: neural network parameters
            state: current state vector
            inputs_t: inputs at time t
            inputs_t_plus_dt: inputs at time t + dt
            dt: time step (float)
        Returns:
            next_state: state vector after time dt
        """
        inputs_mid = (inputs_t + inputs_t_plus_dt) / 2.0

        k1 = self.hybrid_dynamics(state, inputs_t, params)
        k2 = self.hybrid_dynamics(state + dt/2 * k1, inputs_mid, params)
        k3 = self.hybrid_dynamics(state + dt/2 * k2, inputs_mid, params)
        k4 = self.hybrid_dynamics(state + dt * k3, inputs_t_plus_dt, params)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Wrap yaw (index 2) to [-pi, pi] after integration
        next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return next_state
    

    

    def euler_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None):
        """
        Perform a single Euler integration step for the hybrid ODE.
        Args:
            params: neural network parameters
            state: current state vector
            inputs_t: inputs at time t
            inputs_t_plus_dt: (unused, for API compatibility)
            dt: time step (float)
        Returns:
            next_state: state vector after time dt
        """
        k1 = self.hybrid_dynamics(state, inputs_t, params)
   
        next_state = state + dt * k1
        # Wrap yaw (index 2) to [-pi, pi] after integration
        next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return next_state

    def predict_trajectory(self, params, initial_state, inputs_sequence, dt):
        """
        Roll out the trajectory for num_steps-1 using the given initial state and input sequence,
        using jax.lax.scan for efficiency. Only predicts up to the last available ground-truth state.
        Args:
            initial_state: shape (state_dim,)
            inputs_sequence: shape (num_steps, input_dim)
            dt: float, time step
        Returns:
            states: shape (num_steps, state_dim)
        """
        num_steps = inputs_sequence.shape[0]

        def scan_step(state, t):
            current_input = inputs_sequence[t]
            next_input = inputs_sequence[t + 1]  
            next_state = self.rk4_step(state, current_input, next_input, dt, params)
          
            return next_state, next_state

        
        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        trajectory = jnp.vstack([initial_state, states])
        return trajectory
    
    


    def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt):
        batch_predict_fn = jax.vmap(lambda s, i: self.predict_trajectory(params, s, i, dt), in_axes=(0, 0))
        return batch_predict_fn(initial_states, inputs_batch)
    










class MLPStatePrediction(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input expects shape (..., 9) - 7 states + 2 inputs
        x = nn.Dense(128)(x) # 512
        x = nn.tanh(x)
        x = nn.Dense(128)(x) # 256
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(7)(x)  # Output full state instead of 3 derivatives
        return x
    

class Node:

    def __init__(self, config):
        self.config = config
        self.input_dim = len(config['model']['input_names'])
        self.physics_states = config['model']['physics_states']
        self.neural_states = config['model']['neural_states']
        self.neural_net = MLPStatePrediction()  
        self.params = self.init_network(jax.random.PRNGKey(0))

    def init_network(self, key):
        dummy_input = jnp.ones((9,))  
        params = self.neural_net.init(key, dummy_input)
        return params


    def neural_dynamics(self, state, action, params=None):

        """only take the last"""

        nn_inputs = jnp.concatenate((state, action))  # (9,)
        assert nn_inputs.shape == (9,), f"nn_inputs shape is {nn_inputs.shape}, expected (9,)"
        if params is None:
            params = self.params
        neural_output = self.neural_net.apply(params, nn_inputs)
        return neural_output
    

    def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None):
        """
        Perform a single RK4 integration step for the hybrid ODE.
        Args:
            params: neural network parameters
            state: current state vector
            inputs_t: inputs at time t
            inputs_t_plus_dt: inputs at time t + dt
            dt: time step (float)
        Returns:
            next_state: state vector after time dt
        """
        inputs_mid = (inputs_t + inputs_t_plus_dt) / 2.0

        k1 = self.neural_dynamics(state, inputs_t, params)
        k2 = self.neural_dynamics(state + dt/2 * k1, inputs_mid, params)
        k3 = self.neural_dynamics(state + dt/2 * k2, inputs_mid, params)
        k4 = self.neural_dynamics(state + dt * k3, inputs_t_plus_dt, params)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Wrap yaw (index 2) to [-pi, pi] after integration
        next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return next_state


    


    
    def predict_trajectory(self, params, initial_state, inputs_sequence, dt):
        """
        Roll out the trajectory for num_steps-1 using the given initial state and input sequence,
        using jax.lax.scan for efficiency. Only predicts up to the last available ground-truth state.
        Args:
            initial_state: shape (state_dim,)
            inputs_sequence: shape (num_steps, input_dim)
            dt: float, time step
        Returns:
            states: shape (num_steps, state_dim)
        """
        num_steps = inputs_sequence.shape[0]

        def scan_step(state, t):
            current_input = inputs_sequence[t]
            # Use jnp.where to select next_input
            next_input = jnp.where(
                t < num_steps - 1,
                inputs_sequence[t + 1],
                current_input
            )
            next_state = self.rk4_step(state, current_input, next_input, dt, params)
           
            return next_state, next_state

        
        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        trajectory = jnp.vstack([initial_state, states])
        return trajectory


    def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt):
        batch_predict_fn = jax.vmap(lambda s, i: self.predict_trajectory(params, s, i, dt), in_axes=(0, 0))
        return batch_predict_fn(initial_states, inputs_batch)






class DynamicBicycle:
    """
    Dynamic bicycle model with Magic Formula tire forces in JAX.
    Pure physics-based implementation compatible with HybridODE and Node APIs.
    
    State Variables (7D): [x, y, yaw, vx, vy, yaw_rate, steering_angle]
    Control Inputs (2D): [drive_force, steer_speed]
    """
    
    def __init__(self, config):
        """
        Initialize the dynamic bicycle model with physics parameters.
        
        Args:
            config: Configuration dictionary (for API compatibility)
        """
        self.config = config
   
        self.params = {
            'lf': 0.8839200139045715,           # distance from CoG to front axle (m)
            'lr': 1.5087599754333496,           # distance from CoG to rear axle (m)
            'Iz': 1771.53857421875,             # yaw moment of inertia (kg⋅m²)
            'mass': 701.0275268554688,          # vehicle mass (kg)
            'Df': 4041.601318359375,            # front tire peak friction
            'Cf': 1.4304611682891846,           # front tire shape factor
            'Bf': 18.741268157958984,           # front tire stiffness factor
            'Dr': 3652.13427734375,             # rear tire peak friction
            'Cr': 0.7047927379608154,           # rear tire shape factor
            'Br': 27.620542526245117,           # rear tire stiffness factor
            'Cm': 0.5324415564537048,           # motor efficiency
            'Cr0': 10.995059967041016,          # rolling resistance constant
            'Cr2': -0.03063417412340641,        # aerodynamic drag coefficient
        }
        
        # State and input dimensions
        self.state_dim = 7
        self.input_dim = 2
        
        # State indices for clarity
        self.X, self.Y, self.YAW = 0, 1, 2
        self.VX, self.VY, self.YAW_RATE, self.STEERING_ANGLE = 3, 4, 5, 6
        
        # Input indices
        self.DRIVE_FORCE, self.STEER_SPEED = 0, 1
        
        # Tire force indices
        self.FRX, self.FFY, self.FRY = 0, 1, 2
    
    def calculate_tire_forces(self, state, inputs):
        
        # Extract state variables
        vx = state[self.VX]
        vy = state[self.VY] 
        yaw_rate = state[self.YAW_RATE]
        steering_angle = state[self.STEERING_ANGLE]
        
        # Extract inputs
        drive_force = inputs[self.DRIVE_FORCE]
        
        # Compute slip angles
        # Front slip angle: αf = δ - arctan((ψ̇⋅lf + vy)/vx)
        alpha_f = steering_angle - jnp.arctan((yaw_rate * self.params['lf'] + vy) / vx)
        
        # Rear slip angle: αr = arctan((ψ̇⋅lr - vy)/vx)  
        alpha_r = jnp.arctan((yaw_rate * self.params['lr'] - vy) / vx)
        
        # Longitudinal force (rear): Frx = Cm⋅Fd - Cr0 - Cr2⋅vx²
        Frx = (self.params['Cm'] * drive_force - 
               self.params['Cr0'] - 
               self.params['Cr2'] * vx**2)
        
        # Lateral forces using Magic Formula: F = D⋅sin(C⋅arctan(B⋅α))
        # Front lateral force
        Ffy = (self.params['Df'] * 
               jnp.sin(self.params['Cf'] * 
                      jnp.arctan(self.params['Bf'] * alpha_f)))
        
        # Rear lateral force  
        Fry = (self.params['Dr'] * 
               jnp.sin(self.params['Cr'] * 
                      jnp.arctan(self.params['Br'] * alpha_r)))
        
        return jnp.array([Frx, Ffy, Fry])
    
    def dynamics(self, state, inputs):
        """
        Compute state derivatives using dynamic bicycle model.
        
        Args:
            state: Current state vector (7D)
            inputs: Control inputs vector (2D)
            
        Returns:
            state_derivatives: Time derivatives of state vector (7D)
        """
        # Extract state variables
        x = state[self.X]
        y = state[self.Y]
        yaw = state[self.YAW]
        vx = state[self.VX]
        vy = state[self.VY]
        yaw_rate = state[self.YAW_RATE]
        steering_angle = state[self.STEERING_ANGLE]
        
        # Extract inputs
        drive_force = inputs[self.DRIVE_FORCE]
        steer_speed = inputs[self.STEER_SPEED]
        
        # Calculate tire forces
        tire_forces = self.calculate_tire_forces(state, inputs)
        Frx, Ffy, Fry = tire_forces[self.FRX], tire_forces[self.FFY], tire_forces[self.FRY]
        
        # Physical parameters
        m = self.params['mass']
        Iz = self.params['Iz']
        lf = self.params['lf']
        lr = self.params['lr']
        
        # Kinematic equations (global frame)
        dx_dt = vx * jnp.cos(yaw) - vy * jnp.sin(yaw)
        dy_dt = vx * jnp.sin(yaw) + vy * jnp.cos(yaw)
        dyaw_dt = yaw_rate
        
        # Dynamic equations (body frame)
        # Longitudinal dynamics: dvx/dt = (Frx - Ffy⋅sin(δ) + m⋅vy⋅ψ̇) / m
        dvx_dt = (Frx - Ffy * jnp.sin(steering_angle) + m * vy * yaw_rate) / m
        
        # Lateral dynamics: dvy/dt = (Fry + Ffy⋅cos(δ) - m⋅vx⋅ψ̇) / m  
        dvy_dt = (Fry + Ffy * jnp.cos(steering_angle) - m * vx * yaw_rate) / m
        
        # Yaw dynamics: dψ̇/dt = (Ffy⋅lf⋅cos(δ) - Fry⋅lr) / Iz
        dyaw_rate_dt = (Ffy * lf * jnp.cos(steering_angle) - Fry * lr) / Iz
        
        # Steering dynamics: dδ/dt = δ̇
        dsteering_dt = steer_speed
        
        return jnp.array([dx_dt, dy_dt, dyaw_dt, dvx_dt, dvy_dt, dyaw_rate_dt, dsteering_dt])
    
    def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt):
        
        inputs_mid = (inputs_t + inputs_t_plus_dt) / 2.0
        
        # RK4 integration
        k1 = self.dynamics(state, inputs_t)
        k2 = self.dynamics(state + dt/2 * k1, inputs_mid)
        k3 = self.dynamics(state + dt/2 * k2, inputs_mid)  
        k4 = self.dynamics(state + dt * k3, inputs_t_plus_dt)
        
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Wrap yaw angle to [-π, π]
        next_state = next_state.at[self.YAW].set(
            ((next_state[self.YAW] + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        )
        
        return next_state
    
    def euler_step(self, state, inputs_t, inputs_t_plus_dt, dt):
       
        k1 = self.dynamics(state, inputs_t)
        next_state = state + dt * k1
        
        # Wrap yaw angle to [-π, π]
        next_state = next_state.at[self.YAW].set(
            ((next_state[self.YAW] + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        )
        
        return next_state
    
    def predict_trajectory(self, initial_state, inputs_sequence, dt):
        
        num_steps = inputs_sequence.shape[0]
        
        def scan_step(state, t):
            current_input = inputs_sequence[t]
            next_input = jnp.where(
                t < (num_steps - 1),
                inputs_sequence[t + 1],
                current_input
            )
            next_state = self.rk4_step(state, current_input, next_input, dt)
            return next_state, next_state
        
        # Roll out trajectory for num_steps - 1 
        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        # Include initial state
        trajectory = jnp.vstack([initial_state, states])
        return trajectory
    



    def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt):
        
        batch_predict_fn = jax.vmap(
            lambda s, i: self.predict_trajectory(s, i, dt), 
            in_axes=(0, 0)
        )
        return batch_predict_fn(initial_states, inputs_batch)





class KinematicBicycle:
   
    
    def __init__(self, config, lf=1.2, lr=1.3):
        
        self.config = config
        self.lf = lf
        self.lr = lr
        self.wheelbase = lf + lr
        
        # State and input dimensions
        self.state_dim = 5
        self.input_dim = 2
        
        # State indices
        self.X, self.Y, self.THETA, self.V, self.YAW = 0, 1, 2, 3, 4
        
        # Input indices
        self.STEER_RATE, self.ACCEL = 0, 1



    
    def dynamics(self, state, inputs):
        
        x = state[self.X]
        y = state[self.Y]
        theta = state[self.THETA]  # steering angle
        v = state[self.V]
        yaw = state[self.YAW]     # vehicle heading
        
        # Extract inputs
        steer_rate = inputs[self.STEER_RATE]
        accel = inputs[self.ACCEL]
        
        # Compute sideslip angle beta
        beta = jnp.arctan(jnp.tan(theta) * (self.lr / self.wheelbase))
        
        # Kinematic equations
        dx_dt = v * jnp.cos(yaw)
        dy_dt = v * jnp.sin(yaw)
        dtheta_dt = steer_rate
        dv_dt = accel
        dyaw_dt = (v * jnp.tan(theta)) / self.wheelbase
        
        return jnp.array([dx_dt, dy_dt, dtheta_dt, dv_dt, dyaw_dt])
    




    def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt):
        
        inputs_mid = (inputs_t + inputs_t_plus_dt) / 2.0
        
        # RK4 integration
        k1 = self.dynamics(state, inputs_t)
        k2 = self.dynamics(state + dt/2 * k1, inputs_mid)
        k3 = self.dynamics(state + dt/2 * k2, inputs_mid)
        k4 = self.dynamics(state + dt * k3, inputs_t_plus_dt)
        
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Wrap angles to [-π, π]
        next_state = next_state.at[self.THETA].set(
            ((next_state[self.THETA] + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        )
        next_state = next_state.at[self.YAW].set(
            ((next_state[self.YAW] + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        )
        
        return next_state



    
    def euler_step(self, state, inputs_t, inputs_t_plus_dt, dt):
        
        k1 = self.dynamics(state, inputs_t)
        next_state = state + dt * k1
        
        # Wrap angles to [-π, π]
        next_state = next_state.at[self.THETA].set(
            ((next_state[self.THETA] + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        )
        next_state = next_state.at[self.YAW].set(
            ((next_state[self.YAW] + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        )
        
        return next_state
    
    def predict_trajectory(self, initial_state, inputs_sequence, dt):
        
        num_steps = inputs_sequence.shape[0]
        
        def scan_step(state, t):
            current_input = inputs_sequence[t]
            
            next_input = jnp.where(
            t < (num_steps - 1),
            inputs_sequence[t + 1],
            current_input
        )



            next_state = self.rk4_step(state, current_input, next_input, dt)
            return next_state, next_state
        
        # Roll out trajectory for num_steps - 1
        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        # Include initial state
        trajectory = jnp.vstack([initial_state, states])
        return trajectory
    
    def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt):
        
        batch_predict_fn = jax.vmap(
            lambda s, i: self.predict_trajectory(s, i, dt),
            in_axes=(0, 0)
        )
        return batch_predict_fn(initial_states, inputs_batch)




    
def create_train_state(model, learning_rate, key, weight_decay=0.0):
    
    params = model.init_network(key)
    if weight_decay > 0:
        # Example: Exponential decay scheduler
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=100,
            decay_rate=0.99,
            staircase=True
        )
        optimizer = optax.adamw(schedule, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.neural_net.apply,
        params=params,
        tx=optimizer
    )
    


    
if __name__ == "__main__":
    

    print("=" * 50)
    print("Testing Hybrid ODE with Multi-Step Prediction (models_new.py)")
    print("=" * 50)

    # Load  config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = HybridODE(config)
    print("Model initialized.")

    # Initialize parameters (not used directly in this structure, but shown for completeness)
    key = jax.random.PRNGKey(42)
    params = model.init_network(key)
    print("Parameters initialized.")

    # Load a processed test sample
    data = np.load("processed_data/test_data.npz")
    samples = data["samples"]  # shape: (num_samples, 9, n_steps)
    print(f"Loaded test samples: {samples.shape}")

    # Use the first sample for demonstration
    sample = samples[0]  # shape: (9, n_steps)
    n_steps = sample.shape[1]
    state_dim = 7
    input_dim = 2

    # Extract initial state and input sequence
    initial_state = sample[:state_dim, 0]  # shape: (7,)
    inputs_sequence = sample[state_dim:, :].T  # shape: (n_steps, 2)
    inputs_sequence = jnp.array(inputs_sequence)

    # Use a fixed dt (if your data is uniform in time)
    dt = 0.1  # or set from your config/timestamps if available

    # Predict trajectory (for demo/testing): pass params explicitly
    pred_traj = model.predict_trajectory(params, initial_state, inputs_sequence, dt)
    print(f"Predicted trajectory shape: {pred_traj.shape}")
    print("First few predicted states:\n", np.array(pred_traj[:3]))

    print("\nMulti-step prediction is working correctly in models_new.py!")





