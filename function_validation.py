import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Generator
import numpy as np
import pandas as pd

def kinematic_model(state, inputs):
    """
    Implements the single-track ODE model with small angle approximation.
    Args:
        state: jnp.ndarray, shape (7,) [Δx, Δy, ψ, δ, v, β, ω]
        inputs: jnp.ndarray, shape (2,) [a_x, δ_dot]
        params: dict with vehicle parameters
    Returns:
        dxdt: jnp.ndarray, shape (7,)
    """
    # Unpack state
    dx, dy, psi, delta, v, beta, omega = state
    a_x, delta_dot = inputs



    params = {
    'm': 3.0,
    'I_z': 0.05,
    'l_f': 0.13,
    'l_r': 0.13,
    'mu': 1.0,
    'C_f': 15.0,
    'C_r': 15.0,
    'g': 9.81,
    }

    # Unpack parameters
    m = params['m']
    I_z = params['I_z']
    l_f = params['l_f']
    l_r = params['l_r']
    mu = params['mu']
    C_f = params['C_f']
    C_r = params['C_r']
    g = params['g']

    # Avoid division by zero
    v_safe = jnp.where(jnp.abs(v) < 1e-3, 1e-3, v)

    # Tire forces (small angle approx)
    F_fy = mu * C_f * m * g * l_r / (l_r + l_f) * (delta - omega * l_f / v_safe - beta)
    F_ry = mu * C_r * m * g * l_f / (l_r + l_f) * (omega * l_r / v_safe - beta)

    # ODEs
    dx_dt = v  *np.coss(psi + beta) ≈ 1
    dy_dt = v * (psi + beta)  # sin(psi + beta) ≈ psi + beta
    dpsi_dt = omega
    ddelta_dt = delta_dot
    dv_dt = a_x
    dbeta_dt = (1/(m * v_safe)) * (F_fy + F_ry) - omega
    domega_dt = (1/I_z) * (F_fy * l_f - F_ry * l_r)

    return jnp.array([dx_dt, dy_dt, dpsi_dt, ddelta_dt, dv_dt, dbeta_dt, domega_dt])


def euler_step(state, inputs_t, dt):
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
        k1 = kinematic_model(state, inputs_t)
        next_state = state + dt * k1
        # Wrap yaw (index 2) to [-pi, pi] after integration
        next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return next_state


def predict_trajectory(initial_state, inputs_sequence, dt):
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
            # next_input = inputs_sequence[t + 1]  # Not needed for Euler
            next_state = euler_step(state, current_input, dt)
            return next_state, next_state

        
        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        trajectory = jnp.vstack([initial_state, states])
        return trajectory

class KinematicODEWrapper:
    """Wraps the kinematic_model to match the expected API for euler_step."""
    def __init__(self, params):
        self.params = params

    def hybrid_dynamics(self, state, inputs):
        # Ignore params argument, use self.params
        return kinematic_model(state, inputs)

# --- Main block ---
if __name__ == "__main__":
    data = np.load("processed_data/test_data.npz")
    samples = data["samples"]
    samples = jnp.array(samples[:20], dtype=jnp.float32)
    print(samples.shape)

    # Parameters for the model
    params = {
        'm': 3.0,
        'I_z': 0.05,
        'l_f': 0.13,
        'l_r': 0.13,
        'mu': 1.0,
        'C_f': 15.0,
        'C_r': 15.0,
        'g': 9.81,
    }
    dt = 0.01666666753590107  # from config.yaml

    all_rows = []
    for sample_idx, sample in enumerate(samples):
        state_dim = 7
        n_steps = sample.shape[1]
        initial_state = sample[:state_dim, 0]
        inputs_sequence = sample[state_dim:, :].T  # (n_steps, 2)
        true_states = sample[:state_dim, :].T      # (n_steps, 7)

        # Predict trajectory
        pred_states = [initial_state]
        for u in inputs_sequence[:-1]:
            s = pred_states[-1]
            dsdt = kinematic_model(s, u)

        predicted_trajectory = predict_trajectory(initial_state, inputs_sequence, dt)

        for t in range(n_steps):
            for d in range(state_dim):
                all_rows.append({
                    "sample_idx": sample_idx,
                    "timestep": t,
                    "state_idx": d,
                    "true_state": float(true_states[t, d]),
                    "pred_state": float(predicted_trajectory[t, d])
                })

    df = pd.DataFrame(all_rows)
    df.to_csv("kinematic_vs_true_trajectories.csv", index=False)
    print("Saved results to kinematic_vs_true_trajectories.csv")






