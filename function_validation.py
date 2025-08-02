#!/usr/bin/env python3
"""
Save all true, Euler, RK4, and solve_ivp (RK45) trajectories for a real dataset.
No command-line arguments required.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from scipy.integrate import solve_ivp

def kinematic_model(state: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    dx, dy, psi, delta, v, beta, omega = state
    a_x, delta_dot = inputs
    params = dict(
        m=3.0, I_z=0.05, l_f=0.13, l_r=0.13, mu=1.0,
        C_f=15.0, C_r=15.0, g=9.81
    )
    v_safe = jnp.where(jnp.abs(v) < 1e-3, 1e-3, v)
    F_fy = (
        params["mu"] * params["C_f"] * params["m"] * params["g"] * params["l_r"]
        / (params["l_r"] + params["l_f"])
        * (delta - omega * params["l_f"] / v_safe - beta)
    )
    F_ry = (
        params["mu"] * params["C_r"] * params["m"] * params["g"] * params["l_f"]
        / (params["l_r"] + params["l_f"])
        * (omega * params["l_r"] / v_safe - beta)
    )
    dx_dt = v * jnp.cos(psi + beta)
    dy_dt = v * jnp.sin(psi + beta)
    dpsi_dt = omega
    ddelta_dt = delta_dot
    dv_dt = a_x
    dbeta_dt = (F_fy + F_ry) / (params["m"] * v_safe) - omega
    domega_dt = (F_fy * params["l_f"] - F_ry * params["l_r"]) / params["I_z"]
    return jnp.array([dx_dt, dy_dt, dpsi_dt, ddelta_dt, dv_dt, dbeta_dt, domega_dt], dtype=state.dtype)

def kinematic_model_np(state: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(kinematic_model(jnp.asarray(state), jnp.asarray(inputs)))

def euler_step(state: jnp.ndarray, inp: jnp.ndarray, dt: float) -> jnp.ndarray:
    next_state = state + dt * kinematic_model(state, inp)
    next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
    return next_state

def rk4_step(state, inputs_t, inputs_t_plus_dt, dt):
   
    inputs_mid = (inputs_t + inputs_t_plus_dt) / 2.0
    k1 = kinematic_model(state, inputs_t)
    k2 = kinematic_model(state + dt/2 * k1, inputs_mid)
    k3 = kinematic_model(state + dt/2 * k2, inputs_mid)
    k4 = kinematic_model(state + dt * k3, inputs_t_plus_dt)

    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # Wrap yaw (index 2) to [-pi, pi] after integration
    next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
    return next_state

def rollout_euler(initial, controls, dt):
    T = controls.shape[0]
    states = [initial]
    for k in range(T - 1):
        states.append(euler_step(states[-1], controls[k], dt))
    return np.asarray(states, dtype=np.float32)

def rollout_rk4(initial, controls, dt):

    controls = jnp.array(controls) 
    
    num_steps = controls.shape[0]

    def scan_step(state, k):
        inputs_t = controls[k]
        inputs_t_plus_dt = controls[k + 1]
        next_state = rk4_step(state, inputs_t, inputs_t_plus_dt, dt)
        return next_state, next_state

   
    indices = jnp.arange(num_steps - 1)
    _, states = jax.lax.scan(scan_step, initial, indices)
    trajectory = jnp.vstack([initial, states])
    return np.asarray(trajectory)

def rollout_solve_ivp(initial, controls, dt):
    T = controls.shape[0]
    t_eval = np.arange(T) * dt
    def u_of_t(t):
        idx = min(int(np.floor(t / dt)), T - 1)
        return controls[idx]
    def f(t, y):
        return kinematic_model_np(y, u_of_t(t))
    sol = solve_ivp(f, (0.0, (T - 1) * dt), y0=initial, t_eval=t_eval, method="RK45")
    traj = sol.y.T
    traj[:, 2] = (traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return traj.astype(np.float32)

STATE_NAMES = ["dx", "dy", "psi", "delta", "v", "beta", "omega"]

def main():
    # Load Data
    data = np.load("processed_data/test_data.npz")
    samples = data["samples"]
    samples = samples[:10]
    all_rows = []

    for idx, sample in enumerate(samples):
        print(f"Processing sample {idx + 1}/{len(samples)}")
        true_states = sample[:7, :].T        # (timesteps, 7)
        controls = sample[7:, :].T           # (timesteps, 2)
        initial_state = true_states[0]
        n_steps = true_states.shape[0]
        dt = 0.01666666753590107

        # Rollouts
        euler_traj = rollout_euler(initial_state, controls, dt)
        rk4_traj = rollout_rk4(initial_state, controls, dt)
        ivp_traj = rollout_solve_ivp(initial_state, controls, dt)

        for t in range(n_steps):
            for k, name in enumerate(STATE_NAMES):
                all_rows.append(dict(
                    sample=idx,
                    t=t,
                    state=name,
                    true=true_states[t, k],
                    euler=euler_traj[t, k],
                    rk4=rk4_traj[t, k],
                    solve_ivp=ivp_traj[t, k],
                ))

    df = pd.DataFrame(all_rows)
    df.to_csv("all_methods_vs_true.csv", index=False)
    print("[✓] Saved results to all_methods_vs_true.csv")

if __name__ == "__main__":
    main()
