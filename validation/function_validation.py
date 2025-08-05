"""
synthetic_preprocess_test.py
Generate fake robot trajectories, run them through the original preprocessing
pipeline, and print sanity-check results.
"""

import numpy as np
import yaml, json, shutil
from pathlib import Path
from typing import Tuple
from preprocess_func import process_data
import matplotlib.pyplot as plt

# ---------- your original preprocessing code -----------
# (paste **unchanged** functions: load_config, load_npz_data, validate_data,
#  convert_to_relative_pos, split_data, compute_normalization_params,
#  apply_normalization, normalize_dataset, create_multistep_samples,
#  process_data )
# -------------------------------------------------------

# ------------------------------------------------------------------
# 1. Synthetic-data generator
# ------------------------------------------------------------------
def make_synthetic_dataset(num_robots: int = 12,
                           timesteps: int = 300,
                           dt: float = 0.05
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce simple bicycle-model trajectories plus noisy control inputs.
    Returns states, inputs, timestamps in the exact layout required by
    the preprocessing pipeline.
    """
    states  = np.zeros((timesteps, num_robots, 7), dtype=np.float32)
    inputs  = np.zeros((timesteps, num_robots, 2), dtype=np.float32)
    timevec = np.arange(timesteps, dtype=np.float32) * dt

    wheelbase = 2.0
    rng = np.random.default_rng(0)

    # robot-independent loop for clarity
    for r in range(num_robots):
        # random initial state
        x, y     = rng.uniform(-10, 10, size=2)
        yaw      = rng.uniform(-np.pi, np.pi)
        steer    = 0.0
        vel      = rng.uniform(1.0, 4.0)
        β, yawdot = 0.0, 0.0

        for t in range(timesteps):
            # store state
            states[t, r] = [x, y, yaw, steer, vel, β, yawdot]

            # random control
            acc  = rng.normal(0, 0.4)
            dδ   = rng.normal(0, 0.15)
            inputs[t, r] = [acc, dδ]

            # discrete bicycle update
            vel   = max(0.1, vel + acc*dt)
            steer = np.clip(steer + dδ*dt, -0.5, 0.5)
            yawdot= vel*np.tan(steer)/wheelbase
            yaw   = np.arctan2(np.sin(yaw + yawdot*dt),
                               np.cos(yaw + yawdot*dt))
            x    += vel*np.cos(yaw)*dt
            y    += vel*np.sin(yaw)*dt
            β     = 0.1*np.sin(steer)   # toy side-slip model

    return states, inputs, timevec

# ------------------------------------------------------------------
# 2. Build a fake raw-data folder of multiple NPZ files
# ------------------------------------------------------------------
def write_npz_files(states, inputs, timestamps,
                    outdir="synthetic_raw", robots_per_file=4):
    out = Path(outdir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    R = states.shape[1]
    for i, start in enumerate(range(0, R, robots_per_file)):
        end = min(start + robots_per_file, R)
        np.savez(out / f"robots_{i:02d}.npz",
                 states=states[:, start:end],
                 inputs=inputs[:, start:end],
                 timestamps=timestamps)
    return out.as_posix()

# ------------------------------------------------------------------
# 3. Minimal config creator
# ------------------------------------------------------------------
def make_config(raw_dir: str,
                conf_path="synthetic_config.yaml"):
    cfg = dict(
        data=dict(
            input_dir               = raw_dir,
            train_ratio             = 0.6,
            val_ratio               = 0.2,
            test_ratio              = 0.2,
            noise_std               = 0.02,
            num_multi_step_predictions = 15,
            verbose                 = True
        )
    )
    with open(conf_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return conf_path

# ------------------------------------------------------------------
# 4. End-to-end test run
# ------------------------------------------------------------------
def run_test():
    print(">>> Generating synthetic data")
    S, U, T = make_synthetic_dataset()

    print(">>> Writing NPZ shards")
    raw_dir = write_npz_files(S, U, T)

    print(">>> Creating config")
    cfg = make_config(raw_dir)

    print(">>> Running full preprocessing pipeline")
    process_data(cfg)

    # Load processed data
    train_npz = np.load("processed_data/train_data.npz")
    samples = train_npz["samples"]  # shape: (num_samples, 9, n_steps)
    with open("processed_data/normalization_params.json") as f:
        params = json.load(f)

    # --- Numerical validations ---
    # 1. Check for NaNs or Infs
    assert not np.isnan(samples).any(), "Processed samples contain NaNs!"
    assert not np.isinf(samples).any(), "Processed samples contain Infs!"

    # 2. Check normalization: mean and std for non-yaw states
    state_mean = np.array(params["state_mean"])
    state_std = np.array(params["state_std"])
    # Yaw index is 2, should be mean=0, std=1
    assert np.isclose(state_mean[2], 0.0), "Yaw mean not zero after normalization override!"
    assert np.isclose(state_std[2], 1.0), "Yaw std not one after normalization override!"

    # 3. Check sample shape
    assert samples.shape[1] == 9, "Each sample should have 9 features (7 states + 2 inputs)!"
    # assert samples.shape[2] == cfg["data"]["num_multi_step_predictions"], "Sample window size mismatch!"

    print("All numerical validations passed.")

    # --- Plotting normalized Δx, Δy trajectories for a few samples ---
    plt.figure(figsize=(10, 6))
    for i in range(min(5, samples.shape[0])):
        plt.plot(samples[i, 0, :], samples[i, 1, :], label=f"Robot {i}")
    plt.xlabel("Normalized Δx")
    plt.ylabel("Normalized Δy")
    plt.title("Normalized Relative Trajectories (Train Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()
