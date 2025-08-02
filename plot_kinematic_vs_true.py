import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv("kinematic_vs_true_trajectories.csv")

num_samples = df["sample_idx"].max() + 1

for sample_idx in range(num_samples):
    sample_df = df[df["sample_idx"] == sample_idx]
    # Get all timesteps for this sample
    timesteps = sample_df["timestep"].unique()
    # Get delta_x and delta_y for true and predicted
    true_x = sample_df[(sample_df["state_idx"] == 0)]["true_state"].values
    true_y = sample_df[(sample_df["state_idx"] == 1)]["true_state"].values
    pred_x = sample_df[(sample_df["state_idx"] == 0)]["pred_state"].values
    pred_y = sample_df[(sample_df["state_idx"] == 1)]["pred_state"].values

    plt.figure(figsize=(8, 6))
    plt.plot(true_x, true_y, label="True Trajectory", color="black")
    plt.plot(pred_x, pred_y, label="Kinematic Model", color="red", linestyle="--")

    # Add markers at each point where the trajectory switches (i.e., at every step)
    plt.scatter(true_x, true_y, color="black", marker="o", s=30, alpha=0.6)
    plt.scatter(pred_x, pred_y, color="red", marker="x", s=30, alpha=0.6)

    plt.xlabel("delta_x [m]")
    plt.ylabel("delta_y [m]")
    plt.title(f"XY Trajectory: Sample {sample_idx}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()