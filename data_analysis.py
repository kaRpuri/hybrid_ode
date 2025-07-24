#!/usr/bin/env python3
"""
Data analysis script to examine the NPZ file structure and compare with paper requirements
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_data():
    # Load the NPZ file
    data = np.load('/home/saichand/ros2_ws/src/hybrid_ode/data/hybrid_neural_ode_data.npz')
    
    print("=== NPZ File Contents ===")
    for key in data.keys():
        print(f"Key: {key}, Shape: {data[key].shape}, Type: {data[key].dtype}")
    
    # Extract data
    states = data['states']  # Shape: (500, 5, 7)
    inputs = data['inputs']  # Shape: (500, 5, 2)
    timestamps = data.get('timestamps', np.arange(states.shape[0]))  # (500,)
    
    print(f"\n=== Data Analysis ===")
    print(f"States shape: {states.shape}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Timestamps shape: {timestamps.shape}")
    print(f"Number of timesteps: {states.shape[0]}")
    print(f"Number of robots: {states.shape[1]}")
    print(f"State dimensions: {states.shape[2]}")
    print(f"Input dimensions: {inputs.shape[2]}")
    
    # State variables according to your log:
    # x_pos, y_pos, yaw, steering, velocity, side_slip, yaw_rate
    state_names = ['x_pos', 'y_pos', 'yaw', 'steering', 'velocity', 'side_slip', 'yaw_rate']
    input_names = ['acceleration', 'steering_rate']
    
    print(f"\n=== State Variables ===")
    for i, name in enumerate(state_names):
        data_flat = states[:, :, i].flatten()
        print(f"{i}: {name:12} - Min: {data_flat.min():8.3f}, Max: {data_flat.max():8.3f}, Mean: {data_flat.mean():8.3f}")
    
    print(f"\n=== Input Variables ===")
    for i, name in enumerate(input_names):
        data_flat = inputs[:, :, i].flatten()
        print(f"{i}: {name:15} - Min: {data_flat.min():8.3f}, Max: {data_flat.max():8.3f}, Mean: {data_flat.mean():8.3f}")
    
    return states, inputs, timestamps, state_names, input_names

def compare_with_paper():
    """
    Compare our data with paper requirements
    """
    print(f"\n=== Paper vs Our Data Comparison ===")
    
    # Paper state vector (Eq. 9): [Δx, Δy, ψ, δ, v, β, ω]
    paper_states = ['Δx (x_pos)', 'Δy (y_pos)', 'ψ (yaw)', 'δ (steering)', 'v (velocity)', 'β (side_slip)', 'ω (yaw_rate)']
    our_states = ['x_pos', 'y_pos', 'yaw', 'steering', 'velocity', 'side_slip', 'yaw_rate']
    
    print("Paper states vs Our states:")
    for i, (paper, ours) in enumerate(zip(paper_states, our_states)):
        print(f"{i}: {paper:20} <-> {ours}")
    
    # Paper inputs (Eq. 8): [ax, vδ]
    paper_inputs = ['ax (acceleration)', 'vδ (steering_rate)']
    our_inputs = ['acceleration', 'steering_rate']
    
    print("\nPaper inputs vs Our inputs:")
    for i, (paper, ours) in enumerate(zip(paper_inputs, our_inputs)):
        print(f"{i}: {paper:20} <-> {ours}")
    
    print(f"\nData mapping looks compatible!")
    print(f"We have 7 states matching the paper's 7 states")
    print(f"We have 2 inputs matching the paper's 2 inputs")
    print(f"We have 500 timesteps across 5 robots (good for training)")


def plot_trajectory_paths(states, state_names):
    """
    Plot 2D trajectory paths for all robots (x_pos vs y_pos).
    """
    print(f"\n=== Plotting Trajectory Paths ===")
    
    plt.figure(figsize=(10, 8))
    num_robots = states.shape[1]
    # Use a colormap for any number of robots
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(num_robots)]
    
    for robot_idx in range(num_robots):
        x_pos = states[:, robot_idx, 0]  # x_pos
        y_pos = states[:, robot_idx, 1]  # y_pos
        
        plt.plot(x_pos, y_pos, color=colors[robot_idx], linewidth=2, 
                label=f'Robot {robot_idx}', alpha=0.8)
        
        # Mark start and end points
        plt.plot(x_pos[0], y_pos[0], 'o', color=colors[robot_idx], 
                markersize=8)
        plt.plot(x_pos[-1], y_pos[-1], 's', color=colors[robot_idx], 
                markersize=8)
    
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('Vehicle Trajectory Paths for All Robots')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()


def plot_states_over_time(states, timestamps, state_names):
    """
    Plot all state variables over time for each robot.
    """
    print(f"\n=== Plotting States Over Time ===")
    
    num_robots = states.shape[1]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(num_robots)]
    
    # Plot states in separate figures to avoid layout issues
    for state_idx in range(min(7, len(state_names))):
        plt.figure(figsize=(12, 6))
        
        for robot_idx in range(num_robots):
            plt.plot(timestamps, states[:, robot_idx, state_idx], 
                   color=colors[robot_idx], linewidth=1.5, 
                   label=f'Robot {robot_idx}', alpha=0.8)
        
        plt.xlabel('Time [s]')
        plt.ylabel(state_names[state_idx])
        plt.title(f'State: {state_names[state_idx]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        plt.close()


def plot_inputs_over_time(inputs, timestamps, input_names):
    """
    Plot input variables over time for each robot.
    """
    print(f"\n=== Plotting Inputs Over Time ===")
    
    num_robots = inputs.shape[1]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(num_robots)]
    
    for input_idx in range(len(input_names)):
        plt.figure(figsize=(12, 6))
        
        for robot_idx in range(num_robots):
            plt.plot(timestamps, inputs[:, robot_idx, input_idx], 
                   color=colors[robot_idx], linewidth=1.5, 
                   label=f'Robot {robot_idx}', alpha=0.8)
        
        plt.xlabel('Time [s]')
        plt.ylabel(input_names[input_idx])
        plt.title(f'Input: {input_names[input_idx]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        plt.close()


def plot_individual_robot(states, inputs, timestamps, state_names, input_names, 
                         robot_idx=0, save_plots=True):
    """
    Detailed plots for a specific robot.
    """
    print(f"\n=== Plotting Individual Robot {robot_idx} ===")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Trajectory path
    ax1 = plt.subplot(4, 3, 1)
    x_pos = states[:, robot_idx, 0]
    y_pos = states[:, robot_idx, 1]
    plt.plot(x_pos, y_pos, 'b-', linewidth=2)
    plt.plot(x_pos[0], y_pos[0], 'go', markersize=8, label='Start')
    plt.plot(x_pos[-1], y_pos[-1], 'ro', markersize=8, label='End')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title(f'Robot {robot_idx} Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2-8. State variables
    for state_idx in range(min(7, len(state_names))):
        ax = plt.subplot(4, 3, state_idx + 2)
        plt.plot(timestamps, states[:, robot_idx, state_idx], 'b-', linewidth=1.5)
        plt.xlabel('Time [s]')
        plt.ylabel(state_names[state_idx])
        plt.title(f'{state_names[state_idx]}')
        plt.grid(True, alpha=0.3)
    
    # 9-10. Input variables
    for input_idx in range(len(input_names)):
        ax = plt.subplot(4, 3, 9 + input_idx)
        plt.plot(timestamps, inputs[:, robot_idx, input_idx], 'r-', linewidth=1.5)
        plt.xlabel('Time [s]')
        plt.ylabel(input_names[input_idx])
        plt.title(f'{input_names[input_idx]}')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Complete Data Analysis for Robot {robot_idx}', fontsize=16)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'results/robot_{robot_idx}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: results/robot_{robot_idx}_analysis.png")
    
    plt.show()


def plot_data_statistics(states, inputs, state_names, input_names, save_plots=True):
    """
    Plot statistical analysis of the data.
    """
    print(f"\n=== Plotting Data Statistics ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. State variable ranges across all robots
    ax = axes[0, 0]
    state_mins = []
    state_maxs = []
    state_means = []
    
    for state_idx in range(len(state_names)):
        data_flat = states[:, :, state_idx].flatten()
        state_mins.append(data_flat.min())
        state_maxs.append(data_flat.max())
        state_means.append(data_flat.mean())
    
    x_pos = np.arange(len(state_names))
    ax.bar(x_pos - 0.2, state_mins, 0.2, label='Min', alpha=0.7)
    ax.bar(x_pos, state_means, 0.2, label='Mean', alpha=0.7)
    ax.bar(x_pos + 0.2, state_maxs, 0.2, label='Max', alpha=0.7)
    
    ax.set_xlabel('State Variables')
    ax.set_ylabel('Values')
    ax.set_title('State Variable Statistics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(state_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Input variable ranges
    ax = axes[0, 1]
    input_mins = []
    input_maxs = []
    input_means = []
    
    for input_idx in range(len(input_names)):
        data_flat = inputs[:, :, input_idx].flatten()
        input_mins.append(data_flat.min())
        input_maxs.append(data_flat.max())
        input_means.append(data_flat.mean())
    
    x_pos = np.arange(len(input_names))
    ax.bar(x_pos - 0.2, input_mins, 0.2, label='Min', alpha=0.7)
    ax.bar(x_pos, input_means, 0.2, label='Mean', alpha=0.7)
    ax.bar(x_pos + 0.2, input_maxs, 0.2, label='Max', alpha=0.7)
    
    ax.set_xlabel('Input Variables')
    ax.set_ylabel('Values')
    ax.set_title('Input Variable Statistics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(input_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity vs time for all robots
    ax = axes[1, 0]
    num_robots = states.shape[1]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(num_robots)]
    velocity_idx = state_names.index('velocity')
    
    for robot_idx in range(num_robots):
        ax.plot(states[:, robot_idx, velocity_idx], 
               color=colors[robot_idx], linewidth=1.5, 
               label=f'Robot {robot_idx}', alpha=0.8)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Steering angle vs time for all robots
    ax = axes[1, 1]
    steering_idx = state_names.index('steering')
    
    for robot_idx in range(num_robots):
        ax.plot(states[:, robot_idx, steering_idx], 
               color=colors[robot_idx], linewidth=1.5, 
               label=f'Robot {robot_idx}', alpha=0.8)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Steering Angle [rad]')
    ax.set_title('Steering Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Data Statistics and Key Variables', fontsize=16)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('results/data_statistics.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: results/data_statistics.png")
    
    plt.show()


def plot_all_data(states, inputs, timestamps, state_names, input_names):
    """
    Generate plots for data visualization.
    """
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS FOR DATA VISUALIZATION")
    print(f"{'='*60}")
    
    # Generate core plots
    plot_trajectory_paths(states, state_names)
    plot_states_over_time(states, timestamps, state_names)
    plot_inputs_over_time(inputs, timestamps, input_names)
    
   
    

if __name__ == "__main__":
    # Analyze data structure
    states, inputs, timestamps, state_names, input_names = analyze_data()
    
    # Compare with paper requirements
    compare_with_paper()
    
    # Generate comprehensive plots
    plot_all_data(states, inputs, timestamps, state_names, input_names)
