import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_track(env, ax=None):
    """Visualize the track layout."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define colors for different track elements:
    # 0 = track (white), 1 = wall (black), 2 = start line (red), 3 = finish line (green)
    cmap = ListedColormap(['white', 'black', 'red', 'green'])
    
    # Plot the track
    ax.imshow(env.track, cmap=cmap)
    
    # Add grid
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set axis labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{env.track_type.capitalize()} Track')
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', label='Track'),
        Patch(facecolor='black', edgecolor='gray', label='Wall'),
        Patch(facecolor='red', edgecolor='gray', label='Start'),
        Patch(facecolor='green', edgecolor='gray', label='Finish')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return ax

def visualize_trajectory(env, trajectory, ax=None):
    """Visualize a trajectory on the track."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    visualize_track(env, ax)
    
    # Plot trajectory points
    xs = [s[0] for s, _, _, _ in trajectory]
    ys = [s[1] for s, _, _, _ in trajectory]
    ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
    ax.plot(xs[0], ys[0], 'go', markersize=10)  # Start
    ax.plot(xs[-1], ys[-1], 'ro', markersize=10)  # End
    
    return ax

def visualize_policy(env, policy, ax=None):
    """Visualize the policy as arrows on the track."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    visualize_track(env, ax)
    
    # Sample states to visualize (avoid overcrowding)
    step = 5  # Adjust spacing as needed
    
    for x in range(0, env.shape[1], step):
        for y in range(0, env.shape[0], step):
            # Skip walls and check if position is valid
            if env.track[y, x] == 1:  # Wall
                continue
            
            # Display policy for states with zero velocity
            state = (x, y, 0, 0)
            if state in policy:
                dx, dy = policy[state]
                ax.arrow(x, y, dx*2, -dy*2, head_width=0.8, head_length=0.8, 
                         fc='blue', ec='blue', alpha=0.6)
    
    return ax

def plot_learning_curves(results_dict, figsize=(15, 5)):
    """Plot learning curves from multiple experiments."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot returns
    ax = axes[0]
    for name, result in results_dict.items():
        ax.plot(result['stats']['returns'], label=name)
    ax.set_title('Returns per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.legend()
    
    # Plot episode lengths
    ax = axes[1]
    for name, result in results_dict.items():
        ax.plot(result['stats']['lengths'], label=name)
    ax.set_title('Episode Length')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    
    # Plot crashes
    ax = axes[2]
    for name, result in results_dict.items():
        ax.plot(result['stats']['crashes'], label=name)
    ax.set_title('Crashes per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of Crashes')
    ax.legend()
    
    plt.tight_layout()
    return fig, axes

def plot_smoothed_curves(data, window=100, label=None, ax=None):
    """Plot smoothed curve using a moving average."""
    if ax is None:
        fig, ax = plt.subplots()
    
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, label=label)
    
    return ax
