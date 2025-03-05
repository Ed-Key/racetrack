import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from racetrack import RacetrackEnv, LargePenaltyEnv, CumulativeDamageEnv, FuelConsumptionEnv, PositionDependentPenaltyEnv
from monte_carlo import monte_carlo_control, generate_policy_trajectory, off_policy_monte_carlo
from visualization import visualize_track, visualize_policy, visualize_trajectory, plot_learning_curves

def experiment_basic(track_type='l_shaped', num_episodes=5000, epsilon=0.1):
    """
    Run basic Monte Carlo control on the racetrack problem.
    
    Args:
        track_type: Type of track ('l_shaped' or 'diagonal')
        num_episodes: Number of episodes to run
        epsilon: Exploration probability
        
    Returns:
        policy: The learned policy
        Q: The action-value function
        stats: Performance statistics
    """
    print(f"Running basic experiment on {track_type} track...")
    env = RacetrackEnv(track_type)
    
    # Run Monte Carlo control
    policy, Q, stats = monte_carlo_control(env, num_episodes=num_episodes, epsilon=epsilon)
    
    # Plot learning curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(stats['returns'])
    plt.title('Returns per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    
    plt.subplot(1, 3, 2)
    plt.plot(stats['lengths'])
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(1, 3, 3)
    plt.plot(stats['crashes'])
    plt.title('Crashes per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Crashes')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize policy
    plt.figure(figsize=(10, 10))
    visualize_policy(env, policy)
    plt.title(f"Learned Policy for {track_type.capitalize()} Track")
    plt.show()
    
    # Generate and visualize a trajectory
    policy_trajectory = generate_policy_trajectory(env, policy)
    plt.figure(figsize=(10, 10))
    visualize_trajectory(env, policy_trajectory)
    plt.title(f"Policy Trajectory (Steps: {len(policy_trajectory)})")
    plt.show()
    
    # Save policy
    with open(f"{track_type}_policy.pkl", "wb") as f:
        pickle.dump(policy, f)
    
    return policy, Q, stats

def compare_crash_penalties(track_type='l_shaped', num_episodes=5000):
    """
    Compare different crash penalty structures.
    
    Args:
        track_type: Type of track ('l_shaped' or 'diagonal')
        num_episodes: Number of episodes to run
        
    Returns:
        results: Dictionary of results for each penalty type
    """
    print(f"Comparing crash penalties on {track_type} track...")
    
    environments = {
        'Baseline': RacetrackEnv(track_type),
        'Large Penalty': LargePenaltyEnv(track_type),
        'Cumulative Damage': CumulativeDamageEnv(track_type),
        'Fuel Consumption': FuelConsumptionEnv(track_type),
        'Position-Dependent': PositionDependentPenaltyEnv(track_type)
    }
    
    results = {}
    
    for name, env in environments.items():
        print(f"Training with {name} penalty...")
        policy, Q, stats = monte_carlo_control(env, num_episodes=num_episodes)
        results[name] = {
            'policy': policy,
            'Q': Q,
            'stats': stats
        }
        
        # Generate a trajectory using the learned policy
        policy_trajectory = generate_policy_trajectory(env, policy)
        
        # Visualize policy and trajectory
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        visualize_policy(env, policy, ax=axes[0])
        axes[0].set_title(f"{name} Policy")
        
        visualize_trajectory(env, policy_trajectory, ax=axes[1])
        axes[1].set_title(f"{name} Trajectory (Steps: {len(policy_trajectory)})")
        
        plt.tight_layout()
        plt.savefig(f"{track_type}_{name.lower().replace(' ', '_')}.png")
        plt.show()
    
    # Plot comparison
    fig, axes = plot_learning_curves(results)
    plt.suptitle(f"Comparison of Crash Penalties on {track_type.capitalize()} Track")
    plt.tight_layout()
    plt.savefig(f"{track_type}_penalty_comparison.png")
    plt.show()
    
    return results

def experiment_off_policy(track_type='l_shaped', num_episodes=5000):
    """
    Compare on-policy vs off-policy Monte Carlo methods.
    
    Args:
        track_type: Type of track ('l_shaped' or 'diagonal')
        num_episodes: Number of episodes to run
        
    Returns:
        results: Dictionary with results from both methods
    """
    print(f"Comparing on-policy vs off-policy Monte Carlo on {track_type} track...")
    
    env = RacetrackEnv(track_type)
    
    # Run on-policy Monte Carlo
    print("Training with on-policy Monte Carlo...")
    on_policy, on_q, on_stats = monte_carlo_control(env, num_episodes=num_episodes)
    
    # Run off-policy Monte Carlo
    print("Training with off-policy Monte Carlo...")
    off_policy, off_q, off_stats = off_policy_monte_carlo(env, num_episodes=num_episodes)
    
    # Compile results
    results = {
        'On-Policy': {
            'policy': on_policy,
            'Q': on_q,
            'stats': on_stats
        },
        'Off-Policy': {
            'policy': off_policy,
            'Q': off_q,
            'stats': off_stats
        }
    }
    
    # Plot comparison
    fig, axes = plot_learning_curves(results)
    plt.suptitle(f"Comparison of On-Policy vs Off-Policy MC on {track_type.capitalize()} Track")
    plt.tight_layout()
    plt.savefig(f"{track_type}_on_off_policy_comparison.png")
    plt.show()
    
    # Compare trajectories
    on_trajectory = generate_policy_trajectory(env, on_policy)
    off_trajectory = generate_policy_trajectory(env, off_policy)
    
    # Visualize policies and trajectories
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    visualize_policy(env, on_policy, ax=axes[0, 0])
    axes[0, 0].set_title("On-Policy Monte Carlo Policy")
    
    visualize_trajectory(env, on_trajectory, ax=axes[0, 1])
    axes[0, 1].set_title(f"On-Policy Trajectory (Steps: {len(on_trajectory)})")
    
    visualize_policy(env, off_policy, ax=axes[1, 0])
    axes[1, 0].set_title("Off-Policy Monte Carlo Policy")
    
    visualize_trajectory(env, off_trajectory, ax=axes[1, 1])
    axes[1, 1].set_title(f"Off-Policy Trajectory (Steps: {len(off_trajectory)})")
    
    plt.tight_layout()
    plt.savefig(f"{track_type}_on_off_policy_trajectories.png")
    plt.show()
    
    return results

if __name__ == "__main__":
    # Basic experiment on L-shaped track
    l_policy, l_q, l_stats = experiment_basic('l_shaped', num_episodes=1000)
    
    # Basic experiment on diagonal track
    d_policy, d_q, d_stats = experiment_basic('diagonal', num_episodes=1000)
    
    # Compare crash penalties on L-shaped track
    penalty_results = compare_crash_penalties('l_shaped', num_episodes=1000)
    
    # Compare on-policy vs off-policy Monte Carlo
    off_policy_results = experiment_off_policy('l_shaped', num_episodes=1000)
