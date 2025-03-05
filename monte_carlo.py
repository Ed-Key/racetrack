import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

def monte_carlo_control(env, num_episodes=10000, gamma=1.0, epsilon=0.1):
    """
    First-visit Monte Carlo control with exploring starts.
    
    Args:
        env: The environment
        num_episodes: Number of episodes to run
        gamma: Discount factor
        epsilon: Exploration probability
        
    Returns:
        policy: The learned policy
        Q: The action-value function
        stats: Performance statistics
    """
    # Initialize Q(s,a) and policy
    Q = defaultdict(lambda: defaultdict(float))
    returns_count = defaultdict(lambda: defaultdict(int))
    policy = defaultdict(lambda: (0, 0))  # Default to no velocity change
    
    # All possible actions
    actions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
    
    # Track performance
    episode_lengths = []
    total_returns = []
    crash_counts = []
    
    for episode in tqdm(range(num_episodes)):
        # Generate episode with exploring starts
        trajectory = generate_episode(env, policy, epsilon)
        episode_lengths.append(len(trajectory))
        
        # Track returns and crashes
        G = 0
        crashes = sum(1 for _, _, _, info in trajectory if info.get('crash', False))
        crash_counts.append(crashes)
        
        # Process trajectory in reverse for returns calculation
        visited_state_actions = set()
        for t in range(len(trajectory)-1, -1, -1):
            state, action, reward, _ = trajectory[t]
            state_tuple = tuple(state)
            
            # Calculate return
            G = gamma * G + reward
            
            # First-visit MC update
            if (state_tuple, action) not in visited_state_actions:
                visited_state_actions.add((state_tuple, action))
                returns_count[state_tuple][action] += 1
                Q[state_tuple][action] += (G - Q[state_tuple][action]) / returns_count[state_tuple][action]
                
                # Policy improvement - greedy with respect to Q
                policy[state_tuple] = max(actions, key=lambda a: Q[state_tuple][a])
        
        total_returns.append(G)
    
    return policy, Q, {'returns': total_returns, 'lengths': episode_lengths, 'crashes': crash_counts}

def generate_episode(env, policy, epsilon=0.1):
    """Generate an episode using the given policy with epsilon-greedy exploration."""
    trajectory = []
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random action
            action = (np.random.randint(-1, 2), np.random.randint(-1, 2))
        else:
            # Policy action
            state_tuple = tuple(state)
            action = policy[state_tuple]
        
        next_state, reward, done, info = env.step(action)
        trajectory.append((state, action, reward, info))
        state = next_state
    
    return trajectory

def generate_policy_trajectory(env, policy, max_steps=100):
    """Generate a trajectory using the learned policy."""
    trajectory = []
    state = env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        state_tuple = tuple(state)
        # If state not in policy (rare with good exploration), use default
        if state_tuple not in policy:
            action = (0, 0)
        else:
            action = policy[state_tuple]
        
        next_state, reward, done, info = env.step(action)
        trajectory.append((state, action, reward, info))
        state = next_state
        step += 1
    
    return trajectory

def generate_random_trajectory(env, max_steps=100):
    """Generate a random trajectory on the track."""
    trajectory = []
    state = env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        action = (np.random.randint(-1, 2), np.random.randint(-1, 2))
        next_state, reward, done, info = env.step(action)
        trajectory.append((state, action, reward, info))
        state = next_state
        step += 1
    
    return trajectory

def off_policy_monte_carlo(env, num_episodes=10000, gamma=1.0):
    """
    Off-policy Monte Carlo control with weighted importance sampling.
    Based on Exercise 5.14.
    """
    # Initialize C(s,a) and Q(s,a)
    Q = defaultdict(lambda: defaultdict(float))
    C = defaultdict(lambda: defaultdict(float))
    
    # Initialize target policy (greedy)
    target_policy = defaultdict(lambda: (0, 0))
    
    # Initialize behavior policy (random)
    behavior_probs = {}
    actions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
    num_actions = len(actions)
    
    # Track performance
    episode_lengths = []
    total_returns = []
    crash_counts = []
    
    for episode in tqdm(range(num_episodes)):
        # Generate episode using behavior policy
        trajectory = []
        state = env.reset()
        done = False
        
        while not done:
            # Random action selection (behavior policy)
            action_idx = np.random.randint(num_actions)
            action = actions[action_idx]
            
            next_state, reward, done, info = env.step(action)
            trajectory.append((state, action, reward, info))
            state = next_state
        
        episode_lengths.append(len(trajectory))
        crashes = sum(1 for _, _, _, info in trajectory if info.get('crash', False))
        crash_counts.append(crashes)
        
        # Process trajectory for updates
        G = 0
        W = 1.0
        for t in range(len(trajectory)-1, -1, -1):
            state, action, reward, _ = trajectory[t]
            state_tuple = tuple(state)
            
            # Calculate return
            G = gamma * G + reward
            
            # Update weighted returns
            C[state_tuple][action] += W
            Q[state_tuple][action] += (W / C[state_tuple][action]) * (G - Q[state_tuple][action])
            
            # Update target policy (greedy)
            target_policy[state_tuple] = max(actions, key=lambda a: Q[state_tuple][a])
            
            # If the action doesn't match the target policy, break
            if action != target_policy[state_tuple]:
                break
                
            # Update importance sampling weight
            W *= num_actions  # Since behavior policy is uniform random
        
        total_returns.append(G)
    
    return target_policy, Q, {'returns': total_returns, 'lengths': episode_lengths, 'crashes': crash_counts}
