import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

class RacetrackEnv:
    """
    Racetrack environment based on Figure 5.5 from Sutton & Barto.
    
    State: (x, y, vx, vy) - position and velocity
    Action: (ax, ay) - velocity increments (-1, 0, or 1)
    Reward: -1 per step until finish
    """
    
    def __init__(self, track_type='l_shaped'):
        """Initialize the environment with specified track type."""
        self.track_type = track_type
        self.track = self._create_track(track_type)
        self.shape = self.track.shape
        self.start_positions = self._find_start_positions()
        self.finish_positions = self._find_finish_positions()
        
        # State space dimensions
        self.max_velocity = 5  # Maximum velocity in any direction
        
        # Reset to initial state
        self.reset()
    
    def _create_track(self, track_type):
        """Create track layout based on specified type."""
        # Simplified track loading for custom tracks
        if track_type == 'custom_left':
            try:
                # First try direct path
                if os.path.exists('LeftTrack.npy'):
                    track = np.load('LeftTrack.npy', allow_pickle=True)
                    print("Successfully loaded custom left track")
                    return track
                # Try relative path if direct path fails
                elif os.path.exists(os.path.join(os.getcwd(), 'LeftTrack.npy')):
                    track = np.load(os.path.join(os.getcwd(), 'LeftTrack.npy'), allow_pickle=True)
                    print("Successfully loaded custom left track from current directory")
                    return track
                else:
                    print("LeftTrack.npy not found. ")
                    track_type = 'l_shaped'
            except Exception as e:
                print(f"Error loading custom left track: {e}")
        
        elif track_type == 'custom_right':
            try:
                # First try direct path
                if os.path.exists('RightTrack.npy'):
                    track = np.load('RightTrack.npy', allow_pickle=True)  # Removed comma here
                    print("Successfully loaded custom right track")
                    return track
                # Try relative path if direct path fails
                elif os.path.exists(os.path.join(os.getcwd(), 'RightTrack.npy')):
                    track = np.load(os.path.join(os.getcwd(), 'RightTrack.npy'), allow_pickle=True)
                    print("Successfully loaded custom right track from current directory")
                    return track
                else:
                    print("RightTrack.npy not found. ")
                    track_type = 'l_shaped'  # Fall back to l_shaped track
            except Exception as e:
                print(f"Error loading custom right track: {e}")
        
            return track
        
        # Add more track types as needed
        return None  # Should not reach here if all track types are handled
    
    def _find_start_positions(self):
        """Find all valid starting positions."""
        y_indices, x_indices = np.where(self.track == 2)
        return list(zip(x_indices, y_indices))
    
    def _find_finish_positions(self):
        """Find all finish line positions."""
        y_indices, x_indices = np.where(self.track == 3)
        return list(zip(x_indices, y_indices))
    
    def reset(self):
        """Reset car to random starting position with zero velocity."""
        # Choose random starting position
        start_pos = random.choice(self.start_positions)
        self.position = list(start_pos)
        self.velocity = [0, 0]  # Start with zero velocity
        
        return self._get_state()
    
    def _get_state(self):
        """Return current state as (x, y, vx, vy)."""
        return self.position + self.velocity
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Tuple (ax, ay) of velocity increments (-1, 0, or 1)
            
        Returns:
            next_state: New state after action
            reward: Reward for this step (-1)
            done: Whether episode is done
            info: Additional information
        """
        # Action is a tuple (ax, ay) of velocity increments
        ax, ay = action
        
        # 10% chance of acceleration being zero regardless of action
        if random.random() < 0.1:
            ax, ay = 0, 0
        
        # Update velocity
        vx, vy = self.velocity
        vx = max(-self.max_velocity, min(self.max_velocity, vx + ax))
        vy = max(-self.max_velocity, min(self.max_velocity, vy - ay))  # Inverted y-axis
        self.velocity = [vx, vy]
        
        # If velocity is zero in both dimensions, force a move
        if vx == 0 and vy == 0:
            vx, vy = 1, 0  # Default movement
            self.velocity = [vx, vy]
        
        # Update position
        old_position = self.position.copy()
        new_x = self.position[0] + vx
        new_y = self.position[1] + vy
        
        # Check for collisions
        collision = False
        done = False
        finish = False
        info = {'crash': False}
        
        # Check if the path intersects with any walls using Bresenham's line algorithm
        collision = self._check_path_collision(old_position, [new_x, new_y])
        
        if collision:
            # Reset to random start with zero velocity
            info['crash'] = True
            start_pos = random.choice(self.start_positions)
            self.position = list(start_pos)
            self.velocity = [0, 0]
        # Check if crossed finish line
        elif self._check_finish([new_x, new_y]):
            self.position = [new_x, new_y]
            finish = True
            done = True
        else:
            self.position = [new_x, new_y]
        
        # Return step information
        reward = -1  # -1 reward per step
        info['old_position'] = old_position
        info['collision'] = collision
        info['finish'] = finish
        
        return self._get_state(), reward, done, info
    
    def _check_path_collision(self, start_pos, end_pos):
        """
        Check if the path between start_pos and end_pos intersects with any walls
        using Bresenham's line algorithm.
        """
        x0, y0 = int(start_pos[0]), int(start_pos[1])
        x1, y1 = int(end_pos[0]), int(end_pos[1])
        
        # Check if end position is outside boundaries
        if (x1 < 0 or x1 >= self.shape[1] or 
            y1 < 0 or y1 >= self.shape[0]):
            return True
        
        # Bresenham's line algorithm to check intermediate points
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        
        while True:
            # Check current point
            if 0 <= x0 < self.shape[1] and 0 <= y0 < self.shape[0]:
                if self.track[y0, x0] == 1:  # Wall
                    return True
            else:
                # Out of bounds
                return True
            
            # Check if we reached the end position
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 >= dy:
                if x0 == x1:
                    break
                err += dy
                x0 += sx
            if e2 <= dx:
                if y0 == y1:
                    break
                err += dx
                y0 += sy
                
        return False
        
    def _check_collision(self, position):
        """Check if position collides with wall or is outside track."""
        x, y = position
        
        # Check if outside track boundaries
        if x < 0 or x >= self.shape[1] or y < 0 or y >= self.shape[0]:
            return True
        
        # Check if position is a wall (1)
        if self.track[int(y), int(x)] == 1:
            return True
        
        return False
    
    def _check_finish(self, position):
        """Check if position crosses finish line."""
        x, y = position
        
        # Check if outside track boundaries
        if x < 0 or x >= self.shape[1] or y < 0 or y >= self.shape[0]:
            return False
        
        # Check if position is at finish line (3)
        if self.track[int(y), int(x)] == 3:
            return True
        
        return False
    
    def render(self, ax=None):
        """Render the current state of the environment."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Define colors for different track elements
        cmap = ListedColormap(['white', 'black', 'red', 'green'])
        
        # Plot the track
        ax.imshow(self.track, cmap=cmap)
        
        # Plot car position
        ax.plot(self.position[0], self.position[1], 'bo', markersize=10)
        
        # Plot velocity vector
        ax.arrow(self.position[0], self.position[1], 
                 self.velocity[0], self.velocity[1], 
                 color='blue', width=0.1, head_width=0.5)
        
        # Add grid
        ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Set axis labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{self.track_type.capitalize()} Track')
        
        return ax

# Extended environment classes for different crash penalties
class LargePenaltyEnv(RacetrackEnv):
    """Environment with large penalty (-10) for crashes."""
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        if info.get('crash', False):
            reward -= 10  # Additional penalty
        return next_state, reward, done, info

class CumulativeDamageEnv(RacetrackEnv):
    """Environment with cumulative damage model."""
    def __init__(self, track_type='l_shaped'):
        super().__init__(track_type)
        self.damage = 0  # Initialize damage counter
    
    def reset(self):
        state = super().reset()
        self.damage = 0  # Reset damage
        return state
    
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        
        # Reduce damage over time (repair)
        self.damage = max(0, self.damage - 0.1)
        
        if info.get('crash', False):
            # Increase damage on crash
            self.damage += 1
            # Penalty scales with damage
            reward -= self.damage
            
            # Limit max velocity based on damage
            self.max_velocity = max(1, 5 - self.damage/5)
        
        info['damage'] = self.damage
        return next_state, reward, done, info

class FuelConsumptionEnv(RacetrackEnv):
    """Environment with fuel consumption model."""
    def __init__(self, track_type='l_shaped', initial_fuel=100):
        super().__init__(track_type)
        self.initial_fuel = initial_fuel
        self.fuel = initial_fuel
    
    def reset(self):
        state = super().reset()
        self.fuel = self.initial_fuel  # Reset fuel
        return state
    
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        
        # Calculate fuel consumption based on speed
        vx, vy = self.velocity
        fuel_used = 0.1 + 0.05 * (vx**2 + vy**2)
        self.fuel -= fuel_used
        
        # Additional fuel loss on crash
        if info.get('crash', False):
            self.fuel -= 5
        
        # Check if out of fuel
        if self.fuel <= 0:
            self.fuel = 0
            done = True
            reward -= 50  # Large penalty for running out of fuel
        
        info['fuel'] = self.fuel
        return next_state, reward, done, info

class PositionDependentPenaltyEnv(RacetrackEnv):
    """Environment with position-dependent crash penalties."""
    def __init__(self, track_type='l_shaped'):
        super().__init__(track_type)
        self.danger_map = self._create_danger_map()
    
    def _create_danger_map(self):
        """Create a map of danger levels for different positions."""
        danger_map = np.ones_like(self.track)
        
        # Higher penalties for narrow sections
        if self.track_type == 'l_shaped':
            # Corner area (higher risk)
            danger_map[15:25, 7:12] = 20
        elif self.track_type == 'diagonal':
            # Narrowest parts of diagonal section
            for y in range(4, 25):
                center_x = 29 - y
                danger_map[y, center_x-1:center_x+2] = 20
        
        return danger_map
    
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        
        if info.get('crash', False):
            # Get the position where the crash occurred
            old_x, old_y = info['old_position']
            if 0 <= old_y < self.shape[0] and 0 <= old_x < self.shape[1]:
                # Apply position-dependent penalty
                penalty = -self.danger_map[int(old_y), int(old_x)]
                reward += penalty
        
        return next_state, reward, done, info
