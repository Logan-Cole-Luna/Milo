import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PointNavigationEnv(gym.Env):
    """
    A simple 3D point navigation environment.
    The agent's goal is to reach a target in 3D space.
    """
    def __init__(self, max_steps=500, distance_threshold=0.005,
                 start_pos=np.array([-0.5, -0.5, -0.5], dtype=np.float32),
                 goal_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 random_goal=False):
        """Initialize the PointNavigationEnv.

        Args:
            max_steps (int): Maximum number of steps per episode.
            distance_threshold (float): Threshold distance to consider the goal reached.
            start_pos (np.array): Default starting position of the agent.
            goal_pos (np.array): Default goal position.
            random_goal (bool): If True, the goal position is randomized at each reset.
        """
        super(PointNavigationEnv, self).__init__()

        # Scale action space based on environment bounds (-1,1)
        # Allow steps up to 20% of the environment range per step
        self.action_space = spaces.Box(low=-0.6, high=0.6, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Increase max steps for evaluation
        self.max_steps = max_steps if not random_goal else max_steps
        self.distance_threshold = distance_threshold
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.random_goal = random_goal
        self.agent_pos = None
        self.current_step = 0
        self.step_penalty = -0.005 
        self.previous_distance = None # Store previous distance to goal

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state.

        Args:
            seed (int, optional): The seed that is used to initialize the environment's PRNG.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            tuple: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        # Initialize agent position
        self.agent_pos = self.start_pos.copy()
        self.current_step = 0
        
        # Generate random goal if enabled
        if self.random_goal:
            # Generate random goal with controlled distance
            # Generate goals that are neither too close nor too far
            min_distance = 0.3  # Minimum distance for challenge
            max_distance = 1.0  # Maximum reachable distance
            
            # Generate goals until we get one within the desired range
            while True:
                self.goal_pos = np.random.uniform(-0.8, 0.8, size=3).astype(np.float32)
                distance = np.linalg.norm(self.goal_pos - self.start_pos)
                if min_distance <= distance <= max_distance:
                    break
                
        self.previous_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        observation = np.concatenate([self.agent_pos, self.goal_pos])
        info = {}
        return observation, info

    def step(self, action):
        """Execute one time step within the environment.

        Args:
            action (np.array): The action taken by the agent.

        Returns:
            tuple: A tuple containing the observation, reward, done flag, truncated flag, and an info dictionary.
        """
        # Store previous position
        prev_pos = self.agent_pos.copy()
        
        # Move agent in all three dimensions
        self.agent_pos += action  # Full 3D movement
        self.agent_pos = np.clip(self.agent_pos, -1, 1)  # Keep within bounds

        # Calculate distance to goal
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)

        # Calculate reward - normalize to be in a reasonable range (-1 to 1)
        max_possible_distance = np.sqrt(12)
        normalized_distance = distance / max_possible_distance
        
        # Base reward from distance - more moderate scaling
        reward = -normalized_distance * 1.0  # Scale to approximately -1 to 0
        
        # More balanced improvement reward
        distance_improvement = self.previous_distance - distance
        normalized_improvement = distance_improvement / max_possible_distance
        reward += normalized_improvement * 5.0  # More moderate incentive (was 10.0)
        
        # Slightly stronger step penalty for efficiency
        reward += self.step_penalty * 2.0  # Doubled step penalty importance
        
        # More moderate but still significant goal reward
        if distance < self.distance_threshold:
            reward += 15.0  # More balanced bonus (was 20.0)

        # Penalize small movements
        movement = np.linalg.norm(self.agent_pos - prev_pos)
        if movement < 0.1:  # Increased threshold
            reward -= 0.05   # Stronger penalty (was 0.05)
        else:
            # Bonus for taking larger steps in the right direction
            movement_direction = self.goal_pos - prev_pos
            movement_direction = movement_direction / np.linalg.norm(movement_direction)
            actual_direction = self.agent_pos - prev_pos
            if np.linalg.norm(actual_direction) > 0:
                actual_direction = actual_direction / np.linalg.norm(actual_direction)
                direction_alignment = np.dot(movement_direction, actual_direction)
                if direction_alignment > 0.7:  # If moving in roughly the right direction
                    reward += 0.5 * movement  # Reward proportional to step size
        
        self.previous_distance = distance  # Update previous distance
        
        # Check if done
        done = distance < self.distance_threshold or self.current_step >= self.max_steps
        
        truncated = self.current_step >= self.max_steps

        observation = np.concatenate([self.agent_pos, self.goal_pos])
        info = {'distance': distance, 'goal_reached': distance < self.distance_threshold}  # Add info about goal achievement

        self.current_step += 1
        return observation, reward, done, truncated, info

    def render(self):
        """Render the environment.

        This method provides a simple text-based representation of the agent and goal positions.
        """
        # Simple text-based rendering
        print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}, Distance: {np.linalg.norm(self.agent_pos - self.goal_pos):.4f}")

    def close(self):
        """Perform any necessary cleanup.

        This method is called when the environment is no longer needed.
        """
        pass
