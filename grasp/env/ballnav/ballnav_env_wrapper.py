"""
Gym environment wrapper for ballnav world model.
"""

import gym
import numpy as np
from .ballnav_world_model import BallNavWorldModel


class BallNavEnvWrapper(gym.Env):
    """
    Gym environment wrapper for ballnav world model.
    This provides the standard gym interface that the planning framework expects.
    """

    def __init__(self, device="cpu", num_obstacles=1):
        super().__init__()

        # Create the world model
        self.world_model = BallNavWorldModel(num_obstacles=num_obstacles, device=device)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([self.world_model.x_min, self.world_model.y_min]),
            high=np.array([self.world_model.x_max, self.world_model.y_max]),
            dtype=np.float32
        )

        # Current state
        self.current_state = None
        self.action_dim = 2

    def reset(self):
        """Reset environment to start position."""
        self.current_state = self.world_model.start_pos.cpu().numpy()
        return self.current_state

    def step(self, action):
        """Step the environment with the given action."""
        # Convert to torch tensor
        import torch
        state_tensor = torch.tensor(self.current_state, device=self.world_model.device).unsqueeze(0)  # (1, 2)
        action_tensor = torch.tensor(action, device=self.world_model.device).unsqueeze(0)  # (1, 2)

        # Use world model to predict next state
        # Create dummy state sequence for the interface
        dummy_next = torch.zeros_like(state_tensor)
        state_with_dummy = torch.cat([state_tensor, dummy_next], dim=0).unsqueeze(0)  # (1, 2, 2)
        action_expanded = action_tensor.unsqueeze(1)  # (1, 1, 2)
        next_state = self.world_model.next_state(state_with_dummy, action_expanded)  # (1, 1, 2)

        # Update current state
        self.current_state = next_state.squeeze().cpu().numpy()

        # Check if goal is reached
        goal_distance = np.linalg.norm(self.current_state - self.world_model.goal_pos.cpu().numpy())
        done = goal_distance < 0.5
        reward = -goal_distance if not done else 0.0

        info = {"state": self.current_state}

        return self.current_state, reward, done, info

    def render(self, mode='rgb_array'):
        """Render the environment."""
        if self.current_state is None:
            return None

        import torch
        state_tensor = torch.tensor(self.current_state, device=self.world_model.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        fig, ax = self.world_model.visualize(state_tensor)

        if mode == 'rgb_array':
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            import matplotlib.pyplot as plt
            plt.close(fig)
            return buf
        else:
            return fig, ax

    def sample_random_init_goal_states(self, seeds):
        """Sample random initial and goal states."""
        # Handle both single seed and list of seeds
        if isinstance(seeds, (int, np.integer)):
            seeds = [seeds]

        np.random.seed(seeds[0])

        # Random positions within the environment bounds
        init_states = []
        goal_states = []

        for _ in range(len(seeds)):
            init_state = np.array([1.0, 5.0])
            goal_state = np.array([9.0, 5.0])

            init_states.append(init_state)
            goal_states.append(goal_state)

        # Return single arrays if only one seed was provided
        if len(seeds) == 1:
            return init_states[0], goal_states[0]
        else:
            return np.array(init_states), np.array(goal_states)

    def update_env(self, env_info):
        """Update environment configuration (compatibility method)."""
        pass  # ballnav doesn't need updating

    def prepare(self, seeds, init_states):
        """Prepare environment with specific initial states."""
        # Handle both single state and batch of states
        if isinstance(init_states, np.ndarray) and len(init_states.shape) == 1:
            # Single state from vectorized environment
            self.current_state = init_states
            return init_states, init_states  # Return state vector as both obs and state

        # Handle single seed
        if isinstance(seeds, (int, np.integer)):
            seeds = [seeds]

        # Handle batch of states
        state_list = []
        for init_state in init_states:
            state_list.append(init_state)

        return np.array(state_list), np.array(state_list)  # Return state vectors as both obs and states

    def rollout(self, seeds, init_states, actions):
        """Rollout actions from initial states."""
        import torch

        # Convert to tensors
        states_tensor = torch.tensor(init_states, device=self.world_model.device)  # (B, 2)
        actions_tensor = torch.tensor(actions, device=self.world_model.device)  # (B, T, 2)

        # Use world model rollout
        all_states = self.world_model.rollout(states_tensor, actions_tensor)  # (B, T+1, 2)

        # Convert back to numpy and create observations
        all_states_np = all_states.cpu().numpy()

        obs_dict = {}
        obs_dict["visual"] = all_states_np.reshape(all_states_np.shape[0], all_states_np.shape[1], 1, 2)  # (B, T+1, 1, 2)
        obs_dict["proprio"] = np.zeros((all_states_np.shape[0], all_states_np.shape[1], 0))  # (B, T+1, 0)

        return obs_dict, all_states_np

    def eval_state(self, goal_state, cur_state):
        """Evaluate if goal is reached."""
        distance = np.linalg.norm(goal_state - cur_state)
        success = distance < 0.5
        return {
            'success': success,
            'state_dist': distance,
        }
