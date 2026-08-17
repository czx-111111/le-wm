"""
Gym environment wrapper for random_mlp world model.
"""

import gym
import numpy as np
from .random_mlp_world_model import RandomMlpWorldModel

import torch

class RandomMlpEnvWrapper(gym.Env):
    """
    Gym environment wrapper for random_mlp world model.
    This provides the standard gym interface that the planning framework expects.
    """

    def __init__(self, device="cpu", state_dim=64, action_dim=64, H=10):
        super().__init__()

        # Create the world model
        self.world_model = RandomMlpWorldModel(state_dim=state_dim, action_dim=action_dim, H=H, device=device)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Current state
        self.current_state = None
        self.action_dim = action_dim
        self.state_dim = state_dim

    def reset(self):
        """Reset environment to start position."""
        self.current_state = self.world_model.start_pos.cpu().numpy()
        return self.current_state

    def step(self, action):
        """Step the environment with the given action."""
        # Convert to torch tensor
        import torch
        state_tensor = torch.tensor(self.current_state, device=self.world_model.device, dtype=torch.float32)  # (state_dim,)
        action_tensor = torch.tensor(action, device=self.world_model.device, dtype=torch.float32)  # (action_dim,)

        # Use world model to predict next state
        next_state = self.world_model.step(state_tensor, action_tensor)  # (state_dim,)

        # Update current state
        self.current_state = next_state.detach().cpu().numpy()

        # Check if goal is reached
        goal_distance = np.linalg.norm(self.current_state - self.world_model.goal_pos.cpu().numpy())
        done = goal_distance < self.world_model.goal_threshold
        reward = -goal_distance if not done else 0.0

        info = {"state": self.current_state}

        return self.current_state, reward, done, info

    def render(self, mode='rgb_array'):
        """Render the environment."""
        if self.current_state is None:
            return None

        import torch
        state_tensor = torch.tensor(self.current_state, device=self.world_model.device).unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
        fig_tensor = self.world_model.visualize(state_tensor)  # (1, 3, img_size, img_size)

        if mode == 'rgb_array':
            # Convert tensor to numpy array
            buf = fig_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (img_size, img_size, 3)
            buf = (buf * 255).astype(np.uint8)
            return buf
        else:
            # For other modes, create a matplotlib figure
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))

            # Simple visualization showing state magnitude
            if self.state_dim >= 2:
                ax.scatter(self.current_state[0], self.current_state[1], c='red', s=100, label='Current')
                ax.scatter(self.world_model.start_pos[0].cpu(), self.world_model.start_pos[1].cpu(),
                          c='blue', s=100, label='Start')
                ax.scatter(self.world_model.goal_pos[0].cpu(), self.world_model.goal_pos[1].cpu(),
                          c='green', s=100, label='Goal')
            else:
                ax.bar(['Current', 'Start', 'Goal'],
                      [self.current_state[0], self.world_model.start_pos[0].cpu(), self.world_model.goal_pos[0].cpu()])

            ax.legend()
            ax.set_title('Random MLP Environment State')
            return fig, ax

    def sample_random_init_goal_states(self, seeds, goal_H=10):
        """Sample random initial states and reachable goal states using rollout."""
        # Handle both single seed and list of seeds
        if isinstance(seeds, (int, np.integer)):
            seeds = [seeds]

        np.random.seed(seeds[0])

        # For high-dimensional states, generate random initial states and reachable goals
        init_states = []
        goal_states = []
        import pdb; pdb.set_trace()

        for _ in range(len(seeds)):
            # Sample random initial state
            init_state = np.random.randn(self.state_dim).astype(np.float32)

            # Sample random actions for goal_H steps
            random_actions = np.random.randn(goal_H, self.action_dim).astype(np.float32)

            # Convert to torch tensors for rollout
            init_state_tensor = torch.tensor(init_state, device=self.world_model.device, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
            actions_tensor = torch.tensor(random_actions, device=self.world_model.device, dtype=torch.float32).unsqueeze(0)  # (1, goal_H, action_dim)

            # Rollout using world model to get reachable goal state
            states, _ = self.world_model.rollout(init_state_tensor, actions_tensor)  # (1, goal_H+1, state_dim)
            goal_state = states[0, -1].detach().cpu().numpy()  # Take final state as goal

            init_states.append(init_state)
            goal_states.append(goal_state)

        # Return single arrays if only one seed was provided
        if len(seeds) == 1:
            return init_states[0], goal_states[0]
        else:
            return np.array(init_states), np.array(goal_states)

    def update_env(self, env_info):
        """Update environment configuration (compatibility method)."""
        pass  # random_mlp doesn't need updating

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
        states_tensor = torch.tensor(init_states, device=self.world_model.device, dtype=torch.float32)  # (B, state_dim)
        actions_tensor = torch.tensor(actions, device=self.world_model.device, dtype=torch.float32)  # (B, T, action_dim)

        # Use world model rollout
        all_states = self.world_model.rollout(states_tensor, actions_tensor)[0]  # (B, T+1, state_dim)

        # Convert back to numpy and create observations
        all_states_np = all_states.detach().cpu().numpy()

        obs_dict = {}
        obs_dict["visual"] = all_states_np.reshape(all_states_np.shape[0], all_states_np.shape[1], 1, self.state_dim)  # (B, T+1, 1, state_dim)
        obs_dict["proprio"] = np.zeros((all_states_np.shape[0], all_states_np.shape[1], 0))  # (B, T+1, 0)

        return obs_dict, all_states_np

    def eval_state(self, goal_state, cur_state):
        """Evaluate if goal is reached."""
        distance = np.linalg.norm(goal_state - cur_state)
        success = distance < self.world_model.goal_threshold
        return {
            'success': success,
            'state_dist': distance,
        }
