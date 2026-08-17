"""
Simple world model implementation for random_mlp environment.
This provides a differentiable world model that can be used with planners.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class RandomMlpWorldModel(nn.Module):
    """
    World model for random_mlp environment with differentiable dynamics.
    This provides the interface required by planners for planning in the MLP environment.
    """

    def __init__(self, state_dim=64, action_dim=64, H=10, device=None):
        super().__init__()

        # Environment parameters
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Set device with fallback to CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # MLP initialization: (state_dim + action_dim) -> state_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 2*self.state_dim),  # First FC layer
            nn.ReLU(),           # Non-linearity
            nn.Linear(2*self.state_dim, self.state_dim)  # Second FC layer (no activation)
        ).to(self.device)

        # Initialize MLP weights properly
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Keep MLP in training mode for differentiable forward passes
        self.mlp.train()

        self.start_pos = torch.randn(self.state_dim, device=self.device)
        self.H = H
        self.actions = torch.randn(self.H, self.action_dim, device=self.device)
        self.actions.requires_grad = False
        self.goal_threshold = 0.01
        self.goal_pos = None

        # Generate goal by rolling out the predefined actions
        self._generate_goal()

    def _generate_goal(self):
        """Generate goal position by rolling out predefined actions from start."""
        with torch.no_grad():
            curr_state = self.start_pos.clone()
            # roll out the world model to determine self.goal_pos:
            for i in range(self.H):
                curr_state = self.step(curr_state, self.actions[i])
            self.goal_pos = curr_state.detach()

    def is_at_goal(self, state):
        """Check if the point is at the goal."""
        distance = torch.norm(state - self.goal_pos)
        return distance < self.goal_threshold

    def step(self, state, action):
        """
        A differentiable MLP world model.

        Args:
            state: Current state tensor of shape (state_dim,) or (batch_size, state_dim)
            action: Action tensor of shape (action_dim,) or (batch_size, action_dim)

        Returns:
            next_state: The next state after applying the action
        """
        # Ensure inputs are 2D tensors for batch processing
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Concatenate state and action along the feature dimension
        combined_input = torch.cat([state, action], dim=1)

        # Pass through MLP
        next_state = self.mlp(combined_input)

        # Remove batch dimension if input was 1D
        if next_state.shape[0] == 1 and state.shape[0] == 1:
            next_state = next_state.squeeze(0)

        return next_state

    def next_state(self, state, action):
        """
        Compute next state from current state and current action.
        Works on batched tensors.

        Args:
            state: Current state tensor (B, T, state_dim) where T includes current and future states
            action: Action tensor (B, T-1, action_dim) for action commands

        Returns:
            next_states: Tensor (B, T-1, state_dim) of predicted next states
        """
        # Extract current states (all except last)
        current_states = state[:, :-1]  # (B, T-1, state_dim)

        # Get batch and time dimensions
        B, T_minus_1, _ = current_states.shape

        # Reshape for processing
        current_states_flat = current_states.reshape(B * T_minus_1, self.state_dim)  # (B*(T-1), state_dim)
        action_flat = action.reshape(B * T_minus_1, self.action_dim)  # (B*(T-1), action_dim)

        # Use the step function to get next states
        next_states_flat = self.step(current_states_flat, action_flat)  # (B*(T-1), state_dim)

        # Reshape back to original format
        next_states = next_states_flat.reshape(B, T_minus_1, self.state_dim)  # (B, T-1, state_dim)

        return next_states

    def simulate_one_step(self, state, action):
        """
        Same as step function but with consistent naming for planning interface.
        """
        return self.step(state, action)

    def rollout_encoded(self, obs_0, act):
        return self.rollout(obs_0[:,0], act)

    def rollout(self, obs_0, act):
        """
        Roll out all actions from initial state, output all states.

        Args:
            obs_0: Initial state (B, state_dim)
            act: Action sequence (B, T, action_dim)

        Returns:
            states: All states including initial state (B, T+1, state_dim)
            states_with_actions: All states with actions appended (B, T+1, state_dim + action_dim)
        """
        batch_size, T, _ = act.shape

        # Initialize state sequence with initial state
        states = torch.zeros(batch_size, T + 1, self.state_dim, device=self.device)
        states[:, 0] = obs_0

        # Roll out step by step using MLP dynamics
        for t in range(T):
            current_state = states[:, t]  # (B, state_dim)
            action = act[:, t]  # (B, action_dim)
            next_state = self.simulate_one_step(current_state, action)  # (B, state_dim)
            states[:, t + 1] = next_state

        # Prepare states_with_actions: (B, T+1, state_dim + action_dim)
        # For each timestep, concatenate state and action; for last state, repeat last action
        actions_padded = torch.zeros(batch_size, T + 1, self.action_dim, device=self.device)
        actions_padded[:, :-1, :] = act
        actions_padded[:, -1, :] = act[:, -1, :]  # Repeat last action for last state

        states_with_actions = torch.cat([states, actions_padded], dim=-1)  # (B, T+1, state_dim + action_dim)

        return states, states_with_actions

    def encode_observation_to_vector(self, state):
        """
        For us, this is the identity map since we work directly with state vectors.

        Args:
            state: State tensor (B, state_dim) or (B, T, state_dim)

        Returns:
            encoded_state: Same as input (identity mapping)
        """
        return state

    # Alias so CEM/GD planners (which call encode_obs) also work.
    encode_obs = encode_observation_to_vector

    def visualize(self, state, trajectory_index=0, img_size=256):
        """
        Visualize the high-dimensional state as a torch tensor.
        For high-dimensional states, we'll create a simple 2D projection visualization.

        Args:
            state: State tensor (B, T, state_dim) or (T, state_dim)
            trajectory_index: Which trajectory to visualize if batch
            img_size: Size of the output image (img_size x img_size)

        Returns:
            image_tensor: RGB image tensor of shape (B, T, 3, img_size, img_size) or (T, 3, img_size, img_size)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Handle different input shapes
        if len(state.shape) == 3:
            # Batched states (B, T, state_dim)
            B, T, _ = state.shape
            states = state.detach().cpu()
        elif len(state.shape) == 2:
            # Single trajectory (T, state_dim) - add batch dimension
            T, _ = state.shape
            B = 1
            states = state.detach().cpu().unsqueeze(0)  # (1, T, state_dim)
        else:
            raise ValueError(f"Invalid state shape: {state.shape}")

        # Create output tensor for all visualizations
        all_images = torch.zeros(B, T, 3, img_size, img_size, device=self.device)

        # Generate visualization for each batch and timestep
        for b in range(B):
            for t in range(T):
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                ax.set_aspect('equal')
                ax.axis('off')

                # White background
                ax.set_facecolor('white')

                # Project high-dimensional state to 2D for visualization
                # Use first two dimensions or compute a 2D projection
                current_state = states[b, t]  # (state_dim,)

                if self.state_dim >= 2:
                    # Use first two dimensions
                    x_coord = current_state[0].item()
                    y_coord = current_state[1].item()
                else:
                    # For 1D state, use state value as x and 0 as y
                    x_coord = current_state[0].item()
                    y_coord = 0.0

                # Visualize start position (if available)
                if hasattr(self, 'start_pos') and self.start_pos is not None:
                    start_proj = self.start_pos.cpu()
                    if len(start_proj) >= 2:
                        start_x, start_y = start_proj[0].item(), start_proj[1].item()
                    else:
                        start_x, start_y = start_proj[0].item(), 0.0
                    ax.plot(start_x, start_y, 'bo', markersize=10, label='Start', alpha=0.8)

                # Visualize goal position (if available)
                if hasattr(self, 'goal_pos') and self.goal_pos is not None:
                    goal_proj = self.goal_pos.cpu()
                    if len(goal_proj) >= 2:
                        goal_x, goal_y = goal_proj[0].item(), goal_proj[1].item()
                    else:
                        goal_x, goal_y = goal_proj[0].item(), 0.0
                    ax.plot(goal_x, goal_y, 'go', markersize=12, label='Goal', alpha=0.8)

                # Draw current state (black dot)
                ax.plot(x_coord, y_coord, 'ko', markersize=8, label='Current')

                # Add some visualization of the state magnitude
                state_norm = torch.norm(current_state).item()
                circle = plt.Circle((x_coord, y_coord), radius=min(state_norm * 0.1, 1.0),
                                  color='red', alpha=0.3)
                ax.add_patch(circle)

                # Add title with state info
                ax.set_title(f'State norm: {state_norm:.3f}', fontsize=8)

                # Convert matplotlib figure to tensor
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # Convert to tensor and normalize to [0, 1]
                img_tensor = torch.from_numpy(buf).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)

                # Resize if needed
                if img_tensor.shape[1] != img_size or img_tensor.shape[2] != img_size:
                    import torch.nn.functional as F
                    img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)

                all_images[b, t] = img_tensor.to(self.device)

                plt.close(fig)

        # Return appropriate shape based on input
        if len(state.shape) == 2:
            # Input was (T, state_dim), return (T, 3, img_size, img_size)
            return all_images.squeeze(0)  # Remove batch dimension
        else:
            # Input was (B, T, state_dim), return (B, T, 3, img_size, img_size)
            return all_images
