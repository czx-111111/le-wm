"""
Simple world model implementation for ballnav environment.
This provides a differentiable world model that can be used with planners.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


class BallNavWorldModel(nn.Module):
    """
    World model for ballnav environment with differentiable dynamics.
    This provides the interface required by planners for planning in the 2D point mass environment.
    """

    def __init__(self, num_obstacles=1, device=None):
        super().__init__()

        # Environment parameters
        self.x_min, self.x_max = 0.0, 10.0
        self.y_min, self.y_max = 0.0, 10.0
        self.dt = 0.1
        self.max_vel = 1.0

        # Set device with fallback to CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Start and goal positions
        self.start_pos = torch.tensor([1.0, 5.0], device=self.device)
        self.goal_pos = torch.tensor([9.0, 5.0], device=self.device)

        # Generate obstacles
        self.num_obstacles = num_obstacles
        self.obstacles = self._generate_random_obstacles()

        # Smoothing parameters for differentiable operations
        self.collision_smoothing = 0.01
        self.boundary_smoothing = 0.1

        # Action dimension (2D velocity commands)
        self.action_dim = 2

        # Add a dummy parameter so that next(model.parameters()) works
        self.dummy_param = nn.Parameter(torch.zeros(1, device=self.device))

    def _generate_random_obstacles(self):
        """Generate multiple random rectangular obstacles between start and goal positions."""
        if self.num_obstacles == 0:
            return []

        obstacles = []

        # Calculate available space and spacing
        x_margin = 0.5
        available_x_space = self.goal_pos[0] - self.start_pos[0] - 2 * x_margin

        # Calculate maximum width per obstacle to ensure they don't touch
        min_gap = 0.2
        max_width_per_obstacle = (available_x_space - (self.num_obstacles - 1) * min_gap) / self.num_obstacles

        # Clamp max width to reasonable values
        max_width_per_obstacle = min(max_width_per_obstacle, 1)
        min_width = min(1.0, max_width_per_obstacle * 0.5)

        # Generate x positions evenly spaced across available space
        if self.num_obstacles == 1:
            x_centers = [self.start_pos[0] + x_margin + available_x_space / 2]
        else:
            segment_width = available_x_space / self.num_obstacles
            x_centers = [
                self.start_pos[0] + x_margin + segment_width * (i + 0.5)
                for i in range(self.num_obstacles)
            ]

        for i in range(self.num_obstacles):
            # Random width within constraints
            obs_w = torch.rand(1, device=self.device) * (max_width_per_obstacle - min_width) + min_width

            # Use the pre-calculated x position with small random offset
            x_offset = (torch.rand(1, device=self.device) - 0.5) * 0.3
            obs_x = x_centers[i] + x_offset.item()

            # Randomly choose whether obstacle touches floor (bottom) or ceiling (top)
            touches_floor = torch.rand(1, device=self.device) < 0.5

            if touches_floor:
                # Obstacle touches floor and extends upward to at least y=5
                min_height = 5.0
                max_height = 7.0
                obs_h = torch.rand(1, device=self.device) * (max_height - min_height) + min_height
                obs_y = obs_h / 2
            else:
                # Obstacle touches ceiling and extends downward to at least y=5
                min_height = 5.0
                max_height = 7.0
                obs_h = torch.rand(1, device=self.device) * (max_height - min_height) + min_height
                obs_y = self.y_max - obs_h / 2

            obstacle = torch.tensor([obs_x, obs_y, obs_w.item(), obs_h.item()], device=self.device)
            obstacles.append(obstacle)

        return obstacles

    def obstacle_distance(self, state):
        """
        Vectorized computation of distance to the nearest obstacle for batched states.
        Input:
            state: Tensor of shape (B, 2) or (B, T, 2)
        Output:
            distances: Tensor of shape (B,) or (B, T) with signed distances to nearest obstacle
        """
        # Handle both (B, 2) and (B, T, 2) shapes
        original_shape = state.shape
        if len(original_shape) == 3:
            B, T, _ = original_shape
            state = state.reshape(B * T, 2)
        else:
            B = original_shape[0]
            T = None

        if len(self.obstacles) == 0:
            # No obstacles, return large positive distances
            distances = torch.full((state.shape[0],), 1000.0, device=state.device)
        else:
            min_distances = []

            for obstacle in self.obstacles:
                obs_x, obs_y, obs_w, obs_h = obstacle
                x, y = state[:, 0:1], state[:, 1:2]

                dx = torch.clamp(torch.abs(x - obs_x) - obs_w / 2, min=0.0)
                dy = torch.clamp(torch.abs(y - obs_y) - obs_h / 2, min=0.0)

                # Compute distance outside the obstacle
                outside_dist = dx**2 + dy**2

                # Inside obstacle mask
                inside_mask = (dx == 0) & (dy == 0)

                # Penetration depth when inside
                penetration_x = obs_w / 2 - torch.abs(x - obs_x)
                penetration_y = obs_h / 2 - torch.abs(y - obs_y)
                inside_dist = -torch.min(penetration_x, penetration_y)

                # Combine for this obstacle
                obstacle_dist = torch.where(inside_mask, inside_dist, outside_dist)
                min_distances.append(obstacle_dist.squeeze(-1))

            # Return minimum distance to any obstacle
            all_distances = torch.stack(min_distances, dim=1)  # (B*T, num_obstacles)
            distances = torch.min(all_distances, dim=1)[0]  # (B*T,)

        # Reshape back to original shape
        if T is not None:
            distances = distances.reshape(B, T)

        return distances

    def differentiable_obstacle_avoidance(self, state):
        """
        Return a differentiable factor (0 to 1) indicating obstacle avoidance.
        0 means deep inside the obstacle, 1 means far from the obstacle.
        """
        distance = self.obstacle_distance(state)
        return torch.sigmoid(distance / self.collision_smoothing)

    def differentiable_boundary_avoidance(self, state):
        """
        Return a differentiable factor (0 to 1) for each boundary.
        0 means beyond the boundary, 1 means within bounds.
        """
        # Handle both (B, 2) and (B, T, 2) shapes
        if len(state.shape) == 3:
            x, y = state[..., 0:1], state[..., 1:2]
        else:
            x, y = state[:, 0:1], state[:, 1:2]

        # Compute smooth boundary factors
        x_min_factor = torch.sigmoid((x - self.x_min) / self.boundary_smoothing)
        x_max_factor = torch.sigmoid((self.x_max - x) / self.boundary_smoothing)
        y_min_factor = torch.sigmoid((y - self.y_min) / self.boundary_smoothing)
        y_max_factor = torch.sigmoid((self.y_max - y) / self.boundary_smoothing)

        # Combine boundary factors
        return x_min_factor * x_max_factor * y_min_factor * y_max_factor

    def next_state(self, state, action):
        """
        Compute next state from current state and current action.
        Works on batched tensors.

        Args:
            state: Current state tensor (B, T, 2) where T includes current and future states
            action: Action tensor (B, T-1, 2) for velocity commands

        Returns:
            next_states: Tensor (B, T-1, 2) of predicted next states
        """
        # Extract current states (all except last)
        current_states = state[:, :-1]  # (B, T-1, 2)

        # Soft action clipping - smoother than hard clipping
        clipped_action = torch.tanh(action) * self.max_vel

        # Predict the next state without collision or boundary handling
        next_state_raw = current_states + clipped_action * self.dt

        # Get differentiable factors for obstacle and boundary avoidance
        obstacle_factor = self.differentiable_obstacle_avoidance(next_state_raw)
        boundary_factor = self.differentiable_boundary_avoidance(next_state_raw)

        target_shape = [state.shape[0]] + [state.shape[1] - 1] + [1] * (state.dim() - 2)
        obstacle_factor = obstacle_factor.reshape(target_shape)
        boundary_factor = boundary_factor.reshape(target_shape)

        # Combine both factors
        total_factor = obstacle_factor * boundary_factor

        # Smoothly interpolate between staying at the current state (factor=0)
        # and moving to the next state (factor=1)
        next_states = current_states * (1 - total_factor) + next_state_raw * total_factor

        return next_states

    # same as next_state, but here convention is same num of states and actions
    def simulate_one_step(self, state, action):
        # Soft action clipping - smoother than hard clipping
        clipped_action = torch.tanh(action) * self.max_vel

        # Predict the next state without collision or boundary handling
        next_state_raw = state + clipped_action * self.dt

        # Get differentiable factors for obstacle and boundary avoidance
        obstacle_factor = self.differentiable_obstacle_avoidance(next_state_raw)
        boundary_factor = self.differentiable_boundary_avoidance(next_state_raw)

        # Ensure factors have correct shape for broadcasting
        # Reshape obstacle_factor and boundary_factor to match state dimensions for broadcasting
        # Add singleton dimensions after batch to match state
        target_shape = [state.shape[0]] + [1] * (state.dim() - 1)
        obstacle_factor = obstacle_factor.reshape(target_shape)
        boundary_factor = boundary_factor.reshape(target_shape)


        # Combine both factors
        total_factor = obstacle_factor * boundary_factor

        # Smoothly interpolate between staying at the current state (factor=0)
        # and moving to the next state (factor=1)
        next_state = state * (1 - total_factor) + next_state_raw * total_factor

        return next_state

    def rollout_encoded(self, obs_0, act):
        return self.rollout(obs_0[:,0], act)

    def rollout(self, obs_0, act):
        """
        Roll out all actions from initial state, output all states.

        Args:
            state_0: Initial state (B, 2)
            actions: Action sequence (B, T, 2)

        Returns:
            states: All states including initial state (B, T+1, 2)
            states_with_actions: All states with actions appended (B, T+1, 4)
        """
        batch_size, T, _ = act.shape

        # Build state sequence without in-place operations
        state_list = [obs_0]

        # Roll out step by step using simple dynamics
        current_state = obs_0
        for t in range(T):
            action = act[:, t]  # (B, 2)
            next_state = self.simulate_one_step(current_state, action)  # (B, 2)
            state_list.append(next_state)
            current_state = next_state

        # Stack all states to create (B, T+1, 2) tensor
        states = torch.stack(state_list, dim=1)

        # Prepare states_with_actions: (B, T+1, 4)
        # For each timestep, concatenate state and action; for last state, repeat last action
        # Build actions_padded without in-place operations
        actions_padded = torch.cat([act, act[:, -1:, :]], dim=1)  # (B, T+1, 2)

        states_with_actions = torch.cat([states, actions_padded], dim=-1)  # (B, T+1, 4)

        return states, states_with_actions

    def encode_observation_to_vector(self, state):
        """
        For us, this is the identity map since we work directly with 2D positions.

        Args:
            state: State tensor (B, 2) or (B, T, 2)

        Returns:
            encoded_state: Same as input (identity mapping)
        """
        return state

    # Alias so CEM/GD planners (which call encode_obs) also work.
    encode_obs = encode_observation_to_vector

    def visualize(self, state, trajectory_index=0, img_size=256):
        """
        Visualize the environment as a torch tensor.
        Simple visualization: start, goal, current state, and obstacles only.

        Args:
            state: State tensor (B, T, 2) or (T, 2)
            trajectory_index: Which trajectory to visualize if batch
            img_size: Size of the output image (img_size x img_size)

        Returns:
            image_tensor: RGB image tensor of shape (B, T, 3, img_size, img_size) or (T, 3, img_size, img_size)
        """
        import numpy as np

        # Handle different input shapes
        if len(state.shape) == 3:
            # Batched states (B, T, 2)
            B, T, _ = state.shape
            states = state.detach().cpu()
        elif len(state.shape) == 2:
            # Single trajectory (T, 2) - add batch dimension
            T, _ = state.shape
            B = 1
            states = state.detach().cpu().unsqueeze(0)  # (1, T, 2)
        else:
            raise ValueError(f"Invalid state shape: {state.shape}")

        # Create output tensor for all visualizations
        all_images = torch.zeros(B, T, 3, img_size, img_size, device=self.device)

        # Generate visualization for each batch and timestep
        for b in range(B):
            for t in range(T):
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylim(self.y_min, self.y_max)
                ax.set_aspect('equal')
                ax.axis('off')

                # White background
                ax.set_facecolor('white')

                # Draw obstacles (red rectangles)
                for obstacle in self.obstacles:
                    obs_x, obs_y, obs_w, obs_h = obstacle.cpu()
                    obstacle_rect = Rectangle(
                        (obs_x - obs_w / 2, obs_y - obs_h / 2), obs_w, obs_h,
                        color="red", alpha=0.8
                    )
                    ax.add_patch(obstacle_rect)

                # Draw start position (blue circle)
                start_x, start_y = self.start_pos.cpu()
                start_circle = Circle(
                    (start_x, start_y), radius=0.2, color="blue", alpha=0.8
                )
                ax.add_patch(start_circle)

                # Draw goal position (green circle)
                goal_x, goal_y = self.goal_pos.cpu()
                goal_circle = Circle(
                    (goal_x, goal_y), radius=0.5, color="green", alpha=0.8
                )
                ax.add_patch(goal_circle)

                # Draw current state (black dot)
                current_x, current_y = states[b, t, 0], states[b, t, 1]
                ax.plot(current_x, current_y, 'ko', markersize=8)

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
            # Input was (T, 2), return (T, 3, img_size, img_size)
            return all_images.squeeze(0)  # Remove batch dimension
        else:
            # Input was (B, T, 2), return (B, T, 3, img_size, img_size)
            return all_images
