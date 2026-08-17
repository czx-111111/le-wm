"""
BallNav optimization visualization and GIF generation functionality.
Reuses code from dummy_control to create animated visualizations of the optimization process.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import imageio


class BallNavOptimizationVisualizer:
    """
    Visualization handler for ballnav optimization process.
    Supports GIF creation showing optimization progression with trajectory and virtual states.
    """

    def __init__(self, wm, save_gif=True, saved_folder="."):
        """
        Initialize the visualizer.

        Args:
            wm: BallNav world model instance
            save_gif: Whether to save GIF of optimization process
            saved_folder: Folder to save artifacts
        """
        self.wm = wm
        self.save_gif = save_gif
        self.saved_folder = saved_folder
        self.optimization_frames = []

    def save_frame_callback(self, frame_data):
        """Callback to save frames during optimization"""
        if self.save_gif:
            self.optimization_frames.append(frame_data)

    def create_optimization_gif(self, trajectory, gif_path="ballnav_optimization.gif"):
        """
        Create a GIF showing the optimization process for ballnav.
        Shows both rollout trajectory and virtual states with action arrows.

        Args:
            trajectory: Final executed trajectory (list of state arrays)
            gif_path: Path to save the GIF
        """
        if not self.save_gif or len(self.optimization_frames) == 0:
            print("No optimization frames to create GIF")
            return

        print("Creating BallNav optimization GIF...")

        # Create directory for temporary frames
        os.makedirs("frames", exist_ok=True)

        # Reduce number of frames for faster processing
        target_frames = 60
        if len(self.optimization_frames) < target_frames:
            # Duplicate frames to reach target
            frames_to_use = []
            for i in range(target_frames):
                idx = min(
                    i * len(self.optimization_frames) // target_frames,
                    len(self.optimization_frames) - 1,
                )
                frames_to_use.append(self.optimization_frames[idx])
        else:
            # Use first target_frames frames
            frames_to_use = self.optimization_frames[:target_frames]

        frame_paths = []

        for frame_idx, frame_data in enumerate(frames_to_use):
            # Extract frame data
            iteration = frame_data.get("iteration", frame_idx)
            states = frame_data["states"]  # Virtual states from optimization
            actions = frame_data["actions"]  # Actions from optimization
            start_state = frame_data["start_state"]
            goal_state = frame_data["goal_state"]

            # Create figure with side-by-side subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Left plot: World model rollout (current actions applied)
            self._plot_wm_rollout(ax1, start_state, actions, iteration)

            # Right plot: Virtual states with action arrows
            self._plot_virtual_states_with_actions(ax2, states, actions, start_state, goal_state, iteration)

            plt.tight_layout()

            # Save frame
            frames_dir = os.path.join(self.saved_folder, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches="tight")
            frame_paths.append(frame_path)
            plt.close()

            if frame_idx % 10 == 0:
                print(f"Generated frame {frame_idx + 1}/{len(frames_to_use)}")

        # Create GIF
        images = []
        for frame_path in frame_paths:
            images.append(imageio.imread(frame_path))

        # Calculate frame duration for 3 seconds total
        frame_duration = 3.0 / len(frames_to_use)

        # Create durations list: normal speed for all frames except last, then 1 second pause on last frame
        durations = [frame_duration] * (len(frames_to_use) - 1) + [1.0]

        # Save GIF with loop
        imageio.mimsave(gif_path, images, duration=durations, loop=0)

        print(f"GIF saved to {gif_path}")

        # Clean up temporary files
        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir("frames")
        print("Done!")

    def _plot_wm_rollout(self, ax, start_state, actions, iteration):
        """
        Plot the world model rollout trajectory on the left subplot.

        Args:
            ax: Matplotlib axis
            start_state: Starting state
            actions: Action sequence
            iteration: Current iteration number
        """
        # Simulate trajectory using current actions through the world model
        wm_trajectory = [start_state.detach().cpu()]
        current_state = start_state.unsqueeze(0)

        for action in actions:
            next_state = self.wm.simulate_one_step(current_state, action.unsqueeze(0))
            wm_trajectory.append(next_state.detach().cpu().squeeze(0))
            current_state = next_state

        wm_trajectory = torch.stack(wm_trajectory)

        # Set up the environment background
        ax.set_xlim(self.wm.x_min, self.wm.x_max)
        ax.set_ylim(self.wm.y_min, self.wm.y_max)
        ax.set_aspect("equal")

        # Draw obstacles
        for obstacle in self.wm.obstacles:
            obs_x, obs_y, obs_w, obs_h = obstacle.cpu()
            obstacle_rect = Rectangle(
                (obs_x - obs_w / 2, obs_y - obs_h / 2), obs_w, obs_h,
                color="red", alpha=0.5
            )
            ax.add_patch(obstacle_rect)

        # Draw start position
        start_pos_cpu = self.wm.start_pos.cpu()
        start_circle = Circle(
            (start_pos_cpu[0], start_pos_cpu[1]), radius=0.2, color="blue", alpha=0.7
        )
        ax.add_patch(start_circle)
        ax.text(start_pos_cpu[0], start_pos_cpu[1] + 0.5, "Start")

        # Draw goal position
        goal_pos_cpu = self.wm.goal_pos.cpu()
        goal_circle = Circle(
            (goal_pos_cpu[0], goal_pos_cpu[1]),
            radius=0.5,  # Use goal threshold if available
            color="green",
            alpha=0.7,
        )
        ax.add_patch(goal_circle)
        ax.text(goal_pos_cpu[0], goal_pos_cpu[1] + 0.5, "Goal")

        # Draw trajectory
        traj_x = wm_trajectory[:, 0].numpy()
        traj_y = wm_trajectory[:, 1].numpy()
        ax.plot(traj_x, traj_y, 'k-', linewidth=2, alpha=0.7, label="Trajectory")
        ax.plot(traj_x[0], traj_y[0], 'bo', markersize=8, label="Start")
        ax.plot(traj_x[-1], traj_y[-1], 'go', markersize=8, label="End")

        ax.grid(True, alpha=0.3)
        ax.set_title(f"WM Rollout (Iter {iteration})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    def _plot_virtual_states_with_actions(self, ax, states, actions, start_state, goal_state, iteration):
        """
        Plot the virtual states with red action arrows on the right subplot.

        Args:
            ax: Matplotlib axis
            states: Virtual states from optimization
            actions: Action sequence
            start_state: Starting state
            goal_state: Goal state
            iteration: Current iteration number
        """
        # Set up the environment background
        ax.set_xlim(self.wm.x_min, self.wm.x_max)
        ax.set_ylim(self.wm.y_min, self.wm.y_max)
        ax.set_aspect("equal")

        # Draw obstacles
        for obstacle in self.wm.obstacles:
            obs_x, obs_y, obs_w, obs_h = obstacle.cpu()
            obstacle_rect = Rectangle(
                (obs_x - obs_w / 2, obs_y - obs_h / 2), obs_w, obs_h,
                color="red", alpha=0.5
            )
            ax.add_patch(obstacle_rect)

        # Draw start position
        start_pos_cpu = self.wm.start_pos.cpu()
        start_circle = Circle(
            (start_pos_cpu[0], start_pos_cpu[1]), radius=0.2, color="blue", alpha=0.7
        )
        ax.add_patch(start_circle)
        ax.text(start_pos_cpu[0], start_pos_cpu[1] + 0.5, "Start")

        # Draw goal position
        goal_pos_cpu = self.wm.goal_pos.cpu()
        goal_circle = Circle(
            (goal_pos_cpu[0], goal_pos_cpu[1]),
            radius=0.5,
            color="green",
            alpha=0.7,
        )
        ax.add_patch(goal_circle)
        ax.text(goal_pos_cpu[0], goal_pos_cpu[1] + 0.5, "Goal")

        # Draw virtual states with gradient coloring (black to white)
        num_states = len(states)
        for i, state in enumerate(states):
            # Skip drawing if state contains NaN
            if torch.isnan(state).any():
                continue

            # Calculate color intensity (0 = black, 1 = white)
            color_intensity = i / (num_states - 1) if num_states > 1 else 0
            color = (color_intensity, color_intensity, color_intensity)  # RGB grayscale

            # Draw circle with black outline and gradient fill
            circle = Circle(
                (state[0].item(), state[1].item()),
                radius=0.1,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(circle)

        # Draw action arrows over each state dot (except the last state)
        # Actions correspond to states[:-1] (all states except the last)
        for i in range(min(len(states) - 1, len(actions))):
            state = states[i]
            action = actions[i]

            # Skip if state or action contains NaN
            if torch.isnan(state).any() or torch.isnan(action).any():
                continue

            # State position
            state_x, state_y = state[0].item(), state[1].item()

            # Action vector (scaled for visibility)
            action_scale = 0.3  # Scale factor to make arrows visible
            action_x, action_y = action[0].item() * action_scale, action[1].item() * action_scale

            # Draw red arrow indicating action from this state
            ax.arrow(
                state_x, state_y,  # Start position
                action_x, action_y,  # Action vector
                head_width=0.08,
                head_length=0.05,
                fc='red',
                ec='red',
                alpha=0.8,
                linewidth=1.5
            )

        ax.grid(True, alpha=0.3)
        ax.set_title(f"Virtual States + Actions (Iter {iteration})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")


def plot_ballnav_optimization_results(wm, trajectory, states, actions, loss_history):
    """
    Plot the results of ballnav optimization (static visualization).

    Args:
        wm: BallNav world model
        trajectory: Final executed trajectory
        states: Optimized virtual states
        actions: Optimized actions
        loss_history: Loss values during optimization
    """
    # Create a figure with 4 subplots (2x2 grid)
    fig = plt.figure(figsize=(18, 12))

    # 1. Plot actual trajectory
    ax1 = fig.add_subplot(221)
    _render_ballnav_trajectory(ax1, wm, trajectory, title="Actual Trajectory")

    # 2. Plot optimized virtual states
    ax2 = fig.add_subplot(222)
    _render_ballnav_states(ax2, wm, states, title="Optimized Virtual States")

    # 3. Plot optimized actions as vectors
    ax3 = fig.add_subplot(223)
    _render_ballnav_actions(ax3, wm, states, actions, title="Optimized Actions")

    # 4. Plot loss curves
    ax4 = fig.add_subplot(224)
    iterations = range(len(loss_history))
    ax4.plot(iterations, loss_history, "k-", label="Total Loss", linewidth=2)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Loss")
    ax4.set_title("Optimization Loss")
    ax4.set_yscale("log")  # Log scale for better visualization
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("ballnav_optimization_results.png", dpi=300, bbox_inches="tight")
    return fig


def _render_ballnav_trajectory(ax, wm, trajectory, title="Trajectory"):
    """Render ballnav trajectory on given axis."""
    # Set up environment
    ax.set_xlim(wm.x_min, wm.x_max)
    ax.set_ylim(wm.y_min, wm.y_max)
    ax.set_aspect("equal")

    # Draw obstacles
    for obstacle in wm.obstacles:
        obs_x, obs_y, obs_w, obs_h = obstacle.cpu()
        obstacle_rect = Rectangle(
            (obs_x - obs_w / 2, obs_y - obs_h / 2), obs_w, obs_h,
            color="red", alpha=0.5
        )
        ax.add_patch(obstacle_rect)

    # Draw start and goal
    start_pos_cpu = wm.start_pos.cpu()
    goal_pos_cpu = wm.goal_pos.cpu()

    start_circle = Circle((start_pos_cpu[0], start_pos_cpu[1]), radius=0.2, color="blue", alpha=0.7)
    ax.add_patch(start_circle)

    goal_circle = Circle((goal_pos_cpu[0], goal_pos_cpu[1]), radius=0.5, color="green", alpha=0.7)
    ax.add_patch(goal_circle)

    # Convert trajectory to numpy if needed
    if isinstance(trajectory, torch.Tensor):
        traj_array = trajectory.detach().cpu().numpy()
    elif isinstance(trajectory, list) and len(trajectory) > 0:
        if isinstance(trajectory[0], torch.Tensor):
            traj_array = torch.stack(trajectory).detach().cpu().numpy()
        else:
            traj_array = np.array(trajectory)
    else:
        traj_array = np.array(trajectory)

    # Plot trajectory
    if len(traj_array.shape) == 2:  # (T, 2)
        ax.plot(traj_array[:, 0], traj_array[:, 1], 'k-', linewidth=2, alpha=0.7)
        ax.plot(traj_array[0, 0], traj_array[0, 1], 'bo', markersize=8)
        ax.plot(traj_array[-1, 0], traj_array[-1, 1], 'go', markersize=8)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _render_ballnav_states(ax, wm, states, title="States"):
    """Render ballnav virtual states on given axis."""
    # Set up environment
    ax.set_xlim(wm.x_min, wm.x_max)
    ax.set_ylim(wm.y_min, wm.y_max)
    ax.set_aspect("equal")

    # Draw obstacles
    for obstacle in wm.obstacles:
        obs_x, obs_y, obs_w, obs_h = obstacle.cpu()
        obstacle_rect = Rectangle(
            (obs_x - obs_w / 2, obs_y - obs_h / 2), obs_w, obs_h,
            color="red", alpha=0.5
        )
        ax.add_patch(obstacle_rect)

    # Draw start and goal
    start_pos_cpu = wm.start_pos.cpu()
    goal_pos_cpu = wm.goal_pos.cpu()

    start_circle = Circle((start_pos_cpu[0], start_pos_cpu[1]), radius=0.2, color="blue", alpha=0.7)
    ax.add_patch(start_circle)

    goal_circle = Circle((goal_pos_cpu[0], goal_pos_cpu[1]), radius=0.5, color="green", alpha=0.7)
    ax.add_patch(goal_circle)

    # Draw virtual states with gradient coloring
    if states is not None:
        states_cpu = states.detach().cpu() if isinstance(states, torch.Tensor) else states
        num_states = len(states_cpu)

        for i, state in enumerate(states_cpu):
            if torch.is_tensor(state) and torch.isnan(state).any():
                continue

            # Calculate color intensity (0 = black, 1 = white)
            color_intensity = i / (num_states - 1) if num_states > 1 else 0
            color = (color_intensity, color_intensity, color_intensity)

            state_pos = state.numpy() if torch.is_tensor(state) else state
            circle = Circle(
                (state_pos[0], state_pos[1]),
                radius=0.1,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(circle)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _render_ballnav_actions(ax, wm, states, actions, title="Actions"):
    """Render ballnav actions as arrows on given axis."""
    # Set up environment
    ax.set_xlim(wm.x_min, wm.x_max)
    ax.set_ylim(wm.y_min, wm.y_max)
    ax.set_aspect("equal")

    # Draw virtual states as blue dots
    if states is not None:
        states_cpu = states.detach().cpu() if isinstance(states, torch.Tensor) else states
        state_x = states_cpu[:-1, 0].numpy() if torch.is_tensor(states_cpu) else states_cpu[:-1, 0]
        state_y = states_cpu[:-1, 1].numpy() if torch.is_tensor(states_cpu) else states_cpu[:-1, 1]
        ax.scatter(state_x, state_y, c="blue", s=20, alpha=0.5)

    # Plot actions as vectors
    if actions is not None and states is not None:
        actions_cpu = actions.detach().cpu() if isinstance(actions, torch.Tensor) else actions
        action_x = actions_cpu[:, 0].numpy() if torch.is_tensor(actions_cpu) else actions_cpu[:, 0]
        action_y = actions_cpu[:, 1].numpy() if torch.is_tensor(actions_cpu) else actions_cpu[:, 1]

        ax.quiver(
            state_x,
            state_y,
            action_x,
            action_y,
            color="red",
            angles="xy",
            scale_units="xy",
            scale=5,
        )

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
