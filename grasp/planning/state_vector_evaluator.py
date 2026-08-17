"""
Base evaluator for environments that work with state vectors directly.
BallNavEvaluator and RandomMlpEvaluator both inherit from this class.
"""

import torch
import numpy as np
from utils import move_to_device
import matplotlib.pyplot as plt


class StateVectorEvaluator:
    """Base evaluator for environments with state vector observations."""

    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        save_images=True,
        goal_threshold=0.5,
        env_name="StateVector",
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.save_images = save_images
        self.goal_threshold = goal_threshold
        self.env_name = env_name

        # Get device from world model
        try:
            self.device = next(wm.parameters()).device
        except StopIteration:
            if hasattr(wm, 'device'):
                self.device = wm.device
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_traj_last(self, traj_data, length):
        """Get the last valid element for each trajectory."""
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(1)
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _extract_viz_frames(self, img_tensors):
        """Extract visualization frames from world model output.
        Override in subclasses if the world model returns different shapes."""
        return img_tensors

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        Evaluate actions for the environment.

        Args:
            actions: detached torch tensors on device (n_evals, horizon, action_dim)
            action_len: length of valid actions for each trajectory
            filename: base filename for saving outputs
            save_video: whether to save video / trajectory plots

        Returns:
            logs: dictionary of evaluation metrics
            successes: boolean array indicating success for each trajectory
            e_obses: environment observations (state vectors)
            e_states: environment states
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)

        trans_obs_0 = move_to_device(torch.tensor(self.obs_0), self.device)
        trans_obs_g = move_to_device(torch.tensor(self.obs_g), self.device)

        with torch.no_grad():
            wm_states, _ = self.wm.rollout(
                obs_0=trans_obs_0.squeeze(1),
                act=actions
            )

        wm_final_states = self._get_traj_last(wm_states, action_len + 1)[:, 0]

        exec_actions = actions.cpu().numpy()

        e_states_all = []
        e_final_states = []

        for i in range(n_evals):
            valid_len = int(action_len[i]) if action_len[i] != np.inf else actions.shape[1]
            traj_actions = exec_actions[i, :valid_len]

            init_state = self.state_0[i]
            states_traj, _ = self.wm.rollout(
                obs_0=torch.tensor(init_state, device=self.device).unsqueeze(0),
                act=torch.tensor(traj_actions, device=self.device).unsqueeze(0)
            )

            e_states_all.append(states_traj.squeeze(0).detach().cpu().numpy())
            e_final_states.append(states_traj[0, -1].detach().cpu().numpy())

        e_final_states = np.array(e_final_states)

        e_obses = e_states_all

        logs, successes = self._compute_rollout_metrics(
            e_final_states=e_final_states,
            wm_final_states=wm_final_states.cpu().numpy(),
        )

        if self.save_images and save_video:
            self._plot_trajectories(
                e_states_all=e_states_all,
                wm_states=wm_states.cpu().numpy(),
                successes=successes,
                filename=filename,
            )

        return logs, successes, e_obses, e_states_all

    def _compute_rollout_metrics(self, e_final_states, wm_final_states):
        """
        Compute evaluation metrics.

        Args:
            e_final_states: final states from environment rollout (n_evals, state_dim)
            wm_final_states: final states from world model rollout (n_evals, state_dim)

        Returns:
            logs: dictionary of metrics
            successes: boolean array of success indicators
        """
        goal_distances = np.linalg.norm(e_final_states - self.state_g, axis=1)
        successes = goal_distances < self.goal_threshold

        wm_errors = np.linalg.norm(e_final_states - wm_final_states, axis=1)

        logs = {
            "success_rate": np.mean(successes.astype(float)),
            "mean_goal_distance": np.mean(goal_distances),
            "mean_wm_error": np.mean(wm_errors),
            "min_goal_distance": np.min(goal_distances),
            "max_goal_distance": np.max(goal_distances),
        }

        print(f"{self.env_name} Success rate: {logs['success_rate']:.3f}")
        print(f"Mean goal distance: {logs['mean_goal_distance']:.3f}")
        print(f"Mean WM error: {logs['mean_wm_error']:.3f}")

        return logs, successes

    def _plot_trajectories(self, e_states_all, wm_states, successes, filename):
        """
        Plot trajectory comparisons using tensor-based visualization.
        Creates a grid showing 8 equally spaced timesteps from start to end.
        """
        n_plot = min(self.n_plot_samples, len(e_states_all))

        for i in range(n_plot):
            env_traj = e_states_all[i]
            traj_len = len(env_traj)

            state_tensor = torch.tensor(env_traj, device=self.device).unsqueeze(0)

            img_tensors = self._extract_viz_frames(
                self.wm.visualize(state_tensor, trajectory_index=0, img_size=256)
            )

            if traj_len > 8:
                timesteps = np.linspace(0, traj_len - 1, 8, dtype=int)
            else:
                timesteps = np.arange(traj_len)

            n_timesteps = len(timesteps)
            fig, axes = plt.subplots(1, n_timesteps, figsize=(3*n_timesteps, 3))

            if n_timesteps == 1:
                axes = [axes]

            for idx, t in enumerate(timesteps):
                ax = axes[idx]
                img_np = img_tensors[t].cpu().numpy().transpose(1, 2, 0)
                ax.imshow(img_np, origin='upper')
                ax.axis('off')
                ax.set_title(f't={t}', fontsize=12)

            success_str = "Success" if successes[i] else "Failure"
            goal_dist = np.linalg.norm(env_traj[-1] - self.state_g[i])
            fig.suptitle(f'Trajectory {i}: {success_str} - Goal distance: {goal_dist:.3f}',
                        fontsize=16, y=0.98)

            trajectory_filename = f'{filename}_traj_{i}_grid.png'
            plt.savefig(trajectory_filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved trajectory grid visualization to {trajectory_filename}")

        print(f"Saved {n_plot} trajectory visualizations")
