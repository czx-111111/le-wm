"""
GRASP planner: gradient-based planning with virtual states.

The GRASP planner optimizes both actions and intermediate ``virtual`` latent
states in a lifted trajectory space, using stochastic state updates and
Lagrangian-style dynamics penalties to enforce consistency with the learned
world model. This module implements the core optimization logic and logging;
environment setup, world-model construction, and evaluation are handled by
``plan.py`` and the surrounding configuration files.
"""

import numpy as np
import torch
import torch.nn as nn
import time
from utils import move_to_device

from einops import rearrange

from .base_planner import BasePlanner

import wandb

import os

# Mixed precision imports
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    # Fallback for when CUDA is not available
    autocast = None
    GradScaler = None

# Create combined image with labels
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

# Import ballnav visualization
try:
    from .ballnav_visualization import BallNavOptimizationVisualizer
except ImportError:
    BallNavOptimizationVisualizer = None


class GRASPPlanner(BasePlanner):
    """Gradient-based planner operating in the world-model latent space.

    GRASP maintains a sequence of virtual latent states between the encoded
    start and goal observations and jointly optimizes these states together
    with the action sequence. By working in this lifted space, GRASP can
    parallelize world-model rollouts over trajectory segments while using
    stochastic state updates and Lagrangian-style penalties to encourage
    dynamics consistency.

    The many keyword arguments can be roughly grouped as:

    - optimization hyperparameters (learning rates, number of steps, schedulers),
    - regularization weights (gamma_* coefficients),
    - state exploration (init_noise_level, state_noise_scale),
    - refinement strategies (GD / CEM options),
    - numerical / implementation details (mixed precision, compilation, clamping).
    """
    def __init__(
        self,
        horizon,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        num_virtual_states=None,
        visualize_states=False,
        init_noise_level=0.3, # noise level for state initialization
        lr_s=0.01,  # Learning rate for state
        lr_a=0.1,  # Learning rate for actions. Typically lr_a > lr_s
        lr_lagrangian=0.001,
        gradclip_val = 0.0, # magnitude to clip gradient values. <= means no clipping
        opt_steps=30,  # Number of optimization steps
        eval_every=1,
        logging_prefix="plan_0",
        log_filename="logs.json",
        gamma_action=1.0,
        gamma_lagrangian=1.0,
        gamma_action_smooth=0.1,  # weight for action smoothness penalty
        gamma_state_uniform=0.0,  # weight for uniform state spacing penalty
        opt="adam",
        scheduler_type="none",
        rho_init=1.0,  # dynamics loss scaling (gamma_dynamics)
        lagrangian_update_threshold=1.0,  # max primal element threshold for lagrangian updates
        use_fixed_lagrangian=False,  # whether to use fixed lagrangian (ones) instead of learned
        saved_folder=".",  # folder to save artifacts
        init_actions_gaussian=False,  # whether to initialize actions from Gaussian distribution
        init_actions_mean=0.0,  # mean for Gaussian action initialization
        init_actions_std=0.1,  # standard deviation for Gaussian action initialization
        use_mixed_precision=True,  # whether to use mixed precision (bfloat16)
        compile_world_model=True,  # whether to compile the world model rollout_encoded method
        init_states_method="cylinder",  # initialization method: "cylinder" or "gaussian_decoder"
        state_noise_scale=0.0,  # global scale of noise to add to state proportional to dynamics loss
        state_noise_max=1.0,  # maximum absolute magnitude of the noise scale
        state_noise_proprio_only=False,  # if True, only add noise to proprioception part of state
        use_gd=False,  # whether to use GD refinement during optimization (alternative to CEM)
        gd_interval=50,  # how often to run GD refinement (every N iterations)
        gd_opt_steps=10,  # number of GD optimization steps per refinement
        gd_lr=0.1,  # learning rate for GD refinement
        gd_use_final_state_only=True,  # whether to use only final state loss (True) or all states (False)
        gd_accept_reject=True,  # whether to use accept/reject logic across GD steps
        gd_use_mppi=False,  # whether to use MPPI-style softmax weighted averaging instead of argmin selection
        mppi_temperature=1.0,  # temperature for MPPI softmax (lower = more greedy, higher = more uniform)
        cem_gamma_discount=0.99,  # exponential discount factor for CEM time weighting (gamma^t for timestep t)
        cem_variance_scale=0.5,  # initial variance scale for CEM action sampling
        cem_num_samples=20,  # number of samples per CEM iteration
        cem_max_rollouts_per_batch=None,  # maximum number of rollouts per batch for CEM (None = no batching)
        use_cem=False,  # whether to use CEM refinement during optimization
        cem_interval=50,  # how often to run CEM refinement (every N iterations)
        constant_noise_schedule=False,  # if True, keep noise_scale_factor=1 (no decay)
        gamma_decay=0.9,  # decay factor for dynamics loss time weighting
        enable_state_gradient=False,  # if True, don't detach state in rollout and skip extra dynamics loss
        action_clamp=0.0,  # clamp value for actions; 0 means no clamping
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        # Default num_virtual_states to horizon - 1 if not specified
        self.num_virtual_states = num_virtual_states
        self.visualize_states = visualize_states
        self.init_noise_level = init_noise_level
        self.lr_s = lr_s
        self.lr_a = lr_a
        self.lr_lagrangian = lr_lagrangian
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.gamma_action = gamma_action
        self.gamma_lagrangian = gamma_lagrangian
        self.gamma_action_smooth = gamma_action_smooth
        self.gamma_state_uniform = gamma_state_uniform
        self.gradclip_val = gradclip_val
        self.opt = opt
        self.scheduler_type = scheduler_type
        self.rho_init = rho_init
        self.lagrangian_update_threshold = lagrangian_update_threshold
        self.use_fixed_lagrangian = use_fixed_lagrangian
        self.saved_folder = saved_folder
        self.init_actions_gaussian = init_actions_gaussian
        self.init_actions_mean = init_actions_mean
        self.init_actions_std = init_actions_std
        self.use_mixed_precision = use_mixed_precision
        self.compile_world_model = compile_world_model
        self.init_states_method = init_states_method
        self.state_noise_scale = state_noise_scale
        self.state_noise_max = state_noise_max
        self.state_noise_proprio_only = state_noise_proprio_only

        self.use_gd = use_gd
        self.gd_interval = gd_interval
        self.gd_opt_steps = gd_opt_steps
        self.gd_lr = gd_lr
        self.gd_use_final_state_only = gd_use_final_state_only
        self.gd_accept_reject = gd_accept_reject
        self.gd_use_mppi = gd_use_mppi
        self.mppi_temperature = mppi_temperature
        self.num_parallel_trajectories = kwargs.get('num_parallel_trajectories', 1)
        self.gd_top_k = kwargs.get('gd_top_k', 1)  # number of top trajectories to optimize in GD
        self.gd_batch_size = kwargs.get('gd_batch_size', None)  # batch size for GD optimization (None = all at once)

        self.cem_gamma_discount = cem_gamma_discount
        self.cem_variance_scale = cem_variance_scale
        self.cem_num_samples = cem_num_samples
        self.cem_max_rollouts_per_batch = cem_max_rollouts_per_batch
        self.use_cem = use_cem
        self.cem_interval = cem_interval
        self.constant_noise_schedule = constant_noise_schedule
        self.gamma_decay = gamma_decay
        self.enable_state_gradient = enable_state_gradient
        self.action_clamp = action_clamp


        # Initialize upsampler for Gaussian state initialization
        self.upsampler = None

        # Initialize mixed precision scaler if enabled and available
        if self.use_mixed_precision and GradScaler is not None:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            if self.use_mixed_precision and GradScaler is None:
                print("Warning: Mixed precision requested but not available. Falling back to full precision.")
                self.use_mixed_precision = False

        # Initialize ballnav visualizer if available
        self.visualizer = None
        if BallNavOptimizationVisualizer is not None and hasattr(wm, 'x_min'):
            # Check if world model looks like ballnav
            self.visualizer = BallNavOptimizationVisualizer(wm, save_gif=True, saved_folder=self.saved_folder)

        # Apply torch.compile to rollout_encoded method if enabled
        if self.compile_world_model:
            try:
                self.wm.rollout_encoded = torch.compile(self.wm.rollout_encoded)
                print("Applied torch.compile to world model rollout_encoded method")
            except Exception as e:
                print(f"Warning: Failed to apply torch.compile to rollout_encoded: {e}")
                print("Continuing with regular rollout_encoded method")



    def _init_actions(self, z_obs_0: torch.Tensor, actions=None):
        """Initialize or extend the action sequence to match the horizon.

        Args:
            z_obs_0: Encoded initial observations with shape (B, ..., *latent_shape*).
            actions: Optional existing action tensor of shape (B, t, action_dim).

        Returns:
            Tensor of shape (B, H, action_dim), where H is the planner horizon.
        """
        n_evals = z_obs_0.shape[0]

        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.init_actions_gaussian:
                # Initialize from Gaussian distribution
                new_actions = torch.normal(
                    mean=self.init_actions_mean,
                    std=self.init_actions_std,
                    size=(n_evals, remaining_t, self.action_dim)
                ).to(device)
            else:
                # Initialize with zeros (original behavior)
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
            actions = torch.cat([actions, new_actions.to(device)], dim=1)

        return actions

    def _init_states(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        num_virtual_states: int = 1,
    ):
        """Initialize latent states between start and goal.

        Uses either a simple interpolation-based \"cylinder\" scheme or a
        Gaussian decoder-based scheme, depending on ``self.init_states_method``.

        Args:
            start_state: Starting state tensor.
            goal_state: Goal state tensor.
            num_virtual_states: Number of virtual states to initialize.

        Returns:
            Tuple ``(states, indices)`` where ``states`` has gradient tracking
            enabled and ``indices`` records any selection indices used during
            initialization.
        """
        if self.init_states_method == "gaussian_decoder":
            state_shape = start_state.shape[0], num_virtual_states, 32
            return self._init_states_gaussian(start_state, goal_state, num_virtual_states, state_shape)
        elif self.init_states_method == "gaussian":
            state_shape = start_state.shape[0], num_virtual_states, *start_state.shape[2:]
            return self._init_states_gaussian(start_state, goal_state, num_virtual_states, state_shape)
        else:  # default to cylinder method
            return self._init_states_cylinder(start_state, goal_state, num_virtual_states)

    def _init_states_cylinder(self, start_state, goal_state, num_virtual_states=1):
        """
        Initialize states as points along a cylinder from start to goal.

        Args:
            start_state: Starting state (torch.Tensor)
            goal_state: Goal state (torch.Tensor)
            num_virtual_states: Number of virtual states to initialize

        Returns:
            states: Initialized states tensor with gradient tracking
            indices: Indices of the selected states
        """
        # first two dims are batch, time
        state_dim = np.prod(start_state.shape[2:])

        # Create points along the line from start to goal
        t_shape = [1, self.horizon + 1] + [1] * (start_state.dim() - 2)
        t = torch.linspace(0, 1, self.horizon + 1).reshape(*t_shape).to(self.device)

        line_points = start_state + t * (goal_state - start_state)

        # Get direction vector and its norm
        direction = goal_state - start_state
        direction_norm = torch.norm(direction, dim=tuple(range(2, direction.dim())), keepdim=True)

        # Radius of the cylinder
        r = self.init_noise_level*direction_norm / np.sqrt(state_dim - 1)

        # Generate random points in the orthogonal hyperplane
        random_vectors = r * torch.randn_like(line_points).to(self.device)

        # Make vectors orthogonal to the main axis
        projections = (
            torch.sum(random_vectors * direction, dim=tuple(range(2, random_vectors.dim())), keepdim=True)
            * direction / (direction_norm**2)
        )
        orthogonal_vectors = random_vectors - projections

        # Combine line points with orthogonal vectors to get cylinder points
        states = line_points + orthogonal_vectors
        states = states.requires_grad_(True)

        indices = np.linspace(0, states.shape[1]-1, num_virtual_states + 2, dtype=int)[1:-1]
        # choose uniformally the states to optimize via num_virtual_states
        states = states[:,indices].detach().requires_grad_(True).to(self.device)
        return states, indices

    def _init_states_gaussian(self, start_state, goal_state, num_virtual_states=1, state_shape=None):
        """
        Initialize states using pure Gaussian noise with similar shape as start/goal states.

        Args:
            start_state: Starting state (torch.Tensor)
            goal_state: Goal state (torch.Tensor)
            num_virtual_states: Number of virtual states to initialize

        Returns:
            states: Initialized states tensor with gradient tracking
            indices: Indices of the selected states (dummy for Gaussian method)
        """

        # Create states with the same shape as start_state but with num_virtual_states in the time dimension
        # Generate pure Gaussian noise with the desired shape
        mid_point = (goal_state + start_state) / 2
        mu = mid_point.mean()
        mid_point = mid_point - mu
        sigma = mid_point.std()
        noisy_states = torch.randn(state_shape, device=self.device)

        # normalize both to zero mean
        # linear interpolation in standardized space
        z = np.sqrt(1 - self.init_noise_level**2) * mid_point / sigma + self.init_noise_level * noisy_states
        # rescale to target variance sigma and restore mean of X
        states = sigma * z + mu

        # Ensure states require gradients
        # intepolate with noisy states
        # states = (0.5 * mid_point + 0.5 * noisy_states).detach().requires_grad_(True)
        # For Gaussian initialization, we don't need specific indices since we're not sampling from a sequence
        # Return dummy indices
        return states.detach().requires_grad_(True), None

    def _expand_z_state(self, z_state):
        """
        Expand z_state based on the initialization method.

        Args:
            z_state: Input state tensor with shape [B, T, ...]

        Returns:
            z_state_expanded: Expanded state tensor with shape [B, T, H*W, C] for gaussian method,
                             or unchanged z_state for other methods
        """
        if self.init_states_method == "gaussian_decoder":
            # Reshape z_state from BxTx8x1x1 to B*Tx8x1x1 for upsampler
            B, T = z_state.shape[:2]
            z_state_reshaped = z_state.view(B * T, z_state.shape[2], 1, 1)
            # Upsample to B*TxCx16x16
            z_state_upsampled = self.upsampler(z_state_reshaped)
            # Resize from 16x16 to 14x14 using bilinear interpolation
            z_state_resized = torch.nn.functional.interpolate(z_state_upsampled, size=(14, 14), mode='bilinear', align_corners=False)
            # Reshape to BxTxCx14x14 then permute to BxTxH*WxC
            z_state_reshaped = z_state_resized.view(B, T, z_state_resized.shape[1], 14, 14)
            # Permute from BxTxCxHxW to BxTxH*WxC
            z_state_expanded = z_state_reshaped.permute(0, 1, 3, 4, 2).contiguous().view(B, T, 14*14, z_state_resized.shape[1])
        else:
            z_state_expanded = z_state

        return z_state_expanded

    def _cem_planning(self, z_obs_0, frozen_z_state, num_samples=20, num_iterations=20, initial_actions=None):
        """
        Cross-Entropy Method (CEM) for planning with batched sampling for GPU memory efficiency.

        Args:
            z_obs_0: Initial observation
            frozen_z_state: Frozen virtual states optimized by the GRASP planner
            num_samples: Number of samples per CEM iteration
            num_iterations: Number of CEM iterations
            initial_actions: Initial action sequence (used to initialize mean)

        Returns:
            Optimized actions
        """
        n_evals = z_obs_0.shape[0]
        device = z_obs_0.device

        # Initialize mean and std for action distribution
        if initial_actions is not None:
            action_mean = initial_actions.clone()
        else:
            action_mean = torch.zeros(n_evals, self.horizon, self.action_dim, device=device)

        action_std = torch.ones(n_evals, self.horizon, self.action_dim, device=device) * self.cem_variance_scale

        # Split frozen_z_state into visual and proprio parts
        # Assuming the last dimension is proprio (you may need to adjust this)
        visual_dim = frozen_z_state.shape[-1] - self.wm.proprio_dim
        frozen_visual = frozen_z_state[..., :visual_dim]
        frozen_proprio = frozen_z_state[..., visual_dim:]

        for iter_idx in range(num_iterations):
            # Sample actions from current distribution
            action_samples = action_mean.unsqueeze(1) + action_std.unsqueeze(1) * torch.randn(
                n_evals, num_samples, self.horizon, self.action_dim, device=device
            )
            # action_samples = action_samples.clamp(-3.0, 3.0)

            # Determine batch size for processing
            if self.cem_max_rollouts_per_batch is not None and self.cem_max_rollouts_per_batch < num_samples:
                # Process in batches
                batch_size = self.cem_max_rollouts_per_batch
                num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

                # Initialize lists to collect results from batches
                all_rolled_visual = []
                all_rolled_proprio = []

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    batch_num_samples = end_idx - start_idx

                    # Get batch of action samples
                    action_samples_batch = action_samples[:, start_idx:end_idx]

                    # Reshape for batch rollout: [n_evals * batch_num_samples, horizon, action_dim]
                    action_samples_flat = action_samples_batch.reshape(-1, self.horizon, self.action_dim)

                    # Expand initial observation for all samples in this batch
                    z_obs_0_expanded = z_obs_0.expand(-1, batch_num_samples, -1, -1).reshape(-1, *z_obs_0.shape[1:])

                    # Rollout with world model
                    _, rolled_out_stateactions = self.wm.rollout_encoded(z_obs_0_expanded, action_samples_flat)
                    rolled_out_states = rolled_out_stateactions[..., :-self.wm.action_dim]

                    # Keep only states w.r.t. virtual states
                    rolled_out_states = rolled_out_states[:, ::(int)(self.horizon / (self.num_virtual_states+1))]

                    # Reshape back to [n_evals, batch_num_samples, horizon+1, ...]
                    rolled_out_states = rolled_out_states.view(n_evals, batch_num_samples, rolled_out_states.shape[1], *rolled_out_states.shape[2:])

                    # Split rolled out states into visual and proprio
                    rolled_visual_batch = rolled_out_states[..., :visual_dim]
                    rolled_proprio_batch = rolled_out_states[..., visual_dim:]

                    # Collect results
                    all_rolled_visual.append(rolled_visual_batch)
                    all_rolled_proprio.append(rolled_proprio_batch)

                # Concatenate all batches along the num_samples dimension
                rolled_visual = torch.cat(all_rolled_visual, dim=1)
                rolled_proprio = torch.cat(all_rolled_proprio, dim=1)

            else:
                # No batching - process all samples at once (original behavior)
                # Reshape for batch rollout: [n_evals * num_samples, horizon, action_dim]
                action_samples_flat = action_samples.view(-1, self.horizon, self.action_dim)

                # Expand initial observation for all samples
                z_obs_0_expanded = z_obs_0.expand(-1, num_samples, -1, -1).reshape(-1, *z_obs_0.shape[1:])

                # Rollout with world model
                _, rolled_out_stateactions = self.wm.rollout_encoded(z_obs_0_expanded, action_samples_flat)
                rolled_out_states = rolled_out_stateactions[..., :-self.wm.action_dim]
                # Keep only states w.r.t. virtual states
                rolled_out_states = rolled_out_states[:, ::(int)(self.horizon / (self.num_virtual_states+1))]

                # Reshape back to [n_evals, num_samples, horizon+1, ...]
                rolled_out_states = rolled_out_states.view(n_evals, num_samples, rolled_out_states.shape[1], *rolled_out_states.shape[2:])

                # Split rolled out states into visual and proprio
                rolled_visual = rolled_out_states[..., :visual_dim]
                rolled_proprio = rolled_out_states[..., visual_dim:]

            # Compute loss: distance to frozen_z_state (even-weighted between visual and proprio)
            # frozen_z_state shape: [n_evals, num_states, H*W, C]
            # rolled_out_states shape: [n_evals, num_samples, horizon+1, H*W, C]

            # Compute losses for all time steps (dimension 2)
            visual_loss_all_steps = ((rolled_visual - frozen_visual.unsqueeze(1)) ** 2).mean(dim=(3, 4))  # [n_evals, num_samples, horizon+1]
            proprio_loss_all_steps = ((rolled_proprio - frozen_proprio.unsqueeze(1)) ** 2).mean(dim=(3, 4))  # [n_evals, num_samples, horizon+1]

            # # Create exponential time-weighting: [gamma, gamma^2, gamma^3, ..., gamma^T]
            # timesteps = visual_loss_all_steps.shape[2]  # horizon+1
            # gamma = self.cem_gamma_discount
            # time_weights = torch.pow(gamma, torch.arange(0, timesteps, device=visual_loss_all_steps.device))
            # # Reshape to [1, 1, timesteps] for broadcasting
            # time_weights = time_weights.view(1, 1, -1)

            # Apply exponential time-weighting and normalize by sum of weights
            visual_loss = (visual_loss_all_steps[:,:,[-1]]).sum(dim=2) # [n_evals, num_samples]
            proprio_loss = (proprio_loss_all_steps[:,:,[-1]]).sum(dim=2)  # [n_evals, num_samples]

            # Even-weighted loss
            total_loss = 0.5 * visual_loss + 0.5 * proprio_loss  # [n_evals, num_samples]

            # Select elite samples (top 10%)
            num_elites = max(1, num_samples // 20)
            elite_indices = torch.topk(total_loss, num_elites, dim=1, largest=False).indices

            # Gather elite actions
            elite_actions = torch.gather(
                action_samples,
                1,
                elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.action_dim)
            )

            # Update mean and std
            action_mean = elite_actions.mean(dim=1)
            action_std = elite_actions.std(dim=1) + 1e-6  # Add small epsilon to prevent collapse

        return action_mean

    def _select_best_trajectory(self, z_state, actions, z_obs_0, z_obs_g, n_evals_original, n_parallel, top_k=1):
        """
        Select the top K trajectories per original eval based on dynamics loss.
        
        Args:
            z_state: States tensor [n_evals_original * n_parallel, num_virtual_states, ...]
            actions: Actions tensor [n_evals_original * n_parallel, horizon, action_dim]
            z_obs_0: Initial observation [n_evals_original * n_parallel, 1, ...]
            z_obs_g: Goal observation [n_evals_original * n_parallel, 1, ...]
            n_evals_original: Original batch size before expansion
            n_parallel: Number of parallel trajectories
            top_k: Number of top trajectories to select (default 1)
            
        Returns:
            z_state_best: Best states [n_evals_original * top_k, num_virtual_states, ...]
            actions_best: Best actions [n_evals_original * top_k, horizon, action_dim]
            best_indices: Indices of best trajectories [n_evals_original, top_k]
            dynamics_losses: All dynamics losses [n_evals_original, n_parallel]
        """
        with torch.no_grad():
            # Reshape to separate parallel dimension
            z_state_reshaped = z_state.view(n_evals_original, n_parallel, *z_state.shape[1:])
            actions_reshaped = actions.view(n_evals_original, n_parallel, *actions.shape[1:])
            z_obs_0_reshaped = z_obs_0.view(n_evals_original, n_parallel, *z_obs_0.shape[1:])
            z_obs_g_reshaped = z_obs_g.view(n_evals_original, n_parallel, *z_obs_g.shape[1:])
            
            # Expand states with start and goal
            z_state_expanded = torch.cat([z_obs_0, z_state, z_obs_g], dim=1)
            
            # Compute next state predictions using world model (parallelized sub-rollouts)
            action_samples_flat = actions.view(-1, self.horizon // (self.num_virtual_states + 1), self.action_dim)
            z0_virtual = z_state_expanded[:, :-1].flatten(0, 1).unsqueeze(1)
            _, i_z_obses = self.wm.rollout_encoded(z0_virtual.detach(), action_samples_flat)
            f_next_state = i_z_obses[:, [-1]][..., :-self.wm.action_dim]
            f_next_state = f_next_state.view(n_evals_original * n_parallel, self.num_virtual_states + 1, *z_state_expanded.shape[2:])
            # Dynamics loss per trajectory: mean over all dimensions except batch
            dynamics_loss = (f_next_state[:,[-1]] - z_state_expanded[:, [-1]]).pow(2).mean(dim=tuple(range(1, f_next_state.dim())))
            
            # Reshape to [n_evals_original, n_parallel]
            dynamics_loss_reshaped = dynamics_loss.view(n_evals_original, n_parallel)
            
            # Clamp top_k to not exceed n_parallel
            top_k = min(top_k, n_parallel)
            
            # Select top K (smallest losses) per original eval
            _, best_indices = dynamics_loss_reshaped.topk(top_k, dim=1, largest=False)  # [n_evals_original, top_k]
            
            # Gather top K trajectories
            batch_indices = torch.arange(n_evals_original, device=z_state.device).unsqueeze(1).expand(-1, top_k)
            z_state_topk = z_state_reshaped[batch_indices, best_indices]  # [n_evals_original, top_k, ...]
            actions_topk = actions_reshaped[batch_indices, best_indices]  # [n_evals_original, top_k, ...]
            
            # Flatten to [n_evals_original * top_k, ...]
            z_state_best = z_state_topk.view(n_evals_original * top_k, *z_state.shape[1:])
            actions_best = actions_topk.view(n_evals_original * top_k, *actions.shape[1:])
            
            return z_state_best, actions_best, best_indices, dynamics_loss_reshaped

    def _select_best_from_topk(self, actions, rolled_out_states, frozen_z_state,
                                initial_distances_per_timestep, n_evals_original, top_k, device):
        """
        Select the best trajectory from each group of top_k trajectories.
        
        When gd_use_mppi is False (default): selects the single best (argmin) trajectory.
        When gd_use_mppi is True: computes softmax-weighted average over trajectories based on
        negative losses scaled by mppi_temperature.
        
        Args:
            actions: All optimized actions [n_evals_original * top_k, horizon, action_dim]
            rolled_out_states: Rolled out states [n_evals_original * top_k, num_states, ...]
            frozen_z_state: Target frozen states [n_evals_original * top_k, num_states, ...]
            initial_distances_per_timestep: Initial distances [n_evals_original * top_k, num_states] or None
            n_evals_original: Original batch size
            top_k: Number of trajectories per original eval
            device: Device to use
            
        Returns:
            Selected actions, states, and distances for the best trajectory per original eval
        """
        with torch.no_grad():
            # Compute per-trajectory loss using rolled_out_states and frozen_z_state
            visual_dim = frozen_z_state.shape[-1] - self.wm.proprio_dim
            rolled_visual = rolled_out_states[..., :visual_dim]
            rolled_proprio = rolled_out_states[..., visual_dim:]
            frozen_visual = frozen_z_state[..., :visual_dim]
            frozen_proprio = frozen_z_state[..., visual_dim:]
            
            visual_loss_all_steps = ((rolled_visual - frozen_visual) ** 2).mean(dim=(2, 3))
            proprio_loss_all_steps = ((rolled_proprio - frozen_proprio) ** 2).mean(dim=(2, 3))
            
            if self.gd_use_final_state_only:
                per_traj_loss = 0.5 * visual_loss_all_steps[:, -1] + 0.5 * proprio_loss_all_steps[:, -1]
            else:
                per_traj_loss = 0.5 * visual_loss_all_steps.mean(dim=1) + 0.5 * proprio_loss_all_steps.mean(dim=1)
            
            # Reshape losses and tensors
            per_traj_loss_reshaped = per_traj_loss.view(n_evals_original, top_k)
            actions_reshaped = actions.view(n_evals_original, top_k, *actions.shape[1:])
            rolled_out_states_reshaped = rolled_out_states.view(n_evals_original, top_k, *rolled_out_states.shape[1:])
            
            if self.gd_use_mppi:
                # MPPI-style: compute softmax weights over negative losses
                # softmax(-loss / temperature) gives higher weight to lower loss trajectories
                weights = torch.softmax(-per_traj_loss_reshaped / self.mppi_temperature, dim=1)  # [n_evals_original, top_k]
                
                # Weighted average of actions
                # weights: [n_evals_original, top_k] -> [n_evals_original, top_k, 1, 1]
                weights_actions = weights.view(n_evals_original, top_k, *([1] * (actions_reshaped.dim() - 2)))
                actions_best = (weights_actions * actions_reshaped).sum(dim=1)  # [n_evals_original, horizon, action_dim]
                
                # Weighted average of rolled out states
                weights_states = weights.view(n_evals_original, top_k, *([1] * (rolled_out_states_reshaped.dim() - 2)))
                rolled_out_states_best = (weights_states * rolled_out_states_reshaped).sum(dim=1)
                
                # Weighted average of initial distances
                initial_distances_best = None
                if initial_distances_per_timestep is not None:
                    initial_distances_reshaped = initial_distances_per_timestep.view(
                        n_evals_original, top_k, *initial_distances_per_timestep.shape[1:])
                    weights_distances = weights.view(n_evals_original, top_k, *([1] * (initial_distances_reshaped.dim() - 2)))
                    initial_distances_best = (weights_distances * initial_distances_reshaped).sum(dim=1)
                
                # Compute weighted average loss for reporting
                final_loss = (weights * per_traj_loss_reshaped).sum(dim=1).mean().item()
            else:
                # Original behavior: select argmin
                best_indices = per_traj_loss_reshaped.argmin(dim=1)
                batch_indices = torch.arange(n_evals_original, device=device)
                
                actions_best = actions_reshaped[batch_indices, best_indices]
                rolled_out_states_best = rolled_out_states_reshaped[batch_indices, best_indices]
                
                initial_distances_best = None
                if initial_distances_per_timestep is not None:
                    initial_distances_reshaped = initial_distances_per_timestep.view(
                        n_evals_original, top_k, *initial_distances_per_timestep.shape[1:])
                    initial_distances_best = initial_distances_reshaped[batch_indices, best_indices]
                
                # Compute loss for selected trajectories
                final_loss = per_traj_loss_reshaped[batch_indices, best_indices].mean().item()
            
            return actions_best, rolled_out_states_best, initial_distances_best, final_loss

    def _gd_planning(self, z_obs_0, frozen_z_state, num_iterations=10, lr=0.1, initial_actions=None,
                     n_evals_original=None, top_k=1):
        """
        Gradient Descent (GD) based planning with frozen virtual states.
        Rolls out from the initial state z_obs_0, similar to CEM.
        
        When top_k > 1, optimizes multiple trajectories in parallel and returns
        only the single best trajectory per original eval at the end.

        Args:
            z_obs_0: Initial observation [n_evals_original * top_k, 1, ...]
            frozen_z_state: Frozen virtual states optimized by the GRASP planner [n_evals_original * top_k, ...]
            num_iterations: Number of GD optimization steps
            lr: Learning rate for GD optimization
            initial_actions: Initial action sequence [n_evals_original * top_k, horizon, action_dim]
            n_evals_original: Original batch size before top_k expansion (required if top_k > 1)
            top_k: Number of trajectories being optimized per original eval

        Returns:
            Tuple of (optimized_actions, rolled_out_states, initial_loss, final_loss, initial_distances_per_timestep)
            When top_k > 1, actions and states are reduced to [n_evals_original, ...] by selecting the best
        """
        n_evals = z_obs_0.shape[0]
        device = z_obs_0.device

        # If gd_batch_size is set, process in batches and concatenate results
        if self.gd_batch_size is not None and n_evals > self.gd_batch_size:
            all_actions = []
            all_states = []
            all_initial_distances = []
            total_initial_loss = 0.0
            total_final_loss = 0.0
            
            for start_idx in range(0, n_evals, self.gd_batch_size):
                end_idx = min(start_idx + self.gd_batch_size, n_evals)
                batch_size = end_idx - start_idx
                
                # Slice inputs for this batch
                z_obs_0_batch = z_obs_0[start_idx:end_idx]
                frozen_z_state_batch = frozen_z_state[start_idx:end_idx]
                initial_actions_batch = initial_actions[start_idx:end_idx] if initial_actions is not None else None
                
                # For batched processing, we don't use top_k selection within batches
                # (that would be handled after all batches are processed)
                actions_batch, states_batch, init_loss, final_loss, init_distances = self._gd_planning_single_batch(
                    z_obs_0_batch, frozen_z_state_batch, num_iterations, lr, initial_actions_batch
                )
                
                all_actions.append(actions_batch)
                all_states.append(states_batch)
                if init_distances is not None:
                    all_initial_distances.append(init_distances)
                total_initial_loss += init_loss * batch_size
                total_final_loss += final_loss * batch_size
            
            # Concatenate all batches
            actions_opt = torch.cat(all_actions, dim=0)
            rolled_out_states = torch.cat(all_states, dim=0)
            initial_loss = total_initial_loss / n_evals
            final_loss = total_final_loss / n_evals
            initial_distances_per_timestep = torch.cat(all_initial_distances, dim=0) if all_initial_distances else None
            
            # Handle top_k selection after batched processing (same as non-batched)
            if top_k > 1 and n_evals_original is not None:
                actions_best, rolled_out_states_best, initial_distances_best, best_final_loss = self._select_best_from_topk(
                    actions_opt, rolled_out_states, frozen_z_state,
                    initial_distances_per_timestep, n_evals_original, top_k, device
                )
                return actions_best, rolled_out_states_best, initial_loss, best_final_loss, initial_distances_best
            
            return actions_opt, rolled_out_states, initial_loss, final_loss, initial_distances_per_timestep
        
        # Non-batched processing (original behavior)
        return self._gd_planning_single_batch(
            z_obs_0, frozen_z_state, num_iterations, lr, initial_actions,
            n_evals_original, top_k
        )

    def _gd_planning_single_batch(self, z_obs_0, frozen_z_state, num_iterations=10, lr=0.1, initial_actions=None,
                                   n_evals_original=None, top_k=1):
        """
        Single batch GD planning (called by _gd_planning for each batch or for the full batch).
        """
        n_evals = z_obs_0.shape[0]
        device = z_obs_0.device

        # Initialize actions from initial_actions or zeros
        if initial_actions is not None:
            actions_opt = initial_actions.clone().detach().requires_grad_(True)
        else:
            actions_opt = torch.zeros(n_evals, self.horizon, self.action_dim, device=device, requires_grad=True)

        # Create optimizer for actions
        optimizer = torch.optim.Adam([actions_opt], lr=lr)

        # Split frozen_z_state into visual and proprio parts
        visual_dim = frozen_z_state.shape[-1] - self.wm.proprio_dim
        frozen_visual = frozen_z_state[..., :visual_dim]
        frozen_proprio = frozen_z_state[..., visual_dim:]

        initial_loss = None
        final_loss = None
        initial_distances_per_timestep = None  # Store per-timestep distances from first iteration
        initial_per_traj_loss = None  # Store per-trajectory initial loss for best selection

        for iter_idx in range(num_iterations):
            optimizer.zero_grad()

            # Use autocast for mixed precision if enabled
            if self.use_mixed_precision and autocast is not None:
                autocast_context = autocast(enabled=True, dtype=torch.bfloat16)
            else:
                autocast_context = torch.enable_grad()

            with autocast_context:
                # Rollout from initial state z_obs_0 (same as CEM)
                _, rolled_out_stateactions = self.wm.rollout_encoded(z_obs_0, actions_opt)
                rolled_out_states = rolled_out_stateactions[..., :-self.wm.action_dim]

                # Keep only states at virtual state positions (same as CEM)
                rolled_out_states = rolled_out_states[:, ::(int)(self.horizon / (self.num_virtual_states+1))]

                # Split rolled out states into visual and proprio
                rolled_visual = rolled_out_states[..., :visual_dim]
                rolled_proprio = rolled_out_states[..., visual_dim:]

                # Compute losses for all time steps
                visual_loss_all_steps = ((rolled_visual - frozen_visual) ** 2).mean(dim=(2, 3))  # [n_evals, num_states]
                proprio_loss_all_steps = ((rolled_proprio - frozen_proprio) ** 2).mean(dim=(2, 3))  # [n_evals, num_states]

                # Save per-timestep distances on the first iteration
                if iter_idx == 0:
                    # Compute L2 distance per timestep for all batch elements
                    # Shape: [n_evals, num_states]
                    initial_distances_per_timestep = (0.5 * visual_loss_all_steps + 0.5 * proprio_loss_all_steps).detach()
                    
                    # Also save per-trajectory loss for later best selection comparison
                    if self.gd_use_final_state_only:
                        initial_per_traj_loss = (0.5 * visual_loss_all_steps[:, -1] + 0.5 * proprio_loss_all_steps[:, -1]).detach()
                    else:
                        initial_per_traj_loss = (0.5 * visual_loss_all_steps.mean(dim=1) + 0.5 * proprio_loss_all_steps.mean(dim=1)).detach()

                # Use either final state only or all states based on configuration
                if self.gd_use_final_state_only:
                    # Use only the final state loss
                    visual_loss = visual_loss_all_steps[:, -1].mean()
                    proprio_loss = proprio_loss_all_steps[:, -1].mean()
                else:
                    # Use all states
                    visual_loss = visual_loss_all_steps.mean()
                    proprio_loss = proprio_loss_all_steps.mean()

                # Even-weighted loss
                total_loss = 0.5 * visual_loss + 0.5 * proprio_loss

            # Track initial loss
            if iter_idx == 0:
                initial_loss = total_loss.item()

            # Backward pass
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            # Gradient clipping
            if actions_opt.grad is not None:
                max_action_grad = actions_opt.grad.abs().max()
                if max_action_grad > 2:
                    actions_opt.grad.data /= max_action_grad

            # Optimizer step
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            # Clamp actions to valid range
            with torch.no_grad():
                if self.action_clamp > 0:
                    actions_opt.data.clamp_(-self.action_clamp, self.action_clamp)

        # Track final loss
        final_loss = total_loss.item()

        # If top_k > 1, select the single best trajectory per original eval
        if top_k > 1 and n_evals_original is not None:
            with torch.no_grad():
                # Compute per-trajectory loss (not averaged across batch)
                if self.gd_use_final_state_only:
                    per_traj_loss = 0.5 * visual_loss_all_steps[:, -1] + 0.5 * proprio_loss_all_steps[:, -1]  # [n_evals]
                else:
                    per_traj_loss = 0.5 * visual_loss_all_steps.mean(dim=1) + 0.5 * proprio_loss_all_steps.mean(dim=1)  # [n_evals]
                
                # Reshape to [n_evals_original, top_k]
                per_traj_loss_reshaped = per_traj_loss.view(n_evals_original, top_k)
                
                # Select best (argmin) per original eval
                best_indices = per_traj_loss_reshaped.argmin(dim=1)  # [n_evals_original]
                
                # Reshape actions and states to [n_evals_original, top_k, ...]
                actions_reshaped = actions_opt.detach().view(n_evals_original, top_k, *actions_opt.shape[1:])
                rolled_out_states_reshaped = rolled_out_states.detach().view(n_evals_original, top_k, *rolled_out_states.shape[1:])
                
                # Also reshape initial_distances_per_timestep
                if initial_distances_per_timestep is not None:
                    initial_distances_reshaped = initial_distances_per_timestep.view(n_evals_original, top_k, *initial_distances_per_timestep.shape[1:])
                
                # Gather best trajectories
                batch_indices = torch.arange(n_evals_original, device=device)
                actions_opt = actions_reshaped[batch_indices, best_indices]  # [n_evals_original, horizon, action_dim]
                rolled_out_states = rolled_out_states_reshaped[batch_indices, best_indices]  # [n_evals_original, ...]
                
                if initial_distances_per_timestep is not None:
                    initial_distances_per_timestep = initial_distances_reshaped[batch_indices, best_indices]
                
                # Get initial and final loss for the selected best trajectories
                best_initial_loss = initial_per_traj_loss.view(n_evals_original, top_k)[batch_indices, best_indices].mean().item()
                best_final_loss = per_traj_loss_reshaped[batch_indices, best_indices].mean().item()
                
                return actions_opt, rolled_out_states, best_initial_loss, best_final_loss, initial_distances_per_timestep

        del optimizer
        if actions_opt.grad is not None:
            actions_opt.grad = None
        torch.cuda.empty_cache()

        return actions_opt.detach(), rolled_out_states.detach(), initial_loss, final_loss, initial_distances_per_timestep

    def compute_loss(
        self, z_state, z_obs_g, f_next_state, actions, lambda_lagrangian=None, gamma_dynamics=1.0
    ):
        """
        Compute the optimization loss.

        Args:
            z_state: Sequence of states
            z_obs_g: Goal state
            f_next_state: Predicted next states from dynamics model
            actions: Sequence of actions
            lambda_lagrangian: Lagrangian multipliers (optional)
            gamma_dynamics: Scaling factor for dynamics loss (default: 1.0)

        Returns:
            loss components and total loss
        """

        # ASSERTS

        # Validate that horizon is divisible by num_virtual_states + 1, for parallelizing
        # WM sub-rollouts
        if self.horizon % (self.num_virtual_states + 1) != 0:
            raise ValueError(
                f"horizon ({self.horizon}) must be divisible by (num_virtual_states + 1) "
                f"({self.num_virtual_states + 1}), but {self.horizon} % {self.num_virtual_states + 1} = "
                f"{self.horizon % (self.num_virtual_states + 1)}"
            )

        dim_s = np.prod(z_state.shape[2:])
        dim_a = np.prod(actions.shape[2:])
        # constant expected norm, Xavier-type init

        # Goal loss: distance between final state and goal state
        # f_next_state_list dim: B, N, H*W, C
        # goal_loss = self.gamma_goal*((f_next_state[:, [-1], ...] - z_obs_g) ** 2).sum(dim=-1).mean()
        goal_loss=torch.tensor([0], device=self.device)
        dynamics_loss = gamma_dynamics*(f_next_state - z_state[:,1:,...]).pow(2).sum(dim=-1).mean()
        dynamics_loss_last = (f_next_state[:,-1] - z_state[:,-1]).pow(2).mean()
        # Geometric scaling: weight each timestep by gamma^(T-i) where i is the time index
        if not self.enable_state_gradient:
            T = f_next_state.shape[1]
            time_weights = self.gamma_decay ** torch.arange(T, 0, -1, device=f_next_state.device).view(1, T, *([1] * (f_next_state.dim() - 2)))
            dynamics_loss += gamma_dynamics*(time_weights * (f_next_state - z_state[:,[-1]].detach()).pow(2)).sum(dim=-1)[:,:-1].mean()

        action_mag_loss = self.gamma_action*(actions**2).sum(dim=-1).mean()

        action_diff = actions[:, 1:] - actions[:, :-1]
        action_smooth_loss = self.gamma_action_smooth*(action_diff**2).sum(dim=-1).mean()

        if z_state.shape[1] > 2:  # Need at least 3 states to compute spacing differences
            state_embeddings = z_state.mean(dim=2)  # [time, seq, features] - average over spatial dimensions
            state_distances = torch.norm(state_embeddings[:, 1:] - state_embeddings[:, :-1], dim=-1)
            if len(state_distances.shape) > 1:
                mean_distance = state_distances.mean(dim=-1, keepdim=True)
                state_uniform_loss = self.gamma_state_uniform*((state_distances - mean_distance)**2).sum(dim=-1).mean()
            else:
                state_uniform_loss = torch.tensor([0], device=self.device)
        else:
            state_uniform_loss = torch.tensor([0], device=self.device)

        # Lagrangian-style term for enforcing dynamics consistency
        if lambda_lagrangian is not None:
            lagrangian = self.gamma_lagrangian*(lambda_lagrangian*((f_next_state - z_state[:,1:,...])[:,:-1])).sum(dim=-1).mean()
        else:
            lagrangian = torch.tensor([0], device=self.device)

        # Total loss, dimension regularized
        total_loss = (goal_loss + dynamics_loss + action_mag_loss + action_smooth_loss + state_uniform_loss + lagrangian)


        return {
            "loss": total_loss,
            "goal_loss": goal_loss,
            "dynamics_loss": dynamics_loss,
            "dynamics_loss_last": dynamics_loss_last,
            "action_mag_loss": action_mag_loss,
            "action_smooth_loss": action_smooth_loss,
            "state_uniform_loss": state_uniform_loss,
            "lagrangian": lagrangian,
            "total_loss_plot": total_loss,
        }

    def plan(self, obs_0, obs_g, actions=None):
        """Optimize a sequence of actions and latent states between start and goal.

        Args:
            obs_0: Dictionary of initial observations (e.g., visual and proprio).
            obs_g: Dictionary of goal observations.
            actions: Optional initial action sequence; if ``None``, a sequence of
                length ``horizon`` is initialized internally.

        Returns:
            Tuple ``(actions, action_len)`` where ``actions`` is a tensor of
            optimized actions and ``action_len`` is the effective length used
            by the evaluator.
        """

        self.num_virtual_states = self.num_virtual_states if self.num_virtual_states != 'all' else self.horizon - 1

        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(obs_0, obs_g, encode_to_vector=True)

        # Track original batch size before parallel expansion
        n_evals_original = z_obs_0.shape[0]
        n_parallel = self.num_parallel_trajectories
        
        # Expand for parallel trajectories
        if n_parallel > 1:
            z_obs_0 = z_obs_0.repeat_interleave(n_parallel, dim=0)
            z_obs_g = z_obs_g.repeat_interleave(n_parallel, dim=0)
            # Also expand trans_obs_0 if it's a tensor (needed for visualization)
            if isinstance(trans_obs_0, torch.Tensor):
                trans_obs_0 = trans_obs_0.repeat_interleave(n_parallel, dim=0)

        z_state, _ = self._init_states(z_obs_0, z_obs_g, num_virtual_states=self.num_virtual_states)

        if self.init_states_method == "gaussian_decoder" and self.upsampler is None:
            # Upsampler network: BxTx8x1x1 -> BxTxCx14x14
            C = z_obs_0.shape[3]  # Channel dimension
            self.upsampler = nn.Sequential(
                # First expand to 2x2
                nn.ConvTranspose2d(32, 64, kernel_size=4, stride=1, padding=0),  # 8x1x1 -> 64x2x2
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Then upsample to 4x4
                nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x2x2 -> 32x4x4
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # Then upsample to 8x8
                nn.ConvTranspose2d(128, C, kernel_size=4, stride=2, padding=1),  # 32x4x4 -> 16x8x8
            ).to(self.device)

            # Initialize weights properly for DCGAN-style decoder
            def init_weights(m):
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight.data, 1)
                    nn.init.constant_(m.bias.data, 0)

            self.upsampler.apply(init_weights)

        actions = self._init_actions(z_obs_0, actions).to(self.device)
        actions.requires_grad = True
        n_evals = actions.shape[0]

        # for tracking lagrangian progress
        # actions = torch.nn.Parameter((torch.randn(z_obs_0.shape[0], H, self.wm.action_dim)).to(z_obs_0), requires_grad=True)

        if self.opt == "adam":
            opt = torch.optim.Adam
        elif self.opt == "sgd":
            opt = torch.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer: {self.opt}")
        if self.init_states_method == "gaussian_decoder":
            optimizer = opt([
                {"params": z_state, "lr": self.lr_s},
                {"params": self.upsampler.parameters(), "lr": self.lr_s},
                {"params": actions, "lr": self.lr_a}
            ])
        else:
            optimizer = opt([
                {"params": z_state, "lr": self.lr_s},
                {"params": actions, "lr": self.lr_a}
            ])

        if self.use_fixed_lagrangian:
            lambda_lagrangian = torch.ones((z_state.shape[0], z_state.shape[1],*z_obs_0.shape[2:])).to(self.device)
        else:
            # Create Lagrangian as a parameter that can be optimized
            lambda_lagrangian = torch.nn.Parameter(
                torch.zeros((z_state.shape[0], z_state.shape[1],*z_obs_0.shape[2:])).to(self.device),
                requires_grad=False
            )


        gd_initial_distances = torch.ones((z_state.shape[0], z_state.shape[1]+2)).to(self.device)


        gamma_dynamics = self.rho_init

        # Create scheduler for the single optimizer
        scheduler = None
        if self.scheduler_type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.opt_steps
            )
        elif self.scheduler_type != 'none':
            raise ValueError




        # z_state = torch.cat([z_obs_0, z_state, z_obs_g], dim=1)
        print("opt steps", self.opt_steps)
        if self.use_mixed_precision:
            print("Using mixed precision (bfloat16) for optimization")

        # Frame saving for ballnav GIF generation
        frame_interval = max(1, self.opt_steps // 60) if self.visualizer else 0

        # initialize measuring time for planning here

        # Main optimization loop

        total_measured_time = 0.0
        exit_step = self.opt_steps  # Default to max steps if no early exit

        # Track previous best GD loss, state, and actions for accept/reject across GD steps
        prev_best_gd_loss = float('inf')
        prev_best_z_state = None
        prev_best_actions = None

        for iter_idx in range(self.opt_steps):
            start = time.perf_counter()

            # Use mutable containers to capture values from closure
            closure_outputs = {}

            def closure():
                # Zero gradients inside closure
                optimizer.zero_grad()

                # Use autocast for mixed precision if enabled and available
                if self.use_mixed_precision and autocast is not None:
                    autocast_context = autocast(enabled=True, dtype=torch.bfloat16)
                else:
                    autocast_context = torch.enable_grad()  # Dummy context manager

                with autocast_context:
                    # Generate multiple samples from a Gaussian centered at actions
                    # Shape: [num_samples, batch_size, horizon, action_dim]
                    action_samples_flat = actions
                    z_state_expanded = self._expand_z_state(z_state)

                    # Use the batched next state prediction method from the world model

                    action_samples_flat = action_samples_flat.view(-1, self.horizon // (self.num_virtual_states + 1), self.action_dim)
                    z_state_expanded = torch.cat([z_obs_0, z_state_expanded, z_obs_g], dim=1)
                    z0_virtual = z_state_expanded[:, :-1].flatten(0, 1).unsqueeze(1)


                    z0_input = z0_virtual if self.enable_state_gradient else z0_virtual.detach()

                    _, i_z_obses = self.wm.rollout_encoded(
                        z0_input,
                        action_samples_flat
                    )

                    f_next_state = i_z_obses[:, [-1]][..., : -self.wm.action_dim].view(n_evals, self.num_virtual_states+1, *z_state_expanded.shape[2:])

                    # _, i_z_nextnext = self.wm.rollout_encoded(
                    #    f_next_state[:,:-1].flatten(0, 1).unsqueeze(1),
                    #    action_samples_flat[1:].detach()
                    # )

                    # f_nextnext_state = i_z_nextnext[:, [-1]][..., : -self.wm.action_dim].view(n_evals, self.num_virtual_states, *z_state_expanded.shape[2:])

                    z_obs_g_expanded = z_obs_g

                    loss_dict = self.compute_loss(z_state_expanded, z_obs_g_expanded, f_next_state, actions, lambda_lagrangian.detach(), gamma_dynamics)
                    loss = loss_dict["loss"]


                # Scale loss for mixed precision if enabled
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Scale gradients so max element is 1 (separately for states and actions)
                if self.use_mixed_precision and self.scaler is not None:
                    # Unscale gradients before clipping when using mixed precision
                    self.scaler.unscale_(optimizer)

                # if z_state.grad is not None:
                #     max_state_grad = z_state.grad.abs().max()
                #     if max_state_grad > 1.0:
                #         z_state.grad.data /= max_state_grad

                if actions.grad is not None:
                    # Compute max over action dimension for each (batch, time) element
                    # actions.grad shape: [batch_size, horizon, action_dim]
                    max_action_grad = actions.grad.abs().amax(dim=-1, keepdim=True)  # [batch_size, horizon, 1]
                    # Scale down each (i,j) element independently where max > 1.0
                    scale = torch.where(max_action_grad > 1, max_action_grad, torch.ones_like(max_action_grad))
                    actions.grad.data /= scale

                # if iter_idx > 301:
                    # actions.grad.data.zero_()
                    # z_state.grad.data.zero_()
                    # self.state_noise_scale = 0.0
                    # self.gd_lr=0.01

                # if self.gradclip_val > 0:
                #     if self.use_mixed_precision and self.scaler is not None:
                #         # Unscale gradients before clipping when using mixed precision
                #         self.scaler.unscale_(optimizer)
                #     torch.nn.utils.clip_grad_value_(z_state, clip_value=self.gradclip_val)
                #     torch.nn.utils.clip_grad_value_(actions, clip_value=self.gradclip_val)

                # Store outputs in mutable container for access outside closure
                closure_outputs['loss_dict'] = loss_dict
                closure_outputs['f_next_state_list'] = f_next_state
                closure_outputs['z_state_expanded'] = z_state_expanded

                return loss


            if self.use_mixed_precision and self.scaler is not None:
                # Use scaler for mixed precision optimization
                # Call closure to compute gradients with scaling
                closure()
                if self.use_cem and iter_idx % self.cem_interval < 5:
                    actions.grad.data.zero_()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard optimization
                optimizer.step(closure)


            # Extract values computed in closure (no recomputation needed!)
            loss_dict = closure_outputs['loss_dict']
            f_next_state_list = closure_outputs['f_next_state_list']
            z_state_expanded = closure_outputs['z_state_expanded']


            # Calculate constraint tensor from already computed values
            constraint_tensor = f_next_state_list - z_state_expanded[:,1:]
            warmup_steps = 0  # lagrangian updates start from first iteration


            # lambda updated with applying constraint on new states/actions, so equivalently
            # do local update before compute gd step here
            # residual adaptive update, with stepsize proportional to
            # dynamics loss weight
            with torch.no_grad():
                # Compute state-wise primal norms as squared norm with mean over dimensions 2 and later
                # This leaves batch and time dimensions intact
                primal_norms_statewise = (constraint_tensor[:,:-1].pow(2).mean(dim=tuple(range(2, constraint_tensor.dim()))))

                if iter_idx > warmup_steps:
                    # Create a mask for where lagrangian updates should occur
                    # Update lagrangian where primal norm is below threshold
                    update_mask = primal_norms_statewise <= self.lagrangian_update_threshold

                    lagrangian_stepsize = self.lr_lagrangian * gamma_dynamics
                    # Apply update only where mask is True
                    lambda_lagrangian[update_mask] += lagrangian_stepsize * constraint_tensor[:,:-1][update_mask]

                    # Zero lagrangian only at entries where primal norm exceeds threshold
                    zero_mask = primal_norms_statewise > 2*self.lagrangian_update_threshold
                    lambda_lagrangian[zero_mask] = 0.0

                # clamp lagrangian
                lambda_lagrangian.data.clamp_(min=-10, max=10)


                if self.state_noise_scale > 0:
                    # Linearly decay noise scale to 0 over each gd_interval (0 when GD step runs)
                    if self.constant_noise_schedule:
                        noise_scale_factor = 1.0
                    elif self.use_gd and self.gd_interval > 1:
                        position_in_interval = iter_idx % self.gd_interval
                        noise_scale_factor = 1.0 - position_in_interval / (self.gd_interval - 1)
                    else:
                        noise_scale_factor = 1.0
                    current_noise_scale = self.state_noise_scale * noise_scale_factor

                    # Compute vector norm over all dimensions except batch and time
                    constraint_norm = (constraint_tensor ** 2).sum(dim=tuple(range(2, z_state.dim()))).sqrt()

                    # Compute dimensionality for normalization
                    state_dim = np.prod(z_state.shape[2:])

                    # Normalize by sqrt(d) so expected noise norm equals constraint_norm when state_noise_scale=1
                    if self.state_noise_max > 0:
                        # noise determined by error ahead,
                        # want to explore prev. state to get to # better new state
                        noise_scale = torch.clamp(
                            current_noise_scale * gd_initial_distances[:,2:],
                            max=self.state_noise_max
                        )
                    else:
                        noise_scale = current_noise_scale * constraint_norm / np.sqrt(state_dim)

                    # Add dimensions to match z_state shape
                    noise_scale = noise_scale.view(*noise_scale.shape, *([1] * (z_state.dim() - 2)))

                    # Generate random noise with same shape as z_state
                    # noise = torch.randn_like(z_state) * noise_scale
                    noise = torch.randn_like(z_state) * current_noise_scale

                    # If state_noise_proprio_only is True, zero out all but the proprioception part
                    if self.state_noise_proprio_only:
                        noise[..., :-self.wm.proprio_dim] = 0

                    # Apply noise to the optimized states
                    z_state.data += noise

            # Calculate gradient norms for logging
            z_state_grad_norm = torch.norm(z_state.grad[:,1:-1].detach(), p=2).item() if z_state.grad is not None else 0.0
            actions_grad_norm = torch.norm(actions.grad.detach(), p=2).item() if actions.grad is not None else 0.0

            # Calculate per-action gradient norms (for each time step i)
            if actions.grad is not None:
                # actions.grad shape: [batch_size, horizon, action_dim]
                # Compute norm over batch and action_dim dimensions for each time step
                per_action_grad_norms = torch.norm(actions.grad.detach(), p=2, dim=(0, 2))  # shape: [horizon]
            else:
                per_action_grad_norms = torch.zeros(self.horizon, device=self.device)


            # Step scheduler if needed
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                # Save frame for ballnav GIF generation
                if (self.visualizer and
                    (iter_idx % frame_interval == 0 or iter_idx == self.opt_steps - 1)):
                    frame_data = {
                        "iteration": iter_idx,
                        "states": z_state_expanded.detach().clone()[0],  # Use first batch element
                        "actions": actions.detach().clone()[0],  # Use first batch element
                        "start_state": z_obs_0[0, 0],  # Remove time dimension
                        "goal_state": z_obs_g[0, 0],   # Remove time dimension
                    }
                    self.visualizer.save_frame_callback(frame_data)

                # lagrangian_history[:,iter_idx % self.explore_interval] = loss_dict["lagrangian_batch"]


                if self.action_clamp > 0:
                    actions.data.clamp_(-self.action_clamp, self.action_clamp)
            # GD refinement step (if enabled, alternative to CEM)
            # Uses internal trust region scheduler for adaptive LR and step rejection
            if self.use_gd and iter_idx % self.gd_interval == 0 and iter_idx > 0:
                # If using parallel trajectories, select the top K before GD
                if z_state.grad is not None:
                    z_state.grad = None
                if actions.grad is not None:
                    actions.grad = None
                torch.cuda.empty_cache()
                if n_parallel > 1:
                    top_k = min(self.gd_top_k, n_parallel)
                    z_state_topk, actions_topk, best_indices, parallel_dyn_losses = self._select_best_trajectory(
                        z_state, actions, z_obs_0, z_obs_g, n_evals_original, n_parallel, top_k=top_k
                    )
                    
                    # Log parallel trajectory selection info
                    self.wandb_run.log({
                        f"{self.logging_prefix}/parallel_best_dyn_loss": parallel_dyn_losses.min(dim=1).values.mean().item(),
                        f"{self.logging_prefix}/parallel_worst_dyn_loss": parallel_dyn_losses.max(dim=1).values.mean().item(),
                        f"{self.logging_prefix}/parallel_mean_dyn_loss": parallel_dyn_losses.mean().item(),
                        "step": iter_idx + 1,
                    })
                    
                    # Prepare z_obs_0 and z_obs_g for top_k trajectories
                    # They need to match the shape [n_evals_original * top_k, ...]
                    z_obs_0_topk = z_obs_0.view(n_evals_original, n_parallel, *z_obs_0.shape[1:])[:, :top_k].reshape(
                        n_evals_original * top_k, *z_obs_0.shape[1:])
                    z_obs_g_topk = z_obs_g.view(n_evals_original, n_parallel, *z_obs_g.shape[1:])[:, :top_k].reshape(
                        n_evals_original * top_k, *z_obs_g.shape[1:])
                    
                    frozen_z_state = torch.cat([z_obs_0_topk, z_state_topk, z_obs_g_topk], dim=1).detach().clone()
                    gd_actions, gd_rolled_out_states, gd_initial_loss, gd_final_loss, gd_initial_distances = self._gd_planning(
                        z_obs_0=z_obs_0_topk,
                        frozen_z_state=frozen_z_state,
                        num_iterations=self.gd_opt_steps,
                        lr=self.gd_lr,
                        initial_actions=actions_topk.detach(),
                        n_evals_original=n_evals_original,
                        top_k=top_k
                    )
                    # gd_actions is now [n_evals_original, ...] - the single best per original eval
                    # gd_initial_loss and gd_final_loss are now for the selected best trajectory
                else:
                    frozen_z_state = torch.cat([z_obs_0, z_state, z_obs_g], dim=1).detach().clone()
                    gd_actions, gd_rolled_out_states, gd_initial_loss, gd_final_loss, gd_initial_distances = self._gd_planning(
                        z_obs_0=z_obs_0,
                        frozen_z_state=frozen_z_state,
                        num_iterations=self.gd_opt_steps,
                        lr=self.gd_lr,
                        initial_actions=actions.detach()
                    )
                lambda_lagrangian.data = torch.zeros_like(lambda_lagrangian, device=self.device)

                # Apply accept/reject based on loss improvement (same logic for all cases)
                if gd_final_loss < gd_initial_loss:
                    if n_parallel > 1:
                        # Copy best trajectory to all parallel slots
                        z_state_best = gd_rolled_out_states[:, 1:-1]  # Remove start/goal states
                        z_state.data = z_state_best.repeat_interleave(n_parallel, dim=0)
                        actions.data = gd_actions.repeat_interleave(n_parallel, dim=0)
                    else:
                        actions.data = gd_actions.data
                        # Update z_state to the rolled out states from GD (excluding z_obs_0 and z_obs_g)
                        # Only do this when not using gaussian_decoder, since that method uses a compressed z_state format
                        if self.init_states_method != "gaussian_decoder":
                            z_state.data = gd_rolled_out_states[:, 1:-1].data
                    gd_curr_loss = gd_final_loss
                else:
                    # Loss increased, halve the GD learning rate and don't update actions/states
                    self.gd_lr *= 0.5
                    gd_curr_loss = gd_initial_loss

                # Accept/reject across GD steps: compare with previous best
                if self.gd_accept_reject:
                    if gd_curr_loss <= prev_best_gd_loss:
                        # Accept: update previous best state and actions
                        prev_best_gd_loss = gd_curr_loss
                        prev_best_z_state = z_state.data.clone()
                        prev_best_actions = actions.data.clone()
                    elif prev_best_z_state is not None:
                        # Reject: reset state and actions to previous best
                        print('REJECTED')
                        z_state.data = prev_best_z_state.clone()
                        actions.data = prev_best_actions.clone()
            if self.use_cem and iter_idx % self.cem_interval == (self.cem_interval - 1):
                with torch.no_grad():
                    frozen_z_state = z_state_expanded.detach().clone()
                    actions.data = self._cem_planning(
                        z_obs_0=z_obs_0,
                        frozen_z_state=frozen_z_state,
                        num_samples=self.cem_num_samples,
                        num_iterations=3,
                        initial_actions=actions.detach()
                    ).data

            end = time.perf_counter()
            total_measured_time += end - start


            if self.use_gd and iter_idx % self.gd_interval == 0 and iter_idx > 0:
                # Adaptive noise scale based on whether GD improved within this step
                # if gd_final_loss < gd_initial_loss:
                #     self.state_noise_scale *= 0.7  # GD helped, decrease noise
                #     actions.data = gd_actions.data  # accept update
                # else:
                #     self.state_noise_scale *= 1.3  # GD didn't help, increase noise
                    # reject action update

                # Create per-timestep distance plot for GD initial state
                gd_logs = {
                    f"{self.logging_prefix}/gd_loss": gd_final_loss,
                    f"{self.logging_prefix}/gd_lr": self.gd_lr,
                    f"{self.logging_prefix}/state_noise_scale": self.state_noise_scale,
                    "step": iter_idx + 1,
                }
                
                if self.evaluator.save_images and gd_initial_distances is not None:
                    # gd_initial_distances shape: [n_evals, num_states], plot first batch element
                    num_timesteps = gd_initial_distances.shape[1]
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
                    ax_dist.bar(range(num_timesteps), gd_initial_distances[0].cpu().numpy())
                    ax_dist.set_xlabel('Virtual State Index (time step)')
                    ax_dist.set_ylabel('Distance (MSE)')
                    ax_dist.set_title(f'Rollout-to-virtual state distances (iter {iter_idx})')
                    ax_dist.set_xticks(range(0, num_timesteps, max(1, num_timesteps // 10)))
                    plt.tight_layout()

                    # Convert to wandb image
                    fig_dist.canvas.draw()
                    dist_buf = np.frombuffer(fig_dist.canvas.tostring_rgb(), dtype=np.uint8)
                    dist_buf = dist_buf.reshape(fig_dist.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig_dist)

                    gd_logs[f"{self.logging_prefix}/gd_initial_distances"] = wandb.Image(
                        dist_buf,
                        caption=f"Rollout-to-virtual state distances"
                    )

                # Log GD losses, learning rate, noise scale, and distances to wandb
                self.wandb_run.log(gd_logs)
                # print(gd_final_loss)
                
                    # self.state_noise_scale=gd_initial_loss*0.7
                # z_state.data = gd_states[:, 1:-1].data

                # lambda_lagrangian.data = torch.zeros_like(lambda_lagrangian, device=self.device)


            # Evaluate current plan
            if self.evaluator is not None and iter_idx % self.eval_every == 0:
                # Only evaluate the first copy of each original eval (not all parallel copies)
                if n_parallel > 1:
                    actions_to_eval = actions.detach().view(n_evals_original, n_parallel, *actions.shape[1:])[:, 0]
                else:
                    actions_to_eval = actions.detach()

                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions_to_eval,
                    filename=f"{self.logging_prefix}_output_{iter_idx+1}",
                )

                # Add visualization of optimized states (every 50 steps)
                if self.evaluator.save_images and self.visualize_states and iter_idx % 50 == 0:
                    with torch.no_grad():
                        # Separate visual embeddings from proprioception
                        # Log evenly spaced frames from the trajectory, including first and last
                        if z_state_expanded.shape[1] <= 8:
                            # If 10 or fewer frames, include all of them
                            frame_indices = range(z_state_expanded.shape[1])
                        else:
                            # Otherwise, select evenly spaced frames including first and last
                            frame_indices = np.linspace(0, z_state_expanded.shape[1]-1, 8, dtype=int)

                        # Only plot the first example in the batch
                        z_state_virtual = z_state_expanded.detach().clone()[[0]]
                        obs_decoded_virtual = self.wm.visualize(z_state_virtual[:,frame_indices])

                        images_virtual = []
                        for t in range(len(frame_indices)):
                            img_to_display = obs_decoded_virtual[0, t].permute(1, 2, 0).cpu().numpy()
                            # Normalize to [0, 1]
                            img_to_display = (np.clip(img_to_display,-1,1) + 1)/2
                            images_virtual.append(img_to_display)

                        # Decode next_pred (predicted next states from dynamics model) for the first example
                        z_state_next_pred = z_state_expanded.detach().clone()[[0]]
                        z_state_next_pred[:,1:] = f_next_state_list[[0]]
                        obs_decoded_next_pred = self.wm.visualize(z_state_next_pred[:,frame_indices])

                        images_next_pred = []
                        for t in range(len(frame_indices)):
                            img_to_display = obs_decoded_next_pred[0, t].permute(1, 2, 0).cpu().numpy()
                            # Normalize to [0, 1]
                            img_to_display = (np.clip(img_to_display,-1,1) + 1)/2
                            images_next_pred.append(img_to_display)

                        # Decode WM rollout (actual world model rollout from initial state) for the first example
                        # Determine whether to pass a dict or tensor to rollout
                        if isinstance(trans_obs_0, dict):
                            obs_0_for_rollout = trans_obs_0
                            actions_rollout = actions.detach()
                        else:
                            obs_0_for_rollout = trans_obs_0[[0]]
                            actions_rollout = actions.detach()[[0]]

                        _, rolled_out_stateactions = self.wm.rollout(
                            obs_0=obs_0_for_rollout,
                            act=actions_rollout,
                        )
                        # Extract state component (remove action dimension)
                        z_state_rollout = rolled_out_stateactions[...,:-self.wm.action_dim]
                        obs_decoded_rollout = self.wm.visualize(z_state_rollout[:,frame_indices])

                        images_rollout = []
                        for t in range(len(frame_indices)):
                            img_to_display = obs_decoded_rollout[0, t].permute(1, 2, 0).cpu().numpy()
                            # Normalize to [0, 1]
                            img_to_display = (np.clip(img_to_display,-1,1) + 1)/2
                            images_rollout.append(img_to_display)


                        # Concatenate images horizontally for each row
                        virtual_row = np.concatenate(images_virtual, axis=1)
                        next_pred_row = np.concatenate(images_next_pred, axis=1)
                        rollout_row = np.concatenate(images_rollout, axis=1)

                        # Create figure with all three rows
                        total_height = virtual_row.shape[0] + next_pred_row.shape[0] + rollout_row.shape[0] + 80  # 40 pixels spacing between rows
                        fig, ax = plt.subplots(1, 1, figsize=(virtual_row.shape[1]/50, total_height/50))

                        # Combine rows vertically with spacing
                        combined_image = np.concatenate([
                            virtual_row,
                            np.ones((20, virtual_row.shape[1], 3)),
                            next_pred_row,
                            np.ones((20, virtual_row.shape[1], 3)),
                            rollout_row
                        ], axis=0)

                        ax.imshow(combined_image)
                        ax.set_xlim(0, combined_image.shape[1])
                        ax.set_ylim(combined_image.shape[0], 0)

                        # Add text labels
                        ax.text(10, 15, 'Virtual States', fontsize=12, color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                        ax.text(10, virtual_row.shape[0] + 35, 'Next Pred', fontsize=12, color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                        ax.text(10, virtual_row.shape[0] + next_pred_row.shape[0] + 55, 'WM Rollout', fontsize=12, color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                        ax.axis('off')
                        plt.tight_layout()

                        # Convert matplotlib figure to numpy array
                        fig.canvas.draw()
                        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        plt.close(fig)

                        # Add state comparison to logs dict (will be logged together below)
                        logs["state_comparison"] = wandb.Image(
                            buf,
                            caption=f"Virtual states (top), Next pred (middle), WM rollout (bottom) at iteration {iter_idx}"
                        )

                    # logs["virtual_states"] = wandb.Image(
                    #     np.concatenate(images_virtual, axis=1),  # Concatenate horizontally
                    #     caption=f"States at iteration {iter_idx}"
                    # )

                # Log all metrics in a single call with explicit step parameter
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({
                    "step": iter_idx + 1,
                    "total_measured_time": total_measured_time,
                    f"{self.logging_prefix}/goal_loss": loss_dict["goal_loss"].item(),
                    f"{self.logging_prefix}/action_mag_loss": loss_dict["action_mag_loss"].item(),
                    f"{self.logging_prefix}/action_smooth_loss": loss_dict["action_smooth_loss"].item(),
                    f"{self.logging_prefix}/state_uniform_loss": loss_dict["state_uniform_loss"].item(),
                    f"{self.logging_prefix}/dynamics_loss": loss_dict["dynamics_loss"].item(),
                    f"{self.logging_prefix}/dynamics_loss_last": loss_dict["dynamics_loss_last"].item(),
                    f"{self.logging_prefix}/lagrangian": loss_dict["lagrangian"].item(),
                    f"{self.logging_prefix}/loss": loss_dict["total_loss_plot"],
                    f"{self.logging_prefix}/z_state_grad_norm": z_state_grad_norm,
                    f"{self.logging_prefix}/actions_grad_norm": actions_grad_norm,
                })

                # Create per-action gradient norm plot (only if save_images is enabled)
                if self.evaluator.save_images:
                    fig_grad, ax_grad = plt.subplots(figsize=(10, 4))
                    ax_grad.bar(range(self.horizon), per_action_grad_norms.cpu().numpy())
                    ax_grad.set_xlabel('Action Index (time step)')
                    ax_grad.set_ylabel('Gradient Norm')
                    ax_grad.set_title(f'Per-Action Gradient Norms at Iteration {iter_idx}')
                    ax_grad.set_xticks(range(0, self.horizon, max(1, self.horizon // 10)))
                    plt.tight_layout()

                    # Convert to wandb image
                    fig_grad.canvas.draw()
                    grad_buf = np.frombuffer(fig_grad.canvas.tostring_rgb(), dtype=np.uint8)
                    grad_buf = grad_buf.reshape(fig_grad.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig_grad)

                    logs[f"{self.logging_prefix}/per_action_grad_norms"] = wandb.Image(
                        grad_buf,
                        caption=f"Per-action gradient norms at iteration {iter_idx}"
                    )

                self.wandb_run.log(logs)
                # self.dump_logs({k: v for k, v in logs.items() if k != "virtual_states"})  # except virtual states


                if np.all(successes):
                    exit_step = iter_idx + 1
                    break  # terminate planning if all success

            # if measured time is longer than two hours, terminate planning
            if total_measured_time > 1800:
                exit_step = iter_idx + 1
                break


            

#             # CEM refinement step (if enabled)
#             if self.use_cem and iter_idx % self.cem_interval == (self.cem_interval - 1):
#                 with torch.no_grad():
#                     frozen_z_state = z_state_expanded.detach().clone()
#                     actions.data = self._cem_planning(
#                         z_obs_0=z_obs_0,
#                         frozen_z_state=frozen_z_state,
#                         num_samples=self.cem_num_samples,
#                         num_iterations=1,
#                         initial_actions=actions.detach()
#                     ).data



        # Store planning stats for access by plan.py
        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        # Return only the first copy of each original eval (not all parallel copies)
        if n_parallel > 1:
            actions_final = actions.detach().view(n_evals_original, n_parallel, *actions.shape[1:])[:, 0]
        else:
            actions_final = actions.detach()

        return actions_final, np.full(n_evals_original, np.inf)  # all actions are valid
