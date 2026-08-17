import numpy as np
import torch
import time
from utils import move_to_device

from einops import rearrange

from .base_planner import BasePlanner

import wandb

import os
from tqdm import tqdm


class LatCoPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        init_noise_level=0.3, # noise level for state initialization
        lr_s=0.01,  # Learning rate for state
        lr_a=0.1,  # Learning rate for actions. Typically lr_a > lr_s
        opt_steps=30,  # Number of optimization steps
        eval_every=1,
        logging_prefix="plan_0",
        log_filename="logs.json",
        action_mag_coeff=1.0,
        opt="adam",
        scheduler_type="none",
        gamma_dynamics=1.0,  # initial value for lambda_dynamics
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
        self.init_noise_level = init_noise_level
        self.lr_s = lr_s
        self.lr_a = lr_a
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.action_mag_coeff = action_mag_coeff
        self.opt = opt
        self.scheduler_type = scheduler_type
        self.gamma_dynamics = gamma_dynamics

    def _init_actions(self, z_obs_0, actions=None):

        """Initialize action sequence."""
        n_evals = z_obs_0.shape[0]

        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
            actions = torch.cat([actions, new_actions.to(device)], dim=1)

        return actions

    def _init_states(self, start_state, goal_state):
        """
        Initialize states as points along a cylinder from start to goal.

        Args:
            start_state: Starting state (torch.Tensor)
            goal_state: Goal state (torch.Tensor)

        Returns:
            states: Initialized states tensor with gradient tracking
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

        # Fix the start and end states TODO: definitely a cleaner way to
        # do this
        states.data[:, [0]] = start_state.data
        states.data[:, [-1]] = goal_state.data

        states = states.detach().requires_grad_(True)
        return states

    def compute_loss(
        self, z_state, z_obs_g, v_s, actions, gamma_dynamics=1.0
    ):
        """
        Compute the optimization loss.

        Args:
            z_state: Sequence of states
            z_obs_g: Goal state
            f_next_state_list: Predicted next states from dynamics model
            actions: Sequence of actions
            gamma_dynamics: Scaling factor for dynamics loss (default: 1.0)

        Returns:
            loss components and total loss
        """
        # Goal loss: distance between final state and goal state
        goal_loss = ((z_state[:, [-1], ...] - z_obs_g) ** 2).mean()

        # Dynamics loss: error in state transitions
        dynamics_loss_batch = ((z_state[:,:-1,...] + v_s - z_state[:, 1:, ...]) ** 2).mean(dim=tuple(range(1, z_state.dim())))
        dynamics_loss = gamma_dynamics * dynamics_loss_batch.mean()

        # Action magnitude penalty
        action_mag_loss = self.action_mag_coeff*(torch.clamp(actions**2 - 1, 0, torch.inf)).mean()

        # Total loss, dimension regularized
        total_loss = (goal_loss + dynamics_loss + action_mag_loss)

        return {
            "goal_loss": goal_loss,
            "dynamics_loss_batch": dynamics_loss_batch,
            "dynamics_loss": dynamics_loss,
            "action_mag_loss": action_mag_loss,
            "total_loss": total_loss,
        }

    def plan(self, obs_0, obs_g, actions=None):
        """
        Simple alternating optimization planner:
        1. Roll out trajectory with initial actions to get states
        2. Alternate between:
           - Optimize states w.r.t. ||s_T - s_*||_2^2 + Σ||(s_{i+1}-s_i) - F(s_≤i, a_i)||_2^2
           - Optimize actions w.r.t. Σ||(s_{i+1}-s_i) - F(s_≤i, a_i)||_2^2


        OVERALL OPTIMIZATION PARAM:
        We are going to optimize the raw actions (unencoded), and the encoded states directly.
        Note that we don't actually need to output states, so its ok to just optimize latent states
        directly.
        """

        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(obs_0, obs_g, encode_to_vector=True)

        # Initialize actions
        # z_state = torch.nn.Parameter((torch.randn(z_obs_0.shape[0], H - 1, z_obs_0.shape[2], z_obs_0.shape[3]).to(z_obs_0)), requires_grad=True)

        z_state = self._init_states(z_obs_0, z_obs_g).to(self.device)


        actions = self._init_actions(z_obs_0, actions).to(self.device)
        n_evals = actions.shape[0]
        actions.requires_grad = True

        # Create separate optimizers for states and actions
        if self.opt == "adam":
            opt = torch.optim.Adam
        elif self.opt == "sgd":
            opt = torch.optim.SGD
        else:
            raise ValueError


        # Create optimizer for states
        state_optimizer = opt([{"params": z_state, "lr": self.lr_s}])

        # Create optimizer for actions
        action_optimizer = opt([{"params": actions, "lr": self.lr_a}])

        gamma_dynamics = self.gamma_dynamics

        # Create separate schedulers for each optimizer
        state_scheduler = None
        action_scheduler = None
        if self.scheduler_type == 'cos':
            state_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                state_optimizer, T_max=self.opt_steps
            )
            action_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                action_optimizer, T_max=self.opt_steps
            )
        elif self.scheduler_type != 'none':
            raise ValueError

        print("opt steps", self.opt_steps)

        # Main optimization loop
        total_measured_time = 0.0
        exit_step = self.opt_steps  # Default to max steps if no early exit

        for iter_idx in range(self.opt_steps):
            start = time.perf_counter()
            # Zero gradients for both optimizers
            state_optimizer.zero_grad()
            action_optimizer.zero_grad()

            action_samples_flat = actions
            z_state_expanded = z_state

            f_next_state_list = self.wm.next_state(
                z_state_expanded,
                action_samples_flat
            )

            v_s = f_next_state_list - z_state_expanded[:, :-1]

            z_obs_g_expanded = z_obs_g

            loss_dict = self.compute_loss(z_state_expanded, z_obs_g_expanded, v_s, action_samples_flat, gamma_dynamics)
            loss = loss_dict["total_loss"]
            loss.backward()

            # Calculate gradient norms for logging
            # note beginning and end states are fixed, so we don't include
            z_state_grad_norm = torch.norm(z_state.grad[:,1:-1].detach(), p=2).item() if z_state.grad is not None else 0.0
            actions_grad_norm = torch.norm(actions.grad.detach(), p=2).item() if actions.grad is not None else 0.0

            # Apply optimization steps separately
            state_optimizer.step()
            action_optimizer.step()

            # Step schedulers if needed
            if self.scheduler_type != 'none' and state_scheduler and action_scheduler:
                state_scheduler.step()
                action_scheduler.step()

            with torch.no_grad():
                z_state.data[:, [0]] = z_obs_0.data
                z_state.data[:, [-1]] = z_obs_g.data
                actions.data.clamp_(-1.0, 1.0)

            # Evaluate current plan
            end = time.perf_counter()
            total_measured_time += end - start

            if self.evaluator is not None and iter_idx % self.eval_every == 0:

                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(),
                    filename=f"{self.logging_prefix}_output_{iter_idx+1}",
                )


                # Log extra metrics
                self.wandb_run.log(
                    {
                        f"{self.logging_prefix}/goal_loss": loss_dict[
                            "goal_loss"
                        ].item(),
                        f"{self.logging_prefix}/action_mag_loss": loss_dict[
                            "action_mag_loss"
                        ].item(),
                        f"{self.logging_prefix}/dynamics_loss": loss_dict[
                            "dynamics_loss"
                        ].item(),
                        f"{self.logging_prefix}/loss": loss.item(),
                        f"{self.logging_prefix}/z_state_grad_norm": z_state_grad_norm,
                        f"{self.logging_prefix}/actions_grad_norm": actions_grad_norm,
                    }
                )

                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"total_measured_time": total_measured_time})
                logs.update({"step": iter_idx + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)


                if np.all(successes):
                    exit_step = iter_idx + 1
                    break  # terminate planning if all success

            if total_measured_time > 1800:
                exit_step = iter_idx + 1
                break

        # Store planning stats for access by plan.py
        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        return actions.detach(), np.full(n_evals, np.inf)  # all actions are valid
