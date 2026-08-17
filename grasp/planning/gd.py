import torch
import time
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        var_actions=1.0,
        var_states=0.0,
        num_samples=8,
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
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.var_actions = var_actions
        self.var_states = var_states
        self.num_samples = num_samples
        self.action_clamp = action_clamp

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        if isinstance(obs_0, dict):
            n_evals = obs_0["visual"].shape[0]
        else:
            n_evals = obs_0.shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions





    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(obs_0, obs_g, encode_to_vector=False)
        actions = self.init_actions(obs_0, actions).to(self.device)
        actions_noise = torch.full((1, 1, 1, self.action_dim), fill_value=self.var_actions).to(actions)
        actions_noise.requires_grad = False
        actions.requires_grad = True
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        n_evals = actions.shape[0]

        if isinstance(trans_obs_0, dict):
            trans_obs_0 = {k: v.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1) for k,v in trans_obs_0.items()}
            z_obs_g_detached = {k: v.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1).detach() for k, v in z_obs_g.items()}
        else:
            # format needed for rollout func
            trans_obs_0 = trans_obs_0[:,0].unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1)
            z_obs_g_detached = z_obs_g.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1).detach()

        total_measured_time = 0.0
        exit_step = self.opt_steps  # Default to max steps if no early exit

        for i in range(self.opt_steps):
            start = time.perf_counter()
            optimizer.zero_grad()
            eps = torch.randn(self.num_samples, actions.shape[0], actions.shape[1], actions.shape[2]).to(actions)*actions_noise
            noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1)
            if self.var_states > 0:
                i_z_obses, i_zs = self.wm.rollout_noisy(
                    obs_0=trans_obs_0,
                    act=noisy_actions,
                    var_states=self.var_states,
                )
            else:
                i_z_obses, i_zs = self.wm.rollout(
                    obs_0=trans_obs_0,
                    act=noisy_actions,
                )
            loss = self.objective_fn(i_z_obses, z_obs_g_detached, noisy_actions)  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                if self.action_clamp > 0:
                    actions.data.clamp_(-self.action_clamp, self.action_clamp)
            end = time.perf_counter()
            total_measured_time += end - start

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                self.wandb_run.log({
                    f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1,
                    f"{self.logging_prefix}/act_norm": actions.detach().norm(dim=-1).mean().item(),
                    f"{self.logging_prefix}/success": successes.mean(),
                    f"{self.logging_prefix}/noise": actions_noise.mean().item()
                })

                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"total_measured_time": total_measured_time})
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    print("######## Success ########")
                    exit_step = i + 1
                    break  # terminate planning if all success
            if total_measured_time > 1800:
                exit_step = i + 1
                break

        # Log final results
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), filename=f"{self.logging_prefix}_output_final"
        )
        self.wandb_run.log({
            f"{self.logging_prefix}/loss": total_loss.item(), "step": self.opt_steps,
            f"{self.logging_prefix}/act_norm": actions.detach().norm(dim=-1).mean().item(),
            f"{self.logging_prefix}/success": successes.mean(),
            f"{self.logging_prefix}/noise": actions_noise.mean().item()
        })

        logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
        logs.update({"total_measured_time": total_measured_time})
        logs.update({"step": self.opt_steps})
        self.wandb_run.log(logs)
        self.dump_logs(logs)
        # Store planning stats for access by plan.py
        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        return actions, np.full(n_evals, np.inf)
