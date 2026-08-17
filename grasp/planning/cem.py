import torch
import time
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
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
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.action_clamp = action_clamp

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """

        n_evals = obs_0["visual"].shape[0] if isinstance(obs_0, dict) else obs_0.shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(obs_0, obs_g, encode_to_vector=False)
        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        total_measured_time = 0.0
        exit_step = self.opt_steps  # Default to max steps if no early exit

        for i in range(self.opt_steps):
            start = time.perf_counter()
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                if isinstance(obs_0, dict):
                    cur_trans_obs_0 = {
                        key: repeat(
                            arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                        )
                        for key, arr in trans_obs_0.items()
                    }
                    cur_z_obs_g = {
                        key: repeat(
                            arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                        )
                        for key, arr in z_obs_g.items()
                    }
                else:
                    cur_trans_obs_0 = repeat(
                            trans_obs_0[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                        )

                    cur_z_obs_g = repeat(
                            trans_obs_g[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                        )
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        # dealing with different signatures, outside evaluator passes [B, T, D] instead of [B, 1, T, D]
                        obs_0=cur_trans_obs_0 if isinstance(obs_0, dict) else cur_trans_obs_0[:,0],
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g, action)

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu_mean = topk_action.mean(dim=0)
                if self.action_clamp > 0:
                    mu_mean = mu_mean.clamp(-self.action_clamp, self.action_clamp)
                mu[traj] = mu_mean
                sigma[traj] = topk_action.std(dim=0)


            end = time.perf_counter()
            total_measured_time += end - start

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                self.wandb_run.log(
                    {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1,
                    f"{self.logging_prefix}/act_norm": mu.norm(dim=-1).mean().item(),
                    f"{self.logging_prefix}/success": successes.mean(),
                    }
                )

                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"total_measured_time": total_measured_time})
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    exit_step = i + 1
                    break  # terminate planning if all success

            if total_measured_time > 1800:
                exit_step = i + 1
                break

        # Store planning stats for access by plan.py
        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        return mu, np.full(n_evals, np.inf)  # all actions are valid
