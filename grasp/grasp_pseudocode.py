"""
GRASP: Gradient RelAxed Stochastic Planner — minimal pseudocode.

Core planning algorithm with virtual states, stop-gradient dynamics loss,
goal shaping, Langevin-style state noise, and periodic full-rollout sync.

See: grasp.html, grasp.tex
"""

# =============================================================================
# WORLD MODEL INTERFACE (provided by the world model, not this file)
# =============================================================================
#
#   def rollout(s_0, a) -> states
#       """Roll out dynamics from initial state with action sequence.
#
#       Args:
#           s_0: (B, D)  initial state
#           a:   (B, T, action_dim)  action sequence
#
#       Returns:
#           states: (B, T+1, D)  full trajectory with
#               states[:, 0] = s_0
#               states[:, t+1] = F(states[:, t], a[:, t])
#       """
#
#   action_dim: int
#
# Single-step F(s_t, a_t) -> s_{t+1} is rollout(s_t, a[:, t:t+1]).
# GD sync uses rollout(s_0, a)[:, -1] to get s_T.
#
# =============================================================================

from abc import ABC, abstractmethod

import torch


class WorldModelInterface(ABC):
    """World model interface required by GRASP. Implement this for your model."""

    action_dim: int

    def rollout(self, s_0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Roll out dynamics from initial state with action sequence.

        Args:
            s_0: (B, D)  initial state
            a:   (B, T, action_dim)  action sequence

        Returns:
            states: (B, T+1, D)  full trajectory with
                states[:, 0] = s_0
                states[:, t+1] = F(states[:, t], a[:, t])
        """
        raise NotImplementedError


def init_states(s_0, g, num_virtual_states, device=None):
    """Linear interpolation between s_0 and g for virtual states s_1..s_{T-1}.

    Args:
        s_0: (B, D) start state
        g: (B, D) goal state
        num_virtual_states: T-1, number of virtual states between s_0 and g

    Returns:
        s: (B, num_virtual_states, D) virtual states, requires_grad=True
    """
    device = device or s_0.device
    B, D = s_0.shape
    t = torch.linspace(0, 1, num_virtual_states + 2, device=device)  # 0, 1/T, ..., 1
    t = t[1:-1].view(1, -1, 1)  # (1, T-1, 1) for broadcasting
    s = s_0.unsqueeze(1) + t * (g.unsqueeze(1) - s_0.unsqueeze(1))
    return s.requires_grad_(True)


def compute_loss(s, a, g, s_0, wm):
    """L = L_dyn_sg + L_goal_sg with stop-gradient on state inputs.

    Args:
        s: (B, T-1, D) virtual states s_1..s_{T-1}
        a: (B, T, action_dim) actions
        g: (B, D) goal
        s_0: (B, D) start (fixed)
        wm: world model with rollout(s_0, a) -> (B, T+1, D)

    Returns:
        loss: scalar
    """
    B, T_minus_1, D = s.shape
    T = T_minus_1 + 1

    s_full = torch.cat([s_0.unsqueeze(1), s, g.unsqueeze(1)], dim=1)  # (B, T+1, D)

    losses_dyn = []
    losses_goal = []

    for t in range(T):
        s_t = s_full[:, t]  # (B, D)
        a_t = a[:, t : t + 1]  # (B, 1, action_dim)

        # F(bar_s_t, a_t) via 1-step rollout; detach cuts state-input gradients
        traj = wm.rollout(s_t.detach(), a_t)  # (B, 2, D)
        pred_next = traj[:, 1]  # (B, D)

        s_next = s_full[:, t + 1]

        losses_dyn.append(((pred_next - s_next) ** 2).mean())
        losses_goal.append(((pred_next - g) ** 2).mean())

    return sum(losses_dyn) + sum(losses_goal)


def sync_step(wm, s_0, g, a, gd_opt_steps, gd_lr, device=None):
    """Periodic sync: GD on terminal loss |s_T(a) - g|^2, updating only a."""
    device = device or a.device
    a_opt = a.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([a_opt], lr=gd_lr)

    for _ in range(gd_opt_steps):
        optimizer.zero_grad()
        s_T = wm.rollout(s_0, a_opt)[:, -1]
        loss = ((s_T - g) ** 2).mean()
        loss.backward()
        optimizer.step()

    return a_opt.detach()


def grasp_plan(
    s_0,
    g,
    wm,
    horizon,
    lr_s=0.1,
    lr_a=0.001,
    opt_steps=1000,
    state_noise_scale=0.5,
    gd_interval=100,
    gd_opt_steps=25,
    gd_lr=0.1,
    device=None,
):
    """GRASP planning: optimize actions and virtual states.

    Args:
        s_0: (B, D) start state
        g: (B, D) goal state
        wm: world model (see interface at top)
        horizon: T, planning horizon
        lr_s: learning rate for states
        lr_a: learning rate for actions
        opt_steps: total optimization iterations
        state_noise_scale: sigma for Langevin-style state noise
        gd_interval: run sync every N steps
        gd_opt_steps: GD steps per sync
        gd_lr: learning rate for sync phase

    Returns:
        a: (B, horizon, action_dim) optimized actions
    """
    device = device or s_0.device
    B, D = s_0.shape
    T = horizon
    num_virtual_states = T - 1

    s = init_states(s_0, g, num_virtual_states, device)
    a = torch.zeros(B, T, wm.action_dim, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([{"params": s, "lr": lr_s}, {"params": a, "lr": lr_a}])

    for k in range(opt_steps):
        optimizer.zero_grad()
        loss = compute_loss(s, a, g, s_0, wm)
        loss.backward()
        optimizer.step()

        # Langevin-style state noise
        with torch.no_grad():
            s.data += state_noise_scale * torch.randn_like(s, device=device)

        # Periodic sync
        if (k + 1) % gd_interval == 0 and k > 0:
            a = sync_step(wm, s_0, g, a, gd_opt_steps, gd_lr, device)
            a = a.requires_grad_(True)
            optimizer = torch.optim.Adam(
                [{"params": s, "lr": lr_s}, {"params": a, "lr": lr_a}]
            )

    return a.detach()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Minimal test: simple world model + run grasp_plan
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class SimpleWorldModel(WorldModelInterface):
    """Minimal differentiable world model for testing: s_next = s + dt * a."""

    def __init__(self, state_dim=2, action_dim=2, dt=0.1, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt
        self.device = device or torch.device("cpu")

    def rollout(self, s_0, a):
        B, T, _ = a.shape
        states = [s_0]
        for t in range(T):
            next_s = states[-1] + self.dt * a[:, t]
            states.append(next_s)
        return torch.stack(states, dim=1)


if __name__ == "__main__":
    device = torch.device("cpu")
    wm = SimpleWorldModel(state_dim=2, action_dim=2, dt=0.1, device=device)

    # Start at (0, 0), goal at (2, 2)
    s_0 = torch.tensor([[0.0, 0.0]], device=device)
    g = torch.tensor([[2.0, 2.0]], device=device)

    # Fixed constraints: opt_steps=100, gd_opt_steps=5, gd_interval=20, state_noise_scale=0.01
    # Best hyperparams: lr_s=0.2, lr_a=0.2, gd_lr=0.5 (variance with noise; run multiple trials)
    actions = grasp_plan(
        s_0, g, wm, horizon=10,
        opt_steps=100, gd_interval=20, gd_opt_steps=5,
        lr_s=0.2, lr_a=0.2, state_noise_scale=0.01, gd_lr=0.5,
        device=device,
    )

    traj = wm.rollout(s_0, actions)
    s_T = traj[:, -1]
    state_dim = s_0.shape[-1]
    dist_per_dim = (s_T - g).norm() / (state_dim**0.5)
    print(f"Goal distance: {dist_per_dim.item():.4f}")
