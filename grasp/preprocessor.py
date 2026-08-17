"""Preprocessing utilities for actions, states, proprioception, and observations.

The :class:`Preprocessor` normalizes scalar quantities using dataset statistics
and applies visual transforms to image observations before they are passed into
world models and planners.
"""

import torch
from einops import rearrange
from torchvision import tv_tensors

class Preprocessor:
    """Helper for normalizing trajectories and preparing observations."""

    def __init__(
        self,
        action_mean,
        action_std,
        state_mean,
        state_std,
        proprio_mean,
        proprio_std,
        transform,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.transform = transform

    def normalize_actions(self, actions):
        """Normalize actions with shape ``(B, T, action_dim)``."""
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions):
        """Denormalize actions with shape ``(B, T, action_dim)``."""
        return actions * self.action_std + self.action_mean
    
    def normalize_proprios(self, proprio):
        """Normalize proprioceptive inputs of shape ``(..., proprio_dim)``."""
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_states(self, state):
        """Normalize state vectors of shape ``(..., state_dim)``."""
        return (state - self.state_mean) / self.state_std

    def preprocess_obs_visual(self, obs_visual):
        """Convert raw uint8 visual observations to float tensors in BCHW format."""
        return rearrange(obs_visual, "b t h w c -> b t c h w") / 255.0
        

    def transform_obs_visual(self, obs_visual):
        """Apply the configured visual transform to a batch of videos.

        Args:
            obs_visual: Numpy array or tensor of shape ``(B, T, H, W, C)``.

        Returns:
            Torch tensor of transformed videos suitable for model input.
        """
        transformed_obs_visual = torch.tensor(obs_visual)
        transformed_obs_visual = self.preprocess_obs_visual(transformed_obs_visual)
        return torch.stack([self.transform(tv_tensors.Video(elem)) for elem in transformed_obs_visual])
    
    def transform_obs(self, obs):
        """Convert a raw observation dict of numpy arrays to normalized tensors."""
        transformed_obs = {}
        transformed_obs['visual'] = self.transform_obs_visual(obs['visual'])
        transformed_obs['proprio'] = self.normalize_proprios(torch.tensor(obs['proprio']))
        return transformed_obs
