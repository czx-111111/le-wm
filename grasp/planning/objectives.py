import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last", action_mag_coeff=0.0):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
        action_mag_coeff: float, coefficient for action magnitude regularization
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt, actions=None):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            actions: tensor (B, T, action_dim), optional actions for magnitude regularization
        Returns:
            loss: tensor (B, )
        """
        if isinstance(z_obs_pred, dict):
            loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
                dim=tuple(range(1, z_obs_pred["visual"].ndim))
            )
            loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(1, z_obs_pred["proprio"].ndim))
            )
            loss = loss_visual + alpha * loss_proprio

        else:
            loss = metric(z_obs_pred[:,-1:], z_obs_tgt).mean(dim=tuple(range(1,z_obs_pred.ndim)))
        
        # Add action magnitude regularization if actions are provided
        if actions is not None and action_mag_coeff > 0:
            action_mag = torch.mean(actions ** 2, dim=tuple(range(1, actions.ndim)))  # (B,)
            loss = loss + action_mag_coeff * action_mag
            
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, actions=None):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            actions: tensor (B, T, action_dim), optional actions for magnitude regularization
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        
        # Add action magnitude regularization if actions are provided
        if actions is not None and action_mag_coeff > 0:
            action_mag = torch.mean(actions ** 2, dim=tuple(range(1, actions.ndim)))  # (B,)
            loss = loss + action_mag_coeff * action_mag
            
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError
