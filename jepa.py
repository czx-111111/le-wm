"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        print(f'DEBUG encode pixels.shape={pixels.shape}', flush=True)
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            print(f"DEBUG encode action.shape: {info['action'].shape}")
            print(f"DEBUG action_encoder expects channels: {self.action_encoder.patch_embed.in_channels}")
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        # info["action"] = act_0  # 注释掉，避免修改原始info
        n_steps = T - H

        # copy and encode initial info dict
        # pixels needs time dim preserved: (B, T, C, H, W) -> (B, 1, C, H, W)
        _init = {}
        for k, v in info.items():
            if torch.is_tensor(v):
                if k == "pixels":
                    _init[k] = v[:, :1]  # keep time dim
                else:
                    _init[k] = v[:, :1]  # 保留时间维
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        print(f"DEBUG rollout: emb.shape={emb.shape}, act.shape={act.shape}", flush=True)
        for t in range(n_steps):
            print(f"DEBUG t={t}: emb.shape={emb.shape}, act.shape={act.shape}", flush=True)
            act_emb = self.action_encoder(act)
            max_len = min(emb.shape[1], act_emb.shape[1], HS)
            emb_trunc = emb[:, -max_len:]  # (BS, max_len, D)
            act_trunc = act_emb[:, -max_len:]  # (BS, max_len, A_emb)
            print(f"DEBUG trunc: max_len={max_len}, emb_trunc.shape={emb_trunc.shape}, act_trunc.shape={act_trunc.shape}", flush=True)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        max_len = min(emb.shape[1], act_emb.shape[1], HS)
        emb_trunc = emb[:, -max_len:]  # (BS, max_len, D)
        act_trunc = act_emb[:, -max_len:]  # (BS, max_len, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        print(f'DEBUG info_dict keys: {[k for k in info_dict.keys() if torch.is_tensor(info_dict[k])]}', flush=True)
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                print(f'DEBUG info_dict[{k}].shape={v.shape}', flush=True)
        # goal is already (B, C, H, W) or (B, D), no time dim to slice
        goal = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                if k.startswith("goal"):
                    goal[k.replace("goal_", "")] = v  # (B, C, H, W) -> keep as is
        goal["pixels"] = goal["goal"].unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)
        # goal_proprio -> proprio
        if "goal_proprio" in info_dict:
            goal["proprio"] = info_dict["goal_proprio"].unsqueeze(1)  # (B, D) -> (B, 1, D)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
