import json
import numpy as np
from abc import ABC, abstractmethod
from utils import move_to_device


class BasePlanner(ABC):
    def __init__(
        self,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        log_filename,
        **kwargs,
    ):
        self.wm = wm
        self.action_dim = action_dim
        self.objective_fn = objective_fn
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device

        self.evaluator = evaluator
        self.wandb_run = wandb_run
        self.log_filename = log_filename  # do not log if None

    def dump_logs(self, logs):
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        if self.log_filename is not None:
            with open(self.log_filename, "a") as file:
                file.write(json.dumps(logs_entry) + "\n")

    @abstractmethod
    def plan(self):
        pass

    def preprocess_start_goal(self, obs_0, obs_g, encode_to_vector=True):
        if isinstance(obs_0, dict):
            obs_0 = self.preprocessor.transform_obs(obs_0)
            obs_g = self.preprocessor.transform_obs(obs_g)

        trans_obs_0 = move_to_device(obs_0, self.device)
        trans_obs_g = move_to_device(obs_g, self.device)
        
        wm_formatter = self.wm.encode_observation_to_vector if encode_to_vector else self.wm.encode_obs
        z_obs_0 = wm_formatter(trans_obs_0)
        z_obs_g = wm_formatter(trans_obs_g)
        return trans_obs_0, z_obs_0, trans_obs_g, z_obs_g
