"""
Planning entrypoint for GRASP and baseline planners on DINO-WM-style world models.

This module wires together:
- experiment configuration (Hydra),
- trained or analytic world models,
- gym-style environments,
- planners (e.g., GRASP, CEM, GD, LATCO),
- and evaluation / logging utilities.

It is typically invoked via the CLI:

    python plan.py model_name=<model_name> n_evals=5 planner=grasp goal_H=5

The main user-facing entrypoint is the `main` Hydra function at the bottom of
the file, which constructs a config dict and calls `planning_main`.
"""

from __future__ import annotations

import os
import json
import logging
import pickle
import random
from itertools import product
from pathlib import Path

import gym
import hydra
import numpy as np
import submitit
import torch
import wandb
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
import custom_resolvers
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from planning.ballnav_evaluator import BallNavEvaluator
from planning.random_mlp_evaluator import RandomMlpEvaluator
from utils import cfg_to_dict, seed

log = logging.getLogger(__name__)


def _suppress_warnings() -> None:
    """Suppress noisy warnings during planning runs."""
    import warnings

    warnings.filterwarnings("ignore")


_suppress_warnings()

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def planning_main_in_dir(working_dir: str, cfg_dict: dict):
    """Run planning in a given working directory.

    This is used by submitit to launch evaluation jobs in per-experiment
    subdirectories while reusing the main planning entrypoint.
    """
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch: int,
    cfg_dicts: list[dict],
    plan_output_dir: str,
):
    """Launch a set of planning jobs with submitit.

    Each configuration dict produces a subdirectory of the form
    ``{planner_name}_goal_source={...}_goal_H={...}_alpha={...}`` under
    ``plan_output_dir``. Within each subdirectory, `planning_main_in_dir`
    is executed with a cleaned environment to avoid side effects across jobs.
    """
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = (
                f"{cfg_dict['planner']['name']}"
                f"_goal_source={cfg_dict['goal_source']}"
                f"_goal_H={cfg_dict['goal_H']}"
                f"_alpha={cfg_dict['objective']['alpha']}"
            )
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            log.info(
                "Submitted evaluation job for checkpoint: %s, job id: %s",
                subdir_path,
                job.job_id,
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path: str = "",
    ckpt_base_path: str = "",
    model_name: str = "",
    model_epoch: str = "final",
    planner: list[str] | None = None,
    goal_source: list[str] | None = None,
    goal_H: list[int] | None = None,
    alpha: list[float] | None = None,
) -> list[dict]:
    """Build a list of planner configuration dicts for sweeps.

    Args:
        plan_cfg_path: Path to the base planning config (e.g., ``conf/plan.yaml``).
        ckpt_base_path: Root directory containing trained model checkpoints.
        model_name: Name of the model (subdirectory under ``ckpt_base_path/outputs``).
        model_epoch: Checkpoint epoch identifier (e.g., ``\"final\"``).
        planner: List of planner names to evaluate (e.g., ``[\"grasp\", \"cem\"]``).
        goal_source: List of goal-source modes (e.g., ``[\"dset\", \"random_state\"]``).
        goal_H: List of planning horizons to evaluate.
        alpha: List of objective weighting coefficients.

    Returns:
        A list of plain Python dicts suitable to be passed as ``cfg_dict`` into
        :func:`planning_main`. Each dict has the planner config merged in under
        ``cfg_dict[\"planner\"]`` and sets ``cfg_dict[\"planner\"][\"horizon\"]`` to
        ``cfg_dict[\"goal_H\"]``.
    """
    if planner is None:
        planner = ["gd", "cem", "grasp"]
    if goal_source is None:
        goal_source = ["dset"]
    if goal_H is None:
        goal_H = [1, 5, 10]
    if alpha is None:
        alpha = [0, 0.1, 1]

    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts: list[dict] = []
    for override_args in overrides:
        planner_name = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner_name}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        # Assume planning horizon equals goal horizon unless explicitly overridden.
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    """Workspace that bundles together model, data, environment, planner, and logging.

    A ``PlanWorkspace`` is responsible for:
    - preparing start / goal states and observations,
    - constructing the appropriate evaluator and planner instances, and
    - running the planner to produce action sequences and evaluation logs.
    """

    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run

        self._init_device_and_seeds()
        objective_fn = hydra.utils.call(cfg_dict["objective"])
        self._init_preprocessor_and_targets()
        self._init_evaluator()
        self._ensure_wandb_run()
        self._init_planner(objective_fn)
        self.dump_targets()

    def _init_device_and_seeds(self) -> None:
        """Initialize device and per-evaluation seeds."""
        # Get device from world model, handle case where model has no parameters
        try:
            self.device = next(self.wm.parameters()).device
        except StopIteration:
            # Fallback for models without parameters (like ballnav)
            if hasattr(self.wm, "device"):
                self.device = self.wm.device
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

        # Have different seeds for each planning instance
        self.eval_seed = [
            self.cfg_dict["seed"] * self.cfg_dict["n_evals"] + n
            for n in range(self.cfg_dict["n_evals"])
        ]
        log.info("eval_seed: %s", self.eval_seed)
        self.n_evals = self.cfg_dict["n_evals"]
        self.goal_source = self.cfg_dict["goal_source"]
        self.goal_H = self.cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = self.cfg_dict["debug_dset_init"]
        self.save_images = self.cfg_dict.get("save_images", True)

    def _init_preprocessor_and_targets(self) -> None:
        """Create preprocessor and prepare start/goal targets."""
        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(self.cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

    def _init_evaluator(self) -> None:
        """Instantiate the appropriate evaluator for the current environment."""
        if self.env_name == "ballnav":
            self.evaluator = BallNavEvaluator(
                obs_0=self.obs_0,
                obs_g=self.obs_g,
                state_0=self.state_0,
                state_g=self.state_g,
                env=self.env,
                wm=self.wm,
                frameskip=self.frameskip,
                seed=self.eval_seed,
                preprocessor=self.data_preprocessor,
                n_plot_samples=self.cfg_dict["n_plot_samples"],
                save_images=self.save_images,
            )
        elif self.env_name == "random_mlp":
            self.evaluator = RandomMlpEvaluator(
                obs_0=self.obs_0,
                obs_g=self.obs_g,
                state_0=self.state_0,
                state_g=self.state_g,
                env=self.env,
                wm=self.wm,
                frameskip=self.frameskip,
                seed=self.eval_seed,
                preprocessor=self.data_preprocessor,
                n_plot_samples=self.cfg_dict["n_plot_samples"],
                save_images=self.save_images,
            )
        else:
            self.evaluator = PlanEvaluator(
                obs_0=self.obs_0,
                obs_g=self.obs_g,
                state_0=self.state_0,
                state_g=self.state_g,
                env=self.env,
                wm=self.wm,
                frameskip=self.frameskip,
                seed=self.eval_seed,
                preprocessor=self.data_preprocessor,
                n_plot_samples=self.cfg_dict["n_plot_samples"],
                save_images=self.save_images,
            )

    def _ensure_wandb_run(self) -> None:
        """Ensure that a wandb run object is always available."""
        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        # planner and final eval logs are dumped here
        self.log_filename = os.path.join(self.cfg_dict["saved_folder"], "logs.json")

    def _init_planner(self, objective_fn) -> None:
        """Instantiate the planner and align its horizon with the goal horizon."""
        # Apply separate planner seed if provided, otherwise use eval seed
        planner_seed = self.cfg_dict.get("planner_seed")
        if planner_seed is not None:
            seed(planner_seed)

        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for MPC
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
            saved_folder=self.cfg_dict["saved_folder"],
        )

        # Optional: assume planning horizon equals goal horizon
        from planning.mpc import MPCPlanner

        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = self.cfg_dict["goal_H"]
            self.planner.n_taken_actions = self.cfg_dict["goal_H"]
        else:
            self.planner.horizon = self.cfg_dict["goal_H"]

    def prepare_targets(self):
        # Special handling for ballnav - it doesn't have a real dataset
        if self.env_name == "ballnav":
            self.prepare_ballnav_targets()
            return

        # Special handling for random_mlp - it doesn't have a real dataset
        if self.env_name == "random_mlp":
            self.prepare_random_mlp_targets()
            return

        states = []
        actions = []
        observations = []

        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def prepare_ballnav_targets(self):
        """Special target preparation for ballnav environment."""
        print("Preparing ballnav targets...")

        # For ballnav, bypass the vectorized environment system and generate states directly
        np.random.seed(self.eval_seed[0])  # Use first seed for consistency

        # Generate random initial and goal states directly
        rand_init_states = []
        rand_goal_states = []

        for i in range(self.n_evals):
            # Random initial state within environment bounds (avoid edges)
            init_state = np.array([1.0, 5.0])

            # Random goal state within environment bounds
            goal_state = np.array([9.0, 5.0])

            rand_init_states.append(init_state)
            rand_goal_states.append(goal_state)

        # Convert to numpy arrays
        self.state_0 = np.array(rand_init_states)  # (n_evals, 2)
        self.state_g = np.array(rand_goal_states)  # (n_evals, 2)

        # For ballnav, observations are just the state vectors with time dimension
        self.obs_0 = np.expand_dims(self.state_0, axis=1)  # (n_evals, 1, 2)
        self.obs_g = np.expand_dims(self.state_g, axis=1)  # (n_evals, 1, 2)

        self.gt_actions = None  # No ground truth actions for ballnav

        print(f"Created ballnav targets - Initial states: {self.state_0.shape}, Goal states: {self.state_g.shape}")

    def prepare_random_mlp_targets(self):
        """Special target preparation for random_mlp environment."""
        print("Preparing random_mlp targets...")

        # For random_mlp, bypass the vectorized environment system and generate states directly
        np.random.seed(self.eval_seed[0])  # Use first seed for consistency

        # Get state and action dimensions from the world model
        state_dim = self.wm.state_dim
        action_dim = self.wm.action_dim

        # Generate random initial states and reachable goal states using world model rollout
        rand_init_states = []
        rand_goal_states = []

        for i in range(self.n_evals):
            # Sample random initial state
            init_state = np.random.randn(state_dim).astype(np.float32)

            # Sample random actions for goal_H steps
            random_actions = np.random.rand(self.goal_H, action_dim).astype(np.float32)

            # Convert to torch tensors for rollout
            init_state_tensor = torch.tensor(init_state, device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
            actions_tensor = torch.tensor(random_actions, device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, goal_H, action_dim)

            # Rollout using world model to get reachable goal state
            states, _ = self.wm.rollout(init_state_tensor, actions_tensor)  # (1, goal_H+1, state_dim)
            goal_state = states[0, -1].detach().cpu().numpy()  # Take final state as goal

            rand_init_states.append(init_state)
            rand_goal_states.append(goal_state)

        # Convert to numpy arrays
        self.state_0 = np.array(rand_init_states)  # (n_evals, state_dim)
        self.state_g = np.array(rand_goal_states)  # (n_evals, state_dim)

        # For random_mlp, observations are just the state vectors with time dimension
        self.obs_0 = np.expand_dims(self.state_0, axis=1)  # (n_evals, 1, state_dim)
        self.obs_g = np.expand_dims(self.state_g, axis=1)  # (n_evals, 1, state_dim)

        self.gt_actions = None  # No ground truth actions for random_mlp

        print(f"Created random_mlp targets - Initial states: {self.state_0.shape}, Goal states: {self.state_g.shape}")

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        file_path = os.path.join(self.cfg_dict["saved_folder"], "plan_targets.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None

        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, e_states_all = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=self.save_images, filename="output_final"
        )

        # Create optimization GIF for ballnav if planner has visualizer
        if (self.save_images and
            self.env_name == "ballnav" and
            hasattr(self.planner, 'visualizer') and
            self.planner.visualizer is not None):
            print("Creating ballnav optimization GIF...")
            gif_path = os.path.join(self.cfg_dict["saved_folder"], "output_final_optimization.gif")
            self.planner.visualizer.create_optimization_gif(
                trajectory=e_states_all[0] if e_states_all else [],
                gif_path=gif_path
            )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}

        # Add planner stats if available (exit_step and total_measured_time)
        if hasattr(self.planner, 'exit_step'):
            logs["exit_step"] = self.planner.exit_step
        if hasattr(self.planner, 'total_measured_time'):
            logs["total_measured_time"] = self.planner.total_measured_time
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    # Pre-load dinov2 from torch.hub to make it available for unpickling
    # This is needed for old checkpoints that saved full model objects
    try:
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        _ = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", source="github", force_reload=False)
    except Exception as e:
        log.warning(f"Could not pre-load dinov2 from torch.hub: {e}")

    try:
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        _, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
    except Exception as e:
        log.warning(f"Could not pre-load vjepav2 from torch.hub: {e}")

    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            # Handle both old format (full model) and new format (state_dict)
            if isinstance(v, dict):
                # New format: state_dict - will be loaded later in load_model
                result[k] = v
            else:
                # Old format: full model object
                result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    # Special handling for ballnav - it's a direct world model, not a trained model
    if train_cfg.env.name == "ballnav":
        from env.ballnav.ballnav_world_model import BallNavWorldModel
        # Create ballnav world model with config parameters
        model = BallNavWorldModel(
            num_obstacles=train_cfg.model.get('num_obstacles', 1),
            device=str(device)
        )
        model.to(device)
        return model

    # Special handling for random_mlp - it's a direct world model, not a trained model
    if train_cfg.env.name == "random_mlp":
        from env.random_mlp.random_mlp_world_model import RandomMlpWorldModel
        # Create random_mlp world model with config parameters
        model = RandomMlpWorldModel(
            state_dim=train_cfg.model.get('state_dim', 64),
            action_dim=train_cfg.model.get('action_dim', 64),
            H=train_cfg.model.get('H', 10),
            device=str(device)
        )
        model.to(device)
        return model

    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    # Initialize encoder if not loaded or if it's a state_dict
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    elif isinstance(result["encoder"], dict):
        # New format: state_dict - need to instantiate model then load
        encoder = hydra.utils.instantiate(train_cfg.encoder)
        encoder.load_state_dict(result["encoder"])
        result["encoder"] = encoder.to(device)

    # Handle predictor
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")
    elif isinstance(result["predictor"], dict):
        # New format: state_dict
        predictor = hydra.utils.instantiate(
            train_cfg.predictor,
            num_patches=result.get("num_patches", 196),  # Will need proper handling
            num_frames=train_cfg.num_hist,
            dim=train_cfg.encoder.emb_dim + (train_cfg.proprio_emb_dim * train_cfg.num_proprio_repeat + train_cfg.action_emb_dim * train_cfg.num_action_repeat) * train_cfg.concat_dim
        )
        predictor.load_state_dict(result["predictor"])
        result["predictor"] = predictor.to(device)

    # Handle action_encoder and proprio_encoder
    if "action_encoder" not in result:
        result["action_encoder"] = hydra.utils.instantiate(train_cfg.action_encoder, in_chans=result.get("action_dim", 2), emb_dim=train_cfg.action_emb_dim)
    elif isinstance(result["action_encoder"], dict):
        action_encoder = hydra.utils.instantiate(train_cfg.action_encoder, in_chans=result.get("action_dim", 2), emb_dim=train_cfg.action_emb_dim)
        action_encoder.load_state_dict(result["action_encoder"])
        result["action_encoder"] = action_encoder.to(device)

    if "proprio_encoder" not in result:
        result["proprio_encoder"] = hydra.utils.instantiate(train_cfg.proprio_encoder, in_chans=result.get("proprio_dim", 2), emb_dim=train_cfg.proprio_emb_dim)
    elif isinstance(result["proprio_encoder"], dict):
        proprio_encoder = hydra.utils.instantiate(train_cfg.proprio_encoder, in_chans=result.get("proprio_dim", 2), emb_dim=train_cfg.proprio_emb_dim)
        proprio_encoder.load_state_dict(result["proprio_encoder"])
        result["proprio_encoder"] = proprio_encoder.to(device)

    # Handle decoder
    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError("Decoder path not found in model checkpoint and is not provided in config")
    elif train_cfg.has_decoder and isinstance(result.get("decoder"), dict):
        # New format: state_dict
        decoder = hydra.utils.instantiate(train_cfg.decoder, emb_dim=train_cfg.encoder.emb_dim)
        decoder.load_state_dict(result["decoder"])
        result["decoder"] = decoder.to(device)
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]

    model_name = cfg_dict['model_name']

    # Special handling for ballnav and random_mlp - they don't have trained models
    if model_name == "ballnav" or model_name == "random_mlp":
        # Load configuration directly from environment config file
        if cfg_dict.get("env", None) is not None:
            env_cfg = cfg_dict["env"]
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
            env_cfg_path = os.path.join(base_path, f"conf/env/{model_name}.yaml")
            env_cfg = OmegaConf.load(env_cfg_path)

        # Create a minimal model config structure
        model_cfg = OmegaConf.create({
            "env": env_cfg,
            "model": env_cfg.get("model", {}),
            "frameskip": 1,  # default frameskip
            "num_action_repeat": 1,  # default
            "has_decoder": False,
            "has_predictor": False,
        })
    else:
        model_path = f"{ckpt_base_path}/outputs/{model_name}/"
        with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])

    # Special handling for ballnav - create a dummy dataset
    if model_cfg.env.name == "ballnav":
        from types import SimpleNamespace
        dset = SimpleNamespace()
        dset.action_dim = 2
        dset.action_mean = np.zeros(2)
        dset.action_std = np.ones(2)
        dset.state_mean = np.array([5.0, 5.0])  # center of environment
        dset.state_std = np.array([3.0, 3.0])   # reasonable scale
        dset.proprio_mean = np.zeros(0)
        dset.proprio_std = np.ones(0)
        dset.transform = None
    # Special handling for random_mlp - create a dummy dataset
    elif model_cfg.env.name == "random_mlp":
        from types import SimpleNamespace
        state_dim = model_cfg.model.get('state_dim', 64)
        action_dim = model_cfg.model.get('action_dim', 64)
        dset = SimpleNamespace()
        dset.action_dim = action_dim
        dset.action_mean = np.zeros(action_dim)
        dset.action_std = np.ones(action_dim)
        dset.state_mean = np.zeros(state_dim)  # zero mean for high-dim states
        dset.state_std = np.ones(state_dim)    # unit variance
        dset.proprio_mean = np.zeros(0)
        dset.proprio_std = np.ones(0)
        dset.transform = None
    else:
        _, dset = hydra.utils.call(
            model_cfg.env.dataset,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
        dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat

    # Special handling for ballnav - create direct environment wrapper
    # load_model has specific cases for ballnav and random_mlp to instantiate
    # the hard-coded world models and return before trying to ref ckpt
    if model_cfg.env.name == "ballnav":
        from env.ballnav.ballnav_env_wrapper import BallNavEnvWrapper
        env = BallNavEnvWrapper(device=str(device))
        model = load_model(None, model_cfg, num_action_repeat, device=device)
    # Special handling for random_mlp - create direct environment wrapper
    elif model_cfg.env.name == "random_mlp":
        from env.random_mlp.random_mlp_env_wrapper import RandomMlpEnvWrapper
        env = RandomMlpEnvWrapper(
            device=str(device),
            state_dim=model_cfg.model.get('state_dim', 64),
            action_dim=model_cfg.model.get('action_dim', 64),
            H=model_cfg.model.get('H', 10)
        )
        model = load_model(None, model_cfg, num_action_repeat, device=device)
    # above two only models without trained models
    else:
        model_ckpt = (
            Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
        )
        model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)
        # use dummy vector env for wall env
        if model_cfg.env.name == "wall":
            from env.serial_vector_env import SerialVectorEnv
            env = SerialVectorEnv(
                [
                    gym.make(
                        model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                    )
                    for _ in range(cfg_dict["n_evals"])
                ]
            )
        else:
            env = SubprocVectorEnv(
                [
                    lambda: gym.make(
                        model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                    )
                    for _ in range(cfg_dict["n_evals"])
                ]
            )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan_grasp")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
