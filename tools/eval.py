import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    """Get length of each episode."""
    if "ep_len" in dataset.column_names:
        return dataset.get_col_data("ep_len")
    
    episode_idx = dataset.get_col_data("episode_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.sum(episode_idx == ep_id))
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    ds_cfg = cfg.dataset
    kwargs = {
        "name": dataset_name,
        "keys_to_cache": ds_cfg.keys_to_cache,
        "cache_dir": dataset_path,
    }
    if "keys_to_load" in ds_cfg:
        kwargs["keys_to_load"] = ds_cfg.keys_to_load
    if "num_steps" in ds_cfg:
        kwargs["num_steps"] = ds_cfg.num_steps
    if "frameskip" in ds_cfg:
        kwargs["frameskip"] = ds_cfg.frameskip
    return swm.data.HDF5Dataset(**kwargs)


def build_processors(stats_dataset, keys_to_cache):
    process = {}
    for col in keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor
    return process


def apply_processors(sample, process):
    out = dict(sample)
    for col, proc in process.items():
        if col.startswith("goal_") or col == "action":
            continue
        if col not in out:
            continue
        arr = out[col]
        if torch.is_tensor(arr):
            arr = arr.numpy()
        n_features = proc.n_features_in_
        orig_shape = arr.shape
        arr_2d = arr.reshape(-1, n_features)
        transformed = proc.transform(arr_2d)
        out[col] = torch.from_numpy(transformed).float().reshape(orig_shape)
    return out


def sample_eval_points(dataset, cfg):
    col_name = "episode_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    chosen = g.choice(valid_indices, size=cfg.eval.num_eval, replace=False)
    chosen = np.sort(chosen)

    eval_episodes = dataset.get_col_data(col_name)[chosen]
    eval_start_idx = dataset.get_col_data("step_idx")[chosen]
    return chosen, eval_episodes, eval_start_idx


def build_goal_lookup(h5_path):
    with h5py.File(h5_path, "r") as f:
        episode_idx = f["episode_idx"][:]
        step_idx = f["step_idx"][:]
        ep_offset = f["ep_offset"][:]

    lookup = {}
    for global_idx, (ep, step) in enumerate(zip(episode_idx, step_idx)):
        lookup[(int(ep), int(step))] = global_idx

    meta = {
        "episode_idx": episode_idx,
        "episode_idx": episode_idx,
        "ep_offset": ep_offset,
    }
    return lookup, meta


def load_goal_frame(h5_path, episode_id, goal_step, lookup):
    key = (int(episode_id), int(goal_step))
    if key not in lookup:
        raise KeyError(f"goal frame not found: episode={episode_id}, step={goal_step}")
    idx = lookup[key]
    with h5py.File(h5_path, "r") as f:
        pixels = f["pixels"][idx]
        proprio = f["proprio"][idx]
    return pixels, proprio


def build_expert_actions(h5_path, episode_id, start_step, history_size, horizon, action_dim, lookup):
    actions = []
    with h5py.File(h5_path, "r") as f:
        action_all = f["action"]
        for t in range(history_size + horizon):
            step = int(start_step + t)
            idx = lookup[(int(episode_id), step)]
            actions.append(action_all[idx])
    actions = np.stack(actions, axis=0).astype(np.float32)
    # 如果模型期望220维但数据是44维，扩展维度
    if action_dim == 220 and actions.shape[-1] == 44:
        actions = np.tile(actions, (1, 5))  # 44 * 5 = 220
    elif actions.shape[-1] != action_dim:
        raise ValueError(
            f"action dim mismatch: expected {action_dim}, got {actions.shape[-1]}"
        )
    return torch.from_numpy(actions)


def build_info_dict(clip, goal_pixels, goal_proprio, process, transform, device, action_dim=None):
    clip = apply_processors(clip, process)

    pixels = clip["pixels"].float()
    if pixels.ndim == 4:
        pixels = transform["pixels"](pixels)
    pixels = pixels.unsqueeze(0).to(device)

    action_raw = torch.nan_to_num(clip['action'], 0.0)
    if action_dim is not None and action_raw.shape[-1] != action_dim:
        # Reshape flattened multi-step action: (history, steps*action_dim) -> (history, steps, action_dim) -> mean
        n_steps = action_raw.shape[-1] // action_dim
        action_raw = action_raw.reshape(action_raw.shape[0], n_steps, action_dim).mean(dim=1)
    action = action_raw.unsqueeze(0).to(device)
    info = {"pixels": pixels, "action": action}

    if "proprio" in clip:
        info["proprio"] = clip["proprio"].unsqueeze(0).to(device)

    goal_img = torch.from_numpy(np.array(goal_pixels))
    if goal_img.ndim == 3 and goal_img.shape[-1] in (1, 3):
        goal_img = goal_img.permute(2, 0, 1)
    goal_img = transform["goal"](goal_img).unsqueeze(0).to(device)  # (1, 3, H, W)
    info["goal"] = goal_img

    goal_prop = torch.from_numpy(np.array(goal_proprio, dtype=np.float32)).unsqueeze(0)  # (1, 193)
    goal_prop = torch.from_numpy(process["goal_proprio"].transform(goal_prop.numpy())).float()  # (1, 193)
    info["goal_proprio"] = goal_prop.to(device)
    return info


def cem_plan(model, info, horizon, action_dim, solver_cfg, device):
    history = info["pixels"].shape[1]
    print(f"DEBUG cem_plan: history={history}, action.shape={info['action'].shape}", flush=True)
    total_steps = history + horizon
    num_samples = int(solver_cfg.num_samples)
    n_steps = int(solver_cfg.n_steps)
    topk = int(solver_cfg.topk)
    var_scale = float(solver_cfg.var_scale)

    # info["action"] is (1, history, flattened_dim), reshape to (1, history, action_dim)
    action_flat = info["action"]
    if action_flat.shape[-1] != action_dim:
        n_steps = action_flat.shape[-1] // action_dim
        action_2d = action_flat.reshape(1, history, n_steps, action_dim).mean(dim=2)
    else:
        action_2d = action_flat
    mean = torch.zeros(1, total_steps, action_dim, device=device)
    std = torch.ones(1, total_steps, action_dim, device=device) * var_scale
    mean[:, :history] = action_2d[:, :history]

    best_cost = None
    best_actions = None

    for iter_idx in range(n_steps):
        print(f"DEBUG cem iter {iter_idx}: action.shape={info['action'].shape}, mean.shape={mean.shape}", flush=True)
        noise = torch.randn(1, num_samples, total_steps, action_dim, device=device)
        candidates = mean.unsqueeze(1) + std.unsqueeze(1) * noise
        candidates[:, :, :history] = info["action"][:, :history].unsqueeze(1)
        costs = model.get_cost(info, candidates)
        k = min(topk, num_samples)
        _, idx = torch.topk(costs, k, dim=1, largest=False)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, total_steps, action_dim)
        elite = torch.gather(candidates, 1, idx_exp)
        mean = elite.mean(dim=1)
        std = elite.std(dim=1, unbiased=False).clamp_min(0.05)
        mean[:, :history] = info["action"][:, :history]

        min_cost, min_idx = costs.min(dim=1)
        if best_cost is None or min_cost.item() < best_cost:
            best_cost = min_cost.item()
            best_actions = candidates[0, min_idx.item()]

    return best_actions, best_cost


def evaluate_cost(model, info, actions, device):
    candidates = actions.unsqueeze(0).unsqueeze(0).to(device)
    return model.get_cost(info, candidates)[0, 0].item()


def run_offline_eval(cfg, dataset, model, process, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.requires_grad_(False)

    history_size = int(cfg.wm.history_size)
    horizon = int(cfg.plan_config.horizon)
    action_dim = 220  # 模型期望的维度
    lookup, _ = build_goal_lookup(dataset.h5_path)

    eval_indices, eval_episodes, eval_start_idx = sample_eval_points(dataset, cfg)

    planning_costs = []
    expert_costs = []
    successes = []

    start_time = time.time()
    for clip_idx, ep_id, start_step in zip(eval_indices, eval_episodes, eval_start_idx):
        clip = dataset[int(clip_idx)]
        print(f"DEBUG clip action shape: {clip['action'].shape}", flush=True)
        goal_step = int(start_step + cfg.eval.goal_offset_steps)
        goal_pixels, goal_proprio = load_goal_frame(
            dataset.h5_path, ep_id, goal_step, lookup
        )

        info = build_info_dict(
            clip, goal_pixels, goal_proprio, process, transform, device
        )
        expert_actions = build_expert_actions(
            dataset.h5_path,
            ep_id,
            start_step,
            history_size=info["action"].shape[1],
            horizon=horizon,
            action_dim=action_dim,
            lookup=lookup,
        )

        _, plan_cost = cem_plan(
            model,
            info,
            horizon=horizon,
            action_dim=action_dim,
            solver_cfg=cfg.solver,
            device=device,
        )
        expert_cost = evaluate_cost(model, info, expert_actions, device)

        planning_costs.append(plan_cost)
        expert_costs.append(expert_cost)
        ratio = float(cfg.eval.get("success_ratio", 1.0))
        successes.append(plan_cost <= expert_cost * ratio)

    elapsed = time.time() - start_time
    metrics = {
        "mode": "offline",
        "num_eval": len(planning_costs),
        "mean_planning_cost": float(np.mean(planning_costs)),
        "mean_expert_cost": float(np.mean(expert_costs)),
        "median_planning_cost": float(np.median(planning_costs)),
        "success_rate": float(np.mean(successes) * 100.0),
        "evaluation_time_sec": elapsed,
    }
    return metrics


def run_sim_eval(cfg, dataset, policy, process, transform):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    eval_indices, eval_episodes, eval_start_idx = sample_eval_points(dataset, cfg)
    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    world.set_policy(policy)
    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    metrics["evaluation_time_sec"] = time.time() - start_time
    metrics["mode"] = "sim"
    return metrics, results_path


def write_results(cfg, metrics, results_path):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    mode = cfg.eval.get("mode", cfg.get("mode", "sim"))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = build_processors(dataset, cfg.dataset.keys_to_cache)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    if mode == "offline":
        if cfg.get("policy", "random") == "random":
            raise ValueError("offline 评估需要指定训练好的 checkpoint: policy=你的job_id/lewm")

        model = swm.policy.AutoCostModel(cfg.policy)
        if hasattr(model, "interpolate_pos_encoding"):
            model.interpolate_pos_encoding = True

        metrics = run_offline_eval(cfg, dataset, model, process, transform)
        results_path = Path(cfg.output.get("dir", Path(__file__).parent)) / cfg.output.filename
        print(metrics)
        write_results(cfg, metrics, results_path)
        return

    if cfg.get("policy", "random") != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    metrics, results_path = run_sim_eval(cfg, dataset, policy, process, transform)
    print(metrics)
    write_results(cfg, metrics, results_path / cfg.output.filename)


if __name__ == "__main__":
    run()
