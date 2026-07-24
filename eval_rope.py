import sys
sys.path.insert(0, "/workspace/le-wm")
sys.path.insert(0, "/workspace/le-wm/deformable_env")

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import stable_pretraining as spt
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYFLEX_HEADLESS"] = "1"

sys.path.append(str(Path(__file__).parent / "dino_wm-main"))

import pyflex
pyflex.set_screenWidth(224)
pyflex.set_screenHeight(224)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.0)
pyflex.init(True)

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from deformable_env.FlexEnvWrapper import FlexEnvWrapper, chamfer_distance


def img_transform(img_size):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )
    return transform


def get_episodes_length(h5_path):
    with h5py.File(h5_path, "r") as f:
        episode_idx = f["ep_idx"][:]
        step_idx = f.get("step_idx")
        if step_idx is None:
            ep_len = f["ep_len"][:]
            ep_indices = np.arange(len(ep_len))
            return ep_indices, ep_len
        step_idx = step_idx[:]
    
    ep_indices = np.unique(episode_idx)
    lengths = []
    for ep_id in ep_indices:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return ep_indices, np.array(lengths)


def sample_eval_points(h5_path, num_eval, goal_offset_steps, seed=42):
    with h5py.File(h5_path, "r") as f:
        episode_idx = f["ep_idx"][:]
        step_idx = f.get("step_idx")
        
        if step_idx is None:
            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]
            
            max_start_idx = ep_len - goal_offset_steps - 1
            valid_ep_mask = max_start_idx >= 0
            valid_ep_indices = np.nonzero(valid_ep_mask)[0]
            max_start_idx = max_start_idx[valid_ep_mask]
            
            g = np.random.default_rng(seed)
            chosen_ep = g.choice(valid_ep_indices, size=num_eval, replace=False)
            
            chosen_starts = []
            for ep_id in chosen_ep:
                max_start = max_start_idx[list(valid_ep_indices).index(ep_id)]
                chosen_starts.append(g.integers(0, max_start + 1))
            
            eval_indices = ep_offset[chosen_ep] + chosen_starts
            eval_episodes = chosen_ep
            eval_start_idx = np.array(chosen_starts)
            
            print(f"{len(valid_ep_indices)} valid episodes found for evaluation.")
            return eval_indices, eval_episodes, eval_start_idx
        
        step_idx = step_idx[:]
    
    ep_indices, episode_len = get_episodes_length(h5_path)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in episode_idx]
    )
    valid_mask = step_idx <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")
    
    g = np.random.default_rng(seed)
    chosen = g.choice(len(valid_indices), size=num_eval, replace=False)
    chosen = np.sort(valid_indices[chosen])
    
    eval_episodes = episode_idx[chosen]
    eval_start_idx = step_idx[chosen]
    return chosen, eval_episodes, eval_start_idx


def build_goal_lookup(h5_path):
    with h5py.File(h5_path, "r") as f:
        episode_idx = f["ep_idx"][:]
        step_idx = f.get("step_idx")
        
        if step_idx is None:
            ep_offset = f["ep_offset"][:]
            ep_len = f["ep_len"][:]
            
            lookup = {}
            global_idx = 0
            for ep_id, offset, length in zip(range(len(ep_len)), ep_offset, ep_len):
                for step in range(length):
                    lookup[(int(ep_id), int(step))] = int(offset + step)
            return lookup
        
        step_idx = step_idx[:]
    
    lookup = {}
    for global_idx, (ep, step) in enumerate(zip(episode_idx, step_idx)):
        lookup[(int(ep), int(step))] = int(global_idx)
    return lookup


def load_frame(h5_path, idx):
    with h5py.File(h5_path, "r") as f:
        pixels = f["pixels"][idx]
        action = f["action"][idx] if "action" in f else np.zeros(4)
        proprio = f["proprio"][idx] if "proprio" in f else np.zeros(1)
        state = f["state"][idx] if "state" in f else None
    return pixels, action, proprio, state


def load_sequence(h5_path, start_idx, history_size, lookup, episode_id, start_step):
    pixels_list = []
    action_list = []
    proprio_list = []
    
    for t in range(history_size):
        step = int(start_step + t)
        idx = lookup[(int(episode_id), step)]
        pixels, action, proprio, _ = load_frame(h5_path, idx)
        pixels_list.append(pixels)
        action_list.append(action)
        proprio_list.append(proprio)
    
    return {
        "pixels": torch.from_numpy(np.stack(pixels_list)),
        "action": torch.from_numpy(np.stack(action_list)),
        "proprio": torch.from_numpy(np.stack(proprio_list)),
    }


def build_processors(h5_path, keys_to_process):
    process = {}
    with h5py.File(h5_path, "r") as f:
        for col in keys_to_process:
            if col not in f:
                continue
            processor = preprocessing.StandardScaler()
            col_data = f[col][:]
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
            process[col] = processor
            if col != "action":
                process[f"goal_{col}"] = processor
    return process


def apply_processors(sample, process):
    out = dict(sample)
    for col, proc in process.items():
        if col.startswith("goal_"):
            continue
        if col not in out:
            continue
        arr = out[col]
        if torch.is_tensor(arr):
            arr = arr.numpy()
        out[col] = torch.from_numpy(proc.transform(arr)).float()
    return out


def build_info_dict(clip, goal_pixels, goal_proprio, process, transform, device):
    clip = apply_processors(clip, process)
    
    pixels = clip["pixels"].float()
    print(f"  [DEBUG] pixels shape before permute: {pixels.shape}")
    if pixels.ndim == 4:
        pixels = pixels.permute(0, 3, 1, 2)
        print(f"  [DEBUG] pixels shape after permute: {pixels.shape}")
        pixels = transform["pixels"](pixels)
    pixels = pixels.unsqueeze(0).to(device)
    print(f"  [DEBUG] pixels shape after transform: {pixels.shape}")
    
    action = torch.nan_to_num(clip["action"], 0.0).unsqueeze(0).to(device)
    info = {"pixels": pixels, "action": action}
    
    if "proprio" in clip:
        info["proprio"] = clip["proprio"].unsqueeze(0).to(device)
    
    goal_img = torch.from_numpy(np.array(goal_pixels))
    if goal_img.ndim == 3 and goal_img.shape[-1] in (1, 3):
        goal_img = goal_img.permute(2, 0, 1)
    goal_img = transform["goal"](goal_img).unsqueeze(0).to(device)
    info["goal"] = goal_img
    
    goal_prop = torch.from_numpy(np.array(goal_proprio, dtype=np.float32)).unsqueeze(0)
    if "goal_proprio" in process:
        goal_prop = torch.from_numpy(process["goal_proprio"].transform(goal_prop.numpy())).float()
    info["goal_proprio"] = goal_prop.to(device)
    return info


def cem_plan(model, info, horizon, action_dim, num_samples=128, n_steps=5, topk=16, var_scale=0.5, device="cuda"):
    history = info["pixels"].shape[1]
    total_steps = history + horizon
    
    mean = torch.zeros(1, total_steps, action_dim, device=device)
    std = torch.ones(1, total_steps, action_dim, device=device) * var_scale
    mean[:, :history] = info["action"][:, :history]
    
    best_cost = None
    best_actions = None
    
    for _ in range(n_steps):
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
    
    return best_actions[history:], best_cost


def load_lewm_model(ckpt_path, effective_act_dim, history_size=3, embed_dim=192,
                    encoder_scale="tiny", patch_size=14, img_size=224,
                    predictor_depth=6, predictor_heads=16, predictor_mlp_dim=2048,
                    predictor_dim_head=64, predictor_dropout=0.1, predictor_emb_dropout=0.0):
    encoder = spt.backbone.utils.vit_hf(
        encoder_scale,
        patch_size=patch_size,
        image_size=img_size,
        pretrained=False,
        use_mask_token=False,
    )
    
    hidden_dim = encoder.config.hidden_size
    
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=predictor_depth,
        heads=predictor_heads,
        mlp_dim=predictor_mlp_dim,
        dim_head=predictor_dim_head,
        dropout=predictor_dropout,
        emb_dropout=predictor_emb_dropout,
    )
    
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )
    
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state_dict[k[6:]] = v
        elif k.startswith("module.model."):
            model_state_dict[k[13:]] = v
        else:
            model_state_dict[k] = v
    
    result = model.load_state_dict(model_state_dict, strict=False)
    
    if result.missing_keys:
        print(f"[WARN] Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARN] Unexpected keys: {result.unexpected_keys}")
    
    model.eval()
    model.requires_grad_(False)
    
    if hasattr(model.encoder, "interpolate_pos_encoding"):
        model.encoder.interpolate_pos_encoding = True
    
    return model


def run_sim_eval(h5_path, model, process, transform, device,
                 num_eval=50, goal_offset_steps=15, horizon=5,
                 history_size=3, action_dim=4, seed=42,
                 cem_num_samples=128, cem_n_steps=5, cem_topk=16, cem_var_scale=0.5,
                 success_threshold=0.0):
    
    env = FlexEnvWrapper("rope")
    print("[INFO] 仿真环境初始化完成")
    
    lookup = build_goal_lookup(h5_path)
    eval_indices, eval_episodes, eval_start_idx = sample_eval_points(
        h5_path, num_eval, goal_offset_steps, seed
    )
    
    successes = []
    chamfer_distances = []
    planning_costs = []
    
    start_time = time.time()
    
    for i, (ep_id, start_step) in enumerate(zip(eval_episodes, eval_start_idx)):
        print(f"\n[DEBUG] Eval {i+1}/{num_eval} | Episode {ep_id}, Start step {start_step}")
        
        goal_step = int(start_step + goal_offset_steps)
        
        start_idx = lookup[(int(ep_id), int(start_step))]
        goal_idx = lookup[(int(ep_id), goal_step)]
        
        print("[DEBUG] Loading start frame...")
        start_pixels, _, start_proprio, init_state = load_frame(h5_path, start_idx)
        print("[DEBUG] Start frame loaded")
        
        print("[DEBUG] Loading goal frame...")
        goal_pixels, _, goal_proprio, goal_state = load_frame(h5_path, goal_idx)
        print("[DEBUG] Goal frame loaded")
        
        if init_state is None:
            print(f"[WARN] Episode {ep_id} step {start_step} 没有 state 数据，跳过")
            continue
        
        print("[DEBUG] Loading sequence...")
        clip = load_sequence(h5_path, start_idx, history_size, lookup, ep_id, start_step)
        print("[DEBUG] Sequence loaded")
        
        print("[DEBUG] Building info dict...")
        info = build_info_dict(clip, goal_pixels, goal_proprio, process, transform, device)
        print("[DEBUG] Info dict built")
        
        print("[DEBUG] Running CEM planning...")
        planned_actions, plan_cost = cem_plan(
            model, info, horizon, action_dim,
            num_samples=cem_num_samples, n_steps=cem_n_steps,
            topk=cem_topk, var_scale=cem_var_scale, device=device
        )
        print("[DEBUG] CEM planning completed")
        
        planned_actions_np = planned_actions.cpu().numpy()
        
        print(f"\nEval {i+1}/{num_eval} | Episode {ep_id}")
        print(f"  Start step: {start_step}, Goal step: {goal_step}")
        print(f"  Planning cost: {plan_cost:.4f}")
        print(f"  Planned actions shape: {planned_actions_np.shape}")
        
        init_state_reshaped = init_state.reshape(-1, 4).astype(np.float32)
        print(f"  init_state shape: {init_state.shape}, reshaped: {init_state_reshaped.shape}, dtype: {init_state_reshaped.dtype}")
        
        print("  [DEBUG] Calling env.prepare...")
        obs, state_dct = env.prepare(seed + i, init_state_reshaped)
        print("  [DEBUG] env.prepare completed")
        
        print(f"  [DEBUG] Executing {len(planned_actions_np)} actions...")
        for t, action in enumerate(planned_actions_np):
            print(f"    [DEBUG] Step {t}, action: {action}")
            obses, rewards, dones, infos = env.step_multiple([action])
            print(f"    [DEBUG] Step {t} completed")
        
        print("  [DEBUG] Getting final state...")
        final_state = infos["state"][-1]
        
        if goal_state is not None:
            goal_state_reshaped = goal_state.reshape(-1, 4).astype(np.float32)
            cd = chamfer_distance(torch.tensor([goal_state_reshaped]), torch.tensor([final_state])).item()
            success = cd < success_threshold
        else:
            cd = -1
            success = False
        
        successes.append(success)
        chamfer_distances.append(cd)
        planning_costs.append(plan_cost)
        
        print(f"  Chamfer distance: {cd:.4f}")
        print(f"  Success: {success}")
    
    env.close()
    
    elapsed = time.time() - start_time
    valid_eval = len(successes)
    
    metrics = {
        "mode": "sim",
        "num_eval": valid_eval,
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_chamfer_distance": float(np.mean([d for d in chamfer_distances if d >= 0])),
        "mean_planning_cost": float(np.mean(planning_costs)),
        "evaluation_time_sec": elapsed,
    }
    
    return metrics


def write_results(metrics, results_path):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        f.write("==== SIMULATION EVALUATION RESULTS ====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def main():
    parser = argparse.ArgumentParser(description="LeWM Rope 仿真评估")
    parser.add_argument("--dataset", required=True, help="HDF5 数据集路径")
    parser.add_argument("--ckpt", required=True, help="模型权重路径")
    parser.add_argument("--num-eval", type=int, default=50, help="评估样本数")
    parser.add_argument("--goal-offset", type=int, default=15, help="目标偏移步数")
    parser.add_argument("--horizon", type=int, default=5, help="规划视野")
    parser.add_argument("--history-size", type=int, default=3, help="历史帧数")
    parser.add_argument("--action-dim", type=int, default=4, help="动作维度")
    parser.add_argument("--frameskip", type=int, default=1, help="帧跳过")
    parser.add_argument("--img-size", type=int, default=224, help="图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--success-threshold", type=float, default=0.0, help="成功阈值(chamfer distance)")
    parser.add_argument("--output", default="./outputs/rope_sim_results.txt", help="输出路径")
    
    parser.add_argument("--embed-dim", type=int, default=192, help="嵌入维度")
    parser.add_argument("--encoder-scale", default="tiny", help="编码器规模")
    parser.add_argument("--patch-size", type=int, default=14, help="patch尺寸")
    parser.add_argument("--predictor-depth", type=int, default=6, help="预测器深度")
    parser.add_argument("--predictor-heads", type=int, default=16, help="预测器头数")
    parser.add_argument("--predictor-mlp-dim", type=int, default=2048, help="预测器MLP维度")
    parser.add_argument("--predictor-dim-head", type=int, default=64, help="预测器头维度")
    parser.add_argument("--predictor-dropout", type=float, default=0.1, help="预测器dropout")
    
    parser.add_argument("--cem-num-samples", type=int, default=128, help="CEM样本数")
    parser.add_argument("--cem-n-steps", type=int, default=5, help="CEM迭代步数")
    parser.add_argument("--cem-topk", type=int, default=16, help="CEM topk")
    parser.add_argument("--cem-var-scale", type=float, default=0.5, help="CEM方差缩放")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="计算设备")
    
    args = parser.parse_args()
    
    os.environ["MUJOCO_GL"] = "egl"
    
    dataset_path = Path(args.dataset)
    ckpt_path = Path(args.ckpt)
    
    if not dataset_path.exists():
        print(f"[ERROR] 数据集文件不存在: {dataset_path}")
        return
    if not ckpt_path.exists():
        print(f"[ERROR] 权重文件不存在: {ckpt_path}")
        return
    
    print("="*60)
    print("LeWM Rope 仿真评估")
    print("="*60)
    print(f"数据集: {dataset_path}")
    print(f"权重: {ckpt_path}")
    print(f"评估样本数: {args.num_eval}")
    print(f"目标偏移步数: {args.goal_offset}")
    print(f"规划视野: {args.horizon}")
    print(f"历史帧数: {args.history_size}")
    print(f"动作维度: {args.action_dim}")
    print(f"成功阈值: {args.success_threshold}")
    
    effective_act_dim = args.frameskip * args.action_dim
    
    process = build_processors(str(dataset_path), ["action", "proprio"])
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }
    
    print("\n[INFO] 加载模型...")
    model = load_lewm_model(
        str(ckpt_path),
        effective_act_dim=effective_act_dim,
        history_size=args.history_size,
        embed_dim=args.embed_dim,
        encoder_scale=args.encoder_scale,
        patch_size=args.patch_size,
        img_size=args.img_size,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_dim_head=args.predictor_dim_head,
        predictor_dropout=args.predictor_dropout,
    )
    print("[INFO] 模型加载完成")
    
    device = torch.device(args.device)
    model = model.to(device)
    print(f"[INFO] 使用设备: {device}")
    
    print("\n[INFO] 开始仿真评估...")
    metrics = run_sim_eval(
        str(dataset_path), model, process, transform, device,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset,
        horizon=args.horizon,
        history_size=args.history_size,
        action_dim=args.action_dim,
        seed=args.seed,
        cem_num_samples=args.cem_num_samples,
        cem_n_steps=args.cem_n_steps,
        cem_topk=args.cem_topk,
        cem_var_scale=args.cem_var_scale,
        success_threshold=args.success_threshold,
    )
    
    print("\n" + "="*60)
    print("仿真评估结果:")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    write_results(metrics, args.output)
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
