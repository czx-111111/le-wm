import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path


def convert_to_hdf5(data_path, out_dir, seed=42):
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pixels = []
    all_proprio = []
    all_actions = []
    all_states = []
    all_ep_idx = []
    all_step_idx = []
    ep_lens = []
    ep_counter = 0

    ep_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    print(f"[INFO] 找到 {len(ep_dirs)} 个子目录")

    for ep_dir in tqdm(ep_dirs):
        ep_path = os.path.join(data_path, ep_dir)
        
        h5_files = sorted([f for f in os.listdir(ep_path) if f.endswith('.h5')])
        if not h5_files:
            continue
        
        ep_pixels = []
        ep_proprio = []
        ep_action = []
        ep_state = []
        
        for hf in h5_files:
            with h5py.File(os.path.join(ep_path, hf), 'r') as f:
                pixels = f['observations/color/cam_0'][:]
                eef = f['eef_states'][:]
                action = f['action'][:]
                positions = f['positions'][:]
                
                if eef.ndim == 1:
                    eef = eef.reshape(1, -1)  # (14,) -> (1, 14)
                elif eef.ndim > 2:
                    eef = eef.reshape(eef.shape[0], -1)
                
                if action.ndim == 1:
                    action = action.reshape(1, -1)  # (4,) -> (1, 4)
                elif action.ndim > 2:
                    action = action.reshape(action.shape[0], -1)
                
                state = positions.reshape(positions.shape[0], -1)
                
                ep_pixels.append(pixels)
                ep_proprio.append(eef)
                ep_action.append(action)
                ep_state.append(state)
        
        if ep_pixels:
            pixels = np.concatenate(ep_pixels, axis=0)
            proprio = np.concatenate(ep_proprio, axis=0)
            action = np.concatenate(ep_action, axis=0)
            state = np.concatenate(ep_state, axis=0)
            
            T = pixels.shape[0]
            
            all_pixels.append(pixels)
            all_proprio.append(proprio)
            all_actions.append(action)
            all_states.append(state)
            all_ep_idx.append(np.full(T, ep_counter))
            all_step_idx.append(np.arange(T))
            ep_lens.append(T)
            
            ep_counter += 1

    pixels_all = np.concatenate(all_pixels, axis=0)
    proprio_all = np.concatenate(all_proprio, axis=0)
    action_all = np.concatenate(all_actions, axis=0)
    state_all = np.concatenate(all_states, axis=0)
    ep_idx_all = np.concatenate(all_ep_idx, axis=0)
    step_idx_all = np.concatenate(all_step_idx, axis=0)
    ep_len_all = np.array(ep_lens, dtype=np.int64)
    ep_offset_all = np.concatenate([[0], np.cumsum(ep_len_all[:-1])])

    print(f"Total time steps: {pixels_all.shape[0]}, Total episodes: {ep_counter}")
    print(f"action shape: {action_all.shape}")
    print(f"proprio shape: {proprio_all.shape}")
    print(f"state shape: {state_all.shape}")
    print(f"Episode lengths (first 10): {ep_len_all[:10]}")
    print(f"Total frames sum: {ep_len_all.sum()}")

    output_h5 = out_dir / "rope_eval.h5"
    with h5py.File(output_h5, 'w') as f_out:
        f_out.create_dataset('pixels', data=pixels_all)
        f_out.create_dataset('proprio', data=proprio_all)
        f_out.create_dataset('action', data=action_all)
        f_out.create_dataset('state', data=state_all)
        f_out.create_dataset('ep_idx', data=ep_idx_all)
        f_out.create_dataset('step_idx', data=step_idx_all)
        f_out.create_dataset('ep_len', data=ep_len_all)
        f_out.create_dataset('ep_offset', data=ep_offset_all)

    print(f"Saved to {output_h5}")


def main():
    parser = argparse.ArgumentParser(description="DINO-WM rope dataset -> SWM HDF5")
    parser.add_argument("--data-path", required=True, help="DINO-WM rope数据目录")
    parser.add_argument("--out-dir", default="./data", help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert_to_hdf5(args.data_path, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
