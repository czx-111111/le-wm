import h5py 
import numpy as np 
import os 
from tqdm import tqdm 

DATA_DIR = "/mnt/data/wangjianfeng/qwh/czx/le-wm/data/deformable/rope" 
OUTPUT_PATH = "/mnt/data/wangjianfeng/qwh/czx/le-wm/data/rope.h5" 
CAM_ID = "cam_0" 

all_pixels = [] 
all_proprio = [] 
all_actions = [] 
all_ep_idx = [] 
ep_lens = [] 
ep_counter = 0 

ep_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]) 

for ep_dir in tqdm(ep_dirs): 
    ep_path = os.path.join(DATA_DIR, ep_dir) 
    
    h5_files = sorted([f for f in os.listdir(ep_path) if f.endswith('.h5')]) 
    if not h5_files: 
        continue 
    
    ep_pixels = [] 
    ep_proprio = [] 
    ep_action = [] 
    
    for hf in h5_files: 
        with h5py.File(os.path.join(ep_path, hf), 'r') as f: 
            pixels = f[f'observations/color/{CAM_ID}'][:] 
            eef = f['eef_states'][:] 
            action = f['action'][:] 
            
            # 修复：(4,) -> (1, 4), (14,) -> (1, 14)
            if eef.ndim == 1: 
                eef = eef.reshape(1, -1)  # 修复！
            elif eef.ndim > 2: 
                eef = eef.reshape(eef.shape[0], -1) 
            
            if action.ndim == 1: 
                action = action.reshape(1, -1)  # 修复！
            elif action.ndim > 2: 
                action = action.reshape(action.shape[0], -1) 
            
            ep_pixels.append(pixels) 
            ep_proprio.append(eef) 
            ep_action.append(action) 
    
    if ep_pixels: 
        pixels = np.concatenate(ep_pixels, axis=0) 
        proprio = np.concatenate(ep_proprio, axis=0) 
        action = np.concatenate(ep_action, axis=0) 
        
        T = pixels.shape[0] 
        
        all_pixels.append(pixels) 
        all_proprio.append(proprio) 
        all_actions.append(action) 
        all_ep_idx.append(np.full(T, ep_counter)) 
        ep_lens.append(T) 
        
        ep_counter += 1 

pixels_all = np.concatenate(all_pixels, axis=0) 
proprio_all = np.concatenate(all_proprio, axis=0) 
action_all = np.concatenate(all_actions, axis=0) 
ep_idx_all = np.concatenate(all_ep_idx, axis=0) 
ep_len_all = np.array(ep_lens, dtype=np.int64) 
ep_offset_all = np.concatenate([[0], np.cumsum(ep_len_all[:-1])]) 

print(f"Total time steps: {pixels_all.shape[0]}, Total episodes: {ep_counter}") 
print(f"action shape: {action_all.shape}") 
print(f"proprio shape: {proprio_all.shape}") 
print(f"Episode lengths (first 10): {ep_len_all[:10]}") 
print(f"Total frames sum: {ep_len_all.sum()}") 

with h5py.File(OUTPUT_PATH, 'w') as f_out: 
    f_out.create_dataset('pixels', data=pixels_all) 
    f_out.create_dataset('proprio', data=proprio_all) 
    f_out.create_dataset('action', data=action_all) 
    f_out.create_dataset('ep_idx', data=ep_idx_all) 
    f_out.create_dataset('ep_len', data=ep_len_all) 
    f_out.create_dataset('ep_offset', data=ep_offset_all) 

print(f"Saved to {OUTPUT_PATH}") 
