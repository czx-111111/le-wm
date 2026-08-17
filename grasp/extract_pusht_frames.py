#!/usr/bin/env python3
"""
Extract frames from PushT video episodes and save as PNG images.
This helps with data loading speed by avoiding real-time MP4 decoding.
"""

import os
import sys
from pathlib import Path
import decord
from decord import VideoReader
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial

def extract_frames_from_video(args):
    """Extract all frames from a single video and save as PNG files."""
    video_path, output_dir, video_name = args
    try:
        # Create output directory for this video
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load video
        reader = VideoReader(str(video_path))
        num_frames = len(reader)
        
        # Extract each frame
        for frame_idx in range(num_frames):
            frame = reader[frame_idx]  # Get frame as decord NDArray
            frame_np = frame.asnumpy()  # Convert to numpy array (H, W, C)
            
            # Convert to PIL Image and save
            frame_pil = Image.fromarray(frame_np)
            frame_path = video_output_dir / f"frame_{frame_idx:03d}.png"
            frame_pil.save(frame_path)
        
        return (video_name, num_frames, True)
        
    except Exception as e:
        return (video_name, 0, False, str(e))

def process_episode_folder(input_folder, output_folder, split_name, num_processes=64):
    """Process all videos in a folder (train or val) using multiprocessing.
    Writes to output_folder/split_name/obses/ so the loader finds frames at data_path/obses."""
    input_path = Path(input_folder)
    output_path = Path(output_folder) / split_name / "obses"
    
    if not input_path.exists():
        print(f"Input folder does not exist: {input_folder}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all MP4 files
    video_files = list(input_path.glob("*.mp4"))
    video_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(video_files)} videos in {split_name}")
    print(f"Using {num_processes} processes for parallel processing")
    
    # Prepare arguments for multiprocessing
    video_args = [(video_file, output_path, video_file.stem) for video_file in video_files]
    
    total_frames = 0
    successful_videos = 0
    failed_videos = []
    
    # Use multiprocessing with progress bar
    with mp.Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(extract_frames_from_video, video_args),
            total=len(video_args),
            desc=f"Processing {split_name}"
        ))
    
    # Process results
    for result in results:
        if len(result) == 3:  # Success case
            video_name, frames_extracted, success = result
            if success:
                total_frames += frames_extracted
                successful_videos += 1
        else:  # Error case
            video_name, frames_extracted, success, error = result
            failed_videos.append((video_name, error))
    
    print(f"Successfully processed {successful_videos}/{len(video_files)} videos")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output saved to: {output_path}")
    
    if failed_videos:
        print(f"Failed videos: {len(failed_videos)}")
        for video_name, error in failed_videos[:5]:  # Show first 5 errors
            print(f"  {video_name}: {error}")
        if len(failed_videos) > 5:
            print(f"  ... and {len(failed_videos) - 5} more")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from PushT video episodes")
    parser.add_argument("--input-train", 
                       default="/checkpoint/amaia/video/basileterv/dino_wm/data/pusht_noise/train/obses/",
                       help="Input folder for train videos")
    parser.add_argument("--input-val", 
                       default="/checkpoint/amaia/video/basileterv/dino_wm/data/pusht_noise/val/obses/",
                       help="Input folder for val videos")
    parser.add_argument("--output", 
                       default="/checkpoint/amaia/video/amirbar/datasets/pusht",
                       help="Output folder for extracted frames")
    parser.add_argument("--split", 
                       choices=["train", "val", "both"], 
                       default="both",
                       help="Which split to process")
    parser.add_argument("--num-processes", 
                       type=int, 
                       default=64,
                       help="Number of parallel processes to use")
    
    args = parser.parse_args()
    
    print("PushT Frame Extraction Script (Multiprocessing)")
    print("=" * 60)
    print(f"Input train folder: {args.input_train}")
    print(f"Input val folder: {args.input_val}")
    print(f"Output folder: {args.output}")
    print(f"Processing split: {args.split}")
    print(f"Number of processes: {args.num_processes}")
    print("=" * 60)
    
    # Process train split
    if args.split in ["train", "both"]:
        print("\nProcessing TRAIN split...")
        process_episode_folder(args.input_train, args.output, "train", args.num_processes)
    
    # Process val split
    if args.split in ["val", "both"]:
        print("\nProcessing VAL split...")
        process_episode_folder(args.input_val, args.output, "val", args.num_processes)
    
    print("\nFrame extraction completed!")

if __name__ == "__main__":
    main()
