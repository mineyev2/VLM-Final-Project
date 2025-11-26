#!/usr/bin/env python3
"""
Evaluation Script
Specific Config:
- History Input: Last 3 Frames ONLY (GLOBAL COORDINATES)
- Metric: ADE/FDE (filtered by Failure > 10m)
- Coordinate System: PURE GLOBAL (No local transforms except for viz)
"""

import argparse
import os
import csv
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored

# Local imports
from src.models.qwen_clip_model import QwenCLIPModel
from scripts.project_utils import parse_coords_from_text, visualize_trajectories
from scripts.nuscenes_dataset import NuScenesDataset


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda': torch.cuda.empty_cache()

    ########################### CHANGE MODEL LOADING TO LOAD ANY REQUESTED MODEL ##############################
    # Load Model
    model = QwenCLIPModel(device, qwen_model_name=args.llm, checkpoint_path=None)
    if args.checkpoint:
        print(colored(f"Loading checkpoint: {args.checkpoint}", "yellow"))
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'language_model_state_dict' in ckpt:
            model.language_model.load_state_dict(ckpt['language_model_state_dict'], strict=False)
            model.vision_tower.load_state_dict(ckpt['vision_tower_state_dict'])
            try:
                model.mlp_projector.load_state_dict(ckpt['mlp_projector_state_dict'])
            except:
                model.mlp_projector.load_state_dict(ckpt['model_state_dict'])
        else:
            model.mlp_projector.load_state_dict(ckpt)
    
    model.eval()
    #############################################################################################################

    # Dataset
    ds = NuScenesDataset(args.version, args.dataroot, model.prompt_part1, model.prompt_part2)
    n_samples = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))
    
    # Vis indices
    vis_indices = set(np.random.choice(n_samples, size=min(args.num_vis, n_samples), replace=False)) if args.num_vis > 0 else set()
    vis_dir = os.path.join(args.output_dir, 'visualizations')

    # CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_name)
    csv_file = open(csv_path, 'w', newline='')
    fieldnames = ['idx', 'num_valid_waypoints', 'format_compliant', 'ade', 'fde', 
                  'failure_rate', 'error_at_1s', 'processing_time_sec'] + \
                 [f'wp{i}_error' for i in range(10)] + ['gen_text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Save Args
    with open(os.path.join(args.output_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    results = []
    batch_size = args.batch_size
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    print(colored("Configuration: Global Coordinates. Sending last 10 history frames.", "magenta"))

    for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
        start_t = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        # Load Batch
        batch_data = [ds.get_item(i) for i in batch_indices]
        images = [b['image'] for b in batch_data]
        lidar = [b['torch_pointcloud'] for b in batch_data]
        ego_pos_global = [b['ego_positions'] for b in batch_data]

        # Generate (Model produces Global Coords if trained on Global)
        try:
            outputs, gen_texts = model.generateMotion(images, lidar, ego_pos_global)
        except Exception as e:
            print(f"Batch failed: {e}")
            continue

        batch_proc_time = time.time() - start_t
        per_sample_time = batch_proc_time / len(batch_indices)

        for i, idx in enumerate(batch_indices):
            text = gen_texts[i]
            # Parse Global Coords directly from text
            pred_global = parse_coords_from_text(text, 10)
            
            # Pad/Truncate
            num_valid = pred_global.shape[0]
            if num_valid < 10:
                pad = np.full((10 - num_valid, 2), np.nan)
                pred_global = np.vstack([pred_global, pad]) if num_valid > 0 else pad
            else:
                pred_global = pred_global[:10]

            # Calculate Metrics (Global vs Global)
            gt_global = batch_data[i]['waypoints_global']
            diffs = pred_global - gt_global
            l2 = np.linalg.norm(diffs, axis=1)
            
            ade = np.nanmean(l2)
            fde = l2[-1]
            error_at_1s = l2[1]
            
            is_failure = True if (l2[1] > 10.0 or np.isnan(l2[1])) else False

            row = {
                'idx': idx,
                'num_valid_waypoints': num_valid,
                'format_compliant': 1 if num_valid == 10 else 0,
                'ade': float(ade),
                'fde': float(fde),
                'failure_rate': 1 if is_failure else 0,
                'error_at_1s': float(error_at_1s),
                'processing_time_sec': float(per_sample_time),
                'gen_text': text
            }
            for wpi in range(10):
                row[f'wp{wpi}_error'] = float(l2[wpi])
            
            results.append(row)
            writer.writerow(row)
            csv_file.flush()

            # Visualization (Direct Global Overlay)
            if idx in vis_indices:
                visualize_trajectories(
                    batch_data[i]['image'],
                    batch_data[i]['waypoints'], # GT Global
                    pred_global,                       # Pred Global
                    batch_data[i]['cam_to_ego'],
                    batch_data[i]['ego_to_world'],
                    idx,
                    vis_dir
                )

    csv_file.close()

    # Stats
    successful = [r for r in results if r['failure_rate'] == 0]
    failed_count = len(results) - len(successful)
    
    def get_stats(data_list):
        if not data_list: return float('nan'), float('nan')
        return float(np.mean(data_list)), float(np.std(data_list))

    ade_m, ade_s = get_stats([r['ade'] for r in successful])
    fde_m, fde_s = get_stats([r['fde'] for r in successful])
    e1s_m, e1s_s = get_stats([r['error_at_1s'] for r in successful])
    e2s_m, e2s_s = get_stats([r['wp3_error'] for r in successful])
    e3s_m, e3s_s = get_stats([r['wp5_error'] for r in successful])

    failure_rate = failed_count / len(results) if len(results) > 0 else 0.0
    
    summary = {
        'total_samples': len(results),
        'successful_samples': len(successful),
        'failed_samples': failed_count,
        'failure_rate': failure_rate,
        'format_compliance': np.mean([r['format_compliant'] for r in results]) if results else 0,
        'ade_mean': ade_m, 'ade_std': ade_s,
        'fde_mean': fde_m, 'fde_std': fde_s,
        'error_at_1s_mean': e1s_m, 'error_at_1s_std': e1s_s,
        'error_at_2s_mean': e2s_m, 'error_at_1s_std': e2s_s,
        'error_at_3s_mean': e3s_m, 'error_at_3s_std': e3s_s
    }

    with open(os.path.join(args.output_dir, 'eval_results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(colored('\n=== Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Failures (FDE > 10m): {summary['failed_samples']} ({summary['failure_rate']:.2%})")
    print(f"ADE (Successful): {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}")
    print(f"FDE (Successful): {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}")
    print(f"Error @ 1s: {summary['error_at_1s_mean']:.4f}")
    print(f"Error @ 1s: {summary['error_at_2s_mean']:.4f}")
    print(f"Error @ 3s: {summary['error_at_3s_mean']:.4f}")
    print(colored(f"Results saved to: {csv_path}", "green"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs')
    parser.add_argument('--output_name', type=str, default='eval_results.csv')
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--num_vis', type=int, default=10)
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    if args.num_samples == -1: args.num_samples = None
    return args

if __name__ == '__main__':
    evaluate(parse_args())
    
