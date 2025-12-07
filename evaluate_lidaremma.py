# PyTorch Files
import csv
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Local files
from src.models.lidar_emma import LidarEMMA
from scripts.nuscenes_dataset import NuScenesDataset
from src.utils.lidaremma_utils import collate_fn, parse_coords_from_text


# Other
import argparse
import gc
import os
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from datetime import datetime
import wandb
import yaml
from dataclasses import fields
import logging


def load_model(args, device):

    model = LidarEMMA(device,
                llm=args.llm,
                freeze_encoders=True,
                freeze_llm=True,
                use_lidar=args.use_lidar,
                lidar_pooling=False) # Always false for now
    # Load checkpoint
    checkpt = torch.load(args.checkpoint, map_location=device)
    if 'vision_projector_state_dict' in checkpt:
        model.vision_projector.load_state_dict(checkpt['vision_projector_state_dict'])
        print("✓ Vision Projector loaded")

    if 'lidar_projector_state_dict' in checkpt and hasattr(model, 'lidar_projector'):
        model.lidar_projector.load_state_dict(checkpt['lidar_projector_state_dict'])
        print("✓ LiDAR Projector loaded")

    # Load Encoders/LLM
    if 'vision_encoder_state_dict' in checkpt:
        model.vision_tower.load_state_dict(checkpt['vision_encoder_state_dict'])
        print("✓ Vision Encoder loaded")
        
    if 'lidar_encoder_state_dict' in checkpt:
        model.lidar_encoder.load_state_dict(checkpt['lidar_encoder_state_dict'])
        print("✓ LiDAR Encoder loaded")
        
    if 'llm_state_dict' in checkpt:
        model.language_model.load_state_dict(checkpt['llm_state_dict'])
        print("✓ LLM loaded")
    model.eval() # Set to eval mode

    return model


def main():

    # ========================================================================
    # Parse Arguments
    # ========================================================================
    parser = argparse.ArgumentParser(description="Evaluate LIDAR-EMMA model")

    # Required args
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["1a", "2a", "3a", "1b", "2b", "3b",
                 "1a-lidar","2a-lidar", "3a-lidar", "1b-lidar", "2b-lidar", "3b-lidar"],
        help="Select ablation config: 1a, 2a, 3a, 1b, 2b, 3b (no lidar), or 1a-lidar, 2a-lidar, 3a-lidar, 1b-lidar, 2b-lidar, 3b-lidar (with lidar)."
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (final_model.pth or projector state dict)')
    
    # Other args
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_name', type=str, default='eval_results.csv')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for data loading')
    
    args = parser.parse_args()
    args.use_lidar = "lidar" in args.ablation

    # Run name is folder name of stem's parent
    args.run_name = Path(args.checkpoint).parent.name

    # Choose LLM based on ablation
    if args.ablation in ("1a", "2a", "3a", "1a-lidar", "2a-lidar", "3a-lidar"):
        args.llm = "Qwen/Qwen2.5-3B"
    else:
        args.llm = "Qwen/Qwen2.5-3B-Instruct"

    args.output_dir = f"./eval_outputs/{args.run_name}"

    # ========================================================================
    # Initialize device, tokenizer, dataloader
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}\n")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    model = load_model(args, device)
    
    print("\nLoading dataset...")
    ds = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2,
        output_lidar=args.use_lidar,
    )
    print(f"✓ Dataset loaded: {len(ds)} samples\n")

    # Create dataloader
    custom_collate_fn = lambda batch: collate_fn(batch)
    
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False, # Keep order the same for reproducability
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False, # Don't drop last batch if not full
        prefetch_factor=2, # Faster data loading by prefetching batches while running model
        
    )

    print(colored("--- Evaluation Configuration ---", "cyan"))
    if args.use_lidar:
        print(colored("Using LidarEMMA model!", "green"))
    else:
        print(colored("Using QwenCLIP model!", "green"))
    for k, v in vars(args).items():
        print(colored(f"{k}: {v}", "cyan"))
    print(colored("--------------------------", "cyan"))

    # ========================================================================
    # Evaluation Loop
    # ========================================================================
    print("\n" + "="*70)
    print("Starting Evaluation")
    print("="*70 + "\n")

    # --- PREPARE CSV WRITER ---
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_name)
    
    # Save run arguments to JSON
    args_dict = vars(args)
    args_json_path = os.path.join(args.output_dir, 'run_args.json')
    with open(args_json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(colored(f"Run arguments saved to {args_json_path}", "green"))
    
    # Load dark scenes information
    dark_scenes_path = os.path.join(args.output_dir, '../dark_scenes.csv')
    dark_scenes_data = {'<40': [], '<60': [], '<80': []}
    
    if os.path.exists(dark_scenes_path):
        with open(dark_scenes_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dark_scenes_data['<40'].append(row['<40'] == 'True')
                dark_scenes_data['<60'].append(row['<60'] == 'True')
                dark_scenes_data['<80'].append(row['<80'] == 'True')
        print(colored(f"Loaded dark scenes data from {dark_scenes_path}", "green"))
    else:
        print(colored(f"Warning: {dark_scenes_path} not found. Dark scene filtering disabled.", "yellow"))
    
    # Open file and keep it open during the loop
    csv_file = open(csv_path, 'w', newline='')
    fieldnames = ['idx', 'num_valid_waypoints', 'format_compliant', 'ade', 'fde',
                 'failure_rate', 'error_at_1s'] + \
                 [f'wp{i}_error' for i in range(10)] + \
                 ['history_trajectory', 'gt_trajectory', 'pred_trajectory', 'gen_text'] # Added history_trajectory
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    # Open separate CSV files for dark scenes
    csv_files_dark = {}
    writers_dark = {}
    for threshold in ['40', '60', '80']:
        csv_path_dark = os.path.join(args.output_dir, f'eval_results_dark{threshold}.csv')
        csv_files_dark[threshold] = open(csv_path_dark, 'w', newline='')
        writers_dark[threshold] = csv.DictWriter(csv_files_dark[threshold], fieldnames=fieldnames)
        writers_dark[threshold].writeheader()
    # ---------------------------

    results = []
    results_dark = {'40': [], '60': [], '80': []}
    sample_idx = 0  # Track global sample index across batches
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            prompt = batch['prompt']

            images = batch['images']  # List of PIL images
            lidar_data = batch.get("lidar", None) # TODO: Rewrite so it uses device gpu

            # Extra data loaded for evaluation
            batch_gt_waypoints = batch.get("waypoints", None)
            batch_ego_positions = batch.get("ego_positions", None)

            # Process images with CLIP processor
            pixel_values = model.image_processor(images=images, return_tensors="pt").pixel_values.to(device)
            
            # Process LiDAR data
            point_clouds = None
            if args.use_lidar:
                point_clouds = [pc for pc in lidar_data if pc is not None]
                if len(point_clouds) == 0:
                    point_clouds = None
                else:
                    # Move to device
                    point_clouds = [pc.to(device) if isinstance(pc, torch.Tensor) else pc 
                                    for pc in point_clouds]
                    
            try:
                gen_texts = model.generate_trajectory(
                    prompt=prompt,
                    images=pixel_values,
                    point_clouds=point_clouds,
                )
            except Exception as e:
                logging.error(f"Forward pass failed: {e}")
                logging.error(f"Batch size: {len(images)}")
                logging.error(f"Point clouds: {point_clouds is not None}")
                raise

            # Print generated text for inspection
            print(colored(f"\n[Sample 0] Generated text:", "cyan"))
            print(gen_texts[0])

            for idx, gen_text in enumerate(gen_texts): # TODO: Batchify later
                pred_coords = parse_coords_from_text(gen_text)
                num_valid_waypoints = pred_coords.shape[0]
                format_compliant = (num_valid_waypoints == 10)

                if pred_coords.shape[0] < 10:
                    # pad with NaNs so shapes align
                    pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
                    pred_coords = np.vstack([pred_coords, pad])
                elif pred_coords.shape[0] > 10:
                    # Truncate if more than 10
                    pred_coords = pred_coords[:10]

                # === Coordinate-based metrics ===
                gt_wp = batch_gt_waypoints[idx]
                diffs = pred_coords - gt_wp
                l2_per_waypoint = np.linalg.norm(diffs, axis=1)
                
                # ADE (Average Displacement Error)
                ade = np.nanmean(l2_per_waypoint)
                
                # FDE (Final Displacement Error)
                fde = l2_per_waypoint[-1] if len(l2_per_waypoint) >= 10 else np.nan
                
                error_at_1s = l2_per_waypoint[1] if len(l2_per_waypoint) > 1 else np.nan
                any_severe_error = np.any(l2_per_waypoint > 100.0)
                has_nans = np.any(np.isnan(l2_per_waypoint))
                failure_rate = True if (error_at_1s > 10.0 or any_severe_error or has_nans) else False

                result = {
                    'idx': idx,
                    'num_valid_waypoints': int(num_valid_waypoints),
                    'format_compliant': int(format_compliant),
                    'ade': float(ade),
                    'fde': float(fde),
                    'failure_rate': failure_rate,
                    'error_at_1s': error_at_1s,
                    'gen_text': gen_text,
                    'history_trajectory': str(batch_ego_positions[idx]),
                    'gt_trajectory': str(gt_wp),
                    'pred_trajectory': str(pred_coords.tolist())
                }

                # Add per-waypoint errors to result
                for wp_idx in range(10):
                    result[f'wp{wp_idx}_error'] = l2_per_waypoint[wp_idx] if wp_idx < len(l2_per_waypoint) else np.nan

                results.append(result)

                # Write to CSV immediately (incremental saving)
                writer.writerow(result)
                csv_file.flush()  # Ensure data is written to disk
                
                # Write to dark scene CSVs if applicable
                if len(dark_scenes_data['<40']) > sample_idx:
                    if dark_scenes_data['<40'][sample_idx]:
                        results_dark['40'].append(result)
                        writers_dark['40'].writerow(result)
                        csv_files_dark['40'].flush()
                    if dark_scenes_data['<60'][sample_idx]:
                        results_dark['60'].append(result)
                        writers_dark['60'].writerow(result)
                        csv_files_dark['60'].flush()
                    if dark_scenes_data['<80'][sample_idx]:
                        results_dark['80'].append(result)
                        writers_dark['80'].writerow(result)
                        csv_files_dark['80'].flush()
                
                sample_idx += 1

    # Close CSV file
    csv_file.close()
    for threshold in ['40', '60', '80']:
        csv_files_dark[threshold].close()

    # Waypoint-specific errors (only for non-failed samples)
    # Waypoint indices correspond to: 0=0.5s, 1=1s, 2=1.5s, 3=2s, 4=2.5s, 5=3s, etc.
    successful_samples = [r for r in results if not r['failure_rate']]
    
    # ADE and FDE only from successful samples
    ades = [r['ade'] for r in successful_samples if not np.isnan(r['ade'])]
    fdes = [r['fde'] for r in successful_samples if not np.isnan(r['fde'])]
    
    wp1_errors = [r['wp1_error'] for r in successful_samples if not np.isnan(r['wp1_error'])]  # 1s
    wp3_errors = [r['wp3_error'] for r in successful_samples if not np.isnan(r['wp3_error'])]  # 2s
    wp5_errors = [r['wp5_error'] for r in successful_samples if not np.isnan(r['wp5_error'])]  # 3s
    
    # Format compliance
    format_compliant_count = sum(1 for r in results if r['format_compliant'])
    format_compliance_rate = format_compliant_count / len(results) if len(results) > 0 else 0.0
    
    # Failure rate
    failure_count = sum(1 for r in results if r['failure_rate'])
    failure_rate = failure_count / len(results) if len(results) > 0 else 0.0
    
    def compute_summary(results_list, name=""):
        """Helper function to compute summary statistics"""
        successful = [r for r in results_list if not r['failure_rate']]
        
        ades_local = [r['ade'] for r in successful if not np.isnan(r['ade'])]
        fdes_local = [r['fde'] for r in successful if not np.isnan(r['fde'])]
        
        wp1_local = [r['wp1_error'] for r in successful if not np.isnan(r['wp1_error'])]
        wp3_local = [r['wp3_error'] for r in successful if not np.isnan(r['wp3_error'])]
        wp5_local = [r['wp5_error'] for r in successful if not np.isnan(r['wp5_error'])]
        
        format_compliant_local = sum(1 for r in results_list if r['format_compliant'])
        format_compliance_local = format_compliant_local / len(results_list) if len(results_list) > 0 else 0.0
        
        failure_count_local = sum(1 for r in results_list if r['failure_rate'])
        failure_rate_local = failure_count_local / len(results_list) if len(results_list) > 0 else 0.0
        
        return {
            'category': name,
            'total_samples': len(results_list),
            'successful_samples': len(successful),
            'failed_samples': failure_count_local,
            'failure_rate': float(failure_rate_local),
            'format_compliance_rate': float(format_compliance_local),
            'ade_mean': float(np.mean(ades_local)) if len(ades_local) > 0 else float('nan'),
            'ade_std': float(np.std(ades_local)) if len(ades_local) > 0 else float('nan'),
            'fde_mean': float(np.mean(fdes_local)) if len(fdes_local) > 0 else float('nan'),
            'fde_std': float(np.std(fdes_local)) if len(fdes_local) > 0 else float('nan'),
            'error_at_1s_mean': float(np.mean(wp1_local)) if len(wp1_local) > 0 else float('nan'),
            'error_at_1s_std': float(np.std(wp1_local)) if len(wp1_local) > 0 else float('nan'),
            'error_at_2s_mean': float(np.mean(wp3_local)) if len(wp3_local) > 0 else float('nan'),
            'error_at_2s_std': float(np.std(wp3_local)) if len(wp3_local) > 0 else float('nan'),
            'error_at_3s_mean': float(np.mean(wp5_local)) if len(wp5_local) > 0 else float('nan'),
            'error_at_3s_std': float(np.std(wp5_local)) if len(wp5_local) > 0 else float('nan'),
        }
    
    summary = compute_summary(results, "all_samples")

    # Save summary to JSON
    summary_json_path = os.path.join(args.output_dir, 'eval_results_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Compute and save summaries for dark scenes
    summaries_dark = {}
    for threshold in ['40', '60', '80']:
        if len(results_dark[threshold]) > 0:
            summary_dark = compute_summary(results_dark[threshold], f"dark<{threshold}")
            summaries_dark[threshold] = summary_dark
            
            # Save individual summary file
            summary_dark_path = os.path.join(args.output_dir, f'eval_results_dark{threshold}_summary.json')
            with open(summary_dark_path, 'w') as f:
                json.dump(summary_dark, f, indent=4)
            print(colored(f"Dark{threshold} summary saved to {summary_dark_path}", "green"))
    
    print(colored('\n=== Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Successful Samples: {summary['successful_samples']}")
    print(f"Failed Samples: {summary['failed_samples']}")
    print(f"Failure Rate: {summary['failure_rate']:.2%}")
    print(f"Format Compliance: {summary['format_compliance_rate']:.2%}")
    print(f"\nADE: {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f} m")
    print(f"FDE: {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f} m")
    print(f"\nError @ 1s: {summary['error_at_1s_mean']:.4f} ± {summary['error_at_1s_std']:.4f} m")
    print(f"Error @ 2s: {summary['error_at_2s_mean']:.4f} ± {summary['error_at_2s_std']:.4f} m")
    print(f"Error @ 3s: {summary['error_at_3s_mean']:.4f} ± {summary['error_at_3s_std']:.4f} m")
    print(colored(f"\nResults saved to {csv_path}", "green"))
    print(colored(f"Summary saved to {summary_json_path}", "green"))
    
    # Print dark scene summaries
    for threshold in ['40', '60', '80']:
        if threshold in summaries_dark:
            s = summaries_dark[threshold]
            print(colored(f'\n=== Dark<{threshold} Summary ({s["total_samples"]} samples) ===', 'yellow', attrs=['bold']))
            print(f"Successful: {s['successful_samples']} | Failed: {s['failed_samples']} | Failure Rate: {s['failure_rate']:.2%}")
            print(f"ADE: {s['ade_mean']:.4f} ± {s['ade_std']:.4f} m | FDE: {s['fde_mean']:.4f} ± {s['fde_std']:.4f} m")
            print(f"Error @ 1s: {s['error_at_1s_mean']:.4f} m | @ 2s: {s['error_at_2s_mean']:.4f} m | @ 3s: {s['error_at_3s_mean']:.4f} m")

if __name__ == "__main__":
    main()
