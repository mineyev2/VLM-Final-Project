#!/usr/bin/env python3
"""
Evaluation Script for Multimodal LiDAR+CLIP+Qwen Model
(Updated with Incremental CSV Saving and Coordinate Logging)
"""

import os
import sys
import argparse
import json
import csv
import time
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import colored

from src.models.multimodal_qwen_model import MultimodalQwenModel
from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from pyquaternion import Quaternion


def parse_coords_from_text(text, max_points=10):
    """Extract waypoint coordinates from generated text."""
    trajectory_match = re.search(r'Future Trajectory:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    
    if trajectory_match:
        text_to_parse = trajectory_match.group(1)
    else:
        text_to_parse = text
    
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text_to_parse)
    nums = [float(x) for x in nums]
    
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        if len(pairs) >= max_points:
            break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)


def visualize_trajectories(image_pil, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, idx, output_dir):
    """Overlay ground truth and predicted trajectories on the image."""
    
    # --- Safety Check ---
    if pred_waypoints_2d.shape[0] == 0 or len(pred_waypoints_2d.shape) != 2:
        return
    # --------------------

    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    gt_waypoints_3d = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    valid_pred_mask = ~np.isnan(pred_waypoints_3d).any(axis=1)
    pred_waypoints_3d_valid = pred_waypoints_3d[valid_pred_mask]
    
    try:
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
        
        if len(gt_waypoints_3d) > 1:
            gt_left_3d = OffsetTrajectory3D(gt_waypoints_3d, -1.73 / 2)
            gt_right_3d = OffsetTrajectory3D(gt_waypoints_3d, 1.73 / 2)
            gt_left_img = ProjectWorldToImage(gt_left_3d.tolist(), cam_to_ego, ego_to_world)
            gt_right_img = ProjectWorldToImage(gt_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            gt_polygon = np.vstack((np.array(gt_left_img), np.array(gt_right_img)[::-1])).astype(np.int32)
            if gt_polygon.size > 0:
                cv2.fillPoly(img, [gt_polygon], color=(0, 255, 0), lineType=cv2.LINE_AA)
        
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 255, 0), thickness=-1)
        
        if len(pred_waypoints_3d_valid) > 0:
            pred_points_img = ProjectWorldToImage(pred_waypoints_3d_valid.tolist(), cam_to_ego, ego_to_world)
            
            if len(pred_waypoints_3d_valid) > 1:
                pred_left_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, -1.73 / 2)
                pred_right_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, 1.73 / 2)
                pred_left_img = ProjectWorldToImage(pred_left_3d.tolist(), cam_to_ego, ego_to_world)
                pred_right_img = ProjectWorldToImage(pred_right_3d.tolist(), cam_to_ego, ego_to_world)
                
                pred_polygon = np.vstack((np.array(pred_left_img), np.array(pred_right_img)[::-1])).astype(np.int32)
                if pred_polygon.size > 0:
                    cv2.fillPoly(img, [pred_polygon], color=(0, 125, 255), lineType=cv2.LINE_AA)
            
            for pt in pred_points_img:
                cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 125, 255), thickness=-1)
        
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 2)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'vis_sample_{idx:04d}.jpg')
        cv2.imwrite(output_path, img)
        
    except Exception as e:
        print(f"Visualization failed for sample {idx}: {e}")


class EvalNuScenes:
    def __init__(self, version, dataroot, prompt_part1, prompt_part2, nsweeps=5):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.nsweeps = nsweeps
        self.prompt_part1 = prompt_part1
        self.prompt_part2 = prompt_part2
        self.sample_tokens = []
        for scene in self.nusc.scene:
            nbr_samples = scene['nbr_samples']
            if nbr_samples < 13:
                continue
            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)
            for _ in range(2):
                sample = self.nusc.get('sample', sample['next'])
            for _ in range(nbr_samples - 12):
                self.sample_tokens.append(sample['token'])
                sample = self.nusc.get('sample', sample['next'])

    def __len__(self):
        return len(self.sample_tokens)

    def get_item(self, idx):
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        ego_positions = []
        current_sample = sample
        for _ in range(3):
            cam_data = self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_positions.append([float(ego_pose['translation'][0]), float(ego_pose['translation'][1])])
            if current_sample['prev']:
                current_sample = self.nusc.get('sample', current_sample['prev'])
            else:
                while len(ego_positions) < 3:
                    ego_positions.append(ego_positions[-1])
        ego_positions.reverse()

        camera_token = sample['data']['CAM_FRONT']
        camera_data = self.nusc.get('sample_data', camera_token)
        image_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        image = Image.open(image_path).convert('RGB')
        
        cam_calib = self.nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', camera_data['ego_pose_token'])
        cam_to_ego = {
            'translation': cam_calib['translation'],
            'rotation': cam_calib['rotation'],
            'camera_intrinsic': np.array(cam_calib['camera_intrinsic'])
        }
        ego_to_world = {
            'translation': ego_pose['translation'],
            'rotation': ego_pose['rotation']
        }

        nuscenes_pointcloud, _ = LidarPointCloud.from_file_multisweep(
            self.nusc,
            sample,
            chan='LIDAR_TOP',
            ref_chan='LIDAR_TOP',
            nsweeps=self.nsweeps,
            min_distance=1.0
        )
        torch_pointcloud = torch.from_numpy(nuscenes_pointcloud.points.T).float()

        waypoints = []
        current_sample = sample
        for _ in range(10):
            next_sample_token = current_sample['next']
            next_sample = self.nusc.get('sample', next_sample_token)
            next_camera_data = self.nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
            next_ego_pose = self.nusc.get('ego_pose', next_camera_data['ego_pose_token'])['translation']
            waypoints.append([float(next_ego_pose[0]), float(next_ego_pose[1])])
            current_sample = next_sample

        return {
            'image': image,
            'ego_positions': ego_positions,
            'waypoints': np.array(waypoints, dtype=float),
            'cam_to_ego': cam_to_ego,
            'ego_to_world': ego_to_world,
            'lidar': torch_pointcloud
        }


def load_checkpoint_into_model(model, ckpt_path, device):
    print(colored(f"Loading checkpoint from {ckpt_path}...", "yellow"))
    data = torch.load(ckpt_path, map_location=device)
    
    if 'vision_projector_state_dict' in data:
        try:
            model.vision_projector.load_state_dict(data['vision_projector_state_dict'])
            print(colored("  ✓ Loaded vision_projector weights", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading vision_projector: {e}", "red"))
    
    if 'lidar_projector_state_dict' in data:
        try:
            model.lidar_projector.load_state_dict(data['lidar_projector_state_dict'])
            print(colored("  ✓ Loaded lidar_projector weights", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading lidar_projector: {e}", "red"))
    
    if 'lidar_encoder_state_dict' in data:
        try:
            model.lidar_encoder.load_state_dict(data['lidar_encoder_state_dict'])
            print(colored("  ✓ Loaded lidar_encoder weights", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading lidar_encoder: {e}", "red"))


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(colored("\n" + "="*70, "cyan"))
    print(colored("Multimodal Model Evaluation", "cyan", attrs=['bold']))
    if args.disable_lidar:
        print(colored("MODE: LiDAR Disabled (Vision + Text Only)", "magenta", attrs=['bold']))
    print(colored("="*70 + "\n", "cyan"))
    
    model = MultimodalQwenModel(
        device=device,
        qwen_model_name=args.llm,
        clip_model_name=args.clip_model,
        sst_config_path=args.sst_config,
        lidarclip_checkpoint_path=args.lidar_encoder_path,
        freeze_encoders=True,
        freeze_llm=True,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_layers=args.mlp_num_layers,
        mlp_dropout=args.mlp_dropout
    )
    
    if args.checkpoint is not None:
        load_checkpoint_into_model(model, args.checkpoint, device)
    
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    ds = EvalNuScenes(args.version, args.dataroot, model.prompt_part1, model.prompt_part2, nsweeps=args.nsweeps)
    results = []

    n_samples = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))
    vis_indices = set(np.random.choice(n_samples, size=min(args.num_vis, n_samples), replace=False)) if args.num_vis > 0 else set()
    vis_dir = os.path.join(args.output_dir, 'visualizations')

    batch_size = args.batch_size
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    # --- PREPARE CSV WRITER ---
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_name)
    
    # Open file and keep it open during the loop
    csv_file = open(csv_path, 'w', newline='')
    fieldnames = ['idx', 'cross_entropy_loss', 'perplexity', 'token_accuracy', 
                 'num_valid_waypoints', 'format_compliant', 'ade', 'fde', 'miss_rate_10m',
                 'failure_at_1s', 'error_at_1s', 'processing_time_sec'] + \
                 [f'wp{i}_error' for i in range(10)] + \
                 ['gt_trajectory', 'pred_trajectory', 'gen_text'] # Added columns for matching console output
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    # ---------------------------

    for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        batch_items = [ds.get_item(idx) for idx in batch_indices]
        batch_images = [item['image'] for item in batch_items]
        batch_ego_positions = [item['ego_positions'] for item in batch_items]
        batch_gt_waypoints = [item['waypoints'] for item in batch_items]
        batch_cam_to_ego = [item['cam_to_ego'] for item in batch_items]
        batch_ego_to_world = [item['ego_to_world'] for item in batch_items]
        batch_lidar = [item['lidar'] for item in batch_items]
        
        pixel_values = model.image_processor(images=batch_images, return_tensors='pt').pixel_values.to(device)
        batch_lidar_device = [pc.to(device) for pc in batch_lidar]
        batch_ego_positions_py = [[[float(x), float(y)] for (x, y) in ego_pos] for ego_pos in batch_ego_positions]
        
        lidar_input = None if args.disable_lidar else batch_lidar_device
        use_lidar_flag = False if args.disable_lidar else True

        batch_ce_losses = []
        batch_token_accs = []
        batch_perplexities = []

        try:
            with torch.no_grad():
                batch_prompts = []
                batch_targets = []
                for ego_pos, gt_wp in zip(batch_ego_positions_py, batch_gt_waypoints):
                    pos_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_pos])
                    prompt = f"{model.prompt_part1}[{pos_str}]\n{model.prompt_part2}"
                    wp_str = ", ".join([f"[{wp[0]:.2f}, {wp[1]:.2f}]" for wp in gt_wp])
                    target_string = "Future Trajectory: " + wp_str
                    batch_prompts.append(prompt)
                    batch_targets.append(target_string)
                
                full_texts = [p + t for p, t in zip(batch_prompts, batch_targets)]
                full_prompts_formatted = [
                    model.tokenizer.apply_chat_template(
                        [{"role": "user", "content": ft}],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    for ft in full_texts
                ]
                
                tokenized = model.tokenizer(full_prompts_formatted, return_tensors='pt', padding=True).to(device)
                input_ids = tokenized.input_ids
                
                logits = model(pixel_values, lidar_input, input_ids, use_vision=True, use_lidar=use_lidar_flag)
                num_multimodal_tokens = 1 if args.disable_lidar else 2
                
                for i in range(len(batch_indices)):
                    prompt_formatted = model.tokenizer.apply_chat_template(
                        [{"role": "user", "content": batch_prompts[i]}],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    prompt_tokens = model.tokenizer(prompt_formatted, return_tensors='pt').input_ids
                    prompt_length = prompt_tokens.shape[1]
                    
                    labels = input_ids[i].clone()
                    labels[:prompt_length] = -100
                    
                    text_logits = logits[i, num_multimodal_tokens:, :] 
                    shift_logits = text_logits[:-1, :].contiguous()
                    shift_labels = labels[1:].contiguous()
                    
                    loss_per_token = loss_fn(shift_logits, shift_labels)
                    valid_tokens = (shift_labels != -100)
                    
                    if valid_tokens.sum() > 0:
                        ce_loss = loss_per_token[valid_tokens].mean().item()
                        perplexity = np.exp(ce_loss)
                        predictions = shift_logits.argmax(dim=-1)
                        correct = (predictions == shift_labels) & valid_tokens
                        token_acc = correct.sum().item() / valid_tokens.sum().item()
                    else:
                        ce_loss = np.nan
                        perplexity = np.nan
                        token_acc = np.nan
                    
                    batch_ce_losses.append(ce_loss)
                    batch_perplexities.append(perplexity)
                    batch_token_accs.append(token_acc)
                
        except Exception as e:
            print(f"Loss computation failed for batch {batch_idx}: {e}")
            batch_ce_losses = [np.nan] * len(batch_indices)
            batch_perplexities = [np.nan] * len(batch_indices)
            batch_token_accs = [np.nan] * len(batch_indices)
        
        try:
            outputs, gen_texts = model.generate_trajectory(batch_images, lidar_input, batch_ego_positions_py)
        except Exception as e:
            print(f"Generation failed for batch {batch_idx}: {e}")
            continue
        


        # ... [Keep previous code] ...
        
        # Process each sample in batch
        batch_time = time.time() - batch_start_time
        per_sample_time = batch_time / len(batch_indices)
        
        for i, idx in enumerate(batch_indices):
            gen_text = gen_texts[i]
            
            # --- STEP 1: PARSE ALL COORDINATES ---
            raw_pred_coords = parse_coords_from_text(gen_text, max_points=20) # Parse extra points in case we filter some
            
            # --- STEP 2: FILTER OUT HISTORY (Fix for "Predicting the Past") ---
            # Get the history for this sample
            history = np.array(batch_ego_positions_py[i]) # Shape [3, 2]
            
            # Check if the car is effectively stopped (all history points are close to each other)
            # If max distance between any history points is < 0.2m, consider stopped
            is_stopped = False
            if len(history) > 1:
                max_hist_dist = np.max(np.linalg.norm(history - history[0], axis=1))
                if max_hist_dist < 0.2:
                    is_stopped = True

            valid_preds = []
            for p in raw_pred_coords:
                # Calculate distance from this predicted point to ALL history points
                dists = np.linalg.norm(history - p, axis=1)
                min_dist = np.min(dists)
                
                # If the car is MOVING, and this point is identical to a history point (dist < 0.5m),
                # it's likely a hallucination/repetition of the prompt. Skip it.
                if not is_stopped and min_dist < 0.5:
                    continue
                
                valid_preds.append(p)
                if len(valid_preds) == 10:
                    break
            
            pred_coords = np.array(valid_preds)
            # ------------------------------------------------------------------

            num_valid_waypoints = pred_coords.shape[0]
            format_compliant = (num_valid_waypoints == 10)
            
            # Pad or truncate to 10 waypoints
            if pred_coords.shape[0] < 10:
                pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
                pred_coords = np.vstack([pred_coords, pad]) if pred_coords.shape[0] > 0 else pad
            elif pred_coords.shape[0] > 10:
                pred_coords = pred_coords[:10]
            
            gt_wp = batch_gt_waypoints[i]
            
            # Compute trajectory metrics
            diffs = pred_coords - gt_wp
            l2_per_waypoint = np.linalg.norm(diffs, axis=1)
            ade = np.nanmean(l2_per_waypoint)
            fde = l2_per_waypoint[-1]
            
            # ==================================================================
            # LOGGING
            # ==================================================================
            if idx in vis_indices or idx < 5:
                print(colored(f"\n{'='*20} SAMPLE ID: {idx} {'='*20}", "magenta", attrs=['bold']))
                
                print(colored("Input History (Last 3 Positions provided):", "cyan"))
                for h_idx, pos in enumerate(history):
                    print(f"  t-{3-h_idx}: [{pos[0]:.2f}, {pos[1]:.2f}]")

                print(colored(f"Errors:", "cyan"))
                print(f"  ADE: {ade:.4f} meters")
                print(f"  FDE: {fde:.4f} meters")

                print(colored("Future Predictions (Filtered):", "cyan"))
                for k in range(min(10, len(gt_wp))):
                    if not np.isnan(pred_coords[k][0]):
                        p_str = f"[{pred_coords[k][0]:.2f}, {pred_coords[k][1]:.2f}]"
                        pt_err = np.linalg.norm(pred_coords[k] - gt_wp[k])
                        err_str = f"(err: {pt_err:.2f}m)"
                    else:
                        p_str = "[NaN, NaN]"
                        err_str = ""
                    print(f"  GT: [{gt_wp[k][0]:.2f}, {gt_wp[k][1]:.2f}]  ->  Pred: {p_str} {err_str}")

                # 5. Vis Path
                if idx in vis_indices:
                    vis_path = os.path.join(vis_dir, f'vis_sample_{idx:04d}.jpg')
                    print(colored(f"Visualization saved to: {vis_path}", "green", attrs=['bold']))
                print("="*60)
            # ==================================================================

            miss_rate_10m = 1.0 if fde > 10.0 else 0.0
            error_at_1s = l2_per_waypoint[1] if len(l2_per_waypoint) > 1 else np.nan
            failure_at_1s = 1.0 if error_at_1s > 10.0 else 0.0
            
            # Store results (Rest of the loop remains the same...)
            result = {
                'idx': idx,
                'cross_entropy_loss': batch_ce_losses[i],
                'perplexity': batch_perplexities[i],
                'token_accuracy': batch_token_accs[i],
                'num_valid_waypoints': num_valid_waypoints,
                'format_compliant': format_compliant,
                'ade': ade,
                'fde': fde,
                'miss_rate_10m': miss_rate_10m,
                'failure_at_1s': failure_at_1s,
                'error_at_1s': error_at_1s,
                'waypoint_errors': l2_per_waypoint.tolist(),
                'gen_text': gen_text,
                'processing_time_sec': per_sample_time
            }
            results.append(result)
            
            # Visualize
            if idx in vis_indices:
                pred_waypoints_valid = pred_coords[:num_valid_waypoints] if num_valid_waypoints > 0 else np.array([]).reshape(0, 2)
                visualize_trajectories(
                    batch_images[i],
                    gt_wp,
                    pred_waypoints_valid,
                    batch_cam_to_ego[i],
                    batch_ego_to_world[i],
                    idx,
                    vis_dir
                )

    # Close CSV file
    csv_file.close()

    # Summary printing (Optional, kept for end-of-run stats)
    ce_losses = [r['cross_entropy_loss'] for r in results if not np.isnan(r['cross_entropy_loss'])]
    ades = [r['ade'] for r in results if not np.isnan(r['ade'])]
    fdes = [r['fde'] for r in results if not np.isnan(r['fde'])]
    
    summary = {
        'ade_mean': float(np.mean(ades)) if len(ades) > 0 else float('nan'),
        'fde_mean': float(np.mean(fdes)) if len(fdes) > 0 else float('nan'),
    }

    print(colored('\n=== Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(f"ADE Mean: {summary['ade_mean']:.4f}")
    print(f"FDE Mean: {summary['fde_mean']:.4f}")
    print(colored(f"Results saved to {csv_path}", "green"))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Multimodal LiDAR+CLIP+Qwen Model')
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--nsweeps', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--sst_config', type=str, default='src/models/mmdet3d/configs/sst_encoder_only_config.py')
    parser.add_argument('--lidar_encoder_path', type=str)
    parser.add_argument('--mlp_hidden_dim', type=int, default=2048)
    parser.add_argument('--mlp_num_layers', type=int, default=3)
    parser.add_argument('--mlp_dropout', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_vis', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs')
    parser.add_argument('--output_name', type=str, default='eval_results.csv')
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--disable_lidar', action='store_true', help='Disable LiDAR input (Vision + Text only)')
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    if args.num_samples == -1: args.num_samples = None
    return args

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
