#!/usr/bin/env python3
"""
Evaluation Script for Multimodal LiDAR+CLIP+Qwen Model
======================================================

Evaluates trajectory prediction using:
- CLIP vision encoder (frozen)
- SST LiDAR encoder (frozen)
- Qwen LLM (frozen)
- Trained MLP projectors

Outputs:
- Detailed CSV with per-sample metrics
- Summary JSON with aggregate statistics
- Trajectory visualizations (camera, LiDAR, BEV, multi-cam)

Usage:
    python evaluate_lidarqwenclip.py \\
        --checkpoint /path/to/checkpoint.pth \\
        --dataroot /path/to/nuscenes \\
        --version v1.0-test \\
        --num_samples 100 \\
        --num_vis 10 \\
        --run_name my_evaluation
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
    """Extract waypoint coordinates from generated text.
    
    Returns:
        numpy array of shape (N, 2) where N <= max_points
    """
    # Try to extract only from "Future Trajectory:" section to avoid parsing ego positions
    trajectory_match = re.search(r'Future Trajectory:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    
    if trajectory_match:
        text_to_parse = trajectory_match.group(1)
    else:
        # Fallback: use entire text if pattern not found
        text_to_parse = text
    
    # find all floats/ints in text and group into pairs
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text_to_parse)
    nums = [float(x) for x in nums]
    
    # group into pairs
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        if len(pairs) >= max_points:
            break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)


def visualize_trajectories(image_pil, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, idx, output_dir):
    """Overlay ground truth and predicted trajectories on the image.
    
    Args:
        image_pil: PIL Image
        gt_waypoints_2d: ground truth waypoints in world coordinates (N, 2) with z=0
        pred_waypoints_2d: predicted waypoints in world coordinates (N, 2) with z=0
        cam_to_ego: camera calibration data
        ego_to_world: ego pose data
        idx: sample index for filename
        output_dir: directory to save visualization
    """
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Convert 2D waypoints to 3D (add z=0)
    gt_waypoints_3d = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    # Filter out NaN predictions
    valid_pred_mask = ~np.isnan(pred_waypoints_3d).any(axis=1)
    pred_waypoints_3d_valid = pred_waypoints_3d[valid_pred_mask]
    
    try:
        # Project GT waypoints to image
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
        
        # Draw GT trajectory with polygon (green)
        if len(gt_waypoints_3d) > 1:
            gt_left_3d = OffsetTrajectory3D(gt_waypoints_3d, -1.73 / 2)
            gt_right_3d = OffsetTrajectory3D(gt_waypoints_3d, 1.73 / 2)
            gt_left_img = ProjectWorldToImage(gt_left_3d.tolist(), cam_to_ego, ego_to_world)
            gt_right_img = ProjectWorldToImage(gt_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            # Draw GT polygon
            gt_polygon = np.vstack((np.array(gt_left_img), np.array(gt_right_img)[::-1])).astype(np.int32)
            if gt_polygon.size > 0:
                cv2.fillPoly(img, [gt_polygon], color=(0, 255, 0), lineType=cv2.LINE_AA)
        
        # Draw GT waypoints as circles
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 255, 0), thickness=-1)
        
        # Project predicted waypoints to image (if valid)
        if len(pred_waypoints_3d_valid) > 0:
            pred_points_img = ProjectWorldToImage(pred_waypoints_3d_valid.tolist(), cam_to_ego, ego_to_world)
            
            # Draw predicted trajectory with polygon (orange)
            if len(pred_waypoints_3d_valid) > 1:
                pred_left_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, -1.73 / 2)
                pred_right_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, 1.73 / 2)
                pred_left_img = ProjectWorldToImage(pred_left_3d.tolist(), cam_to_ego, ego_to_world)
                pred_right_img = ProjectWorldToImage(pred_right_3d.tolist(), cam_to_ego, ego_to_world)
                
                pred_polygon = np.vstack((np.array(pred_left_img), np.array(pred_right_img)[::-1])).astype(np.int32)
                if pred_polygon.size > 0:
                    cv2.fillPoly(img, [pred_polygon], color=(0, 125, 255), lineType=cv2.LINE_AA)
            
            # Draw predicted waypoints as circles
            for pt in pred_points_img:
                cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 125, 255), thickness=-1)
        
        # Add legend
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 2)
        
        # Save visualization
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
        # build sample list similar to dataset
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

        # ego positions (last 3)
        ego_positions = []
        current_sample = sample
        for _ in range(3):
            cam_data = self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_positions.append([float(ego_pose['translation'][0]), float(ego_pose['translation'][1])])
            if current_sample['prev']:
                current_sample = self.nusc.get('sample', current_sample['prev'])
            else:
                break
        ego_positions.reverse()

        # image and camera calibration
        camera_token = sample['data']['CAM_FRONT']
        camera_data = self.nusc.get('sample_data', camera_token)
        image_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        image = Image.open(image_path).convert('RGB')
        
        # Get camera calibration for visualization
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

        # Get LiDAR point cloud with nsweeps
        nuscenes_pointcloud, _ = LidarPointCloud.from_file_multisweep(
            self.nusc,
            sample,
            chan='LIDAR_TOP',
            ref_chan='LIDAR_TOP',
            nsweeps=self.nsweeps,
            min_distance=1.0
        )
        torch_pointcloud = torch.from_numpy(nuscenes_pointcloud.points.T).float()  # [N, 4]

        # ground truth 10 waypoints
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
    """
    Load checkpoint into MultimodalQwenModel.
    Expects checkpoint format from train_projection_layers.py
    """
    print(colored(f"Loading checkpoint from {ckpt_path}...", "yellow"))
    
    data = torch.load(ckpt_path, map_location=device)
    
    # Load vision projector
    if 'vision_projector_state_dict' in data:
        try:
            model.vision_projector.load_state_dict(data['vision_projector_state_dict'])
            print(colored("  ✓ Loaded vision_projector weights", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading vision_projector: {e}", "red"))
    
    # Load lidar projector
    if 'lidar_projector_state_dict' in data:
        try:
            model.lidar_projector.load_state_dict(data['lidar_projector_state_dict'])
            print(colored("  ✓ Loaded lidar_projector weights", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading lidar_projector: {e}", "red"))
    
    # Optionally load encoder weights if they were trained
    if 'vision_encoder_state_dict' in data:
        try:
            model.vision_tower.load_state_dict(data['vision_encoder_state_dict'])
            print(colored("  ✓ Loaded vision_encoder weights", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading vision_encoder: {e}", "red"))
    
    if 'lidar_encoder_state_dict' in data:
        try:
            model.lidar_encoder.load_state_dict(data['lidar_encoder_state_dict'])
            print(colored("  ✓ Loaded lidar_encoder weights", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading lidar_encoder: {e}", "red"))
    
    if 'llm_state_dict' in data:
        try:
            model.language_model.load_state_dict(data['llm_state_dict'], strict=False)
            print(colored("  ✓ Loaded LLM weights (partial)", "green"))
        except Exception as e:
            print(colored(f"  ✗ Warning loading LLM: {e}", "red"))
    
    epoch = data.get('epoch', 'unknown')
    print(colored(f"  ✓ Checkpoint from epoch {epoch}", "cyan"))


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Clear GPU cache before loading
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(colored("\n" + "="*70, "cyan"))
    print(colored("Multimodal Model Evaluation", "cyan", attrs=['bold']))
    print(colored("="*70 + "\n", "cyan"))
    
    # Create model
    print(colored("Creating MultimodalQwenModel...", "yellow"))
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
    
    # Load checkpoint
    if args.checkpoint is not None:
        # Clear cache again before loading checkpoint weights
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        load_checkpoint_into_model(model, args.checkpoint, device)
    
    model.eval()
    print(colored("✓ Model ready for evaluation\n", "green"))

    # Loss function for computing cross-entropy (same as training)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    # dataset for evaluation
    print(colored("Loading NuScenes dataset...", "yellow"))
    ds = EvalNuScenes(args.version, args.dataroot, model.prompt_part1, model.prompt_part2, nsweeps=args.nsweeps)
    print(colored(f"✓ Dataset loaded: {len(ds)} samples\n", "green"))

    results = []

    n_samples = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))
    
    # Select random samples for visualization
    vis_indices = set(np.random.choice(n_samples, size=min(args.num_vis, n_samples), replace=False)) if args.num_vis > 0 else set()
    vis_dir = os.path.join(args.output_dir, 'visualizations')

    # Process in batches
    batch_size = args.batch_size
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    print(colored(f"Evaluating {n_samples} samples in {num_batches} batches...\n", "cyan"))
    
    for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        # Collect batch data
        batch_items = [ds.get_item(idx) for idx in batch_indices]
        batch_images = [item['image'] for item in batch_items]
        batch_ego_positions = [item['ego_positions'] for item in batch_items]
        batch_gt_waypoints = [item['waypoints'] for item in batch_items]
        batch_cam_to_ego = [item['cam_to_ego'] for item in batch_items]
        batch_ego_to_world = [item['ego_to_world'] for item in batch_items]
        batch_lidar = [item['lidar'] for item in batch_items]
        
        # Process images (CLIP expects pixel values)
        pixel_values = model.image_processor(images=batch_images, return_tensors='pt').pixel_values.to(device)
        
        # Move LiDAR to device
        batch_lidar_device = [pc.to(device) for pc in batch_lidar]
        
        # Convert ego_positions to Python float lists
        batch_ego_positions_py = [[[float(x), float(y)] for (x, y) in ego_pos] for ego_pos in batch_ego_positions]
        
        # === Compute Cross-Entropy Loss (batched) ===
        batch_ce_losses = []
        batch_token_accs = []
        batch_perplexities = []
        
        try:
            with torch.no_grad():
                # Prepare batch prompts and targets
                batch_prompts = []
                batch_targets = []
                for ego_pos, gt_wp in zip(batch_ego_positions_py, batch_gt_waypoints):
                    pos_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_pos])
                    prompt = f"{model.prompt_part1}[{pos_str}]\n{model.prompt_part2}"
                    wp_str = ", ".join([f"[{wp[0]:.2f}, {wp[1]:.2f}]" for wp in gt_wp])
                    target_string = "Future Trajectory: " + wp_str
                    batch_prompts.append(prompt)
                    batch_targets.append(target_string)
                
                # Tokenize prompts and full sequences
                full_texts = [p + t for p, t in zip(batch_prompts, batch_targets)]
                
                # Apply chat template for consistency
                full_prompts_formatted = [
                    model.tokenizer.apply_chat_template(
                        [{"role": "user", "content": ft}],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    for ft in full_texts
                ]
                
                # Tokenize
                tokenized = model.tokenizer(full_prompts_formatted, return_tensors='pt', padding=True).to(device)
                input_ids = tokenized.input_ids
                
                # Forward pass through model
                logits = model(pixel_values, batch_lidar_device, input_ids, use_vision=True, use_lidar=True)
                
                # Number of multimodal tokens prepended (vision + lidar = 2 tokens)
                num_multimodal_tokens = 2  # 1 for vision, 1 for lidar
                
                # Compute loss per sample
                for i in range(len(batch_indices)):
                    # Get prompt length to mask it out
                    prompt_formatted = model.tokenizer.apply_chat_template(
                        [{"role": "user", "content": batch_prompts[i]}],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    prompt_tokens = model.tokenizer(prompt_formatted, return_tensors='pt').input_ids
                    prompt_length = prompt_tokens.shape[1]
                    
                    # Create labels - account for multimodal tokens at start
                    # The logits sequence is: [multimodal_tokens, text_tokens]
                    # We need to align labels with the text portion of logits
                    labels = input_ids[i].clone()
                    labels[:prompt_length] = -100
                    
                    # Extract the portion of logits corresponding to text tokens
                    # logits shape: [batch, num_multimodal_tokens + seq_len, vocab_size]
                    # We want logits for text tokens: [num_multimodal_tokens : num_multimodal_tokens + seq_len]
                    text_logits = logits[i, num_multimodal_tokens:, :]  # Skip multimodal tokens
                    
                    # Standard next-token prediction: shift by 1
                    shift_logits = text_logits[:-1, :].contiguous()  # Predict next token
                    shift_labels = labels[1:].contiguous()  # Target is next token
                    
                    loss_per_token = loss_fn(shift_logits, shift_labels)
                    valid_tokens = (shift_labels != -100)
                    
                    if valid_tokens.sum() > 0:
                        ce_loss = loss_per_token[valid_tokens].mean().item()
                        perplexity = np.exp(ce_loss)
                        
                        # Token accuracy
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
        
        # === Generate trajectories (batched) ===
        try:
            outputs, gen_texts = model.generate_trajectory(batch_images, batch_lidar_device, batch_ego_positions_py)
        except Exception as e:
            print(f"Generation failed for batch {batch_idx}: {e}")
            continue
        
        # Process each sample in batch
        batch_time = time.time() - batch_start_time
        per_sample_time = batch_time / len(batch_indices)
        
        for i, idx in enumerate(batch_indices):
            gen_text = gen_texts[i]
            pred_coords = parse_coords_from_text(gen_text, max_points=10)
            
            # Print generated text for first few samples
            if idx < 5 or idx in vis_indices:
                print(colored(f"\n[Sample {idx}] Generated text:", "yellow"))
                print(gen_text[:500])
            
            num_valid_waypoints = pred_coords.shape[0]
            format_compliant = (num_valid_waypoints == 10)
            
            # Pad or truncate to 10 waypoints
            if pred_coords.shape[0] < 10:
                pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
                pred_coords = np.vstack([pred_coords, pad])
            elif pred_coords.shape[0] > 10:
                pred_coords = pred_coords[:10]
            
            gt_wp = batch_gt_waypoints[i]
            
            # Compute trajectory metrics
            diffs = pred_coords - gt_wp
            l2_per_waypoint = np.linalg.norm(diffs, axis=1)
            ade = np.nanmean(l2_per_waypoint)
            fde = l2_per_waypoint[-1]
            miss_rate_10m = 1.0 if fde > 10.0 else 0.0
            
            # Failure rate at 1 second (index 1 = 2nd waypoint at 1.0s, assuming 0.5s intervals)
            error_at_1s = l2_per_waypoint[1] if len(l2_per_waypoint) > 1 else np.nan
            failure_at_1s = 1.0 if error_at_1s > 10.0 else 0.0
            
            # Store results
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
            
            # Visualize selected samples
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

    # aggregate metrics
    ce_losses = [r['cross_entropy_loss'] for r in results if not np.isnan(r['cross_entropy_loss'])]
    perplexities = [r['perplexity'] for r in results if not np.isnan(r['perplexity'])]
    token_accs = [r['token_accuracy'] for r in results if not np.isnan(r['token_accuracy'])]
    ades = [r['ade'] for r in results if not np.isnan(r['ade'])]
    fdes = [r['fde'] for r in results if not np.isnan(r['fde'])]
    miss_rates = [r['miss_rate_10m'] for r in results]
    failure_rates_1s = [r['failure_at_1s'] for r in results]
    errors_at_1s = [r['error_at_1s'] for r in results if not np.isnan(r['error_at_1s'])]
    format_compliance = [r['format_compliant'] for r in results]
    processing_times = [r['processing_time_sec'] for r in results]
    
    # Per-waypoint aggregate errors
    all_waypoint_errors = [r['waypoint_errors'] for r in results]
    waypoint_means = np.nanmean(all_waypoint_errors, axis=0).tolist() if len(all_waypoint_errors) > 0 else []

    summary = {
        'num_samples': len(results),
        # Loss metrics
        'cross_entropy_loss_mean': float(np.mean(ce_losses)) if len(ce_losses) > 0 else float('nan'),
        'cross_entropy_loss_std': float(np.std(ce_losses)) if len(ce_losses) > 0 else float('nan'),
        'perplexity_mean': float(np.mean(perplexities)) if len(perplexities) > 0 else float('nan'),
        'token_accuracy_mean': float(np.mean(token_accs)) if len(token_accs) > 0 else float('nan'),
        # Coordinate metrics
        'ade_mean': float(np.mean(ades)) if len(ades) > 0 else float('nan'),
        'ade_std': float(np.std(ades)) if len(ades) > 0 else float('nan'),
        'fde_mean': float(np.mean(fdes)) if len(fdes) > 0 else float('nan'),
        'fde_std': float(np.std(fdes)) if len(fdes) > 0 else float('nan'),
        'miss_rate_10m': float(np.mean(miss_rates)) if len(miss_rates) > 0 else float('nan'),
        'failure_rate_1s': float(np.mean(failure_rates_1s)) if len(failure_rates_1s) > 0 else float('nan'),
        'error_at_1s_mean': float(np.mean(errors_at_1s)) if len(errors_at_1s) > 0 else float('nan'),
        'error_at_1s_std': float(np.std(errors_at_1s)) if len(errors_at_1s) > 0 else float('nan'),
        'format_compliance_rate': float(np.mean(format_compliance)) if len(format_compliance) > 0 else float('nan'),
        # Timing metrics
        'avg_processing_time_sec': float(np.mean(processing_times)) if len(processing_times) > 0 else float('nan'),
        'total_processing_time_sec': float(np.sum(processing_times)) if len(processing_times) > 0 else float('nan'),
        'fps': float(1.0 / np.mean(processing_times)) if len(processing_times) > 0 and np.mean(processing_times) > 0 else float('nan'),
        # Per-waypoint breakdown
        'per_waypoint_errors_mean': waypoint_means
    }

    # save detailed CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_name)
    
    # Flatten waypoint_errors for CSV (convert list to individual columns)
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['idx', 'cross_entropy_loss', 'perplexity', 'token_accuracy', 
                     'num_valid_waypoints', 'format_compliant', 'ade', 'fde', 'miss_rate_10m',
                     'failure_at_1s', 'error_at_1s', 'processing_time_sec'] + \
                     [f'wp{i}_error' for i in range(10)] + ['gen_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                'idx': r['idx'],
                'cross_entropy_loss': r['cross_entropy_loss'],
                'perplexity': r['perplexity'],
                'token_accuracy': r['token_accuracy'],
                'num_valid_waypoints': r['num_valid_waypoints'],
                'format_compliant': r['format_compliant'],
                'ade': r['ade'],
                'fde': r['fde'],
                'miss_rate_10m': r['miss_rate_10m'],
                'failure_at_1s': r['failure_at_1s'],
                'error_at_1s': r['error_at_1s'],
                'processing_time_sec': r['processing_time_sec'],
                'gen_text': r['gen_text']
            }
            for i, err in enumerate(r['waypoint_errors']):
                row[f'wp{i}_error'] = err
            writer.writerow(row)
    
    # Save summary JSON
    summary_path = os.path.join(args.output_dir, args.output_name.replace('.csv', '_summary.json'))
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(colored('\n=== Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(colored(f"Samples evaluated: {summary['num_samples']}", 'white'))
    print(colored('\nLoss Metrics:', 'yellow'))
    print(f"  Cross-Entropy Loss: {summary['cross_entropy_loss_mean']:.4f} ± {summary['cross_entropy_loss_std']:.4f}")
    print(f"  Perplexity: {summary['perplexity_mean']:.4f}")
    print(f"  Token Accuracy: {summary['token_accuracy_mean']:.4f}")
    print(colored('\nTrajectory Metrics:', 'yellow'))
    print(f"  Average Displacement Error (mean): {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}")
    print(f"  Final Displacement Error (mean): {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}")
    print(f"  Error at 1s (mean): {summary['error_at_1s_mean']:.4f} ± {summary['error_at_1s_std']:.4f}")
    print(f"  Miss Rate @ 10m: {summary['miss_rate_10m']:.2%}")
    print(f"  Failure Rate @ 1s (>10m): {summary['failure_rate_1s']:.2%}")
    print(f"  Format Compliance: {summary['format_compliance_rate']:.2%}")
    print(colored('\nTiming Metrics:', 'yellow'))
    print(f"  Average Processing Time: {summary['avg_processing_time_sec']:.3f}s")
    print(f"  FPS: {summary['fps']:.2f}")
    print(colored(f'\nResults saved to: {csv_path}', 'green'))
    print(colored(f'Summary saved to: {summary_path}', 'green'))
    if len(vis_indices) > 0:
        print(colored(f'Visualizations saved to: {vis_dir} ({len(vis_indices)} samples)', 'green'))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Multimodal LiDAR+CLIP+Qwen Model')
    
    # Dataset arguments
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3',
                        help='Path to NuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-test',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='NuScenes dataset version')
    parser.add_argument('--nsweeps', type=int, default=5,
                        help='Number of LiDAR sweeps to aggregate')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (from train_projection_layers.py)')
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                        help='Qwen model name')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14',
                        help='CLIP model name')
    parser.add_argument('--sst_config', type=str, default='src/models/mmdet3d/configs/sst_encoder_only_config.py',
                        help='SST config path')
    parser.add_argument('--lidar_encoder_path', type=str, required=True,
                        help='Path to LidarCLIP checkpoint')
    parser.add_argument('--mlp_hidden_dim', type=int, default=2048,
                        help='MLP hidden dimension')
    parser.add_argument('--mlp_num_layers', type=int, default=3,
                        help='Number of MLP layers')
    parser.add_argument('--mlp_dropout', type=float, default=0.1,
                        help='MLP dropout rate')
    
    # Evaluation arguments
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of samples to evaluate (-1 for all)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for evaluation')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of random samples to visualize (0 to disable)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./eval_outputs',
                        help='Base directory for evaluation outputs')
    parser.add_argument('--output_name', type=str, default='eval_results.csv',
                        help='Output CSV filename')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name for this evaluation run')
    
    args = parser.parse_args()
    
    # Append run_name to output_dir
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    
    # Convert -1 to None
    if args.num_samples == -1:
        args.num_samples = None
    
    # Print configuration
    print(colored("--- Evaluation Configuration ---", "cyan"))
    for k, v in vars(args).items():
        print(colored(f"{k}: {v}", "cyan"))
    print(colored("--------------------------", "cyan"))
    
    return args


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
