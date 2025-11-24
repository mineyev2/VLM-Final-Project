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
    """Overlay ground truth and predicted trajectories on the image.
    
    Note: gt_waypoints_2d and pred_waypoints_2d are in LOCAL/EGO frame coordinates.
    We need to transform them to world coordinates for visualization.
    """
    
    # --- Safety Check ---
    if pred_waypoints_2d.shape[0] == 0 or len(pred_waypoints_2d.shape) != 2:
        return
    # --------------------

    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Convert waypoints from list to numpy array if needed
    gt_waypoints_2d = np.array(gt_waypoints_2d) if isinstance(gt_waypoints_2d, list) else gt_waypoints_2d
    pred_waypoints_2d = np.array(pred_waypoints_2d) if isinstance(pred_waypoints_2d, list) else pred_waypoints_2d
    
    # Add z=0 to make 3D points in local frame
    gt_waypoints_3d_local = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d_local = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    # Transform from local/ego frame to world frame
    ego_translation = np.array(ego_to_world['translation'])
    ego_rotation = Quaternion(ego_to_world['rotation'])
    
    # Transform GT waypoints: local → world
    gt_waypoints_3d_world = []
    for local_pt in gt_waypoints_3d_local:
        world_pt = ego_rotation.rotate(local_pt) + ego_translation
        gt_waypoints_3d_world.append(world_pt)
    
    # Transform predicted waypoints: local → world
    pred_waypoints_3d_world = []
    valid_pred_mask = ~np.isnan(pred_waypoints_3d_local).any(axis=1)
    for i, local_pt in enumerate(pred_waypoints_3d_local):
        if valid_pred_mask[i]:
            world_pt = ego_rotation.rotate(local_pt) + ego_translation
            pred_waypoints_3d_world.append(world_pt)
    
    try:
        # Now project world coordinates to image
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d_world, cam_to_ego, ego_to_world)
        
        if len(gt_waypoints_3d_world) > 1:
            gt_waypoints_3d_array = np.array(gt_waypoints_3d_world)
            gt_left_3d = OffsetTrajectory3D(gt_waypoints_3d_array, -1.73 / 2)
            gt_right_3d = OffsetTrajectory3D(gt_waypoints_3d_array, 1.73 / 2)
            gt_left_img = ProjectWorldToImage(gt_left_3d.tolist(), cam_to_ego, ego_to_world)
            gt_right_img = ProjectWorldToImage(gt_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            gt_polygon = np.vstack((np.array(gt_left_img), np.array(gt_right_img)[::-1])).astype(np.int32)
            if gt_polygon.size > 0:
                # Transparent GT corridor (green, 50% opacity)
                frame_gt = np.zeros_like(img)
                cv2.fillPoly(frame_gt, [gt_polygon], color=(0, 255, 0))
                mask_gt = frame_gt.astype(bool)
                img[mask_gt] = cv2.addWeighted(img, 0.5, frame_gt, 0.5, 0)[mask_gt]
        
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 255, 0), thickness=-1)
        
        if len(pred_waypoints_3d_world) > 0:
            pred_points_img = ProjectWorldToImage(pred_waypoints_3d_world, cam_to_ego, ego_to_world)
            
            if len(pred_waypoints_3d_world) > 1:
                pred_waypoints_3d_array = np.array(pred_waypoints_3d_world)
                pred_left_3d = OffsetTrajectory3D(pred_waypoints_3d_array, -1.73 / 2)
                pred_right_3d = OffsetTrajectory3D(pred_waypoints_3d_array, 1.73 / 2)
                pred_left_img = ProjectWorldToImage(pred_left_3d.tolist(), cam_to_ego, ego_to_world)
                pred_right_img = ProjectWorldToImage(pred_right_3d.tolist(), cam_to_ego, ego_to_world)
                
                pred_polygon = np.vstack((np.array(pred_left_img), np.array(pred_right_img)[::-1])).astype(np.int32)
                if pred_polygon.size > 0:
                    # Transparent predicted corridor (orange, 50% opacity)
                    frame_pred = np.zeros_like(img)
                    cv2.fillPoly(frame_pred, [pred_polygon], color=(0, 125, 255))
                    mask_pred = frame_pred.astype(bool)
                    img[mask_pred] = cv2.addWeighted(img, 0.5, frame_pred, 0.5, 0)[mask_pred]
            
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


        # 1. Get Current Ego Pose (Reference Frame)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        ego_trans = np.array(ego_pose_curr['translation'])
        ego_rot = Quaternion(ego_pose_curr['rotation'])

        # 2. Get History (10 frames)
        history_points = []
        curr_hist = sample
        for _ in range(10):
            c_data = self.nusc.get('sample_data', curr_hist['data']['CAM_FRONT'])
            e_pose = self.nusc.get('ego_pose', c_data['ego_pose_token'])
            history_points.append(np.array(e_pose['translation']))
            
            if curr_hist['prev']:
                curr_hist = self.nusc.get('sample', curr_hist['prev'])
            else:
                history_points.append(history_points[-1])
        history_points.reverse()

        # 3. Get Future Ground Truth (10 frames)
        future_points = []
        curr_fut = sample
        for _ in range(10):
            if curr_fut['next'] == '': 
                break
            curr_fut = self.nusc.get('sample', curr_fut['next'])
            c_data = self.nusc.get('sample_data', curr_fut['data']['CAM_FRONT'])
            e_pose = self.nusc.get('ego_pose', c_data['ego_pose_token'])
            future_points.append(np.array(e_pose['translation']))

        while len(future_points) < 10:
            future_points.append(future_points[-1] if len(future_points) > 0 else ego_trans)

        # 4. Transform to Local Coordinates
        history_local = []
        for p in history_points:
            diff = p - ego_trans
            local_p = ego_rot.inverse.rotate(diff)
            history_local.append(local_p[:2])

        future_local = []
        for p in future_points:
            diff = p - ego_trans
            local_p = ego_rot.inverse.rotate(diff)
            future_local.append(local_p[:2])

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

        # waypoints = []
        # current_sample = sample
        # for _ in range(10):
        #     next_sample_token = current_sample['next']
        #     next_sample = self.nusc.get('sample', next_sample_token)
        #     next_camera_data = self.nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
        #     next_ego_pose = self.nusc.get('ego_pose', next_camera_data['ego_pose_token'])['translation']
        #     waypoints.append([float(next_ego_pose[0]), float(next_ego_pose[1])])
        #     current_sample = next_sample

        return {
            'image': image,
            'ego_positions': history_local,
            'waypoints': future_local,
            'cam_to_ego': cam_to_ego,
            'ego_to_world': ego_to_world,
            'lidar': torch_pointcloud,
            'ego_translation': ego_trans,
            'ego_rotation': ego_rot.elements
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
    
    # Check if output directory already exists
    if os.path.exists(args.output_dir):
        print(colored(f"\n⚠️  WARNING: Output directory already exists: {args.output_dir}", "yellow", attrs=['bold']))
        print(colored("This will overwrite existing results!", "yellow"))
        response = input(colored("Continue? (y/n): ", "yellow"))
        if response.lower() != 'y':
            print(colored("Evaluation cancelled.", "red"))
            sys.exit(0)
    
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
    
    # Save run arguments to JSON
    args_dict = vars(args)
    args_json_path = os.path.join(args.output_dir, 'run_args.json')
    with open(args_json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(colored(f"Run arguments saved to {args_json_path}", "green"))
    
    # Open file and keep it open during the loop
    csv_file = open(csv_path, 'w', newline='')
    fieldnames = ['idx', 'num_valid_waypoints', 'format_compliant', 'ade', 'fde',
                 'failure_rate', 'error_at_1s', 'processing_time_sec'] + \
                 [f'wp{i}_error' for i in range(10)] + \
                 ['history_trajectory', 'gt_trajectory', 'pred_trajectory', 'gen_text'] # Added history_trajectory
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
        
        # pixel_values = model.image_processor(images=batch_images, return_tensors='pt').pixel_values.to(device)
        batch_lidar_device = [pc.to(device) for pc in batch_lidar]
        batch_ego_positions_py = [[[float(x), float(y)] for (x, y) in ego_pos] for ego_pos in batch_ego_positions]
        
        lidar_input = None if args.disable_lidar else batch_lidar_device
        use_lidar_flag = False if args.disable_lidar else True

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
            num_valid_waypoints = raw_pred_coords.shape[0]
            format_compliant = (raw_pred_coords.shape[0] == 10)
            
            # # --- STEP 2: FILTER OUT HISTORY (Fix for "Predicting the Past") ---
            # # Get the history for this sample
            history = np.array(batch_ego_positions_py[i]) # Shape [3, 2]
            
            # # Check if the car is effectively stopped (all history points are close to each other)
            # # If max distance between any history points is < 0.2m, consider stopped
            # is_stopped = False
            # if len(history) > 1:
            #     max_hist_dist = np.max(np.linalg.norm(history - history[0], axis=1))
            #     if max_hist_dist < 0.2:
            #         is_stopped = True

            valid_preds = []
            for p in raw_pred_coords:
                # # Calculate distance from this predicted point to ALL history points
                # dists = np.linalg.norm(history - p, axis=1)
                # min_dist = np.min(dists)
                
                # # If the car is MOVING, and this point is identical to a history point (dist < 0.5m),
                # # it's likely a hallucination/repetition of the prompt. Skip it.
                # if not is_stopped and min_dist < 0.5:
                #     continue
                
                valid_preds.append(p)
                if len(valid_preds) == 10:
                    break
            
            pred_coords = np.array(valid_preds)
            # ------------------------------------------------------------------
            
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
                for k in range(min(3, len(gt_wp))):
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
            error_at_1s = l2_per_waypoint[1] if len(l2_per_waypoint) > 1 else np.nan
            failure_rate = True if (error_at_1s > 10.0 or np.isnan(error_at_1s)) else False 
            
            # Store results and write to CSV immediately
            result = {
                'idx': idx,
                'num_valid_waypoints': num_valid_waypoints, # Number of coordinates in the prediction
                'format_compliant': format_compliant,
                'ade': ade, # average distance error
                'fde': fde, # distance error of final point (5s)
                'failure_rate': failure_rate, # True if error at 1s is > 10m or is NaN
                'error_at_1s': error_at_1s,
                'processing_time_sec': per_sample_time,
                'gen_text': gen_text,
                'history_trajectory': str(batch_ego_positions[i]),
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
    
    summary = {
        'total_samples': len(results),
        'successful_samples': len(successful_samples),
        'failed_samples': failure_count,
        'failure_rate': float(failure_rate),
        'format_compliance_rate': float(format_compliance_rate),
        'ade_mean': float(np.mean(ades)) if len(ades) > 0 else float('nan'),
        'ade_std': float(np.std(ades)) if len(ades) > 0 else float('nan'),
        'fde_mean': float(np.mean(fdes)) if len(fdes) > 0 else float('nan'),
        'fde_std': float(np.std(fdes)) if len(fdes) > 0 else float('nan'),
        'error_at_1s_mean': float(np.mean(wp1_errors)) if len(wp1_errors) > 0 else float('nan'),
        'error_at_1s_std': float(np.std(wp1_errors)) if len(wp1_errors) > 0 else float('nan'),
        'error_at_2s_mean': float(np.mean(wp3_errors)) if len(wp3_errors) > 0 else float('nan'),
        'error_at_2s_std': float(np.std(wp3_errors)) if len(wp3_errors) > 0 else float('nan'),
        'error_at_3s_mean': float(np.mean(wp5_errors)) if len(wp5_errors) > 0 else float('nan'),
        'error_at_3s_std': float(np.std(wp5_errors)) if len(wp5_errors) > 0 else float('nan'),
        'avg_processing_time_sec': float(np.mean([r['processing_time_sec'] for r in results])),
    }

    # Save summary to JSON
    summary_json_path = os.path.join(args.output_dir, 'eval_results_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
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


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Multimodal LiDAR+CLIP+Qwen Model')

    # Params to care about:
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--num_samples', type=int, default=-1)


    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--nsweeps', type=int, default=5)
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--sst_config', type=str, default='src/models/mmdet3d/configs/sst_encoder_only_config.py')
    parser.add_argument('--lidar_encoder_path', type=str, default='/home/hice1/rmineyev3/scratch/VLM-Final-Project/Lidar-CLIP/vit_l_14.ckpt')
    parser.add_argument('--mlp_hidden_dim', type=int, default=2048)
    parser.add_argument('--mlp_num_layers', type=int, default=3)
    parser.add_argument('--mlp_dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_vis', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs')
    parser.add_argument('--output_name', type=str, default='eval_results.csv')

    parser.add_argument('--disable_lidar', action='store_true', help='Disable LiDAR input (Vision + Text only)')
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    if args.num_samples == -1: args.num_samples = None
    return args

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
