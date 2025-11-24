#!/usr/bin/env python3
"""
Single Sample Evaluation Script for Multimodal LiDAR+CLIP+Qwen Model
Evaluates one specific sample by index and displays detailed results with visualization.
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import cv2
from termcolor import colored

from src.models.multimodal_qwen_model import MultimodalQwenModel
from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from pyquaternion import Quaternion
import matplotlib.pyplot as plt


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


def visualize_trajectories_with_metrics(image_pil, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, idx, output_path, metrics):
    """Overlay trajectories and metrics on the image."""
    
    if pred_waypoints_2d.shape[0] == 0 or len(pred_waypoints_2d.shape) != 2:
        return
    
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
    
    # Project to image coordinates
    try:
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d_world, cam_to_ego, ego_to_world)
        pred_points_img = ProjectWorldToImage(pred_waypoints_3d_world, cam_to_ego, ego_to_world) if len(pred_waypoints_3d_world) > 0 else []
        
        # Draw GT trajectory polygon (green)
        if len(gt_waypoints_3d_world) > 1:
            gt_left_3d = OffsetTrajectory3D(np.array(gt_waypoints_3d_world), -1.73 / 2)
            gt_right_3d = OffsetTrajectory3D(np.array(gt_waypoints_3d_world), 1.73 / 2)
            gt_left_img = ProjectWorldToImage(gt_left_3d.tolist(), cam_to_ego, ego_to_world)
            gt_right_img = ProjectWorldToImage(gt_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            gt_polygon = np.vstack((np.array(gt_left_img), np.array(gt_right_img)[::-1])).astype(np.int32)
            if gt_polygon.size > 0:
                frame_gt = np.zeros_like(img)
                cv2.fillPoly(frame_gt, [gt_polygon], color=(0, 255, 0))
                mask_gt = frame_gt.astype(bool)
                img[mask_gt] = cv2.addWeighted(img, 0.5, frame_gt, 0.5, 0)[mask_gt]
        
        # Draw predicted trajectory polygon (orange)
        if len(pred_waypoints_3d_world) > 1:
            pred_left_3d = OffsetTrajectory3D(np.array(pred_waypoints_3d_world), -1.73 / 2)
            pred_right_3d = OffsetTrajectory3D(np.array(pred_waypoints_3d_world), 1.73 / 2)
            pred_left_img = ProjectWorldToImage(pred_left_3d.tolist(), cam_to_ego, ego_to_world)
            pred_right_img = ProjectWorldToImage(pred_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            pred_polygon = np.vstack((np.array(pred_left_img), np.array(pred_right_img)[::-1])).astype(np.int32)
            if pred_polygon.size > 0:
                frame_pred = np.zeros_like(img)
                cv2.fillPoly(frame_pred, [pred_polygon], color=(0, 125, 255))
                mask_pred = frame_pred.astype(bool)
                img[mask_pred] = cv2.addWeighted(img, 0.5, frame_pred, 0.5, 0)[mask_pred]
        
        # Draw waypoint markers
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 255, 0), thickness=-1)
        for pt in pred_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 125, 255), thickness=-1)
        
        # Add legend
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 125, 255), 2)
        
        # Add metrics overlay with background box
        ade = metrics['ade']
        fde = metrics['fde']
        error_1s = metrics['error_at_1s']
        failure = metrics['failure_rate']
        
        # Create semi-transparent box for metrics
        box_height = 170
        box_width = 380
        overlay = img.copy()
        cv2.rectangle(overlay, (10, img.shape[0] - box_height - 10), 
                     (box_width, img.shape[0] - 10), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        # Add metrics text
        y_pos = img.shape[0] - box_height
        cv2.putText(img, 'METRICS', (20, y_pos + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'ADE: {ade:.3f} m', (20, y_pos + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f'FDE: {fde:.3f} m', (20, y_pos + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add failure indicator with color coding
        failure_text = 'FAIL' if failure else 'PASS'
        failure_color = (0, 0, 255) if failure else (0, 255, 0)
        cv2.putText(img, f'Failure: {failure_text}', (20, y_pos + 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, failure_color, 2)

        cv2.putText(img, f'Error @1s: {error_1s:.3f} m', (20, y_pos + 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, img)
        
    except Exception as e:
        print(colored(f"Warning: Visualization with metrics failed: {e}", "yellow"))


def visualize_trajectories(image_pil, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, idx, output_path):
    """Overlay ground truth and predicted trajectories on the image."""
    
    if pred_waypoints_2d.shape[0] == 0 or len(pred_waypoints_2d.shape) != 2:
        return
    
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
                    frame_pred = np.zeros_like(img)
                    cv2.fillPoly(frame_pred, [pred_polygon], color=(0, 125, 255))
                    mask_pred = frame_pred.astype(bool)
                    img[mask_pred] = cv2.addWeighted(img, 0.5, frame_pred, 0.5, 0)[mask_pred]
            
            for pt in pred_points_img:
                cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 125, 255), thickness=-1)
        
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 2)
        
        cv2.imwrite(output_path, img)
        print(colored(f"Visualization saved to: {output_path}", "green", attrs=['bold']))
        
    except Exception as e:
        print(colored(f"Visualization failed: {e}", "red"))


def visualize_trajectory_bev(nusc, sample_token, gt_waypoints_local, pred_waypoints_local, ego_translation, ego_rotation, output_path):
    """Add trajectory overlay to bird's eye view image.
    
    BEV is rendered in ego vehicle frame (use_flat_vehicle_coordinates=True by default).
    Our waypoints are already in local/ego frame, so we just need to convert to pixel coordinates.
    
    Args:
        nusc: NuScenes instance
        sample_token: Sample token for rendering
        gt_waypoints_local: Ground truth waypoints in local/ego frame (N, 2)
        pred_waypoints_local: Predicted waypoints in local/ego frame (N, 2)
        ego_translation: Current ego vehicle translation (world coordinates)
        ego_rotation: Current ego vehicle rotation quaternion
        output_path: Path to save the output image
    """
    try:
        # Get sample and render BEV
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        
        # Render BEV (saves to temporary file)
        nusc.render_sample_data(
            lidar_token,
            underlay_map=True,
            out_path=None,
            nsweeps=5,
        )
        
        # Save the matplotlib figure to a temporary file
        temp_bev_path = output_path.replace('.jpg', '_temp.jpg')
        plt.savefig(temp_bev_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Read the BEV image
        bev_img = cv2.imread(temp_bev_path)
        if bev_img is None:
            print(colored("Warning: Could not read BEV image", "yellow"))
            return False
            
        bev_pil = Image.fromarray(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))
        
        # Create figure with the BEV as background
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(bev_pil)
        
        # Get image dimensions
        height, width = bev_img.shape[:2]
        
        # NuScenes render_sample_data uses default axes_limit=40 meters
        bev_range = 40  # meters from center (default in render_sample_data)
        pixels_per_meter = width / (2 * bev_range)
        
        # Center pixel coordinates
        center_x = width / 2
        center_y = height / 2
        
        # Convert local/ego coordinates to pixel coordinates
        # Waypoints are already in ego frame, just need to convert to pixels
        def local_to_pixel(waypoints_local):
            """Convert local/ego frame coordinates to pixel coordinates."""
            # waypoints_local is already in ego frame (forward=x, left=y)
            # BEV pixel coordinates: center is ego vehicle
            pixel_x = center_x + waypoints_local[:, 0] * pixels_per_meter
            pixel_y = center_y - waypoints_local[:, 1] * pixels_per_meter  # Flipped y-axis
            return pixel_x, pixel_y
        
        # Plot ground truth trajectory (green/lime)
        if len(gt_waypoints_local) > 0:
            gt_px, gt_py = local_to_pixel(gt_waypoints_local)
            ax.plot(gt_px, gt_py, 'o-', color='lime', linewidth=3, markersize=8, label='Ground Truth')
        
        # Plot predicted trajectory (orange)
        # Filter out NaN values
        valid_mask = ~np.isnan(pred_waypoints_local).any(axis=1)
        valid_pred = pred_waypoints_local[valid_mask]
        if len(valid_pred) > 0:
            pred_px, pred_py = local_to_pixel(valid_pred)
            ax.plot(pred_px, pred_py, 'o-', color='orange', linewidth=3, markersize=8, label='Predicted')
        
        # Add ego vehicle marker at center
        ax.plot(center_x, center_y, 'r*', markersize=20, label='Ego Vehicle')
        
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        # Set axis limits to match image dimensions
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(temp_bev_path):
            os.remove(temp_bev_path)
        
        return True
    except Exception as e:
        print(colored(f"✗ Failed to create BEV visualization: {e}", "red"))
        import traceback
        traceback.print_exc()
        return False


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
        return self.get_item_by_token(sample_token)
    
    def get_item_by_token(self, sample_token):
        """Get item by sample token."""
        sample = self.nusc.get('sample', sample_token)
        return self._get_item_from_sample(sample)
    
    def _get_item_from_sample(self, sample):
        """Internal method to get item data from a sample."""
        sample_token = sample['token']

        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        ego_trans = np.array(ego_pose_curr['translation'])
        ego_rot = Quaternion(ego_pose_curr['rotation'])

        # Get History (10 frames)
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

        # Get Future Ground Truth (10 frames)
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

        # Transform to Local Coordinates
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

        return {
            'image': image,
            'ego_positions': history_local,
            'waypoints': future_local,
            'cam_to_ego': cam_to_ego,
            'ego_to_world': ego_to_world,
            'lidar': torch_pointcloud,
            'ego_translation': ego_trans,
            'ego_rotation': ego_rot.elements,
            'sample_token': sample_token,
            'cam_token': cam_token
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


def evaluate_single_sample(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(colored("\n" + "="*70, "cyan"))
    print(colored("Single Sample Evaluation", "cyan", attrs=['bold']))
    if args.sample_idx is not None:
        print(colored(f"Sample Index: {args.sample_idx}", "cyan", attrs=['bold']))
    else:
        print(colored(f"Sample Token: {args.sample_token}", "cyan", attrs=['bold']))
    if args.disable_lidar:
        print(colored("MODE: LiDAR Disabled (Vision + Text Only)", "magenta", attrs=['bold']))
    print(colored("="*70 + "\n", "cyan"))
    
    # Load model
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

    # Load dataset
    ds = EvalNuScenes(args.version, args.dataroot, model.prompt_part1, model.prompt_part2, nsweeps=args.nsweeps)
    
    # Get sample - either by index or by token
    if args.sample_idx is not None:
        if args.sample_idx >= len(ds):
            print(colored(f"Error: Sample index {args.sample_idx} out of range (max: {len(ds)-1})", "red"))
            sys.exit(1)
        item = ds.get_item(args.sample_idx)
        sample_identifier = f"idx_{args.sample_idx}"
    else:
        try:
            item = ds.get_item_by_token(args.sample_token)
            sample_identifier = f"token_{args.sample_token[:8]}"
        except Exception as e:
            print(colored(f"Error: Could not load sample with token {args.sample_token}: {e}", "red"))
            sys.exit(1)
    
    # Get sample data
    image = item['image']
    ego_positions = item['ego_positions']
    gt_waypoints = item['waypoints']
    cam_to_ego = item['cam_to_ego']
    ego_to_world = item['ego_to_world']
    lidar = item['lidar']
    sample_token = item['sample_token']
    ego_translation = item['ego_translation']
    ego_rotation = Quaternion(item['ego_rotation'])
    
    # Prepare inputs
    lidar_device = [lidar.to(device)] if not args.disable_lidar else None
    ego_positions_py = [[float(x), float(y)] for (x, y) in ego_positions]
    
    print(colored("Input History (10 positions):", "cyan"))
    for h_idx, pos in enumerate(ego_positions):
        print(f"  Frame {h_idx}: [{pos[0]:.2f}, {pos[1]:.2f}]")
    
    # Generate prediction
    print(colored("\nGenerating prediction...", "yellow"))
    try:
        outputs, gen_texts = model.generate_trajectory([image], lidar_device, [ego_positions_py])
        gen_text = gen_texts[0]
    except Exception as e:
        print(colored(f"Generation failed: {e}", "red"))
        sys.exit(1)
    
    # Parse prediction
    pred_coords = parse_coords_from_text(gen_text, max_points=10)
    num_valid_waypoints = pred_coords.shape[0]
    format_compliant = (pred_coords.shape[0] == 10)
    
    # Pad or truncate to 10 waypoints
    if pred_coords.shape[0] < 10:
        pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
        pred_coords = np.vstack([pred_coords, pad]) if pred_coords.shape[0] > 0 else pad
    elif pred_coords.shape[0] > 10:
        pred_coords = pred_coords[:10]
    
    # Convert gt_waypoints to numpy array
    gt_wp = np.array(gt_waypoints) if isinstance(gt_waypoints, list) else gt_waypoints
    
    # Compute metrics
    diffs = pred_coords - gt_wp
    l2_per_waypoint = np.linalg.norm(diffs, axis=1)
    ade = np.nanmean(l2_per_waypoint)
    fde = l2_per_waypoint[-1]
    error_at_1s = l2_per_waypoint[1] if len(l2_per_waypoint) > 1 else np.nan
    failure_rate = True if (error_at_1s > 10.0 or np.isnan(error_at_1s)) else False
    
    # Print results
    print(colored("\n" + "="*70, "magenta"))
    print(colored("RESULTS", "magenta", attrs=['bold']))
    print(colored("="*70, "magenta"))
    
    print(colored("\nGenerated Text:", "cyan"))
    print(gen_text)
    
    print(colored("\nMetrics:", "cyan"))
    print(f"  Number of valid waypoints: {num_valid_waypoints}")
    print(f"  Format compliant (10 waypoints): {format_compliant}")
    print(f"  ADE (Average Displacement Error): {ade:.4f} m")
    print(f"  FDE (Final Displacement Error): {fde:.4f} m")
    print(f"  Error @ 1s: {error_at_1s:.4f} m")
    print(f"  Failure (>10m @ 1s): {failure_rate}")
    
    print(colored("\nWaypoint-by-Waypoint Comparison:", "cyan"))
    for k in range(min(10, len(gt_wp))):
        if not np.isnan(pred_coords[k][0]):
            p_str = f"[{pred_coords[k][0]:7.2f}, {pred_coords[k][1]:7.2f}]"
            pt_err = np.linalg.norm(pred_coords[k] - gt_wp[k])
            err_str = f"(error: {pt_err:6.2f}m)"
        else:
            p_str = "[    NaN,     NaN]"
            err_str = ""
        print(f"  Waypoint {k}: GT [{gt_wp[k][0]:7.2f}, {gt_wp[k][1]:7.2f}]  ->  Pred: {p_str} {err_str}")
    
    # Create output directory (clear existing contents)
    output_dir = 'eval_outputs/lidar_single_sample'
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save run arguments to JSON
    args_dict = vars(args)
    args_json_path = os.path.join(output_dir, 'run_args.json')
    with open(args_json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(colored(f"\n✓ Run arguments saved to {args_json_path}", "green"))
    
    # Visualize (without metrics)
    output_path = os.path.join(output_dir, f'sample_{sample_identifier}_visualization.jpg')
    pred_waypoints_valid = pred_coords[:num_valid_waypoints] if num_valid_waypoints > 0 else np.array([]).reshape(0, 2)
    visualize_trajectories(
        image,
        gt_wp,
        pred_waypoints_valid,
        cam_to_ego,
        ego_to_world,
        0,  # idx not used in visualization
        output_path
    )
    
    print(colored(f"\n✓ Visualization saved to: {output_path}", "green"))
    
    # Visualize with metrics overlay
    output_path_metrics = os.path.join(output_dir, f'sample_{sample_identifier}_with_metrics.jpg')
    metrics_dict = {
        'ade': ade,
        'fde': fde,
        'error_at_1s': error_at_1s,
        'failure_rate': failure_rate
    }
    visualize_trajectories_with_metrics(
        image,
        gt_wp,
        pred_waypoints_valid,
        cam_to_ego,
        ego_to_world,
        args.sample_idx,
        output_path_metrics,
        metrics_dict
    )
    
    print(colored(f"✓ Visualization with metrics saved to: {output_path_metrics}", "green"))
    
    # Visualize BEV with trajectories
    print(colored("\nRendering bird's eye view...", "yellow"))
    output_path_bev = os.path.join(output_dir, f'sample_{sample_identifier}_bev.jpg')
    bev_success = visualize_trajectory_bev(
        ds.nusc,
        sample_token,
        gt_wp,
        pred_coords,
        ego_translation,
        ego_rotation,
        output_path_bev
    )
    if bev_success:
        print(colored(f"✓ BEV visualization saved to: {output_path_bev}", "green"))
    
    print(colored("\n" + "="*70, "magenta"))
    print(colored("Evaluation Complete!", "green", attrs=['bold']))
    print(colored("="*70 + "\n", "magenta"))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Single Sample - Multimodal LiDAR+CLIP+Qwen Model')

    # Important arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Mutually exclusive: either sample_idx or sample_token
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sample_idx', type=int, help='Sample index to evaluate')
    input_group.add_argument('--sample_token', type=str, help='Sample token to evaluate')
    
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-3B-Instruct')

    # Model arguments
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--sst_config', type=str, default='src/models/mmdet3d/configs/sst_encoder_only_config.py')
    parser.add_argument('--lidar_encoder_path', type=str, default='/home/hice1/rmineyev3/scratch/VLM-Final-Project/Lidar-CLIP/vit_l_14.ckpt')
    parser.add_argument('--mlp_hidden_dim', type=int, default=2048)
    parser.add_argument('--mlp_num_layers', type=int, default=3)
    parser.add_argument('--mlp_dropout', type=float, default=0.1)
    
    # Dataset arguments
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--nsweeps', type=int, default=5)
    
    # Options
    parser.add_argument('--disable_lidar', action='store_true', help='Disable LiDAR input (Vision + Text only)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_single_sample(args)
