#!/usr/bin/env python3
"""
Visualize evaluation results from CSV file.
Reads pre-computed evaluation results and generates visualizations with GT vs Predicted trajectories.
Displays error metrics on the image.
"""

import argparse
import os
import csv
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from termcolor import colored

from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D
from nuscenes import NuScenes
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


def visualize_trajectory_with_metrics(image_pil, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, 
                                     metrics, idx, output_path):
    """
    Overlay ground truth and predicted trajectories on the image with error metrics.
    
    Args:
        image_pil: PIL Image
        gt_waypoints_2d: Ground truth waypoints (N, 2)
        pred_waypoints_2d: Predicted waypoints (M, 2), may contain NaN
        cam_to_ego: Camera to ego transform dict
        ego_to_world: Ego to world transform dict
        metrics: Dict with evaluation metrics (ade, fde, wp errors, etc.)
        idx: Sample index
        output_path: Path to save visualization
    """
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    gt_waypoints_3d = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    valid_pred_mask = ~np.isnan(pred_waypoints_3d).any(axis=1)
    pred_waypoints_3d_valid = pred_waypoints_3d[valid_pred_mask]
    
    try:
        # Project GT waypoints
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
        
        # Draw GT trajectory corridor (green) with transparency
        if len(gt_waypoints_3d) > 1:
            gt_left_3d = OffsetTrajectory3D(gt_waypoints_3d, -1.73 / 2)
            gt_right_3d = OffsetTrajectory3D(gt_waypoints_3d, 1.73 / 2)
            gt_left_img = ProjectWorldToImage(gt_left_3d.tolist(), cam_to_ego, ego_to_world)
            gt_right_img = ProjectWorldToImage(gt_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            gt_polygon = np.vstack((np.array(gt_left_img), np.array(gt_right_img)[::-1])).astype(np.int32)
            if gt_polygon.size > 0:
                frame_gt = np.zeros_like(img)
                cv2.fillPoly(frame_gt, [gt_polygon], color=(0, 255, 0))
                mask_gt = frame_gt.astype(bool)
                img[mask_gt] = cv2.addWeighted(img, 0.5, frame_gt, 0.5, 0)[mask_gt]
        
        # Draw GT waypoints
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 255, 0), thickness=-1)
        
        # Draw predicted trajectory (orange) with transparency
        if len(pred_waypoints_3d_valid) > 0:
            pred_points_img = ProjectWorldToImage(pred_waypoints_3d_valid.tolist(), cam_to_ego, ego_to_world)
            
            if len(pred_waypoints_3d_valid) > 1:
                pred_left_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, -1.73 / 2)
                pred_right_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, 1.73 / 2)
                pred_left_img = ProjectWorldToImage(pred_left_3d.tolist(), cam_to_ego, ego_to_world)
                pred_right_img = ProjectWorldToImage(pred_right_3d.tolist(), cam_to_ego, ego_to_world)
                
                pred_polygon = np.vstack((np.array(pred_left_img), np.array(pred_right_img)[::-1])).astype(np.int32)
                if pred_polygon.size > 0:
                    frame_pred = np.zeros_like(img)
                    cv2.fillPoly(frame_pred, [pred_polygon], color=(0, 125, 255))
                    mask_pred = frame_pred.astype(bool)
                    img[mask_pred] = cv2.addWeighted(img, 0.5, frame_pred, 0.5, 0)[mask_pred]
            
            # Draw predicted waypoints
            for pt in pred_points_img:
                cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 125, 255), thickness=-1)
        
        # Add legend
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 125, 255), 2)
        
        # Add metrics overlay (top-right corner with semi-transparent background)
        metrics_lines = [
            f"Sample {idx}",
            f"ADE: {metrics['ade']:.2f}m",
            f"FDE: {metrics['fde']:.2f}m",
            f"Error@1s: {metrics['error_at_1s']:.2f}m",
            f"Valid WPs: {metrics['num_valid_waypoints']}/10",
            f"Miss@10m: {'YES' if metrics['miss_rate_10m'] > 0 else 'NO'}"
        ]
        
        # Calculate text block size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 30
        padding = 10
        
        max_text_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in metrics_lines])
        block_width = max_text_width + 2 * padding
        block_height = len(metrics_lines) * line_height + 2 * padding
        
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (w - block_width - 10, 10), (w - 10, 10 + block_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Draw text
        y_offset = 10 + padding + 20
        for line in metrics_lines:
            cv2.putText(img, line, (w - block_width - 10 + padding, y_offset), 
                       font, font_scale, (255, 255, 255), font_thickness)
            y_offset += line_height
        
        # Add per-waypoint errors at bottom
        if 'waypoint_errors' in metrics:
            wp_errors = metrics['waypoint_errors']
            error_text = "WP Errors (m): " + ", ".join([f"{e:.1f}" if not np.isnan(e) else "N/A" for e in wp_errors[:5]])
            cv2.putText(img, error_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if len(wp_errors) > 5:
                error_text2 = "              " + ", ".join([f"{e:.1f}" if not np.isnan(e) else "N/A" for e in wp_errors[5:]])
                cv2.putText(img, error_text2, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        
    except Exception as e:
        print(colored(f"✗ Visualization failed for sample {idx}: {e}", "red"))


def visualize_bev_with_metrics(nusc, sample, gt_waypoints, pred_waypoints, metrics, idx, output_path):
    """
    Generate bird's eye view with trajectory overlay and metrics.
    """
    try:
        # Render BEV using NuScenes
        lidar_token = sample['data']['LIDAR_TOP']
        
        nusc.render_sample_data(
            lidar_token,
            underlay_map=True,
            out_path=None,
            nsweeps=5,
        )
        
        # Get current axes
        ax = plt.gca()
        
        # Get ego pose for coordinate transformation
        lidar_data = nusc.get('sample_data', lidar_token)
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # NuScenes BEV uses 40m range by default
        bev_range = 40
        
        # The axes limits are already set by render_sample_data
        # We just need to add trajectory overlay
        
        # Convert world coordinates to ego frame for plotting
        def world_to_ego(waypoints, ego_translation, ego_rotation):
            # Translate to ego-relative coordinates
            rel_x = waypoints[:, 0] - ego_translation[0]
            rel_y = waypoints[:, 1] - ego_translation[1]
            
            # Stack into 2D points for rotation
            rel_points = np.vstack([rel_x, rel_y])
            
            # Rotate from world frame to ego frame
            ego_quat = Quaternion(ego_rotation)
            yaw = ego_quat.yaw_pitch_roll[0]
            
            # Create rotation matrix for yaw only (flat ego frame)
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            
            # Apply rotation
            ego_points = rotation_matrix @ rel_points
            
            return ego_points[0, :], ego_points[1, :]
        
        ego_translation = ego_pose['translation']
        ego_rotation = ego_pose['rotation']
        
        # Plot ground truth trajectory (green)
        if len(gt_waypoints) > 0:
            gt_x, gt_y = world_to_ego(gt_waypoints, ego_translation, ego_rotation)
            ax.plot(gt_x, gt_y, 'o-', color='lime', linewidth=3, markersize=8, label='Ground Truth', zorder=10)
        
        # Plot predicted trajectory (orange)
        valid_pred = pred_waypoints[~np.isnan(pred_waypoints).any(axis=1)]
        if len(valid_pred) > 0:
            pred_x, pred_y = world_to_ego(valid_pred, ego_translation, ego_rotation)
            ax.plot(pred_x, pred_y, 'o-', color='orange', linewidth=3, markersize=8, label='Predicted', zorder=10)
        
        # Add ego vehicle marker at center
        ax.plot(0, 0, 'r*', markersize=20, label='Ego Vehicle', zorder=11)
        
        # Add metrics text
        metrics_text = f"Sample {idx}\n"
        metrics_text += f"ADE: {metrics['ade']:.2f}m\n"
        metrics_text += f"FDE: {metrics['fde']:.2f}m\n"
        metrics_text += f"Error@1s: {metrics['error_at_1s']:.2f}m"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               zorder=12)
        
        ax.legend(loc='upper right', fontsize=12)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(colored(f"✗ Failed to create BEV visualization: {e}", "red"))
        import traceback
        traceback.print_exc()
        return False


def visualize_lidar_with_metrics(nusc, sample, image, gt_waypoints, pred_waypoints, cam_to_ego, ego_to_world, 
                                 metrics, idx, output_path, point_size=5):
    """
    Generate LiDAR overlay visualization with trajectory and metrics.
    """
    try:
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import view_points
        
        # Get sample and sensor data
        cam_token = sample['data']['CAM_FRONT']
        lidar_token = sample['data']['LIDAR_TOP']
        
        cam_data = nusc.get('sample_data', cam_token)
        lidar_data = nusc.get('sample_data', lidar_token)
        
        # Load LiDAR point cloud
        lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(lidar_path)
        
        # Get calibration
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        # Transform from lidar to camera
        pc.rotate(Quaternion(lidar_calib['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_calib['translation']))
        
        # Get poses
        cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Transform to global
        pc.rotate(Quaternion(lidar_pose['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_pose['translation']))
        
        # Transform from global to camera ego
        pc.translate(-np.array(cam_pose['translation']))
        pc.rotate(Quaternion(cam_pose['rotation']).rotation_matrix.T)
        
        # Transform to camera
        pc.translate(-np.array(cam_calib['translation']))
        pc.rotate(Quaternion(cam_calib['rotation']).rotation_matrix.T)
        
        # Project to image
        depths = pc.points[2, :]
        intrinsic = np.array(cam_calib['camera_intrinsic'])
        points = view_points(pc.points[:3, :], intrinsic, normalize=True)
        
        # Filter points in front of camera and within image bounds
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[0, :] < image.size[0])
        mask = np.logical_and(mask, points[1, :] > 0)
        mask = np.logical_and(mask, points[1, :] < image.size[1])
        
        points = points[:, mask]
        depths = depths[mask]
        
        # Create image with LiDAR points
        img_lidar = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Color by depth (closer = red, farther = blue)
        max_depth = 50.0
        for i in range(points.shape[1]):
            depth = min(depths[i], max_depth)
            color_ratio = depth / max_depth
            color = (int(255 * color_ratio), int(255 * (1 - color_ratio) * 0.5), int(255 * (1 - color_ratio)))
            cv2.circle(img_lidar, (int(points[0, i]), int(points[1, i])), 
                     point_size, color, -1)
        
        # Convert to PIL for trajectory overlay
        img_lidar_pil = Image.fromarray(cv2.cvtColor(img_lidar, cv2.COLOR_BGR2RGB))
        
        # Add trajectory overlay with metrics
        visualize_trajectory_with_metrics(
            img_lidar_pil, gt_waypoints, pred_waypoints,
            cam_to_ego, ego_to_world, metrics, idx, output_path
        )
        
        return True
    except Exception as e:
        print(colored(f"✗ LiDAR rendering failed: {e}", "red"))
        import traceback
        traceback.print_exc()
        return False


class EvalNuScenes:
    def __init__(self, version, dataroot):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        # Build sample list
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

        # Get image and calibration
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

        # Get 10 future waypoints
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
            'waypoints': np.array(waypoints, dtype=float),
            'cam_to_ego': cam_to_ego,
            'ego_to_world': ego_to_world,
            'ego_pose': ego_pose,
            'sample_token': sample_token,
            'sample': sample
        }


def load_csv_results(csv_path):
    """Load evaluation results from CSV file."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse waypoint errors
            wp_errors = []
            for i in range(10):
                err_str = row.get(f'wp{i}_error', 'nan')
                try:
                    wp_errors.append(float(err_str))
                except:
                    wp_errors.append(np.nan)
            
            result = {
                'idx': int(row['idx']),
                'cross_entropy_loss': float(row['cross_entropy_loss']) if row['cross_entropy_loss'] else np.nan,
                'perplexity': float(row['perplexity']) if row['perplexity'] else np.nan,
                'token_accuracy': float(row['token_accuracy']) if row['token_accuracy'] else np.nan,
                'num_valid_waypoints': int(row['num_valid_waypoints']),
                'format_compliant': row['format_compliant'].lower() == 'true',
                'ade': float(row['ade']),
                'fde': float(row['fde']),
                'miss_rate_10m': float(row['miss_rate_10m']),
                'failure_at_1s': float(row.get('failure_at_1s', 0)),
                'error_at_1s': float(row.get('error_at_1s', np.nan)),
                'processing_time_sec': float(row.get('processing_time_sec', 0)),
                'waypoint_errors': wp_errors,
                'gen_text': row['gen_text']
            }
            results.append(result)
    
    return results


def visualize_sample(args):
    print(colored(f"\n{'='*60}", "cyan"))
    print(colored("CSV-Based Sample Visualization", "cyan", attrs=['bold']))
    print(colored(f"{'='*60}\n", "cyan"))
    
    # Load CSV results
    print(colored(f"Loading results from: {args.csv_path}", "yellow"))
    results = load_csv_results(args.csv_path)
    print(colored(f"✓ Loaded {len(results)} results\n", "green"))
    
    # Find result for requested sample
    sample_result = None
    for r in results:
        if r['idx'] == args.sample_idx:
            sample_result = r
            break
    
    if sample_result is None:
        print(colored(f"✗ Sample {args.sample_idx} not found in CSV", "red"))
        return
    
    # Load NuScenes dataset
    print(colored("Loading NuScenes dataset...", "yellow"))
    ds = EvalNuScenes(args.version, args.dataroot)
    print(colored(f"✓ Dataset loaded ({len(ds)} samples available)\n", "green"))

    while True:
        
        # Validate index
        if args.sample_idx < 0 or args.sample_idx >= len(ds):
            print(colored(f"✗ Error: Sample index {args.sample_idx} out of range [0, {len(ds)-1}]", "red"))
            return
        
        print(colored(f"Visualizing sample index: {args.sample_idx}", "cyan", attrs=['bold']))
        
        # Get sample data
        item = ds.get_item(args.sample_idx)
        image = item['image']
        gt_waypoints = item['waypoints']
        cam_to_ego = item['cam_to_ego']
        ego_to_world = item['ego_to_world']
        ego_pose = item['ego_pose']
        
        # Parse predicted coordinates from generated text
        pred_coords = parse_coords_from_text(sample_result['gen_text'], max_points=10)
        
        # Pad to 10 waypoints if needed
        if pred_coords.shape[0] < 10:
            pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
            pred_coords = np.vstack([pred_coords, pad])
        elif pred_coords.shape[0] > 10:
            pred_coords = pred_coords[:10]
        
        # Print metrics
        print(colored(f"\n{'='*60}", "cyan"))
        print(colored("METRICS", "cyan", attrs=['bold']))
        print(colored(f"{'='*60}\n", "cyan"))
        
        print(colored("Loss Metrics:", "yellow", attrs=['bold']))
        print(f"  Cross-Entropy Loss:  {sample_result['cross_entropy_loss']:.4f}")
        print(f"  Perplexity:          {sample_result['perplexity']:.4f}")
        print(f"  Token Accuracy:      {sample_result['token_accuracy']:.4f}\n")
        
        print(colored("Trajectory Metrics:", "yellow", attrs=['bold']))
        print(f"  Valid Waypoints:     {sample_result['num_valid_waypoints']}/10")
        print(f"  Format Compliant:    {'✓' if sample_result['format_compliant'] else '✗'}")
        print(f"  ADE (meters):        {sample_result['ade']:.4f}")
        print(f"  FDE (meters):        {sample_result['fde']:.4f}")
        print(f"  Error @ 1s:          {sample_result['error_at_1s']:.4f}")
        print(f"  Miss @ 10m:          {'✗ MISS' if sample_result['miss_rate_10m'] > 0 else '✓ HIT'}\n")
        
        print(colored("Per-Waypoint L2 Errors (meters):", "yellow", attrs=['bold']))
        for i, err in enumerate(sample_result['waypoint_errors']):
            status = "✓" if not np.isnan(err) and err < 10.0 else "✗"
            print(f"  WP {i+1:2d}: {err:6.3f}m {status}")
        
        print(colored("\nGenerated Text:", "yellow", attrs=['bold']))
        print(f"{sample_result['gen_text'][:500]}\n")
        
        # Create visualizations
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 1. Main camera view with metrics
        output_path = os.path.join(args.output_dir, f'sample_camera.jpg')
        print(colored(f"Creating camera visualization...", "yellow"))
        visualize_trajectory_with_metrics(
            image,
            gt_waypoints,
            pred_coords,
            cam_to_ego,
            ego_to_world,
            sample_result,
            args.sample_idx,
            output_path
        )
        print(colored(f"✓ Saved: {output_path}", "green"))
        
        # 2. Bird's eye view
        if args.use_bev:
            print(colored(f"Creating BEV visualization...", "yellow"))
            bev_output_path = os.path.join(args.output_dir, f'sample_bev.jpg')
            
            if visualize_bev_with_metrics(ds.nusc, item['sample'], gt_waypoints, pred_coords, 
                                         sample_result, args.sample_idx, bev_output_path):
                print(colored(f"✓ Saved: {bev_output_path}", "green"))
        
        # 3. LiDAR overlay
        if args.use_lidar:
            print(colored(f"Creating LiDAR visualization...", "yellow"))
            lidar_output_path = os.path.join(args.output_dir, f'sample_lidar.jpg')
            
            if visualize_lidar_with_metrics(ds.nusc, item['sample'], image, gt_waypoints, pred_coords,
                                          cam_to_ego, ego_to_world, sample_result, args.sample_idx,
                                          lidar_output_path, point_size=args.lidar_point_size):
                print(colored(f"✓ Saved: {lidar_output_path}", "green"))
        
        print(colored(f"\n{'='*60}\n", "cyan"))

        # New input
        new_idx = input(colored("Enter new sample index to visualize (or 'q' to quit): ", "yellow"))
        if new_idx.lower() == 'q':
            print(colored("Exiting visualization.", "cyan"))
            break
        elif new_idx.strip() == '':
            print(colored("No input provided. Reading next sample.", "yellow"))
            args.sample_idx += 1
            continue
        try:
            args.sample_idx = int(new_idx)
        except ValueError:
            print(colored("Invalid input. Please enter a valid sample index or 'q' to quit.", "red"))
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize evaluation results from CSV")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to evaluation CSV file')
    parser.add_argument('--sample_idx', type=int, required=True, help='Index of sample to visualize')
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3',
                       help='Path to NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-test', help='NuScenes version')
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/csv_vis', 
                       help='Directory to save visualizations')
    parser.add_argument('--use_bev', action='store_true', default=True, help='Include BEV visualization')
    parser.add_argument('--use_lidar', action='store_true', default=True, help='Include LiDAR overlay visualization')
    parser.add_argument('--lidar_point_size', type=int, default=5, help='Size of LiDAR points in visualization')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_sample(args)
