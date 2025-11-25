#!/usr/bin/env python3
"""
Single Sample Evaluation Script for OpenEMMA Baseline
Evaluates one specific sample by index/token using the OpenEMMA pipeline (Chain-of-Thought -> Kinematics -> Trajectory).
"""

import os
import sys
import argparse
import json
import re
import ast
import numpy as np
import cv2
import torch
from PIL import Image
from termcolor import colored
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

# NuScenes imports
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# OpenEMMA imports
from src.openemma.vlm.base_backbone import BaseOpenEMMA
from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D

def parse_coords_from_text(text, max_points=10):
    """
    Extract waypoint coordinates from OpenEMMA generated text.
    OpenEMMA often returns a python-list style string: "[[x,y], [x,y]...]"
    """
    try:
        # Try direct evaluation if it's a clean list string
        # Clean up any markdown code blocks
        clean_text = text.replace("```python", "").replace("```", "").strip()
        if clean_text.startswith("[") and clean_text.endswith("]"):
            data = ast.literal_eval(clean_text)
            return np.array(data)
    except:
        pass

    # Fallback to regex finding floats
    text = text.replace(";", " ") 
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    nums = [float(x) for x in nums]
    
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        if len(pairs) >= max_points:
            break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)

def visualize_trajectories_with_metrics(image_path, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, output_path, metrics):
    """Overlay trajectories and metrics on the image."""
    
    img = cv2.imread(image_path)
    if img is None:
        print(colored(f"Error reading image: {image_path}", "red"))
        return

    # Convert waypoints from list to numpy array if needed
    gt_waypoints_2d = np.array(gt_waypoints_2d) if isinstance(gt_waypoints_2d, list) else gt_waypoints_2d
    pred_waypoints_2d = np.array(pred_waypoints_2d) if isinstance(pred_waypoints_2d, list) else pred_waypoints_2d
    
    # Add z=0 to make 3D points in local frame
    gt_waypoints_3d_local = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d_local = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    # Transform from local/ego frame to world frame
    ego_translation = np.array(ego_to_world['translation'])
    ego_rotation = Quaternion(ego_to_world['rotation'])
    
    # Transform GT waypoints: local -> world
    gt_waypoints_3d_world = []
    for local_pt in gt_waypoints_3d_local:
        world_pt = ego_rotation.rotate(local_pt) + ego_translation
        gt_waypoints_3d_world.append(world_pt)
    
    # Transform predicted waypoints: local -> world
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
        cv2.putText(img, 'Orange: OpenEMMA', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 125, 255), 2)
        
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
        print(colored(f"✓ Visualization saved to: {output_path}", "green"))
        
    except Exception as e:
        print(colored(f"Warning: Visualization with metrics failed: {e}", "yellow"))
        import traceback
        traceback.print_exc()

def visualize_trajectory_bev(nusc, sample_token, gt_waypoints_local, pred_waypoints_local, ego_translation, ego_rotation, output_path):
    """Add trajectory overlay to bird's eye view image."""
    try:
        # Get sample and render BEV
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        
        # Render BEV (saves to temporary file)
        # Suppress stdout during render to keep console clean
        nusc.render_sample_data(
            lidar_token,
            underlay_map=True,
            out_path=None,
            nsweeps=5,
            verbose=False
        )
        
        # Save the matplotlib figure to a temporary file
        temp_bev_path = output_path.replace('.jpg', '_temp.jpg')
        plt.savefig(temp_bev_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Read the BEV image
        bev_img = cv2.imread(temp_bev_path)
        if bev_img is None:
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
        
        center_x = width / 2
        center_y = height / 2
        
        def local_to_pixel(waypoints_local):
            pixel_x = center_x + waypoints_local[:, 0] * pixels_per_meter
            pixel_y = center_y - waypoints_local[:, 1] * pixels_per_meter  # Flipped y-axis
            return pixel_x, pixel_y
        
        # Plot ground truth trajectory (green/lime)
        if len(gt_waypoints_local) > 0:
            gt_px, gt_py = local_to_pixel(gt_waypoints_local)
            ax.plot(gt_px, gt_py, 'o-', color='lime', linewidth=3, markersize=8, label='Ground Truth')
        
        # Plot predicted trajectory (orange)
        valid_mask = ~np.isnan(pred_waypoints_local).any(axis=1)
        valid_pred = pred_waypoints_local[valid_mask]
        if len(valid_pred) > 0:
            pred_px, pred_py = local_to_pixel(valid_pred)
            ax.plot(pred_px, pred_py, 'o-', color='orange', linewidth=3, markersize=8, label='OpenEMMA')
        
        # Add ego vehicle marker at center
        ax.plot(center_x, center_y, 'r*', markersize=20, label='Ego Vehicle')
        
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(temp_bev_path):
            os.remove(temp_bev_path)
        
        print(colored(f"✓ BEV visualization saved to: {output_path}", "green"))
        return True
    except Exception as e:
        print(colored(f"✗ Failed to create BEV visualization: {e}", "red"))
        return False

class EvalNuScenesOpenEMMAWrapper:
    """Wrapper around NuScenes for OpenEMMA Single Sample Evaluation"""
    def __init__(self, version, dataroot):
        print(colored(f"Initializing NuScenes ({version})...", "cyan"))
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        
        # Build simple index map
        self.sample_tokens = []
        for scene in self.nusc.scene:
            nbr_samples = scene['nbr_samples']
            if nbr_samples < 20: continue
            sample = self.nusc.get('sample', scene['first_sample_token'])
            # Skip first 2 like the training set
            for _ in range(9): sample = self.nusc.get('sample', sample['next'])
            # Add valid samples
            for _ in range(nbr_samples - 19):
                self.sample_tokens.append(sample['token'])
                sample = self.nusc.get('sample', sample['next'])

    def __len__(self):
        return len(self.sample_tokens)

    def get_item(self, idx):
        return self.get_item_by_token(self.sample_tokens[idx])

    def get_item_by_token(self, sample_token):
        sample = self.nusc.get('sample', sample_token)

        # 1. Ego History (Normalized/Local Frame) - Needed for OpenEMMA Logic
        ego_positions = [] 
        curr = sample
        for _ in range(10):
            cam_data = self.nusc.get('sample_data', curr['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_positions.append([float(ego_pose['translation'][0]), float(ego_pose['translation'][1])])
            if curr['prev']: 
                curr = self.nusc.get('sample', curr['prev'])
            else: 
                ego_positions.append(ego_positions[-1]) 
        ego_positions.reverse()

        # 2. Image Path & Calibration
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        image_path = os.path.join(self.dataroot, cam_data['filename'])
        
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        cam_to_ego = {'translation': cam_calib['translation'], 'rotation': cam_calib['rotation'], 'camera_intrinsic': np.array(cam_calib['camera_intrinsic'])}
        ego_to_world = {'translation': ego_pose_curr['translation'], 'rotation': ego_pose_curr['rotation']}

        # 3. Future Waypoints (GT) - World Frame for reference, Local for processing
        waypoints_world = []
        curr = sample
        for _ in range(10):
            if curr['next'] == '': break 
            curr = self.nusc.get('sample', curr['next'])
            cam_data_next = self.nusc.get('sample_data', curr['data']['CAM_FRONT'])
            pose_next = self.nusc.get('ego_pose', cam_data_next['ego_pose_token'])['translation']
            waypoints_world.append([float(pose_next[0]), float(pose_next[1])])
        
        while len(waypoints_world) < 10:
            waypoints_world.append(waypoints_world[-1] if len(waypoints_world)>0 else [0,0])

        # 4. Compute Local Trajectories (Required for OpenEMMA prompts)
        ego_trans = np.array(ego_pose_curr['translation'])
        ego_rot = Quaternion(ego_pose_curr['rotation'])
        
        # History to Local
        his_trajs_local = []
        for p in ego_positions:
            global_p = np.array(p + [0]) 
            local_p = ego_rot.inverse.rotate(global_p - ego_trans)
            his_trajs_local.append([local_p[0], local_p[1]])
            
        # Future to Local
        fut_trajs_local = []
        for p in waypoints_world:
            global_p = np.array(p + [0])
            local_p = ego_rot.inverse.rotate(global_p - ego_trans)
            fut_trajs_local.append([local_p[0], local_p[1]])

        his_diff = np.diff(np.array(his_trajs_local), axis=0)
        fut_diff = np.diff(np.array(fut_trajs_local), axis=0)

        return {
            'sample_token': sample_token,
            'image_path': image_path,
            'gt_waypoints_local': np.array(fut_trajs_local),
            'gt_ego_his_trajs': his_trajs_local, 
            'gt_ego_fut_trajs': fut_trajs_local, 
            'gt_ego_his_diff': his_diff,
            'gt_ego_fut_diff': fut_diff,
            'cam_to_ego': cam_to_ego,
            'ego_to_world': ego_to_world,
            'ego_positions_world': ego_positions,
            'ego_translation': ego_trans,
            'ego_rotation': ego_rot.elements
        }

def evaluate_single_sample(args):
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(colored("\n" + "="*70, "cyan"))
    print(colored("OpenEMMA Single Sample Evaluation", "cyan", attrs=['bold']))
    if args.sample_idx is not None:
        print(colored(f"Sample Index: {args.sample_idx}", "cyan", attrs=['bold']))
    else:
        print(colored(f"Sample Token: {args.sample_token}", "cyan", attrs=['bold']))
    print(colored(f"Model: {args.model_id}", "cyan"))
    print(colored("="*70 + "\n", "cyan"))
    
    # 1. Initialize OpenEMMA Model
    print(colored("Initializing OpenEMMA Model...", "yellow"))
    emma_model = BaseOpenEMMA(args)
    
    # 2. Initialize Dataset
    ds = EvalNuScenesOpenEMMAWrapper(args.version, args.dataroot)
    
    # 3. Get Sample Data
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
            
    print(colored(f"Loaded sample: {item['sample_token']}", "green"))
    
    # 4. OpenEMMA Inference Pipeline
    print(colored("\nRunning OpenEMMA Inference Chain...", "yellow"))
    
    # 4a. Compute Command (Oracle based on Future GT per OpenEMMA paper)
    try:
        command = emma_model.compute_command(item['gt_ego_fut_trajs'])
    except:
        command = "MOVE FORWARD"
    print(f"  Command: {command}")

    # 4b. Prepare data dict for model
    emma_data = {
        "gt_ego_fut_diff": item['gt_ego_fut_diff'],
        "gt_ego_fut_trajs": item['gt_ego_fut_trajs'],
        "gt_ego_his_diff": item['gt_ego_his_diff'],
        "gt_ego_his_trajs": item['gt_ego_his_trajs']
    }

    # 4c. Generate Prediction
    # This runs: Scenedescription -> Object Det -> Meta Action -> Prompt -> Integration
    response_text = emma_model.generate_waypoints(
        command=command,
        image_path=item['image_path'],
        data=emma_data,
        backbone=None,
        args=args
    )
    
    print(colored("\nRaw Model Output:", "cyan"))
    print(response_text)
    
    # 5. Parse Output
    pred_coords = parse_coords_from_text(response_text)
    num_valid_waypoints = pred_coords.shape[0]
    
    # Pad or truncate to 10 waypoints
    if pred_coords.shape[0] < 10:
        pad = np.tile(pred_coords[-1], (10 - pred_coords.shape[0], 1)) if pred_coords.shape[0] > 0 else np.zeros((10, 2))
        pred_coords = np.vstack([pred_coords, pad]) if pred_coords.shape[0] > 0 else pad
    elif pred_coords.shape[0] > 10:
        pred_coords = pred_coords[:10]
        
    # 6. Compute Metrics
    gt_wp = item['gt_waypoints_local']
    diffs = pred_coords - gt_wp
    l2_per_waypoint = np.linalg.norm(diffs, axis=1)
    ade = np.mean(l2_per_waypoint)
    fde = l2_per_waypoint[-1]
    error_at_1s = l2_per_waypoint[1] if len(l2_per_waypoint) > 1 else np.nan
    failure_rate = True if (error_at_1s > 10.0 or np.isnan(error_at_1s)) else False

    # 7. Report Results
    print(colored("\n" + "="*70, "magenta"))
    print(colored("RESULTS", "magenta", attrs=['bold']))
    print(colored("="*70, "magenta"))
    
    print(colored("\nMetrics:", "cyan"))
    print(f"  Valid waypoints parsed: {num_valid_waypoints}")
    print(f"  ADE: {ade:.4f} m")
    print(f"  FDE: {fde:.4f} m")
    print(f"  Error @ 1s: {error_at_1s:.4f} m")
    print(f"  Failure: {failure_rate}")
    
    # 8. Setup Output Directory
    output_dir = 'eval_outputs/openemma_single_sample'
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Args
    with open(os.path.join(output_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # 9. Visualizations
    
    # A. Image with Metrics Overlay
    vis_path = os.path.join(output_dir, f'openemma_{sample_identifier}_vis.jpg')
    metrics_dict = {
        'ade': ade,
        'fde': fde,
        'error_at_1s': error_at_1s,
        'failure_rate': failure_rate
    }
    
    visualize_trajectories_with_metrics(
        item['image_path'],
        gt_wp,
        pred_coords,
        item['cam_to_ego'],
        item['ego_to_world'],
        vis_path,
        metrics_dict
    )
    
    # B. Bird's Eye View (BEV)
    bev_path = os.path.join(output_dir, f'openemma_{sample_identifier}_bev.jpg')
    print(colored("\nRendering bird's eye view...", "yellow"))
    visualize_trajectory_bev(
        ds.nusc,
        item['sample_token'],
        gt_wp,
        pred_coords,
        item['ego_translation'],
        Quaternion(item['ego_rotation']),
        bev_path
    )
    
    print(colored("\n" + "="*70, "magenta"))
    print(colored("Evaluation Complete!", "green", attrs=['bold']))
    print(colored("="*70 + "\n", "magenta"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Single Sample - OpenEMMA Baseline')
    
    # OpenEMMA Model Args
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', 
                        help='Huggingface model ID')
    parser.add_argument('--api_key', type=str, default=None, help='API Key if using GPT')
    
    # Sample Selection (Mutually Exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sample_idx', type=int, help='Sample index to evaluate')
    input_group.add_argument('--sample_token', type=str, help='Sample token to evaluate')
    
    # NuScenes Args
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3/')
    parser.add_argument('--version', type=str, default='v1.0-test')
    
    args = parser.parse_args()
    evaluate_single_sample(args)
