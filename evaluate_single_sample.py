"""
Evaluate a single NuScenes sample by index.
Outputs evaluation metrics to terminal and saves visualization to ./temp_vis/
"""

import argparse
import os
import re
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from termcolor import colored

from src.models.qwen_clip_model import QwenCLIPModel
from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D
from nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion


def parse_coords_from_text(text, max_points=10):
    """Extract waypoint coordinates from generated text.
    
    Returns:
        numpy array of shape (N, 2) where N <= max_points
    """
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


def visualize_trajectory(image_pil, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, idx, output_dir, return_img=False):
    """Overlay ground truth and predicted trajectories on the image."""
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    gt_waypoints_3d = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    valid_pred_mask = ~np.isnan(pred_waypoints_3d).any(axis=1)
    pred_waypoints_3d_valid = pred_waypoints_3d[valid_pred_mask]
    
    try:
        # Project GT waypoints
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
        
        # Draw GT trajectory polygon (green)
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
        
        # Draw predicted trajectory (orange)
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
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 2)
        
        # Save
        if return_img:
            return img
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'sample_{idx:04d}.jpg')
            cv2.imwrite(output_path, img)
            print(colored(f"\n✓ Visualization saved to: {output_path}", "green"))
        
    except Exception as e:
        print(colored(f"✗ Visualization failed: {e}", "red"))
        if return_img:
            return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


class EvalNuScenes:
    def __init__(self, version, dataroot, prompt_part1, prompt_part2):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.prompt_part1 = prompt_part1
        self.prompt_part2 = prompt_part2
        
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

        # Get last 3 ego positions
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
            'ego_positions': ego_positions,
            'waypoints': np.array(waypoints, dtype=float),
            'cam_to_ego': cam_to_ego,
            'ego_to_world': ego_to_world,
            'sample_token': sample_token,
            'sample': sample
        }


def add_trajectory_to_matplotlib(ax, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, img_width=1600, img_height=900):
    """Add trajectory overlay to matplotlib axes (for LiDAR rendering)."""
    try:
        gt_waypoints_3d = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
        pred_waypoints_3d = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
        
        valid_pred_mask = ~np.isnan(pred_waypoints_3d).any(axis=1)
        pred_waypoints_3d_valid = pred_waypoints_3d[valid_pred_mask]
        
        # Project GT waypoints
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
        
        # Draw GT trajectory corridor (green)
        if len(gt_waypoints_3d) > 1:
            gt_left_3d = OffsetTrajectory3D(gt_waypoints_3d, -1.73 / 2)
            gt_right_3d = OffsetTrajectory3D(gt_waypoints_3d, 1.73 / 2)
            gt_left_img = ProjectWorldToImage(gt_left_3d.tolist(), cam_to_ego, ego_to_world)
            gt_right_img = ProjectWorldToImage(gt_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            # Create polygon for GT corridor
            gt_left_arr = np.array(gt_left_img)
            gt_right_arr = np.array(gt_right_img)
            gt_polygon = np.vstack((gt_left_arr, gt_right_arr[::-1]))
            
            from matplotlib.patches import Polygon
            poly_gt = Polygon(gt_polygon, alpha=0.3, facecolor='green', edgecolor='green', linewidth=2)
            ax.add_patch(poly_gt)
        
        # Draw GT waypoints
        gt_points_arr = np.array(gt_points_img)
        ax.plot(gt_points_arr[:, 0], gt_points_arr[:, 1], 'o', color='green', markersize=10, label='Ground Truth')
        
        # Draw predicted trajectory (orange)
        if len(pred_waypoints_3d_valid) > 0:
            pred_points_img = ProjectWorldToImage(pred_waypoints_3d_valid.tolist(), cam_to_ego, ego_to_world)
            
            if len(pred_waypoints_3d_valid) > 1:
                pred_left_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, -1.73 / 2)
                pred_right_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, 1.73 / 2)
                pred_left_img = ProjectWorldToImage(pred_left_3d.tolist(), cam_to_ego, ego_to_world)
                pred_right_img = ProjectWorldToImage(pred_right_3d.tolist(), cam_to_ego, ego_to_world)
                
                # Create polygon for predicted corridor
                pred_left_arr = np.array(pred_left_img)
                pred_right_arr = np.array(pred_right_img)
                pred_polygon = np.vstack((pred_left_arr, pred_right_arr[::-1]))
                
                from matplotlib.patches import Polygon
                poly_pred = Polygon(pred_polygon, alpha=0.3, facecolor='orange', edgecolor='orange', linewidth=2)
                ax.add_patch(poly_pred)
            
            # Draw predicted waypoints
            pred_points_arr = np.array(pred_points_img)
            ax.plot(pred_points_arr[:, 0], pred_points_arr[:, 1], 'o', color='orange', markersize=10, label='Predicted')
        
        ax.legend(loc='upper left', fontsize=12, framealpha=0.8)
        
    except Exception as e:
        print(colored(f"Warning: Could not add trajectory overlay: {e}", "yellow"))


def visualize_trajectory_bev(bev_image_path, gt_waypoints, pred_waypoints, ego_pose, output_path):
    """Add trajectory overlay to bird's eye view image."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Read the BEV image
        bev_img = cv2.imread(bev_image_path)
        bev_pil = Image.fromarray(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))
        
        # Create figure with the BEV as background
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(bev_pil)
        
        # Get image dimensions
        height, width = bev_img.shape[:2]
        
        # NuScenes BEV rendering uses a default range, typically 50m
        # The ego vehicle is at the center of the image
        bev_range = 50  # meters from center
        pixels_per_meter = width / (2 * bev_range)
        
        # Center pixel coordinates
        center_x = width / 2
        center_y = height / 2
        
        # Convert world coordinates to pixel coordinates
        # Ego is at center, x points forward, y points left
        def world_to_pixel(waypoints, ego_translation):
            # Relative to ego position
            rel_x = waypoints[:, 0] - ego_translation[0]
            rel_y = waypoints[:, 1] - ego_translation[1]
            
            # In BEV: x forward -> up (negative y pixel), y left -> left (negative x pixel)
            pixel_x = center_x - rel_y * pixels_per_meter
            pixel_y = center_y - rel_x * pixels_per_meter
            
            return pixel_x, pixel_y
        
        ego_translation = ego_pose['translation']
        
        # Plot ground truth trajectory (green)
        if len(gt_waypoints) > 0:
            gt_px, gt_py = world_to_pixel(gt_waypoints, ego_translation)
            ax.plot(gt_px, gt_py, 'o-', color='lime', linewidth=3, markersize=8, label='Ground Truth')
        
        # Plot predicted trajectory (orange)
        valid_pred = pred_waypoints[~np.isnan(pred_waypoints).any(axis=1)]
        if len(valid_pred) > 0:
            pred_px, pred_py = world_to_pixel(valid_pred, ego_translation)
            ax.plot(pred_px, pred_py, 'o-', color='orange', linewidth=3, markersize=8, label='Predicted')
        
        # Add ego vehicle marker at center
        ax.plot(center_x, center_y, 'r*', markersize=20, label='Ego Vehicle')
        
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(colored(f"✗ Failed to add trajectory to BEV: {e}", "red"))
        return False


def evaluate_single_sample(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(colored(f"\n{'='*60}", "cyan"))
    print(colored("Single Sample Evaluation", "cyan", attrs=['bold']))
    print(colored(f"{'='*60}\n", "cyan"))
    
    # Load model
    print(colored("Loading model...", "yellow"))
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    model = QwenCLIPModel(device, qwen_model_name=args.llm, checkpoint_path=args.checkpoint)
    model.eval()
    print(colored("✓ Model loaded\n", "green"))

    # Load dataset
    print(colored("Loading NuScenes dataset...", "yellow"))
    ds = EvalNuScenes(args.version, args.dataroot, model.prompt_part1, model.prompt_part2)
    print(colored(f"✓ Dataset loaded ({len(ds)} samples available)\n", "green"))
    
    # Validate index
    if args.sample_idx < 0 or args.sample_idx >= len(ds):
        print(colored(f"✗ Error: Sample index {args.sample_idx} out of range [0, {len(ds)-1}]", "red"))
        return
    
    print(colored(f"Evaluating sample index: {args.sample_idx}", "cyan", attrs=['bold']))
    
    # Get sample data
    start_time = time.time()
    item = ds.get_item(args.sample_idx)
    image = item['image']
    ego_positions = item['ego_positions']
    gt_waypoints = item['waypoints']
    cam_to_ego = item['cam_to_ego']
    ego_to_world = item['ego_to_world']
    
    # Process image
    pixel_values = model.image_processor(images=[image], return_tensors='pt').pixel_values.to(device)
    ego_positions_py = [[float(x), float(y)] for (x, y) in ego_positions]
    
    # === Compute Cross-Entropy Loss ===
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    cross_entropy_loss = np.nan
    token_accuracy = np.nan
    perplexity = np.nan
    
    try:
        with torch.no_grad():
            pos_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_positions])
            prompt = f"{model.prompt_part1}[{pos_str}]\n{model.prompt_part2}"
            
            wp_str = ", ".join([f"[{wp[0]:.2f}, {wp[1]:.2f}]" for wp in gt_waypoints])
            target_string = "Future Trajectory: " + wp_str
            
            full_text = prompt + target_string
            input_ids = model.tokenizer(full_text, return_tensors="pt").input_ids.to(device)
            
            labels = input_ids.clone()
            prompt_tokens = model.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_length = prompt_tokens.shape[1]
            labels[:, :prompt_length] = -100
            
            logits = model(pixel_values, input_ids)
            
            num_image_patches = logits.shape[1] - labels.shape[1]
            logits_for_loss = logits[:, num_image_patches:-1, :]
            labels_for_loss = labels[:, 1:]
            
            losses = loss_fn(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                labels_for_loss.reshape(-1)
            )
            
            valid_losses = losses[labels_for_loss.reshape(-1) != -100]
            if len(valid_losses) > 0:
                cross_entropy_loss = valid_losses.mean().item()
                perplexity = np.exp(cross_entropy_loss)
                
                preds = logits_for_loss.argmax(dim=-1)
                valid_mask = labels_for_loss != -100
                if valid_mask.sum() > 0:
                    correct = (preds == labels_for_loss) & valid_mask
                    token_accuracy = correct.sum().item() / valid_mask.sum().item()
    except Exception as e:
        print(colored(f"✗ Loss computation failed: {e}", "red"))
    
    # === Generate trajectory ===
    try:
        outputs, gen_texts = model.generate_trajectory(pixel_values, [ego_positions_py])
        gen_text = gen_texts[0]
    except Exception as e:
        print(colored(f"✗ Generation failed: {e}", "red"))
        return
    
    # Parse coordinates
    pred_coords = parse_coords_from_text(gen_text, max_points=10)
    num_valid_waypoints = pred_coords.shape[0]
    format_compliant = (num_valid_waypoints == 10)
    
    if pred_coords.shape[0] < 10:
        pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
        pred_coords = np.vstack([pred_coords, pad])
    elif pred_coords.shape[0] > 10:
        pred_coords = pred_coords[:10]
    
    # Compute trajectory metrics
    diffs = pred_coords - gt_waypoints
    l2_per_waypoint = np.linalg.norm(diffs, axis=1)
    ade = np.nanmean(l2_per_waypoint)
    fde = l2_per_waypoint[-1]
    miss_rate_10m = 1.0 if fde > 10.0 else 0.0
    
    processing_time = time.time() - start_time
    
    # === Print Results ===
    print(colored(f"\n{'='*60}", "cyan"))
    print(colored("RESULTS", "cyan", attrs=['bold']))
    print(colored(f"{'='*60}\n", "cyan"))
    
    print(colored("Generated Text:", "yellow", attrs=['bold']))
    print(f"{gen_text}\n")
    
    print(colored("Loss Metrics:", "yellow", attrs=['bold']))
    print(f"  Cross-Entropy Loss:  {cross_entropy_loss:.4f}")
    print(f"  Perplexity:          {perplexity:.4f}")
    print(f"  Token Accuracy:      {token_accuracy:.4f}\n")
    
    print(colored("Trajectory Metrics:", "yellow", attrs=['bold']))
    print(f"  Valid Waypoints:     {num_valid_waypoints}/10")
    print(f"  Format Compliant:    {'✓' if format_compliant else '✗'}")
    print(f"  ADE (meters):        {ade:.4f}")
    print(f"  FDE (meters):        {fde:.4f}")
    print(f"  Miss @ 10m:          {'✗ MISS' if miss_rate_10m > 0 else '✓ HIT'}\n")
    
    print(colored("Per-Waypoint L2 Errors (meters):", "yellow", attrs=['bold']))
    for i, err in enumerate(l2_per_waypoint):
        status = "✓" if not np.isnan(err) and err < 10.0 else "✗"
        print(f"  WP {i+1:2d}: {err:6.3f}m {status}")
    
    print(f"\n{colored('Processing Time:', 'yellow', attrs=['bold'])} {processing_time:.3f}s")
    print(f"{colored('FPS:', 'yellow', attrs=['bold'])} {1.0/processing_time:.2f}\n")
    
    # Save visualizations
    if args.save_vis:
        os.makedirs(args.output_dir, exist_ok=True)
        pred_waypoints_valid = pred_coords[:num_valid_waypoints] if num_valid_waypoints > 0 else np.array([]).reshape(0, 2)
        
        # 1. Main trajectory visualization
        visualize_trajectory(
            image,
            gt_waypoints,
            pred_waypoints_valid,
            cam_to_ego,
            ego_to_world,
            args.sample_idx,
            args.output_dir
        )
        
        # 2. LiDAR overlay visualization
        if args.show_lidar:
            try:
                print(colored("\nRendering LiDAR overlay...", "yellow"))
                ds.nusc.render_pointcloud_in_image(
                    sample_token=item['sample_token'],
                    dot_size=args.lidar_point_size,
                    pointsensor_channel='LIDAR_TOP',
                    camera_channel='CAM_FRONT',
                    out_path=None,
                    render_intensity=False,
                    show_lidarseg=False,
                    show_panoptic=False
                )
                
                # Add trajectory overlay to the matplotlib figure before saving
                import matplotlib.pyplot as plt
                ax = plt.gca()
                add_trajectory_to_matplotlib(
                    ax, gt_waypoints, pred_waypoints_valid,
                    cam_to_ego, ego_to_world,
                    img_width=image.size[0], img_height=image.size[1]
                )
                
                output_path = os.path.join(args.output_dir, f'sample_{args.sample_idx:04d}_lidar.jpg')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(colored(f"✓ LiDAR with trajectory overlay saved to: {output_path}", "green"))
            except Exception as e:
                print(colored(f"✗ LiDAR rendering failed: {e}", "red"))
        
        # 3. Bird's eye view
        if args.show_bev:
            try:
                print(colored("Rendering bird's eye view...", "yellow"))
                import matplotlib.pyplot as plt
                lidar_token = item['sample']['data']['LIDAR_TOP']
                
                ds.nusc.render_sample_data(
                    lidar_token,
                    with_anns=False,
                    underlay_map=True,
                    out_path=None,
                    nsweeps=5,
                )
                
                output_path = os.path.join(args.output_dir, f'sample_{args.sample_idx:04d}_bev.jpg')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(colored(f"✓ BEV saved to: {output_path}", "green"))
                
                # Add trajectory overlay to BEV
                print(colored("Adding trajectory overlay to BEV...", "yellow"))
                lidar_data = ds.nusc.get('sample_data', lidar_token)
                ego_pose = ds.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                
                if visualize_trajectory_bev(output_path, gt_waypoints, pred_coords, ego_pose, output_path):
                    print(colored(f"✓ BEV with trajectory overlay saved to: {output_path}", "green"))
            except Exception as e:
                print(colored(f"✗ BEV rendering failed: {e}", "red"))
        
        # 4. All camera views
        if args.show_all_cams:
            try:
                print(colored("Creating multi-camera view...", "yellow"))
                import matplotlib.pyplot as plt
                
                camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
                camera_positions = {
                    'CAM_FRONT_LEFT': (0, 0), 'CAM_FRONT': (0, 1), 'CAM_FRONT_RIGHT': (0, 2),
                    'CAM_BACK_LEFT': (1, 0), 'CAM_BACK': (1, 1), 'CAM_BACK_RIGHT': (1, 2)
                }
                
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                fig.suptitle(f'Sample {args.sample_idx}: All Camera Views', fontsize=16, fontweight='bold')
                
                for cam_channel in camera_channels:
                    if cam_channel in item['sample']['data']:
                        cam_token = item['sample']['data'][cam_channel]
                        cam_data = ds.nusc.get('sample_data', cam_token)
                        img_path = os.path.join(ds.nusc.dataroot, cam_data['filename'])
                        cam_img = Image.open(img_path).convert('RGB')
                        
                        row, col = camera_positions[cam_channel]
                        axes[row, col].imshow(cam_img)
                        axes[row, col].set_title(cam_channel.replace('CAM_', ''), fontsize=12, fontweight='bold')
                        axes[row, col].axis('off')
                
                plt.tight_layout()
                output_path = os.path.join(args.output_dir, f'sample_{args.sample_idx:04d}_all_cams.jpg')
                plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
                plt.close()
                print(colored(f"✓ All cameras view saved to: {output_path}", "green"))
            except Exception as e:
                print(colored(f"✗ Multi-camera rendering failed: {e}", "red"))
    
    print(colored(f"{'='*60}\n", "cyan"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single NuScenes sample")
    parser.add_argument('--sample_idx', type=int, required=True, help='Index of sample to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test', help='NuScenes version')
    parser.add_argument('--llm', type=str, default='Qwen/Qwen3-4B', help='LLM to use')
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/single_sample', help='Directory to save visualization')
    parser.add_argument('--save_vis', action='store_true', default=True, help='Save visualization image')
    parser.add_argument('--show_lidar', action='store_true', default=True, help='Render LiDAR overlay (default: True)')
    parser.add_argument('--no_lidar', dest='show_lidar', action='store_false', help='Disable LiDAR overlay')
    parser.add_argument('--lidar_point_size', type=int, default=5, help='Size of LiDAR points')
    parser.add_argument('--show_bev', action='store_true', default=True, help='Render bird\'s eye view (default: True)')
    parser.add_argument('--no_bev', dest='show_bev', action='store_false', help='Disable bird\'s eye view')
    parser.add_argument('--show_all_cams', action='store_true', default=True, help='Render all 6 camera views (default: True)')
    parser.add_argument('--no_all_cams', dest='show_all_cams', action='store_false', help='Disable all-camera view')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for multi-camera image')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_single_sample(args)
