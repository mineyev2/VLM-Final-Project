import argparse
import os
import re
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from termcolor import colored

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

def visualize_trajectory_bev(bev_image_path, gt_waypoints, pred_waypoints, ego_pose, output_path):
    """Add trajectory overlay to bird's eye view image.
    
    BEV is rendered in ego vehicle frame (use_flat_vehicle_coordinates=True by default),
    so we need to rotate waypoints from world frame to ego frame.
    """
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
        
        # NuScenes render_sample_data uses default axes_limit=40 meters
        bev_range = 40  # meters from center (default in render_sample_data)
        pixels_per_meter = width / (2 * bev_range)
        
        # Center pixel coordinates
        center_x = width / 2
        center_y = height / 2
        
        # Convert world coordinates to ego frame, then to pixel coordinates
        # BEV is rendered in ego vehicle frame (use_flat_vehicle_coordinates=True by default)
        def world_to_pixel(waypoints, ego_translation, ego_rotation):
            # Translate to ego-relative coordinates
            rel_x = waypoints[:, 0] - ego_translation[0]
            rel_y = waypoints[:, 1] - ego_translation[1]
            
            # Stack into 2D points for rotation
            rel_points = np.vstack([rel_x, rel_y])
            
            # Rotate from world frame to ego frame
            # BEV uses flat ego frame (only yaw rotation)
            ego_quat = Quaternion(ego_rotation)
            yaw = ego_quat.yaw_pitch_roll[0]
            
            # Create rotation matrix for yaw only (flat ego frame)
            cos_yaw = np.cos(-yaw)  # Negative yaw to rotate world to ego
            sin_yaw = np.sin(-yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            
            # Apply rotation
            ego_points = rotation_matrix @ rel_points
            
            # Convert to pixel coordinates
            pixel_x = center_x + ego_points[0, :] * pixels_per_meter
            pixel_y = center_y - ego_points[1, :] * pixels_per_meter  # Flipped y-axis
            
            return pixel_x, pixel_y
        
        ego_translation = ego_pose['translation']
        ego_rotation = ego_pose['rotation']
        
        # Plot ground truth trajectory (green)
        if len(gt_waypoints) > 0:
            gt_px, gt_py = world_to_pixel(gt_waypoints, ego_translation, ego_rotation)
            ax.plot(gt_px, gt_py, 'o-', color='lime', linewidth=3, markersize=8, label='Ground Truth')
        
        # Plot predicted trajectory (orange)
        valid_pred = pred_waypoints[~np.isnan(pred_waypoints).any(axis=1)]
        if len(valid_pred) > 0:
            pred_px, pred_py = world_to_pixel(valid_pred, ego_translation, ego_rotation)
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
        
        return True
    except Exception as e:
        print(colored(f"✗ Failed to add trajectory to BEV: {e}", "red"))
        return False