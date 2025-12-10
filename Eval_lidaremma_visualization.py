import os
import sys

# ==============================================================================
# 1. SETUP & ENV VARIABLES
# ==============================================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import csv
import argparse
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

# Local files
from src.models.lidar_emma import LidarEMMA
from scripts.nuscenes_dataset import NuScenesDataset
from src.utils.lidaremma_utils import collate_fn
from nuscenes.utils.geometry_utils import view_points

# ==============================================================================
# 2. Coordinate & Visualization Logic
# ==============================================================================

def global_to_ego(points_global, ego_pose):
    """Transform points from Global frame to Ego vehicle frame."""
    if len(points_global) == 0:
        return np.array([])

    if points_global.shape[1] == 2:
        points_global = np.hstack([points_global, np.zeros((len(points_global), 1))])

    trans = np.array(ego_pose['translation'])
    rot = Quaternion(ego_pose['rotation'])

    points_centered = points_global - trans
    points_ego = np.dot(points_centered, rot.rotation_matrix)
    
    return points_ego

def get_calib_data(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']
    
    cam_data = nusc.get('sample_data', cam_token)
    cs_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    lidar_data = nusc.get('sample_data', lidar_token)
    cs_lidar = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    return {
        'cam_calib': cs_cam,
        'lidar_calib': cs_lidar,
        'lidar_filename': lidar_data['filename'],
        'ego_pose': ego_pose
    }

def project_ego_to_image(pts_ego_3d, cam_calib):
    """
    Projects (3, N) Ego points to Image.
    Returns (3, N) array: [u, v, valid_depth_mask]
    """
    if pts_ego_3d.shape[1] == 0:
        return np.zeros((3, 0))

    rot_cam = Quaternion(cam_calib['rotation']).rotation_matrix
    trans_cam = np.array(cam_calib['translation']).reshape(3, 1)
    pts_cam = rot_cam.T @ (pts_ego_3d - trans_cam)

    intrinsic = np.array(cam_calib['camera_intrinsic'])
    pts_img = view_points(pts_cam, intrinsic, normalize=True)
    
    valid_depth = pts_cam[2, :] > 0.1
    return np.vstack([pts_img[:2, :], valid_depth])

def visualize_sample_robust(output_dir, token, image_pil, gt_traj_global, pred_traj_global, nusc):
    calib = get_calib_data(nusc, token)
    sample_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(sample_dir, exist_ok=True)
    
    # --- CONVERT GLOBAL -> EGO ---
    gt_traj_ego = global_to_ego(gt_traj_global, calib['ego_pose'])
    pred_traj_ego = global_to_ego(pred_traj_global, calib['ego_pose'])
    
    def prep_traj_for_proj(t):
        if len(t) == 0: return np.zeros((3, 0))
        # Z = -1.6m (Approx road level relative to sensor)
        t[:, 2] = -1.6 
        return t.T 

    gt_3d_proj = prep_traj_for_proj(gt_traj_ego.copy())
    pred_3d_proj = prep_traj_for_proj(pred_traj_ego.copy())
    
    # --------------------------------------------------------------------------
    # 1. CAMERA OVERLAY (PATH / CORRIDOR STYLE)
    # --------------------------------------------------------------------------
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    
    def draw_path_corridor(pts_3d, color):
        """Draws a transparent corridor width ~1.8m plus center dots."""
        if pts_3d.shape[1] < 2: return

        # 1. Create Left and Right edges (Ego Y axis is Left)
        # Width = 1.8m -> +/- 0.9m
        left_3d = pts_3d.copy()
        left_3d[1, :] += 0.9
        right_3d = pts_3d.copy()
        right_3d[1, :] -= 0.9

        # 2. Project Edges
        left_uv = project_ego_to_image(left_3d, calib['cam_calib'])
        right_uv = project_ego_to_image(right_3d, calib['cam_calib'])
        center_uv = project_ego_to_image(pts_3d, calib['cam_calib'])

        # 3. Construct Polygon Points
        poly_pts = []
        
        # Add Left points (Forward)
        for i in range(left_uv.shape[1]):
            # Use points only if both edges are valid to prevent twisted polys at image edges
            if left_uv[2, i] and right_uv[2, i]:
                lx, ly = int(left_uv[0, i]), int(left_uv[1, i])
                if 0 <= lx < w and 0 <= ly < h:
                    poly_pts.append([lx, ly])
        
        # Add Right points (Backward)
        # Need to capture the right edge in reverse order to close the loop
        temp_right = []
        for i in range(right_uv.shape[1]):
            if left_uv[2, i] and right_uv[2, i]:
                rx, ry = int(right_uv[0, i]), int(right_uv[1, i])
                if 0 <= rx < w and 0 <= ry < h:
                    temp_right.append([rx, ry])
        
        poly_pts.extend(temp_right[::-1]) # Reverse right side

        # 4. Draw Filled Polygon (The Path)
        if len(poly_pts) > 2:
            overlay = img_cv.copy()
            pts_arr = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_arr], color)
            # Add Weighted blends the overlay with original image (alpha=0.5)
            cv2.addWeighted(overlay, 0.5, img_cv, 0.5, 0, img_cv)

        # 5. Draw Center Dots (Solid)
        for i in range(center_uv.shape[1]):
            if center_uv[2, i]:
                cx, cy = int(center_uv[0, i]), int(center_uv[1, i])
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(img_cv, (cx, cy), 6, color, -1)

    # Draw GT (Green)
    draw_path_corridor(gt_3d_proj, (0, 255, 0))
    # Draw Pred (Orange)
    draw_path_corridor(pred_3d_proj, (0, 165, 255))

    # Legend
    cv2.putText(img_cv, "GT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_cv, "Pred", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    
    cv2.imwrite(os.path.join(sample_dir, f"{token}_cam.jpg"), img_cv)

    # --------------------------------------------------------------------------
    # 2. BEV PLOT (White Background + Colored Depth)
    # --------------------------------------------------------------------------
    pcl_path = os.path.join(nusc.dataroot, calib['lidar_filename'])
    scan = np.fromfile(pcl_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :3] 
    
    cs_lidar = calib['lidar_calib']
    rot_lidar = Quaternion(cs_lidar['rotation']).rotation_matrix
    trans_lidar = np.array(cs_lidar['translation'])
    points_ego = (rot_lidar @ points.T).T + trans_lidar
    
    # Calculate Distances for coloring
    dists = np.sqrt(points_ego[:, 0]**2 + points_ego[:, 1]**2)
    
    # Setup Figure (White Background)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor('white')
    
    # Subsample for speed
    skip = 5
    pts_sub = points_ego[::skip]
    dists_sub = dists[::skip]
    
    # Plot Lidar
    ax.scatter(-pts_sub[:, 1], pts_sub[:, 0], s=0.5, c=dists_sub, cmap='viridis', alpha=0.5, label='Lidar')
    
    # Plot GT
    if gt_traj_ego.shape[0] > 0:
        ax.plot(-gt_traj_ego[:, 1], gt_traj_ego[:, 0], 'o-', color='green', linewidth=3, markersize=7, label='GT', markeredgecolor='black')

    # Plot Pred
    if pred_traj_ego.shape[0] > 0:
        ax.plot(-pred_traj_ego[:, 1], pred_traj_ego[:, 0], 'o-', color='orange', linewidth=3, markersize=7, label='Pred', markeredgecolor='black')
        
    # Plot Ego
    ax.plot(0, 0, 'r*', markersize=18, label='Ego', markeredgecolor='black')
    
    # Limits
    ax.set_xlim(30, -30)
    ax.set_ylim(-10, 60)
    ax.set_aspect('equal')
    ax.axis('off')
    
    leg = ax.legend(loc='upper right', facecolor='white', edgecolor='black', framealpha=1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, f"{token}_bev.jpg"), bbox_inches='tight', dpi=100)
    plt.close()

# ==============================================================================
# 3. Model & Utils
# ==============================================================================

def load_model(args, device):
    print(colored(f"Loading Model (Ablation: {args.ablation})...", "cyan"))
    model = LidarEMMA(device,
                llm=args.llm,
                freeze_encoders=True,
                freeze_llm=True,
                use_lidar=args.use_lidar,
                lidar_pooling=False)
    
    checkpt = torch.load(args.checkpoint, map_location=device)
    
    def load_key(key, target_module):
        if key in checkpt and hasattr(model, target_module):
            getattr(model, target_module).load_state_dict(checkpt[key])
    
    load_key('vision_projector_state_dict', 'vision_projector')
    load_key('lidar_projector_state_dict', 'lidar_projector')
    load_key('vision_encoder_state_dict', 'vision_tower')
    load_key('lidar_encoder_state_dict', 'lidar_encoder')
    load_key('llm_state_dict', 'language_model')
        
    model.eval()
    return model

def parse_coords_from_text(text):
    trajectory_match = re.search(r'Future Trajectory:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    text_to_parse = trajectory_match.group(1) if trajectory_match else text
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text_to_parse)
    nums = [float(x) for x in nums]
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)

# ==============================================================================
# 4. Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate LIDAR-EMMA Dark Scenes")
    parser.add_argument("--ablation", type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dark_csv', type=str, default='eval_outputs/dark_scenes.csv')
    parser.add_argument('--dark_only', type=str, default=None, choices=['40', '60', '80'])
    
    args = parser.parse_args()
    args.use_lidar = "lidar" in args.ablation
    args.run_name = Path(args.checkpoint).parent.name
    args.llm = "Qwen/Qwen2.5-3B" if args.ablation in ("1a", "2a", "3a", "1a-lidar", "2a-lidar", "3a-lidar") else "Qwen/Qwen2.5-3B-Instruct"

    suffix = f"_dark{args.dark_only}" if args.dark_only else ""
    args.output_dir = f"./eval_outputs/{args.run_name}_vis{suffix}"

    dark_mask = None
    if args.dark_only:
        if not os.path.exists(args.dark_csv):
            args.dark_csv = os.path.join(os.path.dirname(args.output_dir), 'dark_scenes.csv')
            if not os.path.exists(args.dark_csv):
                print(colored("Error: dark_scenes.csv not found.", "red"))
                return
        
        col_name = f"<{args.dark_only}"
        dark_mask = []
        with open(args.dark_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dark_mask.append(row[col_name].strip() == 'True')
        print(colored(f"Filtered: {sum(dark_mask)} samples.", "green"))

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False) 
        torch.backends.cuda.enable_math_sdp(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    
    print("\nLoading dataset...")
    ds = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2,
        output_lidar=True 
    )

    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=8,
        collate_fn=lambda batch: collate_fn(batch),
        drop_last=False
    )

    try:
        dataset_list = ds.data
    except:
        dataset_list = ds.samples

    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = open(os.path.join(args.output_dir, 'results.csv'), 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['token', 'ade', 'fde', 'gen_text'])
    writer.writeheader()

    print(colored(f"Visualizations -> {args.output_dir}/visualizations", "magenta"))

    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_size = len(batch['images'])
            
            if dark_mask is not None:
                indices = range(global_idx, global_idx + batch_size)
                batch_mask = [dark_mask[i] for i in indices if i < len(dark_mask)]
                batch_mask += [False] * (batch_size - len(batch_mask))
                if not any(batch_mask):
                    global_idx += batch_size
                    continue
            else:
                batch_mask = [True] * batch_size

            pixel_values = model.image_processor(images=batch['images'], return_tensors="pt").pixel_values.to(device)
            point_clouds = None
            if args.use_lidar and batch.get("lidar") is not None:
                point_clouds = [pc.to(device) if isinstance(pc, torch.Tensor) else pc for pc in batch["lidar"]]

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                gen_texts = model.generate_trajectory(
                    prompt=batch['prompt'],
                    images=pixel_values,
                    point_clouds=point_clouds,
                )

            gt_waypoints = batch['waypoints']

            for i, gen_text in enumerate(gen_texts):
                if not batch_mask[i]: continue

                pred_coords = parse_coords_from_text(gen_text)
                if len(pred_coords) < 10:
                    pad = np.full((10 - len(pred_coords), 2), np.nan)
                    pred_metrics = np.vstack([pred_coords, pad]) if len(pred_coords) > 0 else pad
                else:
                    pred_metrics = pred_coords[:10]
                
                gt_wp = gt_waypoints[i]
                diffs = pred_metrics - gt_wp
                l2 = np.linalg.norm(diffs, axis=1)
                ade = np.nanmean(l2)
                fde = l2[-1] if len(l2) >= 10 else np.nan

                idx = global_idx + i
                if dataset_list and idx < len(dataset_list):
                    entry = dataset_list[idx]
                    token = entry.get('token') if isinstance(entry, dict) else entry
                else:
                    continue

                visualize_sample_robust(
                    output_dir=args.output_dir,
                    token=token,
                    image_pil=batch['images'][i],
                    gt_traj_global=gt_wp,
                    pred_traj_global=pred_metrics,
                    nusc=ds.nusc
                )

                writer.writerow({'token': token, 'ade': ade, 'fde': fde, 'gen_text': gen_text})
                csv_file.flush()

            global_idx += batch_size

    csv_file.close()
    print(colored("Done!", "green"))

if __name__ == "__main__":
    main()