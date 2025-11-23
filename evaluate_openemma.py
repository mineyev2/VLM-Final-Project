import argparse
import os
import re
import csv
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
from termcolor import colored
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# Import OpenEMMA dependencies
from src.openemma.vlm.base_backbone import BaseOpenEMMA

# Import utilities from your project
from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D

def parse_coords_from_text(text, max_points=10):
    """Extract waypoint coordinates from OpenEMMA generated text."""
    text = text.replace(";", " ") 
    # find all floats/ints
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    nums = [float(x) for x in nums]
    
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        if len(pairs) >= max_points:
            break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)

def visualize_trajectories(image_path, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, idx, output_dir):
    """Visualization helper."""
    img = cv2.imread(image_path)
    
    # Convert 2D to 3D (z=0)
    gt_waypoints_3d = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    try:
        # Draw GT (Green)
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=5, color=(0, 255, 0), thickness=-1)
            
        # Draw Pred (Blue)
        if len(pred_waypoints_3d) > 0:
            pred_points_img = ProjectWorldToImage(pred_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
            for pt in pred_points_img:
                cv2.circle(img, tuple(pt.astype(int)), radius=5, color=(255, 0, 0), thickness=-1)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'vis_emma_{idx:04d}.jpg')
        cv2.imwrite(output_path, img)
    except Exception as e:
        print(f"Vis failed: {e}")

class EvalNuScenesOpenEMMA:
    def __init__(self, version, dataroot):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.sample_tokens = []
        
        # Filter samples
        for scene in self.nusc.scene:
            nbr_samples = scene['nbr_samples']
            if nbr_samples < 13: continue
            sample = self.nusc.get('sample', scene['first_sample_token'])
            for _ in range(2): sample = self.nusc.get('sample', sample['next'])
            for _ in range(nbr_samples - 12):
                self.sample_tokens.append(sample['token'])
                sample = self.nusc.get('sample', sample['next'])

    def __len__(self):
        return len(self.sample_tokens)

    def get_item(self, idx):
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        # 1. Ego History (Normalized/Local Frame)
        ego_positions = [] 
        curr = sample
        # Use 10 points (5 seconds) history per OpenEMMA paper
        for _ in range(10):
            cam_data = self.nusc.get('sample_data', curr['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_positions.append([float(ego_pose['translation'][0]), float(ego_pose['translation'][1])])
            if curr['prev']: 
                curr = self.nusc.get('sample', curr['prev'])
            else: 
                ego_positions.append(ego_positions[-1]) 
        ego_positions.reverse()

        # 2. Image Path
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])

        # 3. Calibration
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        cam_to_ego = {'translation': cam_calib['translation'], 'rotation': cam_calib['rotation'], 'camera_intrinsic': np.array(cam_calib['camera_intrinsic'])}
        ego_to_world = {'translation': ego_pose_curr['translation'], 'rotation': ego_pose_curr['rotation']}

        # 4. Future Waypoints (GT)
        waypoints = []
        curr = sample
        for _ in range(10):
            if curr['next'] == '': break 
            curr = self.nusc.get('sample', curr['next'])
            cam_data_next = self.nusc.get('sample_data', curr['data']['CAM_FRONT'])
            pose_next = self.nusc.get('ego_pose', cam_data_next['ego_pose_token'])['translation']
            waypoints.append([float(pose_next[0]), float(pose_next[1])])
        
        while len(waypoints) < 10:
            waypoints.append(waypoints[-1] if len(waypoints)>0 else [0,0])

        # 5. Compute Relative Trajectories (Normalization)
        ego_trans = np.array(ego_pose_curr['translation'])
        ego_rot = Quaternion(ego_pose_curr['rotation'])
        
        # Convert History to Local (Ego) Frame -> Current Position becomes (0,0)
        his_trajs_local = []
        for p in ego_positions:
            global_p = np.array(p + [0]) 
            local_p = ego_rot.inverse.rotate(global_p - ego_trans)
            his_trajs_local.append([local_p[0], local_p[1]])
            
        # Convert Future to Local (Ego) Frame
        fut_trajs_local = []
        for p in waypoints:
            global_p = np.array(p + [0])
            local_p = ego_rot.inverse.rotate(global_p - ego_trans)
            fut_trajs_local.append([local_p[0], local_p[1]])

        his_diff = np.diff(np.array(his_trajs_local), axis=0)
        fut_diff = np.diff(np.array(fut_trajs_local), axis=0)

        return {
            'image_path': image_path,
            'gt_waypoints_world': np.array(waypoints),
            'gt_ego_his_trajs': his_trajs_local, 
            'gt_ego_fut_trajs': fut_trajs_local, 
            'gt_ego_his_diff': his_diff,
            'gt_ego_fut_diff': fut_diff,
            'cam_to_ego': cam_to_ego,
            'ego_to_world': ego_to_world,
            'ego_positions_world': ego_positions
        }

def evaluate(args):
    print(colored(f"Initializing OpenEMMA with model: {args.model_id}", "cyan"))
    emma_model = BaseOpenEMMA(args)
    
    ds = EvalNuScenesOpenEMMA(args.version, args.dataroot)
    results = []
    
    n_samples = len(ds) if args.num_samples == -1 else min(args.num_samples, len(ds))
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    
    print(colored(f"Starting evaluation on {n_samples} samples...", "green"))

    for idx in tqdm(range(n_samples)):
        item = ds.get_item(idx)
        
        # 1. Compute Oracle Command
        try:
            command = emma_model.compute_command(item['gt_ego_fut_trajs'])
        except Exception as e:
            command = "MOVE FORWARD"

        emma_data = {
            "gt_ego_fut_diff": item['gt_ego_fut_diff'],
            "gt_ego_fut_trajs": item['gt_ego_fut_trajs'],
            "gt_ego_his_diff": item['gt_ego_his_diff'],
            "gt_ego_his_trajs": item['gt_ego_his_trajs']
        }

        # 2. Generate Waypoints (Returns Local/Normalized Trajectory)
        try:
            response_text = emma_model.generate_waypoints(
                command=command,
                image_path=item['image_path'],
                data=emma_data,
                backbone=None,
                args=args
            )
            
            # Initial Parsing of Local Coords
            pred_coords_local = parse_coords_from_text(response_text)
            
            # Pad or Truncate
            num_valid_waypoints = pred_coords_local.shape[0]
            format_compliant = 1 if num_valid_waypoints == 10 else 0
            
            if num_valid_waypoints == 0:
                pred_coords_local = np.zeros((10, 2))
            elif num_valid_waypoints < 10:
                pad = np.tile(pred_coords_local[-1], (10 - num_valid_waypoints, 1))
                pred_coords_local = np.vstack([pred_coords_local, pad])
            elif num_valid_waypoints > 10:
                pred_coords_local = pred_coords_local[:10]

            # 3. Metrics (in Local Frame)
            gt_coords_local = np.array(item['gt_ego_fut_trajs'])
            diffs = pred_coords_local - gt_coords_local
            l2_dists = np.linalg.norm(diffs, axis=1)
            
            ade = np.mean(l2_dists)
            fde = l2_dists[-1]
            miss_rate_10m = 1.0 if fde > 10.0 else 0.0
            waypoint_errors = l2_dists.tolist()
            
            # 4. Denormalize to World Frame (for CSV/Logging)
            e2w_t = np.array(item['ego_to_world']['translation'])
            e2w_r = Quaternion(item['ego_to_world']['rotation'])
            
            pred_coords_world = []
            for p in pred_coords_local:
                p3 = np.array([p[0], p[1], 0.0])
                w3 = e2w_r.rotate(p3) + e2w_t
                pred_coords_world.append([w3[0], w3[1]])
            pred_coords_world = np.array(pred_coords_world)

            # --- Console Output (Logging) ---
            tqdm.write(colored(f"\n--- Sample {idx} ---", "yellow"))
            tqdm.write(f"Command: {command}")
            tqdm.write(f"Historical Ego Poses (World): {item['ego_positions_world']}")
            tqdm.write(f"GT Waypoints (Local): {gt_coords_local.tolist()}")
            tqdm.write(f"Pred Waypoints (Local): {pred_coords_local.tolist()}")
            tqdm.write(f"Pred Waypoints (World - Denormalized): {pred_coords_world.tolist()}")

            results.append({
                'idx': idx,
                'num_valid_waypoints': int(num_valid_waypoints),
                'format_compliant': int(format_compliant),
                'ade': float(ade),
                'fde': float(fde),
                'miss_rate_10m': float(miss_rate_10m),
                'waypoint_errors': waypoint_errors,
                'command': command,
                'gen_text': response_text,
                # Save Denormalized (World) coords to CSV for external visualization
                'pred_waypoints_world': json.dumps(pred_coords_world.tolist()) 
            })
            
            # Visualization uses World Coords (already handled by pred_coords_world logic for saving)
            if idx < args.num_vis:
                visualize_trajectories(
                    item['image_path'],
                    item['gt_waypoints_world'],
                    pred_coords_world,
                    item['cam_to_ego'],
                    item['ego_to_world'],
                    idx,
                    vis_dir
                )

        except Exception as e:
            print(colored(f"Sample {idx} failed: {e}", "red"))
            continue

    if not results:
        print("No results generated.")
        return

    # --- Aggregation ---
    ades = [r['ade'] for r in results]
    fdes = [r['fde'] for r in results]
    miss_rates = [r['miss_rate_10m'] for r in results]
    format_compliance = [r['format_compliant'] for r in results]
    all_waypoint_errors = [r['waypoint_errors'] for r in results]
    waypoint_means = np.mean(all_waypoint_errors, axis=0).tolist() if len(all_waypoint_errors) > 0 else []

    summary = {
        'num_samples': len(results),
        'ade_mean': float(np.mean(ades)) if len(ades) > 0 else float('nan'),
        'ade_std': float(np.std(ades)) if len(ades) > 0 else float('nan'),
        'fde_mean': float(np.mean(fdes)) if len(fdes) > 0 else float('nan'),
        'fde_std': float(np.std(fdes)) if len(fdes) > 0 else float('nan'),
        'miss_rate_10m': float(np.mean(miss_rates)) if len(miss_rates) > 0 else float('nan'),
        'format_compliance_rate': float(np.mean(format_compliance)) if len(format_compliance) > 0 else float('nan'),
        'per_waypoint_errors_mean': waypoint_means
    }

    # --- Print Summary ---
    print(colored('\n=== OpenEMMA Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(colored(f"Samples evaluated: {summary['num_samples']}", 'white'))
    print(colored('\nTrajectory Metrics:', 'yellow'))
    print(f"  Average Displacement Error (mean): {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}")
    print(f"  Final Displacement Error (mean): {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}")
    print(f"  Miss Rate @ 10m: {summary['miss_rate_10m']:.2%}")
    print(f"  Format Compliance: {summary['format_compliance_rate']:.2%}")

    # --- Save Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    csv_path = os.path.join(args.output_dir, args.output_name)
    with open(csv_path, 'w', newline='') as f:
        # Saving Denormalized World Waypoints
        fieldnames = ['idx', 'num_valid_waypoints', 'format_compliant', 'ade', 'fde', 'miss_rate_10m'] + \
                     [f'wp{i}_error' for i in range(10)] + \
                     ['command', 'pred_waypoints_world', 'gen_text']
                     
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row = {
                'idx': r['idx'],
                'num_valid_waypoints': r['num_valid_waypoints'],
                'format_compliant': r['format_compliant'],
                'ade': r['ade'],
                'fde': r['fde'],
                'miss_rate_10m': r['miss_rate_10m'],
                'command': r['command'],
                'pred_waypoints_world': r['pred_waypoints_world'], 
                'gen_text': r['gen_text']
            }
            for i, err in enumerate(r['waypoint_errors']):
                row[f'wp{i}_error'] = err
            writer.writerow(row)
            
    print(colored(f"\nResults saved to: {csv_path}", 'green'))
    
    summary_path = os.path.join(args.output_dir, args.output_name.replace('.csv', '_summary.json'))
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(colored(f"Summary saved to: {summary_path}", 'green'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3/')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', 
                        help='Use llava-hf/llava-v1.6-mistral-7b-hf for better transformer support')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_vis', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/openemma_qwen')
    parser.add_argument('--output_name', type=str, default='results.csv')
    
    args = parser.parse_args()
    evaluate(args)
    