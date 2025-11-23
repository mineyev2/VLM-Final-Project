import argparse
import os
import re
import csv
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

        # 1. Ego History
        ego_positions = [] 
        curr = sample
        for _ in range(3):
            cam_data = self.nusc.get('sample_data', curr['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_positions.append([float(ego_pose['translation'][0]), float(ego_pose['translation'][1])])
            if curr['prev']: curr = self.nusc.get('sample', curr['prev'])
            else: ego_positions.append(ego_positions[-1]) 
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

        # 5. Compute Relative Trajectories for OpenEMMA logic
        ego_trans = np.array(ego_pose_curr['translation'])
        ego_rot = Quaternion(ego_pose_curr['rotation'])
        
        his_trajs_local = []
        for p in ego_positions:
            global_p = np.array(p + [0]) 
            local_p = ego_rot.inverse.rotate(global_p - ego_trans)
            his_trajs_local.append([local_p[0], local_p[1]])
            
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
            'ego_to_world': ego_to_world
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

        # 2. Generate Waypoints
        try:
            response_text = emma_model.generate_waypoints(
                command=command,
                image_path=item['image_path'],
                data=emma_data,
                backbone=None,
                args=args
            )
            
            pred_coords_local = parse_coords_from_text(response_text)
            
            if len(pred_coords_local) == 0:
                pred_coords_local = np.zeros((10, 2))
            elif len(pred_coords_local) < 10:
                pad = np.tile(pred_coords_local[-1], (10 - len(pred_coords_local), 1))
                pred_coords_local = np.vstack([pred_coords_local, pad])
            elif len(pred_coords_local) > 10:
                pred_coords_local = pred_coords_local[:10]

            # 3. Metrics (in Local Frame)
            gt_coords_local = np.array(item['gt_ego_fut_trajs'])
            diffs = pred_coords_local - gt_coords_local
            l2_dists = np.linalg.norm(diffs, axis=1)
            ade = np.mean(l2_dists)
            fde = l2_dists[-1]
            
            results.append({
                'idx': idx,
                'ade': ade,
                'fde': fde,
                'command': command,
                'gen_text': response_text
            })
            
            # Visualization (Convert local pred back to world)
            if idx < args.num_vis:
                e2w_t = np.array(item['ego_to_world']['translation'])
                e2w_r = Quaternion(item['ego_to_world']['rotation'])
                
                pred_world = []
                for p in pred_coords_local:
                    p3 = np.array([p[0], p[1], 0.0])
                    w3 = e2w_r.rotate(p3) + e2w_t
                    pred_world.append([w3[0], w3[1]])
                
                visualize_trajectories(
                    item['image_path'],
                    item['gt_waypoints_world'],
                    np.array(pred_world),
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

    ades = [r['ade'] for r in results]
    fdes = [r['fde'] for r in results]
    
    print(colored('\n=== OpenEMMA Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(f"Samples: {len(results)}")
    print(f"ADE: {np.mean(ades):.4f}")
    print(f"FDE: {np.mean(fdes):.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_name)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'ade', 'fde', 'command', 'gen_text'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3/')
    parser.add_argument('--version', type=str, default='v1.0-test')
    # Default updated to llava-v1.6-mistral-7b-hf
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf', 
                        help='Use llava-hf/llava-v1.6-mistral-7b-hf for better transformer support')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_vis', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/openemma_llava')
    parser.add_argument('--output_name', type=str, default='results.csv')
    
    args = parser.parse_args()
    evaluate(args)