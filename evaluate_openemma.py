import argparse
import os
import re
import csv
import json
import numpy as np
from tqdm import tqdm
from termcolor import colored
from nuscenes import NuScenes
from pyquaternion import Quaternion

# Import OpenEMMA dependencies
from src.openemma.vlm.base_backbone import BaseOpenEMMA

def parse_coords_from_text(text, max_points=10):
    """Extract waypoint coordinates from OpenEMMA generated text."""
    # Attempt to clean text
    text = text.replace(";", " ").replace("[", " ").replace("]", " ").replace(",", " ")
    # find all floats/ints
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    nums = [float(x) for x in nums]
    
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        if len(pairs) >= max_points:
            break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)

class EvalNuScenesOpenEMMA:
    def __init__(self, version, dataroot):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.sample_tokens = []
        
        # Filter samples
        for scene in self.nusc.scene:
            nbr_samples = scene['nbr_samples']
            if nbr_samples < 20: continue
            sample = self.nusc.get('sample', scene['first_sample_token'])
            for _ in range(9): sample = self.nusc.get('sample', sample['next'])
            for _ in range(nbr_samples - 19):
                self.sample_tokens.append(sample['token'])
                sample = self.nusc.get('sample', sample['next'])

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        # Ego History (Normalized/Local Frame)
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

        # Image Path
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])

        # Calibration
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        cam_to_ego = {'translation': cam_calib['translation'], 'rotation': cam_calib['rotation'], 'camera_intrinsic': np.array(cam_calib['camera_intrinsic'])}
        ego_to_world = {'translation': ego_pose_curr['translation'], 'rotation': ego_pose_curr['rotation']}

        # Future Waypoints (GT)
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

        # Compute Relative Trajectories (Normalization)
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
            'idx': idx, # Pass index through
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

def collate_fn(batch):
    """Custom collate to handle dictionary batching without converting everything to tensors."""
    return batch # Return list of dicts directly to iterate easily

def compute_summary(results_list, name=""):
    """Helper function to compute summary statistics matching evaluate_lidaremma.py"""
    successful = [r for r in results_list if not r['failure_rate']]
    
    ades_local = [r['ade'] for r in successful if not np.isnan(r['ade'])]
    fdes_local = [r['fde'] for r in successful if not np.isnan(r['fde'])]
    
    wp1_local = [r['wp1_error'] for r in successful if not np.isnan(r['wp1_error'])]
    wp3_local = [r['wp3_error'] for r in successful if not np.isnan(r['wp3_error'])]
    wp5_local = [r['wp5_error'] for r in successful if not np.isnan(r['wp5_error'])]
    
    format_compliant_local = sum(1 for r in results_list if r['format_compliant'])
    format_compliance_local = format_compliant_local / len(results_list) if len(results_list) > 0 else 0.0
    
    failure_count_local = sum(1 for r in results_list if r['failure_rate'])
    failure_rate_local = failure_count_local / len(results_list) if len(results_list) > 0 else 0.0
    
    return {
        'category': name,
        'total_samples': len(results_list),
        'successful_samples': len(successful),
        'failed_samples': failure_count_local,
        'failure_rate': float(failure_rate_local),
        'format_compliance_rate': float(format_compliance_local),
        'ade_mean': float(np.mean(ades_local)) if len(ades_local) > 0 else float('nan'),
        'ade_std': float(np.std(ades_local)) if len(ades_local) > 0 else float('nan'),
        'fde_mean': float(np.mean(fdes_local)) if len(fdes_local) > 0 else float('nan'),
        'fde_std': float(np.std(fdes_local)) if len(fdes_local) > 0 else float('nan'),
        'error_at_1s_mean': float(np.mean(wp1_local)) if len(wp1_local) > 0 else float('nan'),
        'error_at_1s_std': float(np.std(wp1_local)) if len(wp1_local) > 0 else float('nan'),
        'error_at_2s_mean': float(np.mean(wp3_local)) if len(wp3_local) > 0 else float('nan'),
        'error_at_2s_std': float(np.std(wp3_local)) if len(wp3_local) > 0 else float('nan'),
        'error_at_3s_mean': float(np.mean(wp5_local)) if len(wp5_local) > 0 else float('nan'),
        'error_at_3s_std': float(np.std(wp5_local)) if len(wp5_local) > 0 else float('nan'),
    }

def evaluate(args):
    print(colored(f"Initializing OpenEMMA with model: {args.model_id}", "cyan"))
    emma_model = BaseOpenEMMA(args)
    
    ds = EvalNuScenesOpenEMMA(args.version, args.dataroot)
    
    # Subset dataset if num_samples is specified
    if args.num_samples != -1:
        ds.sample_tokens = ds.sample_tokens[:args.num_samples]
        print(f"Subsetting dataset to {len(ds)} samples.")
    
    n_samples = len(ds)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dark scenes information
    dark_scenes_path = os.path.join(args.output_dir, '../dark_scenes.csv')
    dark_scenes_data = {'<40': [], '<60': [], '<80': []}
    
    if os.path.exists(dark_scenes_path):
        with open(dark_scenes_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dark_scenes_data['<40'].append(row['<40'] == 'True')
                dark_scenes_data['<60'].append(row['<60'] == 'True')
                dark_scenes_data['<80'].append(row['<80'] == 'True')
        print(colored(f"Loaded dark scenes data from {dark_scenes_path}", "green"))
    else:
        print(colored(f"Warning: {dark_scenes_path} not found. Dark scene filtering disabled.", "yellow"))

    csv_path = os.path.join(args.output_dir, args.output_name)
    csv_file = open(csv_path, 'w', newline='')
    fieldnames = ['idx', 'num_valid_waypoints', 'format_compliant', 'ade', 'fde',
                 'failure_rate', 'error_at_1s'] + \
                 [f'wp{i}_error' for i in range(10)] + \
                 ['history_trajectory', 'gt_trajectory', 'pred_trajectory', 'gen_text', 'command']
    
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Open separate CSV files for dark scenes
    csv_files_dark = {}
    writers_dark = {}
    for threshold in ['40', '60', '80']:
        csv_path_dark = os.path.join(args.output_dir, f'eval_results_dark{threshold}.csv')
        csv_files_dark[threshold] = open(csv_path_dark, 'w', newline='')
        writers_dark[threshold] = csv.DictWriter(csv_files_dark[threshold], fieldnames=fieldnames)
        writers_dark[threshold].writeheader()
        
    results = []
    results_dark = {'40': [], '60': [], '80': []}
    
    print(colored(f"Starting evaluation on {n_samples} samples...", "green"))

    # Iterate over samples
    for idx in tqdm(range(n_samples), desc="Evaluating"):
        item = ds[idx]
        
        # Compute Oracle Command
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

        # Generate Waypoints (Returns Local/Normalized Trajectory)
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
                pred_coords_local = np.full((10, 2), np.nan)
            elif num_valid_waypoints < 10:
                pad = np.full((10 - num_valid_waypoints, 2), np.nan)
                pred_coords_local = np.vstack([pred_coords_local, pad])
            elif num_valid_waypoints > 10:
                pred_coords_local = pred_coords_local[:10]

            # Metrics (in Local Frame)
            gt_coords_local = np.array(item['gt_ego_fut_trajs'])
            diffs = pred_coords_local - gt_coords_local
            l2_per_waypoint = np.linalg.norm(diffs, axis=1)
            
            # ADE (Average Displacement Error)
            ade = np.nanmean(l2_per_waypoint)
            
            # FDE (Final Displacement Error)
            fde = l2_per_waypoint[-1] if len(l2_per_waypoint) >= 10 else np.nan

            error_at_1s = l2_per_waypoint[1] if len(l2_per_waypoint) > 1 else np.nan
            any_severe_error = np.any(l2_per_waypoint > 100.0)
            has_nans = np.any(np.isnan(l2_per_waypoint))
            
            # Failure Rate Definition: Error@1s > 10m OR Any point > 100m error OR NaNs present
            failure_rate = True if (error_at_1s > 10.0 or any_severe_error or has_nans) else False

            # Construct Result Dict
            result = {
                'idx': idx,
                'num_valid_waypoints': int(num_valid_waypoints),
                'format_compliant': int(format_compliant),
                'ade': float(ade),
                'fde': float(fde),
                'failure_rate': failure_rate,
                'error_at_1s': float(error_at_1s),
                'command': command,
                'gen_text': response_text,
                'history_trajectory': str(item['gt_ego_his_trajs']),
                'gt_trajectory': str(gt_coords_local.tolist()),
                'pred_trajectory': str(pred_coords_local.tolist())
            }
            
            # Add per-waypoint errors
            for wp_idx in range(10):
                result[f'wp{wp_idx}_error'] = l2_per_waypoint[wp_idx] if wp_idx < len(l2_per_waypoint) else np.nan

            results.append(result)

            # Write to CSV
            writer.writerow(result)
            csv_file.flush()

            # Write to Dark Scene CSVs
            if len(dark_scenes_data['<40']) > idx:
                if dark_scenes_data['<40'][idx]:
                    results_dark['40'].append(result)
                    writers_dark['40'].writerow(result)
                    csv_files_dark['40'].flush()
                if dark_scenes_data['<60'][idx]:
                    results_dark['60'].append(result)
                    writers_dark['60'].writerow(result)
                    csv_files_dark['60'].flush()
                if dark_scenes_data['<80'][idx]:
                    results_dark['80'].append(result)
                    writers_dark['80'].writerow(result)
                    csv_files_dark['80'].flush()
            
        except Exception as e:
            print(colored(f"Sample {idx} failed: {e}", "red"))
            continue

    if not results:
        print("No results generated.")
        return

    # Aggregation and Summaries
    summary = compute_summary(results, "all_samples")
    
    # Save Main Summary
    summary_path = os.path.join(args.output_dir, args.output_name.replace('.csv', '_summary.json'))
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # Save Dark Scene Summaries
    for threshold in ['40', '60', '80']:
        if len(results_dark[threshold]) > 0:
            summary_dark = compute_summary(results_dark[threshold], f"dark<{threshold}")
            summary_dark_path = os.path.join(args.output_dir, f'results_dark{threshold}_summary.json')
            with open(summary_dark_path, 'w') as f:
                json.dump(summary_dark, f, indent=4)
            print(colored(f"Dark{threshold} summary saved to {summary_dark_path}", "green"))

    # Close Files
    csv_file.close()
    for f in csv_files_dark.values():
        f.close()

    # Print Summary
    print(colored('\n=== OpenEMMA Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Successful Samples: {summary['successful_samples']}")
    print(f"Failed Samples: {summary['failed_samples']}")
    print(f"Failure Rate: {summary['failure_rate']:.2%}")
    print(f"Format Compliance: {summary['format_compliance_rate']:.2%}")
    print(f"\nADE: {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f} m")
    print(f"FDE: {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f} m")
    print(f"\nError @ 1s: {summary['error_at_1s_mean']:.4f} ± {summary['error_at_1s_std']:.4f} m")
    print(colored(f"\nResults saved to: {csv_path}", 'green'))
    print(colored(f"Summary saved to: {summary_path}", 'green'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3/')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/openemma')
    parser.add_argument('--output_name', type=str, default='eval_results.csv')
    parser.add_argument('--api_key', type=str, default=None, help='For GPT models')
    
    args = parser.parse_args()
    evaluate(args)
