import argparse
import os
import re
import csv
import torch
import numpy as np
from tqdm import tqdm

from src.models.qwen_clip_model import QwenCLIPModel
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image


def parse_coords_from_text(text, max_points=10):
    # find all floats/ints in text and group into pairs
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    nums = [float(x) for x in nums]
    # group into pairs
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        if len(pairs) >= max_points:
            break
    return np.array(pairs, dtype=float)


class EvalNuScenes:
    def __init__(self, version, dataroot, prompt_part1, prompt_part2, nsweeps=5):
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        self.nsweeps = nsweeps
        self.prompt_part1 = prompt_part1
        self.prompt_part2 = prompt_part2
        # build sample list similar to dataset
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

        # ego positions (last 3)
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

        # image
        camera_token = sample['data']['CAM_FRONT']
        camera_data = self.nusc.get('sample_data', camera_token)
        image_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        image = Image.open(image_path).convert('RGB')

        # ground truth 10 waypoints
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
            'waypoints': np.array(waypoints, dtype=float)
        }


def load_checkpoint_into_model(model, ckpt_path, device):
    data = torch.load(ckpt_path, map_location=device)
    # If it's a raw state dict for projector (train saved that as checkpoint_latest.pth)
    if all(isinstance(k, str) for k in data.keys()) and any('weight' in k or 'bias' in k for k in data.keys()):
        try:
            model.mlp_projector.load_state_dict(data)
            print(f"Loaded projector state_dict from {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load projector state_dict: {e}")
        return

    # If it's the 'final_model.pth' dict saved by train_v3
    if isinstance(data, dict) and ('model_state_dict' in data or 'vision_tower_state_dict' in data or 'language_model_state_dict' in data):
        # projector
        if 'model_state_dict' in data:
            try:
                model.mlp_projector.load_state_dict(data['model_state_dict'])
                print("Loaded mlp_projector from final checkpoint.")
            except Exception as e:
                print(f"Warning loading mlp_projector: {e}")
        # vision tower
        if 'vision_tower_state_dict' in data:
            try:
                model.vision_tower.load_state_dict(data['vision_tower_state_dict'])
                print("Loaded vision_tower weights.")
            except Exception as e:
                print(f"Warning loading vision_tower: {e}")
        # language model
        if 'language_model_state_dict' in data:
            try:
                # try non-strict to avoid mismatch
                model.language_model.load_state_dict(data['language_model_state_dict'], strict=False)
                print("Loaded (partial/soft) language_model weights.")
            except Exception as e:
                print(f"Warning loading language_model: {e}")
        return

    print("Unrecognized checkpoint format, attempting to load as state_dict into projector...")
    try:
        model.mlp_projector.load_state_dict(data)
        print("Loaded projector state_dict (fallback).")
    except Exception as e:
        print(f"Failed fallback loading: {e}")


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QwenCLIPModel(device)

    # Load checkpoint
    if args.checkpoint is not None:
        load_checkpoint_into_model(model, args.checkpoint, device)

    model.eval()

    # dataset for evaluation
    ds = EvalNuScenes(args.version, args.dataroot, model.prompt_part1, model.prompt_part2)

    results = []

    n_samples = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))

    for idx in tqdm(range(n_samples), desc='Evaluating'):
        item = ds.get_item(idx)
        image = item['image']
        ego_positions = item['ego_positions']
        gt_waypoints = item['waypoints']

        # prepare image tensor
        pixel_values = model.image_processor(images=[image], return_tensors='pt').pixel_values.to(device)

        # convert ego_positions to Python float lists for compatibility with generate_trajectory
        ego_positions_py = [[float(x), float(y)] for (x, y) in ego_positions]

        # generate
        try:
            outputs, gen_texts = model.generate_trajectory(pixel_values, [ego_positions_py])
        except Exception as e:
            print(f"Generation failed for idx {idx}: {e}")
            continue

        gen_text = gen_texts[0]
        pred_coords = parse_coords_from_text(gen_text, max_points=10)

        if pred_coords.shape[0] < 10:
            # pad with NaNs so shapes align
            pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
            pred_coords = np.vstack([pred_coords, pad])

        # metrics
        diffs = pred_coords - gt_waypoints
        l2_per_waypoint = np.linalg.norm(diffs, axis=1)
        mean_l2 = np.nanmean(l2_per_waypoint)
        final_disp = l2_per_waypoint[-1]

        results.append({
            'idx': idx,
            'mean_l2': float(mean_l2),
            'final_disp': float(final_disp),
            'gen_text': gen_text
        })

    # aggregate
    mean_l2s = [r['mean_l2'] for r in results]
    final_disps = [r['final_disp'] for r in results]

    summary = {
        'num_samples': len(results),
        'mean_l2_mean': float(np.nanmean(mean_l2s)) if len(mean_l2s) > 0 else float('nan'),
        'mean_l2_std': float(np.nanstd(mean_l2s)) if len(mean_l2s) > 0 else float('nan'),
        'final_disp_mean': float(np.nanmean(final_disps)) if len(final_disps) > 0 else float('nan')
    }

    # save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_name)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'mean_l2', 'final_disp', 'gen_text'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print('Evaluation summary:')
    print(summary)
    print(f'Results CSV: {csv_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (final_model.pth or projector state dict)')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to evaluate (default 200 or all if -1)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs')
    parser.add_argument('--output_name', type=str, default='eval_results.csv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.num_samples == -1:
        args.num_samples = None
    evaluate(args)
