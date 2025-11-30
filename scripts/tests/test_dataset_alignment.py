"""
Test script to verify NuScenesDataset alignment and data quality.

This script checks:
1. LiDAR-Image alignment (points should be within image bounds)
2. Data shapes and types
3. Tokenization correctness
4. Label masking
5. Coordinate system consistency
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.lidar_emma import LidarEMMA
import torch
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from pathlib import Path

from scripts.nuscenes_dataset import NuScenesDataset
from transformers import AutoTokenizer
from nuscenes.utils.geometry_utils import view_points

import gc


def visualize_lidar_on_image(sample, save_path=None):
    """Visualize LiDAR points projected onto the image."""
    import matplotlib.patches as mpatches
    
    image = sample['image']
    lidar = sample.get('lidar', None)
    
    if lidar is None:
        print(colored("No LiDAR data in sample", "yellow"))
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(image)
    
    # LiDAR points are in KITTI-style coords (x-forward, y-left, z-up)
    # Need to convert back to camera frame for projection
    # Points are [N, 4] with (x, y, z, intensity) in KITTI format
    points_kitti = lidar[:, :3].numpy()  # [N, 3]
    
    # Convert from KITTI (x-forward, y-left, z-up) back to camera frame
    # Camera frame: x-right, y-down, z-forward
    # Reverse the transformation from dataset:
    #   torch_pointcloud = torch_pointcloud[:, (2, 0, 1, 3)]  # (z, x, y, intensity)
    #   torch_pointcloud[:, 1] = -torch_pointcloud[:, 1]  # negate y
    #   torch_pointcloud[:, 2] = -torch_pointcloud[:, 2]  # negate z
    
    # So to reverse: x_cam = y_kitti (negated), y_cam = z_kitti (negated), z_cam = x_kitti
    points_cam = np.zeros_like(points_kitti)
    points_cam[:, 0] = -points_kitti[:, 1]  # x_cam = -y_kitti (right)
    points_cam[:, 1] = -points_kitti[:, 2]  # y_cam = -z_kitti (down)
    points_cam[:, 2] = points_kitti[:, 0]   # z_cam = x_kitti (forward)
    
    cam_intrinsic = sample['cam_to_ego']['camera_intrinsic']
    
    # Filter points behind camera
    # valid_mask = points_cam[:, 2] > 0
    # points_cam = points_cam[valid_mask]

    # Check num points behind camera
    num_behind_camera = (points_cam[:, 2] <= 0).sum()
    print(colored(f"Number of points behind camera: {num_behind_camera}", "blue"))
    
    if len(points_cam) == 0:
        print(colored("No valid points to project", "yellow"))
        return
    
    # Project using view_points (same as dataset)
    points_2d = view_points(points_cam.T, cam_intrinsic, normalize=True)
    x_img = points_2d[0, :]
    y_img = points_2d[1, :]
    depths = points_cam[:, 2]
    
    # Color by depth
    scatter = ax.scatter(x_img, y_img, c=depths, s=1, cmap='jet', alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    
    # Check bounds
    w, h = image.size
    in_bounds = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    out_bounds = ~in_bounds
    
    ax.set_title(f'LiDAR Projection: {in_bounds.sum()}/{len(x_img)} points in bounds')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    # Add legend
    in_patch = mpatches.Patch(color='green', label=f'In bounds: {in_bounds.sum()}')
    out_patch = mpatches.Patch(color='red', label=f'Out of bounds: {out_bounds.sum()}')
    ax.legend(handles=[in_patch, out_patch])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(colored(f"Saved visualization to {save_path}", "green"))
    else:
        plt.show()
    
    plt.close()
    
    return {
        'total_points': len(x_img),
        'in_bounds': in_bounds.sum(),
        'out_bounds': out_bounds.sum(),
        'percentage_in_bounds': 100 * in_bounds.sum() / len(x_img)
    }


def check_lidar_image_alignment(sample, verbose=True):
    """Check if LiDAR points are properly aligned with image frame."""
    lidar = sample.get('lidar', None)
    image = sample['image']
    
    if lidar is None:
        if verbose:
            print(colored("✗ No LiDAR data in sample", "yellow"))
        return None
    
    w, h = image.size
    cam_intrinsic = sample['cam_to_ego']['camera_intrinsic']
    
    # Convert KITTI coords back to camera frame for projection
    points_kitti = lidar[:, :3].numpy()
    
    # Convert from KITTI (x-forward, y-left, z-up) to camera (x-right, y-down, z-forward)
    points_cam = np.zeros_like(points_kitti)
    points_cam[:, 0] = -points_kitti[:, 1]  # x_cam = -y_kitti
    points_cam[:, 1] = -points_kitti[:, 2]  # y_cam = -z_kitti
    points_cam[:, 2] = points_kitti[:, 0]   # z_cam = x_kitti
    
    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]
    
    if len(points_cam) == 0:
        if verbose:
            print(colored("✗ No valid points (all behind camera)", "red"))
        return {'valid': False, 'reason': 'all_points_behind_camera'}
    
    # Project using view_points (same as dataset)
    points_2d = view_points(points_cam.T, cam_intrinsic, normalize=True)
    x_img = points_2d[0, :]
    y_img = points_2d[1, :]
    
    # Check bounds
    in_bounds = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    percentage_in = 100 * in_bounds.sum() / len(x_img)
    
    result = {
        'valid': True,
        'total_points': len(x_img),
        'in_bounds': int(in_bounds.sum()),
        'out_bounds': int((~in_bounds).sum()),
        'percentage_in_bounds': float(percentage_in),
        'image_size': (w, h)
    }
    
    if verbose:
        if percentage_in > 95:
            print(colored(f"✓ LiDAR-Image alignment: {percentage_in:.1f}% points in bounds", "green"))
        elif percentage_in > 80:
            print(colored(f"⚠ LiDAR-Image alignment: {percentage_in:.1f}% points in bounds", "yellow"))
        else:
            print(colored(f"✗ LiDAR-Image alignment: {percentage_in:.1f}% points in bounds", "red"))
    
    return result


def check_data_shapes(sample, verbose=True):
    """Check shapes and types of all data fields."""
    checks = {}
    
    # Image
    image = sample['image']
    checks['image'] = {
        'type': str(type(image)),
        'size': image.size,
        'mode': image.mode
    }
    if verbose:
        print(colored(f"✓ Image: {image.size}, mode={image.mode}", "green"))
    
    # LiDAR
    lidar = sample.get('lidar', None)
    if lidar is not None:
        checks['lidar'] = {
            'shape': lidar.shape,
            'dtype': str(lidar.dtype),
            'num_points': lidar.shape[0],
            'has_intensity': lidar.shape[1] >= 4
        }
        if verbose:
            print(colored(f"✓ LiDAR: {lidar.shape}, dtype={lidar.dtype}", "green"))
    else:
        checks['lidar'] = None
        if verbose:
            print(colored("⚠ LiDAR: None (disabled)", "yellow"))
    
    # Input IDs
    input_ids = sample['input_ids']
    checks['input_ids'] = {
        'shape': input_ids.shape,
        'dtype': str(input_ids.dtype),
        'max_val': int(input_ids.max()),
        'min_val': int(input_ids.min())
    }
    if verbose:
        print(colored(f"✓ Input IDs: {input_ids.shape}, range=[{input_ids.min()}, {input_ids.max()}]", "green"))
    
    # Labels
    labels = sample['labels']
    checks['labels'] = {
        'shape': labels.shape,
        'dtype': str(labels.dtype),
        'num_masked': int((labels == -100).sum()),
        'num_unmasked': int((labels != -100).sum())
    }
    if verbose:
        print(colored(f"✓ Labels: {labels.shape}, masked={checks['labels']['num_masked']}, "
                     f"unmasked={checks['labels']['num_unmasked']}", "green"))
    
    # Positions
    ego_positions = sample['ego_positions']
    waypoints = sample['waypoints']
    checks['ego_positions'] = {
        'length': len(ego_positions),
        'first': ego_positions[0].tolist() if len(ego_positions) > 0 else None,
        'last': ego_positions[-1].tolist() if len(ego_positions) > 0 else None
    }
    checks['waypoints'] = {
        'length': len(waypoints),
        'first': waypoints[0].tolist() if len(waypoints) > 0 else None,
        'last': waypoints[-1].tolist() if len(waypoints) > 0 else None
    }
    if verbose:
        print(colored(f"✓ Ego positions: {len(ego_positions)} points", "green"))
        print(colored(f"✓ Waypoints: {len(waypoints)} points", "green"))
    
    return checks


def check_tokenization(sample, tokenizer, verbose=True):
    """Check tokenization quality and label masking."""
    input_ids = sample['input_ids']
    labels = sample['labels']
    
    # Decode tokens
    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_clean = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    # Decode only unmasked tokens (what model learns from)
    target_mask = labels != -100
    if target_mask.sum() > 0:
        target_ids = input_ids[target_mask]
        decoded_target = tokenizer.decode(target_ids, skip_special_tokens=True)
    else:
        decoded_target = "[ALL MASKED]"
    
    checks = {
        'total_tokens': len(input_ids),
        'masked_tokens': int((labels == -100).sum()),
        'unmasked_tokens': int((labels != -100).sum()),
        'padding_tokens': int((input_ids == tokenizer.pad_token_id).sum()),
        'decoded_length': len(decoded_clean),
        'decoded_sample': decoded_clean[:200] + "..." if len(decoded_clean) > 200 else decoded_clean,
        'target_sample': decoded_target[:100] + "..." if len(decoded_target) > 100 else decoded_target
    }
    
    if verbose:
        print(colored(f"\n{'='*70}", "cyan"))
        print(colored("Tokenization Analysis:", "cyan", attrs=['bold']))
        print(colored(f"  Total tokens: {checks['total_tokens']}", "white"))
        print(colored(f"  Masked (prompt): {checks['masked_tokens']}", "yellow"))
        print(colored(f"  Unmasked (target): {checks['unmasked_tokens']}", "green"))
        print(colored(f"  Padding: {checks['padding_tokens']}", "white"))
        print(colored(f"\n  Decoded (first 200 chars):", "cyan"))
        print(f"    {checks['decoded_sample']}")
        print(colored(f"\n  Target only (first 100 chars):", "green"))
        print(f"    {checks['target_sample']}")
        print(colored(f"{'='*70}\n", "cyan"))
    
    return checks


def check_coordinates_consistency(sample, verbose=True):
    """Check coordinate transformations and consistency."""
    ego_positions = np.array(sample['ego_positions'])
    waypoints = np.array(sample['waypoints'])
    
    # Check if positions are reasonable (not too far apart)
    ego_diffs = np.linalg.norm(ego_positions[1:] - ego_positions[:-1], axis=1)
    wp_diffs = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
    
    checks = {
        'ego_mean_distance': float(ego_diffs.mean()),
        'ego_max_distance': float(ego_diffs.max()),
        'waypoint_mean_distance': float(wp_diffs.mean()),
        'waypoint_max_distance': float(wp_diffs.max()),
        'ego_position_range': {
            'x': [float(ego_positions[:, 0].min()), float(ego_positions[:, 0].max())],
            'y': [float(ego_positions[:, 1].min()), float(ego_positions[:, 1].max())],
            'z': [float(ego_positions[:, 2].min()), float(ego_positions[:, 2].max())]
        }
    }
    
    if verbose:
        print(colored(f"✓ Ego history: mean step={checks['ego_mean_distance']:.2f}m, "
                     f"max step={checks['ego_max_distance']:.2f}m", "green"))
        print(colored(f"✓ Future waypoints: mean step={checks['waypoint_mean_distance']:.2f}m, "
                     f"max step={checks['waypoint_max_distance']:.2f}m", "green"))
        
        # Sanity checks
        if checks['ego_max_distance'] > 50:
            print(colored(f"  ⚠ Warning: Large ego position jump detected!", "yellow"))
        if checks['waypoint_max_distance'] > 50:
            print(colored(f"  ⚠ Warning: Large waypoint jump detected!", "yellow"))
    
    return checks


def run_full_test(dataset, num_samples=5, save_dir=None):
    """Run comprehensive tests on multiple samples."""
    print(colored(f"\n{'='*70}", "cyan", attrs=['bold']))
    print(colored(f"Running Dataset Validation Tests", "cyan", attrs=['bold']))
    print(colored(f"{'='*70}\n", "cyan"))
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = dataset.tokenizer
    results = []
    
    for i in range(min(num_samples, len(dataset))):
        print(colored(f"\n{'─'*70}", "white"))
        print(colored(f"Sample {i+1}/{num_samples}", "cyan", attrs=['bold']))
        print(colored(f"{'─'*70}\n", "white"))
        
        sample = dataset[i]
        
        result = {
            'sample_idx': i,
            'shapes': check_data_shapes(sample, verbose=True),
            'tokenization': check_tokenization(sample, tokenizer, verbose=True),
            'coordinates': check_coordinates_consistency(sample, verbose=True),
        }
        
        # Check LiDAR alignment if available
        if sample.get('lidar') is not None:
            result['lidar_alignment'] = check_lidar_image_alignment(sample, verbose=True)
            
            # Visualize
            if save_dir:
                vis_path = save_dir / f"sample_{i:03d}_lidar_projection.png"
                vis_stats = visualize_lidar_on_image(sample, save_path=vis_path)
                result['visualization'] = vis_stats
        
        results.append(result)
    
    # Summary
    print(colored(f"\n{'='*70}", "cyan", attrs=['bold']))
    print(colored(f"Test Summary", "cyan", attrs=['bold']))
    print(colored(f"{'='*70}\n", "cyan"))
    
    if any('lidar_alignment' in r for r in results):
        avg_alignment = np.mean([r['lidar_alignment']['percentage_in_bounds'] 
                                for r in results if 'lidar_alignment' in r])
        print(colored(f"Average LiDAR alignment: {avg_alignment:.1f}% points in bounds", "green"))
    
    avg_masked = np.mean([r['tokenization']['masked_tokens'] for r in results])
    avg_unmasked = np.mean([r['tokenization']['unmasked_tokens'] for r in results])
    print(colored(f"Average tokens - Masked: {avg_masked:.1f}, Unmasked: {avg_unmasked:.1f}", "green"))
    
    print(colored(f"\n✓ All tests completed successfully!", "green", attrs=['bold']))
    
    return results


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}\n")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Initialize tokenizer
    print(colored("Loading tokenizer...", "cyan"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    
    # Initialize dataset
    print(colored("Loading dataset...", "cyan"))
    model = LidarEMMA(device, use_lidar=True)
    
    dataset = NuScenesDataset(
        version="v1.0-test",
        dataroot="/storage/ice-shared/cs8803vlm/rmineyev3/",
        tokenizer=tokenizer,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2,
        output_lidar=True
    )
    
    print(colored(f"✓ Dataset loaded: {len(dataset)} samples\n", "green"))
    
    # Run tests
    output_dir = Path("./test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_full_test(dataset, num_samples=5, save_dir=output_dir)
    
    print(colored(f"\n✓ Tests complete! Visualizations saved to {output_dir.absolute()}", "green", attrs=['bold']))