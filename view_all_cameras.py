"""
View all camera images from a single NuScenes sample.
Displays all 6 camera views in a single image using matplotlib subplots.
"""

import argparse
import os
import numpy as np
from nuscenes import NuScenes
from PIL import Image
import matplotlib.pyplot as plt
from termcolor import colored


class NuScenesAllCamerasViewer:
    def __init__(self, version, dataroot):
        print(colored("Loading NuScenes dataset...", "yellow"))
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        # Build sample list (same logic as evaluation)
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
        
        print(colored(f"✓ Dataset loaded ({len(self.sample_tokens)} samples available)\n", "green"))
        
        # Camera channels in NuScenes
        self.camera_channels = [
            'CAM_FRONT_LEFT',
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT',
            'CAM_BACK',
            'CAM_BACK_RIGHT'
        ]

    def __len__(self):
        return len(self.sample_tokens)

    def get_all_camera_images(self, idx):
        """Get all camera images for a sample."""
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Get scene info
        scene = self.nusc.get('scene', sample['scene_token'])
        
        images = {}
        for cam_channel in self.camera_channels:
            if cam_channel in sample['data']:
                camera_token = sample['data'][cam_channel]
                camera_data = self.nusc.get('sample_data', camera_token)
                image_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
                images[cam_channel] = Image.open(image_path).convert('RGB')
        
        return {
            'images': images,
            'scene_name': scene['name'],
            'scene_description': scene['description'],
            'timestamp': sample['timestamp'],
            'sample_token': sample_token
        }


def view_all_cameras(args):
    viewer = NuScenesAllCamerasViewer(args.version, args.dataroot)
    
    # Validate index
    if args.sample_idx < 0 or args.sample_idx >= len(viewer):
        print(colored(f"✗ Error: Sample index {args.sample_idx} out of range [0, {len(viewer)-1}]", "red"))
        return
    
    print(colored(f"{'='*70}", "cyan"))
    print(colored(f"SAMPLE INDEX: {args.sample_idx}", "cyan", attrs=['bold']))
    print(colored(f"{'='*70}\n", "cyan"))
    
    # Get all camera images
    info = viewer.get_all_camera_images(args.sample_idx)
    
    # Print information
    print(colored("Scene Information:", "yellow", attrs=['bold']))
    print(f"  Name:        {info['scene_name']}")
    print(f"  Description: {info['scene_description']}")
    print(f"  Timestamp:   {info['timestamp']}\n")
    
    print(colored(f"Available Cameras: {len(info['images'])}", "yellow", attrs=['bold']))
    for cam_name in info['images'].keys():
        print(f"  - {cam_name}")
    print()
    
    # Create subplot layout
    if args.save_image:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'sample_{args.sample_idx:04d}_all_cams.jpg')
        
        print(colored("Creating multi-camera view...", "yellow"))
        
        # Create figure with 2 rows, 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Sample {args.sample_idx}: {info["scene_name"]}', fontsize=16, fontweight='bold')
        
        # Layout: 
        # Row 0: FRONT_LEFT, FRONT, FRONT_RIGHT
        # Row 1: BACK_LEFT, BACK, BACK_RIGHT
        camera_positions = {
            'CAM_FRONT_LEFT': (0, 0),
            'CAM_FRONT': (0, 1),
            'CAM_FRONT_RIGHT': (0, 2),
            'CAM_BACK_LEFT': (1, 0),
            'CAM_BACK': (1, 1),
            'CAM_BACK_RIGHT': (1, 2)
        }
        
        for cam_name, img in info['images'].items():
            if cam_name in camera_positions:
                row, col = camera_positions[cam_name]
                axes[row, col].imshow(img)
                axes[row, col].set_title(cam_name.replace('CAM_', ''), fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
        
        # Remove any empty subplots
        for cam_name in camera_positions.keys():
            if cam_name not in info['images']:
                row, col = camera_positions[cam_name]
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        
        print(colored(f"✓ All cameras view saved to: {output_path}", "green"))
    
    # Display using matplotlib if requested
    if args.display:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Sample {args.sample_idx}: {info["scene_name"]}', fontsize=16, fontweight='bold')
        
        camera_positions = {
            'CAM_FRONT_LEFT': (0, 0),
            'CAM_FRONT': (0, 1),
            'CAM_FRONT_RIGHT': (0, 2),
            'CAM_BACK_LEFT': (1, 0),
            'CAM_BACK': (1, 1),
            'CAM_BACK_RIGHT': (1, 2)
        }
        
        for cam_name, img in info['images'].items():
            if cam_name in camera_positions:
                row, col = camera_positions[cam_name]
                axes[row, col].imshow(img)
                axes[row, col].set_title(cam_name.replace('CAM_', ''), fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
        
        for cam_name in camera_positions.keys():
            if cam_name not in info['images']:
                row, col = camera_positions[cam_name]
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print(colored(f"\n{'='*70}\n", "cyan"))


def parse_args():
    parser = argparse.ArgumentParser(description="View all camera images from a single NuScenes sample")
    parser.add_argument('--sample_idx', type=int, required=True, help='Index of sample to view')
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test', help='NuScenes version')
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/temp_vis', help='Directory to save image')
    parser.add_argument('--save_image', type=bool, default=True, help='Save image to output directory')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for saved image')
    parser.add_argument('--display', action='store_true', help='Display image using matplotlib')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    view_all_cameras(args)
