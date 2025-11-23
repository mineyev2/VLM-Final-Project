"""
View a single NuScenes sample by index.
Displays sample information and saves camera images with LiDAR and BEV visualizations.
"""

import argparse
import os
import numpy as np
import cv2
from termcolor import colored
from nuscenes import NuScenes
from PIL import Image


class NuScenesViewer:
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
    
    def get_idx_from_token(self, token):
        """Get index from sample token."""
        try:
            return self.sample_tokens.index(token)
        except ValueError:
            return None

    def get_sample_info(self, idx):
        """Get all information about a sample."""
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Get scene info
        scene = self.nusc.get('scene', sample['scene_token'])
        
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

        # Get image
        camera_token = sample['data']['CAM_FRONT']
        camera_data = self.nusc.get('sample_data', camera_token)
        image_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        image = Image.open(image_path).convert('RGB')

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
            'image_path': image_path,
            'ego_positions': ego_positions,
            'waypoints': np.array(waypoints, dtype=float),
            'scene_name': scene['name'],
            'scene_description': scene['description'],
            'timestamp': sample['timestamp'],
            'sample_token': sample_token,
            'sample': sample
        }
    
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


def view_sample(args):
    viewer = NuScenesViewer(args.version, args.dataroot)
    
    # Determine if using token or index
    sample_idx = None
    sample_token = None
    
    if args.sample_token:
        # User provided a token
        sample_token = args.sample_token
        sample_idx = viewer.get_idx_from_token(sample_token)
        if sample_idx is None:
            print(colored(f"✗ Error: Sample token '{sample_token}' not found in dataset", "red"))
            return
        print(colored(f"Token '{sample_token}' found at index {sample_idx}", "cyan"))
    else:
        # User provided an index
        sample_idx = args.sample_idx
        if sample_idx < 0 or sample_idx >= len(viewer):
            print(colored(f"✗ Error: Sample index {sample_idx} out of range [0, {len(viewer)-1}]", "red"))
            return
        sample_token = viewer.sample_tokens[sample_idx]
        print(colored(f"Sample index {sample_idx} corresponds to token: {sample_token}", "cyan"))
    
    print(colored(f"{'='*70}", "cyan"))
    print(colored(f"SAMPLE INDEX: {sample_idx}", "cyan", attrs=['bold']))
    print(colored(f"SAMPLE TOKEN: {sample_token}", "cyan", attrs=['bold']))
    print(colored(f"{'='*70}\n", "cyan"))
    
    # Get sample info
    info = viewer.get_sample_info(sample_idx)
    
    # Print information
    print(colored("Scene Information:", "yellow", attrs=['bold']))
    print(f"  Name:        {info['scene_name']}")
    print(f"  Description: {info['scene_description']}")
    print(f"  Timestamp:   {info['timestamp']}\n")
    
    print(colored("Ego Vehicle History (last 3 positions):", "yellow", attrs=['bold']))
    for i, pos in enumerate(info['ego_positions']):
        print(f"  t-{2-i}: [{pos[0]:8.2f}, {pos[1]:8.2f}]")
    print()
    
    print(colored("Ground Truth Future Trajectory (10 waypoints):", "yellow", attrs=['bold']))
    for i, wp in enumerate(info['waypoints']):
        print(f"  WP {i+1:2d}: [{wp[0]:8.2f}, {wp[1]:8.2f}]")
    print()
    
    print(colored("Image Information:", "yellow", attrs=['bold']))
    print(f"  Size:   {info['image'].size}")
    print(f"  Mode:   {info['image'].mode}")
    print(f"  Source: {info['image_path']}\n")
    
    # Save images
    if args.save_image:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Use NuScenes native rendering for LiDAR
        show_lidar = True
        if show_lidar:
            print(colored("Rendering LiDAR points using NuScenes native method...", "yellow"))
            try:
                # Use NuScenes built-in render_pointcloud_in_image
                viewer.nusc.render_pointcloud_in_image(
                    sample_token=info['sample_token'],
                    dot_size=args.lidar_point_size,
                    pointsensor_channel='LIDAR_TOP',
                    camera_channel='CAM_FRONT',
                    out_path=None,
                    render_intensity=False,
                    show_lidarseg=False,
                    show_panoptic=False
                )
                
                # Save the matplotlib figure
                import matplotlib.pyplot as plt
                output_path = os.path.join(args.output_dir, 'sample_lidar.jpg')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(colored(f"✓ Image with LiDAR saved to: {output_path}", "green"))
                
            except Exception as e:
                print(colored(f"✗ LiDAR rendering failed: {e}", "red"))
                print(colored("Falling back to image without LiDAR...", "yellow"))
                show_lidar = False
        
        # Save regular image (with optional text overlay)
        if not show_lidar:
            output_path = os.path.join(args.output_dir, 'sample.jpg')
            img_cv = cv2.cvtColor(np.array(info['image']), cv2.COLOR_RGB2BGR)
            
            # Add text overlay with sample info
            if args.add_overlay:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_cv, f"Sample Index: {args.sample_idx}", (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(img_cv, f"Scene: {info['scene_name']}", (10, 70), font, 0.7, (255, 255, 255), 2)
                
                # Add ego positions
                y_pos = 110
                cv2.putText(img_cv, "Ego History:", (10, y_pos), font, 0.6, (0, 255, 255), 2)
                for i, pos in enumerate(info['ego_positions']):
                    y_pos += 30
                    cv2.putText(img_cv, f"  t-{2-i}: [{pos[0]:.1f}, {pos[1]:.1f}]", (10, y_pos), font, 0.5, (0, 255, 255), 1)
            
            cv2.imwrite(output_path, img_cv)
            print(colored(f"✓ Image saved to: {output_path}", "green"))
    
    # Save all camera views if requested
    if args.show_all_cams and args.save_image:
        print(colored("Creating multi-camera view...", "yellow"))
        try:
            import matplotlib.pyplot as plt
            
            # Get all camera images
            cam_info = viewer.get_all_camera_images(args.sample_idx)
            
            output_path = os.path.join(args.output_dir, 'sample_all_cams.jpg')
            
            # Create figure with 2 rows, 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Sample {args.sample_idx}: {cam_info["scene_name"]}', fontsize=16, fontweight='bold')
            
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
            
            for cam_name, img in cam_info['images'].items():
                if cam_name in camera_positions:
                    row, col = camera_positions[cam_name]
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(cam_name.replace('CAM_', ''), fontsize=12, fontweight='bold')
                    axes[row, col].axis('off')
            
            # Remove any empty subplots
            for cam_name in camera_positions.keys():
                if cam_name not in cam_info['images']:
                    row, col = camera_positions[cam_name]
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(colored(f"✓ All cameras view saved to: {output_path}", "green"))
            
        except Exception as e:
            print(colored(f"✗ Multi-camera rendering failed: {e}", "red"))
    
    # Save bird's eye view
    show_bev = True
    if show_bev and args.save_image:
        print(colored("Rendering bird's eye view of LiDAR...", "yellow"))
        try:
            import matplotlib.pyplot as plt
            # Use NuScenes built-in render_sample_data for top-down view
            lidar_token = info['sample']['data']['LIDAR_TOP']
            output_path = os.path.join(args.output_dir, 'sample_bev.jpg')
            
            # Note: Annotations only available in v1.0-mini and v1.0-trainval, not in v1.0-test
            viewer.nusc.render_sample_data(
                lidar_token,
                with_anns=True,
                box_vis_level=0,  # Show all boxes (0=any visibility)
                underlay_map=True,
                nsweeps=5,
            )
            
            # Save the figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(colored(f"✓ Bird's eye view saved to: {output_path}", "green"))
            
        except Exception as e:
            print(colored(f"✗ BEV rendering failed: {e}", "red"))
    
    print(colored(f"\n{'='*70}\n", "cyan"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="View a single NuScenes sample without running model",
        epilog="""Examples:
  # View by index
  python view_sample.py --sample_idx 42 --save_image
  
  # View by token
  python view_sample.py --sample_token cc8c0bf57f984915a77078b10eb33198 --save_image
  
  # Disable specific views
  python view_sample.py --sample_idx 42 --save_image --no_all_cams
  
  # With text overlay
  python view_sample.py --sample_idx 42 --save_image --add_overlay
  
Output files (when --save_image is used):
  - sample_lidar.jpg: Camera view with LiDAR overlay
  - sample_bev.jpg: Bird's eye view (top-down)
  - sample_all_cams.jpg: All 6 camera views in 2x3 grid
  - sample.jpg: Plain camera view (only if LiDAR rendering fails)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sample_idx', type=int, default=None, help='Index of sample to view')
    parser.add_argument('--sample_token', type=str, default=None, help='Sample token to view (alternative to --sample_idx)')
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3', help='Path to NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-test', help='NuScenes version (Note: v1.0-test has no annotations; use v1.0-mini or v1.0-trainval to see bounding boxes)')
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/temp_vis', help='Directory to save images')
    parser.add_argument('--save_image', action='store_true', default=True, help='Save images to output directory')
    parser.add_argument('--add_overlay', action='store_true', help='Add text overlay with sample info')
    parser.add_argument('--lidar_point_size', type=int, default=5, help='Size of LiDAR points in pixels')
    parser.add_argument('--no_all_cams', dest='show_all_cams', action='store_false', default=True, help='Disable all-camera view')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Validate that either sample_idx or sample_token is provided
    if args.sample_idx is None and args.sample_token is None:
        print(colored("✗ Error: Must provide either --sample_idx or --sample_token", "red"))
        exit(1)
    
    if args.sample_idx is not None and args.sample_token is not None:
        print(colored("✗ Error: Cannot provide both --sample_idx and --sample_token", "red"))
        exit(1)
    
    view_sample(args)
