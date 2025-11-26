import os
from termcolor import colored
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class NuScenesDataset(Dataset):
    def __init__(self, version, dataroot, tokenizer, prompt_part1, prompt_part2, nsweeps=5, output_lidar=False):
        """
        Initialize the NuScenes dataset for use with PyTorch.
        Serves EGOCENTRIC (Local) coordinates with metadata for Global reconstruction.
        """
        print(colored(f"Initializing NuScenes dataset with version {version} at {dataroot}", "cyan"))
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.nsweeps = nsweeps 
        self.output_lidar = output_lidar

        self.tokenizer = tokenizer
        self.prompt_part1 = prompt_part1
        self.prompt_part2 = prompt_part2

        # Create list of samples
        self.samples = []
        for scene in self.nusc.scene:
            nbr_samples = scene['nbr_samples']
            if nbr_samples < 20: 
                continue
            
            sample = self.nusc.get('sample', scene['first_sample_token'])
            
            # Skip first 9 samples to ensure 10 frames of history
            for _ in range(9): 
                sample = self.nusc.get('sample', sample['next'])
                
            # Add samples (stop 10 frames before end for future ground truth)
            for _ in range(nbr_samples - 19):
                self.samples.append(sample)
                sample = self.nusc.get('sample', sample['next'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Get Current Ego Pose (Reference Frame)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        # 2. Get History (10 frames)
        history_points = []
        curr_hist = sample
        for _ in range(10):
            c_data = self.nusc.get('sample_data', curr_hist['data']['CAM_FRONT'])
            e_pose = self.nusc.get('ego_pose', c_data['ego_pose_token'])
            history_points.append(np.array(e_pose['translation']))
            
            if curr_hist['prev']:
                curr_hist = self.nusc.get('sample', curr_hist['prev'])
            else:
                history_points.append(history_points[-1])
        history_points.reverse()

        # 3. Get Future Ground Truth (10 frames)
        future_points = []
        curr_fut = sample
        for _ in range(10):
            if curr_fut['next'] == '': 
                break
            curr_fut = self.nusc.get('sample', curr_fut['next'])
            c_data = self.nusc.get('sample_data', curr_fut['data']['CAM_FRONT'])
            e_pose = self.nusc.get('ego_pose', c_data['ego_pose_token'])
            future_points.append(np.array(e_pose['translation']))
        
        while len(future_points) < 10:
            future_points.append(future_points[-1] if len(future_points) > 0 else np.array(ego_pose_curr['translation']))

        # 5. Load Image
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(image_path).convert('RGB')

        # 6. Load lidar
        torch_pointcloud = None
        if self.output_lidar:
            nuscenes_pointcloud, _ = LidarPointCloud.from_file_multisweep(
                self.nusc,
                sample,
                chan='LIDAR_TOP',
                ref_chan='LIDAR_TOP',
                nsweeps=self.nsweeps, # TODO: This isn't working right now
                min_distance=1.0  # Filter out points closer than 1 meter
            )
            torch_pointcloud = torch.from_numpy(nuscenes_pointcloud.points.T).float()
        
        # 7. Format Text Prompt
        pos_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in history_points])
        prompt = f"{self.prompt_part1}[{pos_str}]\n{self.prompt_part2}"
        
        wp_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in future_points])
        target_text = "Future Trajectory: [" + wp_str + "]"
        
        full_text = prompt + target_text
        
        # 8. Tokenize & Mask
        input_ids = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=2048,
            truncation=True
        ).input_ids.squeeze(0)
        
        labels = input_ids.clone()
        
        # Calculate prompt length for masking
        # Note: We tokenize the prompt separately to find the boundary
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        prompt_len = prompt_ids.shape[0]
        
        # Apply -100 mask to the prompt part (loss is only calculated on target_text)
        if prompt_len < labels.shape[0]:
            labels[:prompt_len] = -100
        else:
            labels[:] = -100 # Safety fallback if truncation cut off the target
        
        # Calibration
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        return {
            'image': image,
            'lidar': torch_pointcloud,
            'input_ids': input_ids,
            'labels': labels,
            'ego_positions': history_points,
            'waypoints': future_points,
            'cam_to_ego': {
                'translation': cam_calib['translation'],
                'rotation': cam_calib['rotation'],
                'camera_intrinsic': np.array(cam_calib['camera_intrinsic'])
            },
            'ego_to_world': {
                'translation': ego_pose_curr['translation'],
                'rotation': ego_pose_curr['rotation']
            }
        }
    