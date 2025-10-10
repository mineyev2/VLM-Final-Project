import os, sys
from termcolor import colored
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class NuScenesDataset(Dataset):
    def __init__(self, version, dataroot, nsweeps=5):
        """
        Initialize the NuScenes dataset for use with PyTorch.

        Args:
            version (str): Version of the NuScenes dataset (e.g., 'v1.0-mini', 'v1.0-trainval').
            dataroot (str): Path to the root directory of the NuScenes dataset.
            nsweeps (int): Number of sweeps to use for LiDAR data.
        """        
        print(colored(f"Initializing NuScenes dataset with version {version} at {dataroot}", "cyan"))
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        self.nsweeps = nsweeps # TODO: Figure out this value

        # Each scene has multiple samples (frames).
        # We store the tokens from all samples across all scenes
        # This makes __getitem__ easier to implement.
        self.sample_tokens = [] # Around 10 x 40 for mini, hopefully not too large for full dataset

        print(colored("Loading sample tokens from all scenes into a list...", "cyan"))
        for scene in self.nusc.scene:
            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)
            
            # Continuously get next sample in current scene until we run out of all samples
            while sample:
                self.sample_tokens.append(sample['token'])
                if sample['next'] == '':
                    break
                sample = self.nusc.get('sample', sample['next'])

        print(colored(f"Loaded {len(self.sample_tokens)} samples from {len(self.nusc.scene)} scenes.", "green"))

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        # Get LiDAR point cloud with nsweeps
        nuscenes_pointcloud, _ = LidarPointCloud.from_file_multisweep(
            self.nusc,
            sample,
            chan='LIDAR_TOP',
            ref_chan='LIDAR_TOP',
            nsweeps=self.nsweeps, # TODO: This isn't working right now
            min_distance=1.0  # Filter out points closer than 1 meter
        )

        torch_pointcloud = torch.from_numpy(nuscenes_pointcloud.points) # 4 x N

        # Get front camera image
        camera_token = sample['data']['CAM_FRONT']
        camera_data = self.nusc.get('sample_data', camera_token)
        image_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        image = np.array(Image.open(image_path))  # H x W x 3
        torch_image = torch.from_numpy(image)  # H x W x 3

        return torch_pointcloud, torch_image