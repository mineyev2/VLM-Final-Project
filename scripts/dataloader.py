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
            version (str): Version of the NuScenes dataset to use (e.g., 'v1.0-mini', 'v1.0-trainval').
            dataroot (str): Root directory where the NuScenes dataset is stored.
            nsweeps (int): Number of LiDAR sweeps to combine for each sample.
        """

        print(colored(f"Initializing NuScenes dataset with version {version} at {dataroot}", "cyan"))
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        self.nsweeps = nsweeps # TODO: Figure out this value

        # Rewrite so all samples are in a list, for easy indexing
        self.sample_tokens = [] # Around 10 x 40 for mini, hopefully not too large for full dataset

        print(colored("Loading sample tokens from all scenes into a list...", "cyan"))
        for scene in self.nusc.scene:

            # Get number of samples
            nbr_samples = scene['nbr_samples']

            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)

            # Only use samples that have at least 10 future waypoints (model predicts 10 future waypoints)
            for _ in range(nbr_samples - 10):
                self.sample_tokens.append(sample['token'])
                sample = self.nusc.get('sample', sample['next'])

        print(colored(f"Loaded {len(self.sample_tokens)} samples from {len(self.nusc.scene)} scenes.", "green"))

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        """
        Get a data sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            dict: A dictionary containing:
                - 'image': Front camera image as a torch tensor (H, W, 3).
                    - NOTE: This value is not normalized (may need to divide by 255)
                - 'lidar': LiDAR point cloud as a torch tensor (4, N).
                    - NOTE: nsweeps=5 (5 360deg lidar sensor rotations), unsure if this is good/practical approach
                - 'waypoints': 10 future waypoints as a torch tensor (10, 2).
                    - NOTE: car location is retrieved from its position of when
                            the image was taken, rather than when the LIDAR was taken.
                            This isn't a huge difference, but worthy of note.
        """

        # Get sample at idx
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
        torch_image = torch.from_numpy(image).float() # NOTE: This is not normalized

        # Get waypoints
        waypoints = []
        current_sample = sample
        for _ in range(10): # Get 10 future waypoints
            next_sample_token = current_sample['next']
            next_sample = self.nusc.get('sample', next_sample_token)
            next_camera_data = self.nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
            next_ego_pose = self.nusc.get('ego_pose', next_camera_data['ego_pose_token'])['translation']
            waypoints.append(next_ego_pose[:2]) # Only x, y
            current_sample = next_sample
        
        waypoints = torch.tensor(waypoints).float()  # 10 x 2

        return {
                    "image": torch_image,
                    "lidar": torch_pointcloud,
                    "waypoints": waypoints
                }