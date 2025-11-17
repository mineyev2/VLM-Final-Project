import os, sys
from termcolor import colored
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class NuScenesDataset(Dataset):
    def __init__(self, version, dataroot, tokenizer, prompt_part1, prompt_part2, nsweeps=5):
        """
        Initialize the NuScenes dataset for use with PyTorch.

        Args:
            version (str): Version of the NuScenes dataset to use (e.g., 'v1.0-mini', 'v1.0-trainval').
            dataroot (str): Root directory where the NuScenes dataset is stored.
            tokenizer (Tokenizer): Text tokenizer to use for the dataset.
            prompt_part1 (str): First part of the prompt to use for the dataset.
            prompt_part2 (str): Second part of the prompt to use for the dataset.
            nsweeps (int): Number of LiDAR sweeps to combine for each sample.
        """

        print(colored(f"Initializing NuScenes dataset with version {version} at {dataroot}", "cyan"))
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        self.nsweeps = nsweeps # TODO: Figure out this value

        self.tokenizer = tokenizer
        self.prompt_part1 = prompt_part1
        self.prompt_part2 = prompt_part2

        # Rewrite so all samples are in a list, for easy indexing
        self.sample_tokens = []

        print(colored("Loading sample tokens from all scenes into a list...", "cyan"))
        for scene in self.nusc.scene:

            # Get number of samples
            nbr_samples = scene['nbr_samples']

            # Ensure we have at least 2 previous frames and 10 future frames
            # Total frames needed per sample: 1 (current) + 2 (history) + 10 (future) = 13
            if nbr_samples < 13:
                continue

            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)

            # Skip the first two samples of each scene as they don't have enough history
            for _ in range(2):
                sample = self.nusc.get('sample', sample['next'])

            # Add the remaining valid samples
            for _ in range(nbr_samples - 12):
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
                - 'ego_positions': Last 3 ego vehicle XY positions as a torch tensor (3, 2).
                    - NOTE: This is used to get the historical trajectory of the ego vehicle.
                - 'waypoints': 10 future waypoints as a torch tensor (10, 2).
                    - NOTE: car location is retrieved from its position of when
                            the image was taken, rather than when the LIDAR was taken.
                            This isn't a huge difference, but worthy of note.
        """

        # Get sample at idx
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        # Get Last 3 Ego Vehicle XY Positions
        ego_positions = []
        current_sample = sample
        # Get current and two previous positions
        for _ in range(3):
            cam_data = self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_positions.append(ego_pose['translation'][:2]) # Append [x, y]
            # Move to the previous sample
            if current_sample['prev']:
                current_sample = self.nusc.get('sample', current_sample['prev'])
            else:
                # Pad with the oldest available position if at start of scene
                while len(ego_positions) < 3:
                    ego_positions.append(ego_positions[-1])

        ego_positions.reverse() # Order from oldest to newest
        ego_positions = torch.tensor(ego_positions).float() # 3 x 2

        # Get LiDAR point cloud with nsweeps
        nuscenes_pointcloud, _ = LidarPointCloud.from_file_multisweep(
            self.nusc,
            sample,
            chan='LIDAR_TOP',
            ref_chan='LIDAR_TOP',
            nsweeps=self.nsweeps, # TODO: This isn't working right now
            min_distance=1.0  # Filter out points closer than 1 meter
        )
        torch_pointcloud = torch.from_numpy(nuscenes_pointcloud.points.T).float() # [N,4]

        # Get front camera image
        camera_token = sample['data']['CAM_FRONT']
        camera_data = self.nusc.get('sample_data', camera_token)
        image_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        image = Image.open(image_path).convert('RGB')  # H x W x 3

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

        # Prepare Text and Tokenize
        pos_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_positions])
        prompt = f"{self.prompt_part1}[{pos_str}]\n{self.prompt_part2}"
        
        wp_str = ", ".join([f"[{wp[0]:.2f}, {wp[1]:.2f}]" for wp in waypoints])
        target_string = "Future Trajectory: " + wp_str

        full_text = prompt + target_string
        input_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.squeeze(0)
        
        labels = input_ids.clone()
        
        # Mask out the prompt part of the labels
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_length = prompt_tokens.shape[1]

        # The PyTorch nn.CrossEntropyLoss function has a parameter called ignore_index, which is set to -100 by default.
        # When it calculates the loss, it completely skips any position where the label is -100.
        labels[:prompt_length] = -100

        return {
            "image": image,
            "lidar": torch_pointcloud,
            "input_ids": input_ids,
            "labels": labels
        }
