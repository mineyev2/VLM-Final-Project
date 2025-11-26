import argparse
import os
import re
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from termcolor import colored
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from nuscenes import NuScenes

class DarkScenesParsingDataset(Dataset):
    def __init__(self, version, dataroot):
        """
        Initialize the NuScenes dataset for use with PyTorch.
        Serves EGOCENTRIC (Local) coordinates with metadata for Global reconstruction.
        """
        print(colored(f"Initializing NuScenes dataset with version {version} at {dataroot}", "cyan"))
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Create list of samples
        self.samples = []
        num_hist_points = 10 # including current
        num_future_points = 10
        for scene in self.nusc.scene:
            nbr_samples = scene['nbr_samples']
            if nbr_samples < num_hist_points + num_future_points: 
                continue
            
            sample = self.nusc.get('sample', scene['first_sample_token'])
            
            # Skip first 9 samples to ensure 10 frames of history
            for _ in range(num_hist_points - 1):
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

        # 5. Load Image
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(image_path).convert('RGB')
        
        return {
            'image': image,
            'sample_token': sample['token']
        }
    

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single NuScenes sample")
    parser.add_argument('--sample_idx', type=int, required=True, help='Index of sample to evaluate')
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test', help='NuScenes version')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dataset = DarkScenesParsingDataset(
        version=args.version,
        dataroot=args.dataroot
    )

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample['image']

        
        
        # Show with matplotlib
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Sample Index: {idx}")
        plt.show()

        input("Press Enter to continue to the next sample...")