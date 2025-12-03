import os
from termcolor import colored
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

NUSCENES_INTENSITY_MAX = 255.0

class NuScenesDataset(Dataset):
    def __init__(
        self,
        version,
        dataroot,
        prompt_part1,
        prompt_part2,
        nsweeps=5,
        min_dist: float = 0.5,
        output_lidar=False):
        """
        Initialize the NuScenes dataset for use with PyTorch.
        Serves EGOCENTRIC (Local) coordinates with metadata for Global reconstruction.
        """
        print(colored(f"Initializing NuScenes dataset with version {version} at {dataroot}", "cyan"))
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.nsweeps = nsweeps 
        self.output_lidar = output_lidar

        self.min_dist = min_dist
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

        # =============================================================================
        # 1. Current Ego Pose
        # =============================================================================
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        # =============================================================================
        # 2. Get History (10 frames)
        # =============================================================================
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

        # =============================================================================
        # 3. Get Future (10 frames)
        # =============================================================================
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

        # =============================================================================
        # 4. Load Image
        # =============================================================================
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(image_path).convert('RGB') # TODO: Does LidarCLIP assume RGB or BGR?

        torch_pointcloud = None

        if self.output_lidar:
            # NOTE: Code below copied from LidarCLIP

            # Load necessary tokens/filenames
            pointsensor_token = sample['data']['LIDAR_TOP']
            pointsensor = self.nusc.get("sample_data", pointsensor_token)
            lidar_path = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['filename'])
            nuscenes_pointcloud = LidarPointCloud.from_file(
                lidar_path
            )

            # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = self.nusc.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
            nuscenes_pointcloud.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            nuscenes_pointcloud.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            nuscenes_pointcloud.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            nuscenes_pointcloud.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
            nuscenes_pointcloud.translate(-np.array(poserecord["translation"]))
            nuscenes_pointcloud.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            nuscenes_pointcloud.translate(-np.array(cs_record["translation"]))
            nuscenes_pointcloud.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = nuscenes_pointcloud.points[2, :]

            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                nuscenes_pointcloud.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True
            )

            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            # Also make sure points are at least min_dist in front of the camera to avoid seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            w_og, h_og = image.size
            left_border = w_og // 2 - h_og // 2
            right_border = w_og // 2 + h_og // 2
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > self.min_dist)
            mask = np.logical_and(mask, points[0, :] > left_border)
            mask = np.logical_and(mask, points[0, :] < right_border)
            mask = np.logical_and(mask, points[1, :] >= 0)
            mask = np.logical_and(mask, points[1, :] <= h_og)

            torch_pointcloud = torch.as_tensor(nuscenes_pointcloud.points[:, mask].T)
            # shift from cam coords to KITTI style (x-forward, y-left, z-up)
            torch_pointcloud = torch_pointcloud[:, (2, 0, 1, 3)]
            torch_pointcloud[:, 1] = -torch_pointcloud[:, 1]
            torch_pointcloud[:, 2] = -torch_pointcloud[:, 2]
            torch_pointcloud[:, 3] /= NUSCENES_INTENSITY_MAX
        
        # =============================================================================
        # 6. Prepare GT Text & Input IDs
        # =============================================================================
        pos_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in history_points])
        prompt = f"{self.prompt_part1}[{pos_str}]\n{self.prompt_part2}"
        
        wp_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in future_points])
        target_text = "Future Trajectory: [" + wp_str + "]"
        
        # Calibration
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # =============================================================================
        # 7. Return Data
        # =============================================================================
        return {
            'prompt': prompt,
            'target_text': target_text,
            'image': image,
            'lidar': torch_pointcloud,
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
        