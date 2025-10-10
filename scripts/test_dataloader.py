from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import os, sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from dataloader import NuScenesDataset

def main():
    # dataset_root = os.path.join(os.path.dirname(__file__), "..", "datasets", "NuScenes")
    # nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=True)

    # """
    # Data we want for project:
    # 1) LiDAR point clouds
    # 2) Front camera images (I think this is all we will use for now)
    # 3) For now, we will ignore other stuff.
    # """
    # print(f"Number of scenes: {len(nusc.scene)}") 
    # my_scene = nusc.scene[2] # 20 second snippet (at 2 Hz) of car driving
    # print("Number of samples in scene:", my_scene['nbr_samples']) # Number of frames in scene
    # first_sample_token = my_scene['first_sample_token'] # First frame in scene
    # my_sample = nusc.get('sample', first_sample_token) # Get data given token
    # lidar_token = my_sample['data']['LIDAR_TOP'] # Token for LiDAR data
    # print(lidar_token)

    # nusc.render_sample_data(
    #     lidar_token,
    #     nsweeps=5,
    #     box_vis_level=3,
    #     underlay_map=False,
    #     out_path="test_dataloader.png"  # save the rendered plot to this file
    # )

    # Getting LiDAR point clouds
    dataroot = os.path.join(os.path.dirname(__file__), "..", "datasets", "NuScenes")
    dataloader = NuScenesDataset(version='v1.0-mini', dataroot=dataroot, nsweeps=5)

    print(f"Dataset length: {len(dataloader)}")
    pointcloud, image = dataloader[0]
    # Convert pointcloud from torch tensor to numpy
    pointcloud = pointcloud.numpy()
    print(f"Pointcloud shape: {pointcloud.shape}") # (4, N)
    print(f"Image shape: {image.shape}") # (H, W, 3
    print(f"Pointcloud type: {type(pointcloud)}")
    
    plt.figure(figsize=(10,10))
    plt.scatter(pointcloud[0, :], pointcloud[1, :], c=pointcloud[3, :], s=0.5, cmap='viridis')  # X vs Y
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    plt.colorbar(label='Intensity')
    plt.title('LiDAR Top-Down View')
    plt.savefig("./tests/test_images/lidar_topdown.png", dpi=200)
    plt.close()

    # # Plot image
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.title('Front Camera Image')
    # plt.savefig("./tests/test_images/front_camera.png", dpi=200)
    # plt.close()

if __name__ == "__main__":
    main()