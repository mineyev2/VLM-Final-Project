from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import os, sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from dataloader import NuScenesDataset

def main():
    # Getting LiDAR point clouds
    dataroot = os.path.join(os.path.dirname(__file__), "..", "datasets", "NuScenes")
    dataloader = NuScenesDataset(version='v1.0-mini', dataroot=dataroot, nsweeps=5)

    print(f"Dataset length: {len(dataloader)}")
    sample = dataloader[0]

    pointcloud = sample["lidar"].numpy()
    image = sample["image"]
    waypoints = sample["waypoints"].numpy()

    print(f"Pointcloud shape: {pointcloud.shape}") # (4, N)
    print(f"Image shape: {image.shape}") # (H, W, 3)
    print(f"Waypoints shape: {waypoints.shape}") # (10, 2)

    # Plot waypoints
    plt.scatter(waypoints[:, 0], waypoints[:, 1], s=10, c='g')
    plt.savefig("tests/test_images/test_waypoints.png")
    print("Saved test images to tests/test_images/")

    # Plot pointcloud
    plt.scatter(pointcloud[0, :], pointcloud[1, :], s=1, c='b')
    plt.axis('equal')
    plt.savefig("tests/test_images/test_pointcloud.png")
    print("Saved test images to tests/test_images/")

    # Plot image
    plt.imshow(image.numpy().astype("uint8"))
    plt.axis('off')
    plt.savefig("tests/test_images/test_image.png")
    print("Saved test images to tests/test_images/")

if __name__ == "__main__":
    main()