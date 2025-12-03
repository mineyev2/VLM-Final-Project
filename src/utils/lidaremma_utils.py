import numpy as np
import re

def collate_fn(batch):
    """
    Collate function for batching dataset samples.
    
    Args:
        batch: List of dataset samples
    
    Returns:
        dict with batched tensors
    """
    # Images
    images = [item.get("image") for item in batch]

    # convert gt_waypoints and ego_positions to numpy matrix
    gt_waypoints = [np.array(item.get("waypoints"), dtype=float)[:, :2] for item in batch] # just do x, y coords
    ego_positions = [np.array(item.get("ego_positions"), dtype=float) for item in batch]
    
    # LiDAR point clouds (list of tensors, variable size)
    lidar_list = [item.get("lidar", item.get("point_cloud", None)) for item in batch]

    # Prompt and Target
    prompt = [item.get("prompt", "") for item in batch]
    target_text = [item.get("target_text", "") for item in batch]
    
    return {
        "prompt": prompt,
        "target_text": target_text,
        "images": images,
        "lidar": lidar_list,
        "waypoints": gt_waypoints,
        "ego_positions": ego_positions,
    }

def parse_coords_from_text(text):
    """Extract waypoint coordinates from generated text.
    
    Returns:
        numpy array of shape (N, 2) where N <= max_points
    """
    # Try to extract only from "Future Trajectory:" section to avoid parsing ego positions
    trajectory_match = re.search(r'Future Trajectory:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    
    if trajectory_match:
        text_to_parse = trajectory_match.group(1)
    else:
        # Fallback: use entire text if pattern not found
        text_to_parse = text
    
    # find all floats/ints in text and group into pairs
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text_to_parse)
    nums = [float(x) for x in nums]
    
    # group into pairs
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        # if len(pairs) >= max_points:
        #     break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)
