from torch.nn.utils.rnn import pad_sequence
import numpy as np
import re

def collate_fn(batch, tokenizer_pad_id, training=True):
    """
    Collate function for batching dataset samples.
    
    Args:
        batch: List of dataset samples
        tokenizer_pad_id: Padding token ID
    
    Returns:
        dict with batched tensors
    """
    images = [item.get("image") for item in batch]
    input_ids = [item.get("input_ids") for item in batch]
    labels = [item.get("labels") for item in batch]

    if not training:
        text_attention_masks = [item.get("text_attention_mask") for item in batch]
        text_attention_masks_padded = pad_sequence(text_attention_masks, batch_first=True, padding_value=0)

        # # Load gt_waypoints, ego_positions
        # gt_waypoints = [item.get("waypoints") for item in batch]
        # ego_positions = [item.get("ego_positions") for item in batch]

        # convert gt_waypoints and ego_positions to numpy matrix
        gt_waypoints = [np.array(item.get("waypoints"), dtype=float)[:, :2] for item in batch] # just do x, y coords
        ego_positions = [np.array(item.get("ego_positions"), dtype=float) for item in batch]
    
    # LiDAR point clouds (list of tensors, variable size)
    lidar_list = [item.get("lidar", item.get("point_cloud", None)) for item in batch]
    
    # Pad text sequences so they are the same length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer_pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    if not training:
        return {
            "images": images,
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "lidar": lidar_list,
            "text_attention_masks": text_attention_masks_padded,
            "waypoints": gt_waypoints,
            "ego_positions": ego_positions,
        }
        
    else:
        return {
            "images": images,
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "lidar": lidar_list,
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
