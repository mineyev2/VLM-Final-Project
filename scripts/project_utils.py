import os
import re
import cv2
import numpy as np
from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D


def parse_coords_from_text(text, max_points=10):
    """Extract waypoint coordinates from generated text."""
    # Try to find the specific section first
    trajectory_match = re.search(r'Future Trajectory:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    text_to_parse = trajectory_match.group(1) if trajectory_match else text
    
    # Parse numbers
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text_to_parse)
    nums = [float(x) for x in nums]
    
    # Group into pairs
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        pairs.append([nums[i], nums[i+1]])
        if len(pairs) >= max_points:
            break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)


def visualize_trajectories(image_pil, gt_waypoints_global, pred_waypoints_global, cam_to_ego, ego_to_world, idx, output_dir):
    """Overlay trajectories on the image using GLOBAL coordinates."""
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Convert 2D -> 3D (z=0) for projection
    # Note: Global Z is technically not 0, but usually we assume flat ground or rely on calibration if using true 3D.
    # For NuScenes flat bev projection, appending 0 to (x,y) might be inaccurate if ego is at high elevation.
    # Ideally, we should use the ego's Z, but here we stick to the common 2D-bev-to-image assumption
    # or assume input was 2D. 
    # Better approach for global: Use the ego vehicle Z for the trajectory points if not provided.
    
    # Simple assumption: Projection usually handles the extrinsic matrix which accounts for Ego Z.
    # If the points are purely (x,y) global map coords, we need a Z. 
    # Let's use the Ego's Z from the transform matrix if possible, or just 0 if we assume z-flat world.
    # Standard ProjectWorldToImage usually expects (x, y, z).
    
    # Let's extract Ego Z from ego_to_world translation to be safe, or just append 0.
    ego_z = ego_to_world['translation'][2]
    
    gt_3d = np.hstack([gt_waypoints_global, np.full((len(gt_waypoints_global), 1), ego_z)])
    pred_3d = np.hstack([pred_waypoints_global, np.full((len(pred_waypoints_global), 1), ego_z)])
    
    # Filter NaNs from prediction
    valid_mask = ~np.isnan(pred_3d).any(axis=1)
    pred_3d_valid = pred_3d[valid_mask]

    try:
        # 1. Draw Ground Truth (Green)
        gt_points_img = ProjectWorldToImage(gt_3d.tolist(), cam_to_ego, ego_to_world)
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=6, color=(0, 255, 0), thickness=-1)

        if len(gt_3d) > 1:
            gt_l = OffsetTrajectory3D(gt_3d, -0.9) 
            gt_r = OffsetTrajectory3D(gt_3d, 0.9)
            gt_l_img = ProjectWorldToImage(gt_l.tolist(), cam_to_ego, ego_to_world)
            gt_r_img = ProjectWorldToImage(gt_r.tolist(), cam_to_ego, ego_to_world)
            
            poly = np.vstack((np.array(gt_l_img), np.array(gt_r_img)[::-1])).astype(np.int32)
            if poly.size > 0:
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # 2. Draw Prediction (Orange)
        if len(pred_3d_valid) > 0:
            pred_points_img = ProjectWorldToImage(pred_3d_valid.tolist(), cam_to_ego, ego_to_world)
            for pt in pred_points_img:
                cv2.circle(img, tuple(pt.astype(int)), radius=6, color=(0, 125, 255), thickness=-1)

            if len(pred_3d_valid) > 1:
                p_l = OffsetTrajectory3D(pred_3d_valid, -0.9)
                p_r = OffsetTrajectory3D(pred_3d_valid, 0.9)
                p_l_img = ProjectWorldToImage(p_l.tolist(), cam_to_ego, ego_to_world)
                p_r_img = ProjectWorldToImage(p_r.tolist(), cam_to_ego, ego_to_world)
                
                poly = np.vstack((np.array(p_l_img), np.array(p_r_img)[::-1])).astype(np.int32)
                if poly.size > 0:
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [poly], (0, 125, 255))
                    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # Legend
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 125, 255), 2)

        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f'vis_{idx:04d}.jpg'), img)
    except Exception as e:
        print(f"Vis failed for {idx}: {e}")
    