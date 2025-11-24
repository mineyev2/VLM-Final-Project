import argparse
import os
import re
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from termcolor import colored

from src.models.qwen_clip_model import QwenCLIPModel
from src.utils.utils import ProjectWorldToImage, OffsetTrajectory3D
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from pyquaternion import Quaternion


def parse_coords_from_text(text, max_points=10):
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
        if len(pairs) >= max_points:
            break

    return np.array(pairs, dtype=float) if len(pairs) > 0 else np.array([], dtype=float).reshape(0, 2)


def visualize_trajectories(image_pil, gt_waypoints_2d, pred_waypoints_2d, cam_to_ego, ego_to_world, idx, output_dir):
    """Overlay ground truth and predicted trajectories on the image.
    
    Args:
        image_pil: PIL Image
        gt_waypoints_2d: ground truth waypoints in world coordinates (N, 2) with z=0
        pred_waypoints_2d: predicted waypoints in world coordinates (N, 2) with z=0
        cam_to_ego: camera calibration data
        ego_to_world: ego pose data
        idx: sample index for filename
        output_dir: directory to save visualization
    """
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Convert 2D waypoints to 3D (add z=0)
    gt_waypoints_3d = np.hstack([gt_waypoints_2d, np.zeros((len(gt_waypoints_2d), 1))])
    pred_waypoints_3d = np.hstack([pred_waypoints_2d, np.zeros((len(pred_waypoints_2d), 1))])
    
    # Filter out NaN predictions
    valid_pred_mask = ~np.isnan(pred_waypoints_3d).any(axis=1)
    pred_waypoints_3d_valid = pred_waypoints_3d[valid_pred_mask]
    
    try:
        # Project GT waypoints to image
        gt_points_img = ProjectWorldToImage(gt_waypoints_3d.tolist(), cam_to_ego, ego_to_world)
        
        # Draw GT trajectory with polygon (green)
        if len(gt_waypoints_3d) > 1:
            gt_left_3d = OffsetTrajectory3D(gt_waypoints_3d, -1.73 / 2)
            gt_right_3d = OffsetTrajectory3D(gt_waypoints_3d, 1.73 / 2)
            gt_left_img = ProjectWorldToImage(gt_left_3d.tolist(), cam_to_ego, ego_to_world)
            gt_right_img = ProjectWorldToImage(gt_right_3d.tolist(), cam_to_ego, ego_to_world)
            
            # Draw GT polygon
            gt_polygon = np.vstack((np.array(gt_left_img), np.array(gt_right_img)[::-1])).astype(np.int32)
            if gt_polygon.size > 0:
                frame_gt = np.zeros_like(img)
                cv2.fillPoly(frame_gt, [gt_polygon], color=(0, 255, 0))  # Green
                mask_gt = frame_gt.astype(bool)
                img[mask_gt] = cv2.addWeighted(img, 0.5, frame_gt, 0.5, 0)[mask_gt]
        
        # Draw GT waypoints as circles
        for pt in gt_points_img:
            cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 255, 0), thickness=-1)
        
        # Project predicted waypoints to image (if valid)
        if len(pred_waypoints_3d_valid) > 0:
            pred_points_img = ProjectWorldToImage(pred_waypoints_3d_valid.tolist(), cam_to_ego, ego_to_world)
            
            # Draw predicted trajectory with polygon (orange)
            if len(pred_waypoints_3d_valid) > 1:
                pred_left_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, -1.73 / 2)
                pred_right_3d = OffsetTrajectory3D(pred_waypoints_3d_valid, 1.73 / 2)
                pred_left_img = ProjectWorldToImage(pred_left_3d.tolist(), cam_to_ego, ego_to_world)
                pred_right_img = ProjectWorldToImage(pred_right_3d.tolist(), cam_to_ego, ego_to_world)
                
                # Draw predicted polygon
                pred_polygon = np.vstack((np.array(pred_left_img), np.array(pred_right_img)[::-1])).astype(np.int32)
                if pred_polygon.size > 0:
                    frame_pred = np.zeros_like(img)
                    cv2.fillPoly(frame_pred, [pred_polygon], color=(0, 125, 255))  # Orange
                    mask_pred = frame_pred.astype(bool)
                    img[mask_pred] = cv2.addWeighted(img, 0.5, frame_pred, 0.5, 0)[mask_pred]
            
            # Draw predicted waypoints as circles
            for pt in pred_points_img:
                cv2.circle(img, tuple(pt.astype(int)), radius=8, color=(0, 125, 255), thickness=-1)
        
        # Add legend
        cv2.putText(img, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Orange: Predicted', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 255), 2)
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'vis_sample_{idx:04d}.jpg')
        cv2.imwrite(output_path, img)
        
    except Exception as e:
        print(f"Visualization failed for sample {idx}: {e}")


class EvalNuScenes:
    def __init__(self, version, dataroot, prompt_part1, prompt_part2, nsweeps=5):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.nsweeps = nsweeps
        self.prompt_part1 = prompt_part1
        self.prompt_part2 = prompt_part2
        
        # Build sample list
        self.sample_tokens = []
        for scene in self.nusc.scene:
            nbr_samples = scene['nbr_samples']
            if nbr_samples < 20:
                continue
            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)
            # Skip first 2 to ensure we have some history available
            for _ in range(9):
                sample = self.nusc.get('sample', sample['next'])
            # Stop before the end to ensure we have future ground truth
            for _ in range(nbr_samples - 19):
                self.sample_tokens.append(sample['token'])
                sample = self.nusc.get('sample', sample['next'])

    def __len__(self):
        return len(self.sample_tokens)

    def get_item(self, idx):
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        # 1. Get Current Ego Pose (Reference Frame)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        ego_pose_curr = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Store translation and rotation for transforming back later
        ego_trans = np.array(ego_pose_curr['translation'])
        ego_rot = Quaternion(ego_pose_curr['rotation'])

        # 2. Get History (Global Coords)
        history_points_global = []
        current_sample = sample
        for _ in range(10): # Using 3 frames of history as per your previous logic
            c_data = self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
            e_pose = self.nusc.get('ego_pose', c_data['ego_pose_token'])
            history_points_global.append(np.array(e_pose['translation']))
            
            if current_sample['prev']:
                current_sample = self.nusc.get('sample', current_sample['prev'])
            else:
                history_points_global.append(history_points_global[-1])
        history_points_global.reverse()

        # 3. Get Future Ground Truth (Global Coords)
        future_points_global = []
        current_sample = sample
        for _ in range(10):
            if current_sample['next'] == '':
                break
            next_sample_token = current_sample['next']
            next_sample = self.nusc.get('sample', next_sample_token)
            next_cam_data = self.nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
            next_ego_pose = self.nusc.get('ego_pose', next_cam_data['ego_pose_token'])
            future_points_global.append(np.array(next_ego_pose['translation']))
            current_sample = next_sample
            
        # Pad if necessary
        while len(future_points_global) < 10:
             future_points_global.append(future_points_global[-1] if len(future_points_global) > 0 else ego_trans)

        # 4. Transform to Local Coordinates (Egocentric)
        history_local = []
        for p in history_points_global:
            diff = p - ego_trans
            local_p = ego_rot.inverse.rotate(diff)
            history_local.append(local_p[:2]) # Keep only x,y

        future_local = []
        for p in future_points_global:
            diff = p - ego_trans
            local_p = ego_rot.inverse.rotate(diff)
            future_local.append(local_p[:2]) # Keep only x,y

        # Image loading
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(image_path).convert('RGB')
        
        # Camera Calibration info
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        return {
            'image': image,
            'ego_positions_local': history_local, # Input to model
            'waypoints_local': np.array(future_local, dtype=float), # GT for metrics
            'waypoints_global': np.array(future_points_global, dtype=float), # Keep for Visualization
            'ego_translation': ego_trans, # For Viz
            'ego_rotation': ego_rot, # For Viz
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


def load_checkpoint_into_model(model, ckpt_path, device):
    """
    For now, only 2 options:
    1) Model is just glue layer
    2) Model is full checkpoint with projector + vision tower + language model

    So only checking for these two cases in here
    """
    data = torch.load(ckpt_path, map_location=device)

    # try:
    #     model.mlp_projector.load_state_dict(data)
    #     print(colored(f"Loaded projector state_dict from {ckpt_path}", "green"))
    # except Exception as e:
    #     print(colored(f"Warning: failed to load projector state_dict: {e}", "red"))
    # return



    if 'language_model_state_dict' in data:
        try:
            model.language_model.load_state_dict(data['language_model_state_dict'], strict=False)
            model.vision_tower.load_state_dict(data['vision_tower_state_dict'])
            model.mlp_projector.load_state_dict(data['model_state_dict'])
            print(colored("Loaded full checkpoint (projector + vision tower + language model).", "green"))
        except Exception as e:
            print(colored(f"Warning loading model: {e}", "red"))

    else:
        try:
            model.mlp_projector.load_state_dict(data)
            print(colored(f"Loaded projector state_dict from {ckpt_path}", "green"))
        except Exception as e:
            print(colored(f"Warning: failed to load projector state_dict: {e}", "red"))


    # If it's a raw state dict for projector (train saved that as checkpoint_latest.pth)
    # if all(isinstance(k, str) for k in data.keys()) and any('weight' in k or 'bias' in k for k in data.keys()):
    #     try:
    #         model.mlp_projector.load_state_dict(data)
    #         print(f"Loaded projector state_dict from {ckpt_path}")
    #     except Exception as e:
    #         print(f"Warning: failed to load projector state_dict: {e}")
    #     return

    # # If it's the 'final_model.pth' dict saved by train_v3
    # if isinstance(data, dict) and ('model_state_dict' in data or 'vision_tower_state_dict' in data or 'language_model_state_dict' in data):
    #     # projector
    #     if 'model_state_dict' in data:
    #         try:
    #             model.mlp_projector.load_state_dict(data['model_state_dict'])
    #             print("Loaded mlp_projector from final checkpoint.")
    #         except Exception as e:
    #             print(f"Warning loading mlp_projector: {e}")
    #     # vision tower
    #     if 'vision_tower_state_dict' in data:
    #         try:
    #             model.vision_tower.load_state_dict(data['vision_tower_state_dict'])
    #             print("Loaded vision_tower weights.")
    #         except Exception as e:
    #             print(f"Warning loading vision_tower: {e}")
    #     # language model
    #     if 'language_model_state_dict' in data:
    #         try:
    #             # try non-strict to avoid mismatch
    #             model.language_model.load_state_dict(data['language_model_state_dict'], strict=False)
    #             print("Loaded (partial/soft) language_model weights.")
    #         except Exception as e:
    #             print(f"Warning loading language_model: {e}")
    #     return

    # print("Unrecognized checkpoint format, attempting to load as state_dict into projector...")
    # try:
    #     model.mlp_projector.load_state_dict(data)
    #     print("Loaded projector state_dict (fallback).")
    # except Exception as e:
    #     print(f"Failed fallback loading: {e}")




def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Clear GPU cache before loading
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    model = QwenCLIPModel(device, qwen_model_name=args.llm, checkpoint_path=args.checkpoint)
    # Load checkpoint
    if args.checkpoint is not None:
        print(colored(f"Loading model weights from {args.checkpoint}...", "white"))
        
        # Clear cache again before loading checkpoint weights
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # load_checkpoint_into_model(model, args.checkpoint, device)

    model.eval()

    # Loss function for computing cross-entropy (same as training)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    # dataset for evaluation
    # TODO: Add LiDAR here
    ds = EvalNuScenes(args.version, args.dataroot, model.prompt_part1, model.prompt_part2)

    results = []

    n_samples = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))
    
    # Select random samples for visualization
    vis_indices = set(np.random.choice(n_samples, size=min(args.num_vis, n_samples), replace=False)) if args.num_vis > 0 else set()
    vis_dir = os.path.join(args.output_dir, 'visualizations')

    # Process in batches
    batch_size = args.batch_size
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc='Evaluating'):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        # Collect batch data
        batch_items = [ds.get_item(idx) for idx in batch_indices]
        batch_images = [item['image'] for item in batch_items]
        batch_ego_positions_local = [item['ego_positions_local'] for item in batch_items]
        batch_gt_waypoints_local = [item['waypoints_local'] for item in batch_items]
        batch_gt_waypoints_global = [item['waypoints_global'] for item in batch_items] 
        batch_ego_trans = [item['ego_translation'] for item in batch_items]
        batch_ego_rot = [item['ego_rotation'] for item in batch_items]
        batch_cam_to_ego = [item['cam_to_ego'] for item in batch_items]
        batch_ego_to_world = [item['ego_to_world'] for item in batch_items]
        
        # Process images
        pixel_values = model.image_processor(images=batch_images, return_tensors='pt').pixel_values.to(device)
        
        # Convert ego_positions to Python float lists
        batch_ego_positions_py = [[[float(x), float(y)] for (x, y) in ego_pos] for ego_pos in batch_ego_positions_local]
        
        # === Compute Cross-Entropy Loss (batched) ===
        batch_ce_losses = []
        batch_token_accs = []
        batch_perplexities = []
        
        try:
            with torch.no_grad():
                # Prepare batch of texts
                batch_prompts = []
                batch_full_texts = []
                for ego_pos, gt_wp in zip(batch_ego_positions_local, batch_gt_waypoints_local):
                    pos_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_pos])
                    prompt = f"{model.prompt_part1}[{pos_str}]\n{model.prompt_part2}"
                    wp_str = ", ".join([f"[{wp[0]:.2f}, {wp[1]:.2f}]" for wp in gt_wp])
                    target_string = "Future Trajectory: " + wp_str
                    full_text = prompt + target_string
                    batch_prompts.append(prompt)
                    batch_full_texts.append(full_text)
                
                # Tokenize batch with padding
                batch_inputs = model.tokenizer(batch_full_texts, return_tensors="pt", padding=True).to(device)
                batch_input_ids = batch_inputs.input_ids
                
                # Create labels for each sample
                batch_labels = batch_input_ids.clone()
                for i, prompt in enumerate(batch_prompts):
                    prompt_tokens = model.tokenizer(prompt, return_tensors="pt").input_ids
                    prompt_length = prompt_tokens.shape[1]
                    batch_labels[i, :prompt_length] = -100
                
                # Forward pass
                batch_logits = model(pixel_values, batch_input_ids)
                
                # Compute loss per sample
                num_image_patches = batch_logits.shape[1] - batch_labels.shape[1]
                logits_for_loss = batch_logits[:, num_image_patches:-1, :]
                labels_for_loss = batch_labels[:, 1:]
                
                for i in range(len(batch_indices)):
                    sample_logits = logits_for_loss[i:i+1]
                    sample_labels = labels_for_loss[i:i+1]
                    
                    losses = loss_fn(
                        sample_logits.reshape(-1, sample_logits.size(-1)),
                        sample_labels.reshape(-1)
                    )
                    
                    valid_losses = losses[sample_labels.reshape(-1) != -100]
                    if len(valid_losses) > 0:
                        ce_loss = valid_losses.mean().item()
                        batch_ce_losses.append(ce_loss)
                        batch_perplexities.append(np.exp(ce_loss))
                        
                        # Token accuracy
                        preds = sample_logits.argmax(dim=-1)
                        valid_mask = sample_labels != -100
                        if valid_mask.sum() > 0:
                            correct = (preds == sample_labels) & valid_mask
                            token_acc = correct.sum().item() / valid_mask.sum().item()
                            batch_token_accs.append(token_acc)
                        else:
                            batch_token_accs.append(np.nan)
                    else:
                        batch_ce_losses.append(np.nan)
                        batch_perplexities.append(np.nan)
                        batch_token_accs.append(np.nan)
        except Exception as e:
            print(f"Loss computation failed for batch {batch_idx}: {e}")
            batch_ce_losses = [np.nan] * len(batch_indices)
            batch_perplexities = [np.nan] * len(batch_indices)
            batch_token_accs = [np.nan] * len(batch_indices)
        
        # === Generate trajectories (batched) ===
        try:
            outputs, gen_texts = model.generate_trajectory(pixel_values, batch_ego_positions_py)
        except Exception as e:
            print(f"Generation failed for batch {batch_idx}: {e}")
            continue
        
        # Process each sample in batch
        batch_time = time.time() - batch_start_time
        per_sample_time = batch_time / len(batch_indices)
        
        for i, idx in enumerate(batch_indices):
            gen_text = gen_texts[i]
            pred_coords_local = parse_coords_from_text(gen_text, max_points=10)
            
            # Print generated text for first few samples
            if idx < 5 or idx in vis_indices:
                print(colored(f"\n[Sample {idx}] Generated text:", "cyan"))
                print(gen_text)
                print(colored(f"Parsed waypoints: {pred_coords.shape[0]}/10", "yellow"))
            
            num_valid_waypoints = pred_coords.shape[0]
            format_compliant = (num_valid_waypoints == 10)
            
            if pred_coords.shape[0] < 10:
                pad = np.full((10 - pred_coords.shape[0], 2), np.nan)
                pred_coords = np.vstack([pred_coords, pad])
            elif pred_coords.shape[0] > 10:
                pred_coords = pred_coords[:10]
            
            # Coordinate metrics
            gt_waypoints_local = batch_gt_waypoints_local[i]
            diffs = pred_coords_local - gt_waypoints_local
            l2_per_waypoint = np.linalg.norm(diffs, axis=1)
            ade = np.nanmean(l2_per_waypoint)
            fde = l2_per_waypoint[-1]
            miss_rate_10m = 1.0 if fde > 10.0 else 0.0
            waypoint_errors = l2_per_waypoint.tolist()
            
            results.append({
                'idx': idx,
                'cross_entropy_loss': float(batch_ce_losses[i]),
                'perplexity': float(batch_perplexities[i]),
                'token_accuracy': float(batch_token_accs[i]),
                'num_valid_waypoints': int(num_valid_waypoints),
                'format_compliant': int(format_compliant),
                'ade': float(ade),
                'fde': float(fde),
                'miss_rate_10m': float(miss_rate_10m),
                'waypoint_errors': waypoint_errors,
                'processing_time_sec': float(per_sample_time),
                'gen_text': gen_text
            })
            
            # Visualize selected samples
            if idx in vis_indices:
                ego_rot = batch_ego_rot[i]
                ego_trans = batch_ego_trans[i]

                pred_coords_global = []
                for pt in pred_coords_local:
                    if np.isnan(pt).any():
                        pred_coords_global.append(pt)
                        continue
                    # 3D diff (x, y, 0)
                    diff_3d = np.array([pt[0], pt[1], 0.0])
                    # Rotate back to world
                    global_3d = ego_rot.rotate(diff_3d) + ego_trans
                    pred_coords_global.append(global_3d[:2])

                pred_coords_global = np.array(pred_coords_global)


                # Use Global GT and Global Preds for the image overlay
                visualize_trajectories(
                    batch_items[i]['image'],
                    batch_gt_waypoints_global[i][:, :2], # Ensure 2D
                    pred_coords_global,
                    batch_cam_to_ego[i],
                    batch_ego_to_world[i],
                    idx,
                    vis_dir
                )

    # aggregate metrics
    ce_losses = [r['cross_entropy_loss'] for r in results if not np.isnan(r['cross_entropy_loss'])]
    perplexities = [r['perplexity'] for r in results if not np.isnan(r['perplexity'])]
    token_accs = [r['token_accuracy'] for r in results if not np.isnan(r['token_accuracy'])]
    ades = [r['ade'] for r in results if not np.isnan(r['ade'])]
    fdes = [r['fde'] for r in results if not np.isnan(r['fde'])]
    miss_rates = [r['miss_rate_10m'] for r in results]
    format_compliance = [r['format_compliant'] for r in results]
    processing_times = [r['processing_time_sec'] for r in results]
    
    # Per-waypoint aggregate errors
    all_waypoint_errors = [r['waypoint_errors'] for r in results]
    waypoint_means = np.nanmean(all_waypoint_errors, axis=0).tolist() if len(all_waypoint_errors) > 0 else []

    summary = {
        'num_samples': len(results),
        # Loss metrics
        'cross_entropy_loss_mean': float(np.mean(ce_losses)) if len(ce_losses) > 0 else float('nan'),
        'cross_entropy_loss_std': float(np.std(ce_losses)) if len(ce_losses) > 0 else float('nan'),
        'perplexity_mean': float(np.mean(perplexities)) if len(perplexities) > 0 else float('nan'),
        'token_accuracy_mean': float(np.mean(token_accs)) if len(token_accs) > 0 else float('nan'),
        # Coordinate metrics
        'ade_mean': float(np.mean(ades)) if len(ades) > 0 else float('nan'),
        'ade_std': float(np.std(ades)) if len(ades) > 0 else float('nan'),
        'fde_mean': float(np.mean(fdes)) if len(fdes) > 0 else float('nan'),
        'fde_std': float(np.std(fdes)) if len(fdes) > 0 else float('nan'),
        'miss_rate_10m': float(np.mean(miss_rates)) if len(miss_rates) > 0 else float('nan'),
        'format_compliance_rate': float(np.mean(format_compliance)) if len(format_compliance) > 0 else float('nan'),
        # Timing metrics
        'avg_processing_time_sec': float(np.mean(processing_times)) if len(processing_times) > 0 else float('nan'),
        'total_processing_time_sec': float(np.sum(processing_times)) if len(processing_times) > 0 else float('nan'),
        'fps': float(1.0 / np.mean(processing_times)) if len(processing_times) > 0 and np.mean(processing_times) > 0 else float('nan'),
        # Per-waypoint breakdown
        'per_waypoint_errors_mean': waypoint_means
    }

    # save detailed CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_name)
    
    # Flatten waypoint_errors for CSV (convert list to individual columns)
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['idx', 'cross_entropy_loss', 'perplexity', 'token_accuracy', 
                     'num_valid_waypoints', 'format_compliant', 'ade', 'fde', 'miss_rate_10m',
                     'processing_time_sec'] + \
                     [f'wp{i}_error' for i in range(10)] + ['gen_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                'idx': r['idx'],
                'cross_entropy_loss': r['cross_entropy_loss'],
                'perplexity': r['perplexity'],
                'token_accuracy': r['token_accuracy'],
                'num_valid_waypoints': r['num_valid_waypoints'],
                'format_compliant': r['format_compliant'],
                'ade': r['ade'],
                'fde': r['fde'],
                'miss_rate_10m': r['miss_rate_10m'],
                'processing_time_sec': r['processing_time_sec'],
                'gen_text': r['gen_text']
            }
            # Add per-waypoint errors
            for i, err in enumerate(r['waypoint_errors']):
                row[f'wp{i}_error'] = err
            writer.writerow(row)
    
    # Save summary JSON
    import json
    summary_path = os.path.join(args.output_dir, args.output_name.replace('.csv', '_summary.json'))
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(colored('\n=== Evaluation Summary ===', 'cyan', attrs=['bold']))
    print(colored(f"Samples evaluated: {summary['num_samples']}", 'white'))
    print(colored('\nLoss Metrics:', 'yellow'))
    print(f"  Cross-Entropy Loss: {summary['cross_entropy_loss_mean']:.4f} ± {summary['cross_entropy_loss_std']:.4f}")
    print(f"  Perplexity: {summary['perplexity_mean']:.4f}")
    print(f"  Token Accuracy: {summary['token_accuracy_mean']:.4f}")
    print(colored('\nTrajectory Metrics:', 'yellow'))
    print(f"  Average Displacement Error (mean): {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}")
    print(f"  Final Displacement Error (mean): {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}")
    print(f"  Miss Rate @ 10m: {summary['miss_rate_10m']:.2%}")
    print(f"  Format Compliance: {summary['format_compliance_rate']:.2%}")
    print(colored(f'\nResults saved to: {csv_path}', 'green'))
    print(colored(f'Summary saved to: {summary_path}', 'green'))
    if len(vis_indices) > 0:
        print(colored(f'Visualizations saved to: {vis_dir} ({len(vis_indices)} samples)', 'green'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/storage/ice-shared/cs8803vlm/rmineyev3')
    parser.add_argument('--version', type=str, default='v1.0-test')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (final_model.pth or projector state dict)')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to evaluate (default 100 or all if -1)')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./eval_outputs', help='Base directory for evaluation outputs')
    parser.add_argument('--output_name', type=str, default='eval_results.csv')
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-3B-Instruct', help='LLM to use for evaluation')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of random samples to visualize (0 to disable)')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this evaluation run (used to keep track of ablations)')
    
    # Print args used
    args = parser.parse_args()
    
    # Append run_name to output_dir after parsing
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    
    print(colored("--- Evaluation Configuration ---", "cyan"))
    for k, v in vars(args).items():
        print(colored(f"{k}: {v}", "cyan"))
    print(colored("--------------------------", "cyan"))

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.num_samples == -1:
        args.num_samples = None
    evaluate(args)
