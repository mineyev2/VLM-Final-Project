#!/usr/bin/env python3
"""
Single-sample evaluation for LiDAR-EMMA (LiDAR-MR).

Outputs for a given sample index:
  1) Front camera image with GT & predicted waypoints overlaid.
  2) LiDAR BEV image with GT & predicted waypoints overlaid.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from termcolor import colored

from nuscenes.nuscenes import NuScenes
from PIL import Image

# Project imports
from src.models.lidar_emma import LidarEMMA
from scripts.nuscenes_dataset import NuScenesDataset
from src.utils.lidaremma_utils import parse_coords_from_text

# Reuse visualization utils from the image-EMMA single-sample script
# (these already know how to use cam_to_ego, ego_to_world, and ego_pose)
from evaluate_single_sample import visualize_trajectory, visualize_trajectory_bev


# ------------------------------------------------------------
# Model loading (adapted from evaluate_lidaremma.py)
# ------------------------------------------------------------
def load_model(args, device):
    """
    Create a LidarEMMA instance and load checkpoint weights.
    """
    model = LidarEMMA(
        device=device,
        llm=args.llm,
        freeze_encoders=True,
        freeze_llm=True,
        use_lidar=args.use_lidar,
        lidar_pooling=False,   # same as dataset eval
    )

    checkpt = torch.load(args.checkpoint, map_location=device)

    if "vision_projector_state_dict" in checkpt:
        model.vision_projector.load_state_dict(checkpt["vision_projector_state_dict"])
        print("✓ Vision Projector loaded")

    if "lidar_projector_state_dict" in checkpt and hasattr(model, "lidar_projector"):
        model.lidar_projector.load_state_dict(checkpt["lidar_projector_state_dict"])
        print("✓ LiDAR Projector loaded")

    if "vision_encoder_state_dict" in checkpt:
        model.vision_tower.load_state_dict(checkpt["vision_encoder_state_dict"])
        print("✓ Vision Encoder loaded")

    if "lidar_encoder_state_dict" in checkpt:
        model.lidar_encoder.load_state_dict(checkpt["lidar_encoder_state_dict"])
        print("✓ LiDAR Encoder loaded")

    if "llm_state_dict" in checkpt:
        model.language_model.load_state_dict(checkpt["llm_state_dict"])
        print("✓ LLM loaded")

    model.eval()
    return model


# ------------------------------------------------------------
# Main single-sample evaluation
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-sample evaluation for LiDAR-EMMA")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LiDAR-EMMA checkpoint (e.g., final_model.pth)",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/storage/ice-shared/cs8803vlm/rmineyev3",
        help="nuScenes dataroot",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-test",
        help="nuScenes version, e.g. v1.0-trainval or v1.0-test",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample in NuScenesDataset to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./single_sample_outputs",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Qwen model name used in training LiDAR-EMMA",
    )
    parser.add_argument(
        "--use_lidar",
        action="store_true",
        help="Use LiDAR input (should be True for LiDAR-EMMA)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(colored(f"\nUsing device: {device}", "cyan"))

    # ----------------- Load model -----------------
    print(colored("\nLoading model...", "yellow"))
    model = load_model(args, device)
    print(colored("\n======================================================================", "cyan"))
    print(colored("Model initialization complete!", "cyan", attrs=["bold"]))
    print(colored("======================================================================\n", "cyan"))

    # ----------------- Load dataset -----------------
    print(colored("Loading NuScenesDataset...", "yellow"))
    ds = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2,
        output_lidar=args.use_lidar,
    )
    print(colored(f"✓ Dataset loaded: {len(ds)} samples", "green"))

    if args.sample_idx < 0 or args.sample_idx >= len(ds):
        raise IndexError(f"sample_idx {args.sample_idx} out of range [0, {len(ds)-1}]")

    # Use same sample index from the dataset
    item = ds[args.sample_idx]

    # Expected keys from NuScenesDataset:
    image = item["image"]                         # PIL Image
    gt_waypoints = np.array(item["waypoints"])    # (10, 2) global XY
    ego_positions = np.array(item["ego_positions"])
    prompt_text = item["prompt"]                  # already formatted history prompt
    lidar_data = item.get("lidar", None)          # may be None if output_lidar=False

    print(colored(f"\nEvaluating sample index: {args.sample_idx}", "magenta"))

    # ----------------- Prepare model inputs -----------------
    # Image → pixel_values
    pixel_values = model.image_processor(
        images=[image], return_tensors="pt"
    ).pixel_values.to(device)

    # LiDAR → point_clouds (list of torch Tensors or None)
    point_clouds = None
    if args.use_lidar and lidar_data is not None:
        if isinstance(lidar_data, torch.Tensor):
            point_clouds = [lidar_data.to(device)]
        elif isinstance(lidar_data, (list, tuple)):
            pcs = [pc.to(device) for pc in lidar_data if pc is not None]
            point_clouds = pcs if len(pcs) > 0 else None

    # ----------------- Generate trajectory -----------------
    print(colored("\nGenerating trajectory with LiDAR-EMMA...", "yellow"))
    with torch.no_grad():
        gen_texts = model.generate_trajectory(
            prompt=[prompt_text],  # expect list[str]
            images=pixel_values,
            point_clouds=point_clouds,
        )
        gen_text = gen_texts[0]

    print(colored("\nGenerated text:", "yellow"))
    print(gen_text)

    # ----------------- Parse predicted waypoints -----------------
    pred_coords = parse_coords_from_text(gen_text)   # global XY
    num_valid_waypoints = pred_coords.shape[0]
    print(colored(f"\nParsed {num_valid_waypoints} predicted waypoints.", "cyan"))

    # For metrics & visualization, keep up to 10, pad with NaNs if fewer
    if num_valid_waypoints < 10:
        pad = np.full((10 - num_valid_waypoints, 2), np.nan)
        if num_valid_waypoints > 0:
            pred_coords = np.vstack([pred_coords, pad])
        else:
            pred_coords = pad
    elif num_valid_waypoints > 10:
        pred_coords = pred_coords[:10]

    # ----------------- ADE / FDE -----------------
    gt_wp = gt_waypoints
    diffs = pred_coords - gt_wp
    l2_per_wp = np.linalg.norm(diffs, axis=1)
    ade = np.nanmean(l2_per_wp)
    fde = l2_per_wp[-1]

    print(colored("\nMetrics:", "cyan"))
    print(f"  ADE: {ade:.4f} m")
    print(f"  FDE: {fde:.4f} m")

    # ----------------- Visualization setup -----------------
    os.makedirs(args.output_dir, exist_ok=True)
    img_out = os.path.join(args.output_dir, f"sample_{args.sample_idx:05d}_image.jpg")
    bev_out = os.path.join(args.output_dir, f"sample_{args.sample_idx:05d}_bev.jpg")

    # ----------------- nuScenes API (for geometry & BEV) -----------------
    print(colored("\nInitializing nuScenes API...", "yellow"))
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Robustly get sample_token and sample
    sample_token = item.get("sample_token", None)
    sample = item.get("sample", None)

    if sample_token is None and sample is not None:
        sample_token = sample["token"]

    if sample_token is None:
        # Fall back to dataset's internal sample_tokens, if present
        if hasattr(ds, "sample_tokens"):
            sample_token = ds.sample_tokens[args.sample_idx]
        else:
            raise RuntimeError(
                "Cannot determine sample_token from dataset item or NuScenesDataset. "
                "Please ensure NuScenesDataset exposes sample_token or sample."
            )

    if sample is None:
        sample = nusc.get("sample", sample_token)

    # ----------------- Camera transforms for image overlay -----------------
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)
    cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    cam_ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])

    cam_to_ego = {
        "translation": cam_calib["translation"],
        "rotation": cam_calib["rotation"],
        "camera_intrinsic": np.array(cam_calib["camera_intrinsic"]),
    }
    ego_to_world = {
        "translation": cam_ego_pose["translation"],
        "rotation": cam_ego_pose["rotation"],
    }

    # ----------------- 1) Image + trajectory overlay -----------------
    print(colored("\nRendering image with GT & predicted trajectories...", "yellow"))
    visualize_trajectory(
        image_pil=image,
        gt_waypoints_2d=gt_wp,
        pred_waypoints_2d=pred_coords,
        cam_to_ego=cam_to_ego,
        ego_to_world=ego_to_world,
        idx=args.sample_idx,
        output_dir=args.output_dir,
        return_img=False,   # saves as sample_XXXX.jpg
    )
    # Our visualize_trajectory uses its own filename; for convenience, rename/copy:
    if os.path.exists(os.path.join(args.output_dir, f"sample_{args.sample_idx:04d}.jpg")):
        os.replace(
            os.path.join(args.output_dir, f"sample_{args.sample_idx:04d}.jpg"),
            img_out,
        )
        print(colored(f"✓ Image visualization saved to: {img_out}", "green"))

    # ----------------- 2) LiDAR BEV + trajectory overlay -----------------
    print(colored("\nRendering LiDAR BEV with GT & predicted trajectories...", "yellow"))
    try:
        import matplotlib.pyplot as plt

        lidar_token = sample["data"]["LIDAR_TOP"]
        # Render base BEV first
        nusc.render_sample_data(
            lidar_token,
            underlay_map=True,
            out_path=None,
            nsweeps=5,
        )
        plt.savefig(bev_out, dpi=150, bbox_inches="tight")
        plt.close()
        print(colored(f"✓ BEV saved to: {bev_out}", "green"))

        # Get ego pose at LiDAR time for world→ego transforms inside visualize_trajectory_bev
        lidar_data_nusc = nusc.get("sample_data", lidar_token)
        ego_pose_lidar = nusc.get("ego_pose", lidar_data_nusc["ego_pose_token"])

        if visualize_trajectory_bev(bev_out, gt_wp, pred_coords, ego_pose_lidar, bev_out):
            print(colored(f"✓ BEV with trajectory overlay saved to: {bev_out}", "green"))
    except Exception as e:
        print(colored(f"✗ BEV rendering failed: {e}", "red"))

    print(colored("\nDone.", "green"))
    print(colored(f"  Image: {img_out}", "green"))
    print(colored(f"  BEV:   {bev_out}", "green"))


if __name__ == "__main__":
    main()
