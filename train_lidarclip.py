
import argparse
import logging
import os
import gc
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import wandb

# Import the new model class
from lidarclip_qwen_model import LidarCLIPQwenModel  # Assuming you save it as a separate file

# Dataset
from scripts.nuscenes_dataset import NuScenesDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train LidarCLIPQwenModel on the NuScenes dataset.")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes version.")
    parser.add_argument("--dataroot", type=str, default="./datasets/nuscenes", help="Dataset root.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--output_dir", type=str, default="./outputs/latest", help="Output directory for runs.")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--wandb_project", type=str, default="vlm-training", help="WandB project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name.")

    # Model paths
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2-7B-Instruct", help="Qwen model id/path.")
    parser.add_argument("--image_encoder_path", type=str, default="openai/clip-vit-large-patch14", help="CLIP vision encoder id/path.")
    parser.add_argument("--lidar_encoder_path", type=str, default="Lidar-CLIP/vit_l_14.ckpt", required=True, help="Path to pretrained LidarCLIP weights/checkpoint.")

    # Freezing policy
    parser.add_argument("--unfreeze-encoders", action="store_true",
                        help="Unfreeze vision and LiDAR encoders for fine-tuning")
    parser.add_argument("--unfreeze-llm", action="store_true",
                        help="Unfreeze LLM for fine-tuning")

    # (Kept for compatibility; not used directly by the integrated model)
    parser.add_argument("--penalty_weight", type=float, default=5.0, help="(Optional) extra loss weight placeholder.")

    args = parser.parse_args()
    return args


def collate_fn(batch, tokenizer_pad_id):
    """Pads text fields; passes through images and optionally lidar point clouds."""
    images = [item.get("image") for item in batch]
    input_ids = [item.get("input_ids") for item in batch]
    labels = [item.get("labels") for item in batch]

    # Optional lidar in dataset item: 'lidar' or 'point_cloud'
    lidar_list = [item.get("lidar", item.get("point_cloud", None)) for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer_pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "images": images,
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "lidar": lidar_list,
    }


def create_model_and_tokenizer(args, device):
    # Load the integrated LidarCLIPQwen model
    logging.info("Creating LidarCLIPQwen multimodal model...")
    model = LidarCLIPQwenModel(
        device=device,
        qwen_model_name=args.llm_path,
        clip_model_name=args.image_encoder_path,
        lidarclip_config_path="./lidarclip/model/sst_encoder_only_config.py",
        lidarclip_checkpoint_path=args.lidar_encoder_path,
        freeze_encoders=not args.unfreeze_encoders,
        freeze_llm=not args.unfreeze_llm
    )
    model = model.to(device)

    # Get tokenizer from the model
    tokenizer = model.tokenizer
    return model, tokenizer


def count_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    print(colored(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)", "magenta"))
    return trainable_params, total_params


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'vision_projector_state_dict': model.vision_projector.state_dict() if hasattr(model, "vision_projector") else None,
        'lidar_projector_state_dict': model.lidar_projector.state_dict() if hasattr(model, "lidar_projector") else None,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }

    # Optionally save encoder states if they were trained
    if hasattr(model, "vision_tower"):
        if not model.vision_tower.training:
            checkpoint['vision_encoder_state_dict'] = model.vision_tower.state_dict()
    if hasattr(model, "lidar_encoder"):
        if not model.lidar_encoder.training:
            checkpoint['lidar_encoder_state_dict'] = model.lidar_encoder.state_dict()

    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def main():
    # Basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = parse_args()

    # Auto-generate run name
    if args.run_name is None:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{date_str}-epochs{args.epochs}"

    # Output dir per run
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    print(colored("--- Training Configuration ---", "cyan"))
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(colored("--------------------------", "cyan"))

    # Create model + tokenizer
    model, tokenizer = create_model_and_tokenizer(args, device)

    # Build dataset
    # Some older dataset versions expected prompt parts; pass empty strings if unavailable.
    prompt_part1 = getattr(model, "prompt_part1", "")
    prompt_part2 = getattr(model, "prompt_part2", "")
    dataset = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        tokenizer=tokenizer,
        prompt_part1=prompt_part1,
        prompt_part2=prompt_part2
    )

    # Dataloader
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    custom_collate_fn = lambda batch: collate_fn(batch, pad_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    # Create optimizer with model's trainable parameters
    optimizer = optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = None  # (Optional) plug in a scheduler here

    # Log params
    count_trainable_params(model)

    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(colored("Starting training...", "blue"))
    loss_history = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            images = batch['images']
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # lidar: may be list[None] if dataset doesn't provide it
            lidar_data = batch.get("lidar", None)

            optimizer.zero_grad()

            # Convert lidar_data from tensor to list of tensors if needed
            point_clouds = None
            if lidar_data is not None:
                if isinstance(lidar_data, torch.Tensor):
                    point_clouds = [lidar_data[i] for i in range(lidar_data.shape[0])]
                else:
                    point_clouds = lidar_data

            # Get logits from the model
            logits = model(
                images=images,
                point_clouds=point_clouds,
                input_ids=input_ids,
                use_vision=True,
                use_lidar=(point_clouds is not None)
            )

            # Compute loss
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({'loss': loss.item()})
            wandb.log({"step": global_step, "train_loss_step": loss.item()})

        avg_epoch_loss = total_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(colored(f"Epoch {epoch + 1} complete. Average Loss: {avg_epoch_loss:.4f}", "green"))
        wandb.log({"epoch": epoch + 1, "train_loss": avg_epoch_loss})

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_epoch_loss, checkpoint_path)
            print(colored(f"Checkpoint saved: {checkpoint_path}", "yellow"))
            wandb.save(checkpoint_path, policy="now")

    print(colored("Training finished successfully!", "green"))

    # Save final lightweight checkpoint (projectors + optional encoders)
    final_ckpt_path = os.path.join(args.output_dir, "final_checkpoint.pth")
    save_checkpoint(model, optimizer, scheduler, args.epochs, global_step, loss_history[-1], final_ckpt_path)
    wandb.save(final_ckpt_path)

    # Plot loss
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        pass
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(loss_history, linewidth=3, alpha=0.8, marker='o', markersize=4)
    if len(loss_history) > 5:
        x_smooth = np.linspace(0, len(loss_history)-1, len(loss_history)*3)
        f = interpolate.interp1d(range(len(loss_history)), loss_history, kind='cubic')
        y_smooth = f(x_smooth)
        ax.plot(x_smooth, y_smooth, '--', alpha=0.6, linewidth=2, label='Smoothed Trend')

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss Over Time', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    import io
    plt.tight_layout()

    plot_png_path = os.path.join(args.output_dir, "loss_history.png")
    fig.savefig(plot_png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(colored(f"Loss plot saved to: {plot_png_path}", "cyan"))
    wandb.log({"loss_plot": wandb.Image(plot_png_path)})
    wandb.finish()


if __name__ == "__main__":
    main()
