# PyTorch Files
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Local files
from src.models.qwen_clip_model import QwenCLIPModel
from src.models.lidar_emma import LidarEMMA
from scripts.nuscenes_dataset import NuScenesDataset
from training_configs.dataclass import TrainingArgs

# Other
import argparse
import gc
import os
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from datetime import datetime
import wandb
import yaml
from dataclasses import fields
import logging

# def custom_collate_fn(batch):
#     """
#     Custom collate to handle the new dataset output structure.
#     Stacks tensors (input_ids, labels) and collects PIL images into a list.
#     """
#     # Collect raw PIL images into a list (processor handles batching later)
#     raw_images = [item['image'] for item in batch]
    
#     # Stack tensors
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     labels = torch.stack([item['labels'] for item in batch])
    
#     return {
#         'raw_images': raw_images,
#         'input_ids': input_ids,
#         'labels': labels
#     }

def collate_fn(batch, tokenizer_pad_id):
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
    
    # LiDAR point clouds (list of tensors, variable size)
    lidar_list = [item.get("lidar", item.get("point_cloud", None)) for item in batch]
    
    # Pad text sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer_pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "images": images,
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "lidar": lidar_list,
    }

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def count_trainable_params(model):
    """Count and log trainable parameters."""
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    logging.info(
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    
    print(colored(
        f"\nTrainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)",
        "magenta"
    ))
    
    return trainable_params, total_params

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'vision_projector_state_dict': model.vision_projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }
    
    # Optionally save encoder states if they were trained or they exist
    if hasattr(model, "vision_tower") and not model.freeze_encoders:
        checkpoint['vision_encoder_state_dict'] = model.vision_tower.state_dict()
    
    if hasattr(model, "lidar_encoder") and not model.freeze_encoders:
        checkpoint['lidar_encoder_state_dict'] = model.lidar_encoder.state_dict()
    
    if hasattr(model, "language_model") and not model.freeze_llm:
        checkpoint['llm_state_dict'] = model.language_model.state_dict()

    if hasattr(model, "lidar_projector"): # Make sure we only run if we are using lidar
        checkpoint['lidar_projector_state_dict'] = model.lidar_projector.state_dict()
    
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint and resume training."""
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    if 'vision_projector_state_dict' in checkpoint:
        model.vision_projector.load_state_dict(checkpoint['vision_projector_state_dict'])
    
    if 'lidar_projector_state_dict' in checkpoint:
        model.lidar_projector.load_state_dict(checkpoint['lidar_projector_state_dict'])
    
    # Load optimizer
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    logging.info(f"✓ Resumed from epoch {epoch}, step {global_step}")
    
    return epoch, global_step

def main():
    # ========================================================================
    # Setup Logging
    # ========================================================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # ========================================================================
    # Parse Arguments
    # ========================================================================
    parser = argparse.ArgumentParser(description="Train a model on the NuScenes dataset.")
    
    # Required args
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["1a", "2a", "3a", "1b", "2b", "3b",
                 "1a-lidar","2a-lidar", "3a-lidar", "1b-lidar", "2b-lidar", "3b-lidar"],
        help="Select ablation config: 1a, 2a, 3a, 1b, 2b, 3b (no lidar), or 1a-lidar, 2a-lidar, 3a-lidar, 1b-lidar, 2b-lidar, 3b-lidar (with lidar)."
    )
    parser.add_argument("--run_name", type=str, required=True, help="Custom run name (optional).")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name.") # Pranav's: "vlm-training"

    # Optional args
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file to resume training.")

    terminal_args = parser.parse_args()

    # 1) Load dataclasses with default values
    args = TrainingArgs()

    # 2) Override config class with YAML config
    if "lidar" in terminal_args.ablation:
        yaml_path = f"training_configs/{terminal_args.ablation.replace('-lidar', '')}.yaml"
    else:
        yaml_path = f"training_configs/{terminal_args.ablation}.yaml"
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    for key, value in yaml_config.items():
        if hasattr(args, key) and value is not None:
            # Get the expected type from the dataclass
            field_type = next(f.type for f in fields(args) if f.name == key)
            setattr(args, key, field_type(value))

    # 3) Override updated YAML config with command-line args
    for key, value in vars(terminal_args).items():
        if value is not None:
            setattr(args, key, value)

    # 4) Store bool for using lidar
    args.use_lidar = "lidar" in args.ablation

    # ========================================================================
    # Initialize Model, Dataset, Dataloader, Optimizer, Scheduler
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}\n")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # if args.use_lidar:
    #     model = LidarEMMA(device, llm=args.llm, freeze_encoders=args.freeze_encoder, freeze_llm=args.freeze_lang_model, use_lidar=args.use_lidar)
    # else:
    #     model = QwenCLIPModel(device, llm=args.llm, freeze_encoder=args.freeze_encoder, freeze_llm=args.freeze_lang_model)

    model = LidarEMMA(device, llm=args.llm, freeze_encoders=args.freeze_encoder, freeze_llm=args.freeze_lang_model, use_lidar=args.use_lidar)


    tokenizer = model.tokenizer
    
    print("\nLoading dataset...")
    dataset = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        tokenizer=model.tokenizer,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2,
        output_lidar=args.use_lidar,
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    # Create dataloader
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    custom_collate_fn = lambda batch: collate_fn(batch, pad_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == "cuda" else False
    )

    # TODO: Make this variable as well
    # optimizer = torch.optim.Adam(model.mlp_projector.parameters(), lr=args.lr)
    optimizer = optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.lr
        # weight_decay=args.weight_decay # TODO: include back if needed
    )

    # Cosine annealing scheduler with warmup
    total_steps = len(dataloader) * args.epochs
    warmup_steps = len(dataloader) * args.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========================================================================
    # Resume from Checkpoint if specified
    # =======================================================================

    start_epoch = 0
    global_step = 0

    # if args.resume_from: # TODO: Add in later
    #     start_epoch, global_step = load_checkpoint(
    #         model, optimizer, scheduler, args.resume_from, device
    #     )

    loss_history = []
    wandb_run_id = None
    checkpoint = None

    # # Load checkpoint if resuming
    # if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint): # TODO: Make this work for lidaremma
    #     print(colored(f"--- Resuming training from: {args.resume_from_checkpoint} ---", "yellow"))
    #     checkpoint = torch.load(args.resume_from_checkpoint, weights_only=False, map_location=device)
        
    #     model.mlp_projector.load_state_dict(checkpoint['mlp_projector_state_dict'])
    #     model.vision_tower.load_state_dict(checkpoint['vision_tower_state_dict'])
    #     model.language_model.load_state_dict(checkpoint['language_model_state_dict'])
        
    #     start_epoch = checkpoint['epoch']
    #     loss_history = checkpoint['loss_history']
    #     wandb_run_id = checkpoint['wandb_run_id']
    #     args.run_name = checkpoint['run_name']

    #     args.freeze_encoder = checkpoint['args'].freeze_encoder
    #     args.freeze_lang_model = checkpoint['args'].freeze_lang_model
        
    #     print(colored(f"--- Resumed from Epoch {start_epoch}. WandB Run ID: {wandb_run_id} ---", "yellow"))
    # else:
    if args.run_name is None:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{date_str}-epochs{args.epochs}"

    # if args.freeze_encoder: # Removed, assuming get_trainable_parameters() handles this
    #     # For QwenCLIP: Freeze vision tower. For LidarEMMA: Freeze both encoders.
    #     model.freeze_encoder()
    # else:
    #     optimizer.add_param_group({'params': model.vision_tower.parameters(), 'lr': args.lr * 0.1})
    # if args.freeze_lang_model:
    #     model.language_model.requires_grad_(False)
    # else:
    #     optimizer.add_param_group({'params': model.language_model.parameters(), 'lr': args.lr * 0.1})

    # Create output directory
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # if checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     print(colored("--- Optimizer state successfully restored ---", "yellow"))
    
    # args.output_dir = os.path.join(args.output_dir, args.run_name)
    # os.makedirs(args.output_dir, exist_ok=True)

    if wandb_run_id:
        wandb.init(
            project=args.wandb_project,
            id=wandb_run_id, 
            resume="must" 
        )
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )
        wandb_run_id = wandb.run.id

    print(colored("--- Training Configuration ---", "cyan"))
    if args.use_lidar:
        print(colored("Using LidarEMMA model!", "green"))
    else:
        print(colored("Using QwenCLIP model!", "green"))
    for k, v in vars(args).items():
        print(colored(f"{k}: {v}", "cyan"))
    print(colored("--------------------------", "cyan"))

    # ========================================================================
    # Log Parameter Count
    # ========================================================================
    trainable_params, total_params = count_trainable_params(model)

    wandb.config.update({
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": 100 * trainable_params / total_params
    })
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")

    loss_history = []
    best_loss = float('inf')
    model.train()

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            images = batch['images']  # List of PIL images
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lidar_data = batch.get("lidar", None)
            
            # Process images with CLIP processor
            pixel_values = model.image_processor(images=images, return_tensors="pt").pixel_values.to(device)
            
            # Process LiDAR data
            point_clouds = None
            if args.use_lidar:
                point_clouds = [pc for pc in lidar_data if pc is not None]
                if len(point_clouds) == 0:
                    point_clouds = None
                else:
                    # Move to device
                    point_clouds = [pc.to(device) if isinstance(pc, torch.Tensor) else pc 
                                    for pc in point_clouds]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = model(
                    images=pixel_values,
                    point_clouds=point_clouds,
                    input_ids=input_ids,
                    labels=labels,
                )
                
                # # ================================================================
                # # FIX: Adjust labels to match logits length
                # # ================================================================
                # if logits.shape[1] != labels.shape[1]:
                #     # Calculate number of multimodal tokens
                #     num_multimodal_tokens = logits.shape[1] - labels.shape[1]
                    
                #     # Create padding with -100 (ignore_index)
                #     padding = torch.full(
                #         (labels.shape[0], num_multimodal_tokens),
                #         -100,
                #         dtype=labels.dtype,
                #         device=labels.device
                #     )
                    
                #     # Concatenate padding before labels
                #     labels = torch.cat([padding, labels], dim=1)
                    
                #     # Verify shapes match
                #     assert logits.shape[1] == labels.shape[1], \
                #         f"Shape mismatch: logits {logits.shape[1]} vs labels {labels.shape[1]}"

                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()

                # # 3. Compute Loss
                # loss = loss_fn(
                #     shift_logits.view(-1, shift_logits.size(-1)),
                #     shift_labels.view(-1)
                # )

                loss = outputs.loss
                
            except Exception as e:
                logging.error(f"Forward pass failed: {e}")
                logging.error(f"Batch size: {len(images)}")
                logging.error(f"Input IDs shape: {input_ids.shape}")
                logging.error(f"Point clouds: {point_clouds is not None}")
                raise
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = 0.0
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(),
                    max_norm=args.grad_clip
                )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            global_step += 1
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # ================================================================
            # Enhanced WandB Logging (every log_freq steps)
            # ================================================================
            if (batch_idx % args.log_freq == 0):
                log_dict = {
                    # Loss
                    "train/loss_step": loss.item(),
                    
                    # Learning rate
                    "train/learning_rate": current_lr,
                    
                    # Step info
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/batch_idx": batch_idx,
                    
                    # Gradient statistics
                    "train/grad_norm": grad_norm if args.grad_clip > 0 else 0.0,
                }
                
                # Add GPU memory usage if available
                if torch.cuda.is_available():
                    log_dict.update({
                        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                        "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    })
                
                wandb.log(log_dict)
        
        # ====================================================================
        # Epoch Summary
        # ====================================================================
        avg_epoch_loss = total_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Log epoch metrics to WandB
        epoch_log_dict = {
            "epoch/train_loss": avg_epoch_loss,
            "epoch/learning_rate": current_lr,
            "epoch/number": epoch + 1,
        }
        
        # Add data statistics
        epoch_log_dict.update({
            "data/batch_size": args.batch_size,
            "data/num_batches": len(dataloader),
            "data/samples_per_epoch": len(dataset),
        })
        
        wandb.log(epoch_log_dict)
        
        # ====================================================================
        # Save Checkpoint
        # ====================================================================
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step, avg_epoch_loss, checkpoint_path
            )
            print(colored(f"✓ Checkpoint saved: {checkpoint_path}", "yellow"))
            
            # Log checkpoint to WandB
            wandb.save(str(checkpoint_path), policy="now")
            
            # Log checkpoint metadata
            wandb.run.summary[f"checkpoint_epoch_{epoch+1}_loss"] = avg_epoch_loss
            wandb.run.summary[f"checkpoint_epoch_{epoch+1}_step"] = global_step
        
        # ====================================================================
        # Save Best Model
        # ====================================================================
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_checkpoint_path = output_dir / "best_checkpoint.pth"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step, avg_epoch_loss, best_checkpoint_path
            )
            print(colored(f"✓ New best model! Loss: {best_loss:.4f}", "green"))
            
            # Log to WandB
            wandb.run.summary["best_loss"] = best_loss
            wandb.run.summary["best_epoch"] = epoch + 1
        
        print()
    
    # ========================================================================
    # Training Complete
    # ========================================================================
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*70 + "\n")
    
    # Save final checkpoint
    final_ckpt_path = output_dir / "final_checkpoint.pth"
    save_checkpoint(
        model, optimizer, scheduler, args.epochs, global_step, loss_history[-1], final_ckpt_path
    )
    
    # ========================================================================
    # Final WandB Summary
    # ========================================================================
    wandb.save(str(final_ckpt_path))
    
    # Log final summary statistics
    wandb.run.summary.update({
        "final_loss": loss_history[-1],
        "best_loss": best_loss,
        "total_epochs": args.epochs,
        "total_steps": global_step,
        "training_complete": True,
    })
    
    # ========================================================================
    # Plot Loss Curve
    # ========================================================================
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        pass
    
    import matplotlib
    matplotlib.use("Agg")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(loss_history, linewidth=3, alpha=0.8, marker='o', markersize=4, label='Training Loss')
    
    # Smooth curve
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
    plt.tight_layout()
    
    plot_path = output_dir / "loss_history.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(colored(f"✓ Loss plot saved to: {plot_path}", "cyan"))
    
    # Log plot to WandB
    wandb.log({
        "charts/loss_history": wandb.Image(str(plot_path))
    })
    
    # ========================================================================
    # Finish WandB
    # ========================================================================
    print("\n✓ WandB logging complete")
    wandb.finish()
    
    print("\n" + "="*70)
    print("All done! 🎉")
    print("="*70)

if __name__ == "__main__":
    main()
