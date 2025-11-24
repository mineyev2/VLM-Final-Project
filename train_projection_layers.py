#!/usr/bin/env python3
"""
Training Script for Multimodal Projection Layers
=================================================

Trains MLP projectors that map CLIP vision features and SST LiDAR features
to Qwen LLM embedding space for autonomous driving trajectory prediction.

Training Strategy:
    Phase 1 (Current): Train projection layers only
    - CLIP encoder: FROZEN
    - SST encoder: FROZEN  
    - Qwen LLM: FROZEN
    - Vision MLP: TRAINABLE
    - LiDAR MLP: TRAINABLE

Usage:
    python train_projection_layers.py \\
        --dataroot /path/to/nuscenes \\
        --version v1.0-mini \\
        --lidar_encoder_path /path/to/lidarclip_checkpoint.ckpt \\
        --epochs 50 \\
        --batch_size 16 \\
        --wandb
"""

import argparse
import logging
import os
import sys
import gc
from datetime import datetime
from pathlib import Path

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

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Import model
from src.models.multimodal_qwen_model import MultimodalQwenModel

# Import dataset
from scripts.nuscenes_dataset import NuScenesDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multimodal MLP projectors for autonomous driving"
    )
    
    # ========================================================================
    # Dataset Arguments
    # ========================================================================
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Path to nuScenes dataset root directory"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        help="nuScenes dataset version"
    )
    
    # ========================================================================
    # Model Path Arguments
    # ========================================================================
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model name from HuggingFace"
    )
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Qwen model name from HuggingFace"
    )
    parser.add_argument(
        "--sst_config_path",
        type=str,
        default="src/models/mmdet3d/configs/sst_encoder_only_config.py",
        help="Path to SST configuration file"
    )
    parser.add_argument(
        "--lidar_encoder_path",
        type=str,
        required=True,
        help="Path to pretrained LidarCLIP checkpoint (.ckpt file)"
    )
    
    # ========================================================================
    # Model Architecture Arguments
    # ========================================================================
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dimension for MLP projectors"
    )
    parser.add_argument(
        "--mlp_num_layers",
        type=int,
        default=3,
        help="Number of layers in MLP projectors"
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for MLP projectors"
    )
    
    # ========================================================================
    # Training Arguments
    # ========================================================================
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", "--lr",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs"
    )
    
    # ========================================================================
    # Freezing Policy Arguments
    # ========================================================================
    parser.add_argument(
        "--unfreeze_encoders",
        action="store_true",
        help="Unfreeze CLIP and SST encoders for fine-tuning"
    )
    parser.add_argument(
        "--unfreeze_llm",
        action="store_true",
        help="Unfreeze Qwen LLM for fine-tuning"
    )
    
    # ========================================================================
    # Data Loading Arguments
    # ========================================================================
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # ========================================================================
    # Checkpoint Arguments
    # ========================================================================
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/projection_training",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom run name (auto-generated if None)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # ========================================================================
    # Logging Arguments
    # ========================================================================
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="multimodal-projection-training",
        help="W&B project name"
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=10,
        help="Log metrics every N batches"
    )
    
    # ========================================================================
    # Debug Arguments
    # ========================================================================
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, more logging)"
    )
    
    args = parser.parse_args()
    return args


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


def create_model_and_tokenizer(args, device):
    """
    Create model and tokenizer.
    
    Args:
        args: Command line arguments
        device: Device to load model on
    
    Returns:
        model, tokenizer
    """
    logging.info("Creating MultimodalQwenModel...")
    
    # Resolve config path
    config_path = args.sst_config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.isfile(config_path):
        logging.error(f"SST config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logging.info(f"Using SST config: {config_path}")
    
    # Create model
    try:
        model = MultimodalQwenModel(
            device=device,
            qwen_model_name=args.qwen_model_name,
            clip_model_name=args.clip_model_name,
            sst_config_path=config_path,
            lidarclip_checkpoint_path=args.lidar_encoder_path,
            freeze_encoders=not args.unfreeze_encoders,
            freeze_llm=not args.unfreeze_llm,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_num_layers=args.mlp_num_layers,
            mlp_dropout=args.mlp_dropout
        )
        
        model = model.to(device)
        logging.info("✓ Model created successfully")
        
    except Exception as e:
        logging.error(f"✗ Failed to create model: {e}")
        raise
    
    # Get tokenizer
    tokenizer = model.tokenizer
    
    return model, tokenizer


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
        'lidar_projector_state_dict': model.lidar_projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }
    
    # Optionally save encoder states if they were trained
    if hasattr(model, "vision_tower") and not model.freeze_encoders:
        checkpoint['vision_encoder_state_dict'] = model.vision_tower.state_dict()
    
    if hasattr(model, "lidar_encoder") and not model.freeze_encoders:
        checkpoint['lidar_encoder_state_dict'] = model.lidar_encoder.state_dict()
    
    if hasattr(model, "language_model") and not model.freeze_llm:
        checkpoint['llm_state_dict'] = model.language_model.state_dict()
    
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
    """Main training loop."""
    
    # ========================================================================
    # Setup Logging
    # ========================================================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    args = parse_args()
    
    # Auto-generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        freeze_status = "frozen" if not args.unfreeze_encoders else "finetuned"
        args.run_name = f"projection_{freeze_status}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Print Configuration
    # ========================================================================
    print("\n" + "="*70)
    print("Multimodal Projection Layer Training")
    print("="*70)
    print(f"Run name:      {args.run_name}")
    print(f"Output dir:    {output_dir}")
    print(f"Dataset:       nuScenes {args.version}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs:        {args.epochs}")
    print(f"Freeze encoders: {not args.unfreeze_encoders}")
    print(f"Freeze LLM:      {not args.unfreeze_llm}")
    print(f"WandB enabled:   {args.wandb}")
    print("="*70 + "\n")
    
    # ========================================================================
    # Initialize W&B
    # ========================================================================
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )
    
    # ========================================================================
    # Device Setup
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # ========================================================================
    # Create Model and Tokenizer
    # ========================================================================
    model, tokenizer = create_model_and_tokenizer(args, device)
    
    # ========================================================================
    # Create Dataset and DataLoader
    # ========================================================================
    print("\nLoading dataset...")
    
    prompt_part1 = model.prompt_part1
    prompt_part2 = model.prompt_part2
    
    try:
        dataset = NuScenesDataset(
            version=args.version,
            dataroot=args.dataroot,
            tokenizer=tokenizer,
            prompt_part1=prompt_part1,
            prompt_part2=prompt_part2
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples\n")
    except Exception as e:
        logging.error(f"✗ Failed to load dataset: {e}")
        raise
    
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
    
    print(f"✓ DataLoader created: {len(dataloader)} batches\n")
    
    # ========================================================================
    # Create Optimizer and Scheduler
    # ========================================================================
    print("Setting up optimizer...")
    
    optimizer = optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
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
    
    print("✓ Optimizer and scheduler created\n")
    
    # ========================================================================
    # Resume from Checkpoint (Optional)
    # ========================================================================
    start_epoch = 0
    global_step = 0
    
    if args.resume_from:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler, args.resume_from, device
        )
    
    # ========================================================================
    # Log Parameter Count
    # ========================================================================
    trainable_params, total_params = count_trainable_params(model)
    
    # Log to WandB
    if args.wandb:
        wandb.config.update({
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percentage": 100 * trainable_params / total_params
        })
    
    # ========================================================================
    # Loss Function
    # ========================================================================
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        # Set encoder eval modes if frozen
        if model.freeze_encoders:
            model.vision_tower.eval()
            model.lidar_encoder.eval()
        if model.freeze_llm:
            model.language_model.eval()
        
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            images = batch['images']
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lidar_data = batch.get("lidar", None)
            
            # Process LiDAR data
            point_clouds = None
            if lidar_data is not None:
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
                logits = model(
                    images=images,
                    point_clouds=point_clouds,
                    input_ids=input_ids,
                    use_vision=True,
                    use_lidar=(point_clouds is not None)
                )
                
                # ================================================================
                # FIX: Adjust labels to match logits length
                # ================================================================
                if logits.shape[1] != labels.shape[1]:
                    # Calculate number of multimodal tokens
                    num_multimodal_tokens = logits.shape[1] - labels.shape[1]
                    
                    # Create padding with -100 (ignore_index)
                    padding = torch.full(
                        (labels.shape[0], num_multimodal_tokens),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    
                    # Concatenate padding before labels
                    labels = torch.cat([padding, labels], dim=1)
                    
                    # Verify shapes match
                    assert logits.shape[1] == labels.shape[1], \
                        f"Shape mismatch: logits {logits.shape[1]} vs labels {labels.shape[1]}"

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # 3. Compute Loss
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
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
            if args.wandb and (batch_idx % args.log_freq == 0):
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
        if args.wandb:
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
            if args.wandb:
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
            if args.wandb:
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
    if args.wandb:
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
    if args.wandb:
        wandb.log({
            "charts/loss_history": wandb.Image(str(plot_path))
        })
    
    # ========================================================================
    # Finish WandB
    # ========================================================================
    if args.wandb:
        print("\n✓ WandB logging complete")
        wandb.finish()
    
    print("\n" + "="*70)
    print("All done! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()
