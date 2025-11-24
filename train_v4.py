import argparse
import torch
import gc
import os
from torch.utils.data import DataLoader
from termcolor import colored
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from datetime import datetime
import wandb

from src.models.qwen_clip_model import QwenCLIPModel
# UPDATED: Import the new dataset file
from scripts.nuscenes_dataset import NuScenesDataset

def custom_collate_fn(batch):
    """
    Custom collate to handle the new dataset output structure.
    Stacks tensors (input_ids, labels) and collects PIL images into a list.
    """
    # Collect raw PIL images into a list (processor handles batching later)
    raw_images = [item['image'] for item in batch]
    
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'raw_images': raw_images,
        'input_ids': input_ids,
        'labels': labels
    }

def main():
    parser = argparse.ArgumentParser(description="Train the QwenCLIPModel on the NuScenes dataset.")
    parser.add_argument("--version", type=str, default='v1.0-trainval', help="Version of the NuScenes dataset.")
    parser.add_argument("--dataroot", type=str, default="/storage/ice-shared/cs8803vlm/rmineyev3/", help="Root directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--penalty_weight", type=float, default=5.0, help="Weight for the missing waypoint penalty.")
    parser.add_argument("--output_dir", type=str, default="./outputs/latest", help="Directory to save model and plots.")
    parser.set_defaults(freeze_vision_tower=True, freeze_lang_model=True)
    parser.add_argument("--unfreeze-vision-tower", dest="freeze_vision_tower", action="store_false",
                        help="Unfreeze the vision tower for fine-tuning.")
    parser.add_argument("--unfreeze-lang-model", dest="freeze_lang_model", action="store_false",
                        help="Unfreeze the language model for fine-tuning.")
    parser.add_argument("--wandb_project", type=str, default="vlm-training", help="WandB project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name (optional).")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs.")
    # NEW: Resume argument
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file to resume training.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    model = QwenCLIPModel(device)
    
    # Initialize Optimizer (needed before loading state)
    optimizer = torch.optim.Adam(model.mlp_projector.parameters(), lr=args.lr)
    
    # --- RESUME LOGIC START ---
    start_epoch = 0
    loss_history = []
    wandb_run_id = None
    checkpoint = None

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(colored(f"--- Resuming training from: {args.resume_from_checkpoint} ---", "yellow"))
        checkpoint = torch.load(args.resume_from_checkpoint, weights_only=False, map_location=device)
        
        # Load Weights
        model.mlp_projector.load_state_dict(checkpoint['mlp_projector_state_dict'])
        model.vision_tower.load_state_dict(checkpoint['vision_tower_state_dict'])
        model.language_model.load_state_dict(checkpoint['language_model_state_dict'])
        
        # Restore Training State
        start_epoch = checkpoint['epoch']
        loss_history = checkpoint.get('loss_history', [])
        wandb_run_id = checkpoint.get('wandb_run_id')
        
        # Restore Args if available, specifically the run name to keep outputs in same folder
        if 'run_name' in checkpoint:
            args.run_name = checkpoint['run_name']
        
        print(colored(f"--- Resumed from Epoch {start_epoch}. WandB Run ID: {wandb_run_id} ---", "yellow"))
    else:
        # Generate new run name if not resuming
        if args.run_name is None:
            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            args.run_name = f"{date_str}-epochs{args.epochs}"

    # Setup Output Directory (after resolving run_name)
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize WandB (New or Resume)
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
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(colored("--------------------------", "cyan"))

    # Add params to optimizer based on freeze settings
    # Note: If resuming, we might overwrite these groups, but we load state_dict after
    if args.freeze_vision_tower:
        model.vision_tower.requires_grad_(False)
    else:
        optimizer.add_param_group({'params': model.vision_tower.parameters(), 'lr': args.lr * 0.1})
    
    if args.freeze_lang_model:
        model.language_model.requires_grad_(False)
    else:
        optimizer.add_param_group({'params': model.language_model.parameters(), 'lr': args.lr * 0.1})

    # Load Optimizer State (must be done after adding param groups)
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(colored("--- Optimizer state successfully restored ---", "yellow"))
        except Exception as e:
            print(colored(f"--- Warning: Could not restore optimizer state: {e} ---", "red"))

    # --- DATASET & DATALOADER ---
    dataset = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        tokenizer=model.tokenizer,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        collate_fn=custom_collate_fn
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(colored(f"Starting training from epoch {start_epoch+1}...", "blue"))

    for epoch in range(start_epoch, args.epochs):
        model.mlp_projector.train()
        if not args.freeze_vision_tower:
            model.vision_tower.train()
        if not args.freeze_lang_model:
            model.language_model.train()

        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            # Access data using new keys from custom_collate_fn
            images = batch['raw_images'] # List of PIL images
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Process images here in the loop
            image_inputs = model.image_processor(images=images, return_tensors="pt").to(device)

            optimizer.zero_grad()
            logits = model(image_inputs['pixel_values'], input_ids)

            num_image_patches = logits.shape[1] - labels.shape[1]
            logits_for_loss = logits[:, num_image_patches:-1, :]
            labels_for_loss = labels[:, 1:]

            loss = loss_fn(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                labels_for_loss.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_epoch_loss = total_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(colored(f"Epoch {epoch + 1} complete. Average Loss: {avg_epoch_loss:.4f}", "green"))

        # WandB log
        wandb.log({"epoch": epoch + 1, "train_loss": avg_epoch_loss})

        # Save Checkpoint (Resumable)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
            
            torch.save({
                'epoch': epoch + 1,  # Save the NEXT epoch index
                'mlp_projector_state_dict': model.mlp_projector.state_dict(),
                'vision_tower_state_dict': model.vision_tower.state_dict(),
                'language_model_state_dict': model.language_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history,
                'wandb_run_id': wandb_run_id,
                'run_name': args.run_name,
                'args': args # Save args for reference
            }, checkpoint_path)
            
            print(colored(f"Resumable checkpoint saved: {checkpoint_path}", "yellow"))
            wandb.save(checkpoint_path, policy="now")

    print(colored("Training finished successfully!", "green"))

    # Save Final Model (Compatible with evaluation script)
    model_save_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.mlp_projector.state_dict(),
        'vision_tower_state_dict': model.vision_tower.state_dict(),
        'language_model_state_dict': model.language_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'penalty_weight': args.penalty_weight,
        'wandb_run_id': wandb_run_id
    }, model_save_path)
    wandb.save(model_save_path)
    print(colored(f"Full model saved to: {model_save_path}", "green"))

    # Plotting
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(loss_history, linewidth=3, color='#2E86AB', alpha=0.8, marker='o', markersize=4)
    if len(loss_history) > 5:
        try:
            x_smooth = np.linspace(0, len(loss_history)-1, len(loss_history)*3)
            f = interpolate.interp1d(range(len(loss_history)), loss_history, kind='cubic')
            y_smooth = f(x_smooth)
            ax.plot(x_smooth, y_smooth, '--', color='#A23B72', alpha=0.6, linewidth=2, label='Smoothed Trend')
        except Exception:
            pass

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss Over Time', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    plot_png_path = os.path.join(args.output_dir, "loss_history.png")
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(colored(f"Loss plot saved to: {plot_png_path}", "cyan"))

    wandb.log({"loss_plot": wandb.Image(plot_png_path)})
    wandb.finish()

if __name__ == "__main__":
    main()
