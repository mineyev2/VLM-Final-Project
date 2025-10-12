import argparse
import torch
import gc
import os
from torch.utils.data import DataLoader
from termcolor import colored
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from src.models.qwen_clip_model import QwenCLIPModel
from scripts.nuscenes_dataset import NuScenesDataset

def main():
    parser = argparse.ArgumentParser(description="Train the QwenCLIPModel on the NuScenes dataset.")
    parser.add_argument("--version", type=str, default='v1.0-mini', help="Version of the NuScenes dataset.")
    parser.add_argument("--dataroot", type=str, default="./datasets/nuscenes", help="Root directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    # Add a new argument for the penalty weight, allowing it to be tuned
    parser.add_argument("--penalty_weight", type=float, default=5.0, help="Weight for the missing waypoint penalty.")
    parser.add_argument("--output_dir", type=str, default="./outputs/latest", help="Directory to save model and plots.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    print(colored("--- Training Configuration ---", "cyan"))
    print(f"Device: {device}")
    print(f"NuScenes Version: {args.version}")
    print(f"Dataroot: {args.dataroot}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Penalty Weight: {args.penalty_weight}")
    print(f"Output Directory: {args.output_dir}")
    print(colored("--------------------------", "cyan"))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    model = QwenCLIPModel(device)
    dataset = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        tokenizer=model.tokenizer,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2
    )

    # Define the collate function to handle padding
    def collate_fn(batch, tokenizer_pad_id):
        images = [item['image'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad sequences to the max length in the batch
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer_pad_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) # Use -100 for label padding

        return {
            'images': images,
            'input_ids': input_ids_padded,
            'labels': labels_padded
        }

    # Use a lambda to pass the tokenizer's pad_token_id to the collate_fn
    pad_id = model.tokenizer.pad_token_id
    custom_collate_fn = lambda batch: collate_fn(batch, pad_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    
    optimizer = torch.optim.Adam(model.mlp_projector.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(colored("Starting training...", "blue"))
    loss_history = []
    for epoch in range(args.epochs):
        model.mlp_projector.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch in progress_bar:
            images = batch['images']
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
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

    print(colored("Training finished successfully!", "green"))
    
    #########################################################
    ###############  Save the trained model  ################
    #########################################################
    model_save_path = os.path.join(args.output_dir, "image_projector.pth")
    torch.save({
        'model_state_dict': model.mlp_projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'penalty_weight': args.penalty_weight
    }, model_save_path)
    print(colored(f"Model saved to: {model_save_path}", "green"))
    
    #########################################################
    ############# Create a beautiful loss plot ##############
    #########################################################
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the loss curve with styling
    ax.plot(loss_history, linewidth=3, color='#2E86AB', alpha=0.8, marker='o', markersize=4, markevery=max(1, len(loss_history)//20))
    
    # Add smooth trend line
    if len(loss_history) > 5:
        x_smooth = np.linspace(0, len(loss_history)-1, len(loss_history)*3)
        f = interpolate.interp1d(range(len(loss_history)), loss_history, kind='cubic')
        y_smooth = f(x_smooth)
        ax.plot(x_smooth, y_smooth, '--', color='#A23B72', alpha=0.6, linewidth=2, label='Smoothed Trend')
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Average Loss', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title('Training Loss Over Time', fontsize=18, fontweight='bold', color='#2E86AB', pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    
    # Customize ticks and limits
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333')
    ax.set_xlim(0, len(loss_history)-1)
    
    # Add legend if smoothed line exists
    if len(loss_history) > 5:
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Save plot to output directory
    plot_png_path = os.path.join(args.output_dir, "loss_history.png")
    
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(colored(f"Loss plot saved to: {plot_png_path}", "cyan"))

if __name__ == "__main__":
    main()
