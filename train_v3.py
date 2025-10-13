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
from datetime import datetime
import wandb

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
    parser.add_argument("--penalty_weight", type=float, default=5.0, help="Weight for the missing waypoint penalty.")
    parser.add_argument("--output_dir", type=str, default="./outputs/latest", help="Directory to save model and plots.")
    parser.set_defaults(freeze_vision_tower=True, freeze_lang_model=True)
    parser.add_argument("--unfreeze-vision-tower", dest="freeze_vision_tower", action="store_false",
                        help="Unfreeze the vision tower for fine-tuning.")
    parser.add_argument("--unfreeze-lang-model", dest="freeze_lang_model", action="store_false",
                        help="Unfreeze the language model for fine-tuning.")
    parser.add_argument("--wandb_project", type=str, default="vlm-training", help="WandB project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name (optional).")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs.")
    args = parser.parse_args()

    # 🟢 自动生成 run name
    if args.run_name is None:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{date_str}-epochs{args.epochs}"

    # 🟢 输出路径带 run name
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # 🟢 初始化 wandb
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

    model = QwenCLIPModel(device)
    dataset = NuScenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        tokenizer=model.tokenizer,
        prompt_part1=model.prompt_part1,
        prompt_part2=model.prompt_part2
    )

    def collate_fn(batch, tokenizer_pad_id):
        images = [item['image'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer_pad_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {'images': images, 'input_ids': input_ids_padded, 'labels': labels_padded}

    pad_id = model.tokenizer.pad_token_id
    custom_collate_fn = lambda batch: collate_fn(batch, pad_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=custom_collate_fn)

    optimizer = torch.optim.Adam(model.mlp_projector.parameters(), lr=args.lr)
    if args.freeze_vision_tower:
        model.vision_tower.requires_grad_(False)
    else:
        optimizer.add_param_group({'params': model.vision_tower.parameters(), 'lr': args.lr * 0.1})
    if args.freeze_lang_model:
        model.language_model.requires_grad_(False)
    else:
        optimizer.add_param_group({'params': model.language_model.parameters(), 'lr': args.lr * 0.1})

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(colored("Starting training...", "blue"))
    loss_history = []

    for epoch in range(args.epochs):
        model.mlp_projector.train()
        if not args.freeze_vision_tower:
            model.vision_tower.train()
        if not args.freeze_lang_model:
            model.language_model.train()

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

        # 🟢 wandb log
        wandb.log({"epoch": epoch + 1, "train_loss": avg_epoch_loss})

        # 🟢 每N个epoch保存projector权重（覆盖旧文件）
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
            torch.save(model.mlp_projector.state_dict(), checkpoint_path)
            print(colored(f"Lightweight checkpoint saved (MLP only): {checkpoint_path}", "yellow"))

            # 上传到 wandb
            wandb.save(checkpoint_path, policy="now")

    print(colored("Training finished successfully!", "green"))

    # 🟢 保存完整最终模型
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
        'penalty_weight': args.penalty_weight
    }, model_save_path)
    wandb.save(model_save_path)
    print(colored(f"Full model saved to: {model_save_path}", "green"))

    # 🟢 绘制 loss 曲线
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(loss_history, linewidth=3, color='#2E86AB', alpha=0.8, marker='o', markersize=4)
    if len(loss_history) > 5:
        x_smooth = np.linspace(0, len(loss_history)-1, len(loss_history)*3)
        f = interpolate.interp1d(range(len(loss_history)), loss_history, kind='cubic')
        y_smooth = f(x_smooth)
        ax.plot(x_smooth, y_smooth, '--', color='#A23B72', alpha=0.6, linewidth=2, label='Smoothed Trend')

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
