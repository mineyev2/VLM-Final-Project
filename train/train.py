"""
train.py - Training script for multimodal (LiDAR + Image) to LLM projection layers

This script trains MLP projection layers that map:
1. LiDAR features from LidarCLIP encoder
2. Image features from CLIP encoder
Into the token embedding space of an LLM (e.g., Qwen)

Architecture:
    LiDAR → LidarCLIP (frozen) → MLP_lidar (trainable) ┐
                                                          ├→ LLM (frozen) → Output
    Image → CLIP (frozen) → MLP_image (trainable) ------┘
"""

import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Hugging Face transformers for LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# Assuming you have these custom modules in your src/
# from src.models.lidar_encoder import LidarCLIPEncoder
# from src.models.image_encoder import CLIPImageEncoder
# from src.data.nuscenes_dataset import NuScenesMultimodalDataset


# ============================================================================
# 1. MLP Projection Layer
# ============================================================================

class MLPProjection(nn.Module):
    """
    Multi-layer perceptron for projecting encoder features to LLM token embedding space.
    
    Input: Feature vectors from encoders (e.g., [batch, encoder_dim])
    Output: Projected embeddings matching LLM token dimension [batch, llm_dim]
    
    Args:
        input_dim: Dimension of encoder output features
        output_dim: Dimension of LLM token embeddings
        hidden_dim: Dimension of hidden layer(s)
        num_layers: Number of linear layers (minimum 2)
        dropout: Dropout probability for regularization
    """
    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=2, dropout=0.1):
        super().__init__()
        
        layers = []
        
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Final layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            Projected features of shape [batch, seq_len, output_dim] or [batch, output_dim]
        """
        return self.projection(x)


# ============================================================================
# 2. Multimodal Model Wrapper
# ============================================================================

class MultimodalLLM(nn.Module):
    """
    Complete multimodal model combining:
    - LiDAR encoder (frozen)
    - Image encoder (frozen)
    - MLP projections (trainable)
    - LLM (frozen)
    
    Input: Dictionary with 'lidar', 'image', 'text_input_ids', 'labels'
    Output: LLM output logits for next token prediction
    """
    def __init__(
        self,
        lidar_encoder,
        image_encoder,
        llm,
        lidar_dim,
        image_dim,
        llm_dim,
        mlp_hidden_dim=2048,
        mlp_num_layers=2,
        mlp_dropout=0.1
    ):
        super().__init__()
        
        # Frozen encoders
        self.lidar_encoder = lidar_encoder
        self.image_encoder = image_encoder
        self.llm = llm
        
        # Freeze encoders and LLM
        for param in self.lidar_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Trainable MLP projections
        self.lidar_projection = MLPProjection(
            input_dim=lidar_dim,
            output_dim=llm_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout
        )
        
        self.image_projection = MLPProjection(
            input_dim=image_dim,
            output_dim=llm_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout
        )
        
        self.llm_dim = llm_dim
        
    def forward(self, lidar_data, image_data, text_input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the entire multimodal pipeline.
        
        Args:
            lidar_data: LiDAR point cloud data [batch, points, features]
            image_data: Image tensor [batch, channels, height, width]
            text_input_ids: Tokenized text input [batch, seq_len]
            attention_mask: Attention mask for text [batch, seq_len]
            labels: Ground truth tokens for loss computation [batch, seq_len]
            
        Returns:
            Dictionary containing:
                - logits: LLM output logits [batch, total_seq_len, vocab_size]
                - loss: Computed loss (if labels provided)
        """
        batch_size = image_data.shape[0]
        
        # 1. Encode LiDAR (frozen)
        with torch.no_grad():
            lidar_features = self.lidar_encoder(lidar_data)  # [batch, lidar_dim]
        
        # 2. Encode Image (frozen)
        with torch.no_grad():
            image_features = self.image_encoder(image_data)  # [batch, image_dim]
        
        # 3. Project to LLM token space (trainable)
        lidar_tokens = self.lidar_projection(lidar_features)  # [batch, llm_dim]
        image_tokens = self.image_projection(image_features)  # [batch, llm_dim]
        
        # Add sequence dimension if needed: [batch, llm_dim] -> [batch, 1, llm_dim]
        if len(lidar_tokens.shape) == 2:
            lidar_tokens = lidar_tokens.unsqueeze(1)
        if len(image_tokens.shape) == 2:
            image_tokens = image_tokens.unsqueeze(1)
        
        # 4. Get text embeddings from LLM
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)  # [batch, seq_len, llm_dim]
        
        # 5. Concatenate: [lidar_token, image_token, text_tokens]
        # Shape: [batch, 1 + 1 + seq_len, llm_dim]
        combined_embeddings = torch.cat([lidar_tokens, image_tokens, text_embeddings], dim=1)
        
        # 6. Update attention mask to account for LiDAR and image tokens
        if attention_mask is not None:
            # Add attention for LiDAR and image tokens (both are attended to)
            prefix_attention = torch.ones(
                batch_size, 2, 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        
        # 7. Forward through LLM (frozen)
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def get_trainable_parameters(self):
        """Returns only the trainable parameters (MLP projections)"""
        trainable_params = []
        trainable_params.extend(self.lidar_projection.parameters())
        trainable_params.extend(self.image_projection.parameters())
        return trainable_params


# ============================================================================
# 3. Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer, global_step):
    """
    Train for one epoch.
    
    Args:
        model: MultimodalLLM model
        dataloader: Training data loader
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: torch device (cuda/cpu)
        epoch: Current epoch number
        writer: TensorBoard writer
        global_step: Global training step counter
        
    Returns:
        avg_loss: Average loss for the epoch
        global_step: Updated global step counter
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        lidar_data = batch['lidar'].to(device)
        image_data = batch['image'].to(device)
        text_input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            lidar_data=lidar_data,
            image_data=image_data,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        current_lr = scheduler.get_last_lr()[0]
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # TensorBoard logging
        if global_step % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
        
        global_step += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def validate(model, dataloader, device, epoch, writer):
    """
    Validate the model on validation set.
    
    Args:
        model: MultimodalLLM model
        dataloader: Validation data loader
        device: torch device
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            lidar_data = batch['lidar'].to(device)
            image_data = batch['image'].to(device)
            text_input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                lidar_data=lidar_data,
                image_data=image_data,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / num_batches
    
    # Log to TensorBoard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, save_path):
    """
    Save model checkpoint.
    
    Args:
        model: MultimodalLLM model
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch
        global_step: Global step counter
        loss: Current loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


# ============================================================================
# 4. Main Training Script
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train multimodal MLP projections')
    
    # Model paths
    parser.add_argument('--lidar-encoder-path', type=str, required=True,
                        help='Path to pretrained LidarCLIP encoder')
    parser.add_argument('--image-encoder-path', type=str, default='openai/clip-vit-large-patch14',
                        help='Path or HF model name for CLIP image encoder')
    parser.add_argument('--llm-path', type=str, default='Qwen/Qwen2-7B',
                        help='Path or HF model name for LLM')
    
    # Data paths
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='nuScenes dataset version')
    
    # Model architecture
    parser.add_argument('--lidar-dim', type=int, default=768,
                        help='LiDAR encoder output dimension')
    parser.add_argument('--image-dim', type=int, default=1024,
                        help='Image encoder output dimension (CLIP large = 1024)')
    parser.add_argument('--llm-dim', type=int, default=4096,
                        help='LLM embedding dimension (Qwen2-7B = 4096)')
    parser.add_argument('--mlp-hidden-dim', type=int, default=2048,
                        help='Hidden dimension for MLP projections')
    parser.add_argument('--mlp-num-layers', type=int, default=2,
                        help='Number of layers in MLP projections')
    parser.add_argument('--mlp-dropout', type=float, default=0.1,
                        help='Dropout rate for MLP projections')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Number of warmup steps for LR scheduler')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpointing and logging
    parser.add_argument('--output-dir', type=str, default='./experiments/runs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def setup_logging(output_dir):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)


def main():
    # Parse arguments
    args = parse_args()
    
    # Create experiment directory
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logging.info(f"Starting experiment: {args.exp_name}")
    logging.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # ========================================================================
    # Load Models
    # ========================================================================
    logging.info("Loading models...")
    
    # TODO: Load your custom LidarCLIP encoder
    # lidar_encoder = LidarCLIPEncoder.from_pretrained(args.lidar_encoder_path)
    # For now, using a placeholder
    logging.info(f"Loading LiDAR encoder from {args.lidar_encoder_path}")
    # lidar_encoder = ...  # Your custom loader
    
    # Load CLIP image encoder
    logging.info(f"Loading CLIP image encoder: {args.image_encoder_path}")
    # from transformers import CLIPVisionModel
    # image_encoder = CLIPVisionModel.from_pretrained(args.image_encoder_path)
    
    # Load LLM
    logging.info(f"Loading LLM: {args.llm_path}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    
    # Create multimodal model
    logging.info("Creating multimodal model...")
    # model = MultimodalLLM(
    #     lidar_encoder=lidar_encoder,
    #     image_encoder=image_encoder,
    #     llm=llm,
    #     lidar_dim=args.lidar_dim,
    #     image_dim=args.image_dim,
    #     llm_dim=args.llm_dim,
    #     mlp_hidden_dim=args.mlp_hidden_dim,
    #     mlp_num_layers=args.mlp_num_layers,
    #     mlp_dropout=args.mlp_dropout
    # )
    # model = model.to(device)
    
    # Log trainable parameters
    # trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    # total_params = sum(p.numel() for p in model.parameters())
    # logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
    #              f"({100 * trainable_params / total_params:.2f}%)")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    logging.info("Loading datasets...")
    
    # TODO: Create your custom dataset
    # train_dataset = NuScenesMultimodalDataset(
    #     dataroot=args.dataroot,
    #     version=args.version,
    #     split='train',
    #     tokenizer=tokenizer
    # )
    # val_dataset = NuScenesMultimodalDataset(
    #     dataroot=args.dataroot,
    #     version=args.version,
    #     split='val',
    #     tokenizer=tokenizer
    # )
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )
    
    # logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # ========================================================================
    # Setup Training
    # ========================================================================
    logging.info("Setting up optimizer and scheduler...")
    
    # Optimizer - only for trainable MLP parameters
    # optimizer = optim.AdamW(
    #     model.get_trainable_parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay
    # )
    
    # Learning rate scheduler with warmup
    # from torch.optim.lr_scheduler import OneCycleLR
    # total_steps = len(train_loader) * args.num_epochs
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=args.learning_rate,
    #     total_steps=total_steps,
    #     pct_start=args.warmup_steps / total_steps,
    #     anneal_strategy='cos'
    # )
    
    # TensorBoard writer
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from:
        logging.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    logging.info("Starting training...")
    
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"\n{'='*50}")
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logging.info(f"{'='*50}")
        
        # Train
        # train_loss, global_step = train_epoch(
        #     model, train_loader, optimizer, scheduler, 
        #     device, epoch, writer, global_step
        # )
        # logging.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        # val_loss = validate(model, val_loader, device, epoch, writer)
        # logging.info(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        # if (epoch + 1) % args.save_every == 0:
        #     checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        #     save_checkpoint(
        #         model, optimizer, scheduler, epoch, global_step, train_loss, checkpoint_path
        #     )
        
        # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model_path = output_dir / 'best_model.pt'
        #     save_checkpoint(
        #         model, optimizer, scheduler, epoch, global_step, val_loss, best_model_path
        #     )
        #     logging.info(f"New best model saved with val loss: {val_loss:.4f}")
    
    logging.info("\nTraining completed!")
    writer.close()
    
    # Save final model
    # final_model_path = output_dir / 'final_model.pt'
    # save_checkpoint(
    #     model, optimizer, scheduler, args.num_epochs - 1, global_step, train_loss, final_model_path
    # )


if __name__ == '__main__':
    main()