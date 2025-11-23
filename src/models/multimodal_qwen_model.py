"""
Multimodal Qwen Model
=====================

Integrates:
- CLIP Vision Encoder (pre-trained, frozen)
- SST LiDAR Encoder from LidarCLIP (pre-trained, frozen)  
- Qwen LLM (pre-trained, frozen)
- MLP Projectors for vision and LiDAR (trainable)

Architecture:
    Image  → CLIP (frozen) → Vision MLP (trainable) ┐
                                                      ├→ Qwen LLM (frozen) → Output
    LiDAR  → SST (frozen)  → LiDAR MLP (trainable)  ┘
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import SST encoder
from src.models.sst import LidarEncoderSST


class MLPProjector(nn.Module):
    """
    Multi-layer perceptron for projecting encoder features to LLM embedding space.
    
    Args:
        input_dim: Encoder output dimension
        output_dim: LLM embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers (minimum 2)
        dropout: Dropout probability
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, dropout=0.1):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim] or [B, seq_len, input_dim]
        Returns:
            [B, output_dim] or [B, seq_len, output_dim]
        """
        return self.projection(x)


class MultimodalQwenModel(nn.Module):
    """
    Complete multimodal model for autonomous driving trajectory prediction.
    Combines CLIP vision, SST LiDAR, and Qwen LLM with trainable projectors.
    """
    
    def __init__(
        self,
        device,
        qwen_model_name="Qwen/Qwen2.5-3B-Instruct",
        clip_model_name="openai/clip-vit-large-patch14",
        sst_config_path="src/models/configs/sst_encoder_only_config.py",
        lidarclip_checkpoint_path=None,
        freeze_encoders=True,
        freeze_llm=True,
        mlp_hidden_dim=2048,
        mlp_num_layers=3,
        mlp_dropout=0.1
    ):
        super().__init__()
        
        self.device = device
        self.freeze_encoders = freeze_encoders
        self.freeze_llm = freeze_llm
        
        print("\n" + "="*70)
        print("MultimodalQwenModel Initialization")
        print("="*70)
        
        # ====================================================================
        # 1. CLIP Vision Encoder (Pre-trained, Frozen)
        # ====================================================================
        print("\n[1/5] Loading CLIP vision encoder...")
        self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name).to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        
        clip_hidden_size = self.vision_tower.config.hidden_size
        print(f"  ✓ CLIP loaded: {clip_model_name}")
        print(f"  ✓ Hidden size: {clip_hidden_size}")
        
        if freeze_encoders:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            self.vision_tower.eval()
            print(f"  ✓ CLIP frozen")
        
        # ====================================================================
        # 2. SST LiDAR Encoder (Pre-trained from LidarCLIP, Frozen)
        # ====================================================================
        print("\n[2/5] Loading SST LiDAR encoder...")
        self.lidar_encoder = LidarEncoderSST(
            sst_config=sst_config_path,
            clip_embedding_dim=clip_hidden_size,  # Output same dim as CLIP
            checkpoint=lidarclip_checkpoint_path
        )
        self.lidar_encoder = self.lidar_encoder.to(device)
        
        lidar_hidden_size = clip_hidden_size  # SST outputs CLIP-compatible features
        print(f"  ✓ SST encoder loaded")
        print(f"  ✓ Output size: {lidar_hidden_size}")
        
        if freeze_encoders:
            for param in self.lidar_encoder.parameters():
                param.requires_grad = False
            self.lidar_encoder.eval()
            print(f"  ✓ SST frozen")
        
        # ====================================================================
        # 3. Qwen LLM (Pre-trained, Frozen)
        # ====================================================================
        print(f"\n[3/5] Loading Qwen LLM: {qwen_model_name}...")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        
        qwen_hidden_size = self.language_model.config.hidden_size
        print(f"  ✓ Qwen loaded")
        print(f"  ✓ Hidden size: {qwen_hidden_size}")
        
        if freeze_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False
            self.language_model.eval()
            print(f"  ✓ Qwen frozen")
        
        # ====================================================================
        # 4. MLP Projectors (Trainable)
        # ====================================================================
        print("\n[4/5] Creating MLP projectors...")
        
        # Vision projector: CLIP features → Qwen embedding space
        self.vision_projector = MLPProjector(
            input_dim=clip_hidden_size,
            output_dim=qwen_hidden_size,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout
        ).to(device).to(torch.bfloat16)
        
        print(f"  ✓ Vision MLP: {clip_hidden_size} → {qwen_hidden_size}")
        
        # LiDAR projector: SST features → Qwen embedding space
        self.lidar_projector = MLPProjector(
            input_dim=lidar_hidden_size,
            output_dim=qwen_hidden_size,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout
        ).to(device).to(torch.bfloat16)
        
        print(f"  ✓ LiDAR MLP: {lidar_hidden_size} → {qwen_hidden_size}")
        
        # ====================================================================
        # 5. Prompts for Trajectory Prediction
        # ====================================================================
        print("\n[5/5] Setting up prompts...")
        self.prompt_part1 = (
            "You are a self-driving car. Your task is to predict the future trajectory "
            "based on the camera image, LiDAR point cloud, and your recent movement. "
            "Your last three recorded positions (x, y) are: "
        )
        self.prompt_part2 = (
            "It is critical that you output exactly 10 waypoints. "
            "The trajectory must be formatted as a sequence of 10 2D coordinates [x, y]. "
            "For example:\n"
            "Future Trajectory: [[x1, y1], [x2, y2], ..., [x10, y10]]"
        )
        
        print("="*70)
        print("✓ Initialization Complete!")
        print("="*70 + "\n")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print model summary."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print("\n" + "="*70)
        print("Model Summary")
        print("="*70)
        print(f"Vision Encoder:  {'FROZEN' if self.freeze_encoders else 'TRAINABLE'}")
        print(f"LiDAR Encoder:   {'FROZEN' if self.freeze_encoders else 'TRAINABLE'}")
        print(f"LLM:             {'FROZEN' if self.freeze_llm else 'TRAINABLE'}")
        print(f"Vision MLP:      TRAINABLE")
        print(f"LiDAR MLP:       TRAINABLE")
        print(f"\nTotal params:      {total_params:,}")
        print(f"Trainable params:  {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print("="*70 + "\n")
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters for optimizer."""
        trainable_params = []
        
        # Always train projectors
        trainable_params.extend(self.vision_projector.parameters())
        trainable_params.extend(self.lidar_projector.parameters())
        
        # Optionally train encoders
        if not self.freeze_encoders:
            trainable_params.extend(self.vision_tower.parameters())
            trainable_params.extend(self.lidar_encoder.parameters())
        
        # Optionally train LLM
        if not self.freeze_llm:
            trainable_params.extend(self.language_model.parameters())
        
        return trainable_params
    
    def encode_image(self, images):
        """
        Encode images using CLIP.
        
        Args:
            images: Preprocessed images (pixel_values tensor)
        
        Returns:
            [B, hidden_dim] image features
        """
        with torch.no_grad() if self.freeze_encoders else torch.enable_grad():
            # CLIP outputs [B, num_patches, hidden_dim]
            # We take the CLS token (first token)
            image_features = self.vision_tower(pixel_values=images).last_hidden_state
            # Use mean pooling instead of just CLS token for richer features
            image_features = image_features.mean(dim=1)  # [B, hidden_dim]
        
        return image_features
    
    def encode_lidar(self, point_clouds):
        """
        Encode LiDAR point clouds using SST.
        
        Args:
            point_clouds: List of [N_i, 4] point cloud tensors
        
        Returns:
            [B, hidden_dim] LiDAR features
        """
        with torch.no_grad() if self.freeze_encoders else torch.enable_grad():
            lidar_features = self.lidar_encoder(point_clouds)  # [B, hidden_dim]
        
        return lidar_features
    
    def forward(self, images, point_clouds, input_ids, use_vision=True, use_lidar=True):
        """
        Forward pass for training.
        
        Args:
            images: List of PIL images or preprocessed tensors
            point_clouds: List of [N_i, 4] point cloud tensors
            input_ids: Tokenized text input [B, seq_len]
            use_vision: Whether to use vision features
            use_lidar: Whether to use LiDAR features
        
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        batch_size = input_ids.shape[0]
        
        # Encode modalities
        multimodal_features = []
        
        if use_vision and images is not None:
            # Process images if they are a list (convert to batch tensor)
            if isinstance(images, list):
                # Use image processor to convert list of images to batch tensor
                processed = self.image_processor(
                    images=images,
                    return_tensors="pt"
                )
                images = processed.pixel_values.to(self.device)
            
            # Encode images
            image_features = self.encode_image(images)  # [B, clip_dim]
            # Project to LLM space
            projected_vision = self.vision_projector(image_features.to(torch.bfloat16))  # [B, qwen_dim]
            multimodal_features.append(projected_vision.unsqueeze(1))  # [B, 1, qwen_dim]
        
        if use_lidar and point_clouds is not None and len(point_clouds) > 0:
            # Encode LiDAR
            lidar_features = self.encode_lidar(point_clouds)  # [B, lidar_dim]
            # Project to LLM space
            projected_lidar = self.lidar_projector(lidar_features.to(torch.bfloat16))  # [B, qwen_dim]
            multimodal_features.append(projected_lidar.unsqueeze(1))  # [B, 1, qwen_dim]
        
        # Get text embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)  # [B, seq_len, qwen_dim]
        
        # Concatenate all features: [multimodal tokens] + [text tokens]
        if len(multimodal_features) > 0:
            multimodal_tokens = torch.cat(multimodal_features, dim=1)  # [B, num_modalities, qwen_dim]
            combined_embeddings = torch.cat([multimodal_tokens, text_embeddings], dim=1)
        else:
            combined_embeddings = text_embeddings
        
        # Forward through LLM
        outputs = self.language_model(inputs_embeds=combined_embeddings)
        
        return outputs.logits
    
    def generate_trajectory(self, images, point_clouds, ego_positions):
        """
        Generate trajectory predictions (inference mode).
        
        Args:
            images: Preprocessed images
            point_clouds: List of point clouds
            ego_positions: Past ego positions
        
        Returns:
            outputs: Generation outputs
            generated_text: Decoded text
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = len(ego_positions)
            
            # Encode modalities
            multimodal_features = []
            
            if images is not None:
                # Process images if they are a list
                if isinstance(images, list):
                    processed = self.image_processor(
                        images=images,
                        return_tensors="pt"
                    )
                    images = processed.pixel_values.to(self.device)
                
                image_features = self.encode_image(images)
                projected_vision = self.vision_projector(image_features.to(torch.bfloat16))
                multimodal_features.append(projected_vision.unsqueeze(1))
            
            if point_clouds is not None and len(point_clouds) > 0:
                lidar_features = self.encode_lidar(point_clouds)
                projected_lidar = self.lidar_projector(lidar_features.to(torch.bfloat16))
                multimodal_features.append(projected_lidar.unsqueeze(1))
            
            # Create prompts
            prompts = []
            for pos_tensor in ego_positions:
                pos_list = [f"[{pos[0]:.2f}, {pos[1]:.2f}]" for pos in pos_tensor]
                pos_str = ", ".join(pos_list)
                final_prompt = f"{self.prompt_part1}[{pos_str}]\n{self.prompt_part2}"
                prompts.append(final_prompt)
            
            full_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for p in prompts
            ]
            
            inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True).to(self.device)
            text_embeddings = self.language_model.get_input_embeddings()(inputs.input_ids)
            
            # Combine features
            if len(multimodal_features) > 0:
                multimodal_tokens = torch.cat(multimodal_features, dim=1)
                combined_embeddings = torch.cat([multimodal_tokens, text_embeddings], dim=1)
            else:
                combined_embeddings = text_embeddings
            
            # Generate
            outputs = self.language_model.generate(
                inputs_embeds=combined_embeddings,
                max_new_tokens=2048,  # Increased from 512 (set to None for model's max context length)
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            generated_ids = outputs.sequences
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            return outputs, generated_text


# Test code
if __name__ == "__main__":
    print("Testing MultimodalQwenModel...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MultimodalQwenModel(
        device=device,
        qwen_model_name="Qwen/Qwen2.5-3B-Instruct",
        clip_model_name="openai/clip-vit-large-patch14",
        sst_config_path="src/models/configs/sst_encoder_only_config.py",
        lidarclip_checkpoint_path=None,  # Set to checkpoint path to load
        freeze_encoders=True,
        freeze_llm=True
    )
    
    print("\n✓ Model created successfully!")
    print(f"✓ Trainable parameters: {sum(p.numel() for p in model.get_trainable_parameters()):,}")
