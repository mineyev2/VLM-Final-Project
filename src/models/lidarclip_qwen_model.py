import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add LidarCLIP to path if needed
sys.path.append('./lidarclip')
from model.sst import LidarEncoderSST

class LidarCLIPQwenModel(nn.Module):
    """
    Multimodal model combining:
    - Qwen LLM backbone
    - CLIP vision encoder with MLP projection
    - LidarCLIP encoder with MLP projection
    """
    
    def __init__(
        self, 
        device, 
        qwen_model_name="Qwen/Qwen2.5-3B-Instruct",
        clip_model_name="openai/clip-vit-large-patch14",
        lidarclip_config_path="./lidarclip/model/sst_encoder_only_config.py",
        lidarclip_checkpoint_path=None,  # Path to pretrained LidarCLIP weights
        freeze_encoders=True,
        freeze_llm=True
    ):
        super().__init__()
        
        self.device = device
        print(f"Using device {self.device} for LidarCLIPQwenModel.")
        
        # ============================================================
        # 1. Load Vision Encoder (CLIP)
        # ============================================================
        print("Loading CLIP vision model...")
        self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name).to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        
        # ============================================================
        # 2. Load LiDAR Encoder (LidarCLIP's SST)
        # ============================================================
        print("Loading LidarCLIP encoder...")
        # CLIP ViT-L/14 has output dim of 768, ViT-B/32 has 512
        clip_hidden_size = self.vision_tower.config.hidden_size
        self.lidar_encoder = LidarEncoderSST(
            sst_config_path=lidarclip_config_path,
            clip_embedding_dim=clip_hidden_size  # Match CLIP dimension
        ).to(self.device)
        
# Load pretrained LidarCLIP weights if provided
        if lidarclip_checkpoint_path and os.path.exists(lidarclip_checkpoint_path):
            print(f"Loading pretrained LidarCLIP weights from {lidarclip_checkpoint_path}")
            checkpoint = torch.load(lidarclip_checkpoint_path, map_location=self.device)
            
            # Try different checkpoint formats
            if 'lidar_encoder' in checkpoint:
                lidar_state_dict = checkpoint['lidar_encoder']
            elif 'state_dict' in checkpoint:
                # PyTorch Lightning checkpoint format
                lidar_state_dict = {k.replace('lidar_encoder.', ''): v 
                                   for k, v in checkpoint['state_dict'].items() 
                                   if k.startswith('lidar_encoder.')}
            else:
                lidar_state_dict = checkpoint
            
            # Validate we actually loaded something
            if not lidar_state_dict:
                raise ValueError(f"No 'lidar_encoder' weights found in checkpoint. Keys: {list(checkpoint.keys())}")
            
            try:
                self.lidar_encoder.load_state_dict(lidar_state_dict, strict=True)
                print("✓ Loaded pretrained LidarCLIP weights successfully!")
            except RuntimeError as e:
                print(f"⚠️ Warning: Checkpoint loading encountered size mismatch: {e}")
                print("Attempting non-strict loading...")
                self.lidar_encoder.load_state_dict(lidar_state_dict, strict=False)
        
        # ============================================================
        # 3. Load Language Model (Qwen)
        # ============================================================
        print(f"Loading Qwen language model: {qwen_model_name}...")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        
        # Get dimensions for projection layers
        qwen_hidden_size = self.language_model.config.hidden_size
        lidar_output_size = clip_hidden_size  # LidarCLIP outputs CLIP-dim features
        
        # ============================================================
        # 4. Create MLP Projectors (Glue Layers)
        # ============================================================
        
        # Vision MLP Projector (CLIP → Qwen)
        self.vision_projector = nn.Sequential(
            nn.Linear(clip_hidden_size, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size),
        ).to(self.device).to(torch.bfloat16)
        
        # LiDAR MLP Projector (LidarCLIP → Qwen)
        self.lidar_projector = nn.Sequential(
            nn.Linear(lidar_output_size, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size),
        ).to(self.device).to(torch.bfloat16)
        
        # ============================================================
        # 5. Freezing Strategy
        # ============================================================
        if freeze_encoders:
            print("Freezing vision and LiDAR encoders...")
            self.vision_tower.requires_grad_(False)
            self.lidar_encoder.requires_grad_(False)
        
        if freeze_llm:
            print("Freezing language model...")
            self.language_model.requires_grad_(False)
        
        # ============================================================
        # 6. Define Prompts for Multimodal Input
        # ============================================================
        self.vision_token = "<vision>"
        self.lidar_token = "<lidar>"
        self.prompt_template = (
            "You are a self-driving car. "
            "Visual input: {vision_token}. "
            "LiDAR input: {lidar_token}. "
            "Your task is to predict the future trajectory based on the camera image, "
            "LiDAR point cloud, and your recent movement. "
            "Your last three recorded positions (x, y) are: {positions}. "
            "Output exactly 10 waypoints formatted as: "
            "Future Trajectory: [[x1, y1], [x2, y2], ..., [x10, y10]]"
        )
    
    def forward(
        self, 
        images=None,           # Camera images
        point_clouds=None,     # LiDAR point clouds (list of tensors)
        input_ids=None,        # Text token IDs
        use_vision=True,       # Whether to use vision input
        use_lidar=True,        # Whether to use LiDAR input
        return_features=False  # Return intermediate features for analysis
    ):
        """
        Forward pass through the multimodal model.
        
        Args:
            images: Processed images tensor [batch, channels, height, width]
            point_clouds: List of point cloud tensors, each [num_points, 4]
            input_ids: Tokenized text input [batch, seq_len]
            use_vision: Whether to include vision features
            use_lidar: Whether to include LiDAR features
            return_features: If True, also return extracted features
        
        Returns:
            logits: Language model output logits
            features (optional): Dict with 'vision' and 'lidar' features
        """
        
        batch_size = len(point_clouds) if point_clouds else (
            images.shape[0] if images is not None else input_ids.shape[0]
        )
        
        features_dict = {}
        multimodal_embeddings = []
        
        # ============================================================
        # 1. Process Vision Input
        # ============================================================
        if use_vision and images is not None:
            # Extract vision features
            with torch.no_grad() if self.vision_tower.training == False else torch.enable_grad():
                vision_outputs = self.vision_tower(pixel_values=images)
                vision_features = vision_outputs.last_hidden_state  # [batch, num_patches, hidden_size]
            
            # Project vision features to LLM space
            projected_vision = self.vision_projector(vision_features.to(torch.bfloat16))
            multimodal_embeddings.append(projected_vision)
            
            if return_features:
                features_dict['vision'] = vision_features.mean(dim=1)  # Global pool
        
        # ============================================================
        # 2. Process LiDAR Input
        # ============================================================
        if use_lidar and point_clouds is not None:
            # Extract LiDAR features
            with torch.no_grad() if self.lidar_encoder.training == False else torch.enable_grad():
                # LidarCLIP expects a list of point clouds
                lidar_features, attn_weights = self.lidar_encoder(point_clouds)
                # lidar_features: [batch, clip_hidden_size]
            
            # Add sequence dimension to match vision features format
            lidar_features = lidar_features.unsqueeze(1)  # [batch, 1, hidden_size]
            
            # Project LiDAR features to LLM space
            projected_lidar = self.lidar_projector(lidar_features.to(torch.bfloat16))
            multimodal_embeddings.append(projected_lidar)
            
            if return_features:
                features_dict['lidar'] = lidar_features.squeeze(1)
        
        # ============================================================
        # 3. Process Text Input
        # ============================================================
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # ============================================================
        # 4. Combine All Embeddings
        # ============================================================
        if multimodal_embeddings:
            # Concatenate all multimodal embeddings
            multimodal_embeds = torch.cat(multimodal_embeddings, dim=1)
            # Combine with text embeddings
            combined_embeddings = torch.cat([multimodal_embeds, text_embeddings], dim=1)
        else:
            combined_embeddings = text_embeddings
        
        # ============================================================
        # 5. Forward Through Language Model
        # ============================================================
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            use_cache=False,
            return_dict=True
        )
        
        if return_features:
            return outputs.logits, features_dict
        return outputs.logits
    
    def prepare_inputs(
        self,
        images,
        point_clouds,
        ego_positions,
        use_vision=True,
        use_lidar=True
    ):
        """
        Prepare inputs for training or inference.
        
        Args:
            images: List of PIL images or tensor
            point_clouds: List of point cloud tensors
            ego_positions: List of (x, y) positions
            use_vision: Whether to use vision input
            use_lidar: Whether to use LiDAR input
        
        Returns:
            Dict with prepared inputs for forward pass
        """
        
        prepared_inputs = {}
        
        # Process images if using vision
        if use_vision and images is not None:
            if not torch.is_tensor(images):
                image_inputs = self.image_processor(images=images, return_tensors="pt")
                prepared_inputs['images'] = image_inputs['pixel_values'].to(self.device)
            else:
                prepared_inputs['images'] = images.to(self.device)
        
        # Process point clouds if using LiDAR
        if use_lidar and point_clouds is not None:
            # Ensure point clouds are on the correct device
            prepared_inputs['point_clouds'] = [pc.to(self.device) for pc in point_clouds]
        
        # Prepare text prompt
        positions_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_positions])
        prompt = self.prompt_template.format(
            vision_token=self.vision_token if use_vision else "not available",
            lidar_token=self.lidar_token if use_lidar else "not available",
            positions=positions_str
        )
        
        # Tokenize prompt
        text_inputs = self.tokenizer(prompt, return_tensors="pt")
        prepared_inputs['input_ids'] = text_inputs['input_ids'].squeeze(0).to(self.device)
        
        prepared_inputs['use_vision'] = use_vision
        prepared_inputs['use_lidar'] = use_lidar
        
        return prepared_inputs
    
    def get_trainable_parameters(self):
        """Return list of trainable parameters for optimizer."""
        params = []
        
        # Always train projection layers
        params.extend(list(self.vision_projector.parameters()))
        params.extend(list(self.lidar_projector.parameters()))
        
        # Optionally train encoders
        if self.vision_tower.training:
            params.extend(list(self.vision_tower.parameters()))
        if self.lidar_encoder.training:
            params.extend(list(self.lidar_encoder.parameters()))
        
        # Optionally train LLM
        if self.language_model.training:
            params.extend(list(self.language_model.parameters()))
        
        return params
    
    def save_projectors(self, save_path):
        """Save only the trained projector weights."""
        torch.save({
            'vision_projector': self.vision_projector.state_dict(),
            'lidar_projector': self.lidar_projector.state_dict(),
        }, save_path)
        print(f"Saved projectors to {save_path}")
    
    def load_projectors(self, load_path):
        """Load projector weights."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.vision_projector.load_state_dict(checkpoint['vision_projector'])
        self.lidar_projector.load_state_dict(checkpoint['lidar_projector'])
        print(f"Loaded projectors from {load_path}")


# Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = LidarCLIPQwenModel(
        device=device,
        qwen_model_name="Qwen/Qwen2.5-3B-Instruct",
        clip_model_name="openai/clip-vit-large-patch14",
        lidarclip_config_path="./lidarclip/model/sst_encoder_only_config.py",
        lidarclip_checkpoint_path="./checkpoints/lidarclip_vit-l-14.pth",  # Optional
        freeze_encoders=True,  # Freeze pretrained encoders
        freeze_llm=True  # Freeze LLM, only train projectors
    )
    
    # Example forward pass
    batch_size = 2
    
    # Dummy inputs
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_point_clouds = [torch.randn(1000, 4).to(device) for _ in range(batch_size)]
    dummy_ego_positions = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    
    # Prepare inputs
    inputs = model.prepare_inputs(
        images=dummy_images,
        point_clouds=dummy_point_clouds,
        ego_positions=dummy_ego_positions,
        use_vision=True,
        use_lidar=True
    )
    
    # Forward pass
    logits = model(**inputs)
    print(f"Output shape: {logits.shape}")
    
    # Get features for analysis
    logits, features = model(
        images=dummy_images,
        point_clouds=dummy_point_clouds,
        input_ids=inputs['input_ids'].unsqueeze(0),
        return_features=True
    )
    print(f"Vision features shape: {features['vision'].shape}")
    print(f"LiDAR features shape: {features['lidar'].shape}")