import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Import the SST encoder wrapper (OpenMMLab v2 compatible)
from .sst import LidarEncoderSST


class LidarCLIPQwenModel(nn.Module):
    """
    Multimodal model combining:
    - Qwen LLM backbone
    - CLIP vision encoder with MLP projection
    - LiDAR SST encoder with AttentionPool2d and MLP projection
    """

    def __init__(
        self,
        device,
        qwen_model_name="Qwen/Qwen2.5-3B-Instruct",
        clip_model_name="openai/clip-vit-large-patch14",
        lidarclip_config_path="./lidarclip/model/sst_encoder_only_config.py",
        lidarclip_checkpoint_path=None,   # Path to pretrained LiDAR encoder weights (optional)
        freeze_encoders=True,
        freeze_llm=True,
    ):
        super().__init__()

        self.device = device
        print(f"Using device {self.device} for LidarCLIPQwenModel.")

        # ================================
        # 1) Vision Encoder (CLIP)
        # ================================
        print("Loading CLIP vision model...")
        self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name).to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        clip_hidden_size = self.vision_tower.config.hidden_size  # ViT-L/14 -> 1024 or 768 depending on variant

        # ================================
        # 2) LiDAR Encoder (SST wrapper)
        # ================================
        print("Loading LidarCLIP encoder...")
        # Use the new LidarEncoderSST API (no legacy kwargs)
        self.lidar_encoder = LidarEncoderSST(
            sst_config=lidarclip_config_path,
            clip_embedding_dim=clip_hidden_size,
            checkpoint=lidarclip_checkpoint_path,
        ).to(self.device)

        # ================================
        # 3) Language Model (Qwen)
        # ================================
        print(f"Loading Qwen language model: {qwen_model_name}...")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",   # allow HF to shard if multiple GPUs exist
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

        qwen_hidden_size = self.language_model.config.hidden_size
        lidar_output_size = clip_hidden_size  # by construction, SST -> AttentionPool -> CLIP-dim

        # ================================
        # 4) Projectors (to Qwen hidden)
        # ================================
        proj_w = qwen_hidden_size * 4
        self.vision_projector = nn.Sequential(
            nn.Linear(clip_hidden_size, proj_w),
            nn.GELU(),
            nn.Linear(proj_w, proj_w),
            nn.GELU(),
            nn.Linear(proj_w, qwen_hidden_size),
        ).to(self.device).to(self.language_model.dtype)

        self.lidar_projector = nn.Sequential(
            nn.Linear(lidar_output_size, proj_w),
            nn.GELU(),
            nn.Linear(proj_w, proj_w),
            nn.GELU(),
            nn.Linear(proj_w, qwen_hidden_size),
        ).to(self.device).to(self.language_model.dtype)

        # ================================
        # 5) Freezing strategy
        # ================================
        if freeze_encoders:
            print("Freezing vision and LiDAR encoders...")
            self.vision_tower.requires_grad_(False)
            self.lidar_encoder.requires_grad_(False)

        if freeze_llm:
            print("Freezing language model...")
            self.language_model.requires_grad_(False)

        # ================================
        # 6) Prompt bits (kept as-is)
        # ================================
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

    # ----------------------------------------------------------------------

    def forward(
        self,
        images=None,            # [B, 3, H, W] preprocessed tensor
        point_clouds=None,      # list of length B, each (N_i, 4) tensor
        input_ids=None,         # [B, L] token ids
        use_vision=True,
        use_lidar=True,
        return_features=False,  # if True, return {vision, lidar} features and attn
    ):
        """
        Returns:
            logits  (and optionally features dict)
        """
        # Ensure input_ids is [B, L]
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        dtype = self.language_model.dtype
        features_dict = {}
        multimodal_embeddings = []

        # ---------------- Vision ----------------
        if use_vision and images is not None:
            # CLIP forward: returns last_hidden_state [B, P, C]
            with torch.no_grad() if not self.vision_tower.training else torch.enable_grad():
                vision_outputs = self.vision_tower(pixel_values=images.to(self.device))
                vision_features = vision_outputs.last_hidden_state  # [B, P, C]

            # Project to Qwen hidden (keep sequence dim)
            projected_vision = self.vision_projector(vision_features.to(dtype))
            multimodal_embeddings.append(projected_vision)

            if return_features:
                # Global mean for a compact summary
                features_dict['vision'] = vision_features.mean(dim=1).to(dtype)

        # ---------------- LiDAR -----------------
        if use_lidar and point_clouds is not None:
            with torch.no_grad() if not self.lidar_encoder.training else torch.enable_grad():
                # Ask for attention only if user requested features
                if return_features:
                    lidar_features, attn_weights = self.lidar_encoder(
                        point_clouds, return_attention=True
                    )
                else:
                    lidar_features = self.lidar_encoder(point_clouds)  # (B, D)
                    attn_weights = None

            # Match the sequence shape: add a single \"token\" for LiDAR
            lidar_features = lidar_features.unsqueeze(1)  # [B, 1, D]
            projected_lidar = self.lidar_projector(lidar_features.to(dtype))
            multimodal_embeddings.append(projected_lidar)

            if return_features:
                features_dict['lidar'] = lidar_features.squeeze(1).to(dtype)
                if attn_weights is not None:
                    features_dict['lidar_attention'] = attn_weights

        # ---------------- Text ------------------
        if input_ids is None:
            raise ValueError("input_ids must be provided for language modeling.")
        text_embeddings = self.language_model.get_input_embeddings()(input_ids.to(self.device))
        text_embeddings = text_embeddings.to(dtype)

        # ------------- Fuse & Decode ------------
        if multimodal_embeddings:
            multimodal_embeds = torch.cat(multimodal_embeddings, dim=1)  # [B, M, H]
            combined_embeddings = torch.cat([multimodal_embeds, text_embeddings], dim=1)  # [B, M+L, H]
        else:
            combined_embeddings = text_embeddings  # [B, L, H]

        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            use_cache=False,
            return_dict=True,
        )

        if return_features:
            return outputs.logits, features_dict
        return outputs.logits

    # ----------------------------------------------------------------------

    def prepare_inputs(
        self,
        images,
        point_clouds,
        ego_positions,
        use_vision=True,
        use_lidar=True,
    ):
        """
        Prepare & move inputs to device. Returns a dict for forward().
        """
        prepared = {}

        # Vision
        if use_vision and images is not None:
            if not torch.is_tensor(images):
                image_inputs = self.image_processor(images=images, return_tensors="pt")
                prepared['images'] = image_inputs['pixel_values'].to(self.device)
            else:
                prepared['images'] = images.to(self.device)

        # LiDAR
        if use_lidar and point_clouds is not None:
            prepared['point_clouds'] = [pc.to(self.device) for pc in point_clouds]

        # Prompt
        positions_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_positions])
        prompt = self.prompt_template.format(
            vision_token=self.vision_token if use_vision else "not available",
            lidar_token=self.lidar_token if use_lidar else "not available",
            positions=positions_str,
        )
        text_inputs = self.tokenizer(prompt, return_tensors="pt")
        prepared['input_ids'] = text_inputs['input_ids'].to(self.device)  # [1, L]

        prepared['use_vision'] = use_vision
        prepared['use_lidar'] = use_lidar
        return prepared

    # ----------------------------------------------------------------------

    def get_trainable_parameters(self):
        """Return list of params to optimize (projectors + any unfrozen blocks)."""
        params = []
        params.extend(self.vision_projector.parameters())
        params.extend(self.lidar_projector.parameters())

        if self.vision_tower.training:
            params.extend(self.vision_tower.parameters())
        if self.lidar_encoder.training:
            params.extend(self.lidar_encoder.parameters())
        if self.language_model.training:
            params.extend(self.language_model.parameters())
        return list(params)

    def save_projectors(self, save_path):
        torch.save(
            {
                'vision_projector': self.vision_projector.state_dict(),
                'lidar_projector': self.lidar_projector.state_dict(),
            },
            save_path,
        )
        print(f"Saved projectors to {save_path}")

    def load_projectors(self, load_path):
        ckpt = torch.load(load_path, map_location=self.device)
        self.vision_projector.load_state_dict(ckpt['vision_projector'])
        self.lidar_projector.load_state_dict(ckpt['lidar_projector'])
        print(f"Loaded projectors from {load_path}")


# Example (optional)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LidarCLIPQwenModel(
        device=device,
        qwen_model_name="Qwen/Qwen2.5-3B-Instruct",
        clip_model_name="openai/clip-vit-large-patch14",
        lidarclip_config_path="./lidarclip/model/sst_encoder_only_config.py",
        lidarclip_checkpoint_path=None,  # or path to .pth
        freeze_encoders=True,
        freeze_llm=True,
    )

    B = 2
    dummy_images = torch.randn(B, 3, 224, 224).to(device)
    dummy_point_clouds = [torch.randn(1000, 4).to(device) for _ in range(B)]
    dummy_positions = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

    inputs = model.prepare_inputs(
        images=dummy_images,
        point_clouds=dummy_point_clouds,
        ego_positions=dummy_positions,
        use_vision=True,
        use_lidar=True,
    )
    # Forward
    logits, feats = model(return_features=True, **inputs)
    print("logits:", logits.shape)
    print("vision summary:", feats['vision'].shape if 'vision' in feats else None)
    print("lidar summary:", feats['lidar'].shape if 'lidar' in feats else None)
