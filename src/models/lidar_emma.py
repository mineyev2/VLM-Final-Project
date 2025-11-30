import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
import sys
import os
from termcolor import colored

from src.models.base_model import BaseModel

# Import the SST encoder wrapper (OpenMMLab v2 compatible)
from .sst import LidarEncoderSST

from scripts.token_learner import TokenLearner

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class LidarEMMA(BaseModel):
    """
    Multimodal model combining:
    - Qwen LLM backbone
    - CLIP vision encoder with MLP projection
    - LiDAR SST encoder with AttentionPool2d and MLP projection
    
    This version is optimized for mmdet3d >= 1.0 (uses mmengine.Config)
    """

    def __init__(
        self,
        device,
        llm="Qwen/Qwen2.5-3B",
        clip_model_name="openai/clip-vit-large-patch14",
        lidarclip_config_path="src/models/mmdet3d/configs/sst_encoder_only_config.py",
        lidarclip_checkpoint_path=None,
        freeze_encoders=True,
        freeze_llm=True,
        use_lidar=False,
        lidar_pooling=False,
    ):
        super().__init__()

        self.device = device
        self.use_lidar = use_lidar
        self.lidar_pooling = lidar_pooling
        print(f"\n{'='*70}")
        print(f"Initializing LidarEMMA on device: {self.device}")
        print(f"{'='*70}\n")

        # ================================
        # 1) Vision Encoder (CLIP)
        # ================================
        print("[1/4] Loading CLIP vision model...")
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name).to(self.device)
            self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
            clip_hidden_size = self.vision_tower.config.hidden_size
            print(f"      ✓ CLIP loaded. Hidden size: {clip_hidden_size}")
        except Exception as e:
            print(f"      ✗ Failed to load CLIP: {e}")
            raise

        # ================================
        # 2) LiDAR Encoder (SST wrapper)
        # ================================
        print("\n[2/4] Loading LiDAR encoder (SST)...")
        if self.use_lidar:
            try:
                # Verify config path exists
                lidarclip_config_path = os.path.join(project_dir, lidarclip_config_path)
                if isinstance(lidarclip_config_path, str) and not os.path.isfile(lidarclip_config_path):
                    print(f"      Warning: Config path does not exist: {lidarclip_config_path}")
                    print(f"      Will attempt to load anyway (may fail if path is required)...")
                
                self.lidar_encoder = LidarEncoderSST(
                    sst_config=lidarclip_config_path,
                    clip_embedding_dim=clip_hidden_size,
                    checkpoint=lidarclip_checkpoint_path,
                ).to(self.device)
                print(f"      ✓ SST encoder loaded successfully")
            except KeyError as e:
                print(f"      ✗ Model registration error in SST:")
                print(f"         {e}")
                print(f"\n      Troubleshooting:")
                print(f"      • Check that mmdet3d >= 1.0 is installed")
                print(f"      • Ensure SST model is in your mmdet3d installation")
                print(f"      • Verify config file: {lidarclip_config_path}")
                raise
            except Exception as e:
                print(f"      ✗ Failed to load SST encoder: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Only use TokenLearner if pooling is disabled for lidar features
            if not lidar_pooling:
                self.lidar_token_learner = TokenLearner(in_channels=self.lidar_encoder.out_channels,
                                               num_tokens=256)

        else:
            print(colored("      Skipping LiDAR encoder setup (use_lidar=False)", "yellow"))
            self.lidar_encoder = None # TODO: Necessary?

        # ================================
        # 3) Language Model (Qwen)
        # ================================
        print("\n[3/4] Loading Qwen language model...")
        try:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                llm,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llm)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            qwen_hidden_size = self.language_model.config.hidden_size
            print(f"      ✓ Qwen loaded. Hidden size: {qwen_hidden_size}")
        except Exception as e:
            print(f"      ✗ Failed to load Qwen: {e}")
            raise
        
        lidar_output_size = clip_hidden_size  # By construction: SST -> AttentionPool -> CLIP-dim

        # ================================
        # 4) Projectors
        # ================================
        print("\n[4/4] Setting up projection layers...")
        try:
            proj_w = qwen_hidden_size * 4
            
            self.vision_projector = nn.Sequential(
                nn.Linear(clip_hidden_size, proj_w),
                nn.GELU(),
                nn.Linear(proj_w, proj_w),
                nn.GELU(),
                nn.Linear(proj_w, qwen_hidden_size),
            ).to(self.device).to(self.language_model.dtype)

            if self.use_lidar:
                if not lidar_pooling:
                    self.lidar_projector = nn.Sequential(
                        self.lidar_token_learner,
                        nn.Linear(self.lidar_encoder.out_channels, proj_w),
                        nn.GELU(),
                        nn.Linear(proj_w, proj_w),
                        nn.GELU(),
                        nn.Linear(proj_w, qwen_hidden_size),
                    ).to(self.device).to(self.language_model.dtype)
                else:
                    self.lidar_projector = nn.Sequential(
                        nn.Linear(lidar_output_size, proj_w),
                        nn.GELU(),
                        nn.Linear(proj_w, proj_w),
                        nn.GELU(),
                        nn.Linear(proj_w, qwen_hidden_size),
                    ).to(self.device).to(self.language_model.dtype)
            
            print(f"      ✓ Projector(s) initialized")
            print(f"        Vision: {clip_hidden_size} → {proj_w} → {qwen_hidden_size}")
            if self.use_lidar:
                print(f"        LiDAR:  {lidar_output_size} → {proj_w} → {qwen_hidden_size}")
            else:
                print(colored("        Skipping LiDAR projector setup (use_lidar=False)", "yellow"))
                self.lidar_projector = None # TODO: Necessary?
        
        except Exception as e:
            print(f"      ✗ Failed to setup projector(s): {e}")
            raise

        # ================================
        # 5) Freezing strategy
        # ================================
        self.freeze_encoders = freeze_encoders
        self.freeze_llm = freeze_llm

        print("\n[Freezing] Applying freeze strategy...")
        if self.freeze_encoders:
            print("      Freezing vision and LiDAR encoders")
            self.vision_tower.requires_grad_(False)
            if self.use_lidar:
                self.lidar_encoder.requires_grad_(False)
        else:
            print("      Vision and LiDAR encoders are TRAINABLE")

        if self.freeze_llm:
            print("      Freezing language model")
            self.language_model.requires_grad_(False)
        else:
            print("      Language model is TRAINABLE")

        print(f"\n{'='*70}")
        print("Model initialization complete!")
        print(f"{'='*70}\n")

        # # ================================
        # # 6) Prompt template
        # # ================================
        # self.vision_token = "<vision>"
        # self.lidar_token = "<lidar>"
        # self.prompt_template = (
        #     "You are a self-driving car. "
        #     "Visual input: {vision_token}. "
        #     "LiDAR input: {lidar_token}. "
        #     "Your task is to predict the future trajectory based on the camera image, "
        #     "LiDAR point cloud, and your recent movement. "
        #     "Your last three recorded positions (x, y) are: {positions}. "
        #     "Output exactly 10 waypoints formatted as: "
        #     "Future Trajectory: [[x1, y1], [x2, y2], ..., [x10, y10]]"
        # )

    # =========================================================================

    def train(self, mode=True):
        """Override train to handle freezing of components."""

        super().train(mode)  # Call parent to properly set training mode
        
        # Then override specific components if frozen
        if mode and self.freeze_encoders:
            self.vision_tower.eval()
            if self.use_lidar:
                self.lidar_encoder.eval()
        if mode and self.freeze_llm:
            self.language_model.eval()
        
        return self
    
    # def eval(self, mode=True):
    #     raise NotImplementedError("Custom eval method not implemented yet.")
    
    def prepare_multimodal_embeddings(self,
                                        images=None,
                                        point_clouds=None,
                                        input_ids=None,
                                        labels=None,
                                        return_features=False,
                                    ):
        
        # Ensure input_ids is [B, L]
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        dtype = self.language_model.dtype
        features_dict = {}
        multimodal_embeddings = []
        multimodal_len = 0

        # ====================================================================
        # Vision Encoder + Projector
        # ====================================================================
        if images is not None:
            with torch.no_grad() if not self.vision_tower.training else torch.enable_grad():
                vision_outputs = self.vision_tower(pixel_values=images.to(self.device))
                vision_features = vision_outputs.last_hidden_state  # [B, P, C]

            # Project to Qwen hidden (keep sequence dim)
            projected_vision = self.vision_projector(vision_features.to(dtype))
            multimodal_embeddings.append(projected_vision)
            multimodal_len += projected_vision.shape[1]

            if return_features:
                features_dict['vision'] = vision_features.mean(dim=1).to(dtype)
        else:
            raise ValueError("Image input is required for vision encoder.")

        # ====================================================================
        # LiDAR Encoder + Projector
        # ====================================================================
        if self.use_lidar and point_clouds is not None:
            with torch.no_grad() if not self.lidar_encoder.training else torch.enable_grad():
                if return_features:
                    lidar_features, attn_weights = self.lidar_encoder(
                        point_clouds,
                        no_pooling=(not self.lidar_pooling),
                        return_attention=True
                    )
                else:
                    # Using TokenLearner to compress information
                    lidar_features = self.lidar_encoder(point_clouds,
                                                        no_pooling=(not self.lidar_pooling))
                    attn_weights = None
            
            if self.lidar_pooling:
                lidar_features = lidar_features.unsqueeze(1)  # [B, 1, D]
            
            # Match sequence shape: add single "token" for LiDAR
            projected_lidar = self.lidar_projector(lidar_features.to(dtype))
            multimodal_embeddings.append(projected_lidar)
            multimodal_len += projected_lidar.shape[1]

            if return_features:
                features_dict['lidar'] = lidar_features.squeeze(1).to(dtype)
                if attn_weights is not None:
                    features_dict['lidar_attention'] = attn_weights
        elif self.use_lidar:
            raise ValueError("LiDAR input is required for LiDAR encoder.")

        # ====================================================================
        # Convert text to embeddings
        # ====================================================================
        if input_ids is None:
            raise ValueError("input_ids must be provided for language modeling.")
        text_embeddings = self.language_model.get_input_embeddings()(input_ids.to(self.device))
        text_embeddings = text_embeddings.to(dtype)

        # ====================================================================
        # Fuse embeddings
        # ====================================================================
        multimodal_embeds = torch.cat(multimodal_embeddings, dim=1)  # [B, M, H]
        combined_embeddings = torch.cat([multimodal_embeds, text_embeddings], dim=1)  # [B, M+L, H]

        # ====================================================================
        # Pad labels for Loss Computation
        # ====================================================================
        if labels is not None:
            # Create filler for multimodal tokens (ignore index -100)
            batch_size = labels.shape[0]

            if multimodal_len > 0:
                filler = torch.full(
                    (batch_size, multimodal_len),
                    -100,
                    dtype=labels.dtype,
                    device=self.device,
                )
                combined_labels = torch.cat([filler, labels.to(self.device)], dim=1)
            else:
                combined_labels = labels.to(self.device)
        else:
            combined_labels = None
        
        return combined_embeddings, combined_labels, features_dict, multimodal_len

    def forward(
        self,
        images=None,
        point_clouds=None,
        input_ids=None,
        labels=None,
        return_features=False,
    ):
        """
        Forward pass for the multimodal model.
        
        Args:
            images: [B, 3, H, W] preprocessed tensor (optional)
            point_clouds: list of length B, each (N_i, 4) tensor (optional)
            input_ids: [B, L] token ids (required)
            return_features: bool, if True return feature dict alongside logits
            
        Returns:
            logits: [B, L, V] model logits
            features_dict: (optional) dict with 'vision', 'lidar', etc.
        """

        # ====================================================================
        # Prepare multimodal embedddings
        # ====================================================================
        combined_embeddings, combined_labels, features_dict, _ = self.prepare_multimodal_embeddings(images=images, point_clouds=point_clouds, input_ids=input_ids, labels=labels, return_features=return_features)

        # ====================================================================
        # Decode final output with language model
        # ====================================================================
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            use_cache=False, # TODO: What is this?
            labels=combined_labels,
            return_dict=True, # TODO: What is this?
        )

        if return_features:
            return outputs.logits, features_dict
        return outputs

    # =========================================================================
    def generate_trajectory(
        self,
        text_attention_mask,
        images=None,
        point_clouds=None,
        input_ids=None,
        labels=None,
        return_features=False,
    ):
        # """
        # FOR INFERENCE: The non-differentiable generation method.
        # """

        # image_features = self.vision_tower(pixel_values=images).last_hidden_state
        # projected_image_features = self.mlp_projector(image_features.to(torch.bfloat16))

        # # --- Prompt formatting (same as before) ---
        # prompts = []
        # for pos_tensor in ego_positions:
        #     pos_list = [f"[{pos[0]:.2f}, {pos[1]:.2f}]" for pos in pos_tensor]
        #     pos_str = ", ".join(pos_list)
        #     final_prompt = f"{self.prompt_part1}[{pos_str}]\n{self.prompt_part2}"
        #     prompts.append(final_prompt)

        # if self.llm == "Qwen/Qwen2.5-3B":
        #     print(colored("Using Qwen2.5-3B prompt template...", "cyan"))
        #     full_prompts = [self.tokenizer.apply_chat_template(
        #         [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        #     ) for p in prompts]
        # else:
        #     full_prompts = [self.tokenizer.apply_chat_template(
        #         [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
        #     ) for p in prompts]
        
        # inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True).to(self.device)
        # text_embeddings = self.language_model.get_input_embeddings()(inputs.input_ids)
        # combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)

        # # Create attention mask for image embeddings (all real)
        # image_attention = torch.ones(projected_image_features.shape[:2], dtype=torch.long, device=self.device)
        # # Combine with text attention mask
        # combined_attention_mask = torch.cat([image_attention, inputs.attention_mask], dim=1)

        # text_attention_mask = input_ids.attention_mask

        # ====================================================================
        # Create attention mask
        # ====================================================================
        # image_attention = torch.ones(projected_image_features.shape[:2], dtype=torch.long, device=self.device)
        # if (self.use_lidar):



        # ====================================================================
        # Prepare multimodal embeddings
        # ====================================================================
        batch_size = images.shape[0]
        combined_embeddings, combined_labels, features_dict, multimodal_len = self.prepare_multimodal_embeddings(images=images, point_clouds=point_clouds, input_ids=input_ids, labels=labels, return_features=return_features)

        # ====================================================================
        # Generate attention mask
        # ====================================================================
        multimodal_attention_mask = torch.ones([batch_size, multimodal_len], dtype=torch.long, device=self.device)
        combined_attention_mask = torch.cat([multimodal_attention_mask, text_attention_mask], dim=1)

        # print("Attentnion mask for first element in batch:")
        # print(combined_attention_mask[0])

        # --- Generation with strict constraints to prevent thinking mode ---
        outputs = self.language_model.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            max_new_tokens=200,  # Reduced - trajectory should be ~100 tokens max
            min_new_tokens=50,   # Ensure it generates something substantial
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # do_sample=False,     # Greedy decoding for consistency
            # num_beams=5,         # No beam search
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_ids = outputs.sequences
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    # =========================================================================

    def _load_checkpoint_weights(self, checkpoint_data):
        pass # Implement later if needed

    def generateMotion(self, images, lidar, ego_pos_global):
        pixel_values = self.image_processor(images=images, return_tensors='pt').pixel_values.to(self.device)
        _, gen_text = self.generate_trajectory(pixel_values, lidar, ego_pos_global)
        return gen_text
    
    # def prepare_inputs(
    #     self,
    #     images,
    #     point_clouds,
    #     ego_positions,
    #     use_vision=True,
    #     use_lidar=True,
    # ):
    #     """
    #     Prepare & move inputs to device.
        
    #     Returns:
    #         dict with keys: 'images', 'point_clouds', 'input_ids', 'use_vision', 'use_lidar'
    #     """
    #     prepared = {}

    #     # Vision
    #     if use_vision and images is not None:
    #         if not torch.is_tensor(images):
    #             image_inputs = self.image_processor(images=images, return_tensors="pt")
    #             prepared['images'] = image_inputs['pixel_values'].to(self.device)
    #         else:
    #             prepared['images'] = images.to(self.device)

    #     # LiDAR
    #     if use_lidar and point_clouds is not None:
    #         prepared['point_clouds'] = [pc.to(self.device) for pc in point_clouds]

    #     # Prompt
    #     positions_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in ego_positions])
    #     prompt = self.prompt_template.format(
    #         vision_token=self.vision_token if use_vision else "not available",
    #         lidar_token=self.lidar_token if use_lidar else "not available",
    #         positions=positions_str,
    #     )
    #     text_inputs = self.tokenizer(prompt, return_tensors="pt")
    #     prepared['input_ids'] = text_inputs['input_ids'].to(self.device)

    #     prepared['use_vision'] = use_vision
    #     prepared['use_lidar'] = use_lidar
    #     return prepared

    # =========================================================================

    def get_trainable_parameters(self):
        """Return list of parameters to optimize."""
        params = []
        params.extend(self.vision_projector.parameters())

        if self.use_lidar:
            params.extend(self.lidar_projector.parameters())

        if not self.freeze_encoders:
            params.extend(self.vision_tower.parameters())
            if self.use_lidar:
                params.extend(self.lidar_encoder.parameters())
        if not self.freeze_llm:
            params.extend(self.language_model.parameters())
        
        return list(params)

    def save_projectors(self, save_path):
        """Save projection layer weights."""
        if self.use_lidar:
            torch.save(
                {
                    'vision_projector': self.vision_projector.state_dict(),
                    'lidar_projector': self.lidar_projector.state_dict(),
                },
                save_path,
            )
        else:
            torch.save(
                {
                    'vision_projector': self.vision_projector.state_dict(),
                },
                save_path,
            )
        print(f"Saved projectors to {save_path}")

    def load_projectors(self, load_path):
        """Load projection layer weights."""
        ckpt = torch.load(load_path, map_location=self.device)
        self.vision_projector.load_state_dict(ckpt['vision_projector'])
        if self.use_lidar:
            self.lidar_projector.load_state_dict(ckpt['lidar_projector'])
        print(f"Loaded projectors from {load_path}")