import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
import re
from termcolor import colored

import psutil

class QwenCLIPModel(nn.Module):

    def __init__(self, device, qwen_model_name="Qwen/Qwen2.5-3B-Instruct", clip_model_name="openai/clip-vit-large-patch14", checkpoint_path=None):
        super().__init__()

        self.device = device
        print(f"Using device {self.device} for QwenCLIPModel.")

        # Load pre-trained weights if provided
        if checkpoint_path is not None:
            # ram = psutil.virtual_memory()
            # print("Checking system RAM before loading checkpoint...")
            # print(f"Total RAM: {ram.total / 1e9:.2f} GB")
            # print(f"Available RAM: {ram.available / 1e9:.2f} GB")
            # print(f"Used RAM: {ram.used / 1e9:.2f} GB")
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint_data = torch.load(checkpoint_path, weights_only=False, map_location="cpu")  # CPU first to save GPU memory
            print(colored("✓ Checkpoint loaded", "green"))

        print("Loading CLIP vision model...")
        self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name).to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

        print(f"Loading Qwen language model: {qwen_model_name}...")
        self.qwen_model_name = qwen_model_name
        self.language_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

        clip_hidden_size = self.vision_tower.config.hidden_size
        qwen_hidden_size = self.language_model.config.hidden_size
        self.mlp_projector = nn.Sequential(
            nn.Linear(clip_hidden_size, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size  * 4, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size),
        ).to(self.device).to(torch.bfloat16)

        if checkpoint_data is not None:
            print("Loading checkpoint weights into model components...")
            self._load_checkpoint_weights(checkpoint_data)

        self.prompt_part1 = (
            "You are a self-driving car. Your task is to predict the future trajectory based on the camera image and your recent movement. "
            "Your last three recorded positions (x, y) are: "
        )
        self.prompt_part2 = (
            "It is critical that you output exactly 10 waypoints. "
            "The trajectory must be formatted as a sequence of 10 2D coordinates `[x, y]`."
            "For example:\n"
            "Future Trajectory: [[x1, y1], [x2, y2], ..., [x10, y10]]"
        )

    def forward(self, images, input_ids):
        """
        FOR TRAINING:
        This method now takes the final tensors as input and returns logits.
        It does NOT do any data prep like tokenizing or prompt creation.
        That should be done in the training loop.
        """
        # 1. Process the image input
        image_features = self.vision_tower(pixel_values=images).last_hidden_state
        projected_image_features = self.mlp_projector(image_features.to(torch.bfloat16))
        
        # 2. Get the embeddings for the text input
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        # 3. Combine image and text embeddings
        combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)
        
        # 4. Get the raw logits from the language model
        outputs = self.language_model(inputs_embeds=combined_embeddings)

        return outputs.logits

    def generate_trajectory(self, images, ego_positions):
        """
        FOR INFERENCE: The non-differentiable generation method.
        """
        image_features = self.vision_tower(pixel_values=images).last_hidden_state
        projected_image_features = self.mlp_projector(image_features.to(torch.bfloat16))

        # --- Prompt formatting (same as before) ---
        prompts = []
        for pos_tensor in ego_positions:
            pos_list = [f"[{pos[0]:.2f}, {pos[1]:.2f}]" for pos in pos_tensor]
            pos_str = ", ".join(pos_list)
            final_prompt = f"{self.prompt_part1}[{pos_str}]\n{self.prompt_part2}"
            prompts.append(final_prompt)

        if self.qwen_model_name == "Qwen/Qwen3-4B":
            print(colored("Using Qwen3-4B prompt template...", "cyan"))
            full_prompts = [self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            ) for p in prompts]
        else:
            full_prompts = [self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
            ) for p in prompts]
        
        inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True).to(self.device)
        text_embeddings = self.language_model.get_input_embeddings()(inputs.input_ids)
        combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)


        # Create attention mask for image embeddings (all real)
        image_attention = torch.ones(projected_image_features.shape[:2], dtype=torch.long, device=self.device)
        # Combine with text attention mask
        combined_attention_mask = torch.cat([image_attention, inputs.attention_mask], dim=1)

        # --- Generation with strict constraints to prevent thinking mode ---
        outputs = self.language_model.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            max_new_tokens=200,  # Reduced - trajectory should be ~100 tokens max
            min_new_tokens=50,   # Ensure it generates something substantial
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,     # Greedy decoding for consistency
            num_beams=5,         # No beam search
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_ids = outputs.sequences
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs, generated_text

    def _load_checkpoint_weights(self, checkpoint_data):
        """Load weights from checkpoint dict into model components."""

        # Print all keys
        print(colored("Checkpoint keys:", "yellow"))
        for key in checkpoint_data.keys():
            print(f"  - {key}")

        if 'language_model_state_dict' in checkpoint_data:
            # Full checkpoint
            self.language_model.load_state_dict(checkpoint_data['language_model_state_dict'], strict=False)
            self.vision_tower.load_state_dict(checkpoint_data['vision_tower_state_dict'])
            self.mlp_projector.load_state_dict(checkpoint_data['mlp_projector_state_dict'])
            print(colored("✓ Loaded LLM + vision encoder + MLP projector", "green"))
        else:
            # MLP-only checkpoint
            self.mlp_projector.load_state_dict(checkpoint_data['model_state_dict'])
            print(colored("✓ Loaded MLP projector", "green"))