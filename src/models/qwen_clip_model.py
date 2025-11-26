import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from termcolor import colored
from src.models.base_model import BaseModel

class QwenCLIPModel(BaseModel):

    def __init__(self, device, qwen_model_name="Qwen/Qwen2.5-3B-Instruct", clip_model_name="openai/clip-vit-large-patch14", checkpoint_path=None):
        super().__init__()

        if "Qwen2.5" not in qwen_model_name:
            raise Exception("Use Qwen2.5 for QwenCLIP")

        self.device = device
        print(f"Using device {self.device} for QwenCLIPModel.")

        # Load pre-trained weights if provided
        checkpoint_data = None
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint_data = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            print(colored("✓ Checkpoint loaded", "green"))

        print("Loading CLIP vision model...")
        self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name).to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

        print(f"Loading Qwen language model: {qwen_model_name}...")
        self.qwen_model_name = qwen_model_name
        
        # Use the text-only model (AutoModelForCausalLM is correct for Qwen2.5-3B-Instruct)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        clip_hidden_size = self.vision_tower.config.hidden_size
        qwen_hidden_size = self.language_model.config.hidden_size
        
        self.mlp_projector = nn.Sequential(
            nn.Linear(clip_hidden_size, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size),
        ).to(self.device).to(torch.bfloat16)

        if checkpoint_data is not None:
            print("Loading checkpoint weights into model components...")
            self._load_checkpoint_weights(checkpoint_data)

    def forward(self, images, input_ids, labels=None):
        """
        FOR TRAINING:
        Accepts pre-tokenized input_ids and labels from the dataset.
        """
        # 1. Process Images -> Features
        vision_outputs = self.vision_tower(pixel_values=images)
        image_features = vision_outputs.last_hidden_state 
        
        # 2. Project Image Features to LLM Dimension
        projected_image_features = self.mlp_projector(image_features.to(torch.bfloat16)) 

        # 3. Get Text Embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids) 

        # 4. Concatenate: [Image Embeds, Text Embeds]
        combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)
        
        # 5. Adjust Labels for Training
        if labels is not None:
            # Create a filler for image tokens (ignore index -100)
            image_labels_len = projected_image_features.shape[1]
            batch_size = labels.shape[0]
            
            image_labels = torch.full((batch_size, image_labels_len), -100, dtype=labels.dtype, device=self.device)
            combined_labels = torch.cat([image_labels, labels], dim=1)
        else:
            combined_labels = None

        # 6. Forward Pass through LLM
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            labels=combined_labels
        )

        return outputs

    def generate_trajectory(self, images, ego_positions):
        """
        FOR INFERENCE: Generates text from raw inputs.
        """
        # 1. Image Features
        vision_outputs = self.vision_tower(pixel_values=images)
        image_features = vision_outputs.last_hidden_state
        projected_image_features = self.mlp_projector(image_features.to(torch.bfloat16))

        # 2. Format Prompts
        prompts = []
        for pos_tensor in ego_positions:
            pos_list = [f"[{pos[0]:.2f}, {pos[1]:.2f}]" for pos in pos_tensor]
            pos_str = ", ".join(pos_list)
            final_prompt = f"{self.prompt_part1}[{pos_str}]\n{self.prompt_part2}"
            prompts.append(final_prompt)

        # 3. Apply Chat Template
        full_prompts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        ) for p in prompts]
        
        # 4. Tokenize
        inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True).to(self.device)
        text_embeddings = self.language_model.get_input_embeddings()(inputs.input_ids)
        
        # 5. Concatenate [Image, Text]
        combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)

        # 6. Attention Mask
        image_attention = torch.ones(projected_image_features.shape[:2], dtype=torch.long, device=self.device)
        combined_attention_mask = torch.cat([image_attention, inputs.attention_mask], dim=1)

        # 7. Generate
        outputs = self.language_model.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            max_new_tokens=2048,
            min_new_tokens=50,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1, 
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_ids = outputs.sequences
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs, generated_text
    
    def generateMotion(self, images, lidar, ego_pos_global):
        # Prepare Inputs
        pixel_values = self.image_processor(images=images, return_tensors='pt').pixel_values.to(self.device)
        _, gen_text = self.generate_trajectory(pixel_values, ego_pos_global)
        return gen_text


    def _load_checkpoint_weights(self, checkpoint_data):
        """Load weights from checkpoint dict into model components."""
        print(colored("Checkpoint keys:", "yellow"))
        for key in checkpoint_data.keys():
            print(f"  - {key}")

        if 'language_model_state_dict' in checkpoint_data:
            # Full checkpoint
            self.language_model.load_state_dict(checkpoint_data['language_model_state_dict'], strict=False)
            print(colored("✓ Loaded LLM", "green"))

        if 'vision_tower_state_dict' in checkpoint_data:
            self.vision_tower.load_state_dict(checkpoint_data['vision_tower_state_dict'])
            print(colored("✓ Loaded Vision Encoder", "green"))

        self.mlp_projector.load_state_dict(checkpoint_data['model_state_dict'] if 'model_state_dict' in checkpoint_data else checkpoint_data['mlp_projector_state_dict'])
        print(colored("✓ Loaded MLP projector", "green"))
