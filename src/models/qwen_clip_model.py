import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os

class QwenCLIPModel(nn.Module):

    def __init__(self, qwen_model_name="Qwen/Qwen2.5-3B-Instruct", clip_model_name="openai/clip-vit-large-patch14"):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading CLIP vision model...")
        self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        
        print(f"Loading Qwen language model: {qwen_model_name}...")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

        # Freeze the pre-trained models so only the projector is trained
        self.vision_tower.requires_grad_(False)
        self.language_model.requires_grad_(False)

        # Get the embedding dimensions for the projector
        clip_hidden_size = self.vision_tower.config.hidden_size
        qwen_hidden_size = self.language_model.config.hidden_size

        # "Glue" layer
        self.mlp_projector = nn.Sequential(
            nn.Linear(clip_hidden_size, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size)
        ).to(self.device)



    def forward(self, image_path, text_prompt):

        # Process the image with CLIP's processor and model
        pil_image = Image.open(image_path).convert('RGB') # Might have to change this based on how we're loading data
        image_inputs = self.image_processor(images=pil_image, return_tensors="pt").to(self.device)
        image_features = self.vision_tower(**image_inputs).last_hidden_state
        
        # Project the image features through the trainable MLP
        projected_image_features = self.mlp_projector(image_features)

        # Prepare the text prompt using the chat template for better performance
        messages = [
            {"role": "user", "content": text_prompt}
        ]
        # apply_chat_template handles the special tokens and formatting for you
        full_prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize the formatted text prompt
        text_input_ids = self.tokenizer(full_prompt_text, return_tensors="pt").input_ids.to(self.device)
        
        # Get the corresponding text embeddings from the language model
        embedding_layer = self.language_model.get_input_embeddings()
        text_embeddings = embedding_layer(text_input_ids)
        
        # Combine image and text embeddings by concatenating them
        # The LLM will "see" the image first, then read the text prompt
        combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)
        
        # Generate text using the combined embeddings
        # We pass `inputs_embeds` directly, bypassing the model's normal embedding lookup
        output_ids = self.language_model.generate(
            inputs_embeds=combined_embeddings,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id # Suppress warnings
        )
        
        # Decode the output, slicing off the input tokens to get only the generated part
        generated_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Clean up the output to remove the prompt text
        response = generated_text.replace(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False), "").strip()

        return response
    