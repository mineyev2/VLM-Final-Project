import re
import os
import cv2
import sys
import json
import torch
import numpy as np
from PIL import Image
from termcolor import colored
from qwen_vl_utils import process_vision_info
from scipy.spatial.transform import Rotation as R
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Project imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src")) # add /src so we can access files in there
from utils.utils import query_gpt4
from openemma.visualize.visualize import CamParams
from openemma.YOLO3D.inference import yolo3d_nuScenes
from models.base_model import BaseModel

import torch
from termcolor import colored
from models.qwen_clip_model import QwenCLIPModel
from models.base_model import BaseModel


class QwenCLIPDemo(BaseModel):
    """
    Load a locally fine-tuned QwenCLIP model for use in Openemma pipeline.
    """
    def __init__(self, checkpoint_path="./outputs/latest/10-13-unfreeze_clip-250epochs/final_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(colored(f"Initializing QwenCLIPLocalModel on device: {self.device}", "cyan"))
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.load(checkpoint_path)

    def load(self, checkpoint_path):
        print(colored(f"Loading fine-tuned model from {checkpoint_path}", "cyan"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # initialize model structure
        base_model = QwenCLIPModel(self.device)
        base_model.mlp_projector.load_state_dict(checkpoint['model_state_dict'])
        base_model.vision_tower.load_state_dict(checkpoint['vision_tower_state_dict'])
        base_model.language_model.load_state_dict(checkpoint['language_model_state_dict'])
        base_model.eval()

        self.model = base_model
        self.tokenizer = base_model.tokenizer
        self.image_processor = base_model.image_processor
        self.processor = self.image_processor  # for compatibility

        print(colored("✅ Custom Qwen-CLIP model successfully loaded!", "green"))

    # === 以下方法用于与 OpenMR 接口 ===
    def generateMessage(self, prompt, image=None, args=None):
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}]

    def prompt(self, text=None, images=None, args=None):
        """
        For compatibility with OpenMR demo.
        Generates and returns decoded text output from the model.
        """
        # The 'images' argument is a filepath string from the calling script
        if isinstance(images, str):
            pil_image = Image.open(images).convert('RGB')
        else:
            # If it's not a path, assume it's an already-loaded PIL image
            pil_image = images
            
        # 1. Process image input
        image_inputs = self.image_processor(images=[pil_image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            # 2. Get image embeddings from the vision tower and projector
            image_features = self.model.vision_tower(pixel_values=image_inputs.pixel_values.to(torch.bfloat16)).last_hidden_state
            projected_image_features = self.model.mlp_projector(image_features.to(torch.bfloat16))

            # 3. Format and tokenize the text prompt
            full_prompt = self.model.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            inputs = self.model.tokenizer([full_prompt], return_tensors="pt").to(self.device)
            text_embeddings = self.model.language_model.get_input_embeddings()(inputs.input_ids)

            # 4. Combine image and text embeddings to create the multimodal input
            combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)
            
            # The length of the combined input embeddings is needed to slice the output
            # and isolate the newly generated tokens.
            input_len = combined_embeddings.shape[1]

            # 5. Generate token IDs using the language model
            outputs = self.model.language_model.generate(
                inputs_embeds=combined_embeddings,
                max_new_tokens=512,  # Set a reasonable limit for new tokens
                pad_token_id=self.model.tokenizer.eos_token_id
            )

            # 6. Decode only the newly generated tokens into a string
            generated_ids = outputs[0, input_len:]
            response = self.model.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response

    def describeScene(self, images, args=None):
        return self.prompt(text="Describe the driving scene from the images.", images=images)

    def describeObjects(self, images, args=None):
        return self.prompt(text="Identify and describe key objects in the scene.", images=images)

    def generateIntent(self, images, prev_intent=None):
        return self.prompt(text="Describe the ego car's driving intent.", images=images)


    def generateMotion(self, images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, args=None):
        """
        Generate motion prediction using QwenCLIP model and multimodal reasoning.
        """
        # === Step 1. Scene/Object/Intent Reasoning ===
        # These calls will now correctly return strings
        scene_description = self.describeScene(images, args=args)
        object_description = self.describeObjects(images, args=args)
        intent_description = self.generateIntent(images, prev_intent=given_intent)

        print(f"\nScene Description: {scene_description}")
        print(f"Object Description: {object_description}")
        print(f"Intent Description: {intent_description}")

        # === Step 2. Prepare observation info ===
        obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
        obs_waypoints_str = ", ".join(obs_waypoints_str)
        obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
        obs_curvatures = obs_curvatures * 100
        obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
        obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

        print(f"Observed Speed and Curvature: {obs_speed_curvature_str}")

        # === Step 3. Construct reasoning prompt ===
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. 
        The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene_description}. 
        The identified critical objects are {object_description}. 
        The car's intent is {intent_description}. 
        The 5-second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. 
        Generate the predicted future speeds and curvatures in the format 
        [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. 
        Write the raw text, not markdown or latex. 
        Future speeds and curvatures:"""

        # === Step 4. Query Qwen model ===
        result = ""
        for attempt in range(3):
            # This call will now return a string, resolving the error
            result = self.prompt(text=prompt, images=images, args=args)
            print(result)
            if not any(bad_word in result.lower() for bad_word in ["unable", "sorry"]) and "[" in result:
                break
            print(f"[Retry {attempt+1}/3] Model response invalid, retrying...")

        # === Step 5. Return structured reasoning results ===
        return result, scene_description, object_description, intent_description
