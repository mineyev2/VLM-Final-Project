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

class Qwen25Model(BaseModel):
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None

        self.load()

    def load(self):
        """
        Load the model
        """

        print(colored("Loading Qwen-2.5-VL-3B-Instruct model...", "cyan"))

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.tokenizer = None # TODO: Why is tokenizer none for qwen?

    def generateMessage(self, prompt, image=None, args=None): # TODO: Multiple images?
        """
        Create a message formatted for the specific model type.
        
        Args:
            prompt (str): The text prompt to send to the model
            image: The image data (optional, format depends on model type)
            args: Not using for now
        
        Returns:
            message (list): The formatted message
        """

        message = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]   

        return message

    def prompt(self, text=None, images=None, args=None):
        """
        Send and receive messages from the model.

        Args:
            text (str): The text prompt to send to the model
            images: The image data (format depends on model type)
            args: Not using for now
        """

        message = self.generateMessage(text, image=images)
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def describeScene(self, images, args=None):
        """
        Reasoning Step 1: Scene Description
        Describe driving scene from images
        """

        prompt = f"""You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""
        return self.prompt(text=prompt, images=images)

    def describeObjects(self, images, args=None):
        """
        Reasoning Step 2: Major Objects
        Describe critical objects in the driving scene from images
        """

        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""
        return self.prompt(text=prompt, images=images)

    def generateIntent(self, images, prev_intent=None):
        """
        Reasoning Step 3: Intent Command
        Describe the driving intent of the ego car based on the scene and object descriptions
        """

        if prev_intent is None:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
        else:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

        return self.prompt(text=prompt, images=images)

    def generateMotion(self, images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, args=None):
        scene_description = self.describeScene(images, args=args)
        object_description = self.describeObjects(images, args=args)
        intent_description = self.generateIntent(images, prev_intent=given_intent)
        
        print(f'Scene Description: {scene_description}')
        print(f'Object Description: {object_description}')
        print(f'Intent Description: {intent_description}')

        # Convert array waypoints to string.
        obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
        obs_waypoints_str = ", ".join(obs_waypoints_str)
        obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
        obs_curvatures = obs_curvatures * 100
        obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
        obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

        
        print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
                    The scene is described as follows: {scene_description}. 
                    The identified critical objects are {object_description}. 
                    The car's intent is {intent_description}. 
                    The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
                    Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""

        for rho in range(3):
            result = self.prompt(text=prompt, images=images, args=args)
            if not "unable" in result and not "sorry" in result and "[" in result:
                break

        return result, scene_description, object_description, intent_description