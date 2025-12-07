import cv2
import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from qwen_vl_utils import process_vision_info
from src.utils.utils import query_gpt4
from src.openemma.YOLO3D.inference import yolo3d_nuScenes

class BaseOpenEMMA:
    def __init__(self, args):
        """
        Initializes the OpenEMMA model wrapper using Hugging Face Transformers.
        """
        if args.model_id == "liuhaotian/llava-v1.6-mistral-7b":
            self.model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        else:
            self.model_path = args.model_id
            
        self.init_model(args=args)

    def init_model(self, args=None):
        if "llava" in self.model_path.lower():
            print(f"Loading LLaVA-Next model: {self.model_path}...")
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        
        elif "gpt" in self.model_path:
            self.api_key = args.api_key
            
        elif "Llama-3.2" in self.model_path:
             from transformers import MllamaForConditionalGeneration, AutoProcessor
             self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
             self.processor = AutoProcessor.from_pretrained(self.model_path)

        elif "Qwen" in self.model_path:
            print(f"Loading Qwen Instruct model: {self.model_path}...")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)

    def vlm_inference(self, text=None, image_path=None, sys_message=None, args=None):
        if "Qwen" in self.model_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text
        
        elif "llava" in self.model_path.lower():
            raw_image = Image.open(image_path).convert('RGB')
            prompt_text = f"[INST] <image>\n{text} [/INST]"
            inputs = self.processor(text=prompt_text, images=raw_image, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0)
            output_text = self.processor.decode(output[0], skip_special_tokens=True)
            if "[/INST]" in output_text:
                output_text = output_text.split("[/INST]")[-1].strip()
            return output_text

        elif "gpt" in self.model_path:
            return query_gpt4(text, api_key=self.api_key, image_path=image_path, sys_message=sys_message)
            
        elif "Llama-3.2" in self.model_path:
            pass
        return ""

    # --- Prompts ---

    def Scenedescription(self, image_path, backbone):
        prompt = "Provide a short description of the driving scenario."
        return self.vlm_inference(text=prompt, image_path=image_path)
    
    def description_criticalobjects(self, r2, image_path, backbone):
        prompt = f"Identified critical object: {r2}\nProvide a short description of the current status and intended actions to the above identified critical object for the ego car."
        return self.vlm_inference(text=prompt, image_path=image_path)

    def get_objects(self, image_path):
        prompt = "Please list 2-3 key objects the ego car should focus on, specifying only the object's name and its related location in the image of the driving scene."
        return self.vlm_inference(text=prompt, image_path=image_path)

    def getCoT(self, image_path, ego_fut_diff, ego_fut_trajs, ego_his_diff, backbone):
        r1 = self.Scenedescription(image_path, backbone)
        r2 = self.get_objects(image_path)
        r3 = self.description_criticalobjects(r2, image_path, backbone)
        r4 = self.compute_meta_action(ego_fut_diff, ego_fut_trajs, ego_his_diff)
        return f"""Scene description:\n{r1}\nCritical objects:\n{r2}\nBehavior description:\n{r3}\nMeta driving decision:\n{r4}"""

    def compute_meta_action(self, ego_fut_diff, ego_fut_trajs, ego_his_diff):
        constant_eps = 0.5
        his_velos = np.linalg.norm(ego_his_diff, axis=1)
        fut_velos = np.linalg.norm(ego_fut_diff, axis=1)
        cur_velo = his_velos[-1]
        end_velo = fut_velos[-1]

        if cur_velo < constant_eps and end_velo < constant_eps:
            speed_meta = "STOPPED"
        elif end_velo < constant_eps:
            speed_meta = "DECELERATING TO STOP"
        elif np.abs(end_velo - cur_velo) < constant_eps:
            speed_meta = "CONSTANT SPEED FORWARD"
        else:
            speed_meta = "DECELERATING" if cur_velo > end_velo else "ACCELERATING"
        
        lane_changing_th = 4.0
        forward_th = 2.0
        lateral_positions = np.array(ego_fut_trajs)[:, 1] 
        final_lateral = lateral_positions[-1]

        if (np.abs(lateral_positions) < forward_th).all():
             behavior_meta = "MOVING FORWARD"
        else:
             if final_lateral > 0:
                 if np.abs(final_lateral) > lane_changing_th:
                     behavior_meta = "TURNING LEFT"
                 else:
                     behavior_meta = "LANE CHANGING LEFT"
             elif final_lateral < 0:
                 if np.abs(final_lateral) > lane_changing_th:
                     behavior_meta = "TURNING RIGHT"
                 else:
                     behavior_meta = "LANE CHANGING RIGHT"
             else:
                 behavior_meta = "MOVING FORWARD"

        return speed_meta + " AND " + behavior_meta

    def compute_command(self, ego_fut_trajs):
        lane_changing_th = 4.0
        traj_arr = np.array(ego_fut_trajs)
        if (np.abs(traj_arr[:, 1]) < lane_changing_th).all():
            return "MOVE FORWARD"  
        elif traj_arr[-1, 1] > 0:
            return "TURN LEFT"
        elif traj_arr[-1, 1] < 0:
            return "TURN RIGHT"
        else:
            return "MOVE FORWARD"

    def estimate_kinematics(self, history, dt=0.5):
        """
        Calculates FULL history of Speed (m/s) and Curvature (1/m).
        """
        N = len(history)
        if N < 2:
            return [], []

        diffs = history[1:] - history[:-1] 
        dists = np.linalg.norm(diffs, axis=1)
        speeds = dists / dt
        
        curvatures = []
        if N >= 3:
            headings = np.arctan2(diffs[:, 1], diffs[:, 0])
            delta_thetas = headings[1:] - headings[:-1]
            delta_thetas = (delta_thetas + np.pi) % (2 * np.pi) - np.pi
            dists_avg = (dists[1:] + dists[:-1]) / 2.0
            dists_avg[dists_avg < 0.01] = 0.01
            curvatures = delta_thetas / dists_avg

        return speeds.tolist(), curvatures.tolist() if isinstance(curvatures, list) else curvatures.tolist()

    def generate_waypoints(self, command, image_path, data=None, backbone=None, args=None):
        # 1. Normalize History
        raw_his = np.array(data["gt_ego_his_trajs"])
        if len(raw_his) > 0:
            current_pos = raw_his[-1]
            normalized_history = raw_his - current_pos
        else:
            normalized_history = np.zeros((1, 2))

        rationale = self.getCoT(image_path, data["gt_ego_fut_diff"], data["gt_ego_fut_trajs"], data["gt_ego_his_diff"], backbone)
        ego_his_trajs_str = str(normalized_history.tolist()).replace("\n", '')
        sys_message = "You are an expert autonomous driving agent. Your task is to predict the future trajectory of the ego vehicle."

        ################################################################################################
        ##################################### Extra Enhancements #######################################
        ################################################################################################
        # # 2. Estimate Kinematics
        # hist_speeds, hist_curvs = self.estimate_kinematics(normalized_history, dt=0.5)
        
        # # Robustly get current speed
        # current_speed = hist_speeds[-1] if len(hist_speeds) > 0 else 0.0
        # avg_speed = np.mean(hist_speeds[-3:]) if len(hist_speeds) >= 3 else current_speed
        
        # # Determine Trend
        # if len(hist_speeds) >= 3 and hist_speeds[-1] > hist_speeds[0] + 0.5:
        #     trend = "ACCELERATING"
        # elif len(hist_speeds) >= 3 and hist_speeds[-1] < hist_speeds[0] - 0.5:
        #     trend = "DECELERATING"
        # else:
        #     trend = "MAINTAINING SPEED"

        # speed_hist_str = str([round(s, 2) for s in hist_speeds]).replace("'", "")
        # curv_hist_str = str([round(k, 4) for k in hist_curvs]).replace("'", "")
        
        # --- ENHANCED PROMPT ---
        # 1. Explicitly states the Current Speed Goal.
        # 2. Adds a "Trend" indicator to help the model decide to accelerate/brake.
#         prompt = f"""{sys_message}
# ##Context:
# - Command: {command}
# - Historical Waypoints (Local): {ego_his_trajs_str}

# ##Kinematic State:
# - Speed History (m/s): {speed_hist_str}
# - Current Speed: {current_speed:.2f} m/s (Trend: {trend})
# - Curvature History (1/m): {curv_hist_str}

# ##Driving Logic:
# {rationale}

# ##Instruction:
# Predict the future Speed (S) and Curvature (K) for the next 5 seconds (10 frames).
# 1. **Speed (S):** Initialize prediction at **{current_speed:.2f} m/s**. If the path is clear and command is straight, maintain or slightly increase speed. Do NOT drop speed to near zero unless stopped.
# 2. **Curvature (K):** Positive = Left, Negative = Right. Use history to smooth the turn.

# ##Output:
# Return ONLY the vectors in this exact format:
# S: [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
# K: [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10]
# """

        ################################################################################################
        ################################### End Extra Enhancements #####################################
        ################################################################################################

        prompt = f"""{sys_message}
# ##Context:
# - Command: {command}
# - Historical Waypoints (Local): {ego_his_trajs_str}

# ##Driving Logic:
# {rationale}

# ##Instruction:
# Predict the future Speed (S) and Curvature (K) for the next 5 seconds (10 frames).

# ##Output:
# Return ONLY the vectors in this exact format:
# S: [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
# K: [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10]
# """
        output_text = self.vlm_inference(text=prompt, image_path=image_path)
        speed_vec, curv_vec = self.parse_s_k_vectors(output_text)
        
        trajectory = self.integrate_trajectory(speed_vec, curv_vec)
        return str(trajectory)

    def parse_s_k_vectors(self, text):
        try:
            s_match = re.search(r'S(?:peed)?\s*[:=]\s*\[([\d\.,\s-]+)\]', text, re.IGNORECASE)
            k_match = re.search(r'K(?:urvature)?\s*[:=]\s*\[([\d\.,\s-]+)\]', text, re.IGNORECASE)

            if s_match and k_match:
                s_vec = [float(x) for x in s_match.group(1).split(',') if x.strip()]
                k_vec = [float(x) for x in k_match.group(1).split(',') if x.strip()]
                target_len = 10
                s_vec = (s_vec + [0.0]*target_len)[:target_len]
                k_vec = (k_vec + [0.0]*target_len)[:target_len]
                return s_vec, k_vec
        except Exception as e:
            pass
        return [0.0]*10, [0.0]*10

    def integrate_trajectory(self, speed_vec, curv_vec, dt=0.5):
        traj = []
        x, y = 0.0, 0.0
        theta = 0.0
        
        for s, k in zip(speed_vec, curv_vec):
            theta += (s * k) * dt
            x += s * np.cos(theta) * dt
            y += s * np.sin(theta) * dt
            traj.append([round(x, 4), round(y, 4)])
            
        return traj

    def fix_traj(self, traj):
        if len(traj) > 0 and np.abs(traj[0][0]) > 0.0:
            offset_x = traj[0][0]
            for i in range(len(traj)):
                traj[i][0] -= offset_x
        return traj

    def frames_to_video(self, input_folder, output_file, fps):
        images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
        images.sort()
        if not images: return
        frame = cv2.imread(os.path.join(input_folder, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for image in images:
            out.write(cv2.imread(os.path.join(input_folder, image))) 
        out.release()

    def plot_bbx_yolo(self, input_folder, output_file, raw_file):
        images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
        images.sort()
        for i in range(len(images)):
            yolo3d_nuScenes(os.path.join(input_folder, images[i]), output_file, roi_r=-1, roi_w=-1, roi_d=-1)
    