import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models. Point is for any new models we have to have the same functions
    """
    def __init__(self):
        super().__init__()

        # Prompts stored here for consistency across train/eval
        # self.prompt_part1 = (
        #     "You are a self-driving car. Your task is to predict the future trajectory based camera and lidar data (if provided), and your recent movement. " # TODO: Don't say camera image since we have lidar too
        #     "Your last recorded positions (x, y) in the global ego-frame are: "
        # )
        # self.prompt_part2 = (
        #     "It is critical that you output exactly 10 waypoints in the EXACT format specified below: "
        #     "Future Trajectory: [[x1, y1], [x2, y2], ..., [x10, y10]]\n\n"
        # )

        self.prompt_part1 = (
            "You are a self-driving car. Your task is to predict the future trajectory."
            " Your last recorded positions (x, y) in the global frame are: "
        )

        self.prompt_part2 = (
            "It is critical that you output 10 waypoints in the EXACT format specified below:\n"
            "Future Trajectory: [[x1, y1], [x2, y2], ..., [x10, y10]]\n"
            # ", where each waypoint is in the format [x, y] representing global coordinates to the nearest two decimal places.\n" # Seems like specifying decimal places is not necessary
        )

    @abstractmethod
    def forward(self, images, lidar, ego_pos_global):
        """Standard PyTorch forward pass."""
        pass
    
    @abstractmethod
    def _load_checkpoint_weights(self, checkpoint=None):
        pass

    @abstractmethod
    def generateMotion(self, images, lidar, ego_pos_global):
        pass

    @abstractmethod
    def get_trainable_parameters(self):
        pass

    @abstractmethod
    def train(self, mode=True):
        pass