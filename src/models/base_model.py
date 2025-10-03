from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all models. Point is for any new models we have to have the same functions
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generateMessage(self):
        pass

    @abstractmethod
    def prompt(self):
        pass

    @abstractmethod
    def describeScene(self, images):
        pass

    @abstractmethod
    def describeObjects(self, images):
        pass

    @abstractmethod
    def generateIntent(self, images):
        pass

    @abstractmethod
    def generateMotion(self, images):
        pass