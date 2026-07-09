from abc import ABC, abstractmethod


class BaseDetector(ABC):

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def detect(self, image_path: str, classes: list[str]):
        pass