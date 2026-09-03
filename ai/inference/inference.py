from pathlib import Path

import numpy as np

from .detector import Detection, ObjectDetector


class InferenceEngine:
    def __init__(
        self,
        model_path: str = "ai/models/pretrained/yolo11n.pt",
        confidence_threshold: float = 0.5,
    ):
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

    def run(self, image: np.ndarray) -> list[Detection]:
        return self.detector.detect(image)
    