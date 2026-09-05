from pathlib import Path

import numpy as np

from .detecter import Detection, MODEL_PATH, ObjectDetector


class InferenceEngine:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        confidence_threshold: float = 0.5,
    ):
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

    def run(self, image: np.ndarray) -> list[Detection]:
        return self.detector.detect(image)
    