"""
Inference engine for the lawn mower AI system.

This module provides a high-level interface for running object
detection on camera frames.
"""

from pathlib import Path

import numpy as np

from .detecter import Detection, MODEL_PATH, ObjectDetector


class InferenceEngine:
    """
    High-level interface for AI inference.

    The engine delegates object detection to ObjectDetector while
    providing a simple interface for the rest of the system.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the YOLO model file.
            confidence_threshold: Minimum confidence required for
                object detections.
        """
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

    def run(self, image: np.ndarray) -> list[Detection]:
        """
        Run object detection on a camera frame.

        Args:
            image: Three-channel NumPy image array.

        Returns:
            A list of valid object detections.
        """
        return self.detector.detect(image)