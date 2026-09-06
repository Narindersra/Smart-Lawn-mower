"""
YOLO-based object detection for the lawn mower AI system.

This module provides:
    - Detection: structured representation of a detected object.
    - ObjectDetector: YOLO inference and detection parsing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ai" / "models" / "pretrained" / "yolo11n.pt"


# ---------------------------------------------------------------------------
# Detection data model
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Represents a single detected object.

    Attributes:
        class_id: Numeric class identifier returned by the YOLO model.
        class_name: Human-readable class name.
        confidence: Detection confidence score in the range [0.0, 1.0].
        x1: Left coordinate of the bounding box.
        y1: Top coordinate of the bounding box.
        x2: Right coordinate of the bounding box.
        y2: Bottom coordinate of the bounding box.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


# ---------------------------------------------------------------------------
# Object detector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Runs YOLO object detection on camera images.

    The detector validates model availability, validates input frames,
    performs YOLO inference, and converts valid model results into
    Detection objects.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the object detector.

        Args:
            model_path: Path to the YOLO model file.
            confidence_threshold: Minimum confidence required by YOLO
                during inference.

        Raises:
            FileNotFoundError: If the YOLO model does not exist.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect objects in a camera image.

        Invalid input frames, inference failures, and malformed
        individual detections are safely ignored.

        Args:
            image: Three-channel NumPy image array.

        Returns:
            A list of valid Detection objects.
        """
        # Validate the input image before running inference.
        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.size == 0
        ):
            return []

        if image.ndim != 3:
            return []

        if image.shape[2] != 3:
            return []

        # Run YOLO inference.
        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except (RuntimeError, ValueError, TypeError):
            return []

        detections: list[Detection] = []

        # Parse every result returned by the model.
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for box in boxes:
                try:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    coordinates = box.xyxy[0].tolist()

                    if len(coordinates) != 4:
                        continue

                    x1, y1, x2, y2 = (
                        float(value)
                        for value in coordinates
                    )

                    # Validate confidence.
                    if not np.isfinite(confidence):
                        continue

                    if confidence < 0.0 or confidence > 1.0:
                        continue

                    # Validate the model class identifier.
                    if class_id not in self.model.names:
                        continue

                    # Validate bounding-box coordinates.
                    if not all(
                        np.isfinite(value)
                        for value in (x1, y1, x2, y2)
                    ):
                        continue

                    if x2 < x1 or y2 < y1:
                        continue

                    class_name = self.model.names[class_id]

                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )

                # Ignore malformed individual detections without
                # affecting the remaining valid detections.
                except (
                    AttributeError,
                    IndexError,
                    TypeError,
                    ValueError,
                ):
                    continue

        return detections