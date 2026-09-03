from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ai" / "models" / "pretrained" / "yolo11n.pt"


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetector:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        confidence_threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    float(value)
                    for value in box.xyxy[0].tolist()
                )

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

        return detections