from dataclasses import dataclass
from email.mime import image
from pathlib import Path
from unittest import result

import numpy as np
from streamlit import image
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

        
        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except (RuntimeError, ValueError, TypeError):
            return []

        detections: list[Detection] = []

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

                    if not np.isfinite(confidence):
                        continue
                    
                    if confidence < 0.0 or confidence > 1.0:
                        continue
                    
                    if class_id not in self.model.names:
                        continue
                    
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

                except (AttributeError, IndexError, TypeError, ValueError):
                    continue

        return detections