import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeYOLO:
    """Minimal YOLO replacement for unit testing."""

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, *args, **kwargs):
        return []


def install_fake_ultralytics():
    """
    Install a fake ultralytics module so AI unit tests
    do not require the real ML dependency.
    """
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = FakeYOLO
    sys.modules["ultralytics"] = fake_module


install_fake_ultralytics()


from ai.inference.detecter import (
    Detection,
    ObjectDetector,
)

from ai.inference.inference import (
    InferenceEngine,
)


class TestDetection(unittest.TestCase):

    def test_detection_creation(self):
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            x1=10.0,
            y1=20.0,
            x2=100.0,
            y2=200.0,
        )

        self.assertEqual(
            detection.class_id,
            0,
        )

        self.assertEqual(
            detection.class_name,
            "person",
        )

        self.assertAlmostEqual(
            detection.confidence,
            0.95,
        )

        self.assertEqual(
            detection.x1,
            10.0,
        )

        self.assertEqual(
            detection.y1,
            20.0,
        )

        self.assertEqual(
            detection.x2,
            100.0,
        )

        self.assertEqual(
            detection.y2,
            200.0,
        )


class TestAIModuleFiles(unittest.TestCase):

    def test_detector_module_exists(self):
        detector_file = (
            PROJECT_ROOT
            / "ai"
            / "inference"
            / "detecter.py"
        )

        self.assertTrue(
            detector_file.exists()
        )

    def test_inference_module_exists(self):
        inference_file = (
            PROJECT_ROOT
            / "ai"
            / "inference"
            / "inference.py"
        )

        self.assertTrue(
            inference_file.exists()
        )

    def test_ai_requirements_exists(self):
        requirements_file = (
            PROJECT_ROOT
            / "ai"
            / "requirements.txt"
        )

        self.assertTrue(
            requirements_file.exists()
        )


class TestObjectDetector(unittest.TestCase):

    def test_detector_initialization(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        self.assertEqual(
            detector.confidence_threshold,
            0.5,
        )

    def test_detector_rejects_missing_model(self):
        with patch.object(
            Path,
            "exists",
            return_value=False,
        ):
            with self.assertRaises(FileNotFoundError):
                ObjectDetector(
                    model_path="missing_model.pt",
                    confidence_threshold=0.5,
                )

    def test_detector_handles_none_image(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        result = detector.detect(None)

        self.assertEqual(
            result,
            [],
        )

    def test_detector_handles_empty_image(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )
    
        empty_image = np.array([])
    
        result = detector.detect(empty_image)
    
        self.assertEqual(
            result,
            [],
        )


class TestInferenceEngine(unittest.TestCase):

    def test_engine_initialization(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            engine = InferenceEngine(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        self.assertIsNotNone(
            engine.detector
        )

    def test_engine_runs_with_none_image(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            engine = InferenceEngine(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        result = engine.run(None)

        self.assertEqual(
            result,
            [],
        )


if __name__ == "__main__":
    unittest.main()