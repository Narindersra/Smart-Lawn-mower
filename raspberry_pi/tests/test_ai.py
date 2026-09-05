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

    def test_detector_rejects_non_numpy_input(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        result = detector.detect("invalid image")

        self.assertEqual(
            result,
            [],
        )

    def test_detector_rejects_two_dimensional_image(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        image = np.zeros(
            (480, 640),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            result,
            [],
        )

    def test_detector_rejects_invalid_channel_count(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        image = np.zeros(
            (480, 640, 4),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            result,
            [],
        )

    def test_detector_parses_yolo_result_into_detection(self):
        class FakeBox:
            cls = np.array([0])
            conf = np.array([0.95])
            xyxy = np.array(
                [[10.0, 20.0, 100.0, 200.0]]
            )

        class FakeResult:
            boxes = [FakeBox()]

        class FakeDetectionModel:
            names = {0: "person"}

            def predict(self, *args, **kwargs):
                return [FakeResult()]

        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        detector.model = FakeDetectionModel()

        image = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            len(result),
            1,
        )

        detection = result[0]

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

        self.assertAlmostEqual(
            detection.x1,
            10.0,
        )

        self.assertAlmostEqual(
            detection.y1,
            20.0,
        )

        self.assertAlmostEqual(
            detection.x2,
            100.0,
        )

        self.assertAlmostEqual(
            detection.y2,
            200.0,
        )

    def test_detector_skips_result_without_boxes(self):
        class FakeResult:
            boxes = None

        class FakeDetectionModel:
            names = {0: "person"}

            def predict(self, *args, **kwargs):
                return [FakeResult()]

        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        detector.model = FakeDetectionModel()

        image = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            result,
            [],
        )

    def test_detector_skips_invalid_bbox(self):
        class FakeBox:
            cls = np.array([0])
            conf = np.array([0.95])
            xyxy = np.array(
                [[10.0, 20.0, 100.0]]
            )

        class FakeResult:
            boxes = [FakeBox()]

        class FakeDetectionModel:
            names = {0: "person"}

            def predict(self, *args, **kwargs):
                return [FakeResult()]

        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        detector.model = FakeDetectionModel()

        image = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            result,
            [],
        )

    def test_detector_skips_invalid_class_id(self):
        class FakeBox:
            cls = np.array([99])
            conf = np.array([0.95])
            xyxy = np.array(
                [[10.0, 20.0, 100.0, 200.0]]
            )

        class FakeResult:
            boxes = [FakeBox()]

        class FakeDetectionModel:
            names = {0: "person"}

            def predict(self, *args, **kwargs):
                return [FakeResult()]

        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        detector.model = FakeDetectionModel()

        image = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            result,
            [],
        )

    def test_detector_skips_non_finite_confidence(self):
        class FakeBox:
            cls = np.array([0])
            conf = np.array([np.nan])
            xyxy = np.array(
                [[10.0, 20.0, 100.0, 200.0]]
            )

        class FakeResult:
            boxes = [FakeBox()]

        class FakeDetectionModel:
            names = {0: "person"}

            def predict(self, *args, **kwargs):
                return [FakeResult()]

        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        detector.model = FakeDetectionModel()

        image = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            result,
            [],
        )

    def test_detector_handles_inference_failure(self):
        class FakeDetectionModel:
            def predict(self, *args, **kwargs):
                raise RuntimeError(
                    "inference failed"
                )

        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            detector = ObjectDetector(
                model_path="fake_model.pt",
                confidence_threshold=0.5,
            )

        detector.model = FakeDetectionModel()

        image = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        result = detector.detect(image)

        self.assertEqual(
            result,
            [])


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

    def test_engine_uses_configured_confidence_threshold(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            engine = InferenceEngine(
                model_path="fake_model.pt",
                confidence_threshold=0.6,
            )

        self.assertEqual(
            engine.detector.confidence_threshold,
            0.6,
        )

    def test_engine_uses_default_model_path(self):
        with patch.object(
            Path,
            "exists",
            return_value=True,
        ):
            engine = InferenceEngine()

        self.assertEqual(
            Path(engine.detector.model_path),
            Path(
                PROJECT_ROOT
                / "ai"
                / "models"
                / "pretrained"
                / "yolo11n.pt"
            ),
        )


if __name__ == "__main__":
    unittest.main()