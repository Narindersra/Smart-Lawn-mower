import unittest

from safety.safety_manager import SafetyManager


class FakeDetection:
    def __init__(self, class_name):
        self.class_name = class_name


class TestSafetyManager(unittest.TestCase):

    def setUp(self):
        self.safety = SafetyManager(
            ai_stop_classes={"person", "dog", "cat"},
            obstacle_distance_threshold=0.5,
        )

    # ---------------------------------------------------------
    # AI SAFETY
    # ---------------------------------------------------------

    def test_ai_stop_for_person(self):
        detections = [
            FakeDetection("person"),
        ]

        self.assertTrue(
            self.safety.should_stop_for_ai(detections)
        )

    def test_ai_stop_for_dog(self):
        detections = [
            FakeDetection("dog"),
        ]

        self.assertTrue(
            self.safety.should_stop_for_ai(detections)
        )

    def test_ai_stop_for_cat(self):
        detections = [
            FakeDetection("cat"),
        ]

        self.assertTrue(
            self.safety.should_stop_for_ai(detections)
        )

    def test_ai_does_not_stop_for_unknown_class(self):
        detections = [
            FakeDetection("car"),
        ]

        self.assertFalse(
            self.safety.should_stop_for_ai(detections)
        )

    def test_ai_does_not_stop_for_empty_detections(self):
        self.assertFalse(
            self.safety.should_stop_for_ai([])
        )

    # ---------------------------------------------------------
    # OBSTACLE SAFETY
    # ---------------------------------------------------------

    def test_obstacle_stop_below_threshold(self):
        self.assertTrue(
            self.safety.should_stop_for_obstacle(0.3)
        )

    def test_obstacle_stop_at_zero(self):
        self.assertTrue(
            self.safety.should_stop_for_obstacle(0.0)
        )

    def test_obstacle_does_not_stop_above_threshold(self):
        self.assertFalse(
            self.safety.should_stop_for_obstacle(1.0)
        )

    def test_obstacle_at_threshold(self):
        # Actual implementation uses:
        # distance < obstacle_distance_threshold
        self.assertFalse(
            self.safety.should_stop_for_obstacle(0.5)
        )

    # ---------------------------------------------------------
    # EMERGENCY STOP
    # ---------------------------------------------------------

    def test_emergency_stop_for_person(self):
        detections = [
            FakeDetection("person"),
        ]

        self.assertTrue(
            self.safety.should_emergency_stop(detections)
        )

    def test_emergency_stop_for_dog(self):
        detections = [
            FakeDetection("dog"),
        ]

        self.assertTrue(
            self.safety.should_emergency_stop(detections)
        )

    def test_emergency_stop_for_cat(self):
        detections = [
            FakeDetection("cat"),
        ]

        self.assertTrue(
            self.safety.should_emergency_stop(detections)
        )

    def test_no_emergency_stop_for_safe_detection(self):
        detections = [
            FakeDetection("car"),
        ]

        self.assertFalse(
            self.safety.should_emergency_stop(detections)
        )

    def test_no_emergency_stop_for_empty_detections(self):
        self.assertFalse(
            self.safety.should_emergency_stop([])
        )


if __name__ == "__main__":
    unittest.main()