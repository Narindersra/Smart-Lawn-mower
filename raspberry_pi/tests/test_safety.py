import unittest

from safety.safety_manager import SafetyManager
from localization.position_estimator import RobotPose
from navigation.geofence import Geofence
from navigation.navigation_state_machine import NavigationState
from navigation.navigation_state_machine import NavigationStateMachine


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

    # ---------------------------------------------------------
    # COMBINED AI SAFETY
    # ---------------------------------------------------------

    def test_ai_safety_stop_for_person(self):
        safety_manager = SafetyManager(
            {"person", "dog", "cat"},
            0.5,
        )

        class Detection:
            class_name = "person"

        assert safety_manager.should_stop_for_ai(
            [Detection()]
        ) is True

    def test_ai_safety_does_not_stop_for_safe_class(self):
        safety_manager = SafetyManager(
            {"person", "dog", "cat"},
            0.5,
        )

        class Detection:
            class_name = "tree"

        assert safety_manager.should_stop_for_ai(
            [Detection()]
        ) is False

    # ---------------------------------------------------------
    # COMBINED OBSTACLE SAFETY
    # ---------------------------------------------------------

    def test_critical_obstacle_stops(self):
        safety_manager = SafetyManager(
            {"person", "dog", "cat"},
            0.5,
        )

        assert safety_manager.should_stop_for_obstacle(
            0.4
        ) is True

    def test_non_critical_obstacle_does_not_emergency_stop(self):
        safety_manager = SafetyManager(
            {"person", "dog", "cat"},
            0.5,
        )

        assert safety_manager.should_stop_for_obstacle(
            0.8
        ) is False

    # ---------------------------------------------------------
    # GEOFENCE SAFETY
    # ---------------------------------------------------------

    def test_geofence_violation_stops(self):
        geofence = Geofence(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
        )

        safety_manager = SafetyManager(
            {"person", "dog", "cat"},
            0.5,
            geofence=geofence,
        )

        pose = RobotPose(
            x=11.0,
            y=5.0,
            heading=0.0,
        )

        assert safety_manager.should_stop_for_geofence(
            pose
        ) is True

    def test_geofence_safe_position_does_not_stop(self):
        geofence = Geofence(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
        )

        safety_manager = SafetyManager(
            {"person", "dog", "cat"},
            0.5,
            geofence=geofence,
        )

        pose = RobotPose(
            x=5.0,
            y=5.0,
            heading=0.0,
        )

        assert safety_manager.should_stop_for_geofence(
            pose
        ) is False

    # ---------------------------------------------------------
    # RECOVERY
    # ---------------------------------------------------------

    def test_navigation_recovers_from_emergency_stop(self):
        state_machine = NavigationStateMachine()

        state_machine.transition_to(
            NavigationState.EMERGENCY_STOP
        )

        state_machine.resume_from_emergency_stop()

        assert (
            state_machine.get_state()
            == NavigationState.NAVIGATING
        )


if __name__ == "__main__":
    unittest.main()