"""
Safety management for the lawn mower control system.

This module provides safety checks for:
    - AI-based object detections
    - Critical obstacle distance
    - Geofence boundaries
"""

class SafetyManager:
    """
    Central manager for robot safety-stop conditions.

    The SafetyManager evaluates safety conditions and reports
    whether the robot should stop.
    """

    def __init__(
        self,
        ai_stop_classes: set[str],
        obstacle_distance_threshold: float,
        geofence=None,
    ):
        """
        Initialize the safety manager.

        Args:
            ai_stop_classes:
                Object classes that should trigger an AI safety stop.

            obstacle_distance_threshold:
                Distance threshold below which an obstacle should
                trigger a safety stop.

            geofence:
                Optional geofence object used to detect when the
                robot leaves the permitted operating area.
        """
        self.ai_stop_classes = ai_stop_classes
        self.obstacle_distance_threshold = obstacle_distance_threshold
        self.geofence = geofence

    def should_stop_for_ai(self, detections) -> bool:
        """
        Check whether any detected object requires a safety stop.

        Args:
            detections:
                Iterable of object detections containing a
                ``class_name`` attribute.

        Returns:
            True if a detected object belongs to the configured
            AI stop classes, otherwise False.
        """
        return any(
            detection.class_name in self.ai_stop_classes
            for detection in detections
        )

    def should_stop_for_obstacle(self, distance: float) -> bool:
        """
        Check whether an obstacle is within the critical distance.

        Args:
            distance:
                Distance to the detected obstacle.

        Returns:
            True when the distance is below the configured
            obstacle threshold, otherwise False.
        """
        return distance < self.obstacle_distance_threshold

    def should_stop_for_geofence(self, pose) -> bool:
        """
        Check whether the robot is outside the configured geofence.

        Args:
            pose:
                Current robot pose used by the geofence to determine
                whether the robot is inside the permitted area.

        Returns:
            True when the robot is outside the geofence.
            False when no geofence is configured or the robot is
            inside the permitted area.
        """
        if self.geofence is None:
            return False

        return not self.geofence.contains(pose)

    def should_emergency_stop(self, detections) -> bool:
        """
        Check whether AI detections require an emergency stop.

        Args:
            detections:
                Iterable of object detections.

        Returns:
            True when an AI safety-stop condition is detected,
            otherwise False.
        """
        return self.should_stop_for_ai(detections)