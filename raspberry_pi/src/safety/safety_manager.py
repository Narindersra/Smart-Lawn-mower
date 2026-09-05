class SafetyManager:
    def __init__(
        self,
        ai_stop_classes: set[str],
        obstacle_distance_threshold: float,
        geofence=None,
    ):
        self.ai_stop_classes = ai_stop_classes
        self.obstacle_distance_threshold = obstacle_distance_threshold
        self.geofence = geofence

    def should_stop_for_ai(self, detections) -> bool:
        return any(
            detection.class_name in self.ai_stop_classes
            for detection in detections
        )

    def should_stop_for_obstacle(self, distance: float) -> bool:
        return distance < self.obstacle_distance_threshold

    def should_stop_for_geofence(self, pose) -> bool:
        """
        Return True when the robot is outside the configured geofence.
        """
    
        if self.geofence is None:
            return False
    
        return not self.geofence.contains(pose)

    def should_emergency_stop(self, detections) -> bool:
        return self.should_stop_for_ai(detections)

