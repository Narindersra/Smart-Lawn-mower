class SafetyManager:
    def __init__(
        self,
        ai_stop_classes: set[str],
        obstacle_distance_threshold: float,
    ):
        self.ai_stop_classes = ai_stop_classes
        self.obstacle_distance_threshold = obstacle_distance_threshold

    def should_stop_for_ai(self, detections) -> bool:
        return any(
            detection.class_name in self.ai_stop_classes
            for detection in detections
        )

    def should_stop_for_obstacle(self, distance: float) -> bool:
        return distance < self.obstacle_distance_threshold

    def should_emergency_stop(self, detections) -> bool:
        return self.should_stop_for_ai(detections)