from .obstacle_types import ObstacleInformation
from .navigation_types import MotionCommand


class ObstacleAvoidance:
    """
    Generates a temporary motion command for local obstacle avoidance.

    This module does not perform emergency stopping.
    Emergency stopping remains the responsibility of the Safety system.
    """

    def __init__(
        self,
        avoidance_linear_velocity: float = 0.15,
        avoidance_angular_velocity: float = 0.8,
        clearance_distance: float = 0.8,
    ):
        self.avoidance_linear_velocity = avoidance_linear_velocity
        self.avoidance_angular_velocity = avoidance_angular_velocity
        self.clearance_distance = clearance_distance

    def calculate(self, obstacle_information: ObstacleInformation) -> MotionCommand:
        if not obstacle_information.has_obstacle:
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
            )
    
        nearest_distance = obstacle_information.nearest_distance
    
        if nearest_distance is None:
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
            )
    
        if nearest_distance < self.clearance_distance:
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=self.avoidance_angular_velocity,
            )
    
        return MotionCommand(
            linear_velocity=self.avoidance_linear_velocity,
            angular_velocity=self.avoidance_angular_velocity,
        )

    def is_clear(
        self,
        obstacle_information: ObstacleInformation,
    ) -> bool:
        """
        Return True when the robot has sufficient clearance
        from the currently detected obstacle.
        """
    
        if not obstacle_information.has_obstacle:
            return True
    
        return all(
            obstacle.distance >= self.clearance_distance
            for obstacle in obstacle_information.obstacles
        )