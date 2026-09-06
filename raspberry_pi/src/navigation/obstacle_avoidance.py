from .navigation_types import MotionCommand
from .obstacle_types import ObstacleInformation


class ObstacleAvoidance:
    """
    Generates temporary motion commands for local obstacle avoidance.

    This module handles normal obstacle avoidance only. It does not
    perform emergency stopping; emergency stopping remains the
    responsibility of the Safety system.
    """

    def __init__(
        self,
        avoidance_linear_velocity: float = 0.15,
        avoidance_angular_velocity: float = 0.8,
        clearance_distance: float = 0.8,
    ):
        """
        Initialize the obstacle avoidance controller.

        Args:
            avoidance_linear_velocity:
                Linear velocity used during normal obstacle avoidance,
                in meters per second.

            avoidance_angular_velocity:
                Angular velocity used to turn away from an obstacle,
                in radians per second.

            clearance_distance:
                Minimum desired distance from an obstacle in meters.
        """
        self.avoidance_linear_velocity = avoidance_linear_velocity
        self.avoidance_angular_velocity = avoidance_angular_velocity
        self.clearance_distance = clearance_distance

    def calculate(
        self,
        obstacle_information: ObstacleInformation,
    ) -> MotionCommand:
        """
        Calculate a temporary motion command for obstacle avoidance.

        Args:
            obstacle_information:
                Current obstacle detection information.

        Returns:
            MotionCommand containing the linear and angular velocities
            required for normal local obstacle avoidance.
        """

        # No obstacle means there is no avoidance command to execute.
        if not obstacle_information.has_obstacle:
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
            )

        nearest_distance = obstacle_information.nearest_distance

        # Protect against an inconsistent obstacle information object
        # where obstacles are reported but no nearest distance exists.
        if nearest_distance is None:
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
            )

        # When the nearest obstacle is inside the clearance distance,
        # stop forward motion and rotate to create clearance.
        if nearest_distance < self.clearance_distance:
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=self.avoidance_angular_velocity,
            )

        # When there is sufficient distance, move forward slowly while
        # continuing the avoidance turn.
        return MotionCommand(
            linear_velocity=self.avoidance_linear_velocity,
            angular_velocity=self.avoidance_angular_velocity,
        )

    def is_clear(
        self,
        obstacle_information: ObstacleInformation,
    ) -> bool:
        """
        Return whether sufficient clearance exists from all obstacles.

        Args:
            obstacle_information:
                Current obstacle detection information.

        Returns:
            True when no obstacle is detected or every detected obstacle
            is at or beyond the configured clearance distance.
        """

        if not obstacle_information.has_obstacle:
            return True

        return all(
            obstacle.distance >= self.clearance_distance
            for obstacle in obstacle_information.obstacles
        )