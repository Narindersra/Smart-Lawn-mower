from localization.position_estimator import RobotPose


class Geofence:
    """
    Defines the allowed operating area for the robot.

    The geofence is represented as a rectangular boundary in the
    robot's local X/Y coordinate system.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ):
        """
        Initialize the rectangular geofence.

        Args:
            min_x: Minimum allowed X coordinate.
            max_x: Maximum allowed X coordinate.
            min_y: Minimum allowed Y coordinate.
            max_y: Maximum allowed Y coordinate.

        Raises:
            ValueError: If the minimum boundary is greater than or
                equal to the corresponding maximum boundary.
        """

        # Validate the X-axis boundaries.
        if min_x >= max_x:
            raise ValueError("min_x must be smaller than max_x.")

        # Validate the Y-axis boundaries.
        if min_y >= max_y:
            raise ValueError("min_y must be smaller than max_y.")

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def contains(self, pose: RobotPose) -> bool:
        """
        Check whether the robot pose is inside the allowed area.

        Args:
            pose: Current robot pose.

        Returns:
            True if the robot's X/Y position is inside or exactly on
            the geofence boundary; otherwise False.
        """

        return (
            self.min_x <= pose.x <= self.max_x
            and self.min_y <= pose.y <= self.max_y
        )

    def contains_position(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Check whether a position is inside the allowed area.

        Args:
            x: X coordinate to check.
            y: Y coordinate to check.

        Returns:
            True if the position is inside or exactly on the geofence
            boundary; otherwise False.
        """

        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
        )