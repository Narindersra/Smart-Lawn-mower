from localization.position_estimator import RobotPose


class Geofence:
    """
    Defines the allowed operating area for the robot.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ):
        if min_x >= max_x:
            raise ValueError("min_x must be smaller than max_x.")

        if min_y >= max_y:
            raise ValueError("min_y must be smaller than max_y.")

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def contains(self, pose: RobotPose) -> bool:
        """
        Return True if the robot is inside the allowed area.
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
        Return True if a position is inside the allowed area.
        """

        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
        )