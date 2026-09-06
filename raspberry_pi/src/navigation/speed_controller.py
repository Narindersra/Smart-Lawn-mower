class SpeedController:
    """
    Calculates linear velocity based on the distance to the target.

    The controller maintains maximum speed when the target is far away
    and gradually reduces the speed as the robot approaches the target.
    """

    def __init__(
        self,
        maximum_speed: float = 0.5,
        minimum_speed: float = 0.1,
        slowdown_distance: float = 1.0,
    ):
        """
        Initialize the speed controller.

        Args:
            maximum_speed: Maximum linear velocity in meters per second.
            minimum_speed: Minimum linear velocity in meters per second
                while the robot is still approaching the target.
            slowdown_distance: Distance from the target at which the
                speed begins to reduce.
        """
        self.maximum_speed = maximum_speed
        self.minimum_speed = minimum_speed
        self.slowdown_distance = slowdown_distance

    def calculate(
        self,
        distance_to_goal: float,
    ) -> float:
        """
        Calculate linear velocity from the distance to the target.

        Args:
            distance_to_goal: Distance to the target in meters.

        Returns:
            Linear velocity in meters per second.
        """

        # Stop when the target has been reached or passed.
        if distance_to_goal <= 0.0:
            return 0.0

        # Maintain maximum speed outside the slowdown zone.
        if distance_to_goal >= self.slowdown_distance:
            return self.maximum_speed

        # Calculate the robot's position within the slowdown zone.
        speed_ratio = (
            distance_to_goal / self.slowdown_distance
        )

        # Interpolate between minimum and maximum speed.
        linear_velocity = (
            self.minimum_speed
            + (
                self.maximum_speed
                - self.minimum_speed
            )
            * speed_ratio
        )

        return linear_velocity