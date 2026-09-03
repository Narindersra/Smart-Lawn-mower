class SpeedController:
    """
    Calculates linear velocity based on distance to the target.
    """

    def __init__(
        self,
        maximum_speed: float = 0.5,
        minimum_speed: float = 0.1,
        slowdown_distance: float = 1.0,
    ):
        self.maximum_speed = maximum_speed
        self.minimum_speed = minimum_speed
        self.slowdown_distance = slowdown_distance

    def calculate(
        self,
        distance_to_goal: float,
    ) -> float:
        """
        Calculate linear velocity from distance to the target.

        Args:
            distance_to_goal: Distance to the target in meters.

        Returns:
            Linear velocity in meters per second.
        """

        if distance_to_goal <= 0.0:
            return 0.0

        if distance_to_goal >= self.slowdown_distance:
            return self.maximum_speed

        speed_ratio = (
            distance_to_goal / self.slowdown_distance
        )

        linear_velocity = (
            self.minimum_speed
            + (
                self.maximum_speed
                - self.minimum_speed
            )
            * speed_ratio
        )

        return linear_velocity