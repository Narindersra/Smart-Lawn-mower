class HeadingController:
    """
    Converts heading error into an angular velocity command.
    """

    def __init__(
        self,
        proportional_gain: float = 2.0,
        max_angular_velocity: float = 1.5,
    ):
        self.proportional_gain = proportional_gain
        self.max_angular_velocity = max_angular_velocity

    def calculate(
        self,
        heading_error: float,
    ) -> float:
        """
        Calculate angular velocity from heading error.

        Args:
            heading_error: Angular error in radians.

        Returns:
            Angular velocity in radians per second.
        """

        angular_velocity = (
            self.proportional_gain * heading_error
        )

        return max(
            -self.max_angular_velocity,
            min(
                angular_velocity,
                self.max_angular_velocity,
            ),
        )