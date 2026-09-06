class HeadingController:
    """
    Converts heading error into an angular velocity command.

    The controller uses proportional control and limits the resulting
    angular velocity to the configured maximum value.
    """

    def __init__(
        self,
        proportional_gain: float = 2.0,
        max_angular_velocity: float = 1.5,
    ):
        """
        Initialize the heading controller.

        Args:
            proportional_gain: Proportional gain used to convert
                heading error into angular velocity.
            max_angular_velocity: Maximum allowed angular velocity
                in radians per second.
        """
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
            Angular velocity in radians per second, limited to the
            configured maximum magnitude.
        """

        # Apply proportional control to the heading error.
        angular_velocity = (
            self.proportional_gain * heading_error
        )

        # Limit the angular velocity to the configured range.
        return max(
            -self.max_angular_velocity,
            min(
                angular_velocity,
                self.max_angular_velocity,
            ),
        )