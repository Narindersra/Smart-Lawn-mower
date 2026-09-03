class DifferentialDriveController:
    """
    Converts linear and angular velocity commands into
    left and right wheel velocities for a differential-drive robot.
    """

    def __init__(
        self,
        wheel_radius: float = 0.08,
        wheel_track: float = 0.44,
    ):
        self.wheel_radius = wheel_radius
        self.wheel_track = wheel_track

    def calculate_wheel_velocities(
        self,
        linear_velocity: float,
        angular_velocity: float,
    ) -> tuple[float, float]:
        """
        Convert robot linear/angular velocity into wheel
        angular velocities.

        Args:
            linear_velocity:
                Robot linear velocity in meters per second.

            angular_velocity:
                Robot angular velocity in radians per second.

        Returns:
            left_wheel_velocity:
                Left wheel angular velocity in radians per second.

            right_wheel_velocity:
                Right wheel angular velocity in radians per second.
        """

        left_linear_velocity = (
            linear_velocity
            - (angular_velocity * self.wheel_track / 2.0)
        )

        right_linear_velocity = (
            linear_velocity
            + (angular_velocity * self.wheel_track / 2.0)
        )

        left_wheel_velocity = (
            left_linear_velocity / self.wheel_radius
        )

        right_wheel_velocity = (
            right_linear_velocity / self.wheel_radius
        )

        return (
            left_wheel_velocity,
            right_wheel_velocity,
        )