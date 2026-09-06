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
        """
        Initialize the differential-drive controller.

        Args:
            wheel_radius: Radius of the drive wheels in meters.
            wheel_track: Distance between the left and right wheels
                in meters.
        """
        self.wheel_radius = wheel_radius
        self.wheel_track = wheel_track

    def calculate_wheel_velocities(
        self,
        linear_velocity: float,
        angular_velocity: float,
    ) -> tuple[float, float]:
        """
        Convert robot linear and angular velocity into wheel
        angular velocities.

        Args:
            linear_velocity:
                Robot linear velocity in meters per second.

            angular_velocity:
                Robot angular velocity in radians per second.

        Returns:
            Tuple containing:
                - left wheel angular velocity in radians per second.
                - right wheel angular velocity in radians per second.
        """

        # Convert the robot's linear/angular command into the
        # corresponding linear velocity of each wheel.
        left_linear_velocity = (
            linear_velocity
            - (angular_velocity * self.wheel_track / 2.0)
        )

        right_linear_velocity = (
            linear_velocity
            + (angular_velocity * self.wheel_track / 2.0)
        )

        # Convert wheel linear velocities into wheel angular
        # velocities using the wheel radius.
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