import math


class DifferentialDriveOdometry:
    """
    Calculates robot movement from left and right wheel encoder positions.

    Webots encoder values represent wheel rotation angles in radians.
    These rotations are converted into wheel distances using the wheel
    radius, then used to estimate robot distance and heading change.
    """

    def __init__(
        self,
        wheel_radius: float = 0.08,
        wheel_track: float = 0.44,
    ):
        """
        Initialize the differential-drive odometry.

        Args:
            wheel_radius: Radius of each drive wheel in meters.
            wheel_track: Distance between the left and right wheels
                in meters.
        """
        self.wheel_radius = wheel_radius
        self.wheel_track = wheel_track

        # Previous encoder readings are used to calculate wheel movement
        # between consecutive odometry updates.
        self.previous_left_position = None
        self.previous_right_position = None

    def update(
        self,
        left_encoder_position: float,
        right_encoder_position: float,
    ) -> dict[str, float]:
        """
        Update odometry using the current encoder positions.

        The first update establishes the initial encoder reference and
        therefore reports zero movement.

        Args:
            left_encoder_position: Current left wheel encoder position
                in radians.
            right_encoder_position: Current right wheel encoder position
                in radians.

        Returns:
            Dictionary containing:
                - left_distance: Left wheel distance travelled in meters.
                - right_distance: Right wheel distance travelled in meters.
                - distance: Average robot distance travelled in meters.
                - heading_change: Robot heading change in radians.
        """

        # Establish the initial encoder reference.
        if (
            self.previous_left_position is None
            or self.previous_right_position is None
        ):
            self.previous_left_position = left_encoder_position
            self.previous_right_position = right_encoder_position

            return {
                "left_distance": 0.0,
                "right_distance": 0.0,
                "distance": 0.0,
                "heading_change": 0.0,
            }

        # Calculate wheel rotation since the previous update.
        left_delta = (
            left_encoder_position - self.previous_left_position
        )
        right_delta = (
            right_encoder_position - self.previous_right_position
        )

        # Store the current encoder positions for the next update.
        self.previous_left_position = left_encoder_position
        self.previous_right_position = right_encoder_position

        # Convert wheel rotation from radians to linear distance.
        left_distance = left_delta * self.wheel_radius
        right_distance = right_delta * self.wheel_radius

        # Estimate the robot's forward displacement.
        distance = (left_distance + right_distance) / 2.0

        # Estimate the robot's heading change from the difference
        # between the two wheel distances.
        heading_change = (
            right_distance - left_distance
        ) / self.wheel_track

        # Normalize the heading change to the range [-pi, pi].
        heading_change = math.atan2(
            math.sin(heading_change),
            math.cos(heading_change),
        )

        return {
            "left_distance": left_distance,
            "right_distance": right_distance,
            "distance": distance,
            "heading_change": heading_change,
        }