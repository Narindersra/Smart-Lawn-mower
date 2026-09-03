import math


class DifferentialDriveOdometry:
    """
    Calculates robot movement from left and right wheel encoder positions.

    Webots encoder values are wheel rotation angles in radians.
    """

    def __init__(
        self,
        wheel_radius: float = 0.08,
        wheel_track: float = 0.44,
    ):
        self.wheel_radius = wheel_radius
        self.wheel_track = wheel_track

        self.previous_left_position = None
        self.previous_right_position = None

    def update(
        self,
        left_encoder_position: float,
        right_encoder_position: float,
    ) -> dict[str, float]:
        """
        Update odometry using the current encoder positions.

        Returns:
            left_distance: Left wheel distance travelled in meters.
            right_distance: Right wheel distance travelled in meters.
            distance: Average robot distance travelled in meters.
            heading_change: Robot heading change in radians.
        """

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

        left_delta = (
            left_encoder_position - self.previous_left_position
        )
        right_delta = (
            right_encoder_position - self.previous_right_position
        )

        self.previous_left_position = left_encoder_position
        self.previous_right_position = right_encoder_position

        left_distance = left_delta * self.wheel_radius
        right_distance = right_delta * self.wheel_radius

        distance = (left_distance + right_distance) / 2.0

        heading_change = (
            right_distance - left_distance
        ) / self.wheel_track

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