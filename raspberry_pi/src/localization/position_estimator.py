import math
from dataclasses import dataclass


@dataclass
class RobotPose:
    x: float
    y: float
    heading: float


class PositionEstimator:
    """
    Estimates the robot pose from odometry.

    Pose:
        x       -> position in meters
        y       -> position in meters
        heading -> orientation in radians
    """

    def __init__(
        self,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_heading: float = 0.0,
    ):
        self.x = initial_x
        self.y = initial_y
        self.heading = initial_heading
        self.gps_initialized = False
        self.imu_initialized = False
        

    def update(
        self,
        distance: float,
        heading_change: float,
    ) -> RobotPose:
        """
        Update robot pose using incremental odometry.

        Args:
            distance: Robot movement since previous update, in meters.
            heading_change: Heading change since previous update, in radians.

        Returns:
            Current robot pose.
        """

        if not self.is_initialized():
            return self.get_pose()

        previous_heading = self.heading

        self.heading += heading_change

        self.heading = math.atan2(
            math.sin(self.heading),
            math.cos(self.heading),
        )

        midpoint_heading = (
            previous_heading + self.heading
        ) / 2.0

        self.x += distance * math.cos(midpoint_heading)
        self.y += distance * math.sin(midpoint_heading)

        return self.get_pose()

    def update_gps(
        self,
        gps_x: float,
        gps_y: float,
    ) -> None:
        """
        Initialize the estimator position from GPS.

        GPS is used as the initial local position reference.
        """
        if not self.gps_initialized:
            self.x = gps_x
            self.y = gps_y
            self.gps_initialized = True

    def update_imu(self, imu_heading: float) -> None:
        """
        Initialize the estimator heading from IMU.
        Subsequent heading updates come from wheel odometry.
        """
        if not self.imu_initialized:
            self.heading = math.atan2(
                math.sin(imu_heading),
                math.cos(imu_heading),
            )
            self.imu_initialized = True

    def is_initialized(self) -> bool:
        """Return whether the initial robot pose is available."""
        return self.gps_initialized and self.imu_initialized

    def get_pose(self) -> RobotPose:
        """Return the current robot pose."""
        return RobotPose(
            x=self.x,
            y=self.y,
            heading=self.heading,
        )