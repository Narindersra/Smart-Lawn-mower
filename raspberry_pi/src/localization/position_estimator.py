import math
from dataclasses import dataclass


@dataclass
class RobotPose:
    """
    Represents the current robot pose in the local coordinate system.

    Attributes:
        x: Robot X position in meters.
        y: Robot Y position in meters.
        heading: Robot orientation in radians.
    """

    x: float
    y: float
    heading: float


class PositionEstimator:
    """
    Estimates and maintains the robot pose.

    GPS and IMU are used only to initialize the initial position and
    heading. After initialization, incremental pose updates are
    calculated using wheel odometry.
    """

    def __init__(
        self,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_heading: float = 0.0,
    ):
        """
        Initialize the position estimator.

        Args:
            initial_x: Initial X position in meters.
            initial_y: Initial Y position in meters.
            initial_heading: Initial heading in radians.
        """
        self.x = initial_x
        self.y = initial_y
        self.heading = initial_heading

        # Initialization flags ensure GPS and IMU are both available
        # before odometry updates are applied.
        self.gps_initialized = False
        self.imu_initialized = False

    def update(
        self,
        distance: float,
        heading_change: float,
    ) -> RobotPose:
        """
        Update the robot pose using incremental odometry.

        Args:
            distance: Robot movement since the previous update,
                in meters.
            heading_change: Heading change since the previous update,
                in radians.

        Returns:
            The current robot pose.
        """
        # Odometry updates are ignored until both initial position
        # and initial heading have been established.
        if not self.is_initialized():
            return self.get_pose()

        previous_heading = self.heading

        # Apply the incremental heading change.
        self.heading += heading_change

        # Normalize heading to the range [-pi, pi].
        self.heading = math.atan2(
            math.sin(self.heading),
            math.cos(self.heading),
        )

        # Use the midpoint heading during the movement interval.
        # This provides the direction used for the incremental
        # position update.
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

        GPS provides the initial local position reference. Once
        initialized, subsequent GPS readings do not overwrite
        the odometry-based position.
        """
        if not self.gps_initialized:
            self.x = gps_x
            self.y = gps_y
            self.gps_initialized = True

    def update_imu(self, imu_heading: float) -> None:
        """
        Initialize the estimator heading from the IMU.

        The IMU provides the initial heading reference. After
        initialization, subsequent heading updates come from
        wheel odometry.
        """
        if not self.imu_initialized:
            self.heading = math.atan2(
                math.sin(imu_heading),
                math.cos(imu_heading),
            )
            self.imu_initialized = True

    def is_initialized(self) -> bool:
        """
        Return whether the initial robot pose is available.

        Returns:
            True when both GPS position and IMU heading have been
            initialized; otherwise False.
        """
        return self.gps_initialized and self.imu_initialized

    def get_pose(self) -> RobotPose:
        """
        Return the current robot pose.

        Returns:
            Current X position, Y position, and heading.
        """
        return RobotPose(
            x=self.x,
            y=self.y,
            heading=self.heading,
        )