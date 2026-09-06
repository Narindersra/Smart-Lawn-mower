# ============================================================
# Localization Manager
# ============================================================

from .gps import GPSReader
from .imu import IMUReader
from .odometry import DifferentialDriveOdometry
from .position_estimator import PositionEstimator


class LocalizationManager:
    """
    Coordinate localization sensors and maintain the robot pose.

    The manager combines:
        - Wheel encoder odometry
        - GPS position
        - IMU heading

    The resulting sensor information is passed to the
    PositionEstimator to maintain the current robot pose.
    """

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(
        self,
        wheel_radius: float = 0.08,
        wheel_track: float = 0.44,
        gps_origin_x: float = 0.0,
        gps_origin_y: float = 0.0,
    ):
        """Initialize localization components and configuration."""

        self.odometry = DifferentialDriveOdometry(
            wheel_radius=wheel_radius,
            wheel_track=wheel_track,
        )

        self.gps_origin_x = gps_origin_x
        self.gps_origin_y = gps_origin_y

        self.position_estimator = PositionEstimator()

        # Sensor readers are created during initialize().
        self.gps_reader = None
        self.imu_reader = None

    # ========================================================
    # Sensor Initialization
    # ========================================================

    def initialize(
        self,
        gps,
        imu,
        timestep: int,
    ) -> None:
        """
        Initialize the GPS and IMU sensor readers.

        Args:
            gps: Webots GPS device.
            imu: Webots IMU device.
            timestep: Webots simulation timestep in milliseconds.
        """

        self.gps_reader = GPSReader(
            gps,
            origin_x=self.gps_origin_x,
            origin_y=self.gps_origin_y,
        )

        self.imu_reader = IMUReader(imu)

        self.gps_reader.initialize(timestep)
        self.imu_reader.initialize(timestep)

    # ========================================================
    # Localization Update
    # ========================================================

    def update(
        self,
        left_encoder_position: float,
        right_encoder_position: float,
    ):
        """
        Update the robot pose using current sensor measurements.

        Wheel encoder positions are processed through odometry,
        while GPS and IMU provide position and heading updates
        to the position estimator.

        Args:
            left_encoder_position: Current left wheel encoder position.
            right_encoder_position: Current right wheel encoder position.

        Returns:
            Current RobotPose.

        Raises:
            RuntimeError: If the localization sensors have not
                been initialized.
        """

        if self.gps_reader is None or self.imu_reader is None:
            raise RuntimeError(
                "LocalizationManager must be initialized before update()."
            )

        # ----------------------------------------------------
        # Read GPS Position
        # ----------------------------------------------------

        gps_x, gps_y, _ = self.gps_reader.get_position()

        # ----------------------------------------------------
        # Read IMU Heading
        # ----------------------------------------------------

        imu_heading = self.imu_reader.get_heading()

        # ----------------------------------------------------
        # Update Position Estimator with GPS
        # ----------------------------------------------------

        self.position_estimator.update_gps(
            gps_x,
            gps_y,
        )

        # ----------------------------------------------------
        # Update Position Estimator with IMU
        # ----------------------------------------------------

        self.position_estimator.update_imu(
            imu_heading,
        )

        # ----------------------------------------------------
        # Update Wheel Odometry
        # ----------------------------------------------------

        odometry_data = self.odometry.update(
            left_encoder_position,
            right_encoder_position,
        )

        # ----------------------------------------------------
        # Calculate Current Robot Pose
        # ----------------------------------------------------

        return self.position_estimator.update(
            distance=odometry_data["distance"],
            heading_change=odometry_data["heading_change"],
        )

    # ========================================================
    # Localization State
    # ========================================================

    def is_ready(self) -> bool:
        """Return True when the localization estimator is initialized."""

        return self.position_estimator.is_initialized()