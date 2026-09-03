from .odometry import DifferentialDriveOdometry
from .position_estimator import PositionEstimator
from .gps import GPSReader
from .imu import IMUReader


class LocalizationManager:
    """
    Coordinates sensor data and maintains the robot pose.
    """

    def __init__(
        self,
        wheel_radius: float = 0.08,
        wheel_track: float = 0.44,
        gps_origin_x: float = 0.0,
        gps_origin_y: float = 0.0,
    ):
        self.odometry = DifferentialDriveOdometry(
            wheel_radius=wheel_radius,
            wheel_track=wheel_track,
        )

        self.gps_origin_x = gps_origin_x
        self.gps_origin_y = gps_origin_y

        self.position_estimator = PositionEstimator()
        self.gps_reader = None
        self.imu_reader = None

    def initialize(
        self,
        gps,
        imu,
        timestep: int,
    ) -> None:
        """
        Initialize GPS and IMU sensor readers.
        """

        self.gps_reader = GPSReader(
            gps,
            origin_x=self.gps_origin_x,
            origin_y=self.gps_origin_y,
        )
        self.imu_reader = IMUReader(imu)

        self.gps_reader.initialize(timestep)
        self.imu_reader.initialize(timestep)

    def update(
        self,
        left_encoder_position: float,
        right_encoder_position: float,
    ):
        """
        Update localization from encoder, GPS and IMU data.

        Returns:
            Current RobotPose.
        """
        if self.gps_reader is None or self.imu_reader is None:
            raise RuntimeError(
                "LocalizationManager must be initialized before update()."
            )

        gps_x, gps_y, _ = self.gps_reader.get_position()

        imu_heading = self.imu_reader.get_heading()
        
        self.position_estimator.update_gps(
            gps_x,
            gps_y,
        )
        
        self.position_estimator.update_imu(
            imu_heading,
        )
        odometry_data = self.odometry.update(
            left_encoder_position,
            right_encoder_position,
        )

        return self.position_estimator.update(
            distance=odometry_data["distance"],
            heading_change=odometry_data["heading_change"],
        )

    def is_ready(self) -> bool:
        """Return whether localization has been initialized."""
        return self.position_estimator.is_initialized()