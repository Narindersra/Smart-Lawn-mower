class IMUReader:
    """
    Provides a clean interface for reading IMU orientation data.
    """

    def __init__(self, imu):
        self.imu = imu

    def initialize(self, timestep: int) -> None:
        """Enable the IMU sensor."""
        self.imu.enable(timestep)

    def get_heading(self) -> float:
        """
        Read the current robot heading.

        Returns:
            Heading/yaw in radians.
        """
        _, _, yaw = self.imu.getRollPitchYaw()
        return yaw