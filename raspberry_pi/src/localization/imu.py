class IMUReader:
    """
    Provides a clean interface for reading IMU orientation data.

    The reader exposes the robot's yaw angle as its navigation heading.
    """

    def __init__(self, imu):
        """Initialize the IMU reader with the provided IMU device."""
        self.imu = imu

    def initialize(self, timestep: int) -> None:
        """Enable the IMU sensor using the provided simulation timestep."""
        self.imu.enable(timestep)

    def get_heading(self) -> float:
        """
        Read the robot's current heading from the IMU.

        Webots provides roll, pitch, and yaw values. Only the yaw
        component is required for the robot's 2D navigation.

        Returns:
            Robot heading/yaw in radians.
        """
        _, _, yaw = self.imu.getRollPitchYaw()
        return yaw