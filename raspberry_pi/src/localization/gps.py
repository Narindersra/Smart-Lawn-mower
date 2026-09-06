class GPSReader:
    """
    Provides a clean interface for reading GPS position data.

    The Webots GPS coordinates are converted into the project's
    local coordinate system using the configured origin.
    """

    def __init__(
        self,
        gps,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        """Initialize the GPS reader with a Webots GPS device."""
        self.gps = gps
        self.origin_x = origin_x
        self.origin_y = origin_y

    def initialize(self, timestep: int) -> None:
        """Enable the GPS sensor using the provided simulation timestep."""
        self.gps.enable(timestep)

    def get_position(self) -> tuple[float, float, float]:
        """
        Read the current GPS position in the local coordinate system.

        Webots returns coordinates as (x, y, z). For the mower's
        2D navigation plane, Webots X/Z are mapped to the project's
        local X/Y coordinates.

        Returns:
            A tuple containing:
                - local_x: Local X position in meters.
                - local_y: Local Y position in meters.
                - y: Webots vertical position in meters.
        """
        x, y, z = self.gps.getValues()

        # Convert Webots world coordinates to the project's local frame.
        local_x = x - self.origin_x
        local_y = z - self.origin_y

        return local_x, local_y, y