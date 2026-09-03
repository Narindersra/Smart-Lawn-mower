class GPSReader:
    """
    Provides a clean interface for reading GPS data.
    """

    def __init__(
        self,
        gps,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        self.gps = gps
        self.origin_x = origin_x
        self.origin_y = origin_y

    def initialize(self, timestep: int) -> None:
        """Enable the GPS sensor."""
        self.gps.enable(timestep)

    def get_position(self) -> tuple[float, float, float]:
        """
        Read GPS position relative to the configured local origin.
    
        Returns:
            x, y, z position in meters.
        """
        x, y, z = self.gps.getValues()
    
        local_x = x - self.origin_x
        local_y = y - self.origin_y
    
        return local_x, local_y, z