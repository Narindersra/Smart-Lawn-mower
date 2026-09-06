import numpy as np
from controller import Camera as WebotsCamera


# ============================================================
# Webots Camera Adapter
# ============================================================

class WebotsCameraAdapter:
    """Adapt the Webots camera interface for the mower system."""

    def __init__(self, camera: WebotsCamera):
        """Store the Webots camera device."""
        self.camera = camera

    # ========================================================
    # Camera Initialization
    # ========================================================

    def initialize(self, timestep: int) -> None:
        """Enable the Webots camera using the simulation timestep."""
        self.camera.enable(timestep)

    # ========================================================
    # Frame Acquisition
    # ========================================================

    def get_frame(self) -> np.ndarray | None:
        """
        Capture the current camera frame.

        Returns:
            A NumPy array containing the RGB camera frame,
            or None if a valid frame is unavailable.
        """
        image = self.camera.getImage()

        if image is None:
            return None

        width = self.camera.getWidth()
        height = self.camera.getHeight()

        image_data = np.frombuffer(
            image,
            dtype=np.uint8,
        )

        expected_size = width * height * 4

        if image_data.size != expected_size:
            return None

        # Webots provides 4 channels (RGBA).
        # The AI pipeline uses only the first 3 channels (RGB).
        return image_data.reshape(
            (height, width, 4)
        )[:, :, :3].copy()

    # ========================================================
    # Camera Shutdown
    # ========================================================

    def close(self) -> None:
        """Release camera resources when shutdown handling is required."""
        pass