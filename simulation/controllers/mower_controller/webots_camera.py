import numpy as np
from controller import Camera as WebotsCamera


class WebotsCameraAdapter:
    def __init__(self, camera: WebotsCamera):
        self.camera = camera

    def initialize(self, timestep: int) -> None:
        self.camera.enable(timestep)

    def get_frame(self) -> np.ndarray | None:
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

        return image_data.reshape(
            (height, width, 4)
        )[:, :, :3].copy()

    def close(self) -> None:
        pass