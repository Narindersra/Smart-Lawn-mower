from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the camera."""
        raise NotImplementedError

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """Return the latest camera frame."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""
        raise NotImplementedError