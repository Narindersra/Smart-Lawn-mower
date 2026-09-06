"""
Camera interface for the lawn mower perception system.

This module defines the abstract interface that concrete camera
implementations must follow.
"""

from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    """
    Abstract interface for a camera device.

    Concrete camera implementations must provide methods for
    initialization, frame acquisition, and resource cleanup.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the camera device and its resources.
        """
        raise NotImplementedError

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """
        Return the latest available camera frame.

        Returns:
            A NumPy array containing the camera frame, or None
            if no frame is currently available.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Release camera resources and shut down the device.
        """
        raise NotImplementedError