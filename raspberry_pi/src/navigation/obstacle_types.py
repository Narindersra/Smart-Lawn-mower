from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    """
    Represents an obstacle in the local navigation coordinate system.
    """

    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class ObstacleInformation:
    """
    Navigation-level information about detected obstacles.
    """

    obstacles: tuple[Obstacle, ...]

    @property
    def has_obstacle(self) -> bool:
        """Return whether any obstacle is currently known."""
        return bool(self.obstacles)