from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    """
    Represents a detected obstacle.

    Attributes:
        distance: Distance from the robot to the obstacle in meters.
    """

    distance: float


@dataclass(frozen=True)
class ObstacleInformation:
    """
    Contains information about currently detected obstacles.

    Attributes:
        obstacles: Tuple containing all detected obstacles.
    """

    obstacles: tuple[Obstacle, ...]

    @property
    def has_obstacle(self) -> bool:
        """
        Return whether at least one obstacle is detected.

        Returns:
            True if one or more obstacles are present; otherwise False.
        """
        return bool(self.obstacles)

    @property
    def nearest_distance(self) -> float | None:
        """
        Return the distance to the nearest detected obstacle.

        Returns:
            Nearest obstacle distance in meters, or None when no
            obstacles are detected.
        """
        if not self.obstacles:
            return None

        return min(
            obstacle.distance
            for obstacle in self.obstacles
        )