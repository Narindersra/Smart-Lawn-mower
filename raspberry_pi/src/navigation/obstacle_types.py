from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    distance: float


@dataclass(frozen=True)
class ObstacleInformation:
    obstacles: tuple[Obstacle, ...]

    @property
    def has_obstacle(self) -> bool:
        return bool(self.obstacles)

    @property
    def nearest_distance(self) -> float | None:
        if not self.obstacles:
            return None

        return min(
            obstacle.distance
            for obstacle in self.obstacles
        )