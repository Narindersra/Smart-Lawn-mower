from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Waypoint:
    """
    Represents a target position in the local navigation
    coordinate system.

    Attributes:
        x: Target X position in meters.
        y: Target Y position in meters.
    """

    x: float
    y: float


@dataclass(frozen=True)
class Path:
    """
    Represents an ordered navigation path.

    A path consists of one or more waypoints that the robot
    can follow sequentially.

    Attributes:
        waypoints: Ordered tuple of navigation waypoints.
    """

    waypoints: tuple[Waypoint, ...]


class NavigationState(Enum):
    """
    Represents the current state of the navigation system.
    """

    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    REPLANNING = "replanning"
    GOAL_REACHED = "goal_reached"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class MotionCommand:
    """
    Represents the desired robot motion.

    Attributes:
        linear_velocity:
            Forward/backward velocity in meters per second.

        angular_velocity:
            Rotational velocity in radians per second.
    """

    linear_velocity: float
    angular_velocity: float