from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Waypoint:
    """
    A target position in the local navigation coordinate system.
    """

    x: float
    y: float


@dataclass(frozen=True)
class Path:
    """
    An ordered sequence of waypoints forming a navigation path.
    """

    waypoints: tuple[Waypoint, ...]


class NavigationState(Enum):
    """
    Current state of the navigation system.
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
    Desired robot motion.

    linear_velocity:
        Forward/backward velocity in meters per second.

    angular_velocity:
        Rotational velocity in radians per second.
    """

    linear_velocity: float
    angular_velocity: float