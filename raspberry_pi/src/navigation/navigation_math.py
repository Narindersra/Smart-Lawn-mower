import math

from localization.position_estimator import RobotPose
from .navigation_types import Waypoint


def calculate_distance_to_waypoint(
    pose: RobotPose,
    waypoint: Waypoint,
) -> float:
    """
    Calculate straight-line distance from the robot to a waypoint.

    Returns:
        Distance in meters.
    """

    dx = waypoint.x - pose.x
    dy = waypoint.y - pose.y

    return math.hypot(dx, dy)


def calculate_target_heading(
    pose: RobotPose,
    waypoint: Waypoint,
) -> float:
    """
    Calculate the heading from the robot's current position
    toward the target waypoint.

    Returns:
        Target heading in radians.
    """

    dx = waypoint.x - pose.x
    dy = waypoint.y - pose.y

    return math.atan2(dy, dx)


def calculate_heading_error(
    current_heading: float,
    target_heading: float,
) -> float:
    """
    Calculate the shortest angular difference between
    the current heading and target heading.

    Returns:
        Heading error in radians within [-pi, pi].
    """

    error = target_heading - current_heading

    return math.atan2(
        math.sin(error),
        math.cos(error),
    )