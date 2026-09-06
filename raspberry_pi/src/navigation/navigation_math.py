import math

from localization.position_estimator import RobotPose

from .navigation_types import Waypoint


def calculate_distance_to_waypoint(
    pose: RobotPose,
    waypoint: Waypoint,
) -> float:
    """
    Calculate the straight-line distance from the robot to a waypoint.

    Args:
        pose: Current robot pose.
        waypoint: Target waypoint.

    Returns:
        Distance from the robot to the waypoint in meters.
    """
    dx = waypoint.x - pose.x
    dy = waypoint.y - pose.y

    return math.hypot(dx, dy)


def calculate_target_heading(
    pose: RobotPose,
    waypoint: Waypoint,
) -> float:
    """
    Calculate the heading from the robot toward a target waypoint.

    Args:
        pose: Current robot pose.
        waypoint: Target waypoint.

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
    Calculate the shortest angular difference between two headings.

    The resulting error is normalized to the range [-pi, pi].

    Args:
        current_heading: Robot's current heading in radians.
        target_heading: Desired target heading in radians.

    Returns:
        Normalized heading error in radians.
    """
    error = target_heading - current_heading

    return math.atan2(
        math.sin(error),
        math.cos(error),
    )