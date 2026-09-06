from localization.position_estimator import RobotPose

from .navigation_types import Path, Waypoint


class PathPlanner:
    """
    Generates navigation paths between the current robot pose
    and a target waypoint.

    The current implementation uses direct-path planning.
    """

    def create_direct_path(
        self,
        start_pose: RobotPose,
        goal: Waypoint,
    ) -> Path:
        """
        Create a direct path from the robot's current position
        to the goal waypoint.

        The start position is intentionally not included as a waypoint.

        Args:
            start_pose: Current robot pose.
            goal: Target waypoint.

        Returns:
            A Path containing the target waypoint.
        """

        return Path(
            waypoints=(
                goal,
            )
        )

    def replan(
        self,
        current_pose: RobotPose,
        goal: Waypoint,
    ) -> Path:
        """
        Generate a new path from the current robot pose to the goal.

        The current implementation replans using the same direct-path
        strategy as the initial path generation.

        Args:
            current_pose: Robot's current pose.
            goal: Original target waypoint.

        Returns:
            A new direct Path containing the target waypoint.
        """

        return self.create_direct_path(
            start_pose=current_pose,
            goal=goal,
        )