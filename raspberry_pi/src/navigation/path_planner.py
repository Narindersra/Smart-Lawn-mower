from localization.position_estimator import RobotPose

from .navigation_types import Path, Waypoint


class PathPlanner:
    """
    Generates navigation paths between a start pose and a goal waypoint.
    """

    def create_direct_path(
        self,
        start_pose: RobotPose,
        goal: Waypoint,
    ) -> Path:
        """
        Create a direct path from the robot's current position
        to the goal.

        The start position is not included as a waypoint.
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
        Generate a new path from the robot's current position
        to the original goal.
        """
    
        return self.create_direct_path(
            start_pose=current_pose,
            goal=goal,
        )