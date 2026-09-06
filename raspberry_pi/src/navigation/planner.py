from localization.position_estimator import RobotPose

from .geofence import Geofence
from .heading_controller import HeadingController
from .navigation_math import (
    calculate_distance_to_waypoint,
    calculate_heading_error,
    calculate_target_heading,
)
from .navigation_state_machine import NavigationStateMachine
from .navigation_types import (
    MotionCommand,
    NavigationState,
    Path,
    Waypoint,
)
from .obstacle_avoidance import ObstacleAvoidance
from .path_planner import PathPlanner
from .speed_controller import SpeedController


class NavigationPlanner:
    """
    Main navigation planner.

    Coordinates waypoint navigation, path progression, obstacle
    avoidance, geofence protection, heading control, and speed control.
    """

    def __init__(
        self,
        waypoint_tolerance: float = 0.15,
        geofence: Geofence | None = None,
    ):
        """
        Initialize the navigation planner.

        Args:
            waypoint_tolerance:
                Maximum distance in meters at which a waypoint is
                considered reached.

            geofence:
                Optional geofence defining the allowed operating area.
        """
        self.waypoint_tolerance = waypoint_tolerance
        self.geofence = geofence

        # Current navigation target and path state.
        self.current_waypoint: Waypoint | None = None
        self.current_path: Path | None = None
        self.current_waypoint_index = 0
        self.goal_waypoint: Waypoint | None = None

        # Navigation control components.
        self.heading_controller = HeadingController()
        self.speed_controller = SpeedController()
        self.state_machine = NavigationStateMachine()
        self.path_planner = PathPlanner()
        self.obstacle_avoidance = ObstacleAvoidance()

    def transition_to(
        self,
        new_state: NavigationState,
    ) -> None:
        """
        Request a navigation state transition.

        Args:
            new_state: Navigation state to transition into.
        """
        self.state_machine.transition_to(new_state)

    def replan(
        self,
        current_pose: RobotPose,
    ) -> None:
        """
        Generate a new path toward the active goal.

        Args:
            current_pose: Robot's current pose.

        Raises:
            RuntimeError: If no active goal exists.
        """
        if self.goal_waypoint is None:
            raise RuntimeError(
                "Cannot replan without an active goal."
            )

        new_path = self.path_planner.replan(
            current_pose=current_pose,
            goal=self.goal_waypoint,
        )

        self.current_path = new_path
        self.current_waypoint_index = 0
        self.current_waypoint = new_path.waypoints[0]

        self.state_machine.transition_to(
            NavigationState.NAVIGATING
        )

    @property
    def state(self) -> NavigationState:
        """Return the current navigation state."""
        return self.state_machine.get_state()

    def set_waypoint(
        self,
        waypoint: Waypoint,
    ) -> None:
        """
        Set a single navigation target.

        Setting a single waypoint clears any previously active path.
        """
        self.current_path = None
        self.current_waypoint = waypoint
        self.goal_waypoint = waypoint

        self.state_machine.transition_to(
            NavigationState.NAVIGATING
        )

    def set_path(
        self,
        path: Path,
    ) -> None:
        """
        Set an ordered navigation path.

        The planner starts navigation from the first waypoint.

        Args:
            path: Ordered collection of navigation waypoints.
        """
        if not path.waypoints:
            self.clear_path()
            return

        self.goal_waypoint = path.waypoints[-1]
        self.current_path = path
        self.current_waypoint_index = 0
        self.current_waypoint = path.waypoints[0]

        self.state_machine.transition_to(
            NavigationState.NAVIGATING
        )

    def clear_waypoint(self) -> None:
        """Clear the current navigation target and path state."""
        self.current_path = None
        self.current_waypoint = None
        self.current_waypoint_index = 0
        self.goal_waypoint = None

        self.state_machine.transition_to(
            NavigationState.IDLE
        )

    def clear_path(self) -> None:
        """Clear the current navigation path and waypoint."""
        self.current_path = None
        self.current_waypoint = None
        self.current_waypoint_index = 0
        self.goal_waypoint = None

        self.state_machine.transition_to(
            NavigationState.IDLE
        )

    def update(
        self,
        pose: RobotPose,
        obstacle_information=None,
    ) -> tuple[NavigationState, float, float, MotionCommand]:
        """
        Calculate the navigation command for the current robot state.

        The update cycle handles, in order:

        1. Replanning requests.
        2. Normal obstacle detection.
        3. Missing navigation targets.
        4. Geofence validation.
        5. Obstacle avoidance.
        6. Waypoint/path progression.
        7. Heading control.
        8. Speed control.

        Args:
            pose: Current robot pose.
            obstacle_information:
                Optional information about detected obstacles.

        Returns:
            Tuple containing:
                - Current navigation state.
                - Distance to the active waypoint.
                - Heading error in radians.
                - Motion command.
        """

        # Handle a pending replanning state.
        if self.state == NavigationState.REPLANNING:
            self.replan(pose)

        # A detected obstacle changes normal navigation into the
        # temporary avoidance state.
        if (
            self.state == NavigationState.NAVIGATING
            and obstacle_information is not None
            and obstacle_information.has_obstacle
        ):
            self.state_machine.transition_to(
                NavigationState.AVOIDING
            )

        # No target means there is no navigation command to generate.
        if self.current_waypoint is None:
            self.state_machine.transition_to(
                NavigationState.IDLE
            )

            return (
                self.state,
                0.0,
                0.0,
                MotionCommand(
                    linear_velocity=0.0,
                    angular_velocity=0.0,
                ),
            )

        # Validate both the current robot position and the target
        # waypoint against the configured geofence.
        if self.geofence is not None:
            geofence_violation = (
                not self.geofence.contains(pose)
                or not self.geofence.contains_position(
                    self.current_waypoint.x,
                    self.current_waypoint.y,
                )
            )

            if geofence_violation:
                if self.state != NavigationState.EMERGENCY_STOP:
                    self.state_machine.transition_to(
                        NavigationState.EMERGENCY_STOP
                    )

                return (
                    self.state,
                    0.0,
                    0.0,
                    MotionCommand(
                        linear_velocity=0.0,
                        angular_velocity=0.0,
                    ),
                )

            # If the geofence is valid again, resume navigation after
            # an emergency stop.
            if self.state == NavigationState.EMERGENCY_STOP:
                self.state_machine.resume_from_emergency_stop()

        # Handle normal obstacle avoidance.
        if self.state == NavigationState.AVOIDING:
            if obstacle_information is None:
                return (
                    self.state,
                    0.0,
                    0.0,
                    MotionCommand(
                        linear_velocity=0.0,
                        angular_velocity=0.0,
                    ),
                )

            if obstacle_information.has_obstacle:
                avoidance_command = self.obstacle_avoidance.calculate(
                    obstacle_information
                )

                return (
                    self.state,
                    0.0,
                    0.0,
                    avoidance_command,
                )

            # Once the obstacle is clear, request a new path from the
            # current pose toward the original goal.
            self.state_machine.transition_to(
                NavigationState.REPLANNING
            )

            self.replan(pose)

        # Calculate distance to the active waypoint.
        distance = calculate_distance_to_waypoint(
            pose,
            self.current_waypoint,
        )

        # Check whether the current waypoint has been reached.
        if distance <= self.waypoint_tolerance:
            # Single-waypoint navigation is complete.
            if self.current_path is None:
                self.state_machine.transition_to(
                    NavigationState.GOAL_REACHED
                )

                return (
                    self.state,
                    distance,
                    0.0,
                    MotionCommand(
                        linear_velocity=0.0,
                        angular_velocity=0.0,
                    ),
                )

            # Advance to the next waypoint in path mode.
            self.current_waypoint_index += 1

            # The complete path has been reached.
            if self.current_waypoint_index >= len(
                self.current_path.waypoints
            ):
                self.state_machine.transition_to(
                    NavigationState.GOAL_REACHED
                )

                return (
                    self.state,
                    distance,
                    0.0,
                    MotionCommand(
                        linear_velocity=0.0,
                        angular_velocity=0.0,
                    ),
                )

            # Select the next waypoint.
            self.current_waypoint = (
                self.current_path.waypoints[
                    self.current_waypoint_index
                ]
            )

        # Recalculate distance after possible waypoint progression.
        distance = calculate_distance_to_waypoint(
            pose,
            self.current_waypoint,
        )

        # Calculate the desired heading toward the active waypoint.
        target_heading = calculate_target_heading(
            pose,
            self.current_waypoint,
        )

        # Calculate the shortest heading error.
        heading_error = calculate_heading_error(
            pose.heading,
            target_heading,
        )

        # Convert heading error into angular velocity.
        angular_velocity = self.heading_controller.calculate(
            heading_error,
        )

        # Calculate linear velocity based on distance to the waypoint.
        linear_velocity = self.speed_controller.calculate(
            distance,
        )

        motion_command = MotionCommand(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )

        return (
            self.state,
            distance,
            heading_error,
            motion_command,
        )