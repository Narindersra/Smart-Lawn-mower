from .navigation_math import (
    calculate_distance_to_waypoint,
    calculate_target_heading,
    calculate_heading_error,
)
from .navigation_types import (
    MotionCommand,
    NavigationState,
    Path,
    Waypoint,
)
from localization.position_estimator import RobotPose
from .heading_controller import HeadingController
from .speed_controller import SpeedController
from .geofence import Geofence
from .navigation_state_machine import NavigationStateMachine
from .path_planner import PathPlanner
from .obstacle_avoidance import ObstacleAvoidance


class NavigationPlanner:
    """
    Main navigation planner.

    Determines the motion required to move the robot
    toward the current target waypoint.
    """

    def __init__(
        self,
        waypoint_tolerance: float = 0.15,
        geofence: Geofence | None = None,
    ):
        self.waypoint_tolerance = waypoint_tolerance
        self.geofence = geofence
        self.current_waypoint: Waypoint | None = None
        self.current_path: Path | None = None
        self.current_waypoint_index = 0
        self.heading_controller = HeadingController()
        self.speed_controller = SpeedController()
        self.state_machine = NavigationStateMachine()
        self.path_planner = PathPlanner()
        self.obstacle_avoidance = ObstacleAvoidance()
        self.goal_waypoint: Waypoint | None = None

    def transition_to(
        self,
        new_state: NavigationState,
    ) -> None:
        """
        Request a navigation state transition.
        """

        self.state_machine.transition_to(new_state)

    def replan(
        self,
        current_pose: RobotPose,
    ) -> None:
        """
        Generate a new path toward the original goal
        from the robot's current pose.
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

    def set_waypoint(self, waypoint: Waypoint) -> None:
        """
        Set the current navigation target.
        """
        self.current_path = None
        self.current_waypoint = waypoint
        self.goal_waypoint = waypoint
        self.state_machine.transition_to(
            NavigationState.NAVIGATING
        )
        

    def set_path(self, path: Path) -> None:
        """
        Set an ordered navigation path.
    
        The planner starts from the first waypoint.
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
        """
        Clear the current navigation target.
        """
        self.current_path = None
        self.current_waypoint = None
        self.current_waypoint_index = 0
        self.goal_waypoint = None
        self.state_machine.transition_to(
            NavigationState.IDLE
        )
        

    def clear_path(self) -> None:
        """
        Clear the current navigation path and waypoint.
        """
    
        self.current_path = None
        self.current_waypoint = None
        self.current_waypoint_index = 0
        self.goal_waypoint = None
        self.state_machine.transition_to(NavigationState.IDLE)
        

    def update(
        self,
        pose: RobotPose,
        obstacle_information=None,
    ) -> tuple[NavigationState, float, float, MotionCommand]:
        """
        Calculate the robot's relationship to the current waypoint.

        Returns:
            navigation state,
            distance to waypoint,
            heading error.
        """

        if self.state == NavigationState.REPLANNING:
            self.replan(pose)

        if (
            self.state == NavigationState.NAVIGATING
            and obstacle_information is not None
            and obstacle_information.has_obstacle
        ):
            self.state_machine.transition_to(
                NavigationState.AVOIDING
            )

        if self.current_waypoint is None:
            self.state_machine.transition_to(NavigationState.IDLE)

            return (
                self.state,
                0.0,
                0.0,
                MotionCommand(
                    linear_velocity=0.0,
                    angular_velocity=0.0,
                ),
            )

        if self.geofence is not None:
            if (
                not self.geofence.contains(pose)
                or not self.geofence.contains_position(
                    self.current_waypoint.x,
                    self.current_waypoint.y,
                )
            ):
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

            self.state_machine.transition_to(
                NavigationState.REPLANNING
            )
            self.replan(pose)

        distance = calculate_distance_to_waypoint(
            pose,
            self.current_waypoint,
        )

        if distance <= self.waypoint_tolerance:
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

            self.current_waypoint_index += 1

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

            self.current_waypoint = (
                self.current_path.waypoints[
                    self.current_waypoint_index
                ]
            )

        distance = calculate_distance_to_waypoint(
            pose,
            self.current_waypoint,
        )

        target_heading = calculate_target_heading(
            pose,
            self.current_waypoint,
        )

        heading_error = calculate_heading_error(
            pose.heading,
            target_heading,
        )
        angular_velocity = self.heading_controller.calculate(
            heading_error,
        )

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