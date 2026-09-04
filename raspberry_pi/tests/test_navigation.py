import unittest
import math

from localization.position_estimator import RobotPose

from navigation.navigation_types import (
    Waypoint,
    Path,
    NavigationState,
    MotionCommand,
)

from navigation.navigation_math import (
    calculate_distance_to_waypoint,
    calculate_target_heading,
    calculate_heading_error,
)

from navigation.differential_drive import (
    DifferentialDriveController,
)

from navigation.heading_controller import (
    HeadingController,
)

from navigation.speed_controller import (
    SpeedController,
)

from navigation.navigation_state_machine import (
    NavigationStateMachine,
)

from navigation.obstacle_types import (
    Obstacle,
    ObstacleInformation,
)

from navigation.obstacle_avoidance import (
    ObstacleAvoidance,
)

from navigation.path_planner import (
    PathPlanner,
)

from navigation.planner import (
    NavigationPlanner,
)

from navigation.geofence import (
    Geofence,
)


# ============================================================
# NAVIGATION TYPES
# ============================================================

class TestNavigationTypes(unittest.TestCase):

    def test_waypoint(self):
        waypoint = Waypoint(
            x=2.0,
            y=3.0,
        )

        self.assertEqual(waypoint.x, 2.0)
        self.assertEqual(waypoint.y, 3.0)

    def test_path(self):
        waypoint_1 = Waypoint(1.0, 2.0)
        waypoint_2 = Waypoint(3.0, 4.0)

        path = Path(
            waypoints=(
                waypoint_1,
                waypoint_2,
            )
        )

        self.assertEqual(len(path.waypoints), 2)
        self.assertEqual(path.waypoints[0], waypoint_1)
        self.assertEqual(path.waypoints[1], waypoint_2)

    def test_navigation_states_exist(self):
        self.assertEqual(
            NavigationState.IDLE.name,
            "IDLE",
        )

        self.assertEqual(
            NavigationState.NAVIGATING.name,
            "NAVIGATING",
        )

        self.assertEqual(
            NavigationState.AVOIDING.name,
            "AVOIDING",
        )

        self.assertEqual(
            NavigationState.REPLANNING.name,
            "REPLANNING",
        )

        self.assertEqual(
            NavigationState.GOAL_REACHED.name,
            "GOAL_REACHED",
        )

        self.assertEqual(
            NavigationState.EMERGENCY_STOP.name,
            "EMERGENCY_STOP",
        )

    def test_motion_command(self):
        command = MotionCommand(
            linear_velocity=0.4,
            angular_velocity=0.2,
        )

        self.assertEqual(
            command.linear_velocity,
            0.4,
        )

        self.assertEqual(
            command.angular_velocity,
            0.2,
        )


# ============================================================
# NAVIGATION MATH
# ============================================================

class TestNavigationMath(unittest.TestCase):

    def test_distance_to_waypoint(self):
        pose = RobotPose(
            x=0.0,
            y=0.0,
            heading=0.0,
        )

        waypoint = Waypoint(
            x=3.0,
            y=4.0,
        )

        distance = calculate_distance_to_waypoint(
            pose,
            waypoint,
        )

        self.assertAlmostEqual(
            distance,
            5.0,
        )

    def test_distance_same_position(self):
        pose = RobotPose(
            x=2.0,
            y=3.0,
            heading=0.0,
        )

        waypoint = Waypoint(
            x=2.0,
            y=3.0,
        )

        distance = calculate_distance_to_waypoint(
            pose,
            waypoint,
        )

        self.assertAlmostEqual(
            distance,
            0.0,
        )

    def test_target_heading(self):
        pose = RobotPose(
            x=0.0,
            y=0.0,
            heading=0.0,
        )

        waypoint = Waypoint(
            x=1.0,
            y=0.0,
        )

        heading = calculate_target_heading(
            pose,
            waypoint,
        )

        self.assertAlmostEqual(
            heading,
            0.0,
        )

    def test_target_heading_north(self):
        pose = RobotPose(
            x=0.0,
            y=0.0,
            heading=0.0,
        )

        waypoint = Waypoint(
            x=0.0,
            y=1.0,
        )

        heading = calculate_target_heading(
            pose,
            waypoint,
        )

        self.assertAlmostEqual(
            heading,
            math.pi / 2,
        )

    def test_heading_error(self):
        error = calculate_heading_error(
            current_heading=0.0,
            target_heading=math.pi / 2,
        )

        self.assertAlmostEqual(
            error,
            math.pi / 2,
        )

    def test_heading_error_wrap_positive(self):
        error = calculate_heading_error(
            current_heading=math.pi,
            target_heading=-math.pi + 0.1,
        )

        self.assertAlmostEqual(
            error,
            0.1,
        )

    def test_heading_error_wrap_negative(self):
        error = calculate_heading_error(
            current_heading=-math.pi,
            target_heading=math.pi - 0.1,
        )

        self.assertAlmostEqual(
            error,
            -0.1,
        )


# ============================================================
# DIFFERENTIAL DRIVE
# ============================================================

class TestDifferentialDriveController(unittest.TestCase):

    def setUp(self):
        self.controller = DifferentialDriveController(
            wheel_radius=0.08,
            wheel_track=0.44,
        )

    def test_constructor(self):
        self.assertEqual(
            self.controller.wheel_radius,
            0.08,
        )

        self.assertEqual(
            self.controller.wheel_track,
            0.44,
        )

    def test_zero_command(self):
        left, right = (
            self.controller.calculate_wheel_velocities(
                linear_velocity=0.0,
                angular_velocity=0.0,
            )
        )

        self.assertAlmostEqual(left, 0.0)
        self.assertAlmostEqual(right, 0.0)

    def test_forward_command(self):
        left, right = (
            self.controller.calculate_wheel_velocities(
                linear_velocity=0.4,
                angular_velocity=0.0,
            )
        )

        self.assertAlmostEqual(
            left,
            right,
        )

        self.assertGreater(
            left,
            0.0,
        )

    def test_rotation_command(self):
        left, right = (
            self.controller.calculate_wheel_velocities(
                linear_velocity=0.0,
                angular_velocity=1.0,
            )
        )

        self.assertLess(
            left,
            0.0,
        )

        self.assertGreater(
            right,
            0.0,
        )


# ============================================================
# HEADING CONTROLLER
# ============================================================

class TestHeadingController(unittest.TestCase):

    def setUp(self):
        self.controller = HeadingController(
            proportional_gain=2.0,
            max_angular_velocity=1.5,
        )

    def test_zero_error(self):
        output = self.controller.calculate(
            heading_error=0.0,
        )

        self.assertAlmostEqual(
            output,
            0.0,
        )

    def test_positive_error(self):
        output = self.controller.calculate(
            heading_error=0.5,
        )

        self.assertGreater(
            output,
            0.0,
        )

        self.assertAlmostEqual(
            output,
            1.0,
        )

    def test_negative_error(self):
        output = self.controller.calculate(
            heading_error=-0.5,
        )

        self.assertLess(
            output,
            0.0,
        )

    def test_output_limit(self):
        output = self.controller.calculate(
            heading_error=10.0,
        )

        self.assertLessEqual(
            abs(output),
            1.5,
        )


# ============================================================
# SPEED CONTROLLER
# ============================================================

class TestSpeedController(unittest.TestCase):

    def setUp(self):
        self.controller = SpeedController(
            maximum_speed=0.5,
            minimum_speed=0.1,
            slowdown_distance=1.0,
        )

    def test_far_from_goal(self):
        speed = self.controller.calculate(
            distance_to_goal=10.0,
        )

        self.assertAlmostEqual(
            speed,
            0.5,
        )

    def test_near_goal(self):
        speed = self.controller.calculate(
            distance_to_goal=0.1,
        )

        self.assertGreaterEqual(
            speed,
            0.1,
        )

        self.assertLessEqual(
            speed,
            0.5,
        )

    def test_zero_distance(self):
        speed = self.controller.calculate(
            distance_to_goal=0.0,
        )

        self.assertGreaterEqual(
            speed,
            0.0,
        )


# ============================================================
# NAVIGATION STATE MACHINE
# ============================================================

class TestNavigationStateMachine(unittest.TestCase):

    def setUp(self):
        self.machine = NavigationStateMachine()

    def test_initial_state(self):
        self.assertEqual(
            self.machine.get_state(),
            NavigationState.IDLE,
        )

    def test_valid_transition_to_navigating(self):
        self.machine.transition_to(
            NavigationState.NAVIGATING
        )

        self.assertEqual(
            self.machine.get_state(),
            NavigationState.NAVIGATING,
        )

    def test_navigation_to_avoiding(self):
        self.machine.transition_to(
            NavigationState.NAVIGATING
        )

        self.machine.transition_to(
            NavigationState.AVOIDING
        )

        self.assertEqual(
            self.machine.get_state(),
            NavigationState.AVOIDING,
        )

    def test_navigation_to_goal_reached(self):
        self.machine.transition_to(
            NavigationState.NAVIGATING
        )

        self.machine.transition_to(
            NavigationState.GOAL_REACHED
        )

        self.assertEqual(
            self.machine.get_state(),
            NavigationState.GOAL_REACHED,
        )

    def test_navigation_to_emergency_stop(self):
        self.machine.transition_to(
            NavigationState.NAVIGATING
        )

        self.machine.transition_to(
            NavigationState.EMERGENCY_STOP
        )

        self.assertEqual(
            self.machine.get_state(),
            NavigationState.EMERGENCY_STOP,
        )


# ============================================================
# OBSTACLE TYPES
# ============================================================

class TestObstacleTypes(unittest.TestCase):

    def test_obstacle(self):
        obstacle = Obstacle(
            distance=0.4,
        )

        self.assertEqual(
            obstacle.distance,
            0.4,
        )

    def test_obstacle_information_with_obstacle(self):
        information = ObstacleInformation(
            obstacles=[
                Obstacle(distance=0.4),
                Obstacle(distance=1.0),
            ]
        )

        self.assertTrue(
            information.has_obstacle
        )

        self.assertAlmostEqual(
            information.nearest_distance,
            0.4,
        )

    def test_obstacle_information_without_obstacle(self):
        information = ObstacleInformation(
            obstacles=[]
        )

        self.assertFalse(
            information.has_obstacle
        )


# ============================================================
# OBSTACLE AVOIDANCE
# ============================================================

class TestObstacleAvoidance(unittest.TestCase):

    def setUp(self):
        self.avoidance = ObstacleAvoidance(
            avoidance_linear_velocity=0.15,
            avoidance_angular_velocity=0.8,
            clearance_distance=0.8,
        )

    def test_clear_path(self):
        information = ObstacleInformation(
            obstacles=[
                Obstacle(distance=2.0),
            ]
        )

        self.assertTrue(
            self.avoidance.is_clear(information)
        )

    def test_blocked_path(self):
        information = ObstacleInformation(
            obstacles=[
                Obstacle(distance=0.5),
            ]
        )

        self.assertFalse(
            self.avoidance.is_clear(information)
        )

    def test_avoidance_command(self):
        information = ObstacleInformation(
            obstacles=[
                Obstacle(distance=0.5),
            ]
        )

        command = self.avoidance.calculate(
            information
        )

        self.assertGreaterEqual(
            command.linear_velocity,
            0.0,
        )

        self.assertNotEqual(
            command.angular_velocity,
            0.0,
        )


# ============================================================
# PATH PLANNER
# ============================================================

class TestPathPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = PathPlanner()

        self.start_pose = RobotPose(
            x=1.0,
            y=1.0,
            heading=0.0,
        )

        self.goal = Waypoint(
            x=5.0,
            y=5.0,
        )

    def test_create_direct_path(self):
        path = self.planner.create_direct_path(
            self.start_pose,
            self.goal,
        )

        self.assertIsInstance(
            path,
            Path,
        )

        self.assertGreater(
            len(path.waypoints),
            0,
        )

    def test_direct_path_ends_at_goal(self):
        path = self.planner.create_direct_path(
            self.start_pose,
            self.goal,
        )

        self.assertEqual(
            path.waypoints[-1],
            self.goal,
        )

    def test_replan(self):
        path = self.planner.replan(
            self.start_pose,
            self.goal,
        )

        self.assertIsInstance(
            path,
            Path,
        )

        self.assertGreater(
            len(path.waypoints),
            0,
        )


# ============================================================
# NAVIGATION PLANNER
# ============================================================

class TestNavigationPlanner(unittest.TestCase):

    def setUp(self):
        self.geofence = Geofence(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
        )

        self.planner = NavigationPlanner(
            waypoint_tolerance=0.15,
            geofence=self.geofence,
        )

    def test_initial_state(self):
        self.assertEqual(
            self.planner.state_machine.get_state(),
            NavigationState.IDLE,
        )

    def test_set_waypoint(self):
        waypoint = Waypoint(
            x=5.0,
            y=5.0,
        )

        self.planner.set_waypoint(
            waypoint
        )

        self.assertEqual(
            self.planner.current_waypoint,
            waypoint,
        )

    def test_set_path(self):
        path = Path(
            waypoints=(
                Waypoint(2.0, 2.0),
                Waypoint(4.0, 4.0),
            )
        )

        self.planner.set_path(path)

        self.assertIs(
            self.planner.current_path,
            path,
        )

        self.assertEqual(
            self.planner.current_waypoint,
            path.waypoints[0],
        )

    def test_clear_waypoint(self):
        self.planner.set_waypoint(
            Waypoint(2.0, 2.0)
        )

        self.planner.clear_waypoint()

        self.assertIsNone(
            self.planner.current_waypoint
        )

    def test_clear_path(self):
        path = Path(
            waypoints=(
                Waypoint(2.0, 2.0),
            )
        )

        self.planner.set_path(path)
        self.planner.clear_path()

        self.assertIsNone(
            self.planner.current_path
        )

    def test_update_towards_waypoint(self):
        waypoint = Waypoint(
            x=5.0,
            y=2.0,
        )

        self.planner.set_waypoint(
            waypoint
        )

        pose = RobotPose(
            x=2.0,
            y=2.0,
            heading=0.0,
        )

        state, distance, heading_error, command = (
            self.planner.update(pose)
        )

        self.assertEqual(
            state,
            NavigationState.NAVIGATING,
        )

        self.assertAlmostEqual(
            distance,
            3.0,
        )

        self.assertIsNotNone(
            command
        )

    def test_goal_reached(self):
        waypoint = Waypoint(
            x=2.0,
            y=2.0,
        )

        self.planner.set_waypoint(
            waypoint
        )

        pose = RobotPose(
            x=2.0,
            y=2.0,
            heading=0.0,
        )

        state, distance, heading_error, command = (
            self.planner.update(pose)
        )

        self.assertEqual(
            state,
            NavigationState.GOAL_REACHED,
        )

        self.assertAlmostEqual(
            distance,
            0.0,
        )

    def test_geofence_violation(self):
        self.planner.set_waypoint(
            Waypoint(
                x=8.0,
                y=5.0,
            )
        )
    
        pose = RobotPose(
            x=11.0,
            y=5.0,
            heading=0.0,
        )
    
        state, _, _, _ = (
            self.planner.update(pose)
        )
    
        self.assertEqual(
            state,
            NavigationState.EMERGENCY_STOP,
        )


if __name__ == "__main__":
    unittest.main()