import unittest

from localization.position_estimator import RobotPose
from navigation.geofence import Geofence
from navigation.navigation_types import NavigationState, Waypoint
from navigation.planner import NavigationPlanner
from navigation.coverage_planner import (
    CoverageConfig,
    CoverageOrientation,
    CoveragePlanner,
)


class TestCoverageNavigationIntegration(unittest.TestCase):

    def setUp(self):
        self.geofence = Geofence(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=8.0,
        )

        self.coverage_config = CoverageConfig(
            cutting_width=1.0,
            lane_spacing=1.0,
            boundary_margin=0.5,
            turning_margin=0.5,
            waypoint_tolerance=0.1,
            orientation=CoverageOrientation.X,
        )

        self.coverage_planner = CoveragePlanner(
            geofence=self.geofence,
            config=self.coverage_config,
        )

        self.navigation_planner = NavigationPlanner(
            waypoint_tolerance=0.15,
            geofence=self.geofence,
        )

    def test_coverage_planner_generates_navigation_path(self):
        path = self.coverage_planner.generate_navigation_path()

        self.assertGreater(len(path.waypoints), 0)

        for waypoint in path.waypoints:
            self.assertIsInstance(waypoint, Waypoint)

    def test_generated_path_can_be_passed_to_navigation_planner(self):
        coverage_path = (
            self.coverage_planner.generate_navigation_path()
        )

        self.navigation_planner.set_path(coverage_path)

        self.assertIs(
            self.navigation_planner.current_path,
            coverage_path,
        )

    def test_navigation_planner_uses_first_coverage_waypoint(self):
        coverage_path = (
            self.coverage_planner.generate_navigation_path()
        )

        self.navigation_planner.set_path(coverage_path)

        self.assertEqual(
            self.navigation_planner.current_waypoint,
            coverage_path.waypoints[0],
        )

    def test_navigation_update_with_coverage_path(self):
        coverage_path = (
            self.coverage_planner.generate_navigation_path()
        )

        self.navigation_planner.set_path(coverage_path)

        first_waypoint = coverage_path.waypoints[0]

        pose = RobotPose(
            x=first_waypoint.x,
            y=first_waypoint.y,
            heading=0.0,
        )

        state, distance, heading_error, command = (
            self.navigation_planner.update(pose)
        )

        self.assertIsInstance(state, NavigationState)
        self.assertIsInstance(distance, float)
        self.assertIsInstance(heading_error, float)
        self.assertIsNotNone(command)

    def test_coverage_waypoints_inside_geofence(self):
        coverage_path = (
            self.coverage_planner.generate_navigation_path()
        )

        for waypoint in coverage_path.waypoints:
            self.assertTrue(
                self.geofence.contains_position(
                    waypoint.x,
                    waypoint.y,
                )
            )

    def test_coverage_path_has_multiple_waypoints(self):
        coverage_path = (
            self.coverage_planner.generate_navigation_path()
        )

        self.assertGreater(
            len(coverage_path.waypoints),
            1,
        )


class TestLocalizationNavigationIntegration(unittest.TestCase):

    def setUp(self):
        self.geofence = Geofence(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
        )

        self.navigation_planner = NavigationPlanner(
            waypoint_tolerance=0.15,
            geofence=self.geofence,
        )

    def test_localized_pose_can_be_used_by_navigation(self):
        pose = RobotPose(
            x=2.0,
            y=2.0,
            heading=0.0,
        )

        self.navigation_planner.set_waypoint(
            Waypoint(
                x=5.0,
                y=2.0,
            )
        )

        state, distance, heading_error, command = (
            self.navigation_planner.update(pose)
        )

        self.assertIsInstance(state, NavigationState)
        self.assertAlmostEqual(distance, 3.0)
        self.assertIsInstance(heading_error, float)
        self.assertIsNotNone(command)

    def test_navigation_respects_geofence(self):
        pose = RobotPose(
            x=5.0,
            y=5.0,
            heading=0.0,
        )

        self.navigation_planner.set_waypoint(
            Waypoint(
                x=20.0,
                y=5.0,
            )
        )

        state, _, _, _ = self.navigation_planner.update(pose)

        self.assertEqual(
            state,
            NavigationState.EMERGENCY_STOP,
        )


if __name__ == "__main__":
    unittest.main()