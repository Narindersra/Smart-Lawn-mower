import unittest

from navigation.coverage_planner import (
    CoverageBoundary,
    CoverageConfig,
    CoverageLane,
    CoverageOrientation,
    CoveragePlanner,
)
from navigation.geofence import Geofence


class TestCoverageBoundary(unittest.TestCase):

    def setUp(self):
        self.boundary = CoverageBoundary(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=8.0,
        )

    def test_width(self):
        self.assertEqual(self.boundary.width, 10.0)

    def test_height(self):
        self.assertEqual(self.boundary.height, 8.0)

    def test_contains_inside(self):
        self.assertTrue(self.boundary.contains(5.0, 4.0))

    def test_contains_outside(self):
        self.assertFalse(self.boundary.contains(11.0, 4.0))


class TestCoverageLane(unittest.TestCase):

    def setUp(self):
        self.lane = CoverageLane(
            start_x=1.0,
            start_y=2.0,
            end_x=6.0,
            end_y=2.0,
        )

    def test_lane_length(self):
        self.assertAlmostEqual(self.lane.length(), 5.0)

    def test_lane_reverse(self):
        reversed_lane = self.lane.reversed()

        self.assertEqual(reversed_lane.start_x, self.lane.end_x)
        self.assertEqual(reversed_lane.start_y, self.lane.end_y)
        self.assertEqual(reversed_lane.end_x, self.lane.start_x)
        self.assertEqual(reversed_lane.end_y, self.lane.start_y)

    def test_lane_start(self):
        self.assertEqual(self.lane.start, (1.0, 2.0))

    def test_lane_end(self):
        self.assertEqual(self.lane.end, (6.0, 2.0))


class TestCoverageConfig(unittest.TestCase):

    def test_default_orientation(self):
        config = CoverageConfig(
            cutting_width=0.5,
            lane_spacing=0.4,
            boundary_margin=0.2,
            turning_margin=0.3,
            waypoint_tolerance=0.1,
        )

        self.assertEqual(config.orientation, CoverageOrientation.X)

    def test_y_orientation(self):
        config = CoverageConfig(
            cutting_width=0.5,
            lane_spacing=0.4,
            boundary_margin=0.2,
            turning_margin=0.3,
            waypoint_tolerance=0.1,
            orientation=CoverageOrientation.Y,
        )

        self.assertEqual(config.orientation, CoverageOrientation.Y)

    def test_invalid_lane_spacing(self):
        with self.assertRaises(ValueError):
            CoverageConfig(
                cutting_width=0.5,
                lane_spacing=0.6,
                boundary_margin=0.2,
                turning_margin=0.3,
                waypoint_tolerance=0.1,
            )


class TestCoveragePlanner(unittest.TestCase):

    def setUp(self):
        self.geofence = Geofence(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=8.0,
        )

        self.config = CoverageConfig(
            cutting_width=1.0,
            lane_spacing=1.0,
            boundary_margin=0.5,
            turning_margin=0.5,
            waypoint_tolerance=0.1,
            orientation=CoverageOrientation.X,
        )

        self.planner = CoveragePlanner(
            geofence=self.geofence,
            config=self.config,
        )

    def test_boundary(self):
        self.assertEqual(self.planner.boundary.min_x, 0.0)
        self.assertEqual(self.planner.boundary.max_x, 10.0)
        self.assertEqual(self.planner.boundary.min_y, 0.0)
        self.assertEqual(self.planner.boundary.max_y, 8.0)

    def test_usable_area(self):
        area = self.planner.get_usable_area()

        self.assertGreater(area, 0.0)

    def test_contains_position(self):
        self.assertTrue(self.planner.contains_position(5.0, 4.0))
        self.assertFalse(self.planner.contains_position(11.0, 4.0))

    def test_generate_lanes(self):
        lanes = self.planner.generate_lanes()

        self.assertGreater(len(lanes), 0)

        for lane in lanes:
            self.assertTrue(self.planner._is_lane_inside_usable_area(lane))

    def test_generate_ordered_lanes(self):
        lanes = self.planner.generate_ordered_lanes()

        self.assertGreater(len(lanes), 0)

        for lane in lanes:
            self.assertTrue(
                self.planner._is_lane_inside_usable_area(lane)
            )

        def test_ordered_lanes_have_alternating_direction(self):
            lanes = self.planner.generate_ordered_lanes()
    
            self.assertGreaterEqual(len(lanes), 2)
    
            for index, lane in enumerate(lanes):
                if index % 2 == 0:
                    self.assertLess(lane.start_x, lane.end_x)
                else:
                    self.assertGreater(lane.start_x, lane.end_x)

    def test_lane_endpoints(self):
        lanes = self.planner.generate_ordered_lanes()

        endpoints = self.planner.get_lane_endpoints(lanes)

        self.assertEqual(len(endpoints), len(lanes) * 2)

    def test_generate_turn(self):
        lanes = self.planner.generate_ordered_lanes()

        self.assertGreaterEqual(len(lanes), 2)

        turn = self.planner.generate_turn(
            lanes[0],
            lanes[1],
        )

        self.assertGreater(len(turn.points), 0)

    def test_generate_coverage_path(self):
        path = self.planner.generate_coverage_path()

        self.assertGreater(len(path), 0)

        for point in path:
            self.assertEqual(len(point), 2)

    def test_generate_navigation_path(self):
        path = self.planner.generate_navigation_path()

        self.assertGreater(len(path.waypoints), 0)

    def test_start_coverage(self):
        path = self.planner.start_coverage()

        self.assertGreater(len(path.waypoints), 0)
        self.assertNotEqual(
            self.planner.state.name,
            "IDLE",
        )

    def test_current_waypoint(self):
        self.planner.start_coverage()

        waypoint = self.planner.get_current_waypoint()

        self.assertIsNotNone(waypoint)

    def test_advance_waypoint(self):
        self.planner.start_coverage()

        initial_index = self.planner.current_waypoint_index

        self.planner.advance_waypoint()

        self.assertEqual(
            self.planner.current_waypoint_index,
            initial_index + 1,
        )

    def test_progress(self):
        self.planner.start_coverage()

        progress = self.planner.progress

        self.assertGreaterEqual(progress, 0.0)
        self.assertLessEqual(progress, 100.0)

    def test_pause_resume(self):
        self.planner.start_coverage()

        self.planner.pause_coverage()

        self.assertEqual(
            self.planner.state.name,
            "PAUSED",
        )

        self.planner.resume_coverage()

        self.assertEqual(
            self.planner.state.name,
            "COVERING",
        )

    def test_stop_coverage(self):
        self.planner.start_coverage()

        self.planner.stop_coverage()

        self.assertEqual(
            self.planner.state.name,
            "STOPPED",
        )

    def test_coverage_metrics(self):
        self.planner.start_coverage()

        planned_area = self.planner.get_planned_coverage_area()
        coverage_percentage = (
            self.planner.get_planned_coverage_percentage()
        )

        self.assertGreaterEqual(planned_area, 0.0)
        self.assertGreaterEqual(coverage_percentage, 0.0)

    def test_overlap_metrics(self):
        self.planner.start_coverage()

        overlap = self.planner.get_lane_overlap()
        overlap_percentage = self.planner.get_overlap_percentage()

        self.assertGreaterEqual(overlap, 0.0)
        self.assertGreaterEqual(overlap_percentage, 0.0)

    def test_missed_area_metrics(self):
        self.planner.start_coverage()

        missed_area = self.planner.get_missed_area()
        missed_percentage = (
            self.planner.get_missed_area_percentage()
        )

        self.assertGreaterEqual(missed_area, 0.0)
        self.assertGreaterEqual(missed_percentage, 0.0)

    def test_validate_planned_coverage(self):
        self.planner.start_coverage()

        result = self.planner.validate_planned_coverage()

        self.assertIsInstance(result, bool)

    def test_validate_overlap(self):
        self.planner.start_coverage()

        result = self.planner.validate_overlap()

        self.assertIsInstance(result, bool)

    def test_validate_missed_area(self):
        self.planner.start_coverage()

        result = self.planner.validate_missed_area()

        self.assertIsInstance(result, bool)


class TestCoveragePlannerYOrientation(unittest.TestCase):

    def setUp(self):
        geofence = Geofence(
            min_x=0.0,
            max_x=8.0,
            min_y=0.0,
            max_y=10.0,
        )

        config = CoverageConfig(
            cutting_width=1.0,
            lane_spacing=1.0,
            boundary_margin=0.5,
            turning_margin=0.5,
            waypoint_tolerance=0.1,
            orientation=CoverageOrientation.Y,
        )

        self.planner = CoveragePlanner(
            geofence=geofence,
            config=config,
        )

    def test_generate_y_orientation_lanes(self):
        lanes = self.planner.generate_ordered_lanes()

        self.assertGreater(len(lanes), 0)

        for lane in lanes:
            self.assertAlmostEqual(
                lane.start_x,
                lane.end_x,
            )


if __name__ == "__main__":
    unittest.main()