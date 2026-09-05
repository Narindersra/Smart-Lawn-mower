import pytest

from mission.mission_manager import MissionManager, MissionState
from navigation.coverage_planner import (
    CoverageConfig,
    CoverageOrientation,
    CoveragePlanner,
)
from navigation.geofence import Geofence
from navigation.navigation_types import NavigationState
from navigation.planner import NavigationPlanner


@pytest.fixture
def geofence():
    return Geofence(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
    )


@pytest.fixture
def coverage_planner(geofence):
    config = CoverageConfig(
        cutting_width=0.4,
        lane_spacing=0.4,
        boundary_margin=0.2,
        turning_margin=0.2,
        waypoint_tolerance=0.15,
        orientation=CoverageOrientation.X,
    )

    return CoveragePlanner(
        geofence=geofence,
        config=config,
    )


@pytest.fixture
def navigation_planner(geofence):
    return NavigationPlanner(
        waypoint_tolerance=0.15,
        geofence=geofence,
    )


@pytest.fixture
def mission_manager(
    coverage_planner,
    navigation_planner,
):
    return MissionManager(
        coverage_planner=coverage_planner,
        navigation_planner=navigation_planner,
    )


def test_system_components_are_connected(
    mission_manager,
    coverage_planner,
    navigation_planner,
):
    assert mission_manager.coverage_planner is coverage_planner
    assert mission_manager.navigation_planner is navigation_planner


def test_mission_starts_from_generated_coverage(
    mission_manager,
):
    path = mission_manager.prepare()

    assert path.waypoints
    assert mission_manager.state == MissionState.READY

    mission_manager.start()

    assert mission_manager.state == MissionState.RUNNING


def test_first_coverage_waypoint_reaches_navigation(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    assert (
        mission_manager.navigation_planner.current_waypoint
        == mission_manager.current_waypoint
    )


def test_goal_reached_advances_integrated_mission(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    initial_index = mission_manager.current_waypoint_index

    mission_manager.update(
        navigation_state=NavigationState.GOAL_REACHED,
        safety_stop=False,
    )

    assert (
        mission_manager.current_waypoint_index
        == initial_index + 1
    )

    assert (
        mission_manager.navigation_planner.current_waypoint
        == mission_manager.current_waypoint
    )


def test_safety_stop_blocks_integrated_mission_progress(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    initial_index = mission_manager.current_waypoint_index

    mission_manager.update(
        navigation_state=NavigationState.GOAL_REACHED,
        safety_stop=True,
    )

    assert (
        mission_manager.current_waypoint_index
        == initial_index
    )

    assert mission_manager.state == MissionState.RUNNING


def test_emergency_stop_blocks_integrated_mission_progress(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    initial_index = mission_manager.current_waypoint_index

    mission_manager.update(
        navigation_state=NavigationState.EMERGENCY_STOP,
        safety_stop=False,
    )

    assert (
        mission_manager.current_waypoint_index
        == initial_index
    )

    assert mission_manager.state == MissionState.RUNNING


def test_navigation_state_controls_mission_progress(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    initial_index = mission_manager.current_waypoint_index

    mission_manager.update(
        navigation_state=NavigationState.NAVIGATING,
        safety_stop=False,
    )

    assert (
        mission_manager.current_waypoint_index
        == initial_index
    )

    assert mission_manager.state == MissionState.RUNNING


def test_final_waypoint_completes_mission(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    while mission_manager.state == MissionState.RUNNING:
        mission_manager.update(
            navigation_state=NavigationState.GOAL_REACHED,
            safety_stop=False,
        )

    assert mission_manager.state == MissionState.COMPLETED


def test_mission_completion_clears_navigation_target(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    while mission_manager.state == MissionState.RUNNING:
        mission_manager.update(
            navigation_state=NavigationState.GOAL_REACHED,
            safety_stop=False,
        )

    assert mission_manager.state == MissionState.COMPLETED

    assert (
        mission_manager.navigation_planner.current_waypoint
        is None
    )