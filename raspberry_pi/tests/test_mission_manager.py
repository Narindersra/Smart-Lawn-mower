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


def test_initial_state_is_idle(mission_manager):
    assert mission_manager.state == MissionState.IDLE


def test_constructor_rejects_missing_coverage_planner(
    navigation_planner,
):
    with pytest.raises(ValueError):
        MissionManager(
            coverage_planner=None,
            navigation_planner=navigation_planner,
        )


def test_constructor_rejects_missing_navigation_planner(
    coverage_planner,
):
    with pytest.raises(ValueError):
        MissionManager(
            coverage_planner=coverage_planner,
            navigation_planner=None,
        )


def test_prepare_moves_mission_to_ready(mission_manager):
    path = mission_manager.prepare()

    assert path.waypoints
    assert mission_manager.state == MissionState.READY
    assert mission_manager.current_waypoint is not None


def test_start_requires_ready_state(mission_manager):
    with pytest.raises(RuntimeError):
        mission_manager.start()


def test_start_moves_mission_to_running(mission_manager):
    mission_manager.prepare()
    mission_manager.start()

    assert mission_manager.state == MissionState.RUNNING


def test_start_assigns_current_waypoint_to_navigation(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    assert (
        mission_manager.navigation_planner.current_waypoint
        == mission_manager.current_waypoint
    )


def test_goal_reached_advances_coverage_waypoint(
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


def test_goal_reached_assigns_next_waypoint_to_navigation(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    mission_manager.update(
        navigation_state=NavigationState.GOAL_REACHED,
        safety_stop=False,
    )

    assert (
        mission_manager.navigation_planner.current_waypoint
        == mission_manager.current_waypoint
    )


def test_safety_stop_does_not_advance_waypoint(
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


def test_emergency_stop_does_not_advance_waypoint(
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


def test_non_goal_navigation_state_does_not_advance_waypoint(
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


def test_pause_moves_mission_to_paused(mission_manager):
    mission_manager.prepare()
    mission_manager.start()

    mission_manager.pause()

    assert mission_manager.state == MissionState.PAUSED
    assert (
        mission_manager.navigation_planner.current_waypoint
        is None
    )


def test_pause_preserves_current_waypoint(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    current_waypoint = mission_manager.current_waypoint
    current_index = mission_manager.current_waypoint_index

    mission_manager.pause()

    assert (
        mission_manager.current_waypoint
        == current_waypoint
    )

    assert (
        mission_manager.current_waypoint_index
        == current_index
    )


def test_resume_moves_mission_back_to_running(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()
    mission_manager.pause()

    mission_manager.resume()

    assert mission_manager.state == MissionState.RUNNING


def test_resume_restores_current_waypoint(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    current_waypoint = mission_manager.current_waypoint

    mission_manager.pause()
    mission_manager.resume()

    assert (
        mission_manager.navigation_planner.current_waypoint
        == current_waypoint
    )


def test_stop_moves_running_mission_to_stopped(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()

    mission_manager.stop()

    assert mission_manager.state == MissionState.STOPPED
    assert (
        mission_manager.navigation_planner.current_waypoint
        is None
    )


def test_stop_from_paused_moves_to_stopped(
    mission_manager,
):
    mission_manager.prepare()
    mission_manager.start()
    mission_manager.pause()

    mission_manager.stop()

    assert mission_manager.state == MissionState.STOPPED


def test_update_does_nothing_when_not_running(
    mission_manager,
):
    initial_index = mission_manager.current_waypoint_index

    mission_manager.update(
        navigation_state=NavigationState.GOAL_REACHED,
        safety_stop=False,
    )

    assert (
        mission_manager.current_waypoint_index
        == initial_index
    )
    assert mission_manager.state == MissionState.IDLE


def test_progress_is_provided_by_coverage_planner(
    mission_manager,
):
    mission_manager.prepare()

    assert (
        mission_manager.progress
        == mission_manager.coverage_planner.progress
    )


def test_total_waypoints_matches_coverage_planner(
    mission_manager,
):
    mission_manager.prepare()

    assert (
        mission_manager.total_waypoints
        == mission_manager.coverage_planner.total_waypoints
    )