from enum import Enum

from navigation.coverage_planner import CoveragePlanner
from navigation.navigation_types import NavigationState, Path, Waypoint
from navigation.planner import NavigationPlanner


class MissionState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class MissionManager:
    """
    Coordinates the mission-level lifecycle.

    CoveragePlanner owns:
        - coverage path generation
        - coverage waypoint progression
        - coverage progress

    NavigationPlanner owns:
        - navigation toward the current waypoint
        - obstacle handling
        - navigation state

    MissionManager coordinates both subsystems.
    """

    def __init__(
        self,
        coverage_planner: CoveragePlanner,
        navigation_planner: NavigationPlanner,
    ) -> None:
        if coverage_planner is None:
            raise ValueError(
                "MissionManager requires a valid CoveragePlanner."
            )

        if navigation_planner is None:
            raise ValueError(
                "MissionManager requires a valid NavigationPlanner."
            )

        self.coverage_planner = coverage_planner
        self.navigation_planner = navigation_planner

        self._state = MissionState.IDLE

    # --------------------------------------------------
    # Mission State
    # --------------------------------------------------

    @property
    def state(self) -> MissionState:
        """Return the current mission state."""
        return self._state

    # --------------------------------------------------
    # Coverage Information
    # --------------------------------------------------

    @property
    def coverage_path(self) -> Path | None:
        """Return the generated coverage path."""
        return self.coverage_planner.coverage_path

    @property
    def current_waypoint(self) -> Waypoint | None:
        """Return the current coverage waypoint."""
        return self.coverage_planner.get_current_waypoint()

    @property
    def current_waypoint_index(self) -> int:
        """Return the current coverage waypoint index."""
        return self.coverage_planner.current_waypoint_index

    @property
    def total_waypoints(self) -> int:
        """Return the total number of coverage waypoints."""
        return self.coverage_planner.total_waypoints

    @property
    def progress(self) -> float:
        """Return coverage progress as a percentage."""
        return self.coverage_planner.progress

    # --------------------------------------------------
    # Mission Preparation
    # --------------------------------------------------

    def prepare(self) -> Path:
        """
        Generate and prepare the complete coverage mission.
        """
        if self._state not in (
            MissionState.IDLE,
            MissionState.STOPPED,
        ):
            raise RuntimeError(
                "Mission can only be prepared from IDLE or STOPPED."
            )

        path = self.coverage_planner.start_coverage()

        self.navigation_planner.clear_waypoint()

        if not path.waypoints:
            self._state = MissionState.COMPLETED
            return path

        self._state = MissionState.READY

        return path

    # --------------------------------------------------
    # Mission Start
    # --------------------------------------------------

    def start(self) -> None:
        """
        Start the prepared mission.
        """
        if self._state != MissionState.READY:
            raise RuntimeError(
                "Mission can only be started from READY."
            )

        current_waypoint = (
            self.coverage_planner.get_current_waypoint()
        )

        if current_waypoint is None:
            self._state = MissionState.COMPLETED
            return

        self.navigation_planner.set_waypoint(
            current_waypoint
        )

        self._state = MissionState.RUNNING

    # --------------------------------------------------
    # Mission Pause
    # --------------------------------------------------

    def pause(self) -> None:
        """
        Pause the current mission while preserving
        the current coverage waypoint.
        """
        if self._state != MissionState.RUNNING:
            return

        self.coverage_planner.pause_coverage()
        self.navigation_planner.clear_waypoint()

        self._state = MissionState.PAUSED

    # --------------------------------------------------
    # Mission Resume
    # --------------------------------------------------

    def resume(self) -> None:
        """
        Resume the mission from the current coverage waypoint.
        """
        if self._state != MissionState.PAUSED:
            return

        self.coverage_planner.resume_coverage()

        current_waypoint = (
            self.coverage_planner.get_current_waypoint()
        )

        if current_waypoint is None:
            self._state = MissionState.COMPLETED
            return

        self.navigation_planner.set_waypoint(
            current_waypoint
        )

        self._state = MissionState.RUNNING

    # --------------------------------------------------
    # Mission Stop
    # --------------------------------------------------

    def stop(self) -> None:
        """
        Stop the current mission.
        """
        if self._state not in (
            MissionState.RUNNING,
            MissionState.PAUSED,
        ):
            return

        self.coverage_planner.stop_coverage()
        self.navigation_planner.clear_waypoint()

        self._state = MissionState.STOPPED

    # --------------------------------------------------
    # Mission Update
    # --------------------------------------------------

    def update(
        self,
        navigation_state: NavigationState,
        safety_stop: bool = False,
    ) -> MissionState:
        """
        Synchronize mission progress with navigation progress.

        A coverage waypoint is advanced only when:
            1. Mission is RUNNING
            2. No safety stop is active
            3. Navigation reports GOAL_REACHED
        """

        if self._state != MissionState.RUNNING:
            return self._state

        # --------------------------------------------------
        # Safety Protection
        # --------------------------------------------------

        if safety_stop:
            return self._state

        # --------------------------------------------------
        # Emergency Navigation State
        # --------------------------------------------------

        if navigation_state == NavigationState.EMERGENCY_STOP:
            return self._state

        # --------------------------------------------------
        # Wait Until Current Waypoint Is Reached
        # --------------------------------------------------

        if navigation_state != NavigationState.GOAL_REACHED:
            return self._state

        # --------------------------------------------------
        # Advance Coverage Waypoint
        # --------------------------------------------------

        next_waypoint = (
            self.coverage_planner.advance_waypoint()
        )

        # --------------------------------------------------
        # Coverage Complete
        # --------------------------------------------------

        if next_waypoint is None:
            self.navigation_planner.clear_waypoint()
            self._state = MissionState.COMPLETED

            return self._state

        # --------------------------------------------------
        # Navigate To Next Coverage Waypoint
        # --------------------------------------------------

        self.navigation_planner.set_waypoint(
            next_waypoint
        )

        return self._state