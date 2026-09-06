from dataclasses import dataclass
from enum import Enum

from .geofence import Geofence
from .navigation_types import Path, Waypoint


class CoverageOrientation(str, Enum):
    """
    Defines the direction in which coverage lanes are generated.
    """

    X = "x"
    Y = "y"


class CoverageState(str, Enum):
    """
    Represents the current state of the coverage-planning process.
    """

    IDLE = "idle"
    COVERING = "covering"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


@dataclass(frozen=True)
class CoverageBoundary:
    """
    Represents the rectangular lawn boundary.

    Coordinates use the same local coordinate system as the
    navigation Geofence.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        """Validate the boundary dimensions."""
        if self.min_x >= self.max_x:
            raise ValueError(
                "min_x must be smaller than max_x."
            )

        if self.min_y >= self.max_y:
            raise ValueError(
                "min_y must be smaller than max_y."
            )

    @property
    def width(self) -> float:
        """Return the total boundary width in meters."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Return the total boundary height in meters."""
        return self.max_y - self.min_y

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Check whether a position lies inside the boundary.

        Boundary edges are considered valid positions.
        """
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
        )


@dataclass(frozen=True)
class CoverageLane:
    """
    Represents one straight mowing lane.

    Lane endpoints use the same coordinate system as navigation
    Waypoints.
    """

    start_x: float
    start_y: float
    end_x: float
    end_y: float

    def length(self) -> float:
        """
        Calculate the geometric length of the lane in meters.
        """
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y

        return (dx * dx + dy * dy) ** 0.5

    def reversed(self) -> "CoverageLane":
        """
        Return a copy of the lane with start and end points exchanged.
        """
        return CoverageLane(
            start_x=self.end_x,
            start_y=self.end_y,
            end_x=self.start_x,
            end_y=self.start_y,
        )

    @property
    def start(self) -> tuple[float, float]:
        """Return the lane start point as an (x, y) tuple."""
        return self.start_x, self.start_y

    @property
    def end(self) -> tuple[float, float]:
        """Return the lane end point as an (x, y) tuple."""
        return self.end_x, self.end_y


@dataclass(frozen=True)
class CoverageTurn:
    """
    Represents the connection between two consecutive
    coverage lanes.

    The turn contains an ordered sequence of points from the
    end of the current lane to the start of the next lane.
    """

    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CoverageConfig:
    """
    Stores configuration parameters used by the coverage planner.

    All distance-related values are expressed in meters.
    """

    cutting_width: float
    lane_spacing: float
    boundary_margin: float
    turning_margin: float
    waypoint_tolerance: float
    orientation: CoverageOrientation = CoverageOrientation.X

    def __post_init__(self) -> None:
        """Validate coverage configuration parameters."""
        if self.cutting_width <= 0.0:
            raise ValueError(
                "cutting_width must be greater than zero."
            )

        if self.lane_spacing <= 0.0:
            raise ValueError(
                "lane_spacing must be greater than zero."
            )

        if self.lane_spacing > self.cutting_width:
            raise ValueError(
                "lane_spacing must not be greater than "
                "cutting_width."
            )

        if self.boundary_margin < 0.0:
            raise ValueError(
                "boundary_margin cannot be negative."
            )

        if self.turning_margin < 0.0:
            raise ValueError(
                "turning_margin cannot be negative."
            )

        if self.waypoint_tolerance <= 0.0:
            raise ValueError(
                "waypoint_tolerance must be greater than zero."
            )

        if not isinstance(
            self.orientation,
            CoverageOrientation,
        ):
            raise ValueError(
                "orientation must be a CoverageOrientation."
            )


class CoveragePlanner:
    """
    Generates and manages the lawn-mowing coverage path.

    Responsibilities include:

        - Representing the existing Geofence boundary.
        - Applying the configured usable-area margin.
        - Generating parallel mowing lanes.
        - Ordering lanes in a boustrophedon pattern.
        - Generating boundary-safe turning paths.
        - Generating the complete coverage path.
        - Converting coverage points into navigation Path/Waypoint types.
        - Managing sequential coverage waypoint execution.
        - Tracking coverage state and progress.
        - Estimating planned coverage, overlap, and missed area.
        - Handling invalid or empty coverage areas safely.
    """

    def __init__(
        self,
        geofence: Geofence,
        config: CoverageConfig,
    ) -> None:
        """
        Initialize the coverage planner.

        Args:
            geofence: Existing navigation geofence.
            config: Coverage planning configuration.

        Raises:
            ValueError: If the geofence/config is missing or the
                configured boundary margin leaves no usable area.
        """
        if geofence is None:
            raise ValueError(
                "CoveragePlanner requires a valid Geofence."
            )

        if config is None:
            raise ValueError(
                "CoveragePlanner requires a valid CoverageConfig."
            )

        # Convert the navigation geofence into the coverage boundary.
        self._boundary = CoverageBoundary(
            min_x=geofence.min_x,
            max_x=geofence.max_x,
            min_y=geofence.min_y,
            max_y=geofence.max_y,
        )

        self._config = config

        # Coverage execution state.
        self._current_waypoint_index = 0
        self._coverage_path: Path | None = None
        self._state = CoverageState.IDLE

        # Ensure the configured boundary margin leaves a usable lawn.
        if self.usable_width <= 0.0:
            raise ValueError(
                "boundary_margin is too large for the lawn width."
            )

        if self.usable_height <= 0.0:
            raise ValueError(
                "boundary_margin is too large for the lawn height."
            )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def boundary(self) -> CoverageBoundary:
        """Return the original lawn boundary."""
        return self._boundary

    @property
    def config(self) -> CoverageConfig:
        """Return the active coverage configuration."""
        return self._config

    @property
    def coverage_path(self) -> Path | None:
        """Return the currently generated navigation coverage path."""
        return self._coverage_path

    @property
    def orientation(self) -> CoverageOrientation:
        """Return the configured coverage orientation."""
        return self._config.orientation

    @property
    def state(self) -> CoverageState:
        """Return the current coverage state."""
        return self._state

    @property
    def width(self) -> float:
        """Return the total lawn width in meters."""
        return self._boundary.width

    @property
    def height(self) -> float:
        """Return the total lawn height in meters."""
        return self._boundary.height

    # ------------------------------------------------------------------
    # Usable lawn area
    # ------------------------------------------------------------------

    @property
    def usable_min_x(self) -> float:
        """Return the minimum usable X coordinate."""
        return (
            self._boundary.min_x
            + self._config.boundary_margin
        )

    @property
    def usable_max_x(self) -> float:
        """Return the maximum usable X coordinate."""
        return (
            self._boundary.max_x
            - self._config.boundary_margin
        )

    @property
    def usable_min_y(self) -> float:
        """Return the minimum usable Y coordinate."""
        return (
            self._boundary.min_y
            + self._config.boundary_margin
        )

    @property
    def usable_max_y(self) -> float:
        """Return the maximum usable Y coordinate."""
        return (
            self._boundary.max_y
            - self._config.boundary_margin
        )

    @property
    def usable_width(self) -> float:
        """Return the usable lawn width in meters."""
        return self.usable_max_x - self.usable_min_x

    @property
    def usable_height(self) -> float:
        """Return the usable lawn height in meters."""
        return self.usable_max_y - self.usable_min_y

    def contains_position(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Check whether a position lies inside the original lawn boundary.

        This check does not apply the coverage boundary margin.
        """
        return self._boundary.contains(x, y)

    def is_inside_usable_area(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Check whether a position lies inside the usable mowing area.

        The configured boundary margin is applied to the original
        geofence before this check is performed.
        """
        return (
            self.usable_min_x <= x <= self.usable_max_x
            and self.usable_min_y <= y <= self.usable_max_y
        )

    # ------------------------------------------------------------------
    # Lane generation
    # ------------------------------------------------------------------

    def generate_lanes(self) -> tuple[CoverageLane, ...]:
        """
        Generate parallel mowing lanes inside the usable boundary.

        Lanes are generated according to the configured orientation
        and lane spacing.
        """
        lanes: list[CoverageLane] = []

        if (
            self.usable_width <= 0.0
            or self.usable_height <= 0.0
        ):
            return tuple()

        if self.orientation == CoverageOrientation.X:
            coordinate = self.usable_min_y
            maximum = self.usable_max_y

            while coordinate <= maximum:
                lane = CoverageLane(
                    start_x=self.usable_min_x,
                    start_y=coordinate,
                    end_x=self.usable_max_x,
                    end_y=coordinate,
                )

                if self._is_lane_inside_usable_area(lane):
                    lanes.append(lane)

                coordinate += self.config.lane_spacing

        else:
            coordinate = self.usable_min_x
            maximum = self.usable_max_x

            while coordinate <= maximum:
                lane = CoverageLane(
                    start_x=coordinate,
                    start_y=self.usable_min_y,
                    end_x=coordinate,
                    end_y=self.usable_max_y,
                )

                if self._is_lane_inside_usable_area(lane):
                    lanes.append(lane)

                coordinate += self.config.lane_spacing

        # Ignore zero-length lanes.
        return tuple(
            lane
            for lane in lanes
            if lane.length() > 0.0
        )

    def _is_lane_inside_usable_area(
        self,
        lane: CoverageLane,
    ) -> bool:
        """
        Check whether both lane endpoints are inside the usable area.

        For the current rectangular, axis-aligned lanes, checking both
        endpoints is sufficient to validate the complete lane.
        """
        return (
            self.is_inside_usable_area(
                lane.start_x,
                lane.start_y,
            )
            and self.is_inside_usable_area(
                lane.end_x,
                lane.end_y,
            )
        )

    # ------------------------------------------------------------------
    # Lane ordering
    # ------------------------------------------------------------------

    def generate_ordered_lanes(
        self,
    ) -> tuple[CoverageLane, ...]:
        """
        Generate coverage lanes in boustrophedon order.

        Consecutive lanes are traversed in opposite directions to
        reduce unnecessary repositioning between lanes.
        """
        lanes = self.generate_lanes()
        ordered_lanes: list[CoverageLane] = []

        for index, lane in enumerate(lanes):
            if index % 2 == 0:
                ordered_lanes.append(lane)
            else:
                ordered_lanes.append(lane.reversed())

        return tuple(ordered_lanes)

    def get_lane_endpoints(
        self,
        lanes: tuple[CoverageLane, ...],
    ) -> tuple[tuple[float, float], ...]:
        """
        Return lane endpoints in their traversal order.

        Each lane contributes its start point followed by its end point.
        """
        endpoints: list[tuple[float, float]] = []

        for lane in lanes:
            endpoints.append(lane.start)
            endpoints.append(lane.end)

        return tuple(endpoints)

    # ------------------------------------------------------------------
    # Turning
    # ------------------------------------------------------------------

    def generate_turn(
        self,
        current_lane: CoverageLane,
        next_lane: CoverageLane,
    ) -> CoverageTurn:
        """
        Generate a boundary-safe U-turn between two consecutive lanes.

        The generated turn moves inward from the lawn edge, changes
        lane level, and then approaches the start of the next lane.
        """
        if self.orientation == CoverageOrientation.X:
            if current_lane.end_x >= self.usable_max_x:
                turn_x = (
                    self.usable_max_x
                    - self.config.turning_margin
                )
            else:
                turn_x = (
                    self.usable_min_x
                    + self.config.turning_margin
                )

            # Keep the turn point inside the usable X range.
            turn_x = max(
                self.usable_min_x,
                min(self.usable_max_x, turn_x),
            )

            turn = CoverageTurn(
                points=(
                    current_lane.end,
                    (
                        turn_x,
                        current_lane.end_y,
                    ),
                    (
                        turn_x,
                        next_lane.start_y,
                    ),
                    next_lane.start,
                )
            )

        else:
            if current_lane.end_y >= self.usable_max_y:
                turn_y = (
                    self.usable_max_y
                    - self.config.turning_margin
                )
            else:
                turn_y = (
                    self.usable_min_y
                    + self.config.turning_margin
                )

            # Keep the turn point inside the usable Y range.
            turn_y = max(
                self.usable_min_y,
                min(self.usable_max_y, turn_y),
            )

            turn = CoverageTurn(
                points=(
                    current_lane.end,
                    (
                        current_lane.end_x,
                        turn_y,
                    ),
                    (
                        next_lane.start_x,
                        turn_y,
                    ),
                    next_lane.start,
                )
            )

        self._validate_turn(turn)

        return turn

    def _validate_turn(
        self,
        turn: CoverageTurn,
    ) -> None:
        """
        Validate that every turn point remains inside the usable area.

        Raises:
            ValueError: If a generated turn contains a point outside
                the usable mowing area.
        """
        for x, y in turn.points:
            if not self.is_inside_usable_area(x, y):
                raise ValueError(
                    "Generated coverage turn contains a point "
                    "outside the usable lawn area."
                )

    # ------------------------------------------------------------------
    # Complete coverage path
    # ------------------------------------------------------------------

    def generate_coverage_path(
        self,
    ) -> tuple[tuple[float, float], ...]:
        """
        Generate the complete ordered coverage path.

        The generated sequence contains lane endpoints and the
        intermediate turning points required to connect consecutive
        lanes.
        """
        lanes = self.generate_ordered_lanes()

        if not lanes:
            return tuple()

        path_points: list[tuple[float, float]] = []

        for index, lane in enumerate(lanes):
            # Add the first lane's starting point only once.
            if index == 0:
                path_points.append(lane.start)

            path_points.append(lane.end)

            # Connect each lane to the following lane.
            if index < len(lanes) - 1:
                next_lane = lanes[index + 1]

                turn = self.generate_turn(
                    current_lane=lane,
                    next_lane=next_lane,
                )

                # The first turn point is already the current lane end.
                path_points.extend(turn.points[1:])

        return tuple(path_points)

    def generate_navigation_path(self) -> Path:
        """
        Convert coverage coordinates into navigation Path/Waypoint types.
        """
        coverage_points = self.generate_coverage_path()

        waypoints = tuple(
            Waypoint(x=x, y=y)
            for x, y in coverage_points
        )

        return Path(waypoints=waypoints)

    # ------------------------------------------------------------------
    # Coverage execution
    # ------------------------------------------------------------------

    def start_coverage(self) -> Path:
        """
        Generate and initialize the complete coverage path.

        The first generated waypoint becomes the current coverage
        waypoint. An empty path immediately moves the planner to
        COMPLETED.
        """
        self._coverage_path = self.generate_navigation_path()
        self._current_waypoint_index = 0

        if not self._coverage_path.waypoints:
            self._state = CoverageState.COMPLETED
            return self._coverage_path

        self._state = CoverageState.COVERING

        return self._coverage_path

    def get_current_waypoint(self) -> Waypoint | None:
        """
        Return the currently active coverage waypoint.

        Returns:
            Current waypoint, or None if no active waypoint exists.
        """
        if self._coverage_path is None:
            return None

        if self._current_waypoint_index >= len(
            self._coverage_path.waypoints
        ):
            return None

        return self._coverage_path.waypoints[
            self._current_waypoint_index
        ]

    def advance_waypoint(self) -> Waypoint | None:
        """
        Advance the coverage planner to the next waypoint.

        Returns:
            The new current waypoint, or None when coverage is complete
            or no coverage path is available.
        """
        if self._coverage_path is None:
            return None

        if self._state != CoverageState.COVERING:
            return self.get_current_waypoint()

        if self._current_waypoint_index >= len(
            self._coverage_path.waypoints
        ):
            self._state = CoverageState.COMPLETED
            return None

        self._current_waypoint_index += 1

        if self.is_coverage_complete():
            self._state = CoverageState.COMPLETED
            return None

        return self.get_current_waypoint()

    def pause_coverage(self) -> None:
        """Pause an active coverage operation."""
        if self._state == CoverageState.COVERING:
            self._state = CoverageState.PAUSED

    def resume_coverage(self) -> None:
        """Resume a paused coverage operation."""
        if self._state == CoverageState.PAUSED:
            self._state = CoverageState.COVERING

    def stop_coverage(self) -> None:
        """Stop an active or paused coverage operation."""
        if self._state in (
            CoverageState.COVERING,
            CoverageState.PAUSED,
        ):
            self._state = CoverageState.STOPPED

    def is_coverage_complete(self) -> bool:
        """
        Return whether all coverage waypoints have been executed.

        Returns:
            True when the coverage path exists and the waypoint index
            has reached the end of the path.
        """
        if self._coverage_path is None:
            return False

        return (
            self._current_waypoint_index
            >= len(self._coverage_path.waypoints)
        )

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    @property
    def current_waypoint_index(self) -> int:
        """Return the zero-based current waypoint index."""
        return self._current_waypoint_index

    @property
    def total_waypoints(self) -> int:
        """Return the total number of generated navigation waypoints."""
        if self._coverage_path is None:
            return 0

        return len(self._coverage_path.waypoints)

    @property
    def remaining_waypoints(self) -> int:
        """Return the number of coverage waypoints remaining."""
        return max(
            self.total_waypoints
            - self._current_waypoint_index,
            0,
        )

    @property
    def progress(self) -> float:
        """
        Return coverage execution progress as a percentage.

        Returns:
            0.0 before a path is generated and 100.0 after all
            waypoints have been completed.
        """
        total_waypoints = self.total_waypoints

        if total_waypoints == 0:
            return 0.0

        completed_waypoints = min(
            self._current_waypoint_index,
            total_waypoints,
        )

        return (
            completed_waypoints / total_waypoints
        ) * 100.0

    # ------------------------------------------------------------------
    # Coverage validation and estimation
    # ------------------------------------------------------------------

    def get_usable_area(self) -> float:
        """Return the usable lawn area in square meters."""
        return self.usable_width * self.usable_height

    def get_planned_coverage_area(self) -> float:
        """
        Estimate the area covered by the generated mowing lanes.

        This estimate is calculated as:

            lane length × cutting width

        for every generated lane. It is not a geometric union-area
        calculation.
        """
        lanes = self.generate_ordered_lanes()

        return sum(
            lane.length() * self.config.cutting_width
            for lane in lanes
        )

    def get_planned_coverage_percentage(self) -> float:
        """
        Return the estimated percentage of usable lawn covered
        by the mowing lanes.
        """
        usable_area = self.get_usable_area()

        if usable_area <= 0.0:
            return 0.0

        planned_area = self.get_planned_coverage_area()

        return min(
            (planned_area / usable_area) * 100.0,
            100.0,
        )

    def validate_planned_coverage(
        self,
        minimum_percentage: float = 95.0,
    ) -> bool:
        """
        Validate the estimated planned coverage against a minimum.

        Args:
            minimum_percentage: Required minimum coverage percentage.

        Returns:
            True when estimated coverage meets the requirement.

        Raises:
            ValueError: If the requested percentage is outside
                the range 0 to 100.
        """
        if not 0.0 <= minimum_percentage <= 100.0:
            raise ValueError(
                "minimum_percentage must be between 0 and 100."
            )

        return (
            self.get_planned_coverage_percentage()
            >= minimum_percentage
        )

    def get_lane_overlap(self) -> float:
        """
        Estimate the total overlap area between consecutive lanes.

        The calculation is a simplified rectangular-strip estimate.
        """
        lanes = self.generate_ordered_lanes()

        if len(lanes) < 2:
            return 0.0

        overlap_width = max(
            self.config.cutting_width
            - self.config.lane_spacing,
            0.0,
        )

        if overlap_width <= 0.0:
            return 0.0

        return sum(
            lane.length() * overlap_width
            for lane in lanes[:-1]
        )

    def get_overlap_percentage(self) -> float:
        """
        Return estimated lane overlap as a percentage of
        planned coverage area.
        """
        planned_area = self.get_planned_coverage_area()

        if planned_area <= 0.0:
            return 0.0

        overlap_area = self.get_lane_overlap()

        return min(
            (overlap_area / planned_area) * 100.0,
            100.0,
        )

    def validate_overlap(
        self,
        maximum_percentage: float = 10.0,
    ) -> bool:
        """
        Validate estimated lane overlap against a maximum percentage.

        Args:
            maximum_percentage: Maximum allowed overlap percentage.

        Returns:
            True when estimated overlap is within the limit.

        Raises:
            ValueError: If the requested percentage is outside
                the range 0 to 100.
        """
        if not 0.0 <= maximum_percentage <= 100.0:
            raise ValueError(
                "maximum_percentage must be between 0 and 100."
            )

        return (
            self.get_overlap_percentage()
            <= maximum_percentage
        )

    def get_lane_gap(self) -> float:
        """
        Return the estimated gap between consecutive mowing lanes.

        A positive value means lane spacing is greater than the
        configured cutting width.
        """
        return max(
            self.config.lane_spacing
            - self.config.cutting_width,
            0.0,
        )

    def get_missed_area(self) -> float:
        """
        Estimate the area that may remain uncovered because
        lane spacing is greater than cutting width.
        """
        lanes = self.generate_ordered_lanes()

        if len(lanes) < 2:
            return 0.0

        gap_width = self.get_lane_gap()

        if gap_width <= 0.0:
            return 0.0

        return sum(
            lane.length() * gap_width
            for lane in lanes[:-1]
        )

    def get_missed_area_percentage(self) -> float:
        """
        Return the estimated missed area as a percentage of
        usable lawn area.
        """
        usable_area = self.get_usable_area()

        if usable_area <= 0.0:
            return 0.0

        missed_area = self.get_missed_area()

        return min(
            (missed_area / usable_area) * 100.0,
            100.0,
        )

    def validate_missed_area(
        self,
        maximum_percentage: float = 5.0,
    ) -> bool:
        """
        Validate estimated missed area against a maximum percentage.

        Args:
            maximum_percentage: Maximum allowed missed-area percentage.

        Returns:
            True when estimated missed area is within the limit.

        Raises:
            ValueError: If the requested percentage is outside
                the range 0 to 100.
        """
        if not 0.0 <= maximum_percentage <= 100.0:
            raise ValueError(
                "maximum_percentage must be between 0 and 100."
            )

        return (
            self.get_missed_area_percentage()
            <= maximum_percentage
        )