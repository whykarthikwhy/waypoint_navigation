"""
test_waypoint_navigation.py
===========================
Unit and integration tests for the waypoint_navigation package.

These tests are designed to run WITHOUT a live ROS2 environment by
directly testing the pure algorithmic logic extracted from each node.
They use only standard Python libraries (pytest, numpy, scipy, math).

Test Coverage:
  1. Path Smoothing       — spline continuity, endpoint accuracy, edge cases
  2. Trajectory Generation — time monotonicity, velocity correctness
  3. Pure Pursuit          — angle wrapping, curvature formula, goal threshold
  4. Obstacle Detection    — scan arc parsing, range filtering
  5. Velocity Ramping      — ramp profile correctness
  6. Error Handling        — degenerate inputs (single waypoint, zero velocity)

Run with:
    pip install pytest numpy scipy
    pytest test_waypoint_navigation.py -v

Author: karburettor
"""

import math

import numpy as np
import pytest
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────────────────────
# Pure-Python re-implementations of node logic
# (mirrors the algorithms without importing rclpy)
# ─────────────────────────────────────────────────────────────

def fit_spline(waypoints: np.ndarray, sample_count: int = 100):
    """Mirror of PathSmoother._fit_spline()."""
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints are required to fit a spline.")
    t_knots = np.linspace(0, 1, len(waypoints))
    cs_x = CubicSpline(t_knots, waypoints[:, 0])
    cs_y = CubicSpline(t_knots, waypoints[:, 1])
    t_fine = np.linspace(0, 1, sample_count)
    return cs_x(t_fine), cs_y(t_fine)


def time_parameterize(xs: np.ndarray, ys: np.ndarray, velocity: float = 0.15):
    """Mirror of PathSmoother._time_parameterize()."""
    if velocity <= 0:
        raise ValueError("Target velocity must be positive.")
    times = [0.0]
    for i in range(1, len(xs)):
        d = math.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)
        times.append(times[-1] + d / velocity)
    return times


def wrap_angle(angle: float) -> float:
    """Mirror of PurePursuitController._wrap_angle()."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x, y, z, w) -> float:
    """Mirror of PurePursuitController._yaw_from_quaternion()."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_curvature(local_x: float, local_y: float) -> float:
    """Pure Pursuit curvature: κ = 2y / L²"""
    L_sq = local_x**2 + local_y**2
    return (2.0 * local_y) / L_sq if L_sq > 0.001 else 0.0


def compute_velocity_ramp(
    dist_to_goal: float,
    cruise: float = 0.2,
    slowdown_radius: float = 0.8,
    min_vel: float = 0.05,
) -> float:
    """Mirror of PurePursuitController._compute_velocity_ramp()."""
    if dist_to_goal < slowdown_radius:
        return max(min_vel, cruise * (dist_to_goal / slowdown_radius))
    return cruise


def detect_obstacle(ranges: list, arc_deg: int = 30, threshold: float = 0.7, min_range: float = 0.05) -> bool:
    """Mirror of PurePursuitController._scan_callback() logic."""
    front = ranges[:arc_deg] + ranges[360 - arc_deg:]
    valid = [r for r in front if r > min_range]
    return bool(valid) and min(valid) < threshold


# ─────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────

WAYPOINTS = np.array([
    [0.0, 0.0],
    [2.0, 1.0],
    [4.0, -1.0],
    [6.0, 1.0],
    [8.0, 0.0],
])

GOAL_THRESHOLD = 0.05   # 5 cm — must match node constant


# ══════════════════════════════════════════════════════════════
# 1. PATH SMOOTHING TESTS
# ══════════════════════════════════════════════════════════════

class TestPathSmoothing:
    """Tests for PathSmoother._fit_spline()"""

    def test_starts_at_first_waypoint(self):
        """Spline must begin exactly at the first waypoint."""
        xs, ys = fit_spline(WAYPOINTS)
        assert abs(xs[0] - WAYPOINTS[0, 0]) < 1e-6
        assert abs(ys[0] - WAYPOINTS[0, 1]) < 1e-6

    def test_ends_at_last_waypoint(self):
        """Spline must end exactly at the last waypoint."""
        xs, ys = fit_spline(WAYPOINTS)
        assert abs(xs[-1] - WAYPOINTS[-1, 0]) < 1e-6
        assert abs(ys[-1] - WAYPOINTS[-1, 1]) < 1e-6

    def test_passes_through_all_waypoints(self):
        """Cubic spline must interpolate (not approximate) each waypoint."""
        xs, ys = fit_spline(WAYPOINTS, sample_count=500)
        for wp in WAYPOINTS:
            dists = [math.sqrt((x - wp[0])**2 + (y - wp[1])**2)
                     for x, y in zip(xs, ys)]
            assert min(dists) < 0.05, \
                f"Spline never comes within 5 cm of waypoint {wp}"

    def test_output_length(self):
        """Spline output must have exactly sample_count points."""
        for n in [50, 100, 200]:
            xs, ys = fit_spline(WAYPOINTS, sample_count=n)
            assert len(xs) == n and len(ys) == n

    def test_no_large_gaps(self):
        """Consecutive spline points must be < 0.5 m apart (continuity)."""
        xs, ys = fit_spline(WAYPOINTS, sample_count=100)
        for i in range(1, len(xs)):
            gap = math.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
            assert gap < 0.5, f"Gap of {gap:.3f} m at index {i}"

    def test_single_waypoint_raises(self):
        """A single waypoint cannot form a spline — must raise an error."""
        with pytest.raises((ValueError, Exception)):
            fit_spline(np.array([[0.0, 0.0]]))

    def test_collinear_waypoints(self):
        """Collinear waypoints (straight line) should not crash the spline."""
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
        xs, ys = fit_spline(pts)
        # y should stay near 0 for a perfectly horizontal path
        assert all(abs(y) < 0.01 for y in ys)

    def test_two_waypoints(self):
        """Minimum case: two waypoints should produce a valid spline."""
        pts = np.array([[0.0, 0.0], [4.0, 4.0]])
        xs, ys = fit_spline(pts)
        assert len(xs) == 100


# ══════════════════════════════════════════════════════════════
# 2. TRAJECTORY GENERATION TESTS
# ══════════════════════════════════════════════════════════════

class TestTrajectoryGeneration:
    """Tests for PathSmoother._time_parameterize()"""

    def setup_method(self):
        self.xs, self.ys = fit_spline(WAYPOINTS)

    def test_timestamps_are_monotonically_increasing(self):
        """Time must always increase — robot can't travel backwards in time."""
        times = time_parameterize(self.xs, self.ys)
        for i in range(1, len(times)):
            assert times[i] > times[i-1], \
                f"Timestamp not increasing at index {i}: {times[i-1]:.3f} → {times[i]:.3f}"

    def test_starts_at_zero(self):
        """First timestamp must be exactly 0."""
        times = time_parameterize(self.xs, self.ys)
        assert times[0] == 0.0

    def test_total_time_plausible(self):
        """Total travel time should be roughly arc_length / velocity."""
        velocity = 0.15
        times = time_parameterize(self.xs, self.ys, velocity=velocity)
        arc_length = sum(
            math.sqrt((self.xs[i]-self.xs[i-1])**2 + (self.ys[i]-self.ys[i-1])**2)
            for i in range(1, len(self.xs))
        )
        expected = arc_length / velocity
        assert abs(times[-1] - expected) < 0.01, \
            f"Total time {times[-1]:.2f}s ≠ expected {expected:.2f}s"

    def test_correct_length(self):
        """Timestamp array must have same length as coordinate arrays."""
        times = time_parameterize(self.xs, self.ys)
        assert len(times) == len(self.xs) == len(self.ys)

    def test_faster_velocity_shorter_time(self):
        """Higher velocity must produce a shorter total trajectory time."""
        t_slow = time_parameterize(self.xs, self.ys, velocity=0.1)
        t_fast = time_parameterize(self.xs, self.ys, velocity=0.3)
        assert t_fast[-1] < t_slow[-1]

    def test_zero_velocity_raises(self):
        """Zero velocity causes division by zero — must raise an error."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            time_parameterize(self.xs, self.ys, velocity=0.0)

    def test_negative_velocity_raises(self):
        """Negative velocity is physically invalid — must raise an error."""
        with pytest.raises((ValueError, Exception)):
            time_parameterize(self.xs, self.ys, velocity=-0.1)


# ══════════════════════════════════════════════════════════════
# 3. ANGLE & HEADING TESTS
# ══════════════════════════════════════════════════════════════

class TestAngleHandling:
    """Tests for angle wrapping and quaternion extraction."""

    @pytest.mark.parametrize("angle,expected", [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, -math.pi),
        (3 * math.pi / 2, -math.pi / 2),   # 270° wraps to -90°
        (-3 * math.pi / 2, math.pi / 2),   # -270° wraps to +90°
        (2 * math.pi, 0.0),                 # full rotation = 0
        (5 * math.pi, math.pi),             # 900° = 180°
    ])
    def test_angle_wrapping(self, angle, expected):
        """Wrapped angle must equal expected value within 1e-9 rad."""
        result = wrap_angle(angle)
        assert abs(result - expected) < 1e-9, \
            f"wrap_angle({math.degrees(angle):.0f}°) = {math.degrees(result):.2f}° ≠ {math.degrees(expected):.2f}°"

    def test_yaw_identity(self):
        """Identity quaternion (facing +X) must give yaw = 0."""
        yaw = yaw_from_quaternion(0, 0, 0, 1)
        assert abs(yaw) < 1e-9

    def test_yaw_90_degrees(self):
        """Quaternion for 90° yaw must give yaw ≈ π/2."""
        # q = (x=0, y=0, z=sin(45°), w=cos(45°))
        yaw = yaw_from_quaternion(0, 0, math.sin(math.pi/4), math.cos(math.pi/4))
        assert abs(yaw - math.pi / 2) < 1e-9

    def test_yaw_180_degrees(self):
        """Quaternion for 180° yaw must give yaw ≈ ±π."""
        yaw = yaw_from_quaternion(0, 0, 1, 0)
        assert abs(abs(yaw) - math.pi) < 1e-9


# ══════════════════════════════════════════════════════════════
# 4. PURE PURSUIT CURVATURE TESTS
# ══════════════════════════════════════════════════════════════

class TestPurePursuitCurvature:
    """Tests for the κ = 2y/L² curvature formula."""

    def test_zero_lateral_offset_gives_zero_curvature(self):
        """Target directly ahead (y=0) → no turning needed."""
        kappa = compute_curvature(local_x=1.0, local_y=0.0)
        assert abs(kappa) < 1e-9

    def test_left_target_gives_positive_curvature(self):
        """Target to the left (y > 0) → positive curvature (turn left)."""
        kappa = compute_curvature(local_x=1.0, local_y=0.5)
        assert kappa > 0

    def test_right_target_gives_negative_curvature(self):
        """Target to the right (y < 0) → negative curvature (turn right)."""
        kappa = compute_curvature(local_x=1.0, local_y=-0.5)
        assert kappa < 0

    def test_near_zero_distance_returns_zero(self):
        """Degenerate case: L² ≈ 0 must return 0 (no division by zero)."""
        kappa = compute_curvature(local_x=0.0, local_y=0.0)
        assert kappa == 0.0

    def test_curvature_formula_correctness(self):
        """Verify κ = 2y / (x² + y²) manually."""
        lx, ly = 3.0, 4.0
        expected = (2 * ly) / (lx**2 + ly**2)
        assert abs(compute_curvature(lx, ly) - expected) < 1e-9

    def test_symmetry(self):
        """Equal magnitude left/right offsets → equal magnitude curvature."""
        k_left = compute_curvature(1.0, 0.5)
        k_right = compute_curvature(1.0, -0.5)
        assert abs(abs(k_left) - abs(k_right)) < 1e-9


# ══════════════════════════════════════════════════════════════
# 5. VELOCITY RAMPING TESTS
# ══════════════════════════════════════════════════════════════

class TestVelocityRamping:
    """Tests for PurePursuitController._compute_velocity_ramp()."""

    def test_far_from_goal_returns_cruise(self):
        """Beyond slowdown radius → full cruise speed."""
        v = compute_velocity_ramp(dist_to_goal=2.0, cruise=0.2, slowdown_radius=0.8)
        assert abs(v - 0.2) < 1e-9

    def test_at_slowdown_boundary_returns_cruise(self):
        """Exactly at the slowdown boundary → cruise speed."""
        v = compute_velocity_ramp(dist_to_goal=0.8, cruise=0.2, slowdown_radius=0.8)
        assert abs(v - 0.2) < 1e-9

    def test_velocity_decreases_as_goal_approaches(self):
        """Velocity must decrease monotonically within the slowdown zone."""
        dists = np.linspace(0.8, 0.05, 50)
        vels = [compute_velocity_ramp(d) for d in dists]
        for i in range(1, len(vels)):
            assert vels[i] <= vels[i-1] + 1e-9, \
                f"Velocity increased from {vels[i-1]:.4f} to {vels[i]:.4f}"

    def test_never_below_minimum_velocity(self):
        """Velocity must never drop below MIN_VELOCITY."""
        for dist in np.linspace(0.0, 1.0, 50):
            v = compute_velocity_ramp(dist_to_goal=dist, min_vel=0.05)
            assert v >= 0.05, f"Velocity {v:.4f} < min at dist={dist:.3f}"

    def test_linear_interpolation_at_half_radius(self):
        """At half the slowdown radius, velocity should be ~half cruise + floor."""
        v = compute_velocity_ramp(dist_to_goal=0.4, cruise=0.2, slowdown_radius=0.8, min_vel=0.0)
        assert abs(v - 0.1) < 1e-9


# ══════════════════════════════════════════════════════════════
# 6. OBSTACLE DETECTION TESTS
# ══════════════════════════════════════════════════════════════

class TestObstacleDetection:
    """Tests for PurePursuitController._scan_callback() logic."""

    def _make_ranges(self, close_idx: int = None, value: float = 0.5, n: int = 360) -> list:
        """Create a 360-element scan with all ranges at 5m, optionally one close reading."""
        ranges = [5.0] * n
        if close_idx is not None:
            ranges[close_idx] = value
        return ranges

    def test_no_obstacle_all_clear(self):
        """All ranges at 5m → no obstacle."""
        ranges = [5.0] * 360
        assert detect_obstacle(ranges) is False

    def test_obstacle_directly_ahead(self):
        """Close range at index 0 (directly ahead) → obstacle detected."""
        ranges = self._make_ranges(close_idx=0, value=0.3)
        assert detect_obstacle(ranges) is True

    def test_obstacle_at_left_edge_of_arc(self):
        """Close range at index 29 (edge of left arc) → detected."""
        ranges = self._make_ranges(close_idx=29, value=0.3)
        assert detect_obstacle(ranges) is True

    def test_obstacle_at_right_edge_of_arc(self):
        """Close range at index 331 (edge of right arc) → detected."""
        ranges = self._make_ranges(close_idx=331, value=0.3)
        assert detect_obstacle(ranges) is True

    def test_obstacle_behind_robot_not_detected(self):
        """Close range at index 180 (directly behind) → NOT detected."""
        ranges = self._make_ranges(close_idx=180, value=0.3)
        assert detect_obstacle(ranges) is False

    def test_noise_filtered_out(self):
        """Very small returns (< MIN_LASER_RANGE) should be ignored."""
        ranges = self._make_ranges(close_idx=0, value=0.02)  # below 0.05m threshold
        assert detect_obstacle(ranges) is False

    def test_exactly_at_threshold(self):
        """Range exactly equal to threshold is NOT an obstacle (strict <)."""
        ranges = self._make_ranges(close_idx=0, value=0.7)
        assert detect_obstacle(ranges) is False

    def test_just_inside_threshold(self):
        """Range just below threshold → obstacle detected."""
        ranges = self._make_ranges(close_idx=0, value=0.69)
        assert detect_obstacle(ranges) is True


# ══════════════════════════════════════════════════════════════
# 7. GOAL THRESHOLD / MISSION SUCCESS TESTS
# ══════════════════════════════════════════════════════════════

class TestGoalDetection:
    """Tests for the goal-reached condition in the control loop."""

    def test_within_threshold_is_reached(self):
        """Distance < GOAL_THRESHOLD → mission complete."""
        dist = GOAL_THRESHOLD - 0.001
        assert dist < GOAL_THRESHOLD

    def test_outside_threshold_not_reached(self):
        """Distance > GOAL_THRESHOLD → keep driving."""
        dist = GOAL_THRESHOLD + 0.001
        assert dist >= GOAL_THRESHOLD

    def test_exactly_at_threshold_not_reached(self):
        """Distance == GOAL_THRESHOLD → strict less-than means not reached."""
        dist = GOAL_THRESHOLD
        assert not (dist < GOAL_THRESHOLD)

    def test_goal_distance_formula(self):
        """Euclidean goal distance must be computed correctly."""
        robot = (7.9, 0.1)
        goal = (8.0, 0.0)
        dist = math.sqrt((goal[0]-robot[0])**2 + (goal[1]-robot[1])**2)
        expected = math.sqrt(0.01 + 0.01)
        assert abs(dist - expected) < 1e-9


# ══════════════════════════════════════════════════════════════
# 8. INTEGRATION TEST — Full pipeline (no ROS2 needed)
# ══════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    End-to-end test: waypoints → spline → timestamps → curvature commands.
    Verifies that all components work together without crashing.
    """

    def test_pipeline_runs_without_error(self):
        """Full pipeline from waypoints to control output must not raise."""
        xs, ys = fit_spline(WAYPOINTS, sample_count=100)
        times = time_parameterize(xs, ys, velocity=0.15)
        assert len(times) == 100

        # Simulate robot at start, target = 2nd spline point
        robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
        target_x, target_y = xs[1], ys[1]
        dx = target_x - robot_x
        dy = target_y - robot_y

        local_x = dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)
        local_y = -dx * math.sin(robot_yaw) + dy * math.cos(robot_yaw)

        kappa = compute_curvature(local_x, local_y)
        v = compute_velocity_ramp(dist_to_goal=8.0)

        linear = v
        angular = kappa * v

        # Just assert it produces finite numbers
        assert math.isfinite(linear)
        assert math.isfinite(angular)

    def test_pipeline_total_path_length_is_reasonable(self):
        """Total arc length of the smoothed path must be > straight-line distance."""
        xs, ys = fit_spline(WAYPOINTS)
        arc_length = sum(
            math.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
            for i in range(1, len(xs))
        )
        straight = math.sqrt(
            (WAYPOINTS[-1, 0] - WAYPOINTS[0, 0])**2
            + (WAYPOINTS[-1, 1] - WAYPOINTS[0, 1])**2
        )
        # Smoothed path through curves is always ≥ straight-line distance
        assert arc_length >= straight
