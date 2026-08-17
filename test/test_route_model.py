# -*- coding: utf-8 -*-
"""Unit tests for the declarative Stage 5 route model.

这些测试只覆盖纯逻辑（段表一致性、里程累积、双证据门控、转角航向校核、
限速裁剪）。它们不能替代仿真回归，也不构成 G1/G2 证据
（STAGE5_PHYSICAL_REDESIGN_PLAN.md §9）。
"""

import math
import os

import pytest
import yaml

from control_node.route_model import (
    CrossTrackGate,
    ENFORCEMENT_ENFORCE,
    ENFORCEMENT_MONITOR,
    EXIT_ODOMETRY,
    TIER_DEAD_RECKONING,
    TIER_NOMINAL,
    odometry_exit_reached,
    GATE_BELOW_MIN,
    GATE_IN_WINDOW,
    GATE_OVERRUN,
    GATE_UNAVAILABLE,
    PROGRESS_DISTANCE,
    PROGRESS_YAW,
    RouteModel,
    RouteSegment,
    SegmentProgress,
    StallGate,
    ToppleGate,
    angle_delta_rad,
    clamp_speed,
    default_segments,
    evaluate_gate,
    normalize_angle_rad,
    verify_yaw,
)


# ------------------------------------------------------------------
# 段表
# ------------------------------------------------------------------
def test_default_table_is_self_consistent():
    """Default table is self consistent."""
    model = RouteModel()
    assert model.validate() == []


def test_default_table_covers_the_gated_states_once():
    """Default table covers the gated states once."""
    model = RouteModel()
    for state in (
        'P5_STEP_UP',
        'P5_UP_SLOPE',
        'P5_RIGHT_SLOPE_1',
        'P5_RIGHT_SLOPE_2',
        'P5_RIGHT_SLOPE_3',
        'P5_TURN_1',
        'P5_TURN_2',
        'P5_RIGHT_JUMP_AFTER_RESET_BODY',
        'P5_FINAL_LONG_JUMP',
    ):
        assert model.segment_for_state(state) is not None, state
    assert model.segment_for_state('P5_SENSOR_FAULT_HOLD') is None
    assert model.segment_for_state('P5_ROUTE_REALIGN') is None


def test_duplicate_state_in_two_segments_is_rejected():
    """Duplicate state in two segments is rejected."""
    with pytest.raises(ValueError):
        RouteModel((
            RouteSegment(name='a', states=('S',)),
            RouteSegment(name='b', states=('S',)),
        ))


def test_enforced_segments_declare_a_perception_exit_and_a_window():
    """Enforced segments declare a perception exit and a window."""
    for segment in default_segments():
        if not segment.enforced:
            continue
        assert segment.exit_evidence
        if segment.progress == PROGRESS_DISTANCE:
            assert segment.max_m > segment.min_m
            # An enforced perception exit needs a declared degraded successor,
            # otherwise an overrun can only ever fault.
            assert segment.degraded_next_state
        if segment.progress == PROGRESS_YAW:
            assert segment.yaw_tol_deg > 0.0


def test_overrides_apply_and_reject_unknown_names():
    """Overrides apply and reject unknown names."""
    model = RouteModel().with_overrides({'straight_1': {'min_m': 1.0, 'max_m': 2.0}})
    segment = model.segment_by_name('straight_1')
    assert segment.window == (1.0, 2.0)
    # Untouched segments keep their defaults.
    assert model.segment_by_name('straight_2').max_m == \
        RouteModel().segment_by_name('straight_2').max_m
    with pytest.raises(ValueError):
        RouteModel().with_overrides({'straight_9': {'min_m': 1.0}})


def test_validate_reports_inverted_and_missing_windows():
    """Validate reports inverted and missing windows."""
    problems = RouteModel((
        RouteSegment(name='bad', states=('S',), progress=PROGRESS_DISTANCE,
                     min_m=3.0, max_m=1.0),
        RouteSegment(name='nomax', states=('T',), progress=PROGRESS_DISTANCE,
                     enforcement=ENFORCEMENT_ENFORCE, min_m=1.0, max_m=0.0),
        RouteSegment(name='notol', states=('U',), progress=PROGRESS_YAW,
                     enforcement=ENFORCEMENT_ENFORCE, yaw_tol_deg=0.0),
    )).validate()
    assert len(problems) == 3


# ------------------------------------------------------------------
# 里程累积
# ------------------------------------------------------------------
def test_progress_accumulates_path_length_not_displacement():
    """Progress accumulates path length not displacement."""
    progress = SegmentProgress()
    progress.reset('seg')
    # Out and back: 0.2 m of path, zero displacement.
    for seq, x in enumerate([0.0, 0.05, 0.10, 0.05, 0.0]):
        progress.update(seq, x, 0.0, 0.0)
    assert progress.distance_m == pytest.approx(0.20)
    assert progress.samples == 5


def test_repeated_sequence_number_is_ignored():
    """Repeated sequence number is ignored."""
    progress = SegmentProgress()
    progress.reset('seg')
    assert progress.update(1, 0.0, 0.0, 0.0) is True
    assert progress.update(1, 5.0, 5.0, 1.0) is False
    assert progress.distance_m == 0.0
    assert progress.samples == 1


def test_implausible_jump_is_rejected_but_counted():
    """Implausible jump is rejected but counted."""
    progress = SegmentProgress(max_step_m=0.10)
    progress.reset('seg')
    progress.update(1, 0.0, 0.0, 0.0)
    progress.update(2, 0.05, 0.0, 0.0)
    progress.update(3, 3.05, 0.0, 0.0)     # estimator reset / teleport
    progress.update(4, 3.10, 0.0, 0.0)
    assert progress.distance_m == pytest.approx(0.10)
    assert progress.rejected_steps == 1
    assert progress.rejected_distance_m == pytest.approx(3.0)


def test_yaw_is_unwrapped_across_the_pi_boundary():
    """Yaw is unwrapped across the pi boundary."""
    progress = SegmentProgress()
    progress.reset('turn')
    yaws = [3.0, 3.1, -3.05, -2.9]         # keeps rotating positively
    for seq, yaw in enumerate(yaws):
        progress.update(seq, 0.0, 0.0, yaw)
    expected = 0.1 + (2.0 * math.pi - 6.15) + 0.15
    assert progress.yaw_delta_rad == pytest.approx(expected, abs=1e-9)
    assert progress.yaw_delta_deg == pytest.approx(math.degrees(expected))


def test_yaw_accumulates_even_when_the_translation_step_is_rejected():
    """A turn jump barely translates but rotates a lot; keep the rotation."""
    progress = SegmentProgress(max_step_m=0.05)
    progress.reset('turn')
    progress.update(1, 0.0, 0.0, 0.0)
    progress.update(2, 1.0, 0.0, -math.pi / 2.0)
    assert progress.rejected_steps == 1
    assert progress.distance_m == 0.0
    assert progress.yaw_delta_deg == pytest.approx(-90.0)


def test_reset_clears_progress_and_origin():
    """Reset clears progress and origin."""
    progress = SegmentProgress()
    progress.reset('a')
    progress.update(1, 0.0, 0.0, 0.0)
    progress.update(2, 1.0, 0.0, 0.5)
    progress.reset('b')
    assert progress.segment_name == 'b'
    assert progress.distance_m == 0.0
    assert progress.yaw_delta_rad == 0.0
    assert progress.last_seq is None
    assert progress.snapshot()['samples'] == 0


# ------------------------------------------------------------------
# 双证据门控
# ------------------------------------------------------------------
ENFORCED = RouteSegment(
    name='straight', states=('S',), progress=PROGRESS_DISTANCE,
    min_m=2.0, max_m=4.0, enforcement=ENFORCEMENT_ENFORCE,
    degraded_next_state='NEXT')

MONITORED = RouteSegment(
    name='timed', states=('T',), progress=PROGRESS_DISTANCE,
    min_m=2.0, max_m=4.0, enforcement=ENFORCEMENT_MONITOR)


def test_enforced_gate_suppresses_an_early_perception_trigger():
    """Enforced gate suppresses an early perception trigger."""
    decision = evaluate_gate(ENFORCED, progress_m=1.0, exit_confirmed=True,
                             odometry_valid=True)
    assert decision.status == GATE_BELOW_MIN
    assert decision.allow_exit is False


def test_enforced_gate_allows_exit_only_when_both_sources_agree():
    """Enforced gate allows exit only when both sources agree."""
    inside_no_vision = evaluate_gate(ENFORCED, 3.0, False, True)
    assert inside_no_vision.status == GATE_IN_WINDOW
    assert inside_no_vision.allow_exit is False

    both = evaluate_gate(ENFORCED, 3.0, True, True)
    assert both.status == GATE_IN_WINDOW
    assert both.allow_exit is True
    assert both.reason == 'two_sources_agree'


def test_enforced_gate_reports_overrun_and_never_auto_advances():
    """Enforced gate reports overrun and never auto advances."""
    decision = evaluate_gate(ENFORCED, 4.5, True, True)
    assert decision.status == GATE_OVERRUN
    assert decision.allow_exit is False


def test_enforced_gate_fails_closed_without_odometry():
    """Enforced gate fails closed without odometry."""
    decision = evaluate_gate(ENFORCED, 3.0, True, False)
    assert decision.status == GATE_UNAVAILABLE
    assert decision.allow_exit is False


def test_monitored_gate_keeps_its_own_authority_but_still_reports_the_window():
    """Monitored gate keeps its own authority but still reports the window."""
    early = evaluate_gate(MONITORED, 1.0, True, True)
    assert early.allow_exit is True
    assert early.status == GATE_BELOW_MIN
    assert early.reason == 'monitor_only'

    late = evaluate_gate(MONITORED, 9.0, False, True)
    assert late.allow_exit is False
    assert late.status == GATE_OVERRUN

    blind = evaluate_gate(MONITORED, 3.0, True, False)
    assert blind.allow_exit is True
    assert blind.status == GATE_UNAVAILABLE


def test_gate_decision_serializes_for_the_evidence_log():
    """Gate decision serializes for the evidence log."""
    payload = evaluate_gate(ENFORCED, 3.0, True, True).to_dict()
    assert payload['window_m'] == [2.0, 4.0]
    assert payload['progress_m'] == pytest.approx(3.0)
    assert payload['allow_exit'] is True


# ------------------------------------------------------------------
# 转角航向校核
# ------------------------------------------------------------------
CORNER = RouteSegment(name='corner', states=('C',), progress=PROGRESS_YAW,
                      expected_yaw_deg=-90.0, yaw_tol_deg=30.0,
                      enforcement=ENFORCEMENT_ENFORCE)


def test_yaw_verification_accepts_a_turn_inside_tolerance():
    """Yaw verification accepts a turn inside tolerance."""
    ok, error = verify_yaw(CORNER, -75.0)
    assert ok is True
    assert error == pytest.approx(-15.0)


def test_yaw_verification_rejects_a_missed_turn_and_signs_the_correction():
    """Yaw verification rejects a missed turn and signs the correction."""
    ok, error = verify_yaw(CORNER, -20.0)
    assert ok is False
    # Still needs to rotate negatively (clockwise) by ~70 deg.
    assert error == pytest.approx(-70.0)

    ok, error = verify_yaw(CORNER, -170.0)
    assert ok is False
    assert error == pytest.approx(80.0)


def test_yaw_verification_wraps_instead_of_reporting_a_full_turn_error():
    """Yaw verification wraps instead of reporting a full turn error."""
    ok, error = verify_yaw(
        RouteSegment(name='c', states=('C',), progress=PROGRESS_YAW,
                     expected_yaw_deg=-90.0, yaw_tol_deg=30.0),
        270.0)
    assert ok is True
    assert error == pytest.approx(0.0, abs=1e-9)


def test_zero_tolerance_disables_yaw_verification():
    """Zero tolerance disables yaw verification."""
    ok, _ = verify_yaw(
        RouteSegment(name='c', states=('C',), progress=PROGRESS_YAW,
                     expected_yaw_deg=-90.0, yaw_tol_deg=0.0),
        0.0)
    assert ok is True


# ------------------------------------------------------------------
# 角度工具与限速
# ------------------------------------------------------------------
def test_angle_helpers():
    """Angle helpers."""
    assert normalize_angle_rad(math.pi) == pytest.approx(math.pi)
    assert normalize_angle_rad(-math.pi) == pytest.approx(math.pi)
    assert normalize_angle_rad(3.0 * math.pi) == pytest.approx(math.pi)
    assert angle_delta_rad(3.1, -3.1) == pytest.approx(
        2.0 * math.pi - 6.2, abs=1e-9)


def test_speed_cap_scales_direction_preserving_and_leaves_slow_commands_alone():
    """Speed cap scales direction preserving and leaves slow commands alone."""
    vx, vy, capped = clamp_speed(0.2, 0.3, 0.4)     # speed 0.5
    assert capped is True
    assert math.hypot(vx, vy) == pytest.approx(0.2)
    assert vx / vy == pytest.approx(0.3 / 0.4)

    vx, vy, capped = clamp_speed(0.5, 0.1, 0.0)
    assert (vx, vy, capped) == (0.1, 0.0, False)


def test_speed_cap_of_zero_is_disabled_and_zero_command_stays_zero():
    """Speed cap of zero is disabled and zero command stays zero."""
    assert clamp_speed(0.0, 5.0, -5.0) == (5.0, -5.0, False)
    assert clamp_speed(0.2, 0.0, 0.0) == (0.0, 0.0, False)


# ------------------------------------------------------------------
# 里程主导段尾（不依赖黄线）
# ------------------------------------------------------------------
ODOMETRY_EXIT = RouteSegment(
    name='straight', states=('S',), progress=PROGRESS_DISTANCE,
    expected_m=3.0, min_m=2.0, max_m=4.0,
    enforcement=ENFORCEMENT_ENFORCE, exit_source=EXIT_ODOMETRY,
    fallback_tier=TIER_DEAD_RECKONING, degraded_next_state='NEXT')


def test_odometry_exit_fires_at_the_declared_length():
    """Odometry exit fires at the declared length."""
    assert odometry_exit_reached(ODOMETRY_EXIT, 2.9, True).allow_exit is False
    decision = odometry_exit_reached(ODOMETRY_EXIT, 3.0, True)
    assert decision.allow_exit is True
    assert decision.reason == 'odometry_length_reached'
    assert decision.status == GATE_IN_WINDOW


def test_odometry_exit_fails_closed_without_odometry():
    """Odometry exit fails closed without odometry."""
    decision = odometry_exit_reached(ODOMETRY_EXIT, 3.5, False)
    assert decision.allow_exit is False
    assert decision.status == GATE_UNAVAILABLE


def test_odometry_exit_needs_a_declared_length():
    """Odometry exit needs a declared length."""
    segment = ODOMETRY_EXIT.replace(expected_m=0.0)
    decision = odometry_exit_reached(segment, 9.0, True)
    assert decision.allow_exit is False
    assert decision.reason == 'no_expected_length'


def test_odometry_exit_still_reports_an_overrun():
    """Odometry exit still reports an overrun."""
    decision = odometry_exit_reached(ODOMETRY_EXIT, 4.5, True)
    assert decision.allow_exit is False
    assert decision.status == GATE_OVERRUN


def test_odometry_exit_must_declare_the_dead_reckoning_tier():
    """A single-source segment may not claim the nominal fallback tier."""
    problems = RouteModel((ODOMETRY_EXIT.replace(
        fallback_tier=TIER_NOMINAL),)).validate()
    assert any('fallback_tier' in problem for problem in problems)
    assert RouteModel((ODOMETRY_EXIT,)).validate() == []


def test_odometry_exit_length_must_sit_inside_the_window():
    """Odometry exit length must sit inside the window."""
    outside = RouteModel((ODOMETRY_EXIT.replace(expected_m=5.0),)).validate()
    assert any('outside window' in problem for problem in outside)
    missing = RouteModel((ODOMETRY_EXIT.replace(expected_m=0.0),)).validate()
    assert any('positive expected_m' in problem for problem in missing)


def test_vision_segments_keep_the_two_source_gate_by_default():
    """Vision segments keep the two source gate by default."""
    for segment in default_segments():
        assert segment.odometry_exit is False


def test_along_track_ignores_lateral_excursions():
    """Side-stepping adds path length but must not add route progress.

    This is the defect that fired a corner 0.6 m short of the next rail: the
    harder the centring loop worked, the earlier a path-length-gated segment
    ended.
    """
    progress = SegmentProgress()
    progress.reset('straight')
    progress.set_reference_yaw(0.0)
    for seq, (x, y) in enumerate(
            [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.4, 0.1), (0.6, 0.1)], 1):
        progress.update(seq=seq, x=x, y=y, yaw=0.0)
    assert progress.distance_m == pytest.approx(0.7)
    assert progress.along_track_m == pytest.approx(0.6)


def test_along_track_uses_the_reference_heading_not_the_entry_yaw():
    """A sloppy corner must not tilt the axis progress is measured along."""
    progress = SegmentProgress()
    progress.reset('straight')
    progress.update(seq=1, x=0.0, y=0.0, yaw=0.20)
    progress.update(seq=2, x=0.2, y=0.0, yaw=0.20)
    # Entry yaw 0.20 rad: progress along it is only cos(0.20) of the true run.
    assert progress.along_track_m == pytest.approx(0.2 * math.cos(0.20))
    progress.set_reference_yaw(0.0)
    assert progress.along_track_m == pytest.approx(0.2)


def test_along_track_is_signed_so_backwards_motion_undoes_progress():
    """Walking back down a segment must reduce progress, not add to it."""
    progress = SegmentProgress()
    progress.reset('straight')
    progress.set_reference_yaw(0.0)
    progress.update(seq=1, x=0.0, y=0.0, yaw=0.0)
    progress.update(seq=2, x=0.2, y=0.0, yaw=0.0)
    progress.update(seq=3, x=0.1, y=0.0, yaw=0.0)
    assert progress.distance_m == pytest.approx(0.3)
    assert progress.along_track_m == pytest.approx(0.1)


def test_along_track_appears_in_the_snapshot():
    """The evidence log must carry both measures, not silently one."""
    progress = SegmentProgress()
    progress.reset('straight')
    progress.set_reference_yaw(0.0)
    progress.update(seq=1, x=0.0, y=0.0, yaw=0.0)
    progress.update(seq=2, x=0.15, y=0.05, yaw=0.0)
    snap = progress.snapshot()
    assert snap['along_track_m'] == pytest.approx(0.15)
    assert snap['distance_m'] == pytest.approx(math.hypot(0.15, 0.05))
    assert snap['reference_yaw'] == pytest.approx(0.0)


# ------------------------------------------------------------------
# 入段深度完整性门
# ------------------------------------------------------------------

def _entry_gate(min_frames=3):
    from control_node.route_model import EntryDepthGate
    return EntryDepthGate(segment_name='up_slope',
                          min_valid_frames=min_frames)


def test_entry_gate_trips_on_a_blind_climb():
    """The r07 shape: zero valid frames during the climb -> trip."""
    gate = _entry_gate()
    for _ in range(60):
        gate.record_frame(False, 'up_slope')
    assert gate.segment_closed('up_slope') is True
    assert gate.tripped is True


def test_entry_gate_passes_a_seen_climb():
    gate = _entry_gate()
    for _ in range(5):
        gate.record_frame(True, 'up_slope')
    assert gate.segment_closed('up_slope') is False
    assert gate.tripped is False


def test_entry_gate_ignores_frames_outside_the_segment():
    """Both floor runs got valid fixes on the entrance step from the
    ground *before* the climb segment; those frames must not count."""
    gate = _entry_gate()
    for _ in range(10):
        gate.record_frame(True, '')            # before any segment
        gate.record_frame(True, 'entry_step_up')
    assert gate.segment_closed('up_slope') is True


def test_entry_gate_only_examines_the_configured_segment():
    gate = _entry_gate()
    assert gate.segment_closed('straight_1') is False
    assert gate.checked is False
    # blind rails after a good climb must not re-arm it
    for _ in range(5):
        gate.record_frame(True, 'up_slope')
    assert gate.segment_closed('up_slope') is False
    assert gate.segment_closed('up_slope') is False
    assert gate.tripped is False


def test_entry_gate_checks_exactly_once():
    gate = _entry_gate()
    assert gate.segment_closed('up_slope') is True
    # a later re-entry of the same segment name must not trip again
    gate.record_frame(True, 'up_slope')
    assert gate.segment_closed('up_slope') is False
    assert gate.tripped is True


def test_entry_gate_disabled_never_trips():
    from control_node.route_model import EntryDepthGate
    for gate in (EntryDepthGate(),
                 EntryDepthGate(segment_name='up_slope', min_valid_frames=0),
                 EntryDepthGate(segment_name='', min_valid_frames=3)):
        for _ in range(10):
            gate.record_frame(False, 'up_slope')
        assert gate.segment_closed('up_slope') is False
        assert gate.tripped is False


def test_entry_gate_snapshot_carries_the_verdict():
    gate = _entry_gate(min_frames=2)
    gate.record_frame(True, 'up_slope')
    gate.segment_closed('up_slope')
    snap = gate.snapshot()
    assert snap == {
        'segment': 'up_slope',
        'min_valid_frames': 2,
        'valid_frames': 1,
        'checked': True,
        'tripped': True,
    }


# ------------------------------------------------------------------
# 段内横向偏离（计划书第 32 条）
# ------------------------------------------------------------------
def test_cross_track_is_the_perpendicular_of_the_same_projection():
    """Along-track and cross-track split the same displacement."""
    progress = SegmentProgress()
    progress.reset('straight')
    progress.set_reference_yaw(0.0)
    progress.update(seq=1, x=0.0, y=0.0, yaw=0.0)
    progress.update(seq=2, x=0.6, y=0.1, yaw=0.0)
    assert progress.along_track_m == pytest.approx(0.6)
    assert progress.cross_track_m == pytest.approx(0.1)


def test_cross_track_is_signed_left_positive():
    """Left of the entry line is positive, matching the lateral loop."""
    progress = SegmentProgress()
    progress.reset('straight')
    progress.set_reference_yaw(math.pi / 2.0)          # travelling +y
    progress.update(seq=1, x=0.0, y=0.0, yaw=math.pi / 2.0)
    progress.update(seq=2, x=-0.1, y=0.5, yaw=math.pi / 2.0)
    assert progress.cross_track_m == pytest.approx(0.1)


def test_cross_track_resets_with_the_segment():
    """A new segment means a new entry line, so the offset restarts at zero."""
    progress = SegmentProgress()
    progress.reset('a')
    progress.set_reference_yaw(0.0)
    progress.update(seq=1, x=0.0, y=0.0, yaw=0.0)
    progress.update(seq=2, x=0.5, y=0.3, yaw=0.0)
    assert progress.cross_track_m == pytest.approx(0.3)
    progress.reset('b')
    assert progress.cross_track_m == 0.0
    assert progress.snapshot()['cross_track_m'] == 0.0


def test_cross_track_gate_is_off_at_zero_limit():
    """The gate must be opt-in: a zero limit never trips, whatever it sees."""
    gate = CrossTrackGate(limit_m=0.0, consecutive_samples=1)
    assert gate.enabled is False
    for _ in range(100):
        assert gate.record(5.0) is False
    assert gate.tripped is False


def test_cross_track_gate_needs_a_sustained_excursion():
    """One bad fix is noise; half a second of them is a departure."""
    gate = CrossTrackGate(limit_m=0.30, consecutive_samples=3)
    assert gate.record(0.9) is False                   # 1
    assert gate.record(0.1) is False                   # streak broken
    assert gate.record(0.9) is False                   # 1
    assert gate.record(0.9) is False                   # 2
    assert gate.record(0.9) is True                    # 3 -> trips
    assert gate.tripped is True


def test_cross_track_gate_trips_on_either_side():
    """Walking off the inner void is as fatal as walking off the outer edge."""
    gate = CrossTrackGate(limit_m=0.30, consecutive_samples=2)
    assert gate.record(-0.55) is False
    assert gate.record(-0.55) is True


def test_cross_track_gate_trips_once_and_remembers_the_worst():
    """Trip is edge-triggered; the snapshot still carries what was seen."""
    gate = CrossTrackGate(limit_m=0.30, consecutive_samples=1)
    assert gate.record(0.40) is True
    assert gate.record(0.90) is False                  # already tripped
    snap = gate.snapshot()
    assert snap['tripped'] is True
    assert snap['worst_cross_track_m'] == pytest.approx(0.40)
    assert snap['limit_m'] == pytest.approx(0.30)


def test_cross_track_gate_ignores_unusable_samples():
    """A NaN or a non-number is missing data, not a departure."""
    gate = CrossTrackGate(limit_m=0.30, consecutive_samples=1)
    assert gate.record(float('nan')) is False
    assert gate.record(None) is False
    assert gate.record('0.9') is True                  # numeric strings count
    assert gate.tripped is True


def test_cross_track_gate_resets_with_the_segment():
    """A reset clears the verdict as well as the streak."""
    gate = CrossTrackGate(limit_m=0.30, consecutive_samples=1)
    gate.record(0.9)
    gate.reset()
    assert gate.tripped is False
    assert gate.snapshot()['worst_cross_track_m'] == 0.0


def test_cross_track_gate_only_applies_to_segments_with_an_edge():
    """A jump across the floor is a manoeuvre, not a departure."""
    by_name = {s.name: s for s in default_segments()}
    for name in ('up_slope', 'straight_1', 'straight_2', 'straight_3',
                 'straight_3_exit'):
        assert CrossTrackGate.applies_to(by_name[name]) is True, name
    for name in ('corner_1', 'corner_4', 'right_descent_align',
                 'right_descent', 'final_zone', 'entry_step_up'):
        assert CrossTrackGate.applies_to(by_name[name]) is False, name
    assert CrossTrackGate.applies_to(None) is False


# ------------------------------------------------------------------
# 参数档自洽性
# ------------------------------------------------------------------
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'config')
#: Each profile as it is actually loaded: the launch file chains the overlays,
#: so a value is only wrong if it is wrong *after* the chain.
_PROFILE_CHAINS = {
    'sim': ['stage5_sim.yaml'],
    'sim_odometry': ['stage5_sim.yaml', 'stage5_sim_odometry.yaml'],
    'sim_depth': ['stage5_sim.yaml', 'stage5_sim_odometry.yaml',
                  'stage5_sim_depth.yaml'],
    'physical': ['stage5_physical.yaml'],
}


def _load_profile(chain):
    """Merge an overlay chain into one parameter dict, in load order."""
    merged = {}
    for name in chain:
        path = os.path.join(_CONFIG_DIR, name)
        with open(path) as handle:
            doc = yaml.safe_load(handle) or {}
        for node in doc.values():
            merged.update((node or {}).get('ros__parameters', {}) or {})
    return merged


@pytest.mark.parametrize('profile', sorted(_PROFILE_CHAINS))
def test_shipped_profile_builds_a_valid_route_table(profile):
    """Every shipped profile must survive the same check the node applies.

    ``p5_load_params`` raises on an invalid table, so a profile that fails here
    does not start the stage — it is a launch-time crash, not a degraded run.
    Worth a unit test because the numbers are hand-calibrated per profile and a
    length is easy to move outside its own window while tuning.
    """
    params = _load_profile(_PROFILE_CHAINS[profile])
    exit_source = params.get('p5_route_exit_source', 'vision')
    overrides = {}
    for segment in RouteModel().segments:
        prefix = 'p5_route_{}'.format(segment.name)
        fields = {}
        for field in ('expected_m', 'min_m', 'max_m',
                      'expected_yaw_deg', 'yaw_tol_deg', 'speed_cap_mps'):
            key = '{}_{}'.format(prefix, field)
            if key in params:
                fields[field] = float(params[key])
        if segment.enforced and segment.progress == PROGRESS_DISTANCE:
            fields['exit_source'] = exit_source
            if exit_source == EXIT_ODOMETRY:
                fields['fallback_tier'] = TIER_DEAD_RECKONING
                fields['exit_evidence'] = 'odometry distance only'
        overrides[segment.name] = fields
    assert RouteModel().with_overrides(overrides).validate() == []


def test_sim_depth_declares_the_corner_1_approach_correction():
    """The climb gate must fire early enough for the approach that follows it.

    Measured over 31 runs in three batches: `P5_AFTER_UP_SLOPE_FORWARD` carries
    the body +0.611 m and the in-place corner turn a further +0.131 m, so the
    gate has to fire ~0.74 m short of the corner centre.  At 3.75 it fired
    0.68 m short and straight_1 was entered a median +0.067 m past the corner —
    more than the whole in-segment excursion.

    The value is empirical, and deliberately not derived: 3.75, 3.72 and 3.69
    were each measured, and 3.75 and 3.72 turned out statistically
    indistinguishable in entry offset.  3.72 ships because it is the only
    configuration in which all twelve runs of a batch traversed all four rails
    without leaving one — see plan item 36, which also records why the
    ratio-based reasoning that produced 3.69 was wrong.
    """
    params = _load_profile(_PROFILE_CHAINS['sim_depth'])
    assert params['p5_route_up_slope_expected_m'] == pytest.approx(3.72)


def test_topple_gate_is_off_by_default():
    """A zero limit must leave every existing profile untouched."""
    gate = ToppleGate()
    assert not gate.enabled
    for _ in range(100):
        assert not gate.record(3.14, 0.0)


def test_topple_gate_ignores_the_banked_riding_posture():
    """The ring rails are ridden at ~0.48 rad of roll all segment long."""
    gate = ToppleGate(limit_rad=1.0, consecutive_samples=25)
    for _ in range(500):
        assert not gate.record(-0.48, 0.05)
    assert gate.snapshot()['worst_attitude_rad'] == pytest.approx(0.48)


def test_topple_gate_needs_a_sustained_excursion():
    """One bad attitude sample must not stop the stage."""
    gate = ToppleGate(limit_rad=1.0, consecutive_samples=25)
    for _ in range(24):
        assert not gate.record(2.0, 0.0)
    assert not gate.record(-0.48, 0.0)      # back upright: streak resets
    for _ in range(24):
        assert not gate.record(2.0, 0.0)
    assert gate.record(2.0, 0.0)


def test_topple_gate_trips_on_the_measured_false_completion():
    """Replay of the 2026-08-16 run that walked the floor to P5_DONE.

    The corner-3 jump threw the body off the rail; it rolled through 3.14 rad
    for ~10 s, stood back up on the floor 1.16 m outside straight_3, and every
    later distance window and corner check passed.
    """
    gate = ToppleGate(limit_rad=1.0, consecutive_samples=25)
    tripped = False
    for roll in [-0.48] * 100 + [-1.05, -1.15] + [3.14] * 500 + [-0.5] * 100:
        tripped = gate.record(roll, 0.0) or tripped
    assert tripped
    # And it stays tripped after the robot stands up again, because standing up
    # is not evidence of standing up *on the course*.
    assert gate.snapshot()['tripped']
    assert not gate.record(-0.48, 0.0)


def test_topple_gate_treats_missing_attitude_as_no_evidence():
    """NaN must not read as upright, and must not fault either."""
    gate = ToppleGate(limit_rad=1.0, consecutive_samples=1)
    assert not gate.record(float('nan'), float('nan'))
    assert not gate.snapshot()['tripped']


def test_topple_gate_pitch_counts_too():
    """Nose-down over an edge is the same event as rolling off one."""
    gate = ToppleGate(limit_rad=1.0, consecutive_samples=2)
    assert not gate.record(0.0, 1.5)
    assert gate.record(0.0, 1.5)


def test_stall_gate_is_off_by_default():
    """A zero budget must leave every existing profile untouched."""
    gate = StallGate()
    assert not gate.enabled
    for i in range(100):
        assert not gate.record(0.45, 0.20, i * 0.1)


def test_stall_gate_ignores_a_robot_that_is_not_asked_to_move():
    """A stopped robot making no progress is not stalled."""
    gate = StallGate(min_speed=0.05, timeout_s=3.0)
    for i in range(200):
        assert not gate.record(0.0, 0.20, i * 0.1)


def test_stall_gate_ignores_a_robot_that_is_moving():
    """Progress resets the clock, so a walking robot never trips."""
    gate = StallGate(min_speed=0.05, timeout_s=3.0, min_progress_m=0.05)
    for i in range(200):
        assert not gate.record(0.45, i * 0.02, i * 0.1)


def test_stall_gate_trips_on_the_measured_ramp_stall():
    """Replay: vx 0.45 commanded, odometry pinned at 0.20 m."""
    gate = StallGate(min_speed=0.05, timeout_s=6.0, min_progress_m=0.05)
    tripped = [i * 0.1 for i in range(200)
               if gate.record(0.45, 0.20 + (i % 3) * 0.004, i * 0.1)]
    assert len(tripped) == 1                       # latches, fires once
    assert tripped[0] == pytest.approx(6.0, abs=0.15)


def test_stall_gate_tolerates_estimator_jitter():
    """A few mm of drift while standing must not read as progress."""
    gate = StallGate(min_speed=0.05, timeout_s=3.0, min_progress_m=0.05)
    fired = False
    for i in range(100):
        fired = gate.record(0.30, 1.50 + 0.004 * ((i % 5) - 2), i * 0.1) or fired
    assert fired


def test_stall_gate_reset_clears_the_latch_and_the_baseline():
    """A new segment gets a new budget and a new progress reference."""
    gate = StallGate(min_speed=0.05, timeout_s=1.0, min_progress_m=0.05)
    for i in range(40):
        gate.record(0.45, 0.20, i * 0.1)
    assert gate.snapshot()['tripped']
    gate.reset()
    assert not gate.snapshot()['tripped']
    assert not gate.record(0.45, 5.00, 10.0)
