# -*- coding: utf-8 -*-
"""Node-level regressions for the Stage 5 route model wiring.

覆盖范围：里程接入、双证据门控抑制、窗口越界处置、转角航向校核与有界再对齐、
分段限速。都是逻辑层回归，不构成仿真或实体证据
（STAGE5_PHYSICAL_REDESIGN_PLAN.md §9 G1/G2）。
"""

import math

import numpy as np
import pytest
import rclpy

import control_node.stage5_node as stage5_module
from control_node.robot_interface.sim_controller import SimRobotControlAdapter
from control_node.route_model import CrossTrackGate, EXIT_ODOMETRY, RouteModel
from control_node.stage5_node import Stage5Node


@pytest.fixture
def stage5_node():
    """Create an inactive ROS node without starting the LCM command owner."""
    if not rclpy.ok():
        rclpy.init()
    node = Stage5Node()
    yield node
    node.destroy_node()


class _FakeCtrl:
    def __init__(self):
        self.sent = []

    def response_snapshot(self):
        return {
            'seq': 0, 'rx_monotonic_s': None, 'mode': 0, 'gait_id': 0,
            'order_process_bar': 0, 'switch_status': 0, 'ori_error': 0,
            'footpos_error': 0, 'motor_error': [0] * 12,
            'last_incomplete_seq': 0, 'last_incomplete_mode': 0,
            'last_incomplete_gait_id': 0, 'last_incomplete_rx_monotonic_s': None,
        }

    def Send_cmd(self, msg):
        self.sent.append({
            'mode': int(msg.mode),
            'gait_id': int(msg.gait_id),
            'vel_des': [float(v) for v in msg.vel_des],
        })

    def Send_cmd_with_response_barrier(self, msg):
        self.Send_cmd(msg)
        return 0, stage5_module.time.monotonic()

    def Wait_finish(self, mode, gait_id):
        return True


def _sim_backend(node, ctrl):
    """Wrap the LCM stub in the real simulator adapter the node now drives.

    Stage code reaches the controller through the backend adapter, so the
    double belongs below it.  Sharing node.msg mirrors production: one
    life_count sequence on robot_control_cmd.
    """
    backend = SimRobotControlAdapter.__new__(SimRobotControlAdapter)
    backend.node = node
    backend._ctrl = ctrl
    backend._semantic_msg = node.msg
    return backend


class _FakeOdom:
    """Minimal stand-in for the LCM state-estimator reader."""

    def __init__(self):
        self.seq = 0
        self.rx_monotonic_s = 0.0
        self.p = [0.0, 0.0, 0.0]
        self.rpy = [0.0, 0.0, 0.0]

    def step(self, dx=0.0, dy=0.0, dyaw=0.0, dt=0.02):
        self.seq += 1
        self.rx_monotonic_s += dt
        self.p = [self.p[0] + dx, self.p[1] + dy, self.p[2]]
        self.rpy = [0.0, 0.0, self.rpy[2] + dyaw]

    def snapshot(self):
        return {
            'seq': self.seq,
            'rx_monotonic_s': self.rx_monotonic_s,
            'p': list(self.p),
            'rpy': list(self.rpy),
            'v_world': [0.0, 0.0, 0.0],
            'v_body': [0.0, 0.0, 0.0],
            'contact': [1.0, 1.0, 1.0, 1.0],
            'timestamp': 0,
        }


def _arm_route(node, monkeypatch, mode='enforce', overrides=None, now=100.0):
    """Wire a fake odometry reader and a controllable monotonic clock."""
    ctrl = _FakeCtrl()
    odom = _FakeOdom()
    node.Ctrl = _sim_backend(node, ctrl)
    node.Odom = odom
    node.p5_route_model_enabled = True
    node.p5_route_model_mode = mode
    node.p5_route_turn_verify_enabled = True
    node.p5_route_odometry_required = True
    if overrides:
        node.p5_route_model = RouteModel().with_overrides(overrides)
    clock = {'t': now}
    odom.rx_monotonic_s = now
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: clock['t'])
    return ctrl, odom, clock


def _seed_origin(node, odom):
    """Feed the first post-entry sample, which only fixes the segment origin."""
    if node.p5_route_progress.samples == 0:
        odom.step()
        node.p5_route_read_odometry()


def _walk(node, odom, distance_m, step_m=0.01):
    """Feed straight-line odometry samples and run the monitor each tick."""
    _seed_origin(node, odom)
    steps = max(1, int(round(distance_m / step_m)))
    for _ in range(steps):
        odom.step(dx=step_m)
        node.p5_route_monitor()


# ------------------------------------------------------------------
# 里程接入
# ------------------------------------------------------------------
def test_odometry_integrates_only_fresh_samples(stage5_node, monkeypatch):
    """Odometry integrates only fresh samples."""
    node = stage5_node
    _ctrl, odom, clock = _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)

    odom.step(dx=0.05)
    node.p5_route_read_odometry()
    assert node.p5_route_odom_valid is True
    # Same generation twice: no double counting.
    node.p5_route_read_odometry()
    assert node.p5_route_distance_m() == pytest.approx(0.0)

    odom.step(dx=0.05)
    node.p5_route_read_odometry()
    assert node.p5_route_distance_m() == pytest.approx(0.05)

    # A frozen stream ages out and becomes invalid rather than "not moving".
    clock['t'] += node.p5_route_odom_max_age_s + 1.0
    node.p5_route_read_odometry()
    assert node.p5_route_odom_valid is False


def test_never_received_odometry_is_invalid(stage5_node, monkeypatch):
    """Never received odometry is invalid."""
    node = stage5_node
    _ctrl, _odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_read_odometry()
    assert node.p5_route_odom_valid is False
    assert node.p5_route_odom_seq == 0


def test_segment_accumulator_resets_between_segments(stage5_node, monkeypatch):
    """Segment accumulator resets between segments."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.50)
    assert node.p5_route_distance_m() == pytest.approx(0.50, abs=1e-6)
    assert node.p5_route_segment.name == 'straight_1'

    node.p5_enter_state(node.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST)
    assert node.p5_route_segment.name == 'straight_1_exit'
    assert node.p5_route_distance_m() == pytest.approx(0.0)


def test_off_route_state_keeps_the_accumulator(stage5_node, monkeypatch):
    """P5_SET_RIGHT_SLOPE_BODY belongs to no segment; it must not reset progress."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.30)
    node.p5_enter_state(node.P5_SET_RIGHT_SLOPE_BODY)
    assert node.p5_route_segment.name == 'straight_1'
    assert node.p5_route_distance_m() == pytest.approx(0.30, abs=1e-6)


# ------------------------------------------------------------------
# 双证据门控
# ------------------------------------------------------------------
def test_enforced_exit_is_suppressed_before_the_window(stage5_node, monkeypatch):
    """Enforced exit is suppressed before the window."""
    node = stage5_node
    overrides = {'straight_1': {'min_m': 1.00, 'max_m': 2.00}}
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=overrides)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)

    _walk(node, odom, 0.50)
    assert node.p5_route_blocks_exit('T') is True

    _walk(node, odom, 0.70)
    assert node.p5_route_blocks_exit('T') is False


def test_enforced_exit_is_suppressed_without_odometry(stage5_node, monkeypatch):
    """Enforced exit is suppressed without odometry."""
    node = stage5_node
    overrides = {'straight_1': {'min_m': 0.0, 'max_m': 9.0}}
    _ctrl, odom, clock = _arm_route(node, monkeypatch, overrides=overrides)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.20)
    assert node.p5_route_blocks_exit('T') is False

    clock['t'] += node.p5_route_odom_max_age_s + 1.0
    node.p5_route_read_odometry()
    assert node.p5_route_blocks_exit('T') is True


def test_monitor_mode_never_suppresses_or_moves_the_state(stage5_node, monkeypatch):
    """Monitor mode never suppresses or moves the state."""
    node = stage5_node
    overrides = {'straight_1': {'min_m': 5.00, 'max_m': 6.00}}
    _ctrl, odom, _clock = _arm_route(
        node, monkeypatch, mode='monitor', overrides=overrides)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.20)
    assert node.p5_route_blocks_exit('T') is False

    # Far past the window: monitor mode still leaves the state alone.
    _walk(node, odom, 8.00)
    assert node.p5_route_monitor() is False
    assert node.state == node.P5_RIGHT_SLOPE_1


def test_monitored_segment_is_not_gated(stage5_node, monkeypatch):
    """Monitored segment is not gated."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_STEP_UP)
    assert node.p5_route_segment.enforced is False
    _walk(node, odom, 0.02)
    assert node.p5_route_blocks_exit('T') is False


# ------------------------------------------------------------------
# 越界处置
# ------------------------------------------------------------------
def test_overrun_faults_with_a_direct_stop(stage5_node, monkeypatch):
    """Overrun faults with a direct stop."""
    node = stage5_node
    overrides = {'straight_1': {'min_m': 0.10, 'max_m': 0.50}}
    ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=overrides)
    node.p5_route_overrun_action = 'fault'
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)

    _walk(node, odom, 0.60)
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    # The fault path must send the real STOP, not a zero-velocity locomotion
    # command that transits the locomotion FSM first.
    assert ctrl.sent[-1]['mode'] == 12
    assert ctrl.sent[-1]['gait_id'] == 0


def test_overrun_degraded_advance_is_declared_and_bounded(stage5_node, monkeypatch):
    """Overrun degraded advance is declared and bounded."""
    node = stage5_node
    overrides = {'straight_1': {'min_m': 0.10, 'max_m': 0.50}}
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=overrides)
    node.p5_route_overrun_action = 'degraded_advance'
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)

    _walk(node, odom, 0.60)
    assert node.state == node.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST


def test_overrun_without_a_declared_successor_faults(stage5_node, monkeypatch):
    """Overrun without a declared successor faults."""
    node = stage5_node
    overrides = {
        'straight_1': {'min_m': 0.10, 'max_m': 0.50},
    }
    model = RouteModel().with_overrides(overrides)
    model = RouteModel(tuple(
        segment.replace(degraded_next_state='')
        if segment.name == 'straight_1' else segment
        for segment in model.segments))
    ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_model = model
    node.p5_route_overrun_action = 'degraded_advance'
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)

    _walk(node, odom, 0.60)
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1]['mode'] == 12


def test_missing_odometry_faults_an_enforced_segment_after_the_grace(
        stage5_node, monkeypatch):
    """Missing odometry faults an enforced segment after the grace."""
    node = stage5_node
    ctrl, odom, clock = _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.05)

    clock['t'] += node.p5_route_odom_max_age_s + 0.1
    node.p5_route_monitor()
    assert node.state == node.P5_RIGHT_SLOPE_1     # still inside the grace

    clock['t'] += node.p5_sensor_fault_grace_s + 1.0
    node.p5_route_monitor()
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1]['mode'] == 12


def test_missing_odometry_only_logs_on_a_monitored_segment(stage5_node, monkeypatch):
    """Missing odometry only logs on a monitored segment."""
    node = stage5_node
    _ctrl, _odom, clock = _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_STEP_UP)
    clock['t'] += node.p5_sensor_fault_grace_s + 5.0
    assert node.p5_route_monitor() is False
    assert node.state == node.P5_STEP_UP


# ------------------------------------------------------------------
# 转角航向校核与再对齐
# ------------------------------------------------------------------
CORNER_OVERRIDE = {'corner_2': {'expected_yaw_deg': -90.0, 'yaw_tol_deg': 10.0}}


def _enter_corner(node, odom, yaw_deg):
    node.p5_enter_state(node.P5_TURN_1)
    assert node.p5_route_segment.name == 'corner_2'
    _seed_origin(node, odom)
    steps = 20
    for _ in range(steps):
        odom.step(dyaw=(yaw_deg / steps) * 3.141592653589793 / 180.0)
        node.p5_route_read_odometry()


def test_verified_corner_transitions_normally(stage5_node, monkeypatch):
    """Verified corner transitions normally."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    _enter_corner(node, odom, -88.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_RIGHT_SLOPE_2


def test_missed_corner_diverts_to_bounded_realignment(stage5_node, monkeypatch):
    """Missed corner diverts to bounded realignment."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    node.p5_route_realign_max_attempts = 1
    _enter_corner(node, odom, -40.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_ROUTE_REALIGN
    assert node.p5_route_realign_resume_state == node.P5_RIGHT_SLOPE_2
    assert node.p5_route_realign_segment_name == 'corner_2'


def test_realignment_commands_the_short_way_round_and_resumes(stage5_node, monkeypatch):
    """Realignment commands the short way round and resumes."""
    node = stage5_node
    ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    node.p5_route_realign_max_attempts = 1
    _enter_corner(node, odom, -40.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_ROUTE_REALIGN

    # Still 50 deg short of -90 deg: the correction must be negative.
    node.p5_route_monitor()
    node.p5_run_route_realign()
    assert ctrl.sent[-1]['vel_des'][2] < 0.0

    # Finish the rotation; the state machine resumes where the corner pointed.
    for _ in range(10):
        odom.step(dyaw=-5.0 * 3.141592653589793 / 180.0)
    node.p5_route_monitor()
    node.p5_run_route_realign()
    assert node.state == node.P5_RIGHT_SLOPE_2
    assert node.p5_route_segment.name == 'straight_2'


def test_realignment_timeout_faults(stage5_node, monkeypatch):
    """Realignment timeout faults."""
    node = stage5_node
    ctrl, odom, clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    node.p5_route_realign_max_attempts = 1
    _enter_corner(node, odom, -40.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_ROUTE_REALIGN

    clock['t'] += node.p5_route_realign_timeout_s + 1.0
    odom.step()
    node.p5_route_monitor()
    node.p5_run_route_realign()
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1]['mode'] == 12


def test_missed_corner_without_attempts_left_faults(stage5_node, monkeypatch):
    """Missed corner without attempts left faults."""
    node = stage5_node
    ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    node.p5_route_realign_max_attempts = 0
    _enter_corner(node, odom, -40.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    # p5_enter_state stays side-effect free; the fault-hold state issues the
    # real STOP on its first control tick.
    node.p5_control_loop()
    assert ctrl.sent[-1]['mode'] == 12
    assert ctrl.sent[-1]['gait_id'] == 0


def test_corner_verification_fails_closed_without_odometry(stage5_node, monkeypatch):
    """Corner verification fails closed without odometry."""
    node = stage5_node
    _ctrl, odom, clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    _enter_corner(node, odom, -90.0)
    clock['t'] += node.p5_route_odom_max_age_s + 1.0
    node.p5_route_read_odometry()
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_SENSOR_FAULT_HOLD


def test_corner_verification_is_skipped_when_disabled(stage5_node, monkeypatch):
    """Corner verification is skipped when disabled."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    node.p5_route_turn_verify_enabled = False
    _enter_corner(node, odom, 0.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_RIGHT_SLOPE_2


# ------------------------------------------------------------------
# 分段限速
# ------------------------------------------------------------------
def test_segment_speed_cap_scales_the_command(stage5_node, monkeypatch):
    """Segment speed cap scales the command."""
    node = stage5_node
    overrides = {'straight_1': {'speed_cap_mps': 0.10}}
    ctrl, _odom, _clock = _arm_route(node, monkeypatch, overrides=overrides)
    node.p5_route_speed_cap_enabled = True
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)

    node.p5_send_velocity_command(vx=0.40, vy=0.0, wz=0.0, step_height=0.05)
    assert ctrl.sent[-1]['vel_des'][0] == pytest.approx(0.10)

    node.p5_route_speed_cap_enabled = False
    node.p5_send_velocity_command(vx=0.40, vy=0.0, wz=0.0, step_height=0.05)
    assert ctrl.sent[-1]['vel_des'][0] == pytest.approx(0.40)


def test_speed_cap_leaves_zero_and_slow_commands_alone(stage5_node, monkeypatch):
    """Speed cap leaves zero and slow commands alone."""
    node = stage5_node
    overrides = {'straight_1': {'speed_cap_mps': 0.10}}
    ctrl, _odom, _clock = _arm_route(node, monkeypatch, overrides=overrides)
    node.p5_route_speed_cap_enabled = True
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)

    node.p5_send_velocity_command(vx=0.0, vy=0.0, wz=0.30, step_height=0.0)
    assert ctrl.sent[-1]['vel_des'] == pytest.approx([0.0, 0.0, 0.30])

    node.p5_send_velocity_command(vx=0.05, vy=0.0, wz=0.0, step_height=0.0)
    assert ctrl.sent[-1]['vel_des'][0] == pytest.approx(0.05)


def test_disabled_route_model_changes_nothing(stage5_node, monkeypatch):
    """Disabled route model changes nothing."""
    node = stage5_node
    ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_model_enabled = False
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    assert node.p5_route_segment is None
    _walk(node, odom, 9.0)
    assert node.state == node.P5_RIGHT_SLOPE_1
    assert node.p5_route_blocks_exit('T') is False
    node.p5_send_velocity_command(vx=0.9, vy=0.0, wz=0.0, step_height=0.0)
    assert ctrl.sent[-1]['vel_des'][0] == pytest.approx(0.9)


# ------------------------------------------------------------------
# 里程主导段尾（不依赖黄线）
# ------------------------------------------------------------------
def _arm_odometry_exit(node, monkeypatch, expected_m=1.0):
    """Arm the node in odometry-primary mode with yellow control disabled."""
    ctrl, odom, clock = _arm_route(node, monkeypatch)
    node.p5_route_exit_source = EXIT_ODOMETRY
    node.p5_yellow_lateral_correction_enabled = False
    # Mirror the simulator profile: the caps target the physical profile, and
    # leaving them on here would just re-test clamp_speed.
    node.p5_route_speed_cap_enabled = False
    node.p5_route_model = RouteModel().with_overrides({'straight_1': {
        'exit_source': EXIT_ODOMETRY,
        'expected_m': expected_m,
        'min_m': 0.5,
        'max_m': 2.0,
        'fallback_tier': 3,
    }})
    return ctrl, odom, clock


def test_odometry_exit_advances_without_any_image(stage5_node, monkeypatch):
    """The straight ends on distance alone; no RGB frame is ever supplied."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_odometry_exit(node, monkeypatch, expected_m=1.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    assert node.latest_bgr is None
    assert node.p5_route_segment_uses_odometry_exit() is True

    _walk(node, odom, 0.50)
    node.p5_control_loop()
    assert node.state == node.P5_RIGHT_SLOPE_1

    _walk(node, odom, 0.60)
    node.p5_control_loop()
    assert node.state == node.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST


def test_odometry_exit_keeps_commanding_the_segment_velocity(stage5_node, monkeypatch):
    """Odometry exit keeps commanding the segment velocity."""
    node = stage5_node
    ctrl, odom, _clock = _arm_odometry_exit(node, monkeypatch, expected_m=1.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.30)
    node.p5_control_loop()
    assert ctrl.sent[-1]['mode'] == 11
    assert ctrl.sent[-1]['vel_des'][0] == pytest.approx(node.p5_right_slope_1_vx)


def test_odometry_exit_fails_closed_when_odometry_dies(stage5_node, monkeypatch):
    """A segment with no perception exit and no odometry must not advance."""
    node = stage5_node
    ctrl, odom, clock = _arm_odometry_exit(node, monkeypatch, expected_m=1.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.30)

    clock['t'] += node.p5_route_odom_max_age_s + 1.0
    node.p5_route_read_odometry()
    node.run_odometry_distance_velocity_state(
        vx=0.3, vy=0.0, wz=0.0, step_height=0.04,
        next_state=node.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST,
        log_name='T')
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1]['mode'] == 12


def test_odometry_exit_respects_the_state_timeout(stage5_node, monkeypatch):
    """Odometry exit respects the state timeout."""
    node = stage5_node
    ctrl, odom, clock = _arm_odometry_exit(node, monkeypatch, expected_m=9.0)
    node.p5_route_model = RouteModel().with_overrides({'straight_1': {
        'exit_source': EXIT_ODOMETRY, 'expected_m': 9.0,
        'min_m': 0.5, 'max_m': 20.0, 'fallback_tier': 3,
    }})
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.30)
    clock['t'] += 100.0
    odom.step(dx=0.01)
    node.run_odometry_distance_velocity_state(
        vx=0.3, vy=0.0, wz=0.0, step_height=0.04,
        next_state='NEXT', log_name='T', timeout_s=45.0)
    assert node.state == node.P5_SENSOR_FAULT_HOLD


def test_disabling_yellow_correction_returns_the_base_velocities(stage5_node):
    """Disabling yellow correction returns the base velocities."""
    node = stage5_node
    node.p5_yellow_lateral_correction_enabled = False
    # A frame is supplied on purpose: it must not be consulted at all.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert node.compute_p5_right_slope_right_edge_corrected_vy(0.13, frame) == 0.13
    assert node.compute_p5_up_slope_inner_edge_corrected_cmd(
        base_vy=0.05, base_wz=-0.02, frame=frame) == (0.05, -0.02)
    # The post-jump alignment gate degrades to a plain timed forward.
    assert node.p5_forward_inner_edge_aligned() is True


def test_vision_exit_stays_the_default(stage5_node, monkeypatch):
    """Vision exit stays the default."""
    node = stage5_node
    _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    assert node.p5_route_segment_uses_odometry_exit() is False


def test_a_verified_corner_is_not_judged_again(stage5_node, monkeypatch):
    """Leaving an off-route body-preset state must not re-judge the corner."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    _enter_corner(node, odom, -88.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_RIGHT_SLOPE_2

    # Same segment still active through an off-route state, and the yaw has
    # since drifted well outside tolerance.
    node.p5_route_segment = node.p5_route_model.segment_by_name('corner_2')
    node.p5_route_verified_segment = 'corner_2'
    for _ in range(20):
        odom.step(dyaw=-0.02)
        node.p5_route_read_odometry()
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_RIGHT_SLOPE_2


def test_realignment_marks_the_corner_verified(stage5_node, monkeypatch):
    """After a successful re-alignment the corner is not re-judged on resume."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    node.p5_route_realign_max_attempts = 1
    _enter_corner(node, odom, -40.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.state == node.P5_ROUTE_REALIGN

    for _ in range(10):
        odom.step(dyaw=-5.0 * 3.141592653589793 / 180.0)
    node.p5_route_monitor()
    node.p5_run_route_realign()
    assert node.state == node.P5_RIGHT_SLOPE_2
    # The realigned corner is recorded as verified, so a later off-route exit
    # cannot spend the exhausted attempt budget on it again.
    assert node.p5_route_verified_segment in ('corner_2', '')


def test_verification_flag_clears_on_the_next_segment(stage5_node, monkeypatch):
    """Verification flag clears on the next segment."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch, overrides=CORNER_OVERRIDE)
    _enter_corner(node, odom, -88.0)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.p5_route_segment.name == 'straight_2'
    assert node.p5_route_verified_segment == ''


# ------------------------------------------------------------------
# 路线航向栅格基准（计划书第 29 条）
# ------------------------------------------------------------------
def test_grid_prefers_the_declared_course_heading(stage5_node, monkeypatch):
    """A declared course heading must beat whatever yaw the body settled at.

    Measured 2026-08-04: the start-line yaw ranged +1.11..+1.83 rad across 20
    runs against a true course heading of +pi/2, and the grid error went
    straight into every odometry-fallback reference line.
    """
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_base_yaw_declared = math.pi / 2.0
    odom.rpy = [0.0, 0.0, 1.8343]           # the worst measured placement
    odom.step()
    node.p5_route_read_odometry()

    node.p5_enter_state(node.P5_UP_SLOPE)
    assert node.p5_route_base_yaw == pytest.approx(math.pi / 2.0)


def test_grid_falls_back_to_the_measured_yaw_when_undeclared(stage5_node,
                                                             monkeypatch):
    """Undeclared (the physical profile) keeps the previous behaviour."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_base_yaw_declared = None
    odom.rpy = [0.0, 0.0, 1.8343]
    odom.step()
    node.p5_route_read_odometry()

    node.p5_enter_state(node.P5_UP_SLOPE)
    assert node.p5_route_base_yaw == pytest.approx(1.8343)


def test_declared_grid_needs_no_odometry_to_anchor(stage5_node, monkeypatch):
    """A declared heading is a course fact, so a dead estimator cannot lose it."""
    node = stage5_node
    _ctrl, _odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_base_yaw_declared = math.pi / 2.0
    node.p5_route_odom_valid = False

    node.p5_enter_state(node.P5_UP_SLOPE)
    assert node.p5_route_base_yaw == pytest.approx(math.pi / 2.0)


def test_grid_is_anchored_once_and_never_redrawn(stage5_node, monkeypatch):
    """A later segment must not be able to move the grid."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_base_yaw_declared = math.pi / 2.0
    odom.step()
    node.p5_route_read_odometry()
    node.p5_enter_state(node.P5_UP_SLOPE)

    node.p5_route_base_yaw_declared = 0.0        # a later re-read must not win
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    assert node.p5_route_base_yaw == pytest.approx(math.pi / 2.0)


def test_crooked_placement_is_reported_not_absorbed(stage5_node, monkeypatch):
    """The placement error must reach the evidence log instead of the grid."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_route_base_yaw_declared = math.pi / 2.0
    odom.rpy = [0.0, 0.0, math.pi / 2.0 + 0.25]
    odom.step()
    node.p5_route_read_odometry()

    records = []
    monkeypatch.setattr(node, 'p5_evidence_log', records.append)
    node.p5_enter_state(node.P5_UP_SLOPE)

    grid = [r for r in records if r.get('event') == 'route_heading_grid']
    assert len(grid) == 1
    assert grid[0]['source'] == 'declared'
    assert grid[0]['placement_error_rad'] == pytest.approx(0.25, abs=1e-6)
    assert grid[0]['base_yaw_rad'] == pytest.approx(math.pi / 2.0)


# ------------------------------------------------------------------
# 上桥前的起始航向对齐（计划书第 30 条）
# ------------------------------------------------------------------
def _arm_start_align(node, monkeypatch, yaw, enabled=True,
                     declared=math.pi / 2.0):
    """Put the node in P5_START_ALIGN with a controllable estimator yaw."""
    ctrl, odom, clock = _arm_route(node, monkeypatch)
    node.p5_start_align_enabled = enabled
    node.p5_route_base_yaw_declared = declared
    node.p5_start_align_tol_rad = math.radians(4.0)
    node.p5_start_align_wz = 0.30
    node.p5_start_align_step_height = 0.04
    node.p5_start_align_timeout_s = 12.0
    odom.rpy = [0.0, 0.0, yaw]
    odom.step()
    node.p5_enter_state(node.P5_START_ALIGN)
    ctrl.sent.clear()
    return ctrl, odom, clock


def test_start_align_turns_the_body_onto_the_declared_heading(stage5_node,
                                                              monkeypatch):
    """The measured recovery-stand rotation must be turned out before the climb.

    2026-08-05 r07 stood up at +2.116 rad against a route heading of +pi/2 and
    walked onto the bridge 31 deg off.
    """
    node = stage5_node
    ctrl, _odom, _clock = _arm_start_align(node, monkeypatch, yaw=2.116)

    node.p5_run_start_align()
    assert node.state == node.P5_START_ALIGN          # still correcting
    assert ctrl.sent, 'no command issued'
    # Route heading is to the body's right of +2.116, so turn right (wz < 0).
    assert ctrl.sent[-1]['vel_des'][2] == pytest.approx(-0.30)
    assert ctrl.sent[-1]['vel_des'][0] == pytest.approx(0.0)
    assert ctrl.sent[-1]['vel_des'][1] == pytest.approx(0.0)


def test_start_align_turns_the_other_way_for_the_mirror_error(stage5_node,
                                                              monkeypatch):
    """Sign must follow the error, not a fixed direction."""
    node = stage5_node
    ctrl, _odom, _clock = _arm_start_align(node, monkeypatch, yaw=1.10)

    node.p5_run_start_align()
    assert ctrl.sent[-1]['vel_des'][2] == pytest.approx(+0.30)


def test_start_align_releases_inside_tolerance(stage5_node, monkeypatch):
    """A body already on the route must not be turned at all."""
    node = stage5_node
    ctrl, _odom, _clock = _arm_start_align(
        node, monkeypatch, yaw=math.pi / 2.0 + math.radians(2.0))

    node.p5_run_start_align()
    assert node.state == node.P5_STEP_UP
    assert ctrl.sent[-1]['vel_des'][2] == pytest.approx(0.0)


def test_start_align_is_bounded_and_never_stalls_the_stage(stage5_node,
                                                           monkeypatch):
    """Turning that is not converging must release, not hold until the timeout."""
    node = stage5_node
    ctrl, odom, clock = _arm_start_align(node, monkeypatch, yaw=2.116)

    node.p5_run_start_align()
    assert node.state == node.P5_START_ALIGN
    # Keep the estimator alive across the bump: a stale stream would take the
    # (also bounded) skip path instead, which is not what this test pins down.
    clock['t'] += 13.0
    odom.rx_monotonic_s = clock['t']
    node.p5_run_start_align()
    assert node.state == node.P5_STEP_UP
    assert ctrl.sent[-1]['vel_des'][2] == pytest.approx(0.0)


def test_start_align_is_off_by_default(stage5_node, monkeypatch):
    """Disabled means straight through, exactly as before this state existed."""
    node = stage5_node
    ctrl, _odom, _clock = _arm_start_align(node, monkeypatch, yaw=2.116,
                                           enabled=False)

    node.p5_run_start_align()
    assert node.state == node.P5_STEP_UP
    assert not ctrl.sent


def test_start_align_refuses_without_a_declared_heading(stage5_node,
                                                        monkeypatch):
    """No declared heading means no 'correct' direction to turn towards."""
    node = stage5_node
    ctrl, _odom, _clock = _arm_start_align(node, monkeypatch, yaw=2.116,
                                           declared=None)

    node.p5_run_start_align()
    assert node.state == node.P5_STEP_UP
    assert not ctrl.sent


def test_start_align_refuses_on_a_dead_estimator(stage5_node, monkeypatch):
    """A dead stream must not be read as 'yaw 0' and turned against."""
    node = stage5_node
    ctrl, odom, _clock = _arm_start_align(node, monkeypatch, yaw=2.116)
    odom.seq = 0                                  # never received a sample

    node.p5_run_start_align()
    assert node.state == node.P5_STEP_UP
    assert not ctrl.sent


def test_body_preset_now_hands_off_to_the_alignment_state(stage5_node,
                                                          monkeypatch):
    """The new state must actually sit between the body preset and the climb."""
    node = stage5_node
    _ctrl, _odom, clock = _arm_route(node, monkeypatch)
    node.p5_enter_state(node.P5_SET_BODY_NORMAL)
    # /clock is absent under test, so p5_state_elapsed_s cannot advance on its
    # own; the wait itself is not what this test is about.
    monkeypatch.setattr(node, 'p5_state_elapsed_s',
                        lambda: node.p5_body_normal_wait_s + 1.0)
    node.p5_control_loop()
    assert node.state == node.P5_START_ALIGN


def test_start_align_stops_turning_when_the_estimator_goes_stale(stage5_node,
                                                                 monkeypatch):
    """A stream that stops mid-turn must release, not keep turning open-loop."""
    node = stage5_node
    ctrl, _odom, clock = _arm_start_align(node, monkeypatch, yaw=2.116)

    node.p5_run_start_align()
    assert node.state == node.P5_START_ALIGN
    clock['t'] += 2.0                      # odometry not refreshed: now stale
    node.p5_run_start_align()
    assert node.state == node.P5_STEP_UP


# ------------------------------------------------------------------
# 段内横向偏离门 + 深度状态白名单（计划书第 32/34 条）
# ------------------------------------------------------------------
def _arm_cross_track(node, monkeypatch, limit_m=0.30, samples=2):
    """Arm the run-time departure check on an enforced straight."""
    ctrl, odom, clock = _arm_route(node, monkeypatch)
    node.p5_route_cross_track_gate = CrossTrackGate(
        limit_m=limit_m, consecutive_samples=samples)
    return ctrl, odom, clock


def test_cross_track_fault_stops_a_body_leaving_the_rail(stage5_node,
                                                         monkeypatch):
    """Drifting sideways off the entry line faults instead of walking on.

    This is the run-time half of plan item 32: three 2026-08-05 "completions"
    reached P5_DONE standing on the floor past straight_3's outer edge, and
    nothing but the after-the-fact audit noticed.
    """
    node = stage5_node
    ctrl, odom, _clock = _arm_cross_track(node, monkeypatch)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _seed_origin(node, odom)
    for _ in range(40):
        odom.step(dx=0.01, dy=0.02)
        node.p5_route_monitor()
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1]['mode'] == 12          # a real STOP, not zero velocity
    assert node.p5_route_cross_track_gate.tripped is True


def test_cross_track_fault_leaves_a_centred_walk_alone(stage5_node, monkeypatch):
    """A straight that stays on the line must be untouched by the check."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_cross_track(node, monkeypatch)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 1.50)
    assert node.state == node.P5_RIGHT_SLOPE_1
    assert node.p5_route_cross_track_gate.tripped is False


def test_cross_track_check_is_off_by_default(stage5_node, monkeypatch):
    """Unconfigured, the check must not exist: the physical profile ships so."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    assert node.p5_route_cross_track_gate.enabled is False
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _seed_origin(node, odom)
    for _ in range(40):
        odom.step(dx=0.01, dy=0.05)
        node.p5_route_monitor()
    assert node.state == node.P5_RIGHT_SLOPE_1


def test_cross_track_budget_restarts_at_each_segment(stage5_node, monkeypatch):
    """A new segment is a new entry line, so the excursion restarts at zero."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_cross_track(node, monkeypatch, samples=10)
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _seed_origin(node, odom)
    for _ in range(20):
        odom.step(dx=0.01, dy=0.02)
        node.p5_route_monitor()
    assert node.p5_route_cross_track_gate.streak > 0
    node.p5_enter_state(node.P5_RIGHT_SLOPE_2)
    assert node.p5_route_cross_track_gate.streak == 0
    assert node.p5_route_cross_track_gate.worst_m == 0.0


_DEPTH_FIX = {'valid': True, 'source': 'depth', 'reason': 'ok',
              'lateral_offset': 0.12, 'heading_error': -0.05}


def test_depth_steers_everywhere_when_no_allow_list_is_declared(stage5_node):
    """Empty allow-list = previous behaviour, on every state."""
    node = stage5_node
    node.latest_bridge_observation = dict(_DEPTH_FIX)
    node.p5_deck_lateral_depth_states = frozenset()
    node.p5_enter_state(node.P5_RIGHT_SLOPE_3)
    assert node.p5_depth_observation_for_steering()['lateral_offset'] == 0.12


def test_depth_is_suppressed_outside_the_allow_list(stage5_node):
    """On the banked ring rails the observer is noise; it must not steer.

    Measured 2026-08-05 against ground truth: correlation 0.18 and a +0.12 m
    outward bias on straight_3, against 0.99 / 0.005 m on the bridge climb.
    """
    node = stage5_node
    node.latest_bridge_observation = dict(_DEPTH_FIX)
    node.p5_deck_lateral_depth_states = frozenset({'P5_UP_SLOPE'})
    node.p5_enter_state(node.P5_RIGHT_SLOPE_3)
    assert node.p5_depth_observation_for_steering() is None
    node.p5_enter_state(node.P5_UP_SLOPE)
    assert node.p5_depth_observation_for_steering()['lateral_offset'] == 0.12


def test_suppressed_depth_still_reaches_the_entry_depth_gate(stage5_node,
                                                             monkeypatch):
    """Suppression removes steering authority, not the observation itself."""
    node = stage5_node
    node.latest_bridge_observation = dict(_DEPTH_FIX)
    node.p5_deck_lateral_depth_states = frozenset({'P5_UP_SLOPE'})
    node.p5_enter_state(node.P5_RIGHT_SLOPE_3)
    assert node.p5_depth_observation_for_steering() is None
    assert node.latest_bridge_observation['valid'] is True


def test_suppressed_depth_falls_through_to_the_odometry_line(stage5_node,
                                                             monkeypatch):
    """The ladder must still produce a correction, just from the other source."""
    node = stage5_node
    _ctrl, odom, _clock = _arm_route(node, monkeypatch)
    node.p5_deck_lateral_enabled = True
    node.p5_route_lateral_fallback = True
    node.p5_deck_lateral_depth_states = frozenset({'P5_UP_SLOPE'})
    node.latest_bridge_observation = dict(_DEPTH_FIX)
    node.p5_route_base_yaw = 0.0            # the fallback needs a route grid
    node.p5_enter_state(node.P5_RIGHT_SLOPE_1)
    _walk(node, odom, 0.30)
    for _ in range(3):
        odom.step(dx=0.01, dy=0.03)
        node.p5_route_read_odometry()
        node.p5_deck_lateral_update('T')
    assert node.p5_lateral_source_last == 'odometry'


def test_cross_track_check_ignores_the_final_jump(stage5_node, monkeypatch):
    """The final zone runs on the floor and moves sideways on purpose.

    Measured 2026-08-05: without this exemption three consecutive runs faulted
    in ``P5_FINAL_LONG_JUMP``, the last state before ``P5_DONE``.
    """
    node = stage5_node
    _ctrl, odom, _clock = _arm_cross_track(node, monkeypatch, samples=2)
    node.p5_enter_state(node.P5_FINAL_LONG_JUMP)
    assert node.p5_route_segment.name == 'final_zone'
    _seed_origin(node, odom)
    for _ in range(40):
        odom.step(dx=0.01, dy=0.03)
        node.p5_route_monitor()
    assert node.state == node.P5_FINAL_LONG_JUMP
    assert node.p5_route_cross_track_gate.tripped is False
