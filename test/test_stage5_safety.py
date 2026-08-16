# -*- coding: utf-8 -*-
"""Stage 5 fail-safe regressions for clocks, decoded frames and action polling."""

import importlib.util
import time as std_time
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import pytest
import rclpy
import yaml
from launch import LaunchContext
from launch.utilities import perform_substitutions
from sensor_msgs.msg import Image

import control_node.stage5_node as stage5_module
import control_node.stage_common as common_module
from control_node.my_gait import Robot_Ctrl
from control_node.robot_interface.sim_controller import SimRobotControlAdapter
from control_node.stage5_node import Stage5Node
from test_bridge_perception import render_scene


@pytest.fixture
def stage5_node():
    """Create an inactive ROS node without starting the LCM command owner."""
    if not rclpy.ok():
        rclpy.init()
    node = Stage5Node()
    yield node
    node.destroy_node()


class _BridgeResult:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def imgmsg_to_cv2(self, msg, desired_encoding=None):
        if self.error is not None:
            raise self.error
        return self.result


class _FakeCtrl:
    def __init__(self):
        self.snapshot = {
            'seq': 0,
            'rx_monotonic_s': None,
            'mode': 0,
            'gait_id': 0,
            'order_process_bar': 0,
            'switch_status': 0,
            'ori_error': 0,
            'footpos_error': 0,
            'motor_error': [0] * 12,
            'last_incomplete_seq': 0,
            'last_incomplete_mode': 0,
            'last_incomplete_gait_id': 0,
            'last_incomplete_rx_monotonic_s': None,
        }
        self.sent = []

    def response_snapshot(self):
        return dict(self.snapshot)

    def Send_cmd(self, msg):
        self.sent.append((int(msg.mode), int(msg.gait_id), int(msg.life_count)))

    def Send_cmd_with_response_barrier(self, msg):
        baseline = int(self.snapshot['seq'])
        self.Send_cmd(msg)
        return baseline, stage5_module.time.monotonic()

    def Wait_finish(self, mode, gait_id):
        return True


def _sim_backend(node, ctrl):
    """Wrap the LCM stub in the real simulator adapter the node now drives.

    Stage code no longer calls Send_cmd directly for velocity/stop commands,
    so the double has to sit *below* the adapter to keep testing the
    translation the robot actually receives.  Sharing node.msg mirrors
    production: one life_count sequence on robot_control_cmd.
    """
    backend = SimRobotControlAdapter.__new__(SimRobotControlAdapter)
    backend.node = node
    backend._ctrl = ctrl
    backend._semantic_msg = node.msg
    return backend


def test_rgb_freshness_requires_successful_decode(stage5_node, monkeypatch):
    """A received-but-undecodable image cannot refresh the watchdog."""
    node = stage5_node
    node.active = True
    msg = Image()
    node.bridge = _BridgeResult(error=ValueError('bad frame'))
    node.rgb_callback(msg)
    assert node.latest_rgb_seq == 0
    assert node.rgb_age_s() is None

    monkeypatch.setattr(common_module.time, 'monotonic', lambda: 100.0)
    node.bridge = _BridgeResult(result=np.zeros((480, 640, 3), dtype=np.uint8))
    node.rgb_callback(msg)
    assert node.latest_rgb_seq == 1
    assert node.latest_frame_seq == 1
    assert node.rgb_age_s() == pytest.approx(0.0)


def test_depth_freshness_requires_successful_decode(stage5_node, monkeypatch):
    """Depth conversion failures leave both sequence and freshness unchanged."""
    node = stage5_node
    node.active = True
    node.p5_bridge_observer_enabled = False
    msg = Image()
    msg.encoding = '32FC1'
    node.bridge = _BridgeResult(error=ValueError('bad depth'))
    node.depth_callback(msg)
    assert node.latest_depth_seq == 0
    assert node.depth_age_s() is None

    monkeypatch.setattr(common_module.time, 'monotonic', lambda: 200.0)
    node.bridge = _BridgeResult(result=np.ones((2, 2), dtype=np.float32))
    node.depth_callback(msg)
    assert node.latest_depth_seq == 1
    assert node.depth_age_s() == pytest.approx(0.0)


def test_watchdog_uses_monotonic_when_ros_clock_is_frozen(stage5_node, monkeypatch):
    """Safety timeout advances even while use_sim_time has no /clock updates."""
    node = stage5_node
    node.state = node.P5_UP_SLOPE
    node.state_start_monotonic_s = 100.0
    node.last_rgb_rx_time_s = 100.0
    node.p5_sensor_fault_grace_s = 0.0
    node.p5_sensor_max_frame_age_s = 1.0
    sent = []
    entered = []
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 102.0)
    monkeypatch.setattr(common_module.time, 'monotonic', lambda: 102.0)
    monkeypatch.setattr(node, 'p5_send_velocity_command', lambda **kwargs: sent.append(kwargs))
    monkeypatch.setattr(node, 'p5_enter_state', lambda state: entered.append(state))

    assert node.p5_vision_state_guard(0.0, 'TEST') is True
    assert sent[-1]['vx'] == 0.0
    assert entered == [node.P5_SENSOR_FAULT_HOLD]


def test_up_slope_no_image_obeys_fail_safe_profile(stage5_node, monkeypatch):
    """Physical profile holds zero before a valid new state-local RGB frame."""
    node = stage5_node
    node.state = node.P5_UP_SLOPE
    node.state_enter_frame_seq = node.latest_frame_seq
    node.latest_bgr = None
    node.p5_keep_moving_when_no_image = False
    node.p5_sensor_watchdog_enabled = False
    sent = []
    monkeypatch.setattr(node, 'p5_state_elapsed_s', lambda: 1.0)
    monkeypatch.setattr(node, 'p5_send_velocity_command', lambda **kwargs: sent.append(kwargs))

    node.run_right_side_yellow_lost_velocity_state(
        vx=0.2, vy=0.1, wz=0.05, step_height=0.03,
        next_state='NEXT', timeout_s=0.0,
    )
    assert sent[-1]['vx'] == 0.0
    assert sent[-1]['vy'] == 0.0
    assert sent[-1]['wz'] == 0.0


def test_after_up_slope_right_jump_uses_bounded_action_state(
        stage5_node, monkeypatch):
    """Simulation turn method routes the corner through the action handshake."""
    node = stage5_node
    node.state = node.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL
    node.p5_after_up_slope_turn_method = 'right_jump'
    node.p5_after_up_slope_turn_jump_mode = 16
    node.p5_after_up_slope_turn_jump_gait = 0
    calls = []
    monkeypatch.setattr(
        node, 'run_action_state',
        lambda **kwargs: calls.append(('action', kwargs)))
    monkeypatch.setattr(
        node, 'run_timed_velocity_state',
        lambda **kwargs: calls.append(('velocity', kwargs)))

    node.p5_control_loop()

    assert len(calls) == 1
    assert calls[0][0] == 'action'
    assert calls[0][1]['mode'] == 16
    assert calls[0][1]['gait_id'] == 0
    assert calls[0][1]['stop_after_finish'] is True


def test_after_up_slope_velocity_compatibility_path(stage5_node, monkeypatch):
    """Physical calibration profile can retain the legacy timed turn."""
    node = stage5_node
    node.state = node.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL
    node.p5_after_up_slope_turn_method = 'velocity'
    calls = []
    monkeypatch.setattr(
        node, 'run_action_state',
        lambda **kwargs: calls.append(('action', kwargs)))
    monkeypatch.setattr(
        node, 'run_timed_velocity_state',
        lambda **kwargs: calls.append(('velocity', kwargs)))

    node.p5_control_loop()

    assert len(calls) == 1
    assert calls[0][0] == 'velocity'
    assert calls[0][1]['zero_velocity_when_done'] is True


def test_right_slope_2_recenter_is_bounded_and_disables_bad_edge_feedback(
        stage5_node, monkeypatch):
    """Segment 2 recenters only at entry, then holds zero lateral base."""
    node = stage5_node
    node.state = node.P5_RIGHT_SLOPE_2
    node.p5_right_slope_2_entry_recenter_duration_s = 0.8
    node.p5_right_slope_2_entry_recenter_vy = -0.075
    node.p5_right_slope_2_vy = 0.0
    node.p5_right_slope_2_right_edge_adjust_enabled = False
    elapsed = [0.4]
    calls = []
    monkeypatch.setattr(node, 'p5_safety_elapsed_s', lambda: elapsed[0])
    monkeypatch.setattr(
        node, 'run_center_yellow_absence_velocity_state',
        lambda **kwargs: calls.append(kwargs))

    node.p5_control_loop()
    elapsed[0] = 0.8
    node.p5_control_loop()

    assert calls[0]['vy'] == pytest.approx(-0.075)
    assert calls[1]['vy'] == pytest.approx(0.0)
    assert calls[0]['right_edge_adjust_enabled'] is False
    assert calls[1]['right_edge_adjust_enabled'] is False


def test_lost_extra_confirmation_is_fresh_frame_gated(stage5_node, monkeypatch):
    """Repeating one frozen danger frame cannot arm lost-extra memory."""
    node = stage5_node
    node.state = node.P5_RIGHT_SLOPE_1
    node.latest_frame_seq = 10
    node.p5_right_slope_lost_extra_last_eval_frame_seq = 9
    node.p5_right_slope_lost_extra_ignore_after_enter_s = 0.0
    node.p5_right_slope_lost_extra_confirm_count = 3
    node.p5_right_slope_lost_extra_enabled = True
    monkeypatch.setattr(node, 'p5_safety_elapsed_s', lambda: 1.0)
    monkeypatch.setattr(
        node,
        'detect_p5_right_slope_right_inner_edge',
        lambda frame: {'valid': True, 'too_center': True, 'too_right': False},
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    for _ in range(5):
        node.compute_p5_right_slope_right_edge_corrected_vy(0.0, frame)
    assert node.p5_right_slope_too_center_count == 1
    assert node.p5_right_slope_lost_extra_active is False

    node.latest_frame_seq = 11
    node.compute_p5_right_slope_right_edge_corrected_vy(0.0, frame)
    node.latest_frame_seq = 12
    node.compute_p5_right_slope_right_edge_corrected_vy(0.0, frame)
    assert node.p5_right_slope_lost_extra_active is True


def test_action_poll_rejects_stale_response_and_times_out_safe(stage5_node, monkeypatch):
    """Old completion cannot advance state; missing fresh ACK enters fault hold."""
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_RECOVERY_STAND
    node.p5_action_timeout_s = 1.0
    node.p5_action_min_ack_delay_s = 0.1
    node.p5_action_post_complete_hold_s = 0.0
    now = [10.0]
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: now[0])

    node.run_action_state(12, 0, 'NEXT')
    # A delayed duplicate completion arrives after send but has no observed
    # post-send in-progress response; it must not satisfy this action.
    now[0] = 10.2
    ctrl.snapshot.update({
        'seq': 2, 'rx_monotonic_s': 10.15, 'mode': 12,
        'gait_id': 0, 'order_process_bar': 100,
        'last_incomplete_seq': 1, 'last_incomplete_mode': 12,
        'last_incomplete_gait_id': 0,
        'last_incomplete_rx_monotonic_s': 9.99,
    })
    node.run_action_state(12, 0, 'NEXT')
    assert node.state == node.P5_RECOVERY_STAND

    now[0] = 11.1
    node.run_action_state(12, 0, 'NEXT')
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1][0:2] == (12, 0)


def test_action_poll_rejects_late_matching_completion(
    stage5_node, monkeypatch
):
    """A matching completion at or after the hard deadline must fault."""
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_TURN_1
    node.p5_action_timeout_s = 1.0
    node.p5_action_post_complete_hold_s = 0.0
    now = [15.0]
    monkeypatch.setattr(stage5_module.time, "monotonic", lambda: now[0])

    node.run_action_state(16, 3, "NEXT")
    now[0] = 16.0
    ctrl.snapshot.update({
        "seq": 2, "rx_monotonic_s": 15.99, "mode": 16,
        "gait_id": 3, "order_process_bar": 100,
        "last_incomplete_seq": 1, "last_incomplete_mode": 16,
        "last_incomplete_gait_id": 3,
        "last_incomplete_rx_monotonic_s": 15.5,
    })
    node.run_action_state(16, 3, "NEXT")
    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1][0:2] == (12, 0)


def test_action_timeout_sends_stop_before_logging(stage5_node, monkeypatch):
    """Potentially blocking evidence work cannot precede the safety STOP."""
    node = stage5_node
    node.state = node.P5_TURN_1
    node.p5_action_target = (16, 3)
    node.p5_action_phase = "action"
    order = []
    monkeypatch.setattr(
        node, "p5_send_stop_command", lambda: order.append("stop"))
    monkeypatch.setattr(
        node, "p5_evidence_log", lambda _event: order.append("evidence"))
    monkeypatch.setattr(
        node, "p5_enter_state", lambda _state: order.append("enter"))

    node.p5_action_timeout_fault("TEST", 1.0, {})
    assert order == ["stop", "evidence", "enter"]


def test_action_poll_accepts_fresh_matching_response(stage5_node, monkeypatch):
    """A post-send matching completion advances on a later control tick."""
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_RECOVERY_STAND
    node.p5_action_post_complete_hold_s = 0.0
    now = [20.0]
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: now[0])

    node.run_action_state(12, 0, 'NEXT')
    now[0] = 20.1
    ctrl.snapshot.update({
        'seq': 1, 'rx_monotonic_s': 20.08, 'mode': 12,
        'gait_id': 0, 'order_process_bar': 50,
        'last_incomplete_seq': 1, 'last_incomplete_mode': 12,
        'last_incomplete_gait_id': 0,
        'last_incomplete_rx_monotonic_s': 20.08,
    })
    node.run_action_state(12, 0, 'NEXT')
    assert node.state == node.P5_RECOVERY_STAND

    now[0] = 20.2
    ctrl.snapshot.update({
        'seq': 2, 'rx_monotonic_s': 20.18,
        'order_process_bar': 100,
    })
    node.run_action_state(12, 0, 'NEXT')
    assert node.state == 'NEXT'


def test_idempotent_stop_accepts_two_fresh_completed_responses(
    stage5_node, monkeypatch
):
    """Already-complete STOP needs sustained fresh evidence, not progress."""
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_RECOVERY_STAND
    node.p5_action_post_complete_hold_s = 0.0
    now = [22.0]
    monkeypatch.setattr(stage5_module.time, "monotonic", lambda: now[0])

    node.run_action_state(12, 0, "NEXT")
    now[0] = 22.2
    ctrl.snapshot.update({
        "seq": 1, "rx_monotonic_s": 22.15, "mode": 12,
        "gait_id": 0, "order_process_bar": 100,
    })
    node.run_action_state(12, 0, "NEXT")
    assert node.state == node.P5_RECOVERY_STAND

    now[0] = 22.21
    ctrl.snapshot.update({"seq": 2, "rx_monotonic_s": 22.151})
    node.run_action_state(12, 0, "NEXT")
    assert node.state == node.P5_RECOVERY_STAND

    now[0] = 22.3
    ctrl.snapshot.update({"seq": 3, "rx_monotonic_s": 22.26})
    node.run_action_state(12, 0, "NEXT")
    assert node.state == "NEXT"


def test_action_completion_holds_before_stop(stage5_node, monkeypatch):
    """A jump completion gets a separate settle dwell before STOP is sent."""
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_TURN_1
    node.p5_action_post_complete_hold_s = 3.0
    now = [24.0]
    monkeypatch.setattr(stage5_module.time, "monotonic", lambda: now[0])

    node.run_action_state(16, 3, "NEXT", stop_after_finish=True)
    now[0] = 24.2
    ctrl.snapshot.update({
        "seq": 1, "rx_monotonic_s": 24.15, "mode": 16,
        "gait_id": 3, "order_process_bar": 50,
        "last_incomplete_seq": 1, "last_incomplete_mode": 16,
        "last_incomplete_gait_id": 3,
        "last_incomplete_rx_monotonic_s": 24.15,
    })
    node.run_action_state(16, 3, "NEXT", stop_after_finish=True)
    now[0] = 24.4
    ctrl.snapshot.update({
        "seq": 2, "rx_monotonic_s": 24.35, "order_process_bar": 100,
    })
    node.run_action_state(16, 3, "NEXT", stop_after_finish=True)
    assert len(ctrl.sent) == 1

    now[0] = 27.3
    ctrl.snapshot.update({'seq': 3, 'rx_monotonic_s': 27.25})
    node.run_action_state(16, 3, "NEXT", stop_after_finish=True)
    assert len(ctrl.sent) == 1
    now[0] = 27.5
    ctrl.snapshot.update({'seq': 4, 'rx_monotonic_s': 27.45})
    node.run_action_state(16, 3, "NEXT", stop_after_finish=True)
    assert ctrl.sent[-1][0:2] == (12, 0)


@pytest.mark.parametrize(
    'feedback_update, now_after_complete',
    [
        ({'seq': 3, 'rx_monotonic_s': 32.29, 'mode': 9}, 32.3),
        ({'seq': 3, 'rx_monotonic_s': 32.29, 'ori_error': 1}, 32.3),
        ({}, 32.8),
    ],
    ids=['target-lost', 'error-field', 'feedback-stale'],
)
def test_action_completion_hold_faults_on_unsafe_feedback(
    stage5_node, monkeypatch, feedback_update, now_after_complete
):
    '''The settle hold stops on lost, erroneous or stale feedback.'''
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_TURN_1
    node.p5_action_post_complete_hold_s = 3.0
    node.p5_action_feedback_max_age_s = 0.5
    now = [32.0]
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: now[0])

    node.run_action_state(16, 3, 'NEXT', stop_after_finish=True)
    now[0] = 32.1
    ctrl.snapshot.update({
        'seq': 1, 'rx_monotonic_s': 32.08, 'mode': 16,
        'gait_id': 3, 'order_process_bar': 50,
        'last_incomplete_seq': 1, 'last_incomplete_mode': 16,
        'last_incomplete_gait_id': 3,
        'last_incomplete_rx_monotonic_s': 32.08,
    })
    node.run_action_state(16, 3, 'NEXT', stop_after_finish=True)
    now[0] = 32.2
    ctrl.snapshot.update({
        'seq': 2, 'rx_monotonic_s': 32.18, 'order_process_bar': 100,
    })
    node.run_action_state(16, 3, 'NEXT', stop_after_finish=True)
    assert len(ctrl.sent) == 1

    now[0] = now_after_complete
    ctrl.snapshot.update(feedback_update)
    node.run_action_state(16, 3, 'NEXT', stop_after_finish=True)

    assert node.state == node.P5_SENSOR_FAULT_HOLD
    assert ctrl.sent[-1][0:2] == (12, 0)


def test_stop_action_skips_redundant_stop_after_settle(
    stage5_node, monkeypatch
):
    """STOP with stop_after_finish does not send a duplicate command."""
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_RECOVERY_AFTER_JUMP_2
    node.p5_action_post_complete_hold_s = 3.0
    now = [28.0]
    monkeypatch.setattr(stage5_module.time, "monotonic", lambda: now[0])

    node.run_action_state(12, 0, "NEXT", stop_after_finish=True)
    now[0] = 28.2
    ctrl.snapshot.update({
        "seq": 1, "rx_monotonic_s": 28.15, "mode": 12,
        "gait_id": 0, "order_process_bar": 50,
        "last_incomplete_seq": 1, "last_incomplete_mode": 12,
        "last_incomplete_gait_id": 0,
        "last_incomplete_rx_monotonic_s": 28.15,
    })
    node.run_action_state(12, 0, "NEXT", stop_after_finish=True)
    now[0] = 28.4
    ctrl.snapshot.update({
        "seq": 2, "rx_monotonic_s": 28.35, "order_process_bar": 100,
    })
    node.run_action_state(12, 0, "NEXT", stop_after_finish=True)
    assert node.state == "NEXT"
    assert len(ctrl.sent) == 1


def test_depth_observer_wiring_is_read_only_and_preserves_metadata(stage5_node):
    """A decoded D435 frame reaches the observer cache without changing state."""
    node = stage5_node
    node.state = node.P5_UP_SLOPE
    node.latest_depth_seq = 7
    node.p5_bridge_observer_enabled = True
    node.p5_bridge_observer_period_s = 0.0
    msg = Image()
    msg.encoding = '32FC1'
    msg.header.stamp.sec = 3
    msg.header.stamp.nanosec = 500000000

    node.on_depth_frame(np.zeros((480, 640), dtype=np.float32), msg)

    assert node.state == node.P5_UP_SLOPE
    assert node.latest_bridge_observation['control_use'] == 'read_only'
    assert node.latest_bridge_observation['reason'] == 'imu_missing'
    assert node.latest_bridge_observation['frame_seq'] == 7
    assert node.latest_bridge_observation['stamp_s'] == pytest.approx(3.5)


def test_atomic_send_barrier_marks_pre_send_callback_stale():
    """A callback entering during Send_cmd retains a pre-send ingress time."""
    ctrl = Robot_Ctrl.__new__(Robot_Ctrl)
    ctrl.response_lock = Lock()
    ctrl.response_seq = 4
    ctrl.response_rx_monotonic_s = None
    ingress_ready = Event()

    def fake_send(_msg):
        def commit_old_packet():
            ingress = std_time.monotonic()
            ingress_ready.set()
            with ctrl.response_lock:
                ctrl.response_seq += 1
                ctrl.response_rx_monotonic_s = ingress

        worker = Thread(target=commit_old_packet)
        ctrl._barrier_test_worker = worker
        worker.start()
        assert ingress_ready.wait(timeout=1.0)
        std_time.sleep(0.01)

    ctrl.Send_cmd = fake_send
    baseline, sent_time = Robot_Ctrl.Send_cmd_with_response_barrier(ctrl, object())
    ctrl._barrier_test_worker.join(timeout=1.0)

    assert baseline == 4
    assert ctrl.response_seq == 5
    assert ctrl.response_rx_monotonic_s < sent_time


def test_timed_motion_faults_when_ros_clock_freezes(stage5_node, monkeypatch):
    """A non-vision velocity state cannot move forever on frozen /clock."""
    node = stage5_node
    node.state = node.P5_STEP_UP
    sent = []
    entered = []
    monkeypatch.setattr(node, 'p5_state_elapsed_s', lambda: 0.0)
    monkeypatch.setattr(node, 'p5_safety_elapsed_s', lambda: 100.0)
    monkeypatch.setattr(
        node, 'p5_send_velocity_command', lambda **kwargs: sent.append(kwargs))
    monkeypatch.setattr(node, 'p5_enter_state', lambda state: entered.append(state))

    node.run_timed_velocity_state(
        duration_s=1.0, vx=0.3, vy=0.0, wz=0.0,
        step_height=0.03, next_state='NEXT', log_name='FROZEN',
    )
    assert sent[-1]['vx'] == 0.0
    assert entered == [node.P5_SENSOR_FAULT_HOLD]


def test_final_forward_faults_when_ros_clock_freezes(stage5_node, monkeypatch):
    """The then-STOP final forward helper also has a monotonic hard deadline."""
    node = stage5_node
    node.state = node.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY
    sent = []
    entered = []
    monkeypatch.setattr(node, 'p5_state_elapsed_s', lambda: 0.0)
    monkeypatch.setattr(node, 'p5_safety_elapsed_s', lambda: 100.0)
    monkeypatch.setattr(
        node, 'p5_send_velocity_command', lambda **kwargs: sent.append(kwargs))
    monkeypatch.setattr(node, 'p5_enter_state', lambda state: entered.append(state))

    node.run_timed_velocity_then_stop_state(
        duration_s=1.0, vx=0.3, vy=0.0, wz=0.0,
        step_height=0.03, next_state=node.P5_JUMP_EXIT_SLOPE,
        use_inner_edge_align=False, stop_when_done=True,
        log_name='FROZEN_FINAL_FORWARD',
    )
    assert sent[-1]['vx'] == 0.0
    assert entered == [node.P5_SENSOR_FAULT_HOLD]


def test_timed_stop_ack_precedes_next_action_state(stage5_node, monkeypatch):
    """Final timed forward waits for STOP progress/completion before jump state."""
    node = stage5_node
    ctrl = _FakeCtrl()
    node.Ctrl = _sim_backend(node, ctrl)
    node.state = node.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY
    now = [30.0]
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(node, 'p5_state_elapsed_s', lambda: 1.0)
    monkeypatch.setattr(node, 'p5_safety_elapsed_s', lambda: 0.0)

    kwargs = dict(
        duration_s=0.5, vx=0.2, vy=0.0, wz=0.0,
        step_height=0.03, next_state=node.P5_JUMP_EXIT_SLOPE,
        log_name='FINAL_FORWARD', stop_when_done=True,
    )
    node.run_timed_velocity_then_stop_state(**kwargs)
    assert node.state == node.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY
    assert ctrl.sent[-1][0:2] == (12, 0)

    now[0] = 30.1
    ctrl.snapshot.update({
        'seq': 1, 'rx_monotonic_s': 30.08, 'mode': 12,
        'gait_id': 0, 'order_process_bar': 40,
        'last_incomplete_seq': 1, 'last_incomplete_mode': 12,
        'last_incomplete_gait_id': 0,
        'last_incomplete_rx_monotonic_s': 30.08,
    })
    node.run_timed_velocity_then_stop_state(**kwargs)
    assert node.state == node.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY

    now[0] = 30.2
    ctrl.snapshot.update({
        'seq': 2, 'rx_monotonic_s': 30.18, 'order_process_bar': 100,
    })
    node.run_timed_velocity_then_stop_state(**kwargs)
    assert node.state == node.P5_JUMP_EXIT_SLOPE


def test_valid_observer_requires_fresh_synchronized_imu(stage5_node):
    """Fresh depth+IMU can produce a valid read-only body-reference observation."""
    node = stage5_node
    node.state = node.P5_UP_SLOPE
    node.latest_depth_seq = 8
    node.p5_bridge_observer_enabled = True
    node.p5_bridge_observer_period_s = 0.0
    node.p5_last_imu_monotonic_s = std_time.monotonic()
    node.p5_last_imu_stamp_s = 4.0
    node.p5_imu_roll = 0.0
    node.p5_imu_pitch = 0.45
    msg = Image()
    msg.encoding = '32FC1'
    msg.header.stamp.sec = 4

    depth = render_scene(
        camera_pitch=0.45,
        deck_center_y=-node.p5_depth_camera_mount_y,
    )
    node.on_depth_frame(depth, msg)

    obs = node.latest_bridge_observation
    assert obs['valid'], obs
    assert obs['control_use'] == 'read_only'
    assert obs['reference_point'] == 'body_origin'
    assert obs['intrinsics_source'] == 'fov_fallback'
    assert abs(obs['lateral_offset']) < 0.04


def test_observer_rejects_zero_sensor_timestamp(stage5_node):
    """Fresh receive time cannot substitute for a missing depth/IMU timestamp."""
    node = stage5_node
    node.latest_depth_seq = 9
    node.p5_bridge_observer_enabled = True
    node.p5_bridge_observer_period_s = 0.0
    node.p5_last_imu_monotonic_s = std_time.monotonic()
    node.p5_last_imu_stamp_s = 4.0
    msg = Image()
    msg.encoding = '32FC1'

    node.on_depth_frame(np.ones((480, 640), dtype=np.float32), msg)
    assert node.latest_bridge_observation['valid'] is False
    assert node.latest_bridge_observation['reason'] == 'sensor_timestamp_missing'
    for key in (
        'camera_lateral_offset', 'camera_height', 'deck_end_x',
        'deck_end_camera_x', 'd_forward_dropoff_camera',
    ):
        assert key in node.latest_bridge_observation
        assert node.latest_bridge_observation[key] is None


def test_center_absence_counter_advances_once_per_frame(stage5_node, monkeypatch):
    """The shared right-slope absence helper cannot count one frozen frame thrice."""
    node = stage5_node
    node.state = node.P5_RIGHT_SLOPE_1
    node.p5_sensor_watchdog_enabled = False
    node.p5_center_yellow_ignore_after_enter_s = 0.0
    node.p5_center_yellow_absent_confirm_count = 3
    node.latest_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    node.state_enter_frame_seq = 9
    node.latest_frame_seq = 10
    node.p5_center_yellow_last_eval_frame_seq = 9
    entered = []
    monkeypatch.setattr(node, 'p5_state_elapsed_s', lambda: 1.0)
    monkeypatch.setattr(
        node, 'detect_p5_center_yellow_presence',
        lambda _frame: {'has_yellow': False, 'yellow_pixels': 0,
                        'yellow_ratio': 0.0},
    )
    monkeypatch.setattr(
        node, 'compute_p5_right_slope_right_edge_corrected_vy',
        lambda base_vy, frame: base_vy,
    )
    monkeypatch.setattr(node, 'p5_send_velocity_command', lambda **_kwargs: None)
    monkeypatch.setattr(node, 'p5_enter_state', lambda state: entered.append(state))

    kwargs = dict(
        vx=0.2, vy=0.0, wz=0.0, step_height=0.03,
        next_state='NEXT', timeout_s=0.0,
    )
    for _ in range(5):
        node.run_center_yellow_absence_velocity_state(**kwargs)
    assert node.p5_center_yellow_absent_counter == 1
    assert entered == []

    node.latest_frame_seq = 11
    node.run_center_yellow_absence_velocity_state(**kwargs)
    node.latest_frame_seq = 12
    node.run_center_yellow_absence_velocity_state(**kwargs)
    assert entered == ['NEXT']


def test_lost_extra_memory_expires_fail_closed(stage5_node, monkeypatch):
    """A remembered danger direction is revoked after its monotonic hold cap."""
    node = stage5_node
    node.p5_right_slope_lost_extra_enabled = True
    node.p5_right_slope_lost_extra_active = True
    node.p5_right_slope_lost_extra_direction = 'too_center'
    node.p5_right_slope_lost_extra_hold_start_s = 0.0
    node.p5_right_slope_lost_extra_max_hold_s = 0.5
    node.latest_frame_seq = 2
    node.p5_right_slope_lost_extra_last_eval_frame_seq = 1
    monkeypatch.setattr(node, 'p5_safety_elapsed_s', lambda: 1.0)
    monkeypatch.setattr(
        node, 'detect_p5_right_slope_right_inner_edge',
        lambda _frame: {'valid': False, 'reason': 'lost'},
    )

    cmd_vy = node.compute_p5_right_slope_right_edge_corrected_vy(
        0.02, np.zeros((8, 8, 3), dtype=np.uint8))
    assert cmd_vy == pytest.approx(0.02)
    assert node.p5_right_slope_lost_extra_active is False
    assert node.p5_right_slope_lost_extra_direction == 'none'


def test_profiles_and_full_launch_physical_clock_contract():
    """Physical profile is fail-safe and forces wall time for the full launch."""
    package_root = Path(__file__).resolve().parents[1]
    with (package_root / 'config' / 'stage5_sim.yaml').open() as stream:
        sim = yaml.safe_load(stream)['/**']['ros__parameters']
    with (package_root / 'config' / 'stage5_physical.yaml').open() as stream:
        physical = yaml.safe_load(stream)['/**']['ros__parameters']

    assert sim['p5_keep_moving_when_no_image'] is True
    assert sim['p5_bridge_observer_enabled'] is True
    assert physical['use_sim_time'] is False
    assert physical['p5_keep_moving_when_no_image'] is False
    assert physical['p5_bridge_observer_enabled'] is False
    assert physical['p5_timed_motion_timeout_factor'] < sim[
        'p5_timed_motion_timeout_factor']

    launch_path = package_root / 'launch' / 'full_competition.launch.py'
    spec = importlib.util.spec_from_file_location('stage5_full_launch', launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Wall time is forced by the physical Stage-5 profile on its own, and
    # independently by the physical robot backend, so check both switches.
    for configurations in (
        {'platform': 'sim', 'stage5_profile': 'physical', 'use_sim_time': 'true'},
        {'platform': 'real', 'stage5_profile': 'sim', 'use_sim_time': 'true'},
    ):
        context = LaunchContext()
        context.launch_configurations.update(configurations)
        assert module.effective_use_sim_time().perform(context) == 'false'

        checked_nodes = 0
        for entity in module.generate_launch_description().entities:
            parameters = getattr(entity, '_Node__parameters', ())
            if not parameters or not isinstance(parameters[0], dict):
                continue
            for key, value in parameters[0].items():
                if perform_substitutions(context, key) != 'use_sim_time':
                    continue
                assert perform_substitutions(context, value) == 'false'
                checked_nodes += 1
        assert checked_nodes == 7

    # A real backend must never inherit the Gazebo-tuned Stage-5 profile.
    real_context = LaunchContext()
    real_context.launch_configurations.update(
        {'platform': 'real', 'stage5_profile': 'sim'})
    for parameter_file in module.stage5_profile_files():
        assert parameter_file.perform(real_context).endswith(
            'stage5_physical.yaml')


# ------------------------------------------------------------------
# 动作丢单重发（计划 19 条）
# ------------------------------------------------------------------

def _arm_action_poll(node, monkeypatch, t0=1000.0, resend_max=1):
    """Send a (16, 3) jump through the poll machinery against a fake ctrl."""
    node.Ctrl = _sim_backend(node, _FakeCtrl())
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: t0)
    node.p5_action_resend_max = resend_max
    node.p5_action_resend_after_s = 3.0
    node.p5_begin_action_poll(16, 3, 'action')
    node.Ctrl.sent.clear()
    return node.Ctrl


def _dropped_command_snapshot(ctrl, t0=1000.0):
    """The measured signature: controller alive, echoing the previous order."""
    ctrl.snapshot.update({
        'seq': 5,
        'rx_monotonic_s': t0 + 1.0,
        'mode': 7,
        'gait_id': 0,
        'order_process_bar': 100,
    })


def test_protection_mode_takes_the_recovery_ladder(stage5_node, monkeypatch):
    """The measured signature (kPureDamper echo) -> recovery stand first,
    then the original action once recovery completes; budget respected."""
    node = stage5_node
    ctrl = _arm_action_poll(node, monkeypatch)
    _dropped_command_snapshot(ctrl)                    # mode 7, bar 100
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1003.5)
    status, _, _ = node.p5_poll_action(22.0)
    assert status == 'pending'
    assert [entry[:2] for entry in ctrl.sent] == [(12, 0)]
    # recovery stand still running: nothing further is sent
    ctrl.snapshot.update({'seq': 7, 'rx_monotonic_s': 1004.5,
                          'mode': 12, 'order_process_bar': 40})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1005.0)
    node.p5_poll_action(22.0)
    assert len(ctrl.sent) == 1
    # recovery stand complete -> the jump goes out again
    ctrl.snapshot.update({'seq': 9, 'rx_monotonic_s': 1006.5,
                          'mode': 12, 'order_process_bar': 100})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1007.0)
    node.p5_poll_action(22.0)
    assert [entry[:2] for entry in ctrl.sent] == [(12, 0), (16, 3)]
    # budget max=1 spent: a second protection episode is not rescued
    _dropped_command_snapshot(ctrl, t0=1010.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1015.0)
    node.p5_poll_action(22.0)
    assert len(ctrl.sent) == 2


def test_non_protection_mismatch_resends_directly(stage5_node, monkeypatch):
    """A transition race / true drop (echo is not a protection mode) is
    re-sent without the recovery preface."""
    node = stage5_node
    ctrl = _arm_action_poll(node, monkeypatch)
    ctrl.snapshot.update({'seq': 5, 'rx_monotonic_s': 1001.0,
                          'mode': 11, 'gait_id': 3, 'order_process_bar': 0})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1003.5)
    node.p5_poll_action(22.0)
    assert [entry[:2] for entry in ctrl.sent] == [(16, 3)]
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1008.0)
    node.p5_poll_action(22.0)
    assert len(ctrl.sent) == 1


def test_stop_target_in_protection_mode_resends_stop_directly(
        stage5_node, monkeypatch):
    """When the refused target IS (12, 0), the preface would be circular —
    the stop is simply re-sent."""
    node = stage5_node
    node.Ctrl = _sim_backend(node, _FakeCtrl())
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1000.0)
    node.p5_action_resend_max = 1
    node.p5_action_resend_after_s = 3.0
    node.p5_begin_action_poll(12, 0, 'timed_stop')
    node.Ctrl.sent.clear()
    _dropped_command_snapshot(node.Ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1003.5)
    node.p5_poll_action(8.0)
    assert [entry[:2] for entry in node.Ctrl.sent] == [(12, 0)]
    assert node.p5_action_recovery_pending is False


def test_action_resend_never_fires_while_the_action_executes(
        stage5_node, monkeypatch):
    """Echoed mode == target means the jump is running: resending would
    command a second jump."""
    node = stage5_node
    ctrl = _arm_action_poll(node, monkeypatch)
    ctrl.snapshot.update({
        'seq': 5,
        'rx_monotonic_s': 1001.0,
        'mode': 16,
        'gait_id': 3,
        'order_process_bar': 40,
    })
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1010.0)
    status, _, _ = node.p5_poll_action(15.0)
    assert status == 'pending'
    assert ctrl.sent == []


def test_action_resend_waits_out_the_delay(stage5_node, monkeypatch):
    node = stage5_node
    ctrl = _arm_action_poll(node, monkeypatch)
    _dropped_command_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(15.0)
    assert ctrl.sent == []


def test_action_resend_respects_a_seen_progress_edge(stage5_node, monkeypatch):
    """A fresh incomplete edge for the target proves the command arrived;
    a later stale-looking echo must not trigger a duplicate."""
    node = stage5_node
    ctrl = _arm_action_poll(node, monkeypatch)
    ctrl.snapshot.update({
        'last_incomplete_seq': 3,
        'last_incomplete_mode': 16,
        'last_incomplete_gait_id': 3,
        'last_incomplete_rx_monotonic_s': 1000.5,
    })
    _dropped_command_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1001.0)
    node.p5_poll_action(15.0)          # records progress_seen
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1006.0)
    node.p5_poll_action(15.0)
    assert ctrl.sent == []


def test_action_resend_requires_a_live_response_stream(
        stage5_node, monkeypatch):
    """No response since the send: a dead link is the timeout's job — a
    resend could double-queue once the link returns."""
    node = stage5_node
    ctrl = _arm_action_poll(node, monkeypatch)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1010.0)
    node.p5_poll_action(15.0)
    assert ctrl.sent == []


def test_action_resend_is_off_by_default(stage5_node, monkeypatch):
    """Code default resend_max=0 preserves the shipped behaviour."""
    node = stage5_node
    assert node.p5_action_resend_max == 0
    ctrl = _arm_action_poll(node, monkeypatch, resend_max=0)
    _dropped_command_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1010.0)
    node.p5_poll_action(15.0)
    assert ctrl.sent == []


# ------------------------------------------------------------------
# 卡死解楔（计划 33 条）
# ------------------------------------------------------------------

def _stalled_jump_snapshot(ctrl, seq=5, rx=1001.0, mode=16, gait=3):
    """The measured wedge: our own target echoed back, progress bar pinned.

    `FsmStateJump3d` reports this whenever its landing fails
    `height_good_for_trans_`, and its Transition() then refuses every exit
    except kOff — including kPureDamper and kRecoveryStand.
    """
    ctrl.snapshot.update({
        'seq': seq,
        'rx_monotonic_s': rx,
        'mode': mode,
        'gait_id': gait,
        'order_process_bar': 0,
        'switch_status': 0,
        'last_incomplete_seq': seq,
        'last_incomplete_mode': mode,
        'last_incomplete_gait_id': gait,
        'last_incomplete_rx_monotonic_s': rx,
    })


def _arm_stall_unwedge(node, monkeypatch, t0=1000.0, after_s=10.0):
    ctrl = _arm_action_poll(node, monkeypatch, t0=t0, resend_max=1)
    node.p5_action_stall_unwedge_after_s = after_s
    node.p5_action_unwedge_release_timeout_s = 5.0
    return ctrl


def test_stalled_action_is_released_with_koff(stage5_node, monkeypatch):
    """kOff is the only exit `FsmStateJump3d::Transition()` grants
    unconditionally, so it is what the ladder sends."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(40.0)          # starts the stall clock
    assert ctrl.sent == []
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1012.5)
    status, _, _ = node.p5_poll_action(40.0)
    assert status == 'pending'
    assert [entry[:2] for entry in ctrl.sent] == [(0, 0)]
    assert node.p5_action_unwedge_origin == (16, 3)


def test_stall_unwedge_stands_up_and_completes_as_that_recovery_stand(
        stage5_node, monkeypatch):
    """Nothing is declared complete unless the robot got back on its feet:
    the action target becomes the recovery stand and finishes as one."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(40.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1012.5)
    node.p5_poll_action(40.0)                       # kOff out
    # controller lets go: passive echoes mode 0
    ctrl.snapshot.update({'seq': 20, 'rx_monotonic_s': 1013.0,
                          'mode': 0, 'gait_id': 0, 'order_process_bar': 100})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1013.5)
    node.p5_poll_action(40.0)
    assert [entry[:2] for entry in ctrl.sent] == [(0, 0), (12, 0)]
    assert node.p5_action_target == (12, 0)
    assert node.p5_action_unwedge_done is True
    # the recovery stand runs and completes; the poll reports the action done
    ctrl.snapshot.update({'seq': 25, 'rx_monotonic_s': 1014.0, 'mode': 12,
                          'gait_id': 0, 'order_process_bar': 40,
                          'last_incomplete_seq': 25, 'last_incomplete_mode': 12,
                          'last_incomplete_gait_id': 0,
                          'last_incomplete_rx_monotonic_s': 1014.0})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1014.5)
    assert node.p5_poll_action(40.0)[0] == 'pending'
    ctrl.snapshot.update({'seq': 30, 'rx_monotonic_s': 1016.0,
                          'order_process_bar': 100})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1016.5)
    assert node.p5_poll_action(40.0)[0] == 'complete'
    assert len(ctrl.sent) == 2


def test_stall_unwedge_never_resends_the_stalled_jump(stage5_node, monkeypatch):
    """After the kOff the echo stops matching the target, which is exactly
    the resend ladder's trigger — a second jump must not be commanded."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(40.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1012.5)
    node.p5_poll_action(40.0)                       # kOff out
    ctrl.snapshot.update({'seq': 20, 'rx_monotonic_s': 1013.0,
                          'mode': 0, 'gait_id': 0, 'order_process_bar': 100})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1020.0)
    node.p5_poll_action(40.0)
    assert (16, 3) not in [entry[:2] for entry in ctrl.sent[1:]]


def test_stall_unwedge_waits_out_its_window(stage5_node, monkeypatch):
    """The window is measured against the slowest legitimate acknowledgement
    (4.26 s over 326 jumps), so a jump that is merely slow is left alone."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(40.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1011.0)
    node.p5_poll_action(40.0)
    assert ctrl.sent == []


def test_stall_unwedge_requires_the_action_to_have_started(
        stage5_node, monkeypatch):
    """No incomplete edge for our target means the controller never entered
    the action — that is the refusal case, and it belongs to the resend
    ladder, not here."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    ctrl.snapshot.update({'last_incomplete_seq': 0, 'last_incomplete_mode': 0,
                          'last_incomplete_gait_id': 0,
                          'last_incomplete_rx_monotonic_s': None})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(40.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1030.0)
    node.p5_poll_action(40.0)
    assert (0, 0) not in [entry[:2] for entry in ctrl.sent]


def test_stall_clock_restarts_when_the_echo_leaves_the_target(
        stage5_node, monkeypatch):
    """The recovery preface parks the controller in mode 12 for many seconds.
    Timing the stall from the send would then trip the moment the jump is
    re-accepted; it is timed from the stalled echo instead."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(40.0)
    # controller drops out of the target for a while (preface / transition)
    ctrl.snapshot.update({'seq': 9, 'rx_monotonic_s': 1004.0, 'mode': 12,
                          'order_process_bar': 100})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1005.0)
    node.p5_poll_action(40.0)
    ctrl.sent.clear()
    # target echoed again, but only just now: no unwedge yet
    _stalled_jump_snapshot(ctrl, seq=15, rx=1020.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1020.5)
    node.p5_poll_action(40.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1025.0)
    node.p5_poll_action(40.0)
    assert (0, 0) not in [entry[:2] for entry in ctrl.sent]


def test_stall_unwedge_is_off_by_default(stage5_node, monkeypatch):
    """Code default keeps the shipped behaviour; the sim profile arms it."""
    node = stage5_node
    assert node.p5_action_stall_unwedge_after_s == 0.0
    ctrl = _arm_stall_unwedge(node, monkeypatch, after_s=0.0)
    _stalled_jump_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1030.0)
    node.p5_poll_action(40.0)
    assert (0, 0) not in [entry[:2] for entry in ctrl.sent]


def test_stall_unwedge_gives_up_when_koff_is_also_refused(
        stage5_node, monkeypatch):
    """One kOff, then the verdict goes back to the action timeout."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1002.0)
    node.p5_poll_action(40.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1012.5)
    node.p5_poll_action(40.0)
    assert len(ctrl.sent) == 1
    _stalled_jump_snapshot(ctrl, seq=40, rx=1030.0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1030.5)
    node.p5_poll_action(40.0)
    assert node.p5_action_unwedge_phase == 'off_refused'
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1035.0)
    node.p5_poll_action(40.0)
    assert len(ctrl.sent) == 1


def test_action_timeout_releases_a_wedged_controller_before_stopping(
        stage5_node, monkeypatch):
    """A wedged controller ignores the fault path's own STOP. Releasing it
    is not gated on the ladder parameter: the run is already lost, and the
    physical robot has no simulator to restart."""
    node = stage5_node
    node.Ctrl = _sim_backend(node, _FakeCtrl())
    node.state = node.P5_FINAL_LONG_JUMP
    node.p5_action_target = (16, 1)
    node.p5_action_phase = 'action'
    node.p5_action_stall_unwedge_after_s = 0.0
    order = []
    monkeypatch.setattr(
        node, 'p5_send_stop_command', lambda: order.append('stop'))
    monkeypatch.setattr(
        node, 'p5_evidence_log', lambda _event: order.append('evidence'))
    monkeypatch.setattr(node, 'p5_enter_state', lambda _s: order.append('enter'))

    node.p5_action_timeout_fault('TEST', 40.0, {
        'seq': 99, 'mode': 16, 'gait_id': 1, 'order_process_bar': 0})
    assert [entry[:2] for entry in node.Ctrl.sent] == [(0, 0)]
    assert order == ['evidence', 'stop', 'evidence', 'enter']


def test_action_timeout_does_not_release_an_unrelated_stall(
        stage5_node, monkeypatch):
    """A timeout whose echo is not our target is a refusal, not a wedge —
    kOff would drop a robot that is still under control."""
    node = stage5_node
    node.Ctrl = _sim_backend(node, _FakeCtrl())
    node.state = node.P5_FINAL_LONG_JUMP
    node.p5_action_target = (16, 1)
    node.p5_action_phase = 'action'
    monkeypatch.setattr(node, 'p5_send_stop_command', lambda: None)
    monkeypatch.setattr(node, 'p5_evidence_log', lambda _event: None)
    monkeypatch.setattr(node, 'p5_enter_state', lambda _s: None)

    node.p5_action_timeout_fault('TEST', 40.0, {
        'seq': 99, 'mode': 7, 'gait_id': 0, 'order_process_bar': 100})
    assert node.Ctrl.sent == []


def test_stall_clock_restarts_when_the_progress_bar_moves(
        stage5_node, monkeypatch):
    """A wedge pins the bar; a slow action still advances it.

    Measured 2026-08-06: the ladder fired twice on a recovery stand sitting
    below 95, and RecoveryStand from kLifted legitimately takes 17.7 s while
    climbing 20/40/50/60/80 -- far past a window calibrated on jump
    acknowledgements.
    """
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    node.p5_begin_action_poll(12, 0, 'action')
    ctrl.sent.clear()
    for index, (t, bar) in enumerate(
            [(1002.0, 20), (1006.0, 40), (1010.0, 60), (1014.0, 80),
             (1018.0, 80)]):
        ctrl.snapshot.update({
            'seq': 10 + index, 'rx_monotonic_s': t - 0.1, 'mode': 12,
            'gait_id': 0, 'order_process_bar': bar,
            'last_incomplete_seq': 10 + index, 'last_incomplete_mode': 12,
            'last_incomplete_gait_id': 0,
            'last_incomplete_rx_monotonic_s': t - 0.1,
        })
        monkeypatch.setattr(stage5_module.time, 'monotonic', lambda t=t: t)
        node.p5_poll_action(60.0)
    # 16 s in and never idle for 10 s at one value: no interference
    assert ctrl.sent == []
    # now it truly freezes at 80
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1028.5)
    ctrl.snapshot.update({'seq': 40, 'rx_monotonic_s': 1028.0})
    node.p5_poll_action(60.0)
    assert [entry[:2] for entry in ctrl.sent] == [(0, 0)]


def test_pinned_bar_still_trips_the_ladder_on_time(stage5_node, monkeypatch):
    """The measured wedge holds one value forever, so the bar-change reset
    cannot delay it: r35 sat at bar 0 for 10.07 s and fired."""
    node = stage5_node
    ctrl = _arm_stall_unwedge(node, monkeypatch)
    _stalled_jump_snapshot(ctrl)
    for t in (1002.0, 1006.0, 1010.0):
        monkeypatch.setattr(stage5_module.time, 'monotonic', lambda t=t: t)
        node.p5_poll_action(40.0)
    assert ctrl.sent == []
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1012.5)
    node.p5_poll_action(40.0)
    assert [entry[:2] for entry in ctrl.sent] == [(0, 0)]


# ------------------------------------------------------------------
# 转角摔倒扶起（计划 37 条）
# ------------------------------------------------------------------

class _FakeOdom:
    """Minimal Robot_Odom stand-in: position + attitude, always fresh."""

    def __init__(self, x=0.0, y=0.0, rpy=(0.0, 0.0, 0.0), seq=7, rx=1000.0):
        self.p = [float(x), float(y), 0.3]
        self.rpy = [float(v) for v in rpy]
        self.seq = int(seq)
        self.rx = float(rx)

    def snapshot(self):
        return {
            'seq': self.seq,
            'rx_monotonic_s': self.rx,
            'p': list(self.p),
            'rpy': list(self.rpy),
            'v_world': [0.0] * 3,
            'v_body': [0.0] * 3,
            'contact': [0.0] * 4,
            'timestamp': 0,
        }


def _damped_snapshot():
    """The measured post-fall echo: kPureDamper under kEdamp, bar 100."""
    return {'seq': 99, 'mode': 7, 'gait_id': 0, 'order_process_bar': 100,
            'switch_status': 3}


def _arm_fall_recovery(node, monkeypatch, roll=2.0, t0=1000.0):
    node.Ctrl = _sim_backend(node, _FakeCtrl())
    node.Odom = _FakeOdom(x=3.0, y=15.4, rpy=(roll, 0.0, 0.17), rx=t0)
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: t0)
    monkeypatch.setattr(node, 'p5_send_stop_command', lambda: None)
    monkeypatch.setattr(node, 'p5_evidence_log', lambda _event: None)
    node.p5_fall_recovery_enabled = True
    node.p5_fall_recovery_max_attempts = 1
    node.p5_fall_recovery_attempts = 0
    node.p5_fall_recovery_min_rp_rad = 1.20
    node.p5_fall_recovery_release_timeout_s = 5.0
    node.p5_fall_recovery_stand_timeout_s = 25.0
    node.p5_route_odom_max_age_s = 0.5
    node.state = node.P5_RIGHT_JUMP_AFTER_RESET_BODY
    node.p5_action_target = (16, 3)
    node.p5_action_phase = 'action'
    return node.Ctrl


def test_fall_recovery_stands_the_robot_up_and_then_holds(
        stage5_node, monkeypatch):
    """kOff -> recovery stand -> hold. It must never resume the route.

    Measured 2026-08-07: after ~30 s of tumbling and righting, leg odometry's
    position state is meaningless -- both pick-ups stood the robot up 0.29 m
    outside the rail's inner edge while odometry reported 0.024/0.025 m of
    cross-rail displacement, and the one that resumed walked the rest of the
    route across the floor to a false P5_DONE. Nothing on board can answer
    "am I still on the course", so the answer is to stop.
    """
    node = stage5_node
    ctrl = _arm_fall_recovery(node, monkeypatch)
    entered = []
    monkeypatch.setattr(node, 'p5_enter_state',
                        lambda s: (entered.append(s),
                                   setattr(node, 'state', s))[0])

    assert node.p5_fall_recovery_begin('TEST', _damped_snapshot()) is True
    assert entered == [node.P5_FALL_RECOVER]

    node.p5_run_fall_recover()                       # phase '' -> kOff
    assert [entry[:2] for entry in ctrl.sent] == [(0, 0)]

    # controller leaves kEdamp -> the stand goes out
    ctrl.snapshot.update({'seq': 100, 'rx_monotonic_s': 1000.5, 'mode': 0,
                          'switch_status': 0, 'order_process_bar': 0})
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1001.0)
    node.p5_run_fall_recover()
    assert [entry[:2] for entry in ctrl.sent] == [(0, 0), (12, 0)]

    # the robot stands back up, apparently right where it fell
    node.Odom = _FakeOdom(x=2.99, y=15.39, rpy=(0.02, 0.0, 0.15), seq=200,
                          rx=1004.0)
    node.p5_fall_recover_finish({'seq': 120})
    # standing again, but the run is over: hold, do not retry the corner
    assert entered[-1] == node.P5_SENSOR_FAULT_HOLD
    assert node.P5_RIGHT_JUMP_AFTER_RESET_BODY not in entered[1:]


def test_fall_recovery_declines_when_the_body_is_still_upright(
        stage5_node, monkeypatch):
    """kEdamp can latch on a workspace violation without a fall; dropping a
    standing robot with kOff would cause the fall it exists to recover."""
    node = stage5_node
    _arm_fall_recovery(node, monkeypatch, roll=0.55)   # right-slope body preset
    monkeypatch.setattr(node, 'p5_enter_state', lambda _s: None)
    assert node.p5_fall_recovery_begin('TEST', _damped_snapshot()) is False


def test_fall_recovery_declines_on_a_wedge_rather_than_a_fall(
        stage5_node, monkeypatch):
    """A wedged jump echoes its own target, not the damper: that is item 33's
    ladder, and kOff there is already handled by the timeout release."""
    node = stage5_node
    _arm_fall_recovery(node, monkeypatch)
    monkeypatch.setattr(node, 'p5_enter_state', lambda _s: None)
    assert node.p5_fall_recovery_begin('TEST', {
        'seq': 99, 'mode': 16, 'gait_id': 3, 'order_process_bar': 0,
        'switch_status': 0}) is False


def test_fall_recovery_faults_when_the_stand_leaves_the_body_down(
        stage5_node, monkeypatch):
    """A completed stand that did not actually stand must not resume."""
    node = stage5_node
    _arm_fall_recovery(node, monkeypatch)
    entered = []
    monkeypatch.setattr(node, 'p5_enter_state',
                        lambda s: (entered.append(s),
                                   setattr(node, 'state', s))[0])
    node.p5_fall_recovery_begin('TEST', _damped_snapshot())
    node.Odom = _FakeOdom(x=2.95, y=14.82, rpy=(1.95, 0.0, 0.15), seq=200,
                          rx=1004.0)
    node.p5_fall_recover_finish({'seq': 120})
    assert entered[-1] == node.P5_SENSOR_FAULT_HOLD


def test_fall_recovery_spends_its_budget_once_per_run(stage5_node, monkeypatch):
    """An action that keeps toppling the robot must not be retried forever."""
    node = stage5_node
    _arm_fall_recovery(node, monkeypatch)
    monkeypatch.setattr(node, 'p5_enter_state',
                        lambda s: setattr(node, 'state', s))
    assert node.p5_fall_recovery_begin('TEST', _damped_snapshot()) is True
    node.state = node.P5_RIGHT_JUMP_AFTER_RESET_BODY
    assert node.p5_fall_recovery_begin('TEST', _damped_snapshot()) is False


def test_fall_recovery_is_off_by_default(stage5_node, monkeypatch):
    """Code default keeps the shipped fail-closed behaviour."""
    node = stage5_node
    assert node.p5_fall_recovery_enabled is False
    _arm_fall_recovery(node, monkeypatch)
    node.p5_fall_recovery_enabled = False
    monkeypatch.setattr(node, 'p5_enter_state', lambda _s: None)
    assert node.p5_fall_recovery_begin('TEST', _damped_snapshot()) is False


def test_fall_recovery_gives_up_when_koff_is_refused(stage5_node, monkeypatch):
    """Still damped after the release window: fail closed rather than wait."""
    node = stage5_node
    ctrl = _arm_fall_recovery(node, monkeypatch)
    entered = []
    monkeypatch.setattr(node, 'p5_enter_state',
                        lambda s: (entered.append(s),
                                   setattr(node, 'state', s))[0])
    node.p5_fall_recovery_begin('TEST', _damped_snapshot())
    ctrl.snapshot.update(_damped_snapshot())
    node.p5_run_fall_recover()
    assert [entry[:2] for entry in ctrl.sent] == [(0, 0)]
    monkeypatch.setattr(stage5_module.time, 'monotonic', lambda: 1006.0)
    node.p5_run_fall_recover()
    assert entered[-1] == node.P5_SENSOR_FAULT_HOLD


def test_corner_1_expected_yaw_matches_what_the_jump_measures():
    """The sim profile's corner_1 expectation must sit on the measured
    distribution, not on the course's nominal 90 deg: 36 runs measured
    +93.20..+97.20 (median +94.75) and 6 of them tripped a re-alignment that
    shoves the body 0.067 m toward the rail edge."""
    config = (Path(stage5_module.__file__).resolve().parents[1]
              / 'config' / 'stage5_sim.yaml')
    params = yaml.safe_load(config.read_text())['/**']['ros__parameters']
    expected = float(params['p5_route_corner_1_expected_yaw_deg'])
    tol = float(params['p5_route_corner_1_yaw_tol_deg'])
    for measured in (93.20, 94.75, 97.20):
        assert abs(measured - expected) <= tol, measured


def test_fall_recovery_stand_window_covers_the_measured_righting(
        stage5_node, monkeypatch):
    """Standing from fully on-side is much slower than from kLifted.

    Measured 2026-08-07 (`corner_falls` r16/r19): kOff frees kEdamp in 0.09-0.10
    s, the first recovery stand then sits at bar 0 for ~10 s until item 33's
    unwedge ladder kicks it, and righting from there takes 22.6/24.9 s before
    the body even starts to rise. A 25 s window expires on the exact tick the
    robot becomes upright.
    """
    config = (Path(stage5_module.__file__).resolve().parents[1]
              / 'config' / 'stage5_sim.yaml')
    params = yaml.safe_load(config.read_text())['/**']['ros__parameters']
    assert float(params['p5_fall_recovery_stand_timeout_s']) >= 40.0
    node = stage5_node
    assert node.p5_fall_recovery_stand_timeout_s >= 40.0


def test_fall_recovery_gives_up_when_the_total_budget_is_spent(
        stage5_node, monkeypatch):
    """The pick-up must not hold the stage open indefinitely."""
    node = stage5_node
    _arm_fall_recovery(node, monkeypatch)
    entered = []
    monkeypatch.setattr(node, 'p5_enter_state',
                        lambda s: (entered.append(s),
                                   setattr(node, 'state', s))[0])
    node.p5_fall_recovery_total_timeout_s = 90.0
    node.p5_fall_recovery_begin('TEST', _damped_snapshot())
    node.p5_run_fall_recover()                       # phase '' -> kOff
    monkeypatch.setattr(node, 'p5_safety_elapsed_s', lambda: 95.0)
    node.p5_run_fall_recover()
    assert entered[-1] == node.P5_SENSOR_FAULT_HOLD
