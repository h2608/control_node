#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the physical-backend diagnostic gates.

The CyberDog ``protocol`` package only exists on the robot, so the message and
service types are stubbed here, exactly as the pre-migration offline checks in
REAL_ROBOT_MIGRATION_README.md did.  What is under test is the branching added
for the RGB-dropout bisect, not the ROS transport.
"""

import sys
import types

import pytest

pytest.importorskip('rclpy')


# ----------------------------------------------------------------------
# Stub the robot-only `protocol` package before importing the adapter.
# ----------------------------------------------------------------------
class _StubMotionServoCmd(object):
    def __init__(self):
        self.motion_id = 0
        self.cmd_type = 0
        self.cmd_source = 0
        self.value = 0
        self.vel_des = [0.0, 0.0, 0.0]
        self.rpy_des = [0.0, 0.0, 0.0]
        self.pos_des = [0.0, 0.0, 0.0]
        self.acc_des = [0.0, 0.0, 0.0]
        self.ctrl_point = [0.0, 0.0, 0.0]
        self.foot_pose = [0.0, 0.0, 0.0]
        self.step_height = [0.0, 0.0]


class _StubMotionServoResponse(object):
    pass


class _StubMotionResultCmd(object):
    class Request(object):
        def __init__(self):
            self.motion_id = 0
            self.cmd_source = 0
            self.vel_des = [0.0, 0.0, 0.0]
            self.rpy_des = [0.0, 0.0, 0.0]
            self.pos_des = [0.0, 0.0, 0.0]
            self.acc_des = [0.0, 0.0, 0.0]
            self.ctrl_point = [0.0, 0.0, 0.0]
            self.foot_pose = [0.0, 0.0, 0.0]
            self.step_height = [0.0, 0.0]
            self.duration = 0


if 'protocol' not in sys.modules:
    _protocol = types.ModuleType('protocol')
    _protocol_msg = types.ModuleType('protocol.msg')
    _protocol_srv = types.ModuleType('protocol.srv')
    _protocol_msg.MotionServoCmd = _StubMotionServoCmd
    _protocol_msg.MotionServoResponse = _StubMotionServoResponse
    _protocol_srv.MotionResultCmd = _StubMotionResultCmd
    _protocol.msg = _protocol_msg
    _protocol.srv = _protocol_srv
    sys.modules['protocol'] = _protocol
    sys.modules['protocol.msg'] = _protocol_msg
    sys.modules['protocol.srv'] = _protocol_srv

from control_node.robot_interface import real_controller as rc  # noqa: E402


# ----------------------------------------------------------------------
# Minimal fakes for the ROS objects the adapter builds for itself.
# ----------------------------------------------------------------------
class _FakePublisher(object):
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _FakeClient(object):
    def service_is_ready(self):
        return True

    def wait_for_service(self, timeout_sec=0.0):
        return True

    def call_async(self, request):
        raise AssertionError('the service must not be called in this mode')


class _FakeIoNode(object):
    def __init__(self, *args, **kwargs):
        self.publisher = _FakePublisher()
        self.timers = []

    def create_publisher(self, *args, **kwargs):
        return self.publisher

    def create_subscription(self, *args, **kwargs):
        return object()

    def create_client(self, *args, **kwargs):
        return _FakeClient()

    def create_timer(self, period, callback):
        self.timers.append((period, callback))
        return object()

    def destroy_node(self):
        pass


class _FakeExecutor(object):
    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, node):
        pass

    def remove_node(self, node):
        pass

    def spin_once(self, timeout_sec=0.0):
        pass

    def wake(self):
        pass

    def shutdown(self, timeout_sec=0.0):
        pass


class _FakeLogger(object):
    def __init__(self):
        self.lines = []

    def _record(self, text):
        self.lines.append(str(text))

    def info(self, text, **kwargs):
        self._record(text)

    def warning(self, text, **kwargs):
        self._record(text)

    def warn(self, text, **kwargs):
        self._record(text)

    def error(self, text, **kwargs):
        self._record(text)


class _FakeParam(object):
    def __init__(self, value):
        self.value = value


class _FakeClock(object):
    class _Now(object):
        nanoseconds = 0

    def now(self):
        return self._Now()


class _FakeParentNode(object):
    """Just enough of a stage node for RealRobotControlAdapter.__init__."""

    def __init__(self, overrides=None, declared=True):
        self.context = object()
        self.logger = _FakeLogger()
        self.events = []
        self._declared = declared
        self._values = {
            'real_motion_servo_cmd_topic': '/motion_servo_cmd',
            'real_motion_servo_response_topic': '/motion_servo_response',
            'real_motion_result_service': '/motion_result_cmd',
            'real_cmd_source': 0,
            'real_default_servo_motion_id': 303,
            'real_servo_publish_hz': 20.0,
            'real_servo_start_repeat': 5,
            'real_servo_end_repeat': 5,
            'real_servo_start_settle_s': 0.0,
            'real_servo_start_ack_timeout_s': 2.0,
            'real_motion_service_wait_timeout_s': 2.0,
            'real_action_wait_timeout_s': 45.0,
            'real_servo_publish_enabled': True,
            'real_result_actions_enabled': True,
            'real_recovery_motion_id': 111,
            'real_emergency_stop_motion_id': 0,
            'real_lie_down_motion_id': 101,
            'real_left_jump_motion_id': 134,
            'real_right_jump_motion_id': 135,
            'real_forward_jump_motion_id': 132,
            'real_legacy_gait0_motion_id': 303,
            'real_legacy_gait1_motion_id': 303,
            'real_legacy_gait3_motion_id': 303,
            'real_legacy_gait27_motion_id': 303,
        }
        if overrides:
            self._values.update(overrides)

    def get_name(self):
        return 'stage2_node'

    def get_parameter(self, name):
        if name not in self._values:
            raise KeyError(name)
        if not self._declared and name in (
                'real_servo_publish_enabled', 'real_result_actions_enabled'):
            raise RuntimeError('parameter not declared')
        return _FakeParam(self._values[name])

    def get_logger(self):
        return self.logger

    def get_clock(self):
        return _FakeClock()

    def note_stage_event(self, event, detail=''):
        self.events.append((event, detail))


@pytest.fixture(autouse=True)
def _patch_ros(monkeypatch):
    monkeypatch.setattr(rc, 'Node', _FakeIoNode)
    monkeypatch.setattr(rc, 'SingleThreadedExecutor', _FakeExecutor)


def _adapter(**overrides):
    parent = _FakeParentNode(overrides=overrides)
    return rc.RealRobotControlAdapter(parent), parent


# ----------------------------------------------------------------------
# Default configuration
# ----------------------------------------------------------------------
def test_gates_default_to_enabled():
    adapter, _parent = _adapter()
    assert adapter.servo_publish_enabled is True
    assert adapter.result_actions_enabled is True


def test_undeclared_gate_parameters_fall_back_to_enabled():
    # An older stage node (or a debug tool) that never declared the new
    # parameters must still get the normal, fully enabled motion path.
    parent = _FakeParentNode(declared=False)
    adapter = rc.RealRobotControlAdapter(parent)
    assert adapter.servo_publish_enabled is True
    assert adapter.result_actions_enabled is True


# ----------------------------------------------------------------------
# real_servo_publish_enabled
# ----------------------------------------------------------------------
def test_suppressed_servo_publishes_nothing_and_counts_nothing():
    # The counters mean "frames actually put on the wire", so a suppressed
    # session must leave them at zero and be visible only as an event marker.
    adapter, parent = _adapter(real_servo_publish_enabled=False)
    assert adapter.move(0.2, 0.0, 0.0) is True
    assert adapter._io_node.publisher.published == []
    snapshot = adapter.diagnostics_snapshot()
    assert snapshot['servo_tx_start'] == 0
    assert snapshot['servo_tx_data'] == 0
    assert ('SERVO_START_SUPPRESSED', 'motion_id=303') in parent.events


def test_suppressed_servo_timer_ticks_publish_nothing():
    adapter, _parent = _adapter(real_servo_publish_enabled=False)
    adapter.move(0.2, 0.0, 0.0)
    for _ in range(5):
        adapter._servo_timer_cb()
    assert adapter._io_node.publisher.published == []
    assert adapter.diagnostics_snapshot()['servo_tx_data'] == 0


def test_suppressed_servo_start_does_not_block_for_the_ack_timeout():
    # The ACK can never arrive when nothing is published; burning
    # start_ack_timeout_s in the caller's executor would defeat the whole point
    # of the diagnostic mode.
    import time
    adapter, _parent = _adapter(real_servo_publish_enabled=False)
    start = time.monotonic()
    adapter.move(0.2, 0.0, 0.0)
    assert time.monotonic() - start < 0.5


def test_suppressed_servo_still_reports_ready():
    adapter, _parent = _adapter(real_servo_publish_enabled=False)
    adapter.move(0.2, 0.0, 0.0)
    assert adapter.is_servo_ready() is True


def test_enabled_servo_publishes_start_frames():
    adapter, _parent = _adapter(real_servo_start_ack_timeout_s=0.1,
                                real_servo_start_repeat=1)
    # No ACK will ever arrive from the fake robot, so this returns False after
    # the (short) timeout -- but the START frames must have been published.
    assert adapter.move(0.2, 0.0, 0.0) is False
    assert len(adapter._io_node.publisher.published) >= 1


# ----------------------------------------------------------------------
# real_result_actions_enabled
# ----------------------------------------------------------------------
def test_suppressed_result_action_completes_without_calling_the_service():
    adapter, parent = _adapter(real_result_actions_enabled=False)
    # _FakeClient.call_async would raise if it were reached.
    assert adapter.run_result_action(111, legacy_target=(12, 0)) is True
    assert adapter.Wait_finish(12, 0) is True
    assert adapter.diagnostics_snapshot()['result_calls'] == 1
    assert ('ACTION_SUPPRESSED', 'motion_id=111') in parent.events


def test_suppressed_recovery_stand_returns_success_immediately():
    import time
    adapter, _parent = _adapter(real_result_actions_enabled=False)
    start = time.monotonic()
    assert adapter.recovery_stand(wait_finish=True) is True
    assert time.monotonic() - start < 0.5


# ----------------------------------------------------------------------
# Diagnostics plumbing
# ----------------------------------------------------------------------
def test_diagnostics_snapshot_exposes_the_servo_counters():
    adapter, _parent = _adapter(real_servo_publish_enabled=False)
    snapshot = adapter.diagnostics_snapshot()
    for key in ('servo_tx_start', 'servo_tx_data', 'servo_tx_end',
                'servo_rx_resp', 'result_calls', 'servo_active'):
        assert key in snapshot


def test_note_survives_a_parent_without_the_event_hook():
    class _NoHookParent(_FakeParentNode):
        note_stage_event = None

    parent = _NoHookParent()
    adapter = rc.RealRobotControlAdapter(parent)
    adapter._note('ANYTHING')  # must not raise
