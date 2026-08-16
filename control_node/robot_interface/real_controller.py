"""Physical CyberDog adapter using Xiaomi's ROS 2 motion interfaces.

Continuous locomotion:
    protocol/msg/MotionServoCmd      (START=0, DATA=1, END=2)
    protocol/msg/MotionServoResponse
Discrete/result actions:
    protocol/srv/MotionResultCmd

The class owns a small helper ROS node spun in a background executor.  This is
intentional: several legacy stage functions still call Wait_finish() from the
main node's callback thread.  The helper executor can continue receiving the
service result while that callback waits, which preserves the old state-machine
semantics during migration without blocking camera/TF processing forever.
"""

import math
import threading
import time
from typing import Optional, Tuple

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from protocol.msg import MotionServoCmd, MotionServoResponse
from protocol.srv import MotionResultCmd

from . import motion_ids


SERVO_START = 0
SERVO_DATA = 1
SERVO_END = 2


def _fit(values, n, default=0.0):
    data = list(values) if values is not None else []
    data = [float(v) for v in data[:n]]
    while len(data) < n:
        data.append(float(default))
    return data


def _nonzero(values, eps=1e-6):
    return any(abs(float(v)) > eps for v in (values or []))


class RealRobotControlAdapter:
    is_real = True
    backend_name = 'real_motion_api'

    def __init__(self, parent_node):
        self.node = parent_node
        self.log = parent_node.get_logger()
        self._lock = threading.RLock()

        gp = parent_node.get_parameter
        self.servo_cmd_topic = str(gp('real_motion_servo_cmd_topic').value)
        self.servo_response_topic = str(gp('real_motion_servo_response_topic').value)
        self.result_service = str(gp('real_motion_result_service').value)
        self.cmd_source = int(gp('real_cmd_source').value)
        self.default_servo_motion_id = int(gp('real_default_servo_motion_id').value)
        self.publish_hz = max(5.0, float(gp('real_servo_publish_hz').value))
        self.start_repeat = max(1, int(gp('real_servo_start_repeat').value))
        self.end_repeat = max(1, int(gp('real_servo_end_repeat').value))
        self.start_settle_s = max(0.0, float(gp('real_servo_start_settle_s').value))
        self.start_ack_timeout_s = max(0.1, float(
            gp('real_servo_start_ack_timeout_s').value))
        self.service_wait_timeout_s = max(
            0.0, float(gp('real_motion_service_wait_timeout_s').value))
        self.action_wait_timeout_s = max(
            0.1, float(gp('real_action_wait_timeout_s').value))

        self.recovery_motion_id = int(gp('real_recovery_motion_id').value)
        self.emergency_motion_id = int(gp('real_emergency_stop_motion_id').value)
        self.lie_down_motion_id = int(gp('real_lie_down_motion_id').value)
        self.left_jump_motion_id = int(gp('real_left_jump_motion_id').value)
        self.right_jump_motion_id = int(gp('real_right_jump_motion_id').value)
        self.forward_jump_motion_id = int(gp('real_forward_jump_motion_id').value)

        self.legacy_gait_motion = {
            0: int(gp('real_legacy_gait0_motion_id').value),
            1: int(gp('real_legacy_gait1_motion_id').value),
            3: int(gp('real_legacy_gait3_motion_id').value),
            27: int(gp('real_legacy_gait27_motion_id').value),
        }

        # Servo target/lifecycle.
        self._servo_active = False
        self._servo_motion_id = None
        self._servo_legacy_gait = 3
        self._servo_target = self._zero_servo_payload()
        self._servo_start_remaining = 0
        self._servo_start_time = None
        self._servo_data_started = False
        # True only after the physical motion manager has ACKed the current
        # Servo START.  Any discrete MotionResultCmd invalidates this state.
        self._servo_ready_node_time_s = None
        self._servo_ready_monotonic_s = None
        # Timed motion states use this node-clock anchor.  It advances whenever
        # a Servo session is (re)started/accepted or a ResultCmd completes, so
        # time spent waiting for motion-manager handoff is never counted as
        # effective vx/vy/wz motion time.
        self._motion_timer_anchor_node_time_s = None

        # Last raw servo feedback.
        self._last_servo_response = None
        self._last_servo_rx_monotonic_s = None

        # Pseudo legacy response used by Stage 4/5 compatibility code.
        self._response_seq = 0
        self._response_rx_monotonic_s = None
        self._legacy_mode = 0
        self._legacy_gait = 0
        self._legacy_progress = 0
        self._legacy_switch_status = 0
        self._legacy_ori_error = 0
        self._legacy_footpos_error = 0
        self._last_incomplete_seq = 0
        self._last_incomplete_mode = 0
        self._last_incomplete_gait = 0
        self._last_incomplete_rx_monotonic_s = None

        # Discrete result action state.
        self._action_generation = 0
        self._last_action_target: Optional[Tuple[int, int]] = None
        self._last_action_motion_id = None
        self._last_action_complete = False
        self._last_action_success = False
        self._last_action_code = 0

        helper_name = '{}_real_motion_io'.format(parent_node.get_name())
        self._io_node = Node(helper_name, context=parent_node.context)
        self._servo_pub = self._io_node.create_publisher(
            MotionServoCmd, self.servo_cmd_topic, 10)
        self._servo_sub = self._io_node.create_subscription(
            MotionServoResponse,
            self.servo_response_topic,
            self._servo_response_cb,
            10,
        )
        self._result_client = self._io_node.create_client(
            MotionResultCmd, self.result_service)
        self._servo_timer = self._io_node.create_timer(
            1.0 / self.publish_hz, self._servo_timer_cb)

        self._executor = SingleThreadedExecutor(context=parent_node.context)
        self._executor.add_node(self._io_node)
        self._run_executor = True
        self._executor_thread = threading.Thread(
            target=self._executor_loop,
            name=helper_name + '_executor',
            daemon=True,
        )
        self._started = False
        self._servo_starting = False
        self._servo_first_data_logged = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self):
        if self._started:
            return
        self._started = True
        self._executor_thread.start()
        self.log.info(
            '[REAL_CTRL] motion backend ready: servo_cmd={}, servo_response={}, '
            'result_service={}'.format(
                self.servo_cmd_topic,
                self.servo_response_topic,
                self.result_service,
            )
        )

    def quit(self):
        try:
            self.stop_motion()
        except Exception as exc:
            self.log.warning('[REAL_CTRL] SERVO_END during quit failed: {}'.format(exc))
        self._run_executor = False
        try:
            self._executor.wake()
        except Exception:
            pass
        if self._executor_thread.is_alive():
            self._executor_thread.join(timeout=0.5)
        try:
            self._executor.remove_node(self._io_node)
        except Exception:
            pass
        try:
            self._io_node.destroy_node()
        except Exception:
            pass
        try:
            self._executor.shutdown(timeout_sec=0.1)
        except Exception:
            pass
        self._started = False

    def _executor_loop(self):
        while self._run_executor:
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception as exc:
                if self._run_executor:
                    self.log.error('[REAL_CTRL] helper executor error: {}'.format(exc))
                    time.sleep(0.05)

    # ------------------------------------------------------------------
    # Servo locomotion
    # ------------------------------------------------------------------
    @staticmethod
    def _zero_servo_payload():
        return {
            'vel_des': [0.0, 0.0, 0.0],
            'rpy_des': [0.0, 0.0, 0.0],
            'pos_des': [0.0, 0.0, 0.0],
            'acc_des': [0.0, 0.0, 0.0],
            'ctrl_point': [0.0, 0.0, 0.0],
            'foot_pose': [0.0, 0.0, 0.0],
            'step_height': [0.05, 0.05],
            'value': 0,
        }

    def _make_servo_msg(self, cmd_type, payload_override=None):
        with self._lock:
            payload = (
                dict(payload_override)
                if payload_override is not None
                else dict(self._servo_target))
            motion_id = int(self._servo_motion_id or self.default_servo_motion_id)
        msg = MotionServoCmd()
        msg.motion_id = motion_id
        msg.cmd_type = int(cmd_type)
        msg.cmd_source = int(self.cmd_source)
        msg.value = int(payload.get('value', 0))
        msg.vel_des = _fit(payload.get('vel_des'), 3)
        msg.rpy_des = _fit(payload.get('rpy_des'), 3)
        msg.pos_des = _fit(payload.get('pos_des'), 3)
        msg.acc_des = _fit(payload.get('acc_des'), 3)
        msg.ctrl_point = _fit(payload.get('ctrl_point'), 3)
        msg.foot_pose = _fit(payload.get('foot_pose'), 3)
        msg.step_height = _fit(payload.get('step_height'), 2, default=0.05)
        return msg

    def _publish_servo(self, cmd_type, payload_override=None):
        self._servo_pub.publish(
            self._make_servo_msg(cmd_type, payload_override=payload_override))

    def _node_now_s(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _set_motion_timer_anchor_locked(self):
        self._motion_timer_anchor_node_time_s = self._node_now_s()

    def _invalidate_servo_ready_locked(self, update_timer_anchor=False):
        self._servo_ready_node_time_s = None
        self._servo_ready_monotonic_s = None
        self._servo_data_started = False
        self._servo_starting = False
        if update_timer_anchor:
            self._set_motion_timer_anchor_locked()

    def is_servo_ready(self):
        with self._lock:
            return bool(
                self._servo_active and
                self._servo_data_started and
                not self._servo_starting and
                self._servo_ready_node_time_s is not None
            )

    def get_motion_timer_anchor_node_time_s(self):
        with self._lock:
            return self._motion_timer_anchor_node_time_s

    def _begin_servo_locked(self, motion_id, legacy_gait):
        self._servo_active = True
        self._servo_motion_id = int(motion_id)
        self._servo_legacy_gait = int(legacy_gait)
        self._servo_start_remaining = 0
        self._servo_start_time = time.monotonic()
        self._servo_data_started = False
        self._servo_starting = True
        self._servo_first_data_logged = False
        self._servo_ready_node_time_s = None
        self._servo_ready_monotonic_s = None
        # Reset any state timer that may already have started before this
        # blocking START/ACK handoff.
        self._set_motion_timer_anchor_locked()

    def move(self, vx, vy, wz, *, step_height=0.05, roll=0.0, pitch=0.0,
             yaw=0.0, body_height=0.0, motion_id=None, legacy_gait_id=3,
             value=0):
        motion_id = int(
            motion_id if motion_id is not None
            else self.legacy_gait_motion.get(
                int(legacy_gait_id), self.default_servo_motion_id))

        with self._lock:
            changing_motion = (
                self._servo_active and self._servo_motion_id != motion_id)
        if changing_motion:
            self.stop_motion()

        with self._lock:
            self._servo_target = {
                'vel_des': [float(vx), float(vy), float(wz)],
                'rpy_des': [float(roll), float(pitch), float(yaw)],
                'pos_des': [0.0, 0.0, float(body_height)],
                'acc_des': [0.0, 0.0, 0.0],
                'ctrl_point': [0.0, 0.0, 0.0],
                'foot_pose': [0.0, 0.0, 0.0],
                'step_height': [float(step_height), float(step_height)],
                'value': int(value),
            }
            if not self._servo_active:
                self._begin_servo_locked(motion_id, legacy_gait_id)
                publish_start = True
            else:
                self._servo_legacy_gait = int(legacy_gait_id)
                publish_start = False

        if publish_start:
            # Every START frame is neutral.  The important part on the physical
            # robot is not a guessed sleep duration: a Servo START sent
            # immediately after a ResultCmd (for example recovery-stand 111) can
            # be rejected as BUSY.  Therefore send the proven minimum START
            # burst, then keep retrying START until the robot itself returns a
            # successful MotionServoResponse for this motion_id.
            neutral = self._zero_servo_payload()
            interval_s = 1.0 / self.publish_hz
            start_mark = time.monotonic()
            self.log.info(
                '[REAL_CTRL] SERVO_START motion_id={} legacy_gait={} repeat={}, '
                'waiting robot ACK'.format(
                    motion_id, legacy_gait_id, self.start_repeat))

            for _ in range(self.start_repeat):
                self._publish_servo(SERVO_START, payload_override=neutral)
                time.sleep(interval_s)

            deadline = time.monotonic() + self.start_ack_timeout_s
            ack = None
            last_reject = None
            while time.monotonic() < deadline:
                with self._lock:
                    response = self._last_servo_response
                    rx_time = self._last_servo_rx_monotonic_s
                    if (response is not None and rx_time is not None and
                            rx_time >= start_mark and
                            int(response.motion_id) == int(motion_id)):
                        ack = (
                            bool(response.result),
                            int(response.code),
                            int(response.status),
                            int(response.order_process_bar),
                        )

                if ack is not None and ack[0]:
                    break

                if ack is not None:
                    reject_key = ack
                    if reject_key != last_reject:
                        self.log.warning(
                            '[REAL_CTRL] SERVO_START rejected/busy motion_id={} '
                            'result={} code={} status={} progress={}; retrying'.format(
                                motion_id, ack[0], ack[1], ack[2], ack[3]))
                        last_reject = reject_key

                # Retry START instead of sending DATA into a Servo session that
                # the motion manager has not accepted yet.
                self._publish_servo(SERVO_START, payload_override=neutral)
                time.sleep(interval_s)

            if ack is None or not ack[0]:
                with self._lock:
                    self._servo_active = False
                    self._servo_motion_id = None
                    self._invalidate_servo_ready_locked(update_timer_anchor=True)
                self.log.error(
                    '[REAL_CTRL] SERVO_START ACK timeout motion_id={} after {:.2f}s; '
                    'DATA will NOT be sent'.format(
                        motion_id, self.start_ack_timeout_s))
                return False

            # Optional tiny post-ACK delay; default is zero in the real profile.
            if self.start_settle_s > 0.0:
                time.sleep(self.start_settle_s)

            with self._lock:
                if self._servo_active and self._servo_motion_id == motion_id:
                    self._servo_starting = False
                    self._servo_data_started = True
                    self._servo_start_time = time.monotonic()
                    self._servo_ready_monotonic_s = self._servo_start_time
                    self._servo_ready_node_time_s = self._node_now_s()
                    self._motion_timer_anchor_node_time_s = self._servo_ready_node_time_s

            self.log.info(
                '[REAL_CTRL] SERVO_READY motion_id={} robot_ack=True '
                'code={} status={} progress={}'.format(
                    motion_id, ack[1], ack[2], ack[3]))

            # Do not wait another timer period for the first DATA.  Publish the
            # current target immediately after the robot has accepted Servo.
            self._publish_servo(SERVO_DATA)
            with self._lock:
                self._servo_first_data_logged = True
            self.log.info(
                '[REAL_CTRL] FIRST SERVO_DATA motion_id={}'.format(motion_id))
            return True

        return True

    def stop_motion(self):
        with self._lock:
            was_active = self._servo_active
            if not was_active:
                # Do not leave a stale ready flag after a prior ResultCmd or
                # already-ended Servo session.
                self._invalidate_servo_ready_locked(update_timer_anchor=False)
                return True
            self._servo_target = self._zero_servo_payload()

        # Match the proven standalone shutdown sequence: one neutral DATA,
        # then repeated END frames spaced at the Servo period.
        interval_s = 1.0 / self.publish_hz
        try:
            self._publish_servo(
                SERVO_DATA, payload_override=self._zero_servo_payload())
            time.sleep(interval_s)
        except Exception:
            pass
        for _ in range(self.end_repeat):
            self._publish_servo(
                SERVO_END, payload_override=self._zero_servo_payload())
            time.sleep(interval_s)

        with self._lock:
            self._servo_active = False
            self._servo_motion_id = None
            self._servo_start_remaining = 0
            self._servo_start_time = None
            self._servo_data_started = False
            self._servo_starting = False
            self._servo_first_data_logged = False
            self._servo_ready_node_time_s = None
            self._servo_ready_monotonic_s = None
        self.log.info('[REAL_CTRL] SERVO_END')
        return True

    def _servo_timer_cb(self):
        with self._lock:
            if not self._servo_active:
                return
            if self._servo_starting or not self._servo_data_started:
                return
            first_data = not self._servo_first_data_logged
            motion_id = int(
                self._servo_motion_id or self.default_servo_motion_id)

        self._publish_servo(SERVO_DATA)

        if first_data:
            with self._lock:
                if not self._servo_first_data_logged:
                    self._servo_first_data_logged = True
                    should_log = True
                else:
                    should_log = False
            if should_log:
                self.log.info(
                    '[REAL_CTRL] FIRST SERVO_DATA motion_id={}'.format(motion_id))

    def _servo_response_cb(self, msg):
        now = time.monotonic()
        with self._lock:
            self._last_servo_response = msg
            self._last_servo_rx_monotonic_s = now
            # Do not overwrite a discrete action's pseudo response while that
            # service request is outstanding or while its final ACK is being
            # consumed by Stage 5's post-action hold logic.
            if self._last_action_target is None or self._last_action_complete:
                if self._servo_active:
                    self._response_seq += 1
                    self._response_rx_monotonic_s = now
                    self._legacy_mode = 11
                    self._legacy_gait = int(self._servo_legacy_gait)
                    self._legacy_progress = int(msg.order_process_bar)
                    self._legacy_switch_status = int(msg.status)
                    self._legacy_ori_error = 0 if bool(msg.result) else 1
                    if self._legacy_progress < 95:
                        self._record_incomplete_locked(11, self._legacy_gait, now)

    # ------------------------------------------------------------------
    # Result/service actions
    # ------------------------------------------------------------------
    def _record_incomplete_locked(self, mode, gait, now=None):
        now = time.monotonic() if now is None else float(now)
        self._last_incomplete_seq = int(self._response_seq)
        self._last_incomplete_mode = int(mode)
        self._last_incomplete_gait = int(gait)
        self._last_incomplete_rx_monotonic_s = now

    def _set_action_started_locked(self, legacy_target):
        now = time.monotonic()
        # A discrete action owns the motion manager.  From this point onward
        # the previous Servo session is never considered reusable.
        self._servo_active = False
        self._servo_motion_id = None
        self._invalidate_servo_ready_locked(update_timer_anchor=True)
        self._response_seq += 1
        self._response_rx_monotonic_s = now
        self._legacy_mode = int(legacy_target[0])
        self._legacy_gait = int(legacy_target[1])
        self._legacy_progress = 1
        self._legacy_switch_status = 0
        self._legacy_ori_error = 0
        self._legacy_footpos_error = 0
        self._record_incomplete_locked(*legacy_target, now=now)

    def _set_action_completed_locked(self, legacy_target, success, code):
        now = time.monotonic()
        # Even after a successful preset action, continuous locomotion must
        # perform a fresh START -> robot ACK -> DATA handshake.
        self._servo_active = False
        self._servo_motion_id = None
        self._invalidate_servo_ready_locked(update_timer_anchor=True)
        self._response_seq += 1
        self._response_rx_monotonic_s = now
        self._legacy_mode = int(legacy_target[0])
        self._legacy_gait = int(legacy_target[1])
        self._legacy_progress = 100 if success else 0
        self._legacy_switch_status = 0
        self._legacy_ori_error = 0 if success else 1
        self._legacy_footpos_error = 0
        self._last_action_complete = True
        self._last_action_success = bool(success)
        self._last_action_code = int(code)

    def _mark_action_transport_failure(self, legacy_target, code=-1):
        with self._lock:
            self._set_action_started_locked(legacy_target)
            self._set_action_completed_locked(legacy_target, False, code)

    def run_result_action(self, motion_id, *, legacy_target=(0, 0),
                          vel_des=None, rpy_des=None, pos_des=None,
                          acc_des=None, ctrl_point=None, foot_pose=None,
                          step_height=None, duration=0):
        self.stop_motion()
        legacy_target = (int(legacy_target[0]), int(legacy_target[1]))

        if not self._result_client.service_is_ready():
            ready = self._result_client.wait_for_service(
                timeout_sec=self.service_wait_timeout_s)
            if not ready:
                self.log.error(
                    '[REAL_CTRL] result service unavailable: {}'.format(
                        self.result_service))
                self._mark_action_transport_failure(legacy_target, code=-2)
                return False

        request = MotionResultCmd.Request()
        request.motion_id = int(motion_id)
        request.cmd_source = int(self.cmd_source)
        request.vel_des = _fit(vel_des, 3)
        request.rpy_des = _fit(rpy_des, 3)
        request.pos_des = _fit(pos_des, 3)
        request.acc_des = _fit(acc_des, 3)
        request.ctrl_point = _fit(ctrl_point, 3)
        request.foot_pose = _fit(foot_pose, 3)
        request.step_height = _fit(step_height, 2, default=0.05)
        request.duration = int(duration)

        with self._lock:
            self._action_generation += 1
            generation = self._action_generation
            self._last_action_target = legacy_target
            self._last_action_motion_id = int(motion_id)
            self._last_action_complete = False
            self._last_action_success = False
            self._last_action_code = 0
            self._set_action_started_locked(legacy_target)

        try:
            future = self._result_client.call_async(request)
        except Exception as exc:
            self.log.error('[REAL_CTRL] result service call failed: {}'.format(exc))
            self._mark_action_transport_failure(legacy_target, code=-3)
            return False

        def done_cb(done_future):
            success = False
            code = -4
            returned_motion_id = int(motion_id)
            try:
                response = done_future.result()
                success = bool(response.result)
                code = int(response.code)
                returned_motion_id = int(response.motion_id)
            except Exception as exc:
                self.log.error('[REAL_CTRL] result future failed: {}'.format(exc))

            with self._lock:
                if generation != self._action_generation:
                    return
                self._set_action_completed_locked(legacy_target, success, code)
            self.log.info(
                '[REAL_CTRL] ACTION done request_motion_id={} response_motion_id={} '
                'success={} code={}'.format(
                    motion_id, returned_motion_id, success, code))

        future.add_done_callback(done_cb)
        self.log.info(
            '[REAL_CTRL] ACTION start motion_id={} legacy=({}, {})'.format(
                motion_id, legacy_target[0], legacy_target[1]))
        return True

    def run_action(self, action_name, wait_finish=True):
        mapping = {
            'recovery_stand': (self.recovery_motion_id, (12, 0)),
            'emergency_stop': (self.emergency_motion_id, (0, 0)),
            'lie_down': (self.lie_down_motion_id, (7, 1)),
            'left_jump': (self.left_jump_motion_id, (16, 0)),
            'right_jump': (self.right_jump_motion_id, (16, 3)),
            'forward_jump': (self.forward_jump_motion_id, (16, 1)),
        }
        if action_name not in mapping:
            raise ValueError('unsupported physical action: {}'.format(action_name))
        motion_id, legacy_target = mapping[action_name]
        sent = self.run_result_action(motion_id, legacy_target=legacy_target)
        if not sent:
            return False
        return self.Wait_finish(*legacy_target) if wait_finish else True

    def recovery_stand(self, wait_finish=True):
        return self.run_action('recovery_stand', wait_finish=wait_finish)

    # ------------------------------------------------------------------
    # Legacy command compatibility layer
    # ------------------------------------------------------------------
    def _legacy_gait_to_motion(self, gait_id):
        gait_id = int(gait_id)
        if gait_id in self.legacy_gait_motion:
            return self.legacy_gait_motion[gait_id]
        self.log.warning(
            '[REAL_CTRL] unknown legacy gait_id={}, fallback motion_id={}'.format(
                gait_id, self.default_servo_motion_id))
        return self.default_servo_motion_id

    def Send_cmd(self, msg):
        mode = int(getattr(msg, 'mode', 0))
        gait = int(getattr(msg, 'gait_id', 0))
        vel = _fit(getattr(msg, 'vel_des', None), 3)
        rpy = _fit(getattr(msg, 'rpy_des', None), 3)
        pos = _fit(getattr(msg, 'pos_des', None), 3)
        step = _fit(getattr(msg, 'step_height', None), 2, default=0.05)

        if mode == 11:
            self.move(
                vel[0], vel[1], vel[2],
                step_height=max(step[0], step[1]),
                roll=rpy[0], pitch=rpy[1], yaw=rpy[2],
                body_height=pos[2],
                motion_id=self._legacy_gait_to_motion(gait),
                legacy_gait_id=gait,
                value=int(getattr(msg, 'value', 0)),
            )
            return True

        if mode == 12:
            # In the simulator mode=12 is overloaded: sometimes it is a true
            # RecoveryStand, sometimes the stage uses it as a zero-velocity
            # posture command.  Non-zero pose fields clearly mean the latter;
            # send them through the unlocked servo fields instead of invoking
            # motion_id=111.
            if _nonzero(rpy) or _nonzero(pos):
                self.move(
                    0.0, 0.0, 0.0,
                    step_height=max(step[0], step[1]),
                    roll=rpy[0], pitch=rpy[1], yaw=rpy[2],
                    body_height=pos[2],
                    motion_id=self.default_servo_motion_id,
                    legacy_gait_id=0,
                )
                return True
            return self.run_result_action(
                self.recovery_motion_id,
                legacy_target=(12, 0),
                step_height=[0.05, 0.05],
            )

        if mode == 16:
            jump_map = {
                0: self.left_jump_motion_id,
                3: self.right_jump_motion_id,
                1: self.forward_jump_motion_id,
            }
            motion_id = jump_map.get(gait)
            if motion_id is None:
                self.log.error(
                    '[REAL_CTRL] unsupported legacy jump mode=16 gait={}'.format(gait))
                self._mark_action_transport_failure((mode, gait), code=-16)
                return False
            return self.run_result_action(
                motion_id,
                legacy_target=(mode, gait),
                step_height=[0.05, 0.05],
            )

        if mode == 0:
            return self.run_result_action(
                self.emergency_motion_id, legacy_target=(0, gait))

        if mode == 7 and gait == 1:
            return self.run_result_action(
                self.lie_down_motion_id, legacy_target=(7, 1))

        self.log.error(
            '[REAL_CTRL] unsupported legacy command mode={} gait={}; command ignored'.format(
                mode, gait))
        self._mark_action_transport_failure((mode, gait), code=-99)
        return False

    def Send_cmd_with_response_barrier(self, msg):
        with self._lock:
            baseline = int(self._response_seq)
        # The synthetic "progress started" snapshot is written inside
        # Send_cmd().  Capture the barrier time first so Stage 5 can correctly
        # recognise that snapshot as post-command progress.
        sent_monotonic_s = time.monotonic()
        self.Send_cmd(msg)
        return baseline, sent_monotonic_s

    def stop_motion_with_response_barrier(self, legacy_target=(12, 0)):
        """Physical SERVO_END plus a synthetic ACK for legacy Stage-5 polling."""
        with self._lock:
            baseline = int(self._response_seq)
        sent = time.monotonic()
        self.stop_motion()
        with self._lock:
            self._last_action_target = tuple(legacy_target)
            self._last_action_complete = False
            self._last_action_success = True
            self._set_action_started_locked(tuple(legacy_target))
            self._set_action_completed_locked(tuple(legacy_target), True, 0)
        return baseline, sent

    def Wait_finish(self, mode, gait_id):
        target = (int(mode), int(gait_id))
        deadline = time.monotonic() + self.action_wait_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._last_action_target == target and self._last_action_complete:
                    return bool(self._last_action_success)
            time.sleep(0.01)
        self.log.error(
            '[REAL_CTRL] Wait_finish timeout target=({}, {}) after {:.1f}s'.format(
                target[0], target[1], self.action_wait_timeout_s))
        return False

    # ------------------------------------------------------------------
    # Compatibility status APIs used by Stage 4/5
    # ------------------------------------------------------------------
    def response_snapshot(self):
        with self._lock:
            return {
                'seq': int(self._response_seq),
                'rx_monotonic_s': self._response_rx_monotonic_s,
                'mode': int(self._legacy_mode),
                'gait_id': int(self._legacy_gait),
                'order_process_bar': int(self._legacy_progress),
                'switch_status': int(self._legacy_switch_status),
                'ori_error': int(self._legacy_ori_error),
                'footpos_error': int(self._legacy_footpos_error),
                'motor_error': [0] * 12,
                'last_incomplete_seq': int(self._last_incomplete_seq),
                'last_incomplete_mode': int(self._last_incomplete_mode),
                'last_incomplete_gait_id': int(self._last_incomplete_gait),
                'last_incomplete_rx_monotonic_s': (
                    self._last_incomplete_rx_monotonic_s),
                'real_result_code': int(self._last_action_code),
                'real_result_success': bool(self._last_action_success),
                'real_motion_id': self._last_action_motion_id,
            }

    def get_status(self):
        snapshot = self.response_snapshot()
        rx = snapshot['rx_monotonic_s']
        return {
            'mode': snapshot['mode'],
            'gait': snapshot['gait_id'],
            'progress': snapshot['order_process_bar'],
            'switch_status': snapshot['switch_status'],
            'ori_error': snapshot['ori_error'],
            'footpos_error': snapshot['footpos_error'],
            'motor_error': tuple(snapshot['motor_error']),
            'age_s': (
                math.inf if snapshot['seq'] <= 0 or rx is None
                else max(0.0, time.monotonic() - float(rx))
            ),
        }
