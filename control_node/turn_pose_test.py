#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CyberDog fixed-time in-place turn test with body-pose control.

Sequence:
    RecoveryStand (optional)
      -> zero-velocity in-place gait with body pose
      -> fixed wz turn with the SAME body pose
      -> zero-velocity short hold
      -> restore level pose (optional)
      -> Servo END

Only run this node when no other node is publishing motion commands.
"""

import time

import rclpy
from rclpy.node import Node

from control_node.robot_interface import create_robot_controller


class TurnPoseTestNode(Node):
    def __init__(self):
        super().__init__('turn_pose_test')

        # ------------------------------------------------------------
        # Test parameters
        # ------------------------------------------------------------
        self.declare_parameter('platform', 'real')

        # Phase timing
        self.declare_parameter('in_place_time_s', 1.0)
        self.declare_parameter('turn_time_s', 3.85) # 180 7.7  90 3.85
        self.declare_parameter('turn_wz', 0.60)  # + left, - right
        self.declare_parameter('hold_after_turn_s', 0.50)

        # Body pose: these are sent from the FIRST in-place command onward.
        self.declare_parameter('body_height', 0.20)
        self.declare_parameter('body_roll', 0.0)
        self.declare_parameter('body_pitch', 0.0)
        self.declare_parameter('body_yaw', 0.0)
        self.declare_parameter('step_height', 0.02)

        # Startup / shutdown
        self.declare_parameter('do_recovery_stand', True)
        self.declare_parameter('restore_after_test', True)
        self.declare_parameter('restore_body_height', 0.25)
        self.declare_parameter('restore_wait_s', 1.0)
        self.declare_parameter('control_hz', 20.0)

        # ------------------------------------------------------------
        # Parameters required by the physical-robot adapter.
        # Defaults match the current real-robot profile in this project.
        # ------------------------------------------------------------
        self.declare_parameter(
            'real_motion_servo_cmd_topic',
            '/mi_desktop_48_b0_2d_7b_00_e2/motion_servo_cmd')
        self.declare_parameter(
            'real_motion_servo_response_topic',
            '/mi_desktop_48_b0_2d_7b_00_e2/motion_servo_response')
        self.declare_parameter(
            'real_motion_result_service',
            '/mi_desktop_48_b0_2d_7b_00_e2/motion_result_cmd')
        self.declare_parameter('real_cmd_source', 0)
        self.declare_parameter('real_default_servo_motion_id', 303)
        self.declare_parameter('real_servo_publish_hz', 20.0)
        self.declare_parameter('real_servo_start_repeat', 5)
        self.declare_parameter('real_servo_end_repeat', 5)
        self.declare_parameter('real_servo_start_settle_s', 0.0)
        self.declare_parameter('real_servo_start_ack_timeout_s', 2.0)
        self.declare_parameter('real_motion_service_wait_timeout_s', 2.0)
        self.declare_parameter('real_action_wait_timeout_s', 45.0)
        self.declare_parameter('real_recovery_motion_id', 111)
        self.declare_parameter('real_emergency_stop_motion_id', 0)
        self.declare_parameter('real_lie_down_motion_id', 101)
        self.declare_parameter('real_left_jump_motion_id', 134)
        self.declare_parameter('real_right_jump_motion_id', 135)
        self.declare_parameter('real_forward_jump_motion_id', 132)
        self.declare_parameter('real_legacy_gait0_motion_id', 303)
        self.declare_parameter('real_legacy_gait1_motion_id', 303)
        self.declare_parameter('real_legacy_gait3_motion_id', 303)
        self.declare_parameter('real_legacy_gait27_motion_id', 303)

        self.in_place_time_s = max(
            0.0, float(self.get_parameter('in_place_time_s').value))
        self.turn_time_s = max(
            0.0, float(self.get_parameter('turn_time_s').value))
        self.turn_wz = float(self.get_parameter('turn_wz').value)
        self.hold_after_turn_s = max(
            0.0, float(self.get_parameter('hold_after_turn_s').value))

        self.body_height = float(self.get_parameter('body_height').value)
        self.body_roll = float(self.get_parameter('body_roll').value)
        self.body_pitch = float(self.get_parameter('body_pitch').value)
        self.body_yaw = float(self.get_parameter('body_yaw').value)
        self.step_height = float(self.get_parameter('step_height').value)

        self.do_recovery_stand = bool(
            self.get_parameter('do_recovery_stand').value)
        self.restore_after_test = bool(
            self.get_parameter('restore_after_test').value)
        self.restore_body_height = float(
            self.get_parameter('restore_body_height').value)
        self.restore_wait_s = max(
            0.0, float(self.get_parameter('restore_wait_s').value))
        self.control_hz = max(
            5.0, float(self.get_parameter('control_hz').value))

        self.ctrl = create_robot_controller(self)
        self.ctrl.run()
        self.closed = False
        self.phase = 'INIT'
        self.phase_start = time.monotonic()

        self.get_logger().warn(
            'TURN TEST START. Stop every other robot motion/control node first.')
        self.get_logger().info(
            'pose: height=%.3f roll=%.3f pitch=%.3f yaw=%.3f step=%.3f; '
            'in_place=%.2fs turn_wz=%.3f turn_time=%.2fs' % (
                self.body_height,
                self.body_roll,
                self.body_pitch,
                self.body_yaw,
                self.step_height,
                self.in_place_time_s,
                self.turn_wz,
                self.turn_time_s,
            ))

        if self.do_recovery_stand:
            self.get_logger().info('[INIT] RecoveryStand...')
            ok = bool(self.ctrl.recovery_stand(wait_finish=True))
            if not ok:
                self.get_logger().warn(
                    '[INIT] RecoveryStand did not report success; continuing.')

        # IMPORTANT: apply body pose immediately in the in-place gait.
        # The first move() may wait for physical Servo START/ACK, so start the
        # phase timer only AFTER that call returns successfully.
        if not self._send_motion(0.0):
            raise RuntimeError('failed to start locomotion Servo session')
        self.phase = 'IN_PLACE'
        self.phase_start = time.monotonic()
        self.get_logger().warn(
            '[IN_PLACE] zero velocity + requested body pose is now active')

        self.timer = self.create_timer(
            1.0 / self.control_hz, self._timer_callback)

    def _send_motion(self, wz: float) -> bool:
        return bool(self.ctrl.move(
            0.0,
            0.0,
            float(wz),
            step_height=self.step_height,
            roll=self.body_roll,
            pitch=self.body_pitch,
            yaw=self.body_yaw,
            body_height=self.body_height,
            legacy_gait_id=3,
        ))

    def _elapsed(self) -> float:
        return time.monotonic() - self.phase_start

    def _enter_turn(self):
        self._send_motion(self.turn_wz)
        self.phase = 'TURN'
        self.phase_start = time.monotonic()
        direction = 'LEFT' if self.turn_wz > 0.0 else 'RIGHT'
        if self.turn_wz == 0.0:
            direction = 'NONE'
        self.get_logger().warn(
            '[TURN] %s, wz=%.3f rad/s, fixed time=%.3fs; SAME pose kept' % (
                direction, self.turn_wz, self.turn_time_s))

    def _enter_hold(self):
        self._send_motion(0.0)
        self.phase = 'HOLD'
        self.phase_start = time.monotonic()
        self.get_logger().warn(
            '[HOLD] turn finished; zero velocity, SAME pose kept')

    def _enter_restore(self):
        if not self.restore_after_test:
            self._finish()
            return

        # Restore level pose while keeping zero velocity.
        self.ctrl.move(
            0.0, 0.0, 0.0,
            step_height=self.step_height,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            body_height=self.restore_body_height,
            legacy_gait_id=3,
        )
        self.phase = 'RESTORE'
        self.phase_start = time.monotonic()
        self.get_logger().warn(
            '[RESTORE] level pose: height=%.3f, roll=pitch=yaw=0' %
            self.restore_body_height)

    def _finish(self):
        if self.phase == 'DONE':
            return
        self.phase = 'DONE'
        try:
            self.ctrl.stop_motion()
        except Exception as exc:
            self.get_logger().error('stop_motion failed: %s' % exc)
        self.get_logger().warn('[DONE] test finished; Servo session ended')

    def _timer_callback(self):
        if self.closed or self.phase == 'DONE':
            return

        elapsed = self._elapsed()

        if self.phase == 'IN_PLACE':
            # Keep refreshing zero velocity AND the requested pose.
            self._send_motion(0.0)
            self.get_logger().info(
                '[IN_PLACE] %.2f/%.2fs | pose=(r=%.3f,p=%.3f,y=%.3f,h=%.3f)' % (
                    elapsed, self.in_place_time_s,
                    self.body_roll, self.body_pitch,
                    self.body_yaw, self.body_height),
                throttle_duration_sec=0.5)
            if elapsed >= self.in_place_time_s:
                self._enter_turn()

        elif self.phase == 'TURN':
            # Only wz changes. Pose parameters are identical to IN_PLACE.
            self._send_motion(self.turn_wz)
            self.get_logger().info(
                '[TURN] %.2f/%.2fs | wz=%.3f | pose=(r=%.3f,p=%.3f,y=%.3f,h=%.3f)' % (
                    elapsed, self.turn_time_s, self.turn_wz,
                    self.body_roll, self.body_pitch,
                    self.body_yaw, self.body_height),
                throttle_duration_sec=0.5)
            if elapsed >= self.turn_time_s:
                self._enter_hold()

        elif self.phase == 'HOLD':
            self._send_motion(0.0)
            if elapsed >= self.hold_after_turn_s:
                self._enter_restore()

        elif self.phase == 'RESTORE':
            self.ctrl.move(
                0.0, 0.0, 0.0,
                step_height=self.step_height,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                body_height=self.restore_body_height,
                legacy_gait_id=3,
            )
            if elapsed >= self.restore_wait_s:
                self._finish()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        try:
            if hasattr(self, 'timer'):
                self.timer.cancel()
        except Exception:
            pass
        try:
            self.ctrl.stop_motion()
        except Exception:
            pass
        try:
            self.ctrl.quit()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TurnPoseTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
