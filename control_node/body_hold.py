#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold the physical robot in the post-bar target-detection body pose.

This is a stationary hardware debug tool.  It sends zero velocity while
holding the same body height and pitch used by Stage 4 after clearing the
height-limit bar.  Stop every other motion/control node before starting it.
"""

from __future__ import print_function

import sys
import time
from pathlib import Path

# Support direct source-tree execution:
# python3 control_node/bar_target_pose_hold.py
if __package__ is None or __package__ == '':
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import rclpy
from rclpy.node import Node

from control_node.my_gait import Robot_Ctrl
from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt


class BarTargetPoseHoldNode(Node):
    def __init__(self):
        Node.__init__(self, 'bar_target_pose_hold')

        # These defaults exactly match Stage 4 BAR_SEARCH_TARGET.
        self.declare_parameter('body_height', 0.20)
        self.declare_parameter('pitch', 0.15)
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('step_height', 0.02)
        self.declare_parameter('publish_hz', 40.0)
        self.declare_parameter('restore_height', 0.25)

        self.body_height = float(self.get_parameter('body_height').value)
        self.pitch = float(self.get_parameter('pitch').value)
        self.roll = float(self.get_parameter('roll').value)
        self.step_height = float(self.get_parameter('step_height').value)
        self.publish_hz = max(
            1.0, float(self.get_parameter('publish_hz').value))
        self.restore_height = float(
            self.get_parameter('restore_height').value)

        self.ctrl = Robot_Ctrl()
        self.msg = robot_control_cmd_lcmt()
        self.ctrl.run()
        self.closed = False

        self.get_logger().warn(
            'STATIONARY POSE HOLD STARTED. Stop all other control nodes. '
            'height=%.3f, pitch=%.3frad, roll=%.3f, step=%.3f' % (
                self.body_height, self.pitch, self.roll, self.step_height))
        self.get_logger().warn(
            'The robot will remain stationary in this pose until Ctrl+C.')

        # Send once immediately, then keep refreshing the same command.
        self._send_pose(self.body_height, self.pitch, self.roll)
        self.timer = self.create_timer(
            1.0 / self.publish_hz, self._hold_callback)

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 1

    def _send_pose(self, body_height, pitch, roll):
        self.msg.mode = 11
        self.msg.gait_id = 3
        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.msg.rpy_des = [float(roll), float(pitch), 0.0]
        self.msg.pos_des = [0.0, 0.0, float(body_height)]
        self.msg.step_height = [self.step_height, self.step_height]
        self._inc_life_count()
        self.ctrl.Send_cmd(self.msg)

    def _hold_callback(self):
        if self.closed:
            return
        self._send_pose(self.body_height, self.pitch, self.roll)
        self.get_logger().info(
            '[POSE_HOLD] vel=(0,0,0), height=%.3f, pitch=%.3f, roll=%.3f' % (
                self.body_height, self.pitch, self.roll),
            throttle_duration_sec=1.0)

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, 'timer'):
            self.timer.cancel()

        if self.ctrl is None:
            return
        try:
            self.get_logger().warn(
                'Restoring level pose: height=%.3f, pitch=0, roll=0' %
                self.restore_height)
            # Repeat briefly so the restore command is not lost at shutdown.
            for _ in range(10):
                self._send_pose(self.restore_height, 0.0, 0.0)
                time.sleep(0.05)
        finally:
            self.ctrl.quit()
            self.ctrl = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BarTargetPoseHoldNode()
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
