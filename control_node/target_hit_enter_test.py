#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-4 target approach/alignment + manual-enter hit test.

Purpose
-------
Reuse Stage4Node's REAL detector/alignment/hit thresholds for:
    - cola
    - blue_ball
    - white_ball

Test flow
---------
    SEARCH/APPROACH:
        Move forward with the same target-search body pose/gait as Stage 4.
        Detect all three target classes every frame and align to the best target.

    WAIT_ENTER:
        As soon as Stage 4's normal HIT threshold is reached, command zero
        velocity while KEEPING the same low-body + forward-pitch gait pose.
        A background stdin thread waits for ENTER so the ROS executor is never
        blocked.

    HIT:
        After ENTER, use the corresponding Stage-4 hit speed/duration.

    DONE:
        Stop at zero velocity, keep the same body pose, and wait for Ctrl+C.

IMPORTANT: run this test with every other robot-motion publisher stopped.
"""

import os
import sys
import threading
import time

# Allow BOTH:
#   python3 target_hit_enter_test.py
# from control_node/control_node, and:
#   python3 -m control_node.target_hit_enter_test
if __package__ in (None, ''):
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

import cv2
import rclpy

from control_node.stage4_node import Stage4Node


class TargetHitEnterTestNode(Stage4Node):
    SEARCH = 'TEST_SEARCH'
    ALIGN = 'TEST_ALIGN'
    WAIT_ENTER = 'TEST_WAIT_ENTER'
    HIT = 'TEST_HIT'
    DONE = 'TEST_DONE'

    def __init__(self):
        # This loads the exact Stage-4 detectors, target thresholds, gait/body
        # pose parameters and physical-robot adapter parameters.
        super().__init__()

        # Test-only parameters.  Stage-4 parameters themselves can still be
        # overridden normally with --ros-args -p ...
        self.declare_parameter('test_auto_start_delay_s', 0.8)
        self.declare_parameter('test_rgb_timeout_s', 0.8)
        self.declare_parameter('test_do_recovery_stand', False)

        self.test_auto_start_delay_s = max(
            0.05, float(self.get_parameter('test_auto_start_delay_s').value))
        self.test_rgb_timeout_s = max(
            0.1, float(self.get_parameter('test_rgb_timeout_s').value))
        self.test_do_recovery_stand = bool(
            self.get_parameter('test_do_recovery_stand').value)

        self.state = self.SEARCH
        self.locked_target = None
        self.latest_target = None
        self.stable_target_type = None
        self.target_stable_count = 0

        self._enter_event = threading.Event()
        self._enter_thread = None
        self._hit_start_monotonic = None
        self._auto_started = False
        self._last_processed_rgb_seq = -1

        # Start only after rclpy.spin() is running.  The inherited control timer
        # stays idle until self.active becomes True.
        self._auto_start_timer = self.create_timer(
            self.test_auto_start_delay_s, self._auto_start_once)

        self.get_logger().warn(
            'TARGET HIT TEST loaded. STOP all other motion/control nodes before use.')
        self.get_logger().info(
            'Stage4 target pose: height=%.3f, pitch=%.3f, roll=0.000, step=%.3f, gait=3' % (
                self.obstacle_low_body_height,
                self.obstacle_target_forward_pitch,
                self.step_height_cmd,
            ))
        self.get_logger().info(
            'Speeds: search=%.3f, align_far=%.3f, align_near=%.3f; '
            'hit blue=%.3f, white=%.3f, cola=%.3f' % (
                self.target_search_forward_speed,
                self.align_forward_speed_far,
                self.align_forward_speed_near,
                self.hit_params['blue_ball']['speed'],
                self.hit_params['white_ball']['speed'],
                self.hit_params['cola']['speed'],
            ))

    # ------------------------------------------------------------------
    # This is a standalone test. Ignore /mission/active_stage so a latched
    # mission-manager message cannot unexpectedly deactivate this node.
    # ------------------------------------------------------------------
    def _mission_cb(self, msg):
        return

    def _auto_start_once(self):
        if self._auto_started:
            return
        self._auto_started = True
        try:
            self._auto_start_timer.cancel()
        except Exception:
            pass

        self.start_ctrl()
        self.active = True
        self.finished = False

        if self.test_do_recovery_stand:
            self.get_logger().warn('[INIT] sending RecoveryStand before test')
            ok = bool(self.Ctrl.recovery_stand(wait_finish=True))
            if not ok:
                self.get_logger().warn(
                    '[INIT] RecoveryStand did not report success; continuing test')

        self.on_activated()

    def on_activated(self):
        # IMPORTANT: this is the same pose used by Stage4 in
        # SEARCH_TARGET_AFTER_TURNS / APPROACH_AND_ALIGN_TARGET.
        self.send_motion_cmd(
            0.0, 0.0, 0.0,
            roll=0.0,
            pitch=self.obstacle_target_forward_pitch,
            body_height=self.obstacle_low_body_height,
            step_height=0.02,
        )
        self.state = self.SEARCH
        self.locked_target = None
        self.latest_target = None
        self.stable_target_type = None
        self.target_stable_count = 0
        self._enter_event.clear()
        self._hit_start_monotonic = None
        self.get_logger().warn(
            '[SEARCH] started: moving forward, detecting cola/blue_ball/white_ball')

    def _hold_stage4_target_pose(self):
        """Zero velocity while preserving the Stage-4 target-search pose/gait."""
        self.send_motion_cmd(
            0.0, 0.0, 0.0,
            roll=0.0,
            pitch=self.obstacle_target_forward_pitch,
            body_height=self.obstacle_low_body_height,
            step_height=0.02,
        )

    def _start_enter_thread(self):
        if self._enter_thread is not None and self._enter_thread.is_alive():
            return

        target_type = (
            self.locked_target.det_type if self.locked_target is not None else 'unknown')

        def _wait_for_enter():
            try:
                print('\n' + '=' * 72, flush=True)
                print('已满足第四赛段撞击条件，机器狗已停车并保持当前低姿态。', flush=True)
                print('锁定目标: %s' % target_type, flush=True)
                input('确认可以撞击后，按 ENTER 开始撞击 >>> ')
                self._enter_event.set()
            except EOFError:
                # No terminal stdin: stay stopped rather than hitting automatically.
                print('\n[WAIT_ENTER] stdin 不可用；保持停车，不会自动撞击。', flush=True)
            except Exception as exc:
                print('\n[WAIT_ENTER] 输入线程异常: %s；保持停车。' % exc, flush=True)

        self._enter_thread = threading.Thread(
            target=_wait_for_enter,
            name='stage4_target_hit_enter_wait',
            daemon=True,
        )
        self._enter_thread.start()

    def _draw_test_debug(self, frame, candidates, chosen):
        if not self.show_debug_vis:
            return
        self.update_debug_visualization(
            frame,
            obstacle_candidates=[],
            obstacle_pair=None,
            dashed=None,
            target_candidates=candidates,
            chosen_target=chosen,
            final_yellow_line=None,
            bar_det=None,
        )

    def _log_candidates(self, candidates):
        if not candidates:
            return 'none'
        parts = []
        for det in candidates:
            metrics = self.target_visual_metrics(det)
            parts.append(
                '%s(x=%d,area=%s,r=%s)' % (
                    det.det_type,
                    int(det.center_img[0]),
                    ('%.5f' % metrics['area_ratio'])
                    if metrics['area_ratio'] is not None else 'None',
                    ('%.1f' % metrics['radius_px'])
                    if metrics['radius_px'] is not None else 'None',
                ))
        return ', '.join(parts)

    def stage_control_loop(self):
        # --------------------------------------------------------------
        # WAIT/HIT/DONE must keep working even if camera frames pause.
        # --------------------------------------------------------------
        if self.state == self.WAIT_ENTER:
            self._hold_stage4_target_pose()
            if self._enter_event.is_set():
                self._enter_event.clear()
                self._hit_start_monotonic = time.monotonic()
                self.state = self.HIT
                target_type = (
                    self.locked_target.det_type
                    if self.locked_target is not None else 'unknown')
                params = self.hit_params.get(target_type, {})
                self.get_logger().warn(
                    '[HIT] ENTER received: target=%s, speed=%.3f, duration=%.3fs' % (
                        target_type,
                        float(params.get('speed', 0.0)),
                        float(params.get('duration_s', 0.0)),
                    ))
            return

        if self.state == self.HIT:
            if self.locked_target is None:
                self.get_logger().error('[HIT] lost locked target object; stop test')
                self._hold_stage4_target_pose()
                self.state = self.DONE
                return

            params = self.hit_params.get(
                self.locked_target.det_type,
                {'speed': 0.20, 'duration_s': 0.85},
            )
            elapsed = time.monotonic() - float(self._hit_start_monotonic)
            duration = float(params['duration_s'])

            if elapsed >= duration:
                self._hold_stage4_target_pose()
                self.state = self.DONE
                self.get_logger().warn(
                    '[DONE] hit finished: target=%s, elapsed=%.3f/%.3fs. '
                    'Robot is stopped; Ctrl+C to exit.' % (
                        self.locked_target.det_type, elapsed, duration))
                return

            # Stage4 HIT_TARGET sends only vx; send_motion_cmd keeps the same
            # cached low-body/forward-pitch pose and gait=3.
            self.send_motion_cmd(float(params['speed']), 0.0, 0.0)
            self.get_logger().info(
                '[HIT] target=%s elapsed=%.3f/%.3fs vx=%.3f' % (
                    self.locked_target.det_type,
                    elapsed,
                    duration,
                    float(params['speed']),
                ),
                throttle_duration_sec=0.2,
            )
            return

        if self.state == self.DONE:
            self._hold_stage4_target_pose()
            return

        # --------------------------------------------------------------
        # SEARCH / ALIGN are vision-driven.
        # --------------------------------------------------------------
        if self.latest_bgr is None:
            self._hold_stage4_target_pose()
            self.get_logger().warn(
                '[VISION] waiting for RGB frame on %s' % self.rgb_topic,
                throttle_duration_sec=1.0,
            )
            return

        rgb_age = self.rgb_age_s()
        if rgb_age is not None and rgb_age > self.test_rgb_timeout_s:
            self._hold_stage4_target_pose()
            self.get_logger().error(
                '[VISION] RGB stale %.3fs > %.3fs; STOP and keep body pose' % (
                    rgb_age, self.test_rgb_timeout_s),
                throttle_duration_sec=1.0,
            )
            return

        # Process each decoded camera frame at most once.  The physical motion
        # adapter itself continues publishing the last command at its servo Hz.
        if self.latest_rgb_seq == self._last_processed_rgb_seq:
            return
        self._last_processed_rgb_seq = self.latest_rgb_seq

        frame = self.latest_bgr
        candidates = self.detect_all_targets(frame)
        target = self.choose_best_target(candidates)
        self._draw_test_debug(frame, candidates, target)

        # --------------------------------------------------------------
        # SEARCH: exact Stage4 search behavior: forward at search speed and
        # require target_stable_frames before entering visual alignment.
        # --------------------------------------------------------------
        if self.state == self.SEARCH:
            self.send_motion_cmd(
                self.target_search_forward_speed, 0.0, 0.0,
                roll=0.0,
                pitch=self.obstacle_target_forward_pitch,
                body_height=self.obstacle_low_body_height,
                step_height=0.02,
            )

            if target is None:
                self.stable_target_type = None
                self.target_stable_count = 0
                self.get_logger().info(
                    '[SEARCH] detections=none, vx=%.3f' %
                    self.target_search_forward_speed,
                    throttle_duration_sec=0.5,
                )
                return

            if self.stable_target_type == target.det_type:
                self.target_stable_count += 1
            else:
                self.stable_target_type = target.det_type
                self.target_stable_count = 1

            self.latest_target = target
            self.get_logger().info(
                '[SEARCH] all=[%s], chosen=%s center=%s stable=%d/%d' % (
                    self._log_candidates(candidates),
                    target.det_type,
                    str(target.center_img),
                    self.target_stable_count,
                    self.target_stable_frames,
                ),
                throttle_duration_sec=0.2,
            )

            if self.target_stable_count >= self.target_stable_frames:
                self.locked_target = target
                self.state = self.ALIGN
                self.get_logger().warn(
                    '[ALIGN] locked %s; start forward + lateral visual alignment' %
                    target.det_type)
            return

        # --------------------------------------------------------------
        # ALIGN: exact Stage4 target alignment and exact Stage4 hit threshold.
        # --------------------------------------------------------------
        if self.state == self.ALIGN:
            if target is None:
                self.locked_target = None
                self.stable_target_type = None
                self.target_stable_count = 0
                self.state = self.SEARCH
                self.send_motion_cmd(
                    self.target_search_forward_speed, 0.0, 0.0,
                    pitch=self.obstacle_target_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                    step_height=0.02,
                )
                self.get_logger().warn(
                    '[ALIGN] target lost; return to SEARCH')
                return

            self.locked_target = target
            vx, vy = self.compute_target_align_cmd(target)
            metrics = self.target_visual_metrics(target)

            if self.use_rgb_distance_triggers:
                hit_reached = self.target_visual_threshold_reached(target, 'hit')
                depth_m = None
            else:
                depth_m = self.estimate_depth_at_center(target.center_img)
                hit_reached = (
                    depth_m is not None and depth_m < self.hit_trigger_distance_m)

            if hit_reached:
                # CRITICAL: stop BEFORE changing state or waiting for stdin.
                self._hold_stage4_target_pose()
                self.state = self.WAIT_ENTER
                self._start_enter_thread()
                self.get_logger().warn(
                    '[WAIT_ENTER] HIT threshold reached -> STOP. '
                    'target=%s center=%s area=%s radius=%s depth=%s' % (
                        target.det_type,
                        str(target.center_img),
                        str(metrics['area_ratio']),
                        str(metrics['radius_px']),
                        str(depth_m),
                    ))
                return

            self.send_motion_cmd(
                vx, vy, 0.0,
                roll=0.0,
                pitch=self.obstacle_target_forward_pitch,
                body_height=self.obstacle_low_body_height,
                step_height=0.02,
            )
            self.get_logger().info(
                '[ALIGN] all=[%s], chosen=%s center=%s area=%s radius=%s '
                'hit=%s cmd=(%.3f, %.3f, 0.000)' % (
                    self._log_candidates(candidates),
                    target.det_type,
                    str(target.center_img),
                    str(metrics['area_ratio']),
                    str(metrics['radius_px']),
                    str(hit_reached),
                    vx,
                    vy,
                ),
                throttle_duration_sec=0.2,
            )

    def shutdown_test(self):
        try:
            if self.Ctrl is not None:
                # Stop velocity while preserving current target-search pose first.
                self._hold_stage4_target_pose()
                time.sleep(0.05)
        except Exception:
            pass
        try:
            self.stop_ctrl()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TargetHitEnterTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().warn('Shutting down target hit test...')
        node.shutdown_test()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
