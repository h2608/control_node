#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三赛段节点：弯道-直道-弯道黄线赛道，不越界跟随。

原 control_node_123456.py 的 P3_* 状态机（使用整合版 p3_control_loop，
即 FullCompetitionNode 里的重载版本）。P3_ALIGN_TRACK 完成或超时后向
任务控制节点上报完成（原来是 handoff_to_fourth_stage()）。
"""

from typing import Optional

import cv2
import numpy as np

import rclpy

from control_node.stage_common import StageNodeBase, clamp
from control_node.stage_entry import EntryPoint, StageEntryTable


def p3_entry_table():
    """第三赛段调试入口表（顺序即流程顺序）。"""
    states = ('P3_STAND_WAIT', 'P3_S_CURVE_CRUISE', 'P3_ALIGN_TRACK')
    return StageEntryTable(3, 'P3_STAND_WAIT', states, (
        EntryPoint('start', 'P3_STAND_WAIT', '低身站立等待，完整第三赛段'),
        EntryPoint('s_curve', 'P3_S_CURVE_CRUISE', '弯-直-弯黄线巡航',
                   requires=('赛道黄线必须在前向 RGB 视野内',)),
        EntryPoint('align', 'P3_ALIGN_TRACK', '收尾：赛道居中 + 航向对齐'),
    ))


class P3TrackVisionMixin:
    """第三赛段黄线赛道视觉与参数。

    第三赛段节点全程使用；第四赛段节点在 GLOBAL_FINAL_P3_ALIGN 状态
    直接调用 p3_process_yellow_track / p3_show_debug_window。
    """

    def p3_declare_params(self):
        self.declare_parameter('p3_stand_wait_sec', 2.0)
        self.declare_parameter('p3_stand_body_height', 0.20)
        self.declare_parameter('p3_stand_pitch', 0.17)
        self.declare_parameter('p3_step_height', 0.05)
        self.declare_parameter('p3_align_step_height', 0.10)

        self.declare_parameter('p3_s_curve_duration_sec', 16.0)
        self.declare_parameter('p3_base_forward_speed', 0.35)
        self.declare_parameter('p3_min_forward_speed', 0.00)
        self.declare_parameter('p3_kp_turn', 1.2)
        self.declare_parameter('p3_kp_lat', 0.2)
        self.declare_parameter('p3_kd_slowdown', 0.10)
        self.declare_parameter('p3_vision_timeout_sec', 1.0)
        self.declare_parameter('p3_fallback_forward_speed', 0.10)

        self.declare_parameter('p3_align_max_duration_sec', 8.0)
        self.declare_parameter('p3_align_lat_tol', 0.006)
        self.declare_parameter('p3_align_yaw_tol', 0.006)
        self.declare_parameter('p3_align_lat_gain', 0.6)
        self.declare_parameter('p3_align_yaw_gain', 2.0)
        self.declare_parameter('p3_align_lat_max', 0.15)
        self.declare_parameter('p3_align_yaw_max', 0.30)
        self.declare_parameter('p3_align_search_vx', 0.10)
        self.declare_parameter('p3_align_search_wz', 0.14)

        self.declare_parameter('p3_yellow_h_min', 20)
        self.declare_parameter('p3_yellow_h_max', 40)
        self.declare_parameter('p3_yellow_s_min', 50)
        self.declare_parameter('p3_yellow_s_max', 255)
        self.declare_parameter('p3_yellow_v_min', 150)
        self.declare_parameter('p3_yellow_v_max', 255)
        self.declare_parameter('p3_crop_left_ratio', 0.05)
        self.declare_parameter('p3_crop_right_ratio', 0.95)
        self.declare_parameter('p3_mid_top_ratio', 0.85)
        self.declare_parameter('p3_mid_bottom_ratio', 0.95)
        self.declare_parameter('p3_near_top_ratio', 0.95)
        self.declare_parameter('p3_near_bottom_ratio', 1.00)

        self.declare_parameter('p3_align_near_y_ratio', 0.90)
        self.declare_parameter('p3_align_far_y_ratio', 0.70)
        self.declare_parameter('p3_align_roi_left_ratio', 0.05)
        self.declare_parameter('p3_align_roi_right_ratio', 0.95)
        self.declare_parameter('p3_align_min_gap_px', 30)

    def p3_load_params(self):
        self.p3_stand_wait_sec = float(self.get_parameter('p3_stand_wait_sec').value)
        self.p3_stand_body_height = float(self.get_parameter('p3_stand_body_height').value)
        self.p3_stand_pitch = float(self.get_parameter('p3_stand_pitch').value)
        self.p3_step_height = float(self.get_parameter('p3_step_height').value)
        self.p3_align_step_height = float(self.get_parameter('p3_align_step_height').value)

        self.p3_s_curve_duration_sec = float(self.get_parameter('p3_s_curve_duration_sec').value)
        self.p3_base_forward_speed = float(self.get_parameter('p3_base_forward_speed').value)
        self.p3_min_forward_speed = float(self.get_parameter('p3_min_forward_speed').value)
        self.p3_kp_turn = float(self.get_parameter('p3_kp_turn').value)
        self.p3_kp_lat = float(self.get_parameter('p3_kp_lat').value)
        self.p3_kd_slowdown = float(self.get_parameter('p3_kd_slowdown').value)
        self.p3_vision_timeout_sec = float(self.get_parameter('p3_vision_timeout_sec').value)
        self.p3_fallback_forward_speed = float(self.get_parameter('p3_fallback_forward_speed').value)

        self.p3_align_max_duration_sec = float(self.get_parameter('p3_align_max_duration_sec').value)
        self.p3_align_lat_tol = float(self.get_parameter('p3_align_lat_tol').value)
        self.p3_align_yaw_tol = float(self.get_parameter('p3_align_yaw_tol').value)
        self.p3_align_lat_gain = float(self.get_parameter('p3_align_lat_gain').value)
        self.p3_align_yaw_gain = float(self.get_parameter('p3_align_yaw_gain').value)
        self.p3_align_lat_max = float(self.get_parameter('p3_align_lat_max').value)
        self.p3_align_yaw_max = float(self.get_parameter('p3_align_yaw_max').value)
        self.p3_align_search_vx = float(self.get_parameter('p3_align_search_vx').value)
        self.p3_align_search_wz = float(self.get_parameter('p3_align_search_wz').value)

        self.p3_yellow_h_min = int(self.get_parameter('p3_yellow_h_min').value)
        self.p3_yellow_h_max = int(self.get_parameter('p3_yellow_h_max').value)
        self.p3_yellow_s_min = int(self.get_parameter('p3_yellow_s_min').value)
        self.p3_yellow_s_max = int(self.get_parameter('p3_yellow_s_max').value)
        self.p3_yellow_v_min = int(self.get_parameter('p3_yellow_v_min').value)
        self.p3_yellow_v_max = int(self.get_parameter('p3_yellow_v_max').value)
        self.p3_crop_left_ratio = float(self.get_parameter('p3_crop_left_ratio').value)
        self.p3_crop_right_ratio = float(self.get_parameter('p3_crop_right_ratio').value)
        self.p3_mid_top_ratio = float(self.get_parameter('p3_mid_top_ratio').value)
        self.p3_mid_bottom_ratio = float(self.get_parameter('p3_mid_bottom_ratio').value)
        self.p3_near_top_ratio = float(self.get_parameter('p3_near_top_ratio').value)
        self.p3_near_bottom_ratio = float(self.get_parameter('p3_near_bottom_ratio').value)

        self.p3_align_near_y_ratio = float(self.get_parameter('p3_align_near_y_ratio').value)
        self.p3_align_far_y_ratio = float(self.get_parameter('p3_align_far_y_ratio').value)
        self.p3_align_roi_left_ratio = float(self.get_parameter('p3_align_roi_left_ratio').value)
        self.p3_align_roi_right_ratio = float(self.get_parameter('p3_align_roi_right_ratio').value)
        self.p3_align_min_gap_px = int(self.get_parameter('p3_align_min_gap_px').value)

    def p3_init_vision_caches(self):
        # =========================
        # 第三赛段运行缓存（p3_ 前缀）
        # =========================
        self.p3_state_start_time: Optional[float] = None
        self.p3_stand_sent = False
        self.p3_error_mid = 0.0
        self.p3_error_near = 0.0
        self.p3_last_update_time = 0.0
        self.p3_s4_lat = 0.0
        self.p3_s4_yaw = 0.0
        self.p3_s4_valid = 0.0
        self.p3_latest_mask = None
        self.p3_latest_mask_mid = None
        self.p3_latest_mask_near = None
        self.p3_align_near_center = -1.0
        self.p3_align_far_center = -1.0

    def p3_process_yellow_track(self, frame: np.ndarray):
        """
        合并 part3_vision.py：
        1. S 弯阶段：计算中距离 error_mid 和近距离 error_near。
        2. 出弯对齐阶段：双行前瞻，计算 s4_lat / s4_yaw / s4_valid。
        """
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([self.p3_yellow_h_min, self.p3_yellow_s_min, self.p3_yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.p3_yellow_h_max, self.p3_yellow_s_max, self.p3_yellow_v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        crop_left = int(width * self.p3_crop_left_ratio)
        crop_right = int(width * self.p3_crop_right_ratio)
        mask[:, 0:crop_left] = 0
        mask[:, crop_right:width] = 0

        mid_top = int(height * self.p3_mid_top_ratio)
        mid_bottom = int(height * self.p3_mid_bottom_ratio)
        near_top = int(height * self.p3_near_top_ratio)
        near_bottom = int(height * self.p3_near_bottom_ratio)

        mask_mid = np.zeros_like(mask)
        mask_near = np.zeros_like(mask)
        mask_mid[mid_top:mid_bottom, 0:width] = mask[mid_top:mid_bottom, 0:width]
        mask_near[near_top:near_bottom, 0:width] = mask[near_top:near_bottom, 0:width]

        err_mid = 0.0
        err_near = 0.0

        M_mid = cv2.moments(mask_mid)
        if M_mid['m00'] > 0:
            cx_mid = int(M_mid['m10'] / M_mid['m00'])
            dist_mid = abs(cx_mid - width / 2)
            force_mid = ((width / 2 - dist_mid) / (width / 2)) ** 3
            err_mid = float(force_mid) if cx_mid > width / 2 else -float(force_mid)

        M_near = cv2.moments(mask_near)
        if M_near['m00'] > 0:
            cx_near = int(M_near['m10'] / M_near['m00'])
            dist_near = abs(cx_near - width / 2)
            force_near = ((width / 2 - dist_near) / (width / 2)) ** 3
            err_near = float(force_near) if cx_near > width / 2 else -float(force_near)

        self.p3_error_mid = err_mid
        self.p3_error_near = err_near
        self.p3_last_update_time = self.now_sec()

        near_y = int(height * self.p3_align_near_y_ratio)
        far_y = int(height * self.p3_align_far_y_ratio)
        roi_left = int(width * self.p3_align_roi_left_ratio)
        roi_right = int(width * self.p3_align_roi_right_ratio)

        def get_road_center(y_idx: int) -> float:
            y_idx = max(0, min(height - 1, int(y_idx)))
            row = mask[y_idx, :]
            yellow_idx = np.where(row > 128)[0]
            valid_idx = [idx for idx in yellow_idx if roi_left < idx < roi_right]
            if len(valid_idx) < 2:
                return -1.0
            diffs = np.diff(valid_idx)
            if len(diffs) == 0:
                return -1.0
            max_gap_idx = int(np.argmax(diffs))
            if diffs[max_gap_idx] > self.p3_align_min_gap_px:
                l_edge = valid_idx[max_gap_idx]
                r_edge = valid_idx[max_gap_idx + 1]
                return 0.5 * (l_edge + r_edge)
            return -1.0

        cx_n = get_road_center(near_y)
        cx_f = get_road_center(far_y)
        self.p3_align_near_center = cx_n
        self.p3_align_far_center = cx_f

        if cx_n != -1 and cx_f != -1:
            self.p3_s4_lat = (width / 2.0 - cx_n) / (width / 2.0)
            self.p3_s4_yaw = (cx_n - cx_f) / (width / 2.0)
            self.p3_s4_valid = 1.0
        else:
            self.p3_s4_lat = 0.0
            self.p3_s4_yaw = 0.0
            self.p3_s4_valid = 0.0

        self.p3_latest_mask = mask
        self.p3_latest_mask_mid = mask_mid
        self.p3_latest_mask_near = mask_near

    def p3_show_debug_window(self, frame: np.ndarray):
        try:
            vis = frame.copy()
            height, width = vis.shape[:2]
            crop_left = int(width * self.p3_crop_left_ratio)
            crop_right = int(width * self.p3_crop_right_ratio)
            mid_top = int(height * self.p3_mid_top_ratio)
            mid_bottom = int(height * self.p3_mid_bottom_ratio)
            near_top = int(height * self.p3_near_top_ratio)
            near_y = int(height * self.p3_align_near_y_ratio)
            far_y = int(height * self.p3_align_far_y_ratio)
            roi_left = int(width * self.p3_align_roi_left_ratio)
            roi_right = int(width * self.p3_align_roi_right_ratio)

            cv2.putText(vis, f'P3 state={self.state}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(vis, f'err_mid={self.p3_error_mid:.3f} err_near={self.p3_error_near:.3f}', (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(vis, f's4_valid={self.p3_s4_valid:.1f} lat={self.p3_s4_lat:.3f} yaw={self.p3_s4_yaw:.3f}',
                        (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            vx, vy, wz = getattr(self, 'motion_cmd', (0.0, 0.0, 0.0))
            cv2.putText(
                vis,
                f'cmd vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}',
                (10, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            if self.state == getattr(self, 'GLOBAL_FINAL_P3_ALIGN', None):
                cv2.putText(
                    vis,
                    'P4 FINAL uses P3 align logic',
                    (10, 133),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

            cv2.line(vis, (crop_left, 0), (crop_left, height), (0, 255, 255), 2)
            cv2.line(vis, (crop_right, 0), (crop_right, height), (0, 255, 255), 2)
            cv2.line(vis, (0, mid_top), (width, mid_top), (255, 0, 0), 2)
            cv2.line(vis, (0, mid_bottom), (width, mid_bottom), (0, 255, 0), 2)
            cv2.line(vis, (0, near_top), (width, near_top), (0, 180, 255), 1)

            cv2.line(vis, (0, near_y), (width, near_y), (255, 255, 0), 1)
            cv2.line(vis, (0, far_y), (width, far_y), (255, 255, 0), 1)
            cv2.line(vis, (roi_left, 0), (roi_left, height), (255, 0, 255), 2)
            cv2.line(vis, (roi_right, 0), (roi_right, height), (255, 0, 255), 2)

            if self.p3_align_near_center != -1 and self.p3_align_far_center != -1:
                cv2.line(
                    vis,
                    (int(self.p3_align_near_center), near_y),
                    (int(self.p3_align_far_center), far_y),
                    (0, 0, 255),
                    3
                )

            cv2.imshow('part3_origin_debug', vis)
            if self.p3_latest_mask_mid is not None:
                cv2.imshow('part3_mask_mid', self.p3_latest_mask_mid)
            if self.p3_latest_mask_near is not None:
                cv2.imshow('part3_mask_near', self.p3_latest_mask_near)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'p3_show_debug_window failed: {e}', throttle_duration_sec=1.0)


class Stage3Node(P3TrackVisionMixin, StageNodeBase):

    STAGE_ID = 3

    def __init__(self):
        super().__init__('stage3_node', self.STAGE_ID)
        # 调试入口在 Stage3Node 而不是 P3TrackVisionMixin 里声明：第四赛段也
        # 复用这个 mixin（GLOBAL_FINAL_P3_ALIGN），它有自己的入口表。
        self.declare_parameter('p3_initial_state', 'default')
        self.p3_initial_state = self.resolve_stage_entry(
            p3_entry_table(),
            str(self.get_parameter('p3_initial_state').value))
        self.p3_declare_params()
        self.p3_load_params()
        self.p3_init_vision_caches()
        self.get_logger().info('Stage3Node ready.')

    # ============================================================
    # 状态切换（原 set_state 的 P3 部分）
    # ============================================================
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state
            if new_state.startswith('P3_'):
                self.p3_state_start_time = None
                if new_state == 'P3_STAND_WAIT':
                    self.p3_stand_sent = False

    def on_activated(self):
        self.p3_state_start_time = None
        self.state = self.p3_initial_state
        # 低身站立命令对任何入口都是安全的起手；下一个控制周期该入口接管。
        self.p3_send_stand_command()
        self.p3_stand_sent = True
        self.get_logger().info(f'[P3] activated, initial_state={self.state}')

    def on_rgb_frame(self, frame: np.ndarray):
        self.p3_process_yellow_track(frame)
        if self.show_debug_vis:
            self.p3_show_debug_window(frame)

    # ============================================================
    # 第三赛段：视觉处理 + 控制状态机
    # ============================================================
    def p3_elapsed_in_state(self) -> float:
        now = self.now_sec()
        if self.p3_state_start_time is None:
            self.p3_state_start_time = now
        self.p3_state_start_time = self.align_motion_timer_start(
            self.p3_state_start_time, now)
        return max(0.0, now - self.p3_state_start_time)

    def p3_send_stand_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.msg.rpy_des = [0.0, self.p3_stand_pitch, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.p3_stand_body_height]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info('[P3 CMD] STAND / LOW BODY', throttle_duration_sec=1.0)

    def p3_send_velocity_command(self, vx: float, vy: float, wz: float, step_height: Optional[float] = None):
        h = self.p3_step_height if step_height is None else float(step_height)
        self.Ctrl.move(
            float(vx), float(vy), float(wz),
            step_height=h,
            pitch=float(self.p3_stand_pitch),
            body_height=float(self.p3_stand_body_height),
            legacy_gait_id=3,
        )
        self.get_logger().info(
            f'[P3 CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}], step_height={h:.3f}',
            throttle_duration_sec=0.4
        )


    def stage_control_loop(self):
        elapsed = self.p3_elapsed_in_state()
        now = self.now_sec()

        if self.state == 'P3_STAND_WAIT':
            if not self.p3_stand_sent:
                self.p3_send_stand_command()
                self.p3_stand_sent = True
                self.get_logger().info(
                    f'P3_STAND_WAIT start: duration={self.p3_stand_wait_sec:.2f}s, '
                    f'body_height={self.p3_stand_body_height:.2f}, pitch={self.p3_stand_pitch:.2f}'
                )
            if elapsed >= self.p3_stand_wait_sec:
                self.set_state('P3_S_CURVE_CRUISE')
                return
            return

        if self.state == 'P3_S_CURVE_CRUISE':
            if elapsed >= self.p3_s_curve_duration_sec:
                self.set_state('P3_ALIGN_TRACK')
                return

            if now - self.p3_last_update_time < self.p3_vision_timeout_sec:
                err_mid = self.p3_error_mid
                err_near = self.p3_error_near
                raw_turn = (err_mid / 500.0 + err_near) * self.p3_kp_turn
                turn_speed = clamp(raw_turn, -0.5, 0.5)
                raw_lateral = err_near * self.p3_kp_lat
                lateral_speed = clamp(raw_lateral, -0.10, 0.10)
                speed_drop = abs(err_near) * self.p3_kd_slowdown
                forward_speed = max(self.p3_min_forward_speed, self.p3_base_forward_speed - speed_drop)
            else:
                forward_speed = self.p3_fallback_forward_speed
                lateral_speed = 0.0
                turn_speed = 0.0

            self.get_logger().info(
                f'P3_S_CURVE_CRUISE elapsed={elapsed:.2f}/{self.p3_s_curve_duration_sec:.2f}s | '
                f'err_mid={self.p3_error_mid:.3f}, err_near={self.p3_error_near:.3f}, '
                f'cmd=[{forward_speed:.3f},{lateral_speed:.3f},{turn_speed:.3f}]',
                throttle_duration_sec=0.5
            )
            self.p3_send_velocity_command(forward_speed, lateral_speed, turn_speed, step_height=self.p3_step_height)
            return

        if self.state == 'P3_ALIGN_TRACK':
            if elapsed >= self.p3_align_max_duration_sec:
                self.complete_stage('P3_ALIGN_TRACK timeout')
                return

            if self.p3_s4_valid > 0.5:
                err_lat = self.p3_s4_lat
                err_yaw = self.p3_s4_yaw
                if abs(err_lat) < self.p3_align_lat_tol and abs(err_yaw) < self.p3_align_yaw_tol:
                    self.complete_stage('P3 centered and aligned')
                    return
                lateral_speed = clamp(err_lat * self.p3_align_lat_gain, -self.p3_align_lat_max, self.p3_align_lat_max)
                turn_speed = clamp(err_yaw * self.p3_align_yaw_gain, -self.p3_align_yaw_max, self.p3_align_yaw_max)
                self.p3_send_velocity_command(0.0, lateral_speed, turn_speed, step_height=self.p3_align_step_height)
            else:
                self.p3_send_velocity_command(self.p3_align_search_vx, 0.0, self.p3_align_search_wz,
                                              step_height=self.p3_align_step_height)
            return


def main(args=None):
    rclpy.init(args=args)
    node = Stage3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        try:
            if node.Ctrl is not None:
                node.send_stop_command()
        except Exception:
            pass
        try:
            if node.Ctrl is not None:
                node.Ctrl.quit()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()