#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一赛段节点：石板路段 + 第一个弯道。

原 control_node_123456.py 的 P1_* 状态机：
前倾黄线纠偏巡航 -> 前倾刹车缓冲 -> 前倾停止线调平 -> 恢复正常姿态
-> 短暂前进 -> 左转 -> 找蓝球前进 -> 盲走左移。
盲走左移结束后向任务控制节点上报完成（原来是直接切入第二赛段）。
"""

from typing import Optional

import cv2
import numpy as np

import rclpy

from control_node.stage_common import StageNodeBase, clamp
from control_node.stage_entry import EntryPoint, StageEntryTable


def p1_entry_table():
    """第一赛段调试入口表（顺序即流程顺序）。"""
    states = (
        'P1_STAND_WAIT',
        'P1_STAGE1_CRUISE',
        'P1_BRAKE_BUFFER',
        'P1_ALIGN_STOP_LINE',
        'P1_RESTORE_BODY',
        'P1_POST_ALIGN_FORWARD',
        'P1_TURN_LEFT_TO_STAGE2',
        'P1_APPROACH_BLUE_BALL',
        'P1_BLIND_LEFT_SHIFT',
    )
    return StageEntryTable(1, 'P1_STAND_WAIT', states, (
        EntryPoint('start', 'P1_STAND_WAIT', '起步站立等待，完整第一赛段'),
        EntryPoint('cruise', 'P1_STAGE1_CRUISE', '石板路黄线纠偏巡航'),
        EntryPoint('brake', 'P1_BRAKE_BUFFER', '看到停止线后的刹车缓冲'),
        EntryPoint('align', 'P1_ALIGN_STOP_LINE', '对停止线调平机身',
                   requires=('停止线必须在前向 RGB 视野内',)),
        EntryPoint('restore', 'P1_RESTORE_BODY', '从前倾恢复正常姿态'),
        EntryPoint('forward', 'P1_POST_ALIGN_FORWARD', '调平后的定时前进'),
        EntryPoint('turn', 'P1_TURN_LEFT_TO_STAGE2', '左转朝向第二赛段'),
        EntryPoint('ball', 'P1_APPROACH_BLUE_BALL', '朝蓝球前进',
                   requires=('蓝球必须在前向 RGB 视野内',)),
        EntryPoint('shift', 'P1_BLIND_LEFT_SHIFT', '收尾盲走左横移'),
    ))


class Stage1Node(StageNodeBase):

    STAGE_ID = 1

    def __init__(self):
        super().__init__('stage1_node', self.STAGE_ID)

        # =========================
        # 第一赛段参数（全部加 p1_ 前缀，避免和第二赛段变量冲突）
        # =========================
        # 调试入口：也可以直接写状态名。统一的 entry_point 参数优先。
        self.declare_parameter('p1_initial_state', 'default')

        self.declare_parameter('p1_stand_wait_sec', 0)
        self.declare_parameter('p1_stand_body_height', 0.28)
        self.declare_parameter('p1_forward_pitch', 0.10)
        self.declare_parameter('p1_restore_body_duration_sec', 0.3)

        self.declare_parameter('p1_stage1_max_duration_sec', 8.0)
        self.declare_parameter('p1_base_forward_speed', 0.40)
        self.declare_parameter('p1_min_forward_speed', 0.20)
        self.declare_parameter('p1_kp_turn', 0.25)
        self.declare_parameter('p1_kp_lat', 0.15)
        self.declare_parameter('p1_kd_slowdown', 0.05)
        self.declare_parameter('p1_max_turn_speed', 0.15)
        self.declare_parameter('p1_max_lateral_speed', 0.15)
        self.declare_parameter('p1_vision_timeout_sec', 1.0)

        self.declare_parameter('p1_brake_duration_sec', 0.3)
        self.declare_parameter('p1_align_max_duration_sec', 3.0)
        self.declare_parameter('p1_align_angle_deadband_rad', 0.05)
        self.declare_parameter('p1_align_turn_kp', 0.4)
        self.declare_parameter('p1_align_turn_max_wz', 0.10)

        self.declare_parameter('p1_post_align_forward_duration_sec', 1.7)
        self.declare_parameter('p1_post_align_forward_speed', 0.40)

        self.declare_parameter('p1_turn_duration_sec', 3.5)
        self.declare_parameter('p1_turn_forward_vel', 0.13)
        self.declare_parameter('p1_turn_yaw_vel', 0.51)

        self.declare_parameter('p1_blue_target_distance_m', 0.25)
        self.declare_parameter('p1_approach_blue_max_duration_sec', 3.0)
        self.declare_parameter('p1_approach_blue_forward_speed', 0.20)

        self.declare_parameter('p1_blind_left_duration_sec', 3.0)
        self.declare_parameter('p1_blind_left_vy', 0.14)
        self.declare_parameter('p1_blind_left_vx', 0.12)

        self.declare_parameter('p1_yellow_h_min', 20)
        self.declare_parameter('p1_yellow_h_max', 40)
        self.declare_parameter('p1_yellow_s_min', 50)
        self.declare_parameter('p1_yellow_s_max', 255)
        self.declare_parameter('p1_yellow_v_min', 150)
        self.declare_parameter('p1_yellow_v_max', 255)

        self.declare_parameter('p1_stop_top_ratio', 0.80)
        self.declare_parameter('p1_stop_bottom_ratio', 0.95)
        self.declare_parameter('p1_stop_left_ratio', 0.35)
        self.declare_parameter('p1_stop_right_ratio', 0.65)
        self.declare_parameter('p1_stop_yellow_pixel_threshold', 5000)

        self.declare_parameter('p1_nav_top_ratio', 0.90)
        self.declare_parameter('p1_nav_bottom_ratio', 1.00)
        self.declare_parameter('p1_nav_crop_left_ratio', 0.15)
        self.declare_parameter('p1_nav_crop_right_ratio', 0.85)

        self.declare_parameter('p1_blue_h_min', 100)
        self.declare_parameter('p1_blue_h_max', 130)
        self.declare_parameter('p1_blue_s_min', 100)
        self.declare_parameter('p1_blue_s_max', 255)
        self.declare_parameter('p1_blue_v_min', 50)
        self.declare_parameter('p1_blue_v_max', 255)
        self.declare_parameter('p1_blue_min_area', 6500.0)
        self.declare_parameter('p1_blue_depth_patch_half', 1)
        # P1 的蓝球测距沿用原整合节点的深度有效范围。拆分节点时这两个
        # 参数只被带到了 Stage2Node，导致 P1 首次处理蓝球深度时访问未初始化属性。
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        # =========================
        # 读取第一赛段参数（p1_ 前缀）
        # =========================
        self.p1_initial_state = self.resolve_stage_entry(
            p1_entry_table(),
            str(self.get_parameter('p1_initial_state').value))

        self.p1_stand_wait_sec = float(self.get_parameter('p1_stand_wait_sec').value)
        self.p1_stand_body_height = float(self.get_parameter('p1_stand_body_height').value)
        self.p1_forward_pitch = float(self.get_parameter('p1_forward_pitch').value)
        self.p1_restore_body_duration_sec = float(
            self.get_parameter('p1_restore_body_duration_sec').value)

        self.p1_stage1_max_duration_sec = float(self.get_parameter('p1_stage1_max_duration_sec').value)
        self.p1_base_forward_speed = float(self.get_parameter('p1_base_forward_speed').value)
        self.p1_min_forward_speed = float(self.get_parameter('p1_min_forward_speed').value)
        self.p1_kp_turn = float(self.get_parameter('p1_kp_turn').value)
        self.p1_kp_lat = float(self.get_parameter('p1_kp_lat').value)
        self.p1_kd_slowdown = float(self.get_parameter('p1_kd_slowdown').value)
        self.p1_max_turn_speed = float(self.get_parameter('p1_max_turn_speed').value)
        self.p1_max_lateral_speed = float(self.get_parameter('p1_max_lateral_speed').value)
        self.p1_vision_timeout_sec = float(self.get_parameter('p1_vision_timeout_sec').value)

        self.p1_brake_duration_sec = float(self.get_parameter('p1_brake_duration_sec').value)
        self.p1_align_max_duration_sec = float(self.get_parameter('p1_align_max_duration_sec').value)
        self.p1_align_angle_deadband_rad = float(self.get_parameter('p1_align_angle_deadband_rad').value)
        self.p1_align_turn_kp = float(self.get_parameter('p1_align_turn_kp').value)
        self.p1_align_turn_max_wz = float(self.get_parameter('p1_align_turn_max_wz').value)

        self.p1_post_align_forward_duration_sec = float(
            self.get_parameter('p1_post_align_forward_duration_sec').value)
        self.p1_post_align_forward_speed = float(
            self.get_parameter('p1_post_align_forward_speed').value)

        self.p1_turn_duration_sec = float(self.get_parameter('p1_turn_duration_sec').value)
        self.p1_turn_forward_vel = float(self.get_parameter('p1_turn_forward_vel').value)
        self.p1_turn_yaw_vel = float(self.get_parameter('p1_turn_yaw_vel').value)

        self.p1_blue_target_distance_m = float(self.get_parameter('p1_blue_target_distance_m').value)
        self.p1_approach_blue_max_duration_sec = float(self.get_parameter('p1_approach_blue_max_duration_sec').value)
        self.p1_approach_blue_forward_speed = float(self.get_parameter('p1_approach_blue_forward_speed').value)

        self.p1_blind_left_duration_sec = float(self.get_parameter('p1_blind_left_duration_sec').value)
        self.p1_blind_left_vy = float(self.get_parameter('p1_blind_left_vy').value)
        self.p1_blind_left_vx = float(self.get_parameter('p1_blind_left_vx').value)

        self.p1_yellow_h_min = int(self.get_parameter('p1_yellow_h_min').value)
        self.p1_yellow_h_max = int(self.get_parameter('p1_yellow_h_max').value)
        self.p1_yellow_s_min = int(self.get_parameter('p1_yellow_s_min').value)
        self.p1_yellow_s_max = int(self.get_parameter('p1_yellow_s_max').value)
        self.p1_yellow_v_min = int(self.get_parameter('p1_yellow_v_min').value)
        self.p1_yellow_v_max = int(self.get_parameter('p1_yellow_v_max').value)

        self.p1_stop_top_ratio = float(self.get_parameter('p1_stop_top_ratio').value)
        self.p1_stop_bottom_ratio = float(self.get_parameter('p1_stop_bottom_ratio').value)
        self.p1_stop_left_ratio = float(self.get_parameter('p1_stop_left_ratio').value)
        self.p1_stop_right_ratio = float(self.get_parameter('p1_stop_right_ratio').value)
        self.p1_stop_yellow_pixel_threshold = int(self.get_parameter('p1_stop_yellow_pixel_threshold').value)

        self.p1_nav_top_ratio = float(self.get_parameter('p1_nav_top_ratio').value)
        self.p1_nav_bottom_ratio = float(self.get_parameter('p1_nav_bottom_ratio').value)
        self.p1_nav_crop_left_ratio = float(self.get_parameter('p1_nav_crop_left_ratio').value)
        self.p1_nav_crop_right_ratio = float(self.get_parameter('p1_nav_crop_right_ratio').value)

        self.p1_blue_h_min = int(self.get_parameter('p1_blue_h_min').value)
        self.p1_blue_h_max = int(self.get_parameter('p1_blue_h_max').value)
        self.p1_blue_s_min = int(self.get_parameter('p1_blue_s_min').value)
        self.p1_blue_s_max = int(self.get_parameter('p1_blue_s_max').value)
        self.p1_blue_v_min = int(self.get_parameter('p1_blue_v_min').value)
        self.p1_blue_v_max = int(self.get_parameter('p1_blue_v_max').value)
        self.p1_blue_min_area = float(self.get_parameter('p1_blue_min_area').value)
        self.p1_blue_depth_patch_half = int(self.get_parameter('p1_blue_depth_patch_half').value)
        self.valid_min_depth_m = float(self.get_parameter('valid_min_depth_m').value)
        self.valid_max_depth_m = float(self.get_parameter('valid_max_depth_m').value)

        # =========================
        # 第一赛段运行缓存（全部 p1_ 前缀，避免覆盖第二赛段 latest_* / yellow_* / ball_*）
        # =========================
        self.p1_state_start_time: Optional[float] = None
        self.p1_stand_sent = False
        self.p1_lateral_force = 0.0
        self.p1_stop_angle = 0.0
        self.p1_stop_flag = 0.0
        self.p1_last_update_time = 0.0
        self.p1_blue_distance_m = 0.0
        self.p1_blue_count = 0.0
        self.p1_blue_detections = []
        self.p1_latest_mask_yellow = None

        self.get_logger().info('Stage1Node ready.')

    # ============================================================
    # 状态切换（原 set_state 的 P1 部分）
    # ============================================================
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state
            if new_state.startswith('P1_'):
                self.p1_state_start_time = None

    def on_activated(self):
        # 该函数位于 ROS 单线程订阅回调中，不能调用 Wait_finish()；否则会连带
        # 阻塞 RGB、深度和控制定时器。立即发送站立命令，再由状态机继续推进。
        self.p1_state_start_time = None
        self.state = self.p1_initial_state
        # 站立命令对任何入口都是安全的起手：从中段入口启动时它只是把机身摆正，
        # 下一个控制周期该入口的动作就接管。
        self.p1_send_stand_command()
        self.p1_stand_sent = True
        self.get_logger().info(
            '[P1] boot stand command sent without blocking ROS callbacks, '
            f'initial_state={self.state}')

    def on_rgb_frame(self, frame: np.ndarray):
        self.p1_process_stage1_yellow(frame)
        self.p1_process_blue_ball(frame)
        if self.show_debug_vis:
            self.show_p1_debug_window(frame)

    def show_p1_debug_window(self, frame: np.ndarray):
        """在第一赛段 RGB 回调内显示已有识别结果，不重复运行视觉算法。"""
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]

            stop_top = int(h * self.p1_stop_top_ratio)
            stop_bottom = int(h * self.p1_stop_bottom_ratio)
            stop_left = int(w * self.p1_stop_left_ratio)
            stop_right = int(w * self.p1_stop_right_ratio)
            nav_top = int(h * self.p1_nav_top_ratio)
            nav_bottom = int(h * self.p1_nav_bottom_ratio)
            nav_left = int(w * self.p1_nav_crop_left_ratio)
            nav_right = int(w * self.p1_nav_crop_right_ratio)

            stop_pixels = 0
            if self.p1_latest_mask_yellow is not None:
                stop_pixels = cv2.countNonZero(
                    self.p1_latest_mask_yellow[
                        stop_top:stop_bottom, stop_left:stop_right])

            stop_color = (0, 0, 255) if self.p1_stop_flag > 0.5 else (0, 165, 255)
            cv2.rectangle(
                vis, (stop_left, stop_top), (stop_right, stop_bottom), stop_color, 2)
            cv2.rectangle(
                vis, (nav_left, nav_top), (nav_right, nav_bottom), (0, 255, 0), 2)
            cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1)

            for detection in self.p1_blue_detections:
                x, y, box_w, box_h = detection['bbox']
                cx, cy = detection['center']
                cv2.rectangle(
                    vis, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)
                cv2.circle(vis, (cx, cy), 4, (255, 0, 255), -1)
                cv2.putText(
                    vis, f"{detection['depth_m']:.2f}m",
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 2)

            cv2.putText(vis, f'P1 state={self.state}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(
                vis,
                f'stop={self.p1_stop_flag:.0f} pixels={stop_pixels}/'
                f'{self.p1_stop_yellow_pixel_threshold} '
                f'angle={np.degrees(self.p1_stop_angle):.1f}deg',
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(
                vis,
                f'lat_force={self.p1_lateral_force:.3f} blue_cnt={self.p1_blue_count:.0f} '
                f'blue_dist={self.p1_blue_distance_m:.2f}',
                (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('stage1_debug', vis)
            if self.show_yellow_mask and self.p1_latest_mask_yellow is not None:
                mask_vis = self.p1_latest_mask_yellow.copy()
                cv2.rectangle(
                    mask_vis, (stop_left, stop_top), (stop_right, stop_bottom), 180, 2)
                cv2.rectangle(
                    mask_vis, (nav_left, nav_top), (nav_right, nav_bottom), 128, 2)
                cv2.imshow('stage1_yellow_mask', mask_vis)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'show_p1_debug_window failed: {e}', throttle_duration_sec=1.0)

    def stage_control_loop(self):
        self.p1_control_loop()

    # ============================================================
    # 第一赛段工具 / 视觉 / 控制状态机
    # ============================================================
    def p1_elapsed_in_state(self) -> float:
        now = self.now_sec()
        if self.p1_state_start_time is None:
            self.p1_state_start_time = now
        self.p1_state_start_time = self.align_motion_timer_start(
            self.p1_state_start_time, now)
        return max(0.0, now - self.p1_state_start_time)

    def p1_send_stand_command(self):
        self.msg.mode = 12
        self.msg.gait_id = 0
        self._inc_life_count()
        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.msg.step_height = [0.0, 0.0]
        self.msg.rpy_des = [0.0, self.p1_forward_pitch, 0.0]
        self.msg.pos_des = [0.0, 0.0, self.p1_stand_body_height]
        self.Ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[P1 CMD] STAND / FORWARD LEAN pitch={self.p1_forward_pitch:.3f}',
            throttle_duration_sec=1.0)

    def p1_send_velocity_command(
            self, vx: float, vy: float, wz: float, step_height: float,
            pitch: float):
        """第一赛段专用速度命令，可在运动和原地调平时保持指定俯仰姿态。"""
        self.Ctrl.move(
            float(vx), float(vy), float(wz),
            step_height=float(step_height),
            pitch=float(pitch),
            body_height=float(self.p1_stand_body_height),
            legacy_gait_id=3,
        )
        self.get_logger().info(
            f'[P1 CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}], pitch={pitch:.3f}',
            throttle_duration_sec=0.3)

    def p1_depth_to_meters_patch(self, patch: np.ndarray):
        if patch is None or patch.size == 0:
            return None

        if self.latest_depth_encoding == '16UC1':
            patch_m = patch.astype(np.float32) / 1000.0
        elif self.latest_depth_encoding == '32FC1':
            patch_m = patch.astype(np.float32)
        else:
            patch_m = patch.astype(np.float32)

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid > self.valid_min_depth_m) & (valid < self.valid_max_depth_m)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def p1_process_stage1_yellow(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([self.p1_yellow_h_min, self.p1_yellow_s_min, self.p1_yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.p1_yellow_h_max, self.p1_yellow_s_max, self.p1_yellow_v_max], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self.p1_latest_mask_yellow = mask_yellow

        lateral_force = 0.0
        stop_angle = 0.0
        stop_flag = 0.0

        stop_top = int(h * self.p1_stop_top_ratio)
        stop_bottom = int(h * self.p1_stop_bottom_ratio)
        stop_left = int(w * self.p1_stop_left_ratio)
        stop_right = int(w * self.p1_stop_right_ratio)
        mask_stop = mask_yellow[stop_top:stop_bottom, stop_left:stop_right]

        if cv2.countNonZero(mask_stop) > self.p1_stop_yellow_pixel_threshold:
            stop_flag = 1.0
            contours, _ = cv2.findContours(mask_stop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.int32)
                box_sorted = sorted(box, key=lambda p: p[0])
                left_pt, right_pt = box_sorted[0], box_sorted[-1]
                dx = right_pt[0] - left_pt[0]
                dy = right_pt[1] - left_pt[1]
                stop_angle = float(np.arctan2(dy, dx)) if dx != 0 else 0.0

        nav_top = int(h * self.p1_nav_top_ratio)
        nav_bottom = int(h * self.p1_nav_bottom_ratio)
        crop_left = int(w * self.p1_nav_crop_left_ratio)
        crop_right = int(w * self.p1_nav_crop_right_ratio)
        mask_nav = np.zeros_like(mask_yellow)
        mask_nav[nav_top:nav_bottom, crop_left:crop_right] = mask_yellow[nav_top:nav_bottom, crop_left:crop_right]

        M_nav = cv2.moments(mask_nav)
        if M_nav['m00'] > 0:
            cx_nav = int(M_nav['m10'] / M_nav['m00'])
            dist_nav = abs(cx_nav - w / 2)
            force_nav = ((w / 2 - dist_nav) / (w / 2)) ** 3
            lateral_force = float(force_nav) if cx_nav > w / 2 else -float(force_nav)

        self.p1_lateral_force = lateral_force
        self.p1_stop_angle = stop_angle
        self.p1_stop_flag = stop_flag
        self.p1_last_update_time = self.now_sec()

    def p1_process_blue_ball(self, frame: np.ndarray):
        self.p1_blue_detections = []
        self.p1_blue_distance_m = 0.0
        self.p1_blue_count = 0.0

        if self.latest_depth is None:
            return

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([self.p1_blue_h_min, self.p1_blue_s_min, self.p1_blue_v_min], dtype=np.uint8)
        upper_blue = np.array([self.p1_blue_h_max, self.p1_blue_s_max, self.p1_blue_v_max], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_depths = []
        dh, dw = self.latest_depth.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= self.p1_blue_min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] <= 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dx = int(cx * dw / max(w, 1))
            dy = int(cy * dh / max(h, 1))
            half = self.p1_blue_depth_patch_half
            x1 = max(0, dx - half)
            x2 = min(dw, dx + half + 1)
            y1 = max(0, dy - half)
            y2 = min(dh, dy + half + 1)
            depth_m = self.p1_depth_to_meters_patch(self.latest_depth[y1:y2, x1:x2])
            if depth_m is None:
                continue
            valid_depths.append(depth_m)
            self.p1_blue_detections.append({
                'center': (cx, cy),
                'depth_m': depth_m,
                'area': float(area),
                'bbox': cv2.boundingRect(cnt),
            })

        if valid_depths:
            self.p1_blue_count = float(len(valid_depths))
            self.p1_blue_distance_m = float(min(valid_depths))

    def p1_control_loop(self):
        now = self.now_sec()
        if now <= 0.0:
            return

        elapsed = self.p1_elapsed_in_state()

        if self.state == 'P1_STAND_WAIT':
            if not self.p1_stand_sent:
                self.get_logger().info('[P1] 起立')
                self.p1_send_stand_command()
                self.p1_stand_sent = True
            if elapsed >= self.p1_stand_wait_sec:
                self.get_logger().info('[P1] 开始第一赛段黄线纠偏巡航')
                self.set_state('P1_STAGE1_CRUISE')
            return

        if self.state == 'P1_STAGE1_CRUISE':
            if self.p1_stop_flag > 0.5:
                self.get_logger().info('[P1] 看到横向黄线，进入刹车缓冲')
                self.set_state('P1_BRAKE_BUFFER')
                return

            if elapsed >= self.p1_stage1_max_duration_sec:
                self.get_logger().info('[P1] 第一赛段行走超时，进入刹车缓冲')
                self.set_state('P1_BRAKE_BUFFER')
                return

            if now - self.p1_last_update_time < self.p1_vision_timeout_sec:
                err = self.p1_lateral_force
                turn_speed = clamp(err * self.p1_kp_turn, -self.p1_max_turn_speed, self.p1_max_turn_speed)
                lateral_speed = clamp(err * self.p1_kp_lat, -self.p1_max_lateral_speed, self.p1_max_lateral_speed)
                speed_drop = abs(err) * self.p1_kd_slowdown
                forward_speed = max(self.p1_min_forward_speed, self.p1_base_forward_speed - speed_drop)
            else:
                forward_speed = self.p1_base_forward_speed
                lateral_speed = 0.0
                turn_speed = 0.0

            self.p1_send_velocity_command(
                forward_speed, lateral_speed, turn_speed,
                step_height=0.13, pitch=self.p1_forward_pitch)
            return

        if self.state == 'P1_BRAKE_BUFFER':
            if elapsed >= self.p1_brake_duration_sec:
                self.get_logger().info('[P1] 开始根据横线角度调平')
                self.set_state('P1_ALIGN_STOP_LINE')
                return
            self.p1_send_velocity_command(
                0.0, 0.0, 0.0,
                step_height=0.10, pitch=self.p1_forward_pitch)
            return

        if self.state == 'P1_ALIGN_STOP_LINE':
            angle_err = self.p1_stop_angle
            if abs(angle_err) < self.p1_align_angle_deadband_rad or self.p1_stop_flag < 0.5:
                self.get_logger().info(f'[P1] 调平完成或横线离开视野，angle_err={angle_err:.3f}')
                self.set_state('P1_RESTORE_BODY')
                return

            if elapsed >= self.p1_align_max_duration_sec:
                self.get_logger().info('[P1] 调平超时，开始恢复正常姿态')
                self.set_state('P1_RESTORE_BODY')
                return

            turn_speed = clamp(angle_err * self.p1_align_turn_kp, -self.p1_align_turn_max_wz, self.p1_align_turn_max_wz)
            self.p1_send_velocity_command(
                0.0, 0.0, turn_speed,
                step_height=0.10, pitch=self.p1_forward_pitch)
            return

        if self.state == 'P1_RESTORE_BODY':
            if elapsed >= self.p1_restore_body_duration_sec:
                self.get_logger().info('[P1] 正常姿态恢复完成，进入调平后短暂前进')
                self.set_state('P1_POST_ALIGN_FORWARD')
                return
            self.p1_send_velocity_command(
                0.0, 0.0, 0.0,
                step_height=0.10, pitch=0.0)
            return

        if self.state == 'P1_POST_ALIGN_FORWARD':
            if elapsed >= self.p1_post_align_forward_duration_sec:
                self.get_logger().info('[P1] 调平后前进结束，进入左转')
                self.set_state('P1_TURN_LEFT_TO_STAGE2')
                return
            self.p1_send_velocity_command(
                self.p1_post_align_forward_speed, 0.0, 0.0,
                step_height=0.15, pitch=0.0)
            return

        if self.state == 'P1_TURN_LEFT_TO_STAGE2':
            if elapsed >= self.p1_turn_duration_sec:
                self.get_logger().info('[P1] 左转结束，开始寻找蓝球并前进')
                self.set_state('P1_APPROACH_BLUE_BALL')
                return
            self.send_velocity_command(self.p1_turn_forward_vel, 0.0, self.p1_turn_yaw_vel, step_height=0.10)
            return

        if self.state == 'P1_APPROACH_BLUE_BALL':
            if self.p1_blue_count >= 1.0:
                self.get_logger().info(
                    f'[P1] 锁定蓝球距离: {self.p1_blue_distance_m:.2f}m',
                    throttle_duration_sec=0.5
                )
                if self.p1_blue_distance_m <= self.p1_blue_target_distance_m:
                    self.get_logger().info('[P1] 到达蓝球目标距离，进入盲走左移')
                    self.set_state('P1_BLIND_LEFT_SHIFT')
                    return

            if elapsed >= self.p1_approach_blue_max_duration_sec:
                self.get_logger().info('[P1] 找蓝球前进超时，进入盲走左移')
                self.set_state('P1_BLIND_LEFT_SHIFT')
                return

            self.send_velocity_command(self.p1_approach_blue_forward_speed, 0.0, 0.0, step_height=0.10)
            return

        if self.state == 'P1_BLIND_LEFT_SHIFT':
            if elapsed >= self.p1_blind_left_duration_sec:
                # 关键：这里仍然不发 STOP，但进入第二赛段前重置第二赛段缓存，
                # 让第二赛段表现更接近“单独启动第二赛段”。
                self.get_logger().info('[P1] 第一赛段结束，向任务控制节点上报完成')
                self.complete_stage('P1_BLIND_LEFT_SHIFT finished')
                return

            self.send_velocity_command(self.p1_blind_left_vx, self.p1_blind_left_vy, 0.0, step_height=0.10)
            return

        # 兜底：如果 P1 状态写错，直接切入第二赛段，避免卡死。
        self.get_logger().warn(f'[P1] unknown state={self.state}, finish stage 1')
        self.complete_stage('P1 unknown state fallback')


def main(args=None):
    rclpy.init(args=args)
    node = Stage1Node()
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
