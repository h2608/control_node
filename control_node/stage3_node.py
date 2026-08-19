#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三赛段节点：弯道-直道-弯道黄线赛道，不越界跟随。

第三赛段尾部改为纯前向 RGB：
S 弯黄线纠偏 -> 固定预运动 -> 固定右转 -> 保持 0.20m 机身高度并前倾做前方黄线角度调平
-> 原地踏步 0.5s 恢复水平姿态/body_height=0.20 -> 固定左转 90° -> 完成。
鱼眼不参与第三赛段尾部矫正。
"""

import time
import threading
import math
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from control_node.stage_common import (
    StageNodeBase,
    clamp,
    find_contours,
    image_qos,
)
from control_node.stage_entry import EntryPoint, StageEntryTable


def p3_entry_table():
    """第三赛段调试入口表（顺序即流程顺序）。"""
    states = (
        'P3_STAND_WAIT',
        'P3_S_CURVE_CRUISE',
        'P3_PRE_MOTION',
        'P3_TURN_RIGHT_FIXED',
        'P3_ALIGN_FRONT_YELLOW',
        'P3_RESTORE_LEVEL',
        'P3_TURN_LEFT_90',
    )
    return StageEntryTable(3, 'P3_STAND_WAIT', states, (
        EntryPoint('start', 'P3_STAND_WAIT', '低身站立等待，完整第三赛段'),
        EntryPoint('s_curve', 'P3_S_CURVE_CRUISE', '弯-直-弯 RGB 黄线巡航',
                   requires=('赛道黄线必须在前向 RGB 视野内',)),
        EntryPoint('pre_motion', 'P3_PRE_MOTION', 'S 弯后固定预运动：前进+左移'),
        EntryPoint('right_turn', 'P3_TURN_RIGHT_FIXED', '固定时间右转'),
        EntryPoint('align', 'P3_ALIGN_FRONT_YELLOW', '保持 0.20m 机身高度并前倾，用前向 RGB 横线调平',
                   requires=('前方黄色横线必须在 RGB 视野内',)),
        EntryPoint('restore', 'P3_RESTORE_LEVEL', '原地踏步恢复水平姿态和 0.20m 机身高度'),
        EntryPoint('left90', 'P3_TURN_LEFT_90', '固定时间左转 90°'),
    ))


class _Stage3RgbReceiverNode(Node):
    """Dedicated raw RGB receiver isolated from the Stage3 control executor.

    回调只覆盖保存最新 ROS Image，不做 cv_bridge/OpenCV。
    因此即使第三赛段控制/视觉处理耗时，RGB 接收线程仍能持续收图。
    """

    def __init__(self, context, namespace: str, rgb_topic: str,
                 qos=qos_profile_sensor_data, rx_diag=None):
        # 忽略 launch 对主节点的 __node:=stage3_node 全局重映射，
        # 避免辅助节点也被重命名成 stage3_node。
        super().__init__(
            'stage3_rgb_rx',
            context=context,
            namespace=namespace,
            use_global_arguments=False,
        )
        self._lock = threading.Lock()
        self._latest_msg = None
        self._seq = 0
        self._last_rx_monotonic_s = None
        self._rx_diag = rx_diag
        self._sub = self.create_subscription(
            Image,
            rgb_topic,
            self._rgb_cb,
            qos,
        )

    def _rgb_cb(self, msg: Image):
        now = time.monotonic()
        if self._rx_diag is not None:
            self._rx_diag.record('rgb', now)
        with self._lock:
            self._latest_msg = msg
            self._seq += 1
            self._last_rx_monotonic_s = now

    def snapshot(self):
        with self._lock:
            return self._seq, self._latest_msg, self._last_rx_monotonic_s



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
        self.declare_parameter('p3_align_step_height', 0.05)

        # 前段 RGB 黄线巡航纠偏持续时间：这就是你说的‘矫正时间’。
        self.declare_parameter('p3_s_curve_duration_sec', 33.0)
        self.declare_parameter('p3_base_forward_speed', 0.15)
        self.declare_parameter('p3_min_forward_speed', 0.00)
        self.declare_parameter('p3_kp_turn', 1.2)
        self.declare_parameter('p3_kp_lat', 0.2)
        self.declare_parameter('p3_kd_slowdown', 0.10)
        # 5 Hz 控制下允许最多约 3 个控制拍没有新 RGB；超过后立即零速度等待。
        self.declare_parameter('p3_vision_timeout_sec', 0.60)
        # 兼容旧 launch/yaml 参数；第三赛段 RGB stale 时不再使用 fallback 前进。
        self.declare_parameter('p3_fallback_forward_speed', 0.00)

        # S 弯 RGB 纠偏结束后的固定预运动。vx>0 为前进，vy>0 为左移。
        self.declare_parameter('p3_final_forward_speed', 0.15)
        self.declare_parameter('p3_final_left_turn_speed', 0.0)
        self.declare_parameter('p3_final_left_shift_speed', 0.20)
        self.declare_parameter('p3_final_left_shift_duration_sec', 1.5)

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
        self.p3_final_forward_speed = float(
            self.get_parameter('p3_final_forward_speed').value)
        self.p3_final_left_turn_speed = float(
            self.get_parameter('p3_final_left_turn_speed').value)
        self.p3_final_left_shift_speed = float(
            self.get_parameter('p3_final_left_shift_speed').value)
        self.p3_final_left_shift_duration_sec = float(
            self.get_parameter('p3_final_left_shift_duration_sec').value)

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
        # 只缓存最新 RGB 原始消息；真正的 cv_bridge + OpenCV 处理放到 5 Hz 控制拍。
        self._pending_rgb_msg: Optional[Image] = None
        # RGB stale 期间暂停赛段计时，恢复图像后再继续原状态。
        self.p3_rgb_stale_since: Optional[float] = None
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

        # 与第二赛段一致：固定时间状态 30 Hz，RGB 视觉状态 5 Hz。
        self.declare_parameter('stage3_fixed_control_hz', 30.0)
        self.declare_parameter('stage3_vision_control_hz', 5.0)
        self.stage3_fixed_control_hz = max(
            0.1, float(self.get_parameter('stage3_fixed_control_hz').value))
        self.stage3_vision_control_hz = max(
            0.1, float(self.get_parameter('stage3_vision_control_hz').value))
        if abs(self.control_hz - self.stage3_fixed_control_hz) > 1e-6:
            self.control_timer.cancel()
            self.destroy_timer(self.control_timer)
            self.control_timer = self.create_timer(
                1.0 / self.stage3_fixed_control_hz,
                self._control_timer_cb,
            )
        self.control_hz = self.stage3_fixed_control_hz
        self._p3_last_vision_tick_monotonic_s = None

        self.declare_parameter('p3_initial_state', 'default')
        self.p3_initial_state = self.resolve_stage_entry(
            p3_entry_table(),
            str(self.get_parameter('p3_initial_state').value))

        # 原第三赛段 S 弯参数/视觉参数继续保留。
        self.p3_declare_params()
        self.p3_load_params()
        self.p3_init_vision_caches()

        # ============================================================
        # 新尾部流程参数
        # ============================================================
        # 预运动：按用户指定参数。p3_final_left_turn_speed 保留参数名，但这里默认 0。
        # 已由 mixin 声明：
        #   p3_final_forward_speed=0.15
        #   p3_final_left_shift_speed=0.20
        #   p3_final_left_turn_speed=0.0
        #   p3_final_left_shift_duration_sec=1.5

        # 固定右转：用户指定 |wz|=0.60，2.5s；右转实际发送负 wz。
        self.declare_parameter('p3_right_turn_wz', 0.60)
        self.declare_parameter('p3_right_turn_duration_sec', 2.5)
        # 参考第二赛段 timed_turn_step_height。
        self.declare_parameter('p3_timed_turn_step_height', 0.02)

        # 右转后的前方 RGB 横线调平。注意：这个 5.0s 只是该第二个调平状态的超时保护，
        # 不是前面的 P3_S_CURVE_CRUISE 纠偏时间。
        self.declare_parameter('p3_front_align_max_duration_sec', 5.0)
        # 前方黄线角度使用比例/梯度调速：偏差越大转得越快，接近对齐自动减速。
        self.declare_parameter('p3_front_align_wz_k', 0.012)
        self.declare_parameter('p3_front_align_wz_min', 0.02)
        self.declare_parameter('p3_front_align_wz_max', 0.10)
        self.declare_parameter('p3_front_align_deadband_deg', 1.0)
        self.declare_parameter('p3_front_align_stable_frames', 2)
        # 右转后若前向 RGB 还看不到黄线，则先以固定速度后退寻找。
        # 参数保存速度幅值；实际搜索阶段发送 vx=-abs(speed)。
        self.declare_parameter('p3_front_search_back_speed', 0.15)
        # 第一次识别到黄线后不要按固定时间后退，而是继续后退，直到
        # 黄线轮廓下边缘 line_bottom_y / image_height 到达目标位置。
        # 真机后退时地面横线会向图像上方移动，因此从 >0.90 后退到 <=0.90 时停止。
        self.declare_parameter('p3_front_after_detect_target_y_ratio', 0.90)

        # “降低机身 + 前倾”阶段。前倾 0.15 直接参考第二赛段 stage2_body_pitch；
        # 机身高度按你最新要求固定为 0.20m。
        self.declare_parameter('p3_front_align_pitch', 0.15)
        self.declare_parameter('p3_front_align_body_height', 0.20)

        # 第二赛段前方横线筛选参数。
        self.declare_parameter('p3_front_yellow_roi_top_ratio', 0.65)
        self.declare_parameter('p3_front_yellow_roi_left_ratio', 0.40)
        self.declare_parameter('p3_front_yellow_roi_right_ratio', 0.60)
        self.declare_parameter('p3_front_yellow_h_min', 15)
        self.declare_parameter('p3_front_yellow_h_max', 40)
        self.declare_parameter('p3_front_yellow_s_min', 80)
        self.declare_parameter('p3_front_yellow_s_max', 255)
        self.declare_parameter('p3_front_yellow_v_min', 80)
        self.declare_parameter('p3_front_yellow_v_max', 255)
        self.declare_parameter('p3_front_yellow_min_contour_area', 100.0)
        self.declare_parameter('p3_front_yellow_min_width_height_ratio', 2.5)
        self.declare_parameter('p3_front_yellow_center_tolerance_ratio', 0.15)
        self.declare_parameter('p3_front_yellow_min_width_ratio', 0.45)

        # 调平完成后原地踏步 0.5s，恢复不前倾、body_height=0.20。
        self.declare_parameter('p3_restore_level_duration_sec', 0.5)
        self.declare_parameter('p3_restore_body_height', 0.20)

        # 左转 90°：参考第二赛段 timed_turn_wz_90=0.60, duration=3.85s。
        self.declare_parameter('p3_left_turn_90_wz', 0.60)
        self.declare_parameter('p3_left_turn_90_duration_sec', 3.85)

        self.p3_right_turn_wz = abs(float(self.get_parameter('p3_right_turn_wz').value))
        self.p3_right_turn_duration_sec = max(
            0.0, float(self.get_parameter('p3_right_turn_duration_sec').value))
        self.p3_timed_turn_step_height = float(
            self.get_parameter('p3_timed_turn_step_height').value)

        self.p3_front_align_max_duration_sec = max(
            0.0, float(self.get_parameter('p3_front_align_max_duration_sec').value))
        self.p3_front_align_wz_k = abs(float(
            self.get_parameter('p3_front_align_wz_k').value))
        self.p3_front_align_wz_min = abs(float(
            self.get_parameter('p3_front_align_wz_min').value))
        self.p3_front_align_wz_max = abs(float(
            self.get_parameter('p3_front_align_wz_max').value))
        if self.p3_front_align_wz_min > self.p3_front_align_wz_max:
            self.p3_front_align_wz_min = self.p3_front_align_wz_max
        self.p3_front_align_deadband_deg = abs(float(
            self.get_parameter('p3_front_align_deadband_deg').value))
        self.p3_front_align_stable_frames = max(
            1, int(self.get_parameter('p3_front_align_stable_frames').value))
        self.p3_front_search_back_speed = abs(float(
            self.get_parameter('p3_front_search_back_speed').value))
        self.p3_front_after_detect_target_y_ratio = max(
            0.0, min(1.0, float(
                self.get_parameter('p3_front_after_detect_target_y_ratio').value)))
        self.p3_front_align_pitch = float(
            self.get_parameter('p3_front_align_pitch').value)
        self.p3_front_align_body_height = float(
            self.get_parameter('p3_front_align_body_height').value)

        self.p3_front_yellow_roi_top_ratio = float(
            self.get_parameter('p3_front_yellow_roi_top_ratio').value)
        self.p3_front_yellow_roi_left_ratio = float(
            self.get_parameter('p3_front_yellow_roi_left_ratio').value)
        self.p3_front_yellow_roi_right_ratio = float(
            self.get_parameter('p3_front_yellow_roi_right_ratio').value)
        self.p3_front_yellow_h_min = int(self.get_parameter('p3_front_yellow_h_min').value)
        self.p3_front_yellow_h_max = int(self.get_parameter('p3_front_yellow_h_max').value)
        self.p3_front_yellow_s_min = int(self.get_parameter('p3_front_yellow_s_min').value)
        self.p3_front_yellow_s_max = int(self.get_parameter('p3_front_yellow_s_max').value)
        self.p3_front_yellow_v_min = int(self.get_parameter('p3_front_yellow_v_min').value)
        self.p3_front_yellow_v_max = int(self.get_parameter('p3_front_yellow_v_max').value)
        self.p3_front_yellow_min_contour_area = float(
            self.get_parameter('p3_front_yellow_min_contour_area').value)
        self.p3_front_yellow_min_width_height_ratio = float(
            self.get_parameter('p3_front_yellow_min_width_height_ratio').value)
        self.p3_front_yellow_center_tolerance_ratio = float(
            self.get_parameter('p3_front_yellow_center_tolerance_ratio').value)
        self.p3_front_yellow_min_width_ratio = float(
            self.get_parameter('p3_front_yellow_min_width_ratio').value)

        self.p3_restore_level_duration_sec = max(
            0.0, float(self.get_parameter('p3_restore_level_duration_sec').value))
        self.p3_restore_body_height = float(
            self.get_parameter('p3_restore_body_height').value)
        self.p3_left_turn_90_wz = abs(float(
            self.get_parameter('p3_left_turn_90_wz').value))
        self.p3_left_turn_90_duration_sec = max(
            0.0, float(self.get_parameter('p3_left_turn_90_duration_sec').value))

        self.p3_front_yellow_result = {
            'has_line': False,
            'line_bottom_y': None,
            'line_center': None,
            'img_shape': None,
            'angle_deg': None,
            'bbox': None,
            'width_ratio': None,
            'wh_ratio': None,
        }
        self.p3_front_align_stable_count = 0
        # P3_ALIGN_FRONT_YELLOW 子阶段锁存：
        # acquired=False：还在后退搜索；
        # acquired=True 且 extra_back_active=True：已看到黄线，按 line_bottom_y_ratio 后退到目标位置；
        # acquired=True 且 extra_back_active=False：已经停住，进入角度调平。
        self.p3_front_yellow_acquired = False
        self.p3_front_extra_back_active = False
        self.p3_front_yellow_mask = None

        # ============================================================
        # Dedicated RGB receiver。第三赛段不再启动任何鱼眼 receiver。
        # ============================================================
        try:
            if getattr(self, 'rgb_sub', None) is not None:
                self.destroy_subscription(self.rgb_sub)
                self.get_logger().info(
                    '[P3_RGB_RX] destroyed inherited RGB subscription; '
                    'dedicated receiver will be used')
        except Exception as exc:
            self.get_logger().warning(
                f'[P3_RGB_RX] destroy inherited RGB subscription failed: {exc}')
        self.rgb_sub = None

        self._p3_rgb_rx_node = None
        self._p3_rgb_rx_executor = None
        self._p3_rgb_rx_thread = None
        self._p3_rgb_rx_running = False
        self._p3_rgb_rx_local_consumed_seq = -1
        self._p3_rgb_global_seq = 0
        self._p3_rgb_last_rx_monotonic_s = None
        self._p3_rgb_last_restart_monotonic_s = 0.0
        self._p3_rgb_restart_count = 0
        self._p3_rgb_stale_stop_active = False
        self._p3_rgb_rx_ready = True
        self._apply_raw_subscriptions(activated=self.active, reason='stage3_init')

        self.get_logger().info(
            f'[P3_RATE] fixed={self.stage3_fixed_control_hz:.1f}Hz, '
            f'vision={self.stage3_vision_control_hz:.1f}Hz, '
            f'rgb_timeout={self.p3_vision_timeout_sec:.2f}s, '
            'final_align=FRONT_RGB, fisheye=OFF')
        self.get_logger().info(
            '[P3_FLOW] S_CURVE -> PRE_MOTION -> RIGHT_TURN -> RGB_ALIGN -> '
            'RESTORE_LEVEL -> LEFT_90 -> DONE')
        self.get_logger().info(
            f'[P3_PARAMS] s_curve_correction={self.p3_s_curve_duration_sec:.2f}s | '
            f'pre_motion=[{self.p3_final_forward_speed:.3f},'
            f'{self.p3_final_left_shift_speed:.3f},'
            f'{self.p3_final_left_turn_speed:.3f}] '
            f'{self.p3_final_left_shift_duration_sec:.2f}s | '
            f'right_turn_wz=-{self.p3_right_turn_wz:.2f} '
            f'{self.p3_right_turn_duration_sec:.2f}s | '
            f'left90_wz=+{self.p3_left_turn_90_wz:.2f} '
            f'{self.p3_left_turn_90_duration_sec:.2f}s')
        self.get_logger().info('Stage3Node ready.')

    def on_apply_extra_raw_subscriptions(self, want_rgb, want_depth):
        """Run the dedicated Stage 3 RGB receiver under the shared policy."""
        del want_depth
        if not getattr(self, '_p3_rgb_rx_ready', False):
            return
        if want_rgb:
            self._p3_start_rgb_receiver()
        else:
            self._p3_stop_rgb_receiver()

    # ============================================================
    # 状态 / RGB receiver
    # ============================================================
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state
            if new_state.startswith('P3_'):
                self.p3_state_start_time = None
                self.p3_rgb_stale_since = None
                if new_state == 'P3_STAND_WAIT':
                    self.p3_stand_sent = False
                if new_state == 'P3_ALIGN_FRONT_YELLOW':
                    self.p3_front_align_stable_count = 0
                    self.p3_front_yellow_acquired = False
                    self.p3_front_extra_back_active = False
                    self.p3_front_yellow_result = {
                        'has_line': False,
                        'line_bottom_y': None,
                        'line_center': None,
                        'img_shape': None,
                        'angle_deg': None,
                        'bbox': None,
                        'width_ratio': None,
                        'wh_ratio': None,
                    }
                if new_state in ('P3_S_CURVE_CRUISE', 'P3_ALIGN_FRONT_YELLOW'):
                    self._p3_last_vision_tick_monotonic_s = None

    def _p3_rgb_executor_loop(self):
        executor = self._p3_rgb_rx_executor
        while self._p3_rgb_rx_running and rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.10)
            except Exception as exc:
                if self._p3_rgb_rx_running:
                    self.get_logger().error(
                        f'[P3_RGB_RX] receiver executor exception: {repr(exc)}')
                break

    def _p3_start_rgb_receiver(self):
        if self._p3_rgb_rx_node is not None:
            return
        namespace = self.get_namespace()
        self._p3_rgb_rx_node = _Stage3RgbReceiverNode(
            context=self.context,
            namespace=namespace,
            rgb_topic=self.rgb_topic,
            qos=image_qos(self.image_qos_depth),
            rx_diag=self.rx_diag,
        )
        self._p3_rgb_rx_executor = SingleThreadedExecutor(context=self.context)
        self._p3_rgb_rx_executor.add_node(self._p3_rgb_rx_node)
        self._p3_rgb_rx_running = True
        self._p3_rgb_rx_local_consumed_seq = -1
        self._p3_rgb_rx_thread = threading.Thread(
            target=self._p3_rgb_executor_loop,
            name='stage3_rgb_rx_executor',
            daemon=True,
        )
        self._p3_rgb_rx_thread.start()
        self._p3_rgb_last_restart_monotonic_s = time.monotonic()
        self.get_logger().warning(
            f'[P3_RGB_RX] dedicated receiver started: '
            f'node={namespace}/stage3_rgb_rx topic={self.rgb_topic}')

    def _p3_stop_rgb_receiver(self):
        self._p3_rgb_rx_running = False
        thread = self._p3_rgb_rx_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.25)
        executor = self._p3_rgb_rx_executor
        node = self._p3_rgb_rx_node
        if executor is not None and node is not None:
            try:
                executor.remove_node(node)
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.10)
            except Exception:
                pass
        self._p3_rgb_rx_node = None
        self._p3_rgb_rx_executor = None
        self._p3_rgb_rx_thread = None

    def _p3_restart_rgb_receiver(self, reason: str):
        self._p3_rgb_restart_count += 1
        self.get_logger().error(
            f'[P3_RGB_RX] restarting dedicated receiver: '
            f'count={self._p3_rgb_restart_count}, reason={reason}')
        self._p3_stop_rgb_receiver()
        self._p3_rgb_last_rx_monotonic_s = None
        self._p3_start_rgb_receiver()

    def _p3_sync_rgb_from_receiver(self):
        node = self._p3_rgb_rx_node
        if node is None:
            return False
        local_seq, msg, rx_mono = node.snapshot()
        if rx_mono is not None:
            self._p3_rgb_last_rx_monotonic_s = rx_mono
        if msg is None or local_seq == self._p3_rgb_rx_local_consumed_seq:
            return False
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(
                f'[P3_RGB_RX] cv_bridge convert failed: {exc}',
                throttle_duration_sec=1.0)
            return False
        self._p3_rgb_rx_local_consumed_seq = local_seq
        self._p3_rgb_global_seq += 1
        self.latest_rgb_seq = self._p3_rgb_global_seq
        self.latest_rgb_msg = msg
        self.latest_bgr = frame
        self.on_rgb_frame(frame)
        return True

    def p3_rgb_age_s(self):
        last = self._p3_rgb_last_rx_monotonic_s
        if last is None:
            node = self._p3_rgb_rx_node
            if node is not None:
                _seq, _msg, last = node.snapshot()
                if last is not None:
                    self._p3_rgb_last_rx_monotonic_s = last
        if last is None:
            return None
        return max(0.0, time.monotonic() - float(last))

    # ============================================================
    # 前方 RGB 黄色横线检测：筛选/角度符号参考第二赛段
    # ============================================================
    def p3_is_front_horizontal_yellow_line(self, cnt, roi_shape) -> bool:
        _, roi_w = roi_shape[:2]
        area = cv2.contourArea(cnt)
        if area < self.p3_front_yellow_min_contour_area:
            return False
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh <= 0:
            return False
        wh_ratio = bw / float(bh)
        if wh_ratio < self.p3_front_yellow_min_width_height_ratio:
            return False
        width_ratio = bw / float(max(roi_w, 1))
        if width_ratio < self.p3_front_yellow_min_width_ratio:
            return False
        cx = x + bw / 2.0
        roi_cx = roi_w / 2.0
        center_offset_ratio = abs(cx - roi_cx) / float(max(roi_w, 1))
        if center_offset_ratio > self.p3_front_yellow_center_tolerance_ratio:
            return False
        return True

    @staticmethod
    def p3_signed_yellow_line_angle_deg(cnt) -> float:
        if cnt is None or len(cnt) < 2:
            return 0.0
        vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx)
        vy = float(vy)
        angle = math.degrees(math.atan2(vy, vx))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return float(angle)

    def p3_process_front_yellow_line(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        roi_top = max(0, min(h - 1, int(h * self.p3_front_yellow_roi_top_ratio)))
        roi_left = max(0, min(w - 1, int(w * self.p3_front_yellow_roi_left_ratio)))
        roi_right = max(roi_left + 1, min(w, int(w * self.p3_front_yellow_roi_right_ratio)))
        roi = frame[roi_top:h, roi_left:roi_right]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([
            self.p3_front_yellow_h_min,
            self.p3_front_yellow_s_min,
            self.p3_front_yellow_v_min,
        ], dtype=np.uint8)
        upper = np.array([
            self.p3_front_yellow_h_max,
            self.p3_front_yellow_s_max,
            self.p3_front_yellow_v_max,
        ], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.p3_front_yellow_mask = mask

        contours = find_contours(mask)
        best = None
        best_score = -1.0
        for cnt in contours:
            if not self.p3_is_front_horizontal_yellow_line(cnt, roi.shape):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            score = y + bh
            if score > best_score:
                best_score = score
                best = cnt

        if best is None:
            self.p3_front_yellow_result = {
                'has_line': False,
                'line_bottom_y': None,
                'line_center': None,
                'img_shape': (h, w),
                'angle_deg': None,
                'bbox': None,
                'width_ratio': None,
                'wh_ratio': None,
            }
            return

        x, y, bw, bh = cv2.boundingRect(best)
        angle_deg = self.p3_signed_yellow_line_angle_deg(best)
        line_bottom_y = roi_top + y + bh
        cx = roi_left + x + bw // 2
        cy = roi_top + y + bh // 2
        self.p3_front_yellow_result = {
            'has_line': True,
            'line_bottom_y': int(line_bottom_y),
            'line_center': (int(cx), int(cy)),
            'img_shape': (h, w),
            'angle_deg': float(angle_deg),
            'bbox': (int(roi_left + x), int(roi_top + y),
                     int(roi_left + x + bw), int(roi_top + y + bh)),
            'width_ratio': float(bw / float(max(roi_right - roi_left, 1))),
            'wh_ratio': float(bw / float(max(bh, 1))),
        }

    # ============================================================
    # 生命周期 / RGB 回调
    # ============================================================
    def on_activated(self):
        self.p3_state_start_time = None
        self.p3_rgb_stale_since = None
        self._p3_rgb_stale_stop_active = False
        self.p3_front_align_stable_count = 0
        self.p3_front_yellow_acquired = False
        self.p3_front_extra_back_active = False
        self.state = self.p3_initial_state
        self.p3_send_stand_command()
        self.p3_stand_sent = True
        self.get_logger().info(f'[P3] activated, initial_state={self.state}')

    def on_rgb_frame(self, frame: np.ndarray):
        if self.state == 'P3_S_CURVE_CRUISE':
            self.p3_process_yellow_track(frame)
            if self.show_debug_vis:
                self.p3_show_debug_window(frame)
            return
        if self.state == 'P3_ALIGN_FRONT_YELLOW':
            self.p3_process_front_yellow_line(frame)
            if self.show_debug_vis:
                self.p3_show_front_align_debug(frame)
            return

    def p3_show_front_align_debug(self, frame: np.ndarray):
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            top = int(h * self.p3_front_yellow_roi_top_ratio)
            left = int(w * self.p3_front_yellow_roi_left_ratio)
            right = int(w * self.p3_front_yellow_roi_right_ratio)
            cv2.rectangle(vis, (left, top), (right, h - 1), (0, 255, 255), 2)
            result = self.p3_front_yellow_result
            bbox = result.get('bbox')
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            center = result.get('line_center')
            angle = result.get('angle_deg')
            if center is not None:
                cv2.circle(vis, center, 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f'P3 RGB ALIGN angle={angle} stable={self.p3_front_align_stable_count}/'
                f'{self.p3_front_align_stable_frames}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow('stage3_front_rgb_align', vis)
            if self.show_yellow_mask and self.p3_front_yellow_mask is not None:
                cv2.imshow('stage3_front_rgb_yellow_mask', self.p3_front_yellow_mask)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().warning(
                f'p3 front align debug failed: {exc}', throttle_duration_sec=1.0)

    # ============================================================
    # Servo 命令 / 计时
    # ============================================================
    def p3_elapsed_in_state(self) -> float:
        now = self.now_sec()
        if self.p3_state_start_time is None:
            self.p3_state_start_time = now
        self.p3_state_start_time = self.align_motion_timer_start(
            self.p3_state_start_time, now)
        return max(0.0, now - self.p3_state_start_time)

    def p3_send_motion_command(
            self,
            vx: float = 0.0,
            vy: float = 0.0,
            wz: float = 0.0,
            *,
            step_height: Optional[float] = None,
            pitch: Optional[float] = None,
            body_height: Optional[float] = None):
        if getattr(self, 'Ctrl', None) is None:
            self.get_logger().warning(
                '[P3 CMD] Robot_Ctrl is not active; motion command ignored',
                throttle_duration_sec=1.0)
            return
        h = self.p3_step_height if step_height is None else float(step_height)
        command_pitch = self.p3_stand_pitch if pitch is None else float(pitch)
        command_body_height = (
            self.p3_stand_body_height if body_height is None else float(body_height))
        vx = float(vx)
        vy = float(vy)
        wz = float(wz)
        self.motion_cmd = (vx, vy, wz)
        self.Ctrl.move(
            vx, vy, wz,
            step_height=h,
            roll=0.0,
            pitch=command_pitch,
            yaw=0.0,
            body_height=command_body_height,
            legacy_gait_id=3,
        )
        self.get_logger().info(
            f'[P3 CMD] servo vel=[{vx:.3f}, {vy:.3f}, {wz:.3f}], '
            f'pitch={command_pitch:.3f}, body_height={command_body_height:.3f}, '
            f'step_height={h:.3f}',
            throttle_duration_sec=0.3)

    def p3_send_stand_command(self):
        self.p3_send_motion_command(
            0.0, 0.0, 0.0,
            step_height=self.p3_step_height,
            pitch=self.p3_stand_pitch,
            body_height=self.p3_stand_body_height)
        self.get_logger().info(
            '[P3 CMD] SERVO HOLD / LOW BODY', throttle_duration_sec=1.0)

    def p3_send_velocity_command(
            self, vx: float, vy: float, wz: float,
            step_height: Optional[float] = None):
        self.p3_send_motion_command(
            vx, vy, wz,
            step_height=step_height,
            pitch=self.p3_stand_pitch,
            body_height=self.p3_stand_body_height)

    # ============================================================
    # 30 Hz fixed / 5 Hz RGB visual dispatch
    # ============================================================
    def stage_control_loop(self):
        visual_states = {'P3_S_CURVE_CRUISE', 'P3_ALIGN_FRONT_YELLOW'}
        if self.state not in visual_states:
            self.p3_control_loop()
            return

        now_mono = time.monotonic()
        period = 1.0 / self.stage3_vision_control_hz
        last = self._p3_last_vision_tick_monotonic_s
        if last is not None and (now_mono - last) < period:
            return
        self._p3_last_vision_tick_monotonic_s = now_mono

        rgb_new = self._p3_sync_rgb_from_receiver()
        now = self.now_sec()
        rgb_age = self.p3_rgb_age_s()
        if rgb_age is None or rgb_age >= self.p3_vision_timeout_sec:
            if self.p3_rgb_stale_since is None:
                self.p3_rgb_stale_since = now
            self._p3_rgb_stale_stop_active = True
            age_text = 'none' if rgb_age is None else f'{rgb_age:.3f}s'
            thread_alive = bool(
                self._p3_rgb_rx_thread is not None and self._p3_rgb_rx_thread.is_alive())
            self.get_logger().warning(
                f'[P3_RGB_STALE_STOP] state={self.state}, age={age_text}, '
                f'timeout={self.p3_vision_timeout_sec:.3f}s, '
                f'rgb_seq={self._p3_rgb_global_seq}, '
                f'rx_thread_alive={thread_alive} -> zero velocity',
                throttle_duration_sec=0.5)
            if self.state == 'P3_ALIGN_FRONT_YELLOW':
                self.p3_send_motion_command(
                    0.0, 0.0, 0.0,
                    step_height=self.p3_timed_turn_step_height,
                    pitch=self.p3_front_align_pitch,
                    body_height=self.p3_front_align_body_height)
            else:
                self.p3_send_velocity_command(
                    0.0, 0.0, 0.0, step_height=self.p3_step_height)
            if rgb_age is not None:
                restart_after = max(2.0, 2.0 * self.p3_vision_timeout_sec)
                since_restart = now_mono - self._p3_rgb_last_restart_monotonic_s
                if rgb_age >= restart_after and since_restart >= 3.0:
                    self._p3_restart_rgb_receiver(
                        f'age={rgb_age:.3f}s state={self.state}')
            return

        if self._p3_rgb_stale_stop_active:
            self.get_logger().warning(
                f'[P3_RGB_RECOVER] RGB recovered: state={self.state}, '
                f'rgb_seq={self._p3_rgb_global_seq}, age={rgb_age:.3f}s, new={rgb_new}')
            self._p3_rgb_stale_stop_active = False
        if self.p3_rgb_stale_since is not None:
            stale_duration = max(0.0, now - self.p3_rgb_stale_since)
            if self.p3_state_start_time is not None:
                self.p3_state_start_time += stale_duration
            self.get_logger().info(
                f'[P3_RGB_RECOVER] state={self.state}, paused={stale_duration:.3f}s')
            self.p3_rgb_stale_since = None

        self.p3_control_loop()

    # ============================================================
    # 新第三赛段状态机
    # ============================================================
    def p3_control_loop(self):
        elapsed = self.p3_elapsed_in_state()

        if self.state == 'P3_STAND_WAIT':
            if not self.p3_stand_sent:
                self.p3_send_stand_command()
                self.p3_stand_sent = True
            if elapsed >= self.p3_stand_wait_sec:
                self.set_state('P3_S_CURVE_CRUISE')
            return

        if self.state == 'P3_S_CURVE_CRUISE':
            if elapsed >= self.p3_s_curve_duration_sec:
                self.get_logger().info(
                    '[P3] S-curve correction complete -> fixed pre-motion')
                self.set_state('P3_PRE_MOTION')
                return

            raw_turn = (self.p3_error_mid / 500.0 + self.p3_error_near) * self.p3_kp_turn
            turn_speed = clamp(raw_turn, -0.5, 0.5)
            lateral_speed = clamp(self.p3_error_near * self.p3_kp_lat, -0.10, 0.10)
            speed_drop = abs(self.p3_error_near) * self.p3_kd_slowdown
            forward_speed = max(
                self.p3_min_forward_speed,
                self.p3_base_forward_speed - speed_drop)
            self.p3_send_velocity_command(
                forward_speed, lateral_speed, turn_speed,
                step_height=self.p3_step_height)
            self.get_logger().info(
                f'P3_S_CURVE_CRUISE elapsed={elapsed:.2f}/'
                f'{self.p3_s_curve_duration_sec:.2f}s | '
                f'err_mid={self.p3_error_mid:.3f}, err_near={self.p3_error_near:.3f}, '
                f'cmd=[{forward_speed:.3f},{lateral_speed:.3f},{turn_speed:.3f}]',
                throttle_duration_sec=0.5)
            return

        if self.state == 'P3_PRE_MOTION':
            if elapsed >= self.p3_final_left_shift_duration_sec:
                self.get_logger().info(
                    '[P3] pre-motion complete -> fixed right turn')
                self.set_state('P3_TURN_RIGHT_FIXED')
                return
            self.p3_send_motion_command(
                self.p3_final_forward_speed,
                self.p3_final_left_shift_speed,
                self.p3_final_left_turn_speed,
                step_height=self.p3_step_height,
                pitch=self.p3_stand_pitch,
                body_height=self.p3_stand_body_height)
            self.get_logger().info(
                f'P3_PRE_MOTION elapsed={elapsed:.2f}/'
                f'{self.p3_final_left_shift_duration_sec:.2f}s | '
                f'cmd=[{self.p3_final_forward_speed:.3f},'
                f'{self.p3_final_left_shift_speed:.3f},'
                f'{self.p3_final_left_turn_speed:.3f}]',
                throttle_duration_sec=0.2)
            return

        if self.state == 'P3_TURN_RIGHT_FIXED':
            if elapsed >= self.p3_right_turn_duration_sec:
                self.p3_send_motion_command(
                    0.0, 0.0, 0.0,
                    step_height=self.p3_timed_turn_step_height,
                    pitch=0.0,
                    body_height=self.p3_stand_body_height)
                self.get_logger().info(
                    '[P3] fixed right turn complete -> forward-lean body + front RGB align')
                self.set_state('P3_ALIGN_FRONT_YELLOW')
                return
            self.p3_send_motion_command(
                0.0, 0.0, -self.p3_right_turn_wz,
                step_height=self.p3_timed_turn_step_height,
                pitch=0.0,
                body_height=self.p3_stand_body_height)
            self.get_logger().info(
                f'P3_TURN_RIGHT_FIXED elapsed={elapsed:.2f}/'
                f'{self.p3_right_turn_duration_sec:.2f}s | '
                f'cmd=[0.000,0.000,{-self.p3_right_turn_wz:.3f}]',
                throttle_duration_sec=0.2)
            return

        if self.state == 'P3_ALIGN_FRONT_YELLOW':
            # 搜索黄线阶段不使用 5s 调平超时；只有首次识别到黄线并停住后，
            # 才重新计时并对真正的角度调平应用 p3_front_align_max_duration_sec。
            if (self.p3_front_yellow_acquired
                    and not self.p3_front_extra_back_active
                    and elapsed >= self.p3_front_align_max_duration_sec):
                self.get_logger().warning(
                    '[P3] front RGB yellow alignment timeout -> restore level body')
                self.set_state('P3_RESTORE_LEVEL')
                return

            result = self.p3_front_yellow_result
            has_front_line = (
                result.get('has_line', False)
                and result.get('angle_deg') is not None
            )

            # 子阶段 1：固定右转后，如果还没有找到前方黄线，就只向后退寻找。
            # 一旦首次识别到黄线，立即停住一拍并锁存，下一控制拍才开始角度调平。
            if not self.p3_front_yellow_acquired:
                if not has_front_line:
                    self.p3_front_align_stable_count = 0
                    search_vx = -abs(self.p3_front_search_back_speed)
                    self.p3_send_motion_command(
                        search_vx, 0.0, 0.0,
                        step_height=self.p3_timed_turn_step_height,
                        pitch=self.p3_front_align_pitch,
                        body_height=self.p3_front_align_body_height)
                    self.get_logger().warning(
                        f'[P3_RGB_ALIGN_SEARCH] front yellow not detected -> BACKWARD, '
                        f'vx={search_vx:.3f}',
                        throttle_duration_sec=0.3)
                    return

                self.p3_front_yellow_acquired = True
                self.p3_front_align_stable_count = 0
                self.p3_front_extra_back_active = True
                # 首次识别后先按黄线下边缘位置决定是否还需要继续后退。
                img_shape = result.get('img_shape')
                line_bottom_y = result.get('line_bottom_y')
                line_bottom_ratio = None
                if img_shape is not None and line_bottom_y is not None and img_shape[0] > 0:
                    line_bottom_ratio = float(line_bottom_y) / float(img_shape[0])

                if (line_bottom_ratio is not None
                        and line_bottom_ratio <= self.p3_front_after_detect_target_y_ratio):
                    # 已经达到/超过目标位置（黄线已经足够进入画面），直接停住。
                    self.p3_front_extra_back_active = False
                    self.p3_state_start_time = None
                    self.p3_send_motion_command(
                        0.0, 0.0, 0.0,
                        step_height=self.p3_timed_turn_step_height,
                        pitch=self.p3_front_align_pitch,
                        body_height=self.p3_front_align_body_height)
                    self.get_logger().info(
                        f'[P3_RGB_ALIGN_SEARCH] front yellow acquired, '
                        f'bottom_y_ratio={line_bottom_ratio:.3f} <= target='
                        f'{self.p3_front_after_detect_target_y_ratio:.3f} -> STOP; '
                        f'next tick start angle alignment')
                    return

                search_vx = -abs(self.p3_front_search_back_speed)
                self.p3_send_motion_command(
                    search_vx, 0.0, 0.0,
                    step_height=self.p3_timed_turn_step_height,
                    pitch=self.p3_front_align_pitch,
                    body_height=self.p3_front_align_body_height)
                ratio_text = 'None' if line_bottom_ratio is None else f'{line_bottom_ratio:.3f}'
                self.get_logger().info(
                    f'[P3_RGB_ALIGN_SEARCH] front yellow acquired, angle='
                    f'{float(result["angle_deg"]):.2f}deg, bottom_y_ratio={ratio_text} '
                    f'-> BACKWARD until <= '
                    f'{self.p3_front_after_detect_target_y_ratio:.3f}, '
                    f'vx={search_vx:.3f}')
                return

            # 子阶段 2：已看到黄线后，按黄线下边缘的纵向比例后退到 0.90。
            # 在尚未确认 bottom_y_ratio <= 目标值之前，无论暂时丢线还是位置无效，
            # 都继续后退，避免因为单帧识别失败停在图像底边附近。
            if self.p3_front_extra_back_active:
                if not has_front_line:
                    search_vx = -abs(self.p3_front_search_back_speed)
                    self.p3_send_motion_command(
                        search_vx, 0.0, 0.0,
                        step_height=self.p3_timed_turn_step_height,
                        pitch=self.p3_front_align_pitch,
                        body_height=self.p3_front_align_body_height)
                    self.get_logger().warning(
                        f'[P3_RGB_ALIGN_POSITION_BACK] yellow temporarily lost, target not reached '
                        f'-> CONTINUE BACKWARD vx={search_vx:.3f}',
                        throttle_duration_sec=0.5)
                    return

                img_shape = result.get('img_shape')
                line_bottom_y = result.get('line_bottom_y')
                if img_shape is None or line_bottom_y is None or img_shape[0] <= 0:
                    search_vx = -abs(self.p3_front_search_back_speed)
                    self.p3_send_motion_command(
                        search_vx, 0.0, 0.0,
                        step_height=self.p3_timed_turn_step_height,
                        pitch=self.p3_front_align_pitch,
                        body_height=self.p3_front_align_body_height)
                    self.get_logger().warning(
                        f'[P3_RGB_ALIGN_POSITION_BACK] invalid yellow y position, target not reached '
                        f'-> CONTINUE BACKWARD vx={search_vx:.3f}',
                        throttle_duration_sec=0.5)
                    return

                line_bottom_ratio = float(line_bottom_y) / float(img_shape[0])
                target_ratio = self.p3_front_after_detect_target_y_ratio

                if line_bottom_ratio > target_ratio:
                    search_vx = -abs(self.p3_front_search_back_speed)
                    self.p3_send_motion_command(
                        search_vx, 0.0, 0.0,
                        step_height=self.p3_timed_turn_step_height,
                        pitch=self.p3_front_align_pitch,
                        body_height=self.p3_front_align_body_height)
                    self.get_logger().info(
                        f'[P3_RGB_ALIGN_POSITION_BACK] bottom_y_ratio='
                        f'{line_bottom_ratio:.3f} > target={target_ratio:.3f} '
                        f'-> BACKWARD vx={search_vx:.3f}',
                        throttle_duration_sec=0.2)
                    return

                self.p3_front_extra_back_active = False
                self.p3_front_align_stable_count = 0
                # 到达目标 y 位置后停住，并从这一刻重新计算真正角度调平的 7s 超时。
                self.p3_state_start_time = None
                self.p3_send_motion_command(
                    0.0, 0.0, 0.0,
                    step_height=self.p3_timed_turn_step_height,
                    pitch=self.p3_front_align_pitch,
                    body_height=self.p3_front_align_body_height)
                self.get_logger().info(
                    f'[P3_RGB_ALIGN_POSITION_BACK] bottom_y_ratio='
                    f'{line_bottom_ratio:.3f} <= target={target_ratio:.3f} '
                    f'-> STOP; next tick start angle alignment')
                return

            # 子阶段 3：已经找到过黄线且额外后退完成后，只做原地角度调平。
            # 后续偶发单帧丢线时保持原地，不重新进入后退搜索，避免前后抖动。
            if not has_front_line:
                self.p3_front_align_stable_count = 0
                self.p3_send_motion_command(
                    0.0, 0.0, 0.0,
                    step_height=self.p3_timed_turn_step_height,
                    pitch=self.p3_front_align_pitch,
                    body_height=self.p3_front_align_body_height)
                self.get_logger().warning(
                    '[P3_RGB_ALIGN] yellow temporarily lost after acquisition -> HOLD',
                    throttle_duration_sec=0.5)
                return

            angle_deg = float(result['angle_deg'])

            # 先判断是否已经进入对齐死区。
            if abs(angle_deg) <= self.p3_front_align_deadband_deg:
                self.p3_front_align_stable_count += 1
                self.p3_send_motion_command(
                    0.0, 0.0, 0.0,
                    step_height=self.p3_timed_turn_step_height,
                    pitch=self.p3_front_align_pitch,
                    body_height=self.p3_front_align_body_height)
                self.get_logger().info(
                    f'[P3_RGB_ALIGN] READY angle={angle_deg:.2f}deg '
                    f'deadband={self.p3_front_align_deadband_deg:.2f}deg '
                    f'stable={self.p3_front_align_stable_count}/'
                    f'{self.p3_front_align_stable_frames}',
                    throttle_duration_sec=0.2)
                if self.p3_front_align_stable_count >= self.p3_front_align_stable_frames:
                    self.get_logger().info(
                        '[P3] front RGB yellow alignment complete -> restore level body 0.5s')
                    self.set_state('P3_RESTORE_LEVEL')
                return

            self.p3_front_align_stable_count = 0
            # 梯度/比例调速：角度越大转得越快，接近 deadband 时自动减速。
            # 方向保持第二赛段约定：angle>0 -> 负 wz（右转）；angle<0 -> 正 wz（左转）。
            abs_angle = abs(angle_deg)
            wz_mag = abs_angle * self.p3_front_align_wz_k
            wz_mag = max(self.p3_front_align_wz_min,
                         min(self.p3_front_align_wz_max, wz_mag))
            wz = -wz_mag if angle_deg > 0.0 else wz_mag
            self.p3_send_motion_command(
                0.0, 0.0, wz,
                step_height=self.p3_timed_turn_step_height,
                pitch=self.p3_front_align_pitch,
                body_height=self.p3_front_align_body_height)
            self.get_logger().info(
                f'[P3_RGB_ALIGN] angle={angle_deg:.2f}deg -> '
                f'{"RIGHT" if wz < 0.0 else "LEFT"}, '
                f'gradient_wz={wz:.3f} '
                f'(k={self.p3_front_align_wz_k:.3f}, '
                f'min={self.p3_front_align_wz_min:.3f}, '
                f'max={self.p3_front_align_wz_max:.3f}), '
                f'pitch={self.p3_front_align_pitch:.3f}, '
                f'body_height={self.p3_front_align_body_height:.3f}',
                throttle_duration_sec=0.2)
            return

        if self.state == 'P3_RESTORE_LEVEL':
            if elapsed >= self.p3_restore_level_duration_sec:
                self.get_logger().info(
                    '[P3] level-body recovery complete -> fixed left turn 90deg')
                self.set_state('P3_TURN_LEFT_90')
                return
            self.p3_send_motion_command(
                0.0, 0.0, 0.0,
                step_height=self.p3_timed_turn_step_height,
                pitch=0.0,
                body_height=self.p3_restore_body_height)
            self.get_logger().info(
                f'P3_RESTORE_LEVEL elapsed={elapsed:.2f}/'
                f'{self.p3_restore_level_duration_sec:.2f}s | '
                f'pitch=0.000 body_height={self.p3_restore_body_height:.3f}',
                throttle_duration_sec=0.2)
            return

        if self.state == 'P3_TURN_LEFT_90':
            if elapsed >= self.p3_left_turn_90_duration_sec:
                self.p3_send_motion_command(
                    0.0, 0.0, 0.0,
                    step_height=self.p3_timed_turn_step_height,
                    pitch=0.0,
                    body_height=self.p3_restore_body_height)
                self.get_logger().info(
                    '[P3] fixed left 90deg complete -> stage complete')
                self.complete_stage(
                    'P3 RGB align + restore + fixed left 90 complete')
                return
            self.p3_send_motion_command(
                0.0, 0.0, self.p3_left_turn_90_wz,
                step_height=self.p3_timed_turn_step_height,
                pitch=0.0,
                body_height=self.p3_restore_body_height)
            self.get_logger().info(
                f'P3_TURN_LEFT_90 elapsed={elapsed:.2f}/'
                f'{self.p3_left_turn_90_duration_sec:.2f}s | '
                f'cmd=[0.000,0.000,{self.p3_left_turn_90_wz:.3f}]',
                throttle_duration_sec=0.2)
            return

        self.get_logger().warning(
            f'[P3] unknown state={self.state}; stop and finish stage')
        self.p3_send_motion_command(
            0.0, 0.0, 0.0,
            step_height=self.p3_timed_turn_step_height,
            pitch=0.0,
            body_height=self.p3_restore_body_height)
        self.complete_stage('P3 unknown state fallback')


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
            node._p3_stop_rgb_receiver()
        except Exception:
            pass
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
