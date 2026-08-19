#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一赛段节点：石板路段 + 第一个弯道。

原 control_node_123456.py 的 P1_* 状态机：
前倾黄线纠偏巡航 -> 前倾刹车缓冲 -> 前倾停止线调平 -> 恢复正常姿态
-> 短暂前进 -> 左转 -> 找蓝球前进 -> 盲走左移。
盲走左移结束后向任务控制节点上报完成（原来是直接切入第二赛段）。
"""

import time
import threading
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


def p1_entry_table():
    """第一赛段调试入口表（顺序即流程顺序）。"""
    states = (
        'P1_STAND_WAIT',
        'P1_STAGE1_CRUISE',
        'P1_BRAKE_BUFFER',
        'P1_ALIGN_STOP_LINE',
        'P1_RESTORE_BODY',
        'P1_POST_ALIGN_FORWARD',
        'P1_PRE_TURN_LOWER_BODY',
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
        EntryPoint('pre_turn', 'P1_PRE_TURN_LOWER_BODY', '转向前原地踏步并降低机身'),
        EntryPoint('turn', 'P1_TURN_LEFT_TO_STAGE2', '左转朝向第二赛段'),
        EntryPoint('ball', 'P1_APPROACH_BLUE_BALL', '朝蓝球前进',
                   requires=('蓝球必须在前向 RGB 视野内',)),
        EntryPoint('shift', 'P1_BLIND_LEFT_SHIFT', '收尾盲走左横移'),
    ))


def p1_cruise_velocity(
        lateral_force, vision_age, *, timeout_s, hold_s, decay_s,
        blind_min_speed, base_speed, min_speed, kp_turn, kp_lat,
        kd_slowdown, max_turn, max_lat):
    """Return the safe Stage 1 cruise command for fresh or stale vision."""
    err = float(lateral_force)
    wz = clamp(err * kp_turn, -max_turn, max_turn)
    vy = clamp(err * kp_lat, -max_lat, max_lat)
    vx = max(min_speed, base_speed - abs(err) * kd_slowdown)

    if vision_age < timeout_s:
        return vx, vy, wz, 'fresh'

    blind_age = vision_age - timeout_s
    if blind_age < hold_s:
        return vx, vy, wz, 'hold'

    ramp_age = blind_age - hold_s
    frac = 1.0 if decay_s <= 0.0 else clamp(ramp_age / decay_s, 0.0, 1.0)
    vx = vx + (blind_min_speed - vx) * frac
    return vx, 0.0, 0.0, 'decay'


class _Stage1RgbReceiverNode(Node):
    """Dedicated raw RGB receiver isolated from the Stage1 control executor.

    回调只覆盖保存最新 ROS Image，不做 cv_bridge/OpenCV。
    因此即使第一赛段控制/视觉处理耗时，RGB 接收线程仍能持续收图。
    """

    def __init__(self, context, namespace: str, rgb_topic: str,
                 qos=qos_profile_sensor_data, rx_diag=None):
        # 忽略 launch 对主节点的 __node:=stage1_node 全局重映射，
        # 避免辅助节点也被重命名成 stage1_node。
        super().__init__(
            'stage1_rgb_rx',
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


class Stage1Node(StageNodeBase):

    STAGE_ID = 1

    def __init__(self):
        super().__init__('stage1_node', self.STAGE_ID)

        # =========================
        # 第一赛段双频率控制
        # =========================
        # 视觉状态：5 Hz 控制 + 每拍只处理最新一帧 RGB。
        # 固定时间/非视觉状态：30 Hz 控制，并完全跳过 cv_bridge/OpenCV。
        # 底层 timer 以 30 Hz 运行，stage_control_loop() 内对视觉状态节流到 5 Hz。
        self.declare_parameter('stage1_fixed_control_hz', 30.0)
        self.declare_parameter('stage1_vision_control_hz', 5.0)
        self.stage1_fixed_control_hz = max(
            0.1, float(self.get_parameter('stage1_fixed_control_hz').value))
        self.stage1_vision_control_hz = max(
            0.1, float(self.get_parameter('stage1_vision_control_hz').value))
        if abs(self.control_hz - self.stage1_fixed_control_hz) > 1e-6:
            self.control_timer.cancel()
            self.destroy_timer(self.control_timer)
            self.control_timer = self.create_timer(
                1.0 / self.stage1_fixed_control_hz,
                self._control_timer_cb
            )
        self.control_hz = self.stage1_fixed_control_hz
        self._p1_last_vision_tick_monotonic_s = None

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
        self.declare_parameter('p1_vision_timeout_sec', 0.60)

        self.declare_parameter('p1_brake_duration_sec', 0.3)
        self.declare_parameter('p1_align_max_duration_sec', 3.0)
        self.declare_parameter('p1_align_angle_deadband_rad', 0.05)
        self.declare_parameter('p1_align_turn_kp', 0.4)
        self.declare_parameter('p1_align_turn_max_wz', 0.10)

        self.declare_parameter('p1_post_align_forward_duration_sec', 1.2)
        self.declare_parameter('p1_post_align_forward_speed', 0.15)

        # 转向前先原地踏步 1.0s，同时把机身高度从正常 0.28m 降到 0.20m。
        # 随后的整个定时转向阶段也保持 0.20m。
        self.declare_parameter('p1_pre_turn_lower_duration_sec', 1.0)
        self.declare_parameter('p1_turn_body_height', 0.20)
        self.declare_parameter('p1_turn_duration_sec', 3.85)
        self.declare_parameter('p1_turn_forward_vel', 0.0)
        self.declare_parameter('p1_turn_yaw_vel', 0.60)

        # 蓝球靠近阶段改为纯 RGB 判定：不使用深度，按检测轮廓的最小外接圆半径触发。
        self.declare_parameter('p1_blue_trigger_radius_px', 120)
        self.declare_parameter('p1_approach_blue_max_duration_sec', 6.0)
        self.declare_parameter('p1_approach_blue_forward_speed', 0.20)
        # 蓝球 RGB 居中：圆心偏离画面中心超过死区时，使用固定横移速度修正。
        self.declare_parameter('p1_blue_center_deadband_px', 30.0)
        self.declare_parameter('p1_blue_center_vy', 0.06)

        self.declare_parameter('p1_blind_left_duration_sec', 5.0)
        self.declare_parameter('p1_blind_left_vy', 0.11)
        self.declare_parameter('p1_blind_left_vx', 0.11)

        self.declare_parameter('p1_yellow_h_min', 20)
        self.declare_parameter('p1_yellow_h_max', 40)
        self.declare_parameter('p1_yellow_s_min', 50)
        self.declare_parameter('p1_yellow_s_max', 255)
        self.declare_parameter('p1_yellow_v_min', 150)
        self.declare_parameter('p1_yellow_v_max', 255)

        # 与第二赛段类似，按停止黄线下沿在图像中的纵向比例做两级控制：
        # 0.80：进入减速区；0.90：触发停车。
        self.declare_parameter('p1_yellow_slowdown_ratio', 0.85)
        self.declare_parameter('p1_yellow_slow_speed', 0.10)
        self.declare_parameter('p1_stop_line_y_ratio', 0.95)
        self.declare_parameter('p1_stop_min_contour_area', 100.0)
        # 黄线达到停止阈值后，后续所有连续步态统一使用较低抬腿高度。
        self.declare_parameter('p1_post_stop_step_height', 0.02)
        # 停止线候选仍只看图像中央区域，避免两侧黄色物体误触发。
        self.declare_parameter('p1_stop_top_ratio', 0.65)
        self.declare_parameter('p1_stop_bottom_ratio', 1.00)
        self.declare_parameter('p1_stop_left_ratio', 0.35)
        self.declare_parameter('p1_stop_right_ratio', 0.65)
        # 旧参数保留兼容 launch/yaml，但停止判定不再使用黄色像素总数。
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

        self.p1_pre_turn_lower_duration_sec = float(
            self.get_parameter('p1_pre_turn_lower_duration_sec').value)
        self.p1_turn_body_height = float(
            self.get_parameter('p1_turn_body_height').value)
        self.p1_turn_duration_sec = float(self.get_parameter('p1_turn_duration_sec').value)
        self.p1_turn_forward_vel = float(self.get_parameter('p1_turn_forward_vel').value)
        self.p1_turn_yaw_vel = float(self.get_parameter('p1_turn_yaw_vel').value)

        self.p1_blue_trigger_radius_px = float(
            self.get_parameter('p1_blue_trigger_radius_px').value)
        self.p1_approach_blue_max_duration_sec = float(
            self.get_parameter('p1_approach_blue_max_duration_sec').value)
        self.p1_approach_blue_forward_speed = float(self.get_parameter('p1_approach_blue_forward_speed').value)
        self.p1_blue_center_deadband_px = float(
            self.get_parameter('p1_blue_center_deadband_px').value)
        self.p1_blue_center_vy = abs(float(
            self.get_parameter('p1_blue_center_vy').value))

        self.p1_blind_left_duration_sec = float(self.get_parameter('p1_blind_left_duration_sec').value)
        self.p1_blind_left_vy = float(self.get_parameter('p1_blind_left_vy').value)
        self.p1_blind_left_vx = float(self.get_parameter('p1_blind_left_vx').value)

        self.p1_yellow_h_min = int(self.get_parameter('p1_yellow_h_min').value)
        self.p1_yellow_h_max = int(self.get_parameter('p1_yellow_h_max').value)
        self.p1_yellow_s_min = int(self.get_parameter('p1_yellow_s_min').value)
        self.p1_yellow_s_max = int(self.get_parameter('p1_yellow_s_max').value)
        self.p1_yellow_v_min = int(self.get_parameter('p1_yellow_v_min').value)
        self.p1_yellow_v_max = int(self.get_parameter('p1_yellow_v_max').value)

        self.p1_yellow_slowdown_ratio = float(
            self.get_parameter('p1_yellow_slowdown_ratio').value)
        self.p1_yellow_slow_speed = float(
            self.get_parameter('p1_yellow_slow_speed').value)
        self.p1_stop_line_y_ratio = float(self.get_parameter('p1_stop_line_y_ratio').value)
        self.p1_stop_min_contour_area = float(self.get_parameter('p1_stop_min_contour_area').value)
        self.p1_post_stop_step_height = float(
            self.get_parameter('p1_post_stop_step_height').value)
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
        # stop_flag 只表示“黄线下沿已达到停止阈值”；visible 与其分离，
        # 这样进入调平后即使黄线下沿略微退回阈值上方，也不会误判成黄线消失。
        self.p1_stop_flag = 0.0
        self.p1_yellow_slowdown_flag = 0.0
        self.p1_stop_line_visible = False
        self.p1_stop_line_bottom_y: Optional[int] = None
        self.p1_stop_line_bottom_ratio = 0.0
        self.p1_last_update_time = 0.0
        self.p1_blue_max_radius_px = 0.0
        self.p1_blue_count = 0.0
        self.p1_blue_center_x_px: Optional[float] = None
        self.p1_blue_center_error_px = 0.0
        self.p1_blue_detections = []
        self.p1_latest_mask_yellow = None
        # RGB stale 期间暂停依赖视觉状态的赛段计时，恢复图像后再继续原状态。
        self.p1_rgb_stale_since: Optional[float] = None

        # ============================================================
        # Dedicated RGB receiver（与第三赛段同一架构）
        # ============================================================
        # StageNodeBase 已在主 Stage1Node 上创建 RGB 订阅。这里销毁它，
        # 改由轻量辅助节点 + 独立 executor/thread 接收，避免主状态机阻塞图像回调。
        try:
            if getattr(self, 'rgb_sub', None) is not None:
                self.destroy_subscription(self.rgb_sub)
                self.get_logger().info(
                    '[P1_RGB_RX] destroyed inherited RGB subscription; '
                    'dedicated receiver will be used'
                )
        except Exception as exc:
            self.get_logger().warning(
                f'[P1_RGB_RX] destroy inherited RGB subscription failed: {exc}'
            )
        self.rgb_sub = None

        self._p1_rgb_rx_node = None
        self._p1_rgb_rx_executor = None
        self._p1_rgb_rx_thread = None
        self._p1_rgb_rx_running = False
        self._p1_rgb_rx_local_consumed_seq = -1
        self._p1_rgb_global_seq = 0
        self._p1_rgb_last_rx_monotonic_s = None
        self._p1_rgb_last_restart_monotonic_s = 0.0
        self._p1_rgb_restart_count = 0
        self._p1_rgb_stale_stop_active = False
        self._p1_rgb_rx_ready = True
        self._apply_raw_subscriptions(activated=self.active, reason='stage1_init')

        self.get_logger().info(
            f'[P1_RATE] fixed_control={self.stage1_fixed_control_hz:.1f} Hz, '
            f'vision_control={self.stage1_vision_control_hz:.1f} Hz, '
            f'rgb_timeout={self.p1_vision_timeout_sec:.2f}s, '
            'fixed_states_rgb_processing=OFF'
        )
        self.get_logger().info('Stage1Node ready.')

    def on_apply_extra_raw_subscriptions(self, want_rgb, want_depth):
        """Run the dedicated Stage 1 RGB receiver under the shared policy."""
        del want_depth
        if not getattr(self, '_p1_rgb_rx_ready', False):
            return
        if want_rgb:
            self._p1_start_rgb_receiver()
        else:
            self._p1_stop_rgb_receiver()

    # ============================================================
    # 状态切换（原 set_state 的 P1 部分）
    # ============================================================
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state
            if new_state.startswith('P1_'):
                self.p1_state_start_time = None
                self.p1_rgb_stale_since = None
                # 进入视觉状态时让下一次 30 Hz master tick 立即执行一次 5 Hz 视觉拍。
                if new_state in (
                    'P1_STAGE1_CRUISE',
                    'P1_ALIGN_STOP_LINE',
                    'P1_APPROACH_BLUE_BALL',
                ):
                    self._p1_last_vision_tick_monotonic_s = None

    def _p1_rgb_executor_loop(self):
        executor = self._p1_rgb_rx_executor
        while self._p1_rgb_rx_running and rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.10)
            except Exception as exc:
                if self._p1_rgb_rx_running:
                    self.get_logger().error(
                        f'[P1_RGB_RX] receiver executor exception: {repr(exc)}'
                    )
                break

    def _p1_start_rgb_receiver(self):
        if self._p1_rgb_rx_node is not None:
            return
        namespace = self.get_namespace()
        self._p1_rgb_rx_node = _Stage1RgbReceiverNode(
            context=self.context,
            namespace=namespace,
            rgb_topic=self.rgb_topic,
            qos=image_qos(self.image_qos_depth),
            rx_diag=self.rx_diag,
        )
        self._p1_rgb_rx_executor = SingleThreadedExecutor(context=self.context)
        self._p1_rgb_rx_executor.add_node(self._p1_rgb_rx_node)
        self._p1_rgb_rx_running = True
        self._p1_rgb_rx_local_consumed_seq = -1
        self._p1_rgb_rx_thread = threading.Thread(
            target=self._p1_rgb_executor_loop,
            name='stage1_rgb_rx_executor',
            daemon=True,
        )
        self._p1_rgb_rx_thread.start()
        self._p1_rgb_last_restart_monotonic_s = time.monotonic()
        self.get_logger().warning(
            f'[P1_RGB_RX] dedicated receiver started: '
            f'node={namespace}/stage1_rgb_rx topic={self.rgb_topic}'
        )

    def _p1_stop_rgb_receiver(self):
        self._p1_rgb_rx_running = False
        thread = self._p1_rgb_rx_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.25)

        executor = self._p1_rgb_rx_executor
        node = self._p1_rgb_rx_node
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

        self._p1_rgb_rx_node = None
        self._p1_rgb_rx_executor = None
        self._p1_rgb_rx_thread = None

    def _p1_restart_rgb_receiver(self, reason: str):
        self._p1_rgb_restart_count += 1
        self.get_logger().error(
            f'[P1_RGB_RX] restarting dedicated receiver: '
            f'count={self._p1_rgb_restart_count}, reason={reason}'
        )
        self._p1_stop_rgb_receiver()
        self._p1_rgb_last_rx_monotonic_s = None
        self._p1_start_rgb_receiver()

    def _p1_sync_rgb_from_receiver(self):
        """视觉状态的 5 Hz 控制拍只复制并处理辅助节点收到的最新一帧。"""
        node = self._p1_rgb_rx_node
        if node is None:
            return False

        local_seq, msg, rx_mono = node.snapshot()
        if rx_mono is not None:
            self._p1_rgb_last_rx_monotonic_s = rx_mono

        if msg is None or local_seq == self._p1_rgb_rx_local_consumed_seq:
            return False

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(
                f'[P1_RGB_RX] cv_bridge convert failed: {exc}',
                throttle_duration_sec=1.0,
            )
            return False

        self._p1_rgb_rx_local_consumed_seq = local_seq
        self._p1_rgb_global_seq += 1
        self.latest_rgb_seq = self._p1_rgb_global_seq
        self.latest_rgb_msg = msg
        self.latest_bgr = frame

        # 第一赛段真正的 OpenCV 黄线/蓝球处理只在 5 Hz 控制线程执行。
        self.on_rgb_frame(frame)
        return True

    def p1_rgb_age_s(self):
        """辅助 receiver 最近一次收到 RGB 的墙钟年龄。"""
        last = self._p1_rgb_last_rx_monotonic_s
        if last is None:
            node = self._p1_rgb_rx_node
            if node is not None:
                _seq, _msg, last = node.snapshot()
                if last is not None:
                    self._p1_rgb_last_rx_monotonic_s = last
        if last is None:
            return None
        return max(0.0, time.monotonic() - float(last))

    def on_activated(self):
        # 该函数位于 ROS 单线程订阅回调中，不能调用 Wait_finish()；否则会连带
        # 阻塞深度和控制定时器。RGB 已由独立 receiver 线程持续接收。
        self.p1_state_start_time = None
        self.p1_rgb_stale_since = None
        self._p1_rgb_stale_stop_active = False
        self.state = self.p1_initial_state
        # 站立命令对任何入口都是安全的起手：从中段入口启动时它只是把机身摆正，
        # 下一个控制周期该入口的动作就接管。
        self.p1_send_stand_command()
        self.p1_stand_sent = True
        self.get_logger().info(
            '[P1] boot stand command sent without blocking ROS callbacks, '
            f'initial_state={self.state}')

    def on_rgb_frame(self, frame: np.ndarray):
        """只运行当前视觉状态真正需要的算法，固定时间状态不会调用这里。"""
        if self.state in ('P1_STAGE1_CRUISE', 'P1_ALIGN_STOP_LINE'):
            self.p1_process_stage1_yellow(frame)
        elif self.state == 'P1_APPROACH_BLUE_BALL':
            self.p1_process_blue_ball(frame)
        else:
            return

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

            stop_color = (0, 0, 255) if self.p1_stop_flag > 0.5 else (0, 165, 255)
            cv2.rectangle(
                vis, (stop_left, stop_top), (stop_right, stop_bottom), stop_color, 2)
            slowdown_y = int(h * self.p1_yellow_slowdown_ratio)
            threshold_y = int(h * self.p1_stop_line_y_ratio)
            # 黄线：减速阈值；红线：停车阈值。仅用于调试显示。
            cv2.line(vis, (0, slowdown_y), (w - 1, slowdown_y), (0, 255, 255), 2)
            cv2.line(vis, (0, threshold_y), (w - 1, threshold_y), (0, 0, 255), 2)
            if self.p1_stop_line_bottom_y is not None:
                cv2.circle(
                    vis, (w // 2, int(self.p1_stop_line_bottom_y)),
                    5, (255, 0, 255), -1)
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
                    vis, f"r={detection['radius_px']:.1f}px",
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 2)

            cv2.putText(vis, f'P1 state={self.state}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(
                vis,
                f'slow={self.p1_yellow_slowdown_flag:.0f} stop={self.p1_stop_flag:.0f} '
                f'visible={int(self.p1_stop_line_visible)} '
                f'bottom={self.p1_stop_line_bottom_ratio:.3f} '
                f'slow@{self.p1_yellow_slowdown_ratio:.2f} stop@{self.p1_stop_line_y_ratio:.2f} '
                f'angle={np.degrees(self.p1_stop_angle):.1f}deg',
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(
                vis,
                f'lat_force={self.p1_lateral_force:.3f} blue_cnt={self.p1_blue_count:.0f} '
                f'blue_r={self.p1_blue_max_radius_px:.1f}px '
                f'err_x={self.p1_blue_center_error_px:.1f}px '
                f'trigger={self.p1_blue_trigger_radius_px:.1f}px',
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
        # master timer 固定 30 Hz。纯定时/非视觉状态每拍都执行控制，而且完全不做
        # RGB 消费、cv_bridge 转换或 OpenCV；视觉状态则在这里节流到 5 Hz。
        vision_states = (
            'P1_STAGE1_CRUISE',
            'P1_ALIGN_STOP_LINE',
            'P1_APPROACH_BLUE_BALL',
        )

        if self.state not in vision_states:
            # 固定时间流程：30 Hz，零视觉处理。独立 receiver 仍只做轻量 ROS Image 缓存。
            self.p1_control_loop()
            return

        # 视觉状态：控制 + RGB/OpenCV 都限制到 stage1_vision_control_hz。
        now_mono = time.monotonic()
        vision_period = 1.0 / self.stage1_vision_control_hz
        last_tick = self._p1_last_vision_tick_monotonic_s
        if last_tick is not None and (now_mono - last_tick) < vision_period:
            return
        self._p1_last_vision_tick_monotonic_s = now_mono

        rgb_new = self._p1_sync_rgb_from_receiver()
        now = self.now_sec()
        rgb_age = self.p1_rgb_age_s()

        if rgb_age is None or rgb_age >= self.p1_vision_timeout_sec:
            if self.p1_rgb_stale_since is None:
                self.p1_rgb_stale_since = now

            self._p1_rgb_stale_stop_active = True
            age_text = 'none' if rgb_age is None else f'{rgb_age:.3f}s'
            thread_alive = bool(
                self._p1_rgb_rx_thread is not None
                and self._p1_rgb_rx_thread.is_alive()
            )
            self.get_logger().warning(
                f'[P1_RGB_STALE_STOP] state={self.state}, '
                f'dedicated_rgb_age={age_text}, '
                f'timeout={self.p1_vision_timeout_sec:.3f}s, '
                f'rgb_seq={self._p1_rgb_global_seq}, '
                f'rx_thread_alive={thread_alive} -> zero velocity',
                throttle_duration_sec=0.5,
            )

            if self.state in ('P1_STAGE1_CRUISE', 'P1_ALIGN_STOP_LINE'):
                step_h = (
                    0.13
                    if self.state == 'P1_STAGE1_CRUISE'
                    else self.p1_post_stop_step_height
                )
                self.p1_send_velocity_command(
                    0.0, 0.0, 0.0,
                    step_height=step_h,
                    pitch=self.p1_forward_pitch,
                )
            else:
                self.p1_send_motion_command(
                    0.0, 0.0, 0.0,
                    step_height=self.p1_post_stop_step_height,
                    pitch=0.0)

            # receiver 长时间不更新时重建，并限制重启频率。
            if rgb_age is not None:
                restart_after = max(2.0, 2.0 * self.p1_vision_timeout_sec)
                restart_cooldown = 3.0
                since_restart = now_mono - self._p1_rgb_last_restart_monotonic_s
                if (
                    rgb_age >= restart_after
                    and since_restart >= restart_cooldown
                ):
                    self._p1_restart_rgb_receiver(
                        f'age={rgb_age:.3f}s state={self.state}'
                    )
            return

        # RGB 恢复后自动继续，并补偿断图期间的视觉状态计时。
        if self._p1_rgb_stale_stop_active:
            self.get_logger().warning(
                f'[P1_RGB_RECOVER] RGB recovered: state={self.state}, '
                f'rgb_seq={self._p1_rgb_global_seq}, '
                f'age={rgb_age:.3f}s, rgb_new={rgb_new}'
            )
            self._p1_rgb_stale_stop_active = False

        if self.p1_rgb_stale_since is not None:
            stale_duration = max(0.0, now - self.p1_rgb_stale_since)
            if self.p1_state_start_time is not None:
                self.p1_state_start_time += stale_duration
            self.get_logger().info(
                f'[P1_RGB_RECOVER] state={self.state}, '
                f'paused={stale_duration:.3f}s'
            )
            self.p1_rgb_stale_since = None

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

    def p1_send_motion_command(
            self,
            vx: float = 0.0,
            vy: float = 0.0,
            wz: float = 0.0,
            *,
            step_height: float = 0.10,
            pitch: Optional[float] = None,
            body_height: Optional[float] = None):
        """第一赛段统一连续 Servo 运动命令，格式与第三赛段一致。"""
        if getattr(self, 'Ctrl', None) is None:
            self.get_logger().warning(
                '[P1 CMD] Robot_Ctrl is not active; motion command ignored',
                throttle_duration_sec=1.0)
            return

        command_pitch = self.p1_forward_pitch if pitch is None else float(pitch)
        command_body_height = (
            self.p1_stand_body_height if body_height is None else float(body_height)
        )

        vx = float(vx)
        vy = float(vy)
        wz = float(wz)
        self.motion_cmd = (vx, vy, wz)

        # 与第三赛段统一：所有状态都走 Ctrl.move() -> 303 Servo 连续步态，
        # 不再混用旧 Send_cmd() 或 StageNodeBase.send_velocity_command()。
        self.Ctrl.move(
            vx, vy, wz,
            step_height=float(step_height),
            roll=0.0,
            pitch=command_pitch,
            yaw=0.0,
            body_height=command_body_height,
            legacy_gait_id=3,
        )
        self.get_logger().info(
            f'[P1 CMD] servo vel=[{vx:.3f}, {vy:.3f}, {wz:.3f}], '
            f'pitch={command_pitch:.3f}, body_height={command_body_height:.3f}, '
            f'step_height={float(step_height):.3f}',
            throttle_duration_sec=0.3)

    def p1_send_stand_command(self):
        # 与第三赛段一样，不再发送 gait_id=0 静态站立消息；直接进入统一 Servo 通道。
        # 保留第一赛段原来的前倾姿态与机身高度。
        self.p1_send_motion_command(
            0.0, 0.0, 0.0,
            step_height=0.0,
            pitch=self.p1_forward_pitch,
            body_height=self.p1_stand_body_height)
        self.get_logger().info(
            f'[P1 CMD] SERVO HOLD / FORWARD LEAN pitch={self.p1_forward_pitch:.3f}',
            throttle_duration_sec=1.0)

    def p1_send_velocity_command(
            self, vx: float, vy: float, wz: float, step_height: float,
            pitch: float):
        """兼容原调用点；实际统一转发到 p1_send_motion_command()。"""
        self.p1_send_motion_command(
            vx, vy, wz,
            step_height=step_height,
            pitch=pitch,
            body_height=self.p1_stand_body_height)

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
        stop_top = max(0, min(h - 1, stop_top))
        stop_bottom = max(stop_top + 1, min(h, stop_bottom))
        stop_left = max(0, min(w - 1, stop_left))
        stop_right = max(stop_left + 1, min(w, stop_right))
        mask_stop = mask_yellow[stop_top:stop_bottom, stop_left:stop_right]

        # 与第二赛段类似：不再统计 ROI 内黄色像素总数，而是找有效黄色轮廓，
        # 选“下沿最靠近图像底部”的候选，并用其 line_bottom_y / image_height 判定。
        contours = find_contours(mask_stop)
        best_contour = None
        best_bottom_y = -1
        for cnt in contours:
            if cv2.contourArea(cnt) < self.p1_stop_min_contour_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            bottom_y = stop_top + y + bh
            if bottom_y > best_bottom_y:
                best_bottom_y = bottom_y
                best_contour = cnt

        stop_line_visible = best_contour is not None
        stop_line_bottom_y = None
        stop_line_bottom_ratio = 0.0
        slowdown_flag = 0.0

        if best_contour is not None:
            stop_line_bottom_y = int(best_bottom_y)
            stop_line_bottom_ratio = float(stop_line_bottom_y) / float(max(h, 1))
            slowdown_flag = (
                1.0 if stop_line_bottom_ratio >= self.p1_yellow_slowdown_ratio else 0.0
            )
            stop_flag = 1.0 if stop_line_bottom_ratio >= self.p1_stop_line_y_ratio else 0.0

            # 保留原来的停止线角度估计，供进入 P1_ALIGN_STOP_LINE 后继续调平。
            rect = cv2.minAreaRect(best_contour)
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
        self.p1_yellow_slowdown_flag = slowdown_flag
        self.p1_stop_line_visible = stop_line_visible
        self.p1_stop_line_bottom_y = stop_line_bottom_y
        self.p1_stop_line_bottom_ratio = stop_line_bottom_ratio
        self.p1_last_update_time = self.now_sec()

    def p1_process_blue_ball(self, frame: np.ndarray):
        """RGB-only 蓝球检测：按面积筛选，计算半径，并缓存最大目标的横向中心误差。"""
        self.p1_blue_detections = []
        self.p1_blue_max_radius_px = 0.0
        self.p1_blue_count = 0.0
        self.p1_blue_center_x_px = None
        self.p1_blue_center_error_px = 0.0

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array(
            [self.p1_blue_h_min, self.p1_blue_s_min, self.p1_blue_v_min],
            dtype=np.uint8)
        upper_blue = np.array(
            [self.p1_blue_h_max, self.p1_blue_s_max, self.p1_blue_v_max],
            dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours = find_contours(mask_blue)

        best_detection = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= self.p1_blue_min_area:
                continue

            (circle_x, circle_y), radius = cv2.minEnclosingCircle(cnt)
            if radius <= 0.0:
                continue

            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = int(round(circle_x))
                cy = int(round(circle_y))

            detection = {
                'center': (cx, cy),
                'radius_px': float(radius),
                'area': float(area),
                'bbox': cv2.boundingRect(cnt),
            }
            self.p1_blue_detections.append(detection)

            if best_detection is None or detection['radius_px'] > best_detection['radius_px']:
                best_detection = detection

        self.p1_blue_count = float(len(self.p1_blue_detections))
        if best_detection is not None:
            self.p1_blue_max_radius_px = float(best_detection['radius_px'])
            self.p1_blue_center_x_px = float(best_detection['center'][0])
            # 正误差表示目标在画面右侧；负误差表示目标在画面左侧。
            self.p1_blue_center_error_px = self.p1_blue_center_x_px - (w / 2.0)

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
                self.get_logger().info(
                    f'[P1] 黄线下沿达到停止阈值: '
                    f'{self.p1_stop_line_bottom_ratio:.3f} >= {self.p1_stop_line_y_ratio:.3f}，进入刹车缓冲')
                self.set_state('P1_BRAKE_BUFFER')
                return

            if elapsed >= self.p1_stage1_max_duration_sec:
                self.get_logger().info('[P1] 第一赛段行走超时，进入刹车缓冲')
                self.set_state('P1_BRAKE_BUFFER')
                return

            # 到这里说明 dedicated RGB receiver 是新鲜的；stale 情况已在
            # stage_control_loop() 开头零速度 return，不再盲目前进。
            err = self.p1_lateral_force
            turn_speed = clamp(err * self.p1_kp_turn, -self.p1_max_turn_speed, self.p1_max_turn_speed)
            lateral_speed = clamp(err * self.p1_kp_lat, -self.p1_max_lateral_speed, self.p1_max_lateral_speed)
            speed_drop = abs(err) * self.p1_kd_slowdown
            forward_speed = max(self.p1_min_forward_speed, self.p1_base_forward_speed - speed_drop)

            # 黄线下沿进入 0.80~0.90 区间后减速靠近；0.90 停车判定在上方已优先处理。
            # 使用 min() 作为速度上限，避免已有纠偏减速比黄色减速速度更慢时反而被提速。
            if self.p1_yellow_slowdown_flag > 0.5:
                forward_speed = min(forward_speed, self.p1_yellow_slow_speed)
                self.get_logger().info(
                    f'[P1] 黄线进入减速区: bottom={self.p1_stop_line_bottom_ratio:.3f} '
                    f'>= {self.p1_yellow_slowdown_ratio:.3f}, vx={forward_speed:.3f}',
                    throttle_duration_sec=0.5)

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
                step_height=self.p1_post_stop_step_height,
                pitch=self.p1_forward_pitch)
            return

        if self.state == 'P1_ALIGN_STOP_LINE':
            angle_err = self.p1_stop_angle
            if abs(angle_err) < self.p1_align_angle_deadband_rad or not self.p1_stop_line_visible:
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
                step_height=self.p1_post_stop_step_height,
                pitch=self.p1_forward_pitch)
            return

        if self.state == 'P1_RESTORE_BODY':
            if elapsed >= self.p1_restore_body_duration_sec:
                self.get_logger().info('[P1] 正常姿态恢复完成，进入调平后短暂前进')
                self.set_state('P1_POST_ALIGN_FORWARD')
                return
            self.p1_send_velocity_command(
                0.0, 0.0, 0.0,
                step_height=self.p1_post_stop_step_height,
                pitch=0.0)
            return

        if self.state == 'P1_POST_ALIGN_FORWARD':
            if elapsed >= self.p1_post_align_forward_duration_sec:
                self.get_logger().info(
                    '[P1] 调平后前进结束，转向前原地踏步并降低机身到 ' +
                    f'{self.p1_turn_body_height:.2f}m')
                self.set_state('P1_PRE_TURN_LOWER_BODY')
                return
            self.p1_send_velocity_command(
                self.p1_post_align_forward_speed, 0.0, 0.0,
                step_height=self.p1_post_stop_step_height,
                pitch=0.0)
            return

        if self.state == 'P1_PRE_TURN_LOWER_BODY':
            if elapsed >= self.p1_pre_turn_lower_duration_sec:
                self.get_logger().info(
                    f'[P1] 转向前原地踏步完成，body_height={self.p1_turn_body_height:.2f}m，进入左转')
                self.set_state('P1_TURN_LEFT_TO_STAGE2')
                return
            # 30 Hz 固定时间状态：不做视觉。保持连续 Servo 原地踏步，
            # 同时通过 body_height=0.20 将机身降低，为后续转向准备。
            self.p1_send_motion_command(
                0.0, 0.0, 0.0,
                step_height=self.p1_post_stop_step_height,
                pitch=0.0,
                body_height=self.p1_turn_body_height)
            return

        if self.state == 'P1_TURN_LEFT_TO_STAGE2':
            if elapsed >= self.p1_turn_duration_sec:
                self.get_logger().info('[P1] 左转结束，开始寻找蓝球并前进')
                self.set_state('P1_APPROACH_BLUE_BALL')
                return
            self.p1_send_motion_command(
                self.p1_turn_forward_vel, 0.0, self.p1_turn_yaw_vel,
                step_height=self.p1_post_stop_step_height,
                pitch=0.0,
                body_height=self.p1_turn_body_height)
            return

        if self.state == 'P1_APPROACH_BLUE_BALL':
            if self.p1_blue_count >= 1.0:
                self.get_logger().info(
                    f'[P1] 锁定蓝球: radius={self.p1_blue_max_radius_px:.1f}px '
                    f'/ trigger={self.p1_blue_trigger_radius_px:.1f}px',
                    throttle_duration_sec=0.5
                )
                if self.p1_blue_max_radius_px >= self.p1_blue_trigger_radius_px:
                    self.get_logger().info(
                        f'[P1] 蓝球半径达到触发阈值: '
                        f'{self.p1_blue_max_radius_px:.1f} >= '
                        f'{self.p1_blue_trigger_radius_px:.1f}px，进入盲走左移')
                    self.set_state('P1_BLIND_LEFT_SHIFT')
                    return

            if elapsed >= self.p1_approach_blue_max_duration_sec:
                self.get_logger().info('[P1] 找蓝球前进超时，进入盲走左移')
                self.set_state('P1_BLIND_LEFT_SHIFT')
                return

            blue_align_vy = 0.0
            if self.p1_blue_count >= 1.0 and self.p1_blue_center_x_px is not None:
                if self.p1_blue_center_error_px > self.p1_blue_center_deadband_px:
                    # 目标在画面右侧：机器狗向右横移（当前约定 vy<0 向右）。
                    blue_align_vy = -self.p1_blue_center_vy
                elif self.p1_blue_center_error_px < -self.p1_blue_center_deadband_px:
                    # 目标在画面左侧：机器狗向左横移（当前约定 vy>0 向左）。
                    blue_align_vy = self.p1_blue_center_vy

                self.get_logger().info(
                    f'[P1] 蓝球居中: err_x={self.p1_blue_center_error_px:.1f}px, '
                    f'vy={blue_align_vy:+.3f}',
                    throttle_duration_sec=0.5
                )

            self.p1_send_motion_command(
                self.p1_approach_blue_forward_speed, blue_align_vy, 0.0,
                step_height=self.p1_post_stop_step_height,
                pitch=0.0)
            return

        if self.state == 'P1_BLIND_LEFT_SHIFT':
            if elapsed >= self.p1_blind_left_duration_sec:
                # 关键：这里仍然不发 STOP，但进入第二赛段前重置第二赛段缓存，
                # 让第二赛段表现更接近“单独启动第二赛段”。
                self.get_logger().info('[P1] 第一赛段结束，向任务控制节点上报完成')
                self.complete_stage('P1_BLIND_LEFT_SHIFT finished')
                return

            self.p1_send_motion_command(
                self.p1_blind_left_vx, self.p1_blind_left_vy, 0.0,
                step_height=self.p1_post_stop_step_height,
                pitch=0.0)
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
            node._p1_stop_rgb_receiver()
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
