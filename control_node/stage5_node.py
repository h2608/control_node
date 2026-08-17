#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五赛段节点：独木桥/坡道稳渡，四脚过标线后跳下。

原 control_node_123456.py 的 FifthStageMixin（状态机与视觉逻辑原样搬移）。
P5_DONE 后向任务控制节点上报完成（原来是 enter_sixth_stage()）。
"""

import json
import math
import os
import time
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu

from control_node.bridge_perception import (
    BridgePerceptionConfig,
    CameraIntrinsics,
    bridge_observation,
    depth_image_to_meters,
    invalid_bridge_observation,
    rotation_gravity_from_body,
)
from control_node.deck_lateral import (
    CONTROL_BLIND,
    DeckLateralConfig,
    DeckLateralController,
)
from control_node.route_lateral import (
    SOURCE_ODOMETRY,
    USE_FULL,
    USE_HEADING_ONLY,
    anchor_from_depth,
    route_frame_observation,
    select_observation,
    snap_reference_heading,
    wrap_rad,
)
from control_node.my_gait import Robot_Odom
from control_node.route_model import (
    EXIT_ODOMETRY,
    EXIT_VISION,
    GATE_BELOW_MIN,
    GATE_IN_WINDOW,
    GATE_OVERRUN,
    GATE_UNAVAILABLE,
    PROGRESS_DISTANCE,
    PROGRESS_YAW,
    TIER_DEAD_RECKONING,
    CrossTrackGate,
    StallGate,
    ToppleGate,
    EntryDepthGate,
    RouteModel,
    SegmentProgress,
    clamp_speed,
    evaluate_gate,
    odometry_exit_reached,
    verify_yaw,
)
from control_node.stage_common import StageNodeBase
from control_node.stage_entry import EntryPoint, StageEntryTable


#: 开局站姿相对声明航向偏出这么多就值得报警：转角容差是 6 度，比它还大的
#: 摆放误差意味着航向环要先花时间把机体拧上赛道方向。
ROUTE_START_YAW_WARN_RAD = math.radians(6.0)


class Stage5Node(StageNodeBase):

    """
    第五赛段：孤梁稳渡 单独调试节点

    当前版本：
    1. 使用状态机；
    2. P5_STEP_UP 使用仿真时间；
    3. P5_UP_SLOPE 使用右侧赛道黄线消失作为结束标志；
       右侧赛道黄线连续 lost N 帧后，不再左跳/恢复/右移，
       先用固定速度运行一段时间完成转向/位置调整，
       再设置右斜坡 body，然后直接进入右斜坡 1；
    4. P5_RIGHT_SLOPE_1 / 2 使用中间区域黄色消失作为提前结束提示；
       中间区域黄色消失后不立刻 stop，而是继续前进固定时间，然后直接衔接转向；
       P5_RIGHT_SLOPE_3 检测到中间黄色消失后，不再继续额外前进，
       先发送一次速度为 0 的速度命令，再进入 reset body / 右跳转向准备流程，然后执行右跳动作，
       最后固定时间前进，到时 stop 后进入离坡跳跃；
    5. 上坡/离坡跳跃等动作类状态按新 LCM 响应序号非阻塞轮询，并由 monotonic 超时保护；
       右斜路段 1/2 之间的转向使用速度控制；右斜坡 3 结束后使用右跳动作离开/调整方向；
       离坡跳跃恢复站立后，最后再执行一次 16,6 跳远动作；
    6. 加入 OpenCV 可视化窗口；
    7. 结构尽量接近前四赛段整合代码，方便后续改成 FifthStageMixin。
    """

    # ============================================================
    # 第五赛段状态定义
    # ============================================================
    P5_RECOVERY_STAND = 'P5_RECOVERY_STAND'
    P5_SET_BODY_NORMAL = 'P5_SET_BODY_NORMAL'
    # 上桥前把机体转到声明的赛道方向上（计划书第 30 条）：恢复站立会把机体
    # 转歪，而 0.49 m 的桥面容不下 25 度的航向误差。
    P5_START_ALIGN = 'P5_START_ALIGN'

    P5_STEP_UP = 'P5_STEP_UP'
    P5_UP_SLOPE = 'P5_UP_SLOPE'
    P5_AFTER_UP_SLOPE_FORWARD = 'P5_AFTER_UP_SLOPE_FORWARD'
    P5_AFTER_UP_SLOPE_VELOCITY_CONTROL = 'P5_AFTER_UP_SLOPE_VELOCITY_CONTROL'
    P5_SET_RIGHT_SLOPE_BODY = 'P5_SET_RIGHT_SLOPE_BODY'
    P5_RIGHT_SLOPE_1 = 'P5_RIGHT_SLOPE_1'
    P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST = 'P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST'
    P5_TURN_1 = 'P5_TURN_1'
    P5_RECOVERY_AFTER_TURN_1 = 'P5_RECOVERY_AFTER_TURN_1'

    P5_RIGHT_SLOPE_2 = 'P5_RIGHT_SLOPE_2'
    P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST = 'P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST'
    P5_TURN_2 = 'P5_TURN_2'
    P5_RECOVERY_AFTER_TURN_2 = 'P5_RECOVERY_AFTER_TURN_2'

    P5_RIGHT_SLOPE_3 = 'P5_RIGHT_SLOPE_3'
    P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST = (
        'P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST'
    )
    P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP = 'P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP'
    P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2 = 'P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2'
    P5_RIGHT_JUMP_AFTER_RESET_BODY = 'P5_RIGHT_JUMP_AFTER_RESET_BODY'
    P5_ALIGN_AFTER_RIGHT_JUMP = 'P5_ALIGN_AFTER_RIGHT_JUMP'

    # 第三个转角右跳后：先前进并矫正；矫正完成后再固定前进，不再矫正。
    P5_FORWARD_AFTER_RESET_BODY = 'P5_FORWARD_AFTER_RESET_BODY'
    P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY = 'P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY'

    P5_JUMP_EXIT_SLOPE = 'P5_JUMP_EXIT_SLOPE'
    P5_RECOVERY_AFTER_JUMP_2 = 'P5_RECOVERY_AFTER_JUMP_2'
    P5_FINAL_LONG_JUMP = 'P5_FINAL_LONG_JUMP'
    P5_RESET_BODY = 'P5_RESET_BODY'

    P5_DONE = 'P5_DONE'

    # 传感器故障 / 状态超时后的保持状态：停住并等待人工恢复。
    # 视觉门控状态在图像流丢失/冻结或超时后进入这里，不再盲走。
    P5_SENSOR_FAULT_HOLD = 'P5_SENSOR_FAULT_HOLD'

    # 路线模型转角校核失败后的有界再对齐状态（计划 §4.2）：
    # 只做原地小角速度补转，超时/次数用尽即进入故障保持，绝不盲目推进。
    P5_ROUTE_REALIGN = 'P5_ROUTE_REALIGN'

    # 转角摔倒后的有界扶起（计划 37 条）：预设转身跳把机体掀翻后控制器闩在
    # kEdamp，任何命令都被 kPureDamper 覆盖。梯子 kOff -> 恢复站立 -> 重试
    # 该转角一次；扶起后若机体已横向离开赛道则仍然进故障保持。
    P5_FALL_RECOVER = 'P5_FALL_RECOVER'
    P5_STALL_RECOVER = 'P5_STALL_RECOVER'

    STAGE_ID = 5

    def __init__(self):
        super().__init__('stage5_node', self.STAGE_ID)
        self.fifth_stage_init()

    def handle_rgb_msg(self, msg: Image):
        # P5 视觉直接消费原始图像消息（原 rgb_callback 的 P5 分支）。
        self.p5_rgb_callback(msg)

    def p5_camera_info_callback(self, msg: CameraInfo):
        """Cache D435 intrinsics for the read-only observer."""
        if len(msg.k) >= 6 and msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self.p5_bridge_intrinsics = CameraIntrinsics(
                fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5])
            self.p5_bridge_intrinsics_shape = (int(msg.height), int(msg.width))
            self.p5_bridge_camera_info_frame_id = str(msg.header.frame_id)

    def p5_imu_callback(self, msg: Imu):
        """Cache IMU attitude and monotonic receive time."""
        q = msg.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-6:
            return
        x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        self.p5_imu_roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            self.p5_imu_pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            self.p5_imu_pitch = math.asin(sinp)
        self.p5_last_imu_monotonic_s = time.monotonic()
        self.p5_last_imu_stamp_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )

    def on_depth_frame(self, depth_img, msg: Image):
        """Run and record the observer without feeding current control."""
        if not self.p5_bridge_observer_enabled:
            return
        now = time.monotonic()
        if now - self.p5_bridge_last_run_monotonic_s < self.p5_bridge_observer_period_s:
            return
        self.p5_bridge_last_run_monotonic_s = now
        self.p5_run_bridge_observer(depth_img, msg)

    def start_ctrl(self):
        """Bring the read-only odometry reader up with the command link."""
        super().start_ctrl()
        if self.p5_route_model_enabled and self.Odom is None:
            self.Odom = Robot_Odom()
            self.Odom.run()
            self.get_logger().info(
                '[P5_ROUTE] Robot_Odom started (LCM state_estimator, read-only)')

    def stop_ctrl(self):
        """Tear the odometry reader down with the command link."""
        super().stop_ctrl()
        if self.Odom is not None:
            try:
                self.Odom.quit()
            except Exception as e:
                self.get_logger().warn(f'[P5_ROUTE] Robot_Odom quit failed: {e}')
            self.Odom = None
            self.get_logger().info('[P5_ROUTE] Robot_Odom stopped')

    def on_activated(self):
        # 对应原 enter_fifth_stage()。
        self.clear_pre_fifth_vision_caches()
        self.action_sent = False
        self.p5_route_realign_attempts = 0
        self.p5_fall_recovery_attempts = 0
        # Per run, not per segment: "the body went over at some point" is the
        # fact the gate latches, and a new segment does not undo it.
        self.p5_route_topple_gate.reset()
        self.p5_route_stall_gate.reset()
        self.p5_stall_recover_attempts = 0
        self.p5_stall_recover_resume_state = ''
        self.p5_enter_state(self.p5_initial_state)
        # The first active control tick sends the intentional command for this
        # state. Robot_Ctrl does not heartbeat its zero-initialized message.

    def stage_control_loop(self):
        self.p5_control_loop()
        if self.state == self.P5_DONE:
            self.complete_stage('fifth stage P5_DONE')

    def clear_pre_fifth_vision_caches(self):
        # Avoid carrying previous stage visual/debug caches into P5.
        self.latest_p5_yellow_result = {
            'has_line': False, 'line_bottom_y': None, 'line_center': None,
            'img_shape': None, 'angle_deg': None, 'abs_tilt_deg': None,
            'bbox': None, 'width_ratio': None, 'wh_ratio': None,
        }
        self.latest_p5_yellow_mask = None
        self.latest_p5_yellow_roi = None
        self.p5_yellow_stop_counter = 0
        self.p5_center_yellow_absent_counter = 0
        self.p5_center_yellow_last_eval_frame_seq = -1
        self.p5_right_side_yellow_lost_counter = 0
        self.p5_right_side_yellow_last_eval_frame_seq = -1
        self.p5_forward_align_stable_counter = 0
        self.p5_forward_align_lost_counter = 0
        self.p5_forward_align_last_eval_frame_seq = -1
        self.reset_p5_right_slope_lost_extra_state()

    def fifth_stage_init(self):
        # Reuse self.Ctrl/self.msg created by the integrated main node.
        if hasattr(self, 'msg'):
            self.msg.contact = 0

        self.latest_bgr = None
        self.latest_frame_seq = 0
        self.state_enter_frame_seq = 0

        self.latest_p5_yellow_result = {
            'has_line': False,
            'line_bottom_y': None,
            'line_center': None,
            'img_shape': None,
            'angle_deg': None,
            'abs_tilt_deg': None,
            'bbox': None,
            'width_ratio': None,
            'wh_ratio': None,
        }
        self.latest_p5_yellow_mask = None
        self.latest_p5_yellow_roi = None
        self.p5_yellow_stop_counter = 0

        # P5_RIGHT_SLOPE_1/2/3 专用：中间区域黄色存在/消失检测。
        # 右斜坡阶段相机画面会倾斜且抖动，不再强行识别“横向黄线形状”，
        # 只判断图像中间 ROI 内是否还有足够黄色像素。
        # 计数只在新图像帧上推进（帧序号门控），与上坡丢线计数保持一致，
        # 避免相机冻结时按控制 tick 快速误确认“段尾”。
        self.p5_center_yellow_absent_counter = 0
        self.p5_center_yellow_last_eval_frame_seq = -1
        self.latest_p5_center_yellow_presence_result = {
            'has_yellow': False,
            'yellow_pixels': 0,
            'roi_pixels': 0,
            'yellow_ratio': 0.0,
            'bbox': None,
            'roi': None,
            'img_shape': None,
            'reason': 'init',
        }

        # P5_UP_SLOPE 专用：右侧赛道黄线检测。
        # 注意：只有 bottom_ratio 靠近图像底部的黄线，才算“右侧赛道黄线还存在”。
        self.p5_right_side_yellow_lost_counter = 0
        self.p5_right_side_yellow_last_eval_frame_seq = -1
        self.latest_p5_right_side_yellow_result = {
            'has_line': False,
            'valid_bottom': False,
            'bbox': None,
            'center': None,
            'bottom_y': None,
            'bottom_ratio': None,
            'area': None,
            'height': None,
            'width': None,
            'roi': None,
            'candidates': [],
            'img_shape': None,
            'reason': 'init',
        }

        # P5_UP_SLOPE 专用：左右内侧黄线边缘检测，用于上坡角度/居中修正。
        self.latest_p5_inner_edge_result = {
            'mask': None,
            'left_roi': None,
            'right_roi': None,
            'left_edge': None,
            'right_edge': None,
            'has_left': False,
            'has_right': False,
            'has_both': False,
            'center_error': None,
            'heading_error': None,
            'common_valid': False,
            'common_reason': 'init',
            'cmd_vy_correction': 0.0,
            'cmd_wz_correction': 0.0,
        }

        # P5_RIGHT_SLOPE_1/2/3 专用：右侧黄线内侧边缘检测，用于三档 vy 修正。
        # 只检测右侧黄线；right_inner_x 太靠中间则加大 vy，太靠右则减小 vy。
        self.latest_p5_right_slope_right_edge_result = {
            'mask': None,
            'roi': None,
            'raw_points': [],
            'points': [],
            'valid': False,
            'reason': 'init',
            'right_inner_x': None,
            'right_inner_x_ratio': None,
            'too_center': False,
            'too_right': False,
            'cmd_vy': None,
            'base_vy': None,
            'action': 'init',
            'lost_extra_active': False,
            'lost_extra_direction': 'none',
            'too_center_count': 0,
            'too_right_count': 0,
            'record_ignore_active': False,
            'record_ignore_elapsed_s': 0.0,
            'record_ignore_duration_s': 0.0,
        }

        # P5_RIGHT_SLOPE_1/2/3 专用：
        # 右侧黄线危险趋势记忆。当前右斜坡段内，如果前面连续 too_center/too_right，
        # 后面即使右侧黄线无效/识别不到，也继续按最后确认的危险方向给额外 vy。
        # 注意：进入每一段新的 P5_RIGHT_SLOPE_x 时会清零，不跨右斜坡段继承。
        self.p5_right_slope_too_center_count = 0
        self.p5_right_slope_too_right_count = 0
        self.p5_right_slope_lost_extra_active = False
        self.p5_right_slope_lost_extra_direction = 'none'  # none / too_center / too_right
        # 本次连续丢线补偿的起始 elapsed；配合 max_hold 参数做硬时间上限。
        self.p5_right_slope_lost_extra_hold_start_s = None
        self.p5_right_slope_lost_extra_last_eval_frame_seq = -1

        # 与 ROS /clock 无关的安全计时和非阻塞动作握手状态。
        self.state_start_monotonic_s = time.monotonic()
        self.p5_action_phase = 'idle'
        self.p5_action_response_seq = 0
        self.p5_action_sent_monotonic_s = None
        self.p5_action_target = None
        self.p5_action_progress_seen = False
        self.p5_action_completed_monotonic_s = None
        self.p5_action_resends_done = 0
        self.p5_action_last_send_monotonic_s = None
        self.p5_action_recovery_pending = False
        self.p5_stop_complete_seq = 0
        self.p5_stop_complete_rx_monotonic_s = None
        self.p5_action_stall_since_monotonic_s = None
        self.p5_action_stall_bar = None
        self.p5_action_unwedge_phase = ''
        self.p5_action_unwedge_done = False
        self.p5_action_unwedge_origin = None
        self.p5_action_unwedge_sent_monotonic_s = None

        # 只读深度桥面观测状态；结果不驱动当前黄色状态机。
        self.p5_bridge_intrinsics = None
        self.p5_bridge_intrinsics_shape = None
        self.p5_bridge_camera_info_frame_id = ''
        self.p5_bridge_config = BridgePerceptionConfig()
        self.p5_bridge_last_run_monotonic_s = float('-inf')
        self.latest_bridge_observation = {'valid': False, 'reason': 'not_run'}
        # 深度桥面居中闭环（默认关闭；仿真 profile 打开，实体需过 G0/G1）。
        self.p5_deck_lateral = DeckLateralController()
        self.p5_deck_lateral_last = None
        self.p5_deck_lateral_debug_last = ''
        self.p5_reset_body_roll_from = 0.0
        # 里程兜底：段入口基准线 + 路线航向栅格锚点。
        self.p5_route_base_yaw = None
        self.p5_route_base_yaw_declared = None
        self.p5_lateral_source_last = ''
        self.p5_route_lateral_anchor_m = 0.0
        self.p5_route_lateral_anchor_suppressed_depth = False
        self.p5_route_lateral_anchor_source = ''
        self.p5_route_lateral_anchor_seq = None
        self.p5_imu_roll = 0.0
        self.p5_imu_pitch = 0.0
        self.p5_last_imu_monotonic_s = None
        self.p5_last_imu_stamp_s = None

        # 结构化证据日志（JSONL）；目录参数为空时关闭。
        self._p5_evidence_fp = None
        self._p5_evidence_failed = False

        # 路线模型运行时状态。p5_load_params() 之前必须已经是安全默认值：
        # p5_send_velocity_command() 会读 p5_route_speed_cap_enabled。
        self.Odom = None
        self.p5_route_model_enabled = False
        self.p5_route_model_mode = 'monitor'
        self.p5_route_overrun_action = 'fault'
        self.p5_route_speed_cap_enabled = False
        self.p5_route_turn_verify_enabled = False
        self.p5_route_exit_source = EXIT_VISION
        self.p5_yellow_lateral_correction_enabled = True
        self.p5_route_model = None
        self.p5_route_progress = SegmentProgress()
        self.p5_route_segment = None
        # 入段深度完整性门（p5_load_params() 前保持关闭的安全默认值）。
        self.p5_route_entry_depth_gate = EntryDepthGate()
        self.p5_route_entry_depth_faulted = False
        # 段内横向偏离门（同样保持关闭的安全默认值）。
        self.p5_route_cross_track_gate = CrossTrackGate()
        # 翻倒闩锁（整跑一次，不随段重置）；同样保持关闭的安全默认值。
        self.p5_route_topple_gate = ToppleGate()
        # 允许深度参与横向闭环的状态集合；空 = 不限制。
        self.p5_deck_lateral_centre_first_states = frozenset()
        self.p5_deck_lateral_depth_states = frozenset()
        self.p5_lateral_depth_suppressed_state = ''
        self.p5_route_odom_valid = False
        self.p5_route_odom_seq = 0
        self.p5_route_odom_age_s = None
        self.p5_route_logged_status = ''
        self.p5_route_overrun_handled = False
        self.p5_route_realign_resume_state = ''
        self.p5_route_realign_segment_name = ''
        self.p5_route_realign_attempts = 0
        self.p5_route_verified_segment = ''
        # 转角摔倒扶起（计划 37 条）。attempts 按整段计数，不随状态重置：
        # 一次摔倒用掉预算后，下一次摔倒必须直接进故障保持，否则一条持续
        # 掀翻机体的动作会被无限次重试。
        self.p5_fall_recovery_attempts = 0
        self.p5_stall_recover_resume_state = ''
        self.p5_stall_recover_attempts = 0
        self.p5_route_stall_gate = StallGate()
        self.p5_fall_recover_phase = ''
        self.p5_fall_recover_since_monotonic_s = None
        self.p5_fall_recover_retry_state = ''
        # 再对齐时沿用最近一次实际下发的体态，避免顺手改变横滚/高度。
        self.p5_last_cmd_roll = 0.0
        self.p5_last_cmd_pitch = 0.0
        self.p5_last_cmd_body_height = 0.25

        # P5_FORWARD_AFTER_RESET_BODY 专用：右跳后前进并矫正；时间到仍未对齐则 vx=0 继续矫正。
        self.p5_forward_align_stable_counter = 0
        self.p5_forward_align_lost_counter = 0
        self.p5_forward_align_last_eval_frame_seq = -1

        self.p5_declare_params()
        self.p5_load_params()
        self.p5_camera_info_sub = self.create_subscription(
            CameraInfo,
            self.p5_depth_camera_info_topic,
            self.p5_camera_info_callback,
            qos_profile_sensor_data,
        )
        self.p5_imu_sub = self.create_subscription(
            Imu, self.p5_imu_topic, self.p5_imu_callback, qos_profile_sensor_data)
        # Compatibility attribute only; StageNodeBase owns the actual RGB
        # subscription and its rgb_topic parameter.
        self.p5_rgb_topic = self.rgb_topic

        # RGB subscription is owned by the integrated node; it dispatches P5 frames in rgb_callback().

        # Do not switch into P5 here; enter_fifth_stage() performs the handoff after P4.
        self.action_sent = False

        # Main timer is owned by CombinedStage1Stage2Node.

        self.get_logger().info(
            f'[P5] fifth stage mixin initialized, '
            f'p5_initial_state={self.p5_initial_state}, current_state={self.state}, '
            f'use_sim_time={self.get_parameter("use_sim_time").value}, '
            f'rgb_topic={self.p5_rgb_topic}'
        )

    # ============================================================
    # 调试入口
    # ============================================================
    @classmethod
    def p5_entry_table(cls) -> StageEntryTable:
        """第五赛段调试入口表（顺序即绕环一圈的顺序）。

        入口名和 route_model.default_segments() 的段名对齐（ramp = up_slope，
        straight_1/2/3、corner_1..4），这样跑到一半的证据日志和调试入口用的是
        同一套词。route_model 通过状态名反查段，所以从中段进入不会破坏段模型：
        它会直接从对应段开始积分。

        但**里程只能给出沿路线的进度，给不出相对桥面中线的偏移**。从中段进入
        时机器人必须已经被摆在该段的起点、朝向该段方向，否则整段都会带着一个
        固定的横向误差走完。
        """
        states = (
            cls.P5_RECOVERY_STAND,
            cls.P5_SET_BODY_NORMAL,
            cls.P5_START_ALIGN,
            cls.P5_STEP_UP,
            cls.P5_UP_SLOPE,
            cls.P5_AFTER_UP_SLOPE_FORWARD,
            cls.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL,
            cls.P5_SET_RIGHT_SLOPE_BODY,
            cls.P5_RIGHT_SLOPE_1,
            cls.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST,
            cls.P5_TURN_1,
            cls.P5_RECOVERY_AFTER_TURN_1,
            cls.P5_RIGHT_SLOPE_2,
            cls.P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST,
            cls.P5_TURN_2,
            cls.P5_RECOVERY_AFTER_TURN_2,
            cls.P5_RIGHT_SLOPE_3,
            cls.P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST,
            cls.P5_RESET_BODY,
            cls.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP,
            cls.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2,
            cls.P5_RIGHT_JUMP_AFTER_RESET_BODY,
            cls.P5_ALIGN_AFTER_RIGHT_JUMP,
            cls.P5_FORWARD_AFTER_RESET_BODY,
            cls.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY,
            cls.P5_JUMP_EXIT_SLOPE,
            cls.P5_RECOVERY_AFTER_JUMP_2,
            cls.P5_FINAL_LONG_JUMP,
            cls.P5_DONE,
        )
        placed = '机器人必须已经被摆在该段起点并朝向该段方向'
        return StageEntryTable(cls.STAGE_ID, cls.P5_SET_BODY_NORMAL, states, (
            EntryPoint('start', cls.P5_SET_BODY_NORMAL,
                       '设置机身姿态，完整第五赛段'),
            EntryPoint('recovery', cls.P5_RECOVERY_STAND, '先恢复站立再走完整流程'),
            EntryPoint('align', cls.P5_START_ALIGN, '上桥前对齐声明的赛道方向'),
            EntryPoint('step_up', cls.P5_STEP_UP, '上台阶（route 段 entry_step_up）',
                       requires=(placed,)),
            EntryPoint('ramp', cls.P5_UP_SLOPE, '上坡段（route 段 up_slope）',
                       requires=(placed, '已经站在坡道上')),
            EntryPoint('ramp_exit', cls.P5_AFTER_UP_SLOPE_FORWARD,
                       '上坡结束后的定时前进', requires=(placed,)),
            EntryPoint('corner_1', cls.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL,
                       '第一个转角', requires=(placed,)),
            EntryPoint('slope_body', cls.P5_SET_RIGHT_SLOPE_BODY, '设置侧坡机身姿态'),
            EntryPoint('straight_1', cls.P5_RIGHT_SLOPE_1, '直道 1（route 段 straight_1）',
                       requires=(placed,)),
            EntryPoint('corner_2', cls.P5_TURN_1, '第二个转角', requires=(placed,)),
            EntryPoint('straight_2', cls.P5_RIGHT_SLOPE_2, '直道 2（route 段 straight_2）',
                       requires=(placed,)),
            EntryPoint('corner_3', cls.P5_TURN_2, '第三个转角', requires=(placed,)),
            EntryPoint('straight_3', cls.P5_RIGHT_SLOPE_3, '直道 3（route 段 straight_3）',
                       requires=(placed,)),
            EntryPoint('reset_body', cls.P5_RESET_BODY, '直道 3 结束后恢复机身姿态'),
            EntryPoint('corner_4', cls.P5_RIGHT_JUMP_AFTER_RESET_BODY,
                       '第四个转角（转向跳）', requires=(placed,)),
            EntryPoint('descent', cls.P5_FORWARD_AFTER_RESET_BODY,
                       '下坡直道（route 段 right_descent）', requires=(placed,)),
            EntryPoint('final', cls.P5_JUMP_EXIT_SLOPE,
                       '收尾区：跳下坡道（route 段 final_zone）', requires=(placed,)),
            EntryPoint('final_jump', cls.P5_FINAL_LONG_JUMP, '收尾远跳',
                       requires=(placed,)),
        ))

    # ============================================================
    # 参数声明
    # ============================================================
    def p5_declare_params(self):
        # 调试入口：可写具名入口（ramp、straight_2、final_jump …）或直接写
        # 状态名。统一的 entry_point 参数（launch 里的 stage5_entry）优先。
        self.declare_parameter('p5_initial_state', 'default')
        self.declare_parameter('p5_control_period_s', 0.02)

        # RGB / 可视化
        self.declare_parameter('p5_show_debug_vis', self.show_debug_vis)
        # 可视化详细程度：
        # 0 = 不显示窗口；1 = 简洁模式，只显示关键控制信息；2 = 详细模式，显示所有调试点/ROI/原因。
        self.declare_parameter('p5_debug_vis_detail_level', 2)
        self.declare_parameter('p5_show_yellow_mask', self.show_yellow_mask)

        # 启动时先设置 normal body
        self.declare_parameter('p5_body_normal_roll', 0.0)
        self.declare_parameter('p5_body_normal_height', 0.25)
        self.declare_parameter('p5_body_normal_wait_s', 0.3)

        # 第五赛段上台阶步态：这一段仍然按仿真时间
        self.declare_parameter('p5_step_up_vx', 0.40)
        self.declare_parameter('p5_step_up_vy', 0.0)
        self.declare_parameter('p5_step_up_wz', 0.0)
        self.declare_parameter('p5_step_up_step_height', 0.13)
        self.declare_parameter('p5_step_up_duration_s', 3.0)

        # 第五赛段上坡路段步态：视觉黄线结束
        self.declare_parameter('p5_up_slope_vx', 0.30)
        self.declare_parameter('p5_up_slope_vy', 0.0)
        self.declare_parameter('p5_up_slope_wz', 0.0)
        self.declare_parameter('p5_up_slope_step_height', 0.10)
        self.declare_parameter('p5_up_slope_roll', 0.0)
        self.declare_parameter('p5_up_slope_pitch', 0.20)

        # P5 上坡阶段：左右内侧黄线边缘矫正
        # 只在 P5_UP_SLOPE 中使用；右侧黄线消失仍然负责上坡结束。
        self.declare_parameter('p5_inner_edge_align_enabled', True)
        self.declare_parameter('p5_inner_edge_enable_vy', True)
        self.declare_parameter('p5_inner_edge_enable_wz', True)

        # ROI：左右下方区域。左 ROI 取黄色区域最右边缘，右 ROI 取黄色区域最左边缘。
        self.declare_parameter('p5_inner_edge_left_roi_x_min', 0.05)
        self.declare_parameter('p5_inner_edge_left_roi_x_max', 0.48)
        self.declare_parameter('p5_inner_edge_right_roi_x_min', 0.52)
        self.declare_parameter('p5_inner_edge_right_roi_x_max', 0.95)
        self.declare_parameter('p5_inner_edge_roi_y_min', 0.60)
        self.declare_parameter('p5_inner_edge_roi_y_max', 1.00)

        # 内侧边缘有效性判断
        self.declare_parameter('p5_inner_edge_min_points', 80)
        self.declare_parameter('p5_inner_edge_min_y_span', 80.0)
        self.declare_parameter('p5_inner_edge_bottom_min_ratio', 0.95)
        self.declare_parameter('p5_inner_edge_x_std_max', 90.0)
        self.declare_parameter('p5_inner_edge_use_bottom_connected_segment', True)
        self.declare_parameter('p5_inner_edge_max_y_gap', 8)
        self.declare_parameter('p5_inner_edge_min_common_y_span', 50.0)
        self.declare_parameter('p5_inner_edge_row_step', 1)
        self.declare_parameter('p5_inner_edge_top_bottom_band_ratio', 0.20)

        # 控制增益：center_error 像素 -> vy；heading_error 像素 -> wz。
        # 当前默认符号：center_error>0 表示赛道中心在图像右侧，给负 vy 向右修正；
        # heading_error>0 表示远端中心偏右，给负 wz 右转修正。
        self.declare_parameter('p5_inner_edge_center_k_vy', 0.0012)
        self.declare_parameter('p5_inner_edge_heading_k_wz', 0.0020)
        self.declare_parameter('p5_inner_edge_vy_max_correction', 0.08)
        self.declare_parameter('p5_inner_edge_wz_max_correction', 0.18)
        self.declare_parameter('p5_inner_edge_center_deadband_px', 8.0)
        self.declare_parameter('p5_inner_edge_heading_deadband_px', 6.0)

        # P5 上坡阶段：右侧赛道黄线消失判定
        self.declare_parameter('p5_right_side_yellow_roi_x_min', 0.50)
        self.declare_parameter('p5_right_side_yellow_roi_x_max', 1.00)
        self.declare_parameter('p5_right_side_yellow_roi_y_min', 0.50)
        self.declare_parameter('p5_right_side_yellow_roi_y_max', 1.00)

        # 右侧黄线检测不需要太严格，但必须接近图像底部才算有效
        self.declare_parameter('p5_right_side_yellow_min_area', 80.0)
        self.declare_parameter('p5_right_side_yellow_min_height', 20)
        self.declare_parameter('p5_right_side_yellow_min_width', 2)
        self.declare_parameter('p5_right_side_yellow_bottom_valid_ratio', 0.95)

        # 上坡黄线消失后，先沿桥中心线继续前进，确保机身进入下一段坡面，
        # 再原地转向。不要把“补足前进距离”和“大角度转向”合并成弧线运动：
        # 1.22 rad 前视场下黄线会在桥末前提前离开图像底部，旧弧线会把机器人
        # 推到桥面左边缘，再叠加右斜坡侧倾后直接跌落。
        self.declare_parameter('p5_after_up_slope_forward_duration_s', 1.9)
        self.declare_parameter('p5_after_up_slope_forward_vx', 0.30)
        self.declare_parameter('p5_after_up_slope_forward_vy', 0.0)
        self.declare_parameter('p5_after_up_slope_forward_wz', 0.0)
        self.declare_parameter('p5_after_up_slope_forward_step_height', 0.04)

        # 补足前进距离后在角点转向。race.world 的 0.5 m 窄桥上，
        # 速度原地转向仍会产生明显质心漂移；仿真档使用已验证的 +90°
        # 定点动作，velocity 仅保留为兼容/标定选项。
        self.declare_parameter('p5_after_up_slope_turn_method', 'right_jump')
        self.declare_parameter('p5_after_up_slope_turn_jump_mode', 16)
        self.declare_parameter('p5_after_up_slope_turn_jump_gait', 0)
        self.declare_parameter('p5_after_up_slope_control_duration_s', 3.2)
        self.declare_parameter('p5_after_up_slope_control_vx', 0.0)
        self.declare_parameter('p5_after_up_slope_control_vy', 0.00)
        self.declare_parameter('p5_after_up_slope_control_wz', 0.55)
        self.declare_parameter('p5_after_up_slope_control_step_height', 0.04)

        # 右侧赛道黄线连续消失 N 帧后，不再左跳/恢复/右移，
        # 先固定速度运行一段时间完成转向/位置调整，再设置右斜坡 body。
        # 入口处机身俯仰会让黄线底点短暂抬高；一帧低于 0.95 不能代表桥末。
        # 先让机身稳定进入桥面，再用连续帧确认，避免在上坡起点过早转弯。
        self.declare_parameter('p5_right_side_yellow_lost_confirm_count', 3)
        self.declare_parameter('p5_right_side_yellow_ignore_after_enter_s', 1.5)

        # 右斜坡 body
        self.declare_parameter('p5_right_slope_roll', -0.60)
        self.declare_parameter('p5_right_slope_height', 0.25)
        self.declare_parameter('p5_right_slope_body_wait_s', 0.3)

        # 右斜坡阶段：右侧黄线内侧边缘三档 vy 修正。
        # 只在 P5_RIGHT_SLOPE_1/2/3 主运动阶段使用，不参与转向和额外前进。
        # right_inner_x 太靠中间：认为机器狗往坡下滑，固定加大 vy；
        # right_inner_x 太靠右：认为机器狗往坡上爬，固定减小 vy。
        self.declare_parameter('p5_right_slope_right_edge_vy_adjust_enabled', True)

        self.declare_parameter('p5_right_slope_right_edge_roi_x_min', 0.45)
        self.declare_parameter('p5_right_slope_right_edge_roi_x_max', 1.00)
        self.declare_parameter('p5_right_slope_right_edge_roi_y_min', 0.60)
        self.declare_parameter('p5_right_slope_right_edge_roi_y_max', 1.00)

        self.declare_parameter('p5_right_slope_right_edge_row_step', 1)
        self.declare_parameter('p5_right_slope_right_edge_use_bottom_connected_segment', True)
        self.declare_parameter('p5_right_slope_right_edge_max_y_gap', 5)

        self.declare_parameter('p5_right_slope_right_edge_min_points', 100)
        self.declare_parameter('p5_right_slope_right_edge_min_y_span', 100.0)
        self.declare_parameter('p5_right_slope_right_edge_x_std_max', 20.0)
        self.declare_parameter('p5_right_slope_right_edge_bottom_min_ratio', 0.90)
        self.declare_parameter('p5_right_slope_right_edge_bottom_band_ratio', 0.20)

        self.declare_parameter('p5_right_slope_right_too_center_ratio', 0.50)
        self.declare_parameter('p5_right_slope_right_too_right_ratio', 0.65)
        self.declare_parameter('p5_right_slope_right_too_center_add_vy', 0.1)
        self.declare_parameter('p5_right_slope_right_too_right_reduce_vy', 0.1)

        # 右斜坡阶段：危险趋势确认后，如果后续右侧黄线识别不到，
        # 则持续给额外 vy，直到当前 P5_RIGHT_SLOPE_x 状态结束。
        self.declare_parameter('p5_right_slope_lost_extra_enabled', True)

        # 每段 P5_RIGHT_SLOPE_1/2/3 刚进入后的前一小段时间，
        # 只执行可见黄线 vy 修正，但不累计 too_center / too_right 危险次数，
        # 也不激活后续丢线持续补偿。
        self.declare_parameter('p5_right_slope_lost_extra_ignore_after_enter_s', 1.0)

        # 连续多少帧 too_center / too_right 后，才认为危险趋势成立。
        self.declare_parameter('p5_right_slope_lost_extra_confirm_count', 3)

        # 后续丢线时，如果之前确认的是 too_center，则叠加这个 vy。
        # 默认方向与“too_center 时加大 vy”一致。
        self.declare_parameter('p5_right_slope_lost_extra_too_center_vy', 0.035)

        # 后续丢线时，如果之前确认的是 too_right，则叠加这个 vy。
        # 默认方向与“too_right 时减小 vy”一致，所以是负数。
        # 如果实测方向反了，直接把这个参数改成正数。
        self.declare_parameter('p5_right_slope_lost_extra_too_right_vy', -0.035)


        # 右斜坡 1：视觉黄线结束
        self.declare_parameter('p5_right_slope_1_vx', 0.30)
        self.declare_parameter('p5_right_slope_1_vy', 0.1325)
        self.declare_parameter('p5_right_slope_1_wz', 0.0)
        self.declare_parameter('p5_right_slope_1_step_height', 0.04)
        # 3_slope.stl 的第一段中心线拐点约为 (-0.380, 12.410)。
        # run5 在 x≈0.016 就因 1.22 rad 视场丢失中间黄线，需继续约 0.40 m。
        self.declare_parameter('p5_right_slope_1_after_center_lost_duration_s', 1.35)
        self.declare_parameter('p5_right_slope_1_after_center_lost_vx', 0.30)
        self.declare_parameter('p5_right_slope_1_after_center_lost_vy', 0.13)
        self.declare_parameter('p5_right_slope_1_after_center_lost_wz', 0.0)
        self.declare_parameter('p5_right_slope_1_after_center_lost_step_height', 0.04)

        # 右斜赛段 1/2 之间的转向方式：
        # "velocity"   ：使用当前的速度控制转向参数 p5_turn_1_* / p5_turn_2_*
        # "right_jump" ：使用右跳动作 mode/gait 转向
        self.declare_parameter('p5_right_slope_turn_method', 'right_jump')
        self.declare_parameter('p5_right_slope_turn_1_jump_mode', 16)
        self.declare_parameter('p5_right_slope_turn_1_jump_gait', 3)
        self.declare_parameter('p5_right_slope_turn_2_jump_mode', 16)
        self.declare_parameter('p5_right_slope_turn_2_jump_gait', 3)
        self.declare_parameter('p5_right_slope_turn_jump_stop_after_finish', True)

        # 转弯 1：不再使用右跳动作，改为速度控制固定时间转向
        # 由于机器狗仍在右斜坡上，转向时保留较小 vx / vy，让它继续贴着坡面运动。
        self.declare_parameter('p5_turn_1_duration_s', 5.85)
        self.declare_parameter('p5_turn_1_vx', 0.05)
        self.declare_parameter('p5_turn_1_vy', 0.0)
        self.declare_parameter('p5_turn_1_wz', -0.70)
        self.declare_parameter('p5_turn_1_step_height', 0.04)

        # 右斜坡 2：视觉黄线结束
        self.declare_parameter('p5_right_slope_2_vx', 0.30)
        # run8 转向落点 x≈-0.524、深度横移≈-0.15 m（机体偏左）；
        # 小幅负 vy 在第二直段向东回中，避免原正 vy 继续推向西侧边界。
        self.declare_parameter('p5_right_slope_2_entry_recenter_duration_s', 0.8)
        self.declare_parameter('p5_right_slope_2_entry_recenter_vy', -0.075)
        self.declare_parameter('p5_right_slope_2_vy', 0.0)
        self.declare_parameter('p5_right_slope_2_right_edge_adjust_enabled', False)
        self.declare_parameter('p5_right_slope_2_wz', 0.0)
        self.declare_parameter('p5_right_slope_2_step_height', 0.04)
        # 3_slope.stl 的第二段中心线拐点约为 (-0.380, 15.410)。
        # run6 的 1.40 s 到 y≈15.42，实测第三个转弯稍晚；提前约 0.06 m。
        self.declare_parameter('p5_right_slope_2_after_center_lost_duration_s', 1.20)
        self.declare_parameter('p5_right_slope_2_after_center_lost_vx', 0.30)
        self.declare_parameter('p5_right_slope_2_after_center_lost_vy', 0.0)
        self.declare_parameter('p5_right_slope_2_after_center_lost_wz', 0.0)
        self.declare_parameter('p5_right_slope_2_after_center_lost_step_height', 0.04)

        # 转弯 2：不再使用右跳动作，改为速度控制固定时间转向
        self.declare_parameter('p5_turn_2_duration_s', 5.9)
        self.declare_parameter('p5_turn_2_vx', 0.05)
        self.declare_parameter('p5_turn_2_vy', 0.00)
        self.declare_parameter('p5_turn_2_wz', -0.7)
        self.declare_parameter('p5_turn_2_step_height', 0.04)

        # 右斜坡 3：视觉黄线结束
        self.declare_parameter('p5_right_slope_3_vx', 0.30)
        # run6 第三转后实际轨迹维持在 y≈15.47，较 15.410 中心线偏左；
        # 只降低第三段的横向补偿，前两段保持已验证的 0.1325。
        self.declare_parameter('p5_right_slope_3_vy', 0.11)
        self.declare_parameter('p5_right_slope_3_wz', 0.0)
        self.declare_parameter('p5_right_slope_3_step_height', 0.04)

        # 第三段中心线终点约为 (3.120, 15.410)。run5 在 x≈2.68 丢线，
        # 所以保持斜坡 body 再直行约 0.44 m；到达拐点后才 reset body。
        self.declare_parameter('p5_right_slope_3_after_center_lost_duration_s', 1.50)
        self.declare_parameter('p5_right_slope_3_after_center_lost_vx', 0.30)
        self.declare_parameter('p5_right_slope_3_after_center_lost_vy', 0.11)
        self.declare_parameter('p5_right_slope_3_after_center_lost_wz', 0.0)
        self.declare_parameter('p5_right_slope_3_after_center_lost_step_height', 0.04)

        # 旧流程在 reset body 后直接侧移 1.25 s，run5 实测在拐点前滑倒。
        # 参数保留用于兼容旧 launch 覆盖；当前状态机不再进入这两个侧移状态。
        self.declare_parameter('p5_right_shift_before_right_jump_duration_s', 0.25)
        self.declare_parameter('p5_right_shift_before_right_jump_vx', 0.10)
        self.declare_parameter('p5_right_shift_before_right_jump_vy', -0.30)
        self.declare_parameter('p5_right_shift_before_right_jump_wz', 0.0)
        self.declare_parameter('p5_right_shift_before_right_jump_step_height', 0.04)

        # 第一段右移结束后，再继续右移一段固定时间，然后才执行右跳动作。
        self.declare_parameter('p5_right_shift_before_right_jump_2_duration_s', 1.0)
        self.declare_parameter('p5_right_shift_before_right_jump_2_vx', 0.0)
        self.declare_parameter('p5_right_shift_before_right_jump_2_vy', -0.30)
        self.declare_parameter('p5_right_shift_before_right_jump_2_wz', 0.0)
        self.declare_parameter('p5_right_shift_before_right_jump_2_step_height', 0.04)

        self.declare_parameter('p5_right_jump_after_reset_body_mode', 16)
        self.declare_parameter('p5_right_jump_after_reset_body_gait', 3)

        # run6 第四个右跳落地 yaw≈-1.33 rad，距离桥中心方向 -pi/2
        # 仍差约 -0.24 rad；若立刻前进会持续向世界 +x 偏出赛道。
        # 先原地补齐航向，再沿 2_bridge 中心线前进。
        self.declare_parameter('p5_align_after_right_jump_duration_s', 1.30)
        self.declare_parameter('p5_align_after_right_jump_vx', 0.0)
        self.declare_parameter('p5_align_after_right_jump_vy', 0.0)
        self.declare_parameter('p5_align_after_right_jump_wz', -0.20)
        self.declare_parameter('p5_align_after_right_jump_step_height', 0.04)

        # 第三个转角右跳后第一段：固定时间前进，同时使用内侧边缘做 vy/wz 矫正。
        # 如果时间到了还没对齐，则 vx=0，继续原地横移/转向矫正。
        self.declare_parameter('p5_forward_after_reset_body_duration_s', 1.5)
        self.declare_parameter('p5_forward_after_reset_body_vx', 0.50)
        self.declare_parameter('p5_forward_after_reset_body_vy', 0.0)
        self.declare_parameter('p5_forward_after_reset_body_wz', 0.0)
        self.declare_parameter('p5_forward_after_reset_body_step_height', 0.04)

        self.declare_parameter('p5_forward_after_reset_body_hold_align_enabled', True)
        self.declare_parameter('p5_forward_after_reset_body_align_center_done_px', 12.0)
        self.declare_parameter('p5_forward_after_reset_body_align_heading_done_px', 8.0)
        self.declare_parameter('p5_forward_after_reset_body_align_stable_frames', 5)
        # 前进结束后如果连续多帧看不到两侧边缘，原地等待不会恢复可观测性；
        # 立即结束视觉矫正，避免按 max_extra_s 长时间发送零速度。
        self.declare_parameter('p5_forward_after_reset_body_align_lost_confirm_frames', 3)
        # 0 表示不设额外最长矫正时间；如果担心卡死，可以设成 2.0 / 3.0 等。
        self.declare_parameter('p5_forward_after_reset_body_align_max_extra_s', 5.0)

        # 第三个转角右跳后第二段：矫正完成后，再固定前进一段时间，不再叠加视觉矫正。
        # 这一段结束后才 stop，然后进入右跳/跳远流程。
        self.declare_parameter('p5_forward_no_align_after_reset_body_duration_s', 1.5)
        self.declare_parameter('p5_forward_no_align_after_reset_body_vx', 0.50)
        self.declare_parameter('p5_forward_no_align_after_reset_body_vy', 0.0)
        self.declare_parameter('p5_forward_no_align_after_reset_body_wz', 0.0)
        self.declare_parameter('p5_forward_no_align_after_reset_body_step_height', 0.04)

        # 离开坡度区右跳
        self.declare_parameter('p5_jump_exit_slope_mode', 16)
        self.declare_parameter('p5_jump_exit_slope_gait', 3)

        # 最后跳远动作
        self.declare_parameter('p5_final_long_jump_mode', 16)
        self.declare_parameter('p5_final_long_jump_gait', 1)

        # 离开坡度区前 reset body
        self.declare_parameter('p5_reset_roll', 0.0)
        self.declare_parameter('p5_reset_height', 0.25)
        self.declare_parameter('p5_reset_body_wait_s', 0.3)
        # 从赛道侧倾姿态恢复水平的斜坡时间（秒，0 = 一次到位，保持旧行为）。
        # 一次到位等于把重心一步挪过整个横坡；race.world 有路缘接着，
        # race_physical 没有，实测机体在三号边尾端 roll -0.47 -> +0.55 翻下去。
        self.declare_parameter('p5_reset_body_ramp_s', 0.0)

        # =========================
        # 严格前方横向黄线检测参数
        # =========================
        self.declare_parameter('p5_yellow_roi_top_ratio', 0.45)
        self.declare_parameter('p5_yellow_roi_left_ratio', 0.40)
        self.declare_parameter('p5_yellow_roi_right_ratio', 0.60)

        self.declare_parameter('p5_yellow_h_min', 15)
        self.declare_parameter('p5_yellow_h_max', 40)
        self.declare_parameter('p5_yellow_s_min', 80)
        self.declare_parameter('p5_yellow_s_max', 255)
        self.declare_parameter('p5_yellow_v_min', 80)
        self.declare_parameter('p5_yellow_v_max', 255)

        self.declare_parameter('p5_yellow_min_contour_area', 100.0)
        self.declare_parameter('p5_yellow_min_width_height_ratio', 2.0)
        self.declare_parameter('p5_yellow_max_tilt_deg', 30.0)
        self.declare_parameter('p5_yellow_center_tolerance_ratio', 0.28)
        self.declare_parameter('p5_yellow_min_width_ratio', 0.18)

        # 黄线到底判定
        self.declare_parameter('p5_yellow_stop_line_y_ratio', 0.95)
        self.declare_parameter('p5_yellow_stop_confirm_count', 1)

        # 进入新状态后，忽略刚进入时的旧帧/旧黄线一小段时间
        self.declare_parameter('p5_yellow_ignore_after_enter_s', 0.3)

        # =========================
        # 右斜坡阶段：中间区域黄色消失检测
        # =========================
        # 右斜坡阶段相机歪斜、抖动大，因此不再要求横线形状。
        # 只要中间 ROI 内黄色像素少于阈值，并连续确认 N 帧，就认为当前斜坡段结束。
        self.declare_parameter('p5_center_yellow_roi_x_min', 0.35)
        self.declare_parameter('p5_center_yellow_roi_x_max', 0.65)
        self.declare_parameter('p5_center_yellow_roi_y_min', 0.35)
        self.declare_parameter('p5_center_yellow_roi_y_max', 1.00)
        self.declare_parameter('p5_center_yellow_min_pixels', 80)
        self.declare_parameter('p5_center_yellow_min_ratio', 0.002)
        self.declare_parameter('p5_center_yellow_absent_confirm_count', 3)
        self.declare_parameter('p5_center_yellow_ignore_after_enter_s', 0.3)

        # 状态进入后一直没有新图像时是否继续走。
        # 实体课程默认 fail-safe：停住（发零速度）而不是盲走。
        # 仿真档在 config/stage5_sim.yaml 里显式改回 True 以保持旧行为。
        self.declare_parameter('p5_keep_moving_when_no_image', False)

        # =========================
        # 传感器健康 watchdog / 状态超时（fail-safe 层）
        # =========================
        # 视觉门控状态里，RGB 流超过 max_frame_age 没有新帧即判定故障，
        # 切入 P5_SENSOR_FAULT_HOLD 停住等待人工恢复；grace 为状态进入后的宽限期
        # （阻塞动作结束后队列里的旧帧需要一点时间被处理掉）。
        self.declare_parameter('p5_sensor_watchdog_enabled', True)
        self.declare_parameter('p5_sensor_max_frame_age_s', 2.0)
        self.declare_parameter('p5_sensor_fault_grace_s', 2.0)

        # 各视觉门控状态的整体超时（0 = 不限制）。
        # P5_UP_SLOPE 原实现没有超时，相机冻结在“有黄帧”上会无限前进。
        self.declare_parameter('p5_up_slope_timeout_s', 60.0)
        self.declare_parameter('p5_right_slope_1_timeout_s', 45.0)
        self.declare_parameter('p5_right_slope_2_timeout_s', 45.0)
        self.declare_parameter('p5_right_slope_3_timeout_s', 45.0)

        # 右斜坡丢线补偿的最长持续时间（0 = 不限制，兼容旧仿真行为）。
        # 到达上限后撤销危险记忆，需要重新连续确认才会再次补偿。
        self.declare_parameter('p5_right_slope_lost_extra_max_hold_s', 2.0)

        # 结构化证据日志目录（JSONL，每次激活一个文件）；空字符串 = 关闭。
        self.declare_parameter('p5_evidence_log_dir', '')

        # 动作握手使用 monotonic 非阻塞轮询；超时必须进入故障保持，不能盲目进下一状态。
        self.declare_parameter('p5_action_timeout_s', 12.0)
        self.declare_parameter('p5_stop_timeout_s', 5.0)
        self.declare_parameter('p5_action_min_ack_delay_s', 0.10)
        self.declare_parameter('p5_action_require_progress', True)
        # 丢单重发（计划 19 条）：控制器活着却始终回显别的 mode 时重发动作
        # 命令。resend_max=0 关闭（代码默认保持原行为）。
        self.declare_parameter('p5_action_resend_max', 0)
        self.declare_parameter('p5_action_resend_after_s', 3.0)
        # 卡死解楔（计划 33 条）：控制器回显的正是本目标、却把
        # order_process_bar 永久钉在 0 —— 动作被接收、执行完毕，只是它自己的
        # 完成判据不成立。此时 FsmStateJump3d::Transition() 对除 kOff 以外的
        # 每一个出口都要求同一组判据，所以连 kPureDamper / kRecoveryStand 都
        # 会被无限期挂起；唯一无条件放行的出口是 kOff（mode 0，手柄 back 键
        # 映射的那个）。梯子：kOff 解锁 -> kRecoveryStand 站起 -> 把动作目标
        # 改写成这次恢复站立，由既有完成逻辑收尾。
        # 0 = 关闭（代码默认保持原行为）。仿真档按实测启用。
        self.declare_parameter('p5_action_stall_unwedge_after_s', 0.0)
        self.declare_parameter('p5_action_unwedge_release_timeout_s', 5.0)
        # 转角摔倒扶起（计划 37 条）。与上面的解楔互补：解楔处理"控制器接受了
        # 动作却永远不报完成"，这里处理"动作把机体掀翻了"。
        #
        # 2026-08-06 36 跑实测：四条失败全部是预设转身跳（mode 16 / gait 3）
        # 把机体掀翻，回显 mode 7 / switch_status 3（kEdamp）/ bar 100。
        # control_fsm.cpp 的 kEdamp 分支把 current_state_ 强制换成 kPureDamper，
        # 并且只有 edamp_iter > 1550 才重新读 CheckTransition()；恢复站立一旦
        # 触发 SafetyPostCheck 又会立刻退回 kEdamp 并把计数清零，所以实测 37 s
        # 里机体一直躺着。唯一能离开的出口和解楔一样是 kOff。
        #
        # 因此这条梯子只在"机体确实已经躺倒"时才发 kOff：躺倒的机体本来就在
        # 地上，kOff 的代价是零；直立时误发才是危险的，所以用里程计的
        # roll/pitch 做前置判据，而不是只看回显。
        self.declare_parameter('p5_fall_recovery_enabled', False)
        self.declare_parameter('p5_fall_recovery_max_attempts', 1)
        # 机体确实躺倒的判据：|roll| 或 |pitch| 超过该值。实测四次摔倒静止时
        # |roll| 为 1.95~2.13 rad，正常行走全程 < 0.62 rad（右坡体态 -0.60）。
        self.declare_parameter('p5_fall_recovery_min_rp_rad', 1.20)
        self.declare_parameter('p5_fall_recovery_release_timeout_s', 5.0)
        # 从完全侧躺起身比从 kLifted 起身慢得多（第 19 条实测 17.7 s）。
        # 2026-08-07 两次实测：kOff 释放 kEdamp 只要 0.09~0.10 s，但随后的
        # 恢复站立会在 bar=0 上停 10 s，直到第 33 条的解楔梯子再发一次 kOff
        # 才真正开始；从那一刻起翻正（|roll| 回到 0.20 以内）要 22.6/24.9 s，
        # 再撑起身体还要几秒。25 s 的旧值正好卡在翻正完成的那一拍上，两跑
        # 都在机体已经翻正、正在起身时被判超时。
        self.declare_parameter('p5_fall_recovery_stand_timeout_s', 45.0)
        # 整条梯子的总预算：解楔梯子每个动作只放行一次，所以站立超时本身
        # 已经有界；这一项是防止阶段状态在极端情况下无限期停在扶起里。
        self.declare_parameter('p5_fall_recovery_total_timeout_s', 90.0)
        # 注意：扶起之后**不重试**，一律进故障保持。曾经按"相对转角出口方向的
        # 横向位移"判断机体是否还在赛道上并重试该转角，2026-08-07 实测证明这个
        # 判据不成立：翻滚 30 s 之后腿式里程计的位置状态已经没有意义，两次扶起
        # 真实位置都落在赛道内缘外 0.29 m（x=2.60，赛道 x∈[2.88,3.37]），
        # 里程计却都只报 0.024/0.025 m。其中一次因此在地面上跑完剩余航段，
        # 报出了假完赛（终点 (1.51,13.23)，终点框是 2.34~2.51 x 13.60~13.75）。
        # 假完赛比故障更糟：它会向 mission control 上报 stage_complete。
        # 没有任何可信信号能回答"扶起来之后我还在赛道上吗"——深度观测器在环形
        # 边上是盲的，黄线在无黄线档里不用——所以这条路直接关掉。
        # 非 STOP 动作完成后继续保持该动作一段时间，让跳跃/转身落地稳定。
        # 该窗口独立于 ACK 防陈旧延时，不能用放大 min_ack_delay 代替。
        self.declare_parameter('p5_action_post_complete_hold_s', 3.0)
        # Hold feedback must remain current and healthy; otherwise fail closed.
        self.declare_parameter('p5_action_feedback_max_age_s', 0.50)
        # /clock 冻结时，普通定时运动也必须在 wall-monotonic 硬期限内停住。
        self.declare_parameter('p5_timed_motion_timeout_factor', 3.0)
        self.declare_parameter('p5_timed_motion_timeout_margin_s', 2.0)
        self.declare_parameter('p5_forward_after_reset_body_timeout_s', 20.0)

        # D3 仿真 D435 只读 smoke 路径。G0 前只使用 CameraInfo + IMU，不进入控制闭环。
        self.declare_parameter('p5_bridge_observer_enabled', True)
        self.declare_parameter('p5_bridge_observer_period_s', 0.20)
        self.declare_parameter(
            'p5_depth_camera_info_topic',
            '/d435/depth/d435_depth/camera_info',
        )
        self.declare_parameter('p5_imu_topic', '/imu')
        self.declare_parameter('p5_depth_horizontal_fov', 1.22)
        self.declare_parameter('p5_depth_camera_mount_roll', 0.0)
        self.declare_parameter('p5_depth_camera_mount_pitch', 0.0)
        # 仿真 D435 相对 body 原点；实体 G0 必须用实测外参覆盖。
        self.declare_parameter('p5_depth_camera_mount_x', 0.271994)
        self.declare_parameter('p5_depth_camera_mount_y', 0.025)
        self.declare_parameter('p5_depth_camera_mount_z', 0.114912)
        self.declare_parameter('p5_bridge_max_imu_age_s', 0.50)
        self.declare_parameter('p5_bridge_max_imu_depth_skew_s', 0.10)

        # 深度居中闭环：把观测器的 lateral_offset / heading_error 变成有界的
        # vy / wz 修正。里程只给沿路进度，给不出相对桥中心线的横向偏差，
        # 这条回路是唯一能闭合它的东西。默认关闭：实体启用前必须过 G0/G1。
        self.declare_parameter('p5_deck_lateral_enabled', False)
        self.declare_parameter('p5_deck_lateral_k_vy', 0.40)
        self.declare_parameter('p5_deck_lateral_k_wz', 0.80)
        self.declare_parameter('p5_deck_lateral_deadband_m', 0.02)
        self.declare_parameter('p5_deck_lateral_heading_deadband_rad', 0.02)
        self.declare_parameter('p5_deck_lateral_max_vy', 0.12)
        self.declare_parameter('p5_deck_lateral_max_wz', 0.25)
        self.declare_parameter('p5_deck_lateral_max_age_s', 1.0)
        self.declare_parameter('p5_deck_lateral_min_consecutive_valid', 2)
        self.declare_parameter('p5_deck_lateral_max_plausible_offset_m', 0.40)
        self.declare_parameter('p5_deck_lateral_blind_vx_scale', 0.5)
        # 允许深度观测参与横向闭环的状态白名单。空列表 = 全部状态（旧行为）。
        #
        # 为什么需要（2026-08-05 两批 24 跑真值复核，计划书第 34 条）：
        # 观测器的准确度是**分段**的，而不是全局的。以真值为基准逐帧比对——
        #   爬坡（2_bridge，平桥面、两侧真跌落）  n=184  相关 0.99  rms 0.005 m
        #   一号边                                n=10   相关 0.99  rms 0.012 m
        #   二号边                                n=7    相关 0.99  rms 0.009 m
        #   三号边（3_slope，横向倾 11.7 度）     n=146  相关 0.18  rms 0.136 m
        # 三号边上观测器 40% 的帧“有效”，但那些值与真实偏差几乎不相关，
        # 而且带 +0.12 m 的固定偏置，方向恰好指向赛道外侧。
        # 因果检验同样干净：三号边上按当时选中的观测源分段统计真实漂移——
        # 深度在手时 +0.19~+0.25 m/m，里程兜底时 +0.02~+0.04 m/m，两批一致、
        # 每一跑都一致。三号边的全部失败（2 次走出赛道、3 次假完赛、2 次带
        # 着入口偏差进跳跃段）都出在这里。
        #
        # 几何原因在网格里：环形边是 0.49 m 宽、横向倾 11.7 度的斜面，两侧
        # 各紧贴一条与相邻边等高的黄色路缘（3_border），所以既没有平面桥面，
        # 也没有两侧跌落——双边提取器的两条前提都不成立。爬坡段两条都成立。
        #
        # 逗号分隔而不是字符串数组：Galactic 下声明空数组参数推不出类型，
        # 而这个白名单的默认值必须是“空 = 不限制”，否则默认档就被悄悄改了。
        self.declare_parameter('p5_deck_lateral_depth_states', '')
        # 横向偏差过大时先归位再前进（0 表示关闭，实体默认关闭）。
        self.declare_parameter('p5_deck_lateral_centre_first_offset_m', 0.0)
        self.declare_parameter('p5_deck_lateral_centre_first_release_m', 0.03)
        self.declare_parameter('p5_deck_lateral_centre_first_vx_scale', 0.0)
        # 允许“先归位再前进”降低 vx 的状态白名单（逗号分隔，空 = 不限制）。
        # 归位保持是按环形边标定的：那里半宽 0.245 m 而实测 0.11 m 就掉下去。
        # 爬坡段不是这样——桥面 0.504 m 宽、两侧各 0.252 m，0.08 m 的偏差
        # 完全在横向回路自己能修的范围内，可是 vx 被压到 0.180 m·s⁻¹ 之后
        # 机体上不了入口台阶：2026-08-16 十二跑里两跑就卡在坡底 y=8.05、
        # z 掉到 0.13，然后 40 s 一动不动。爬坡要的是动量，不是站住横移。
        self.declare_parameter('p5_deck_lateral_centre_first_states', '')
        # 进入一段之后前多少米不参与横向/航向修正（0 = 关闭，行为不变）。
        #
        # **实测否决，默认必须保持 0。** 2026-08-16 十六跑：设成 0.5 之后
        # 爬坡超时 2/11 -> 7/16、一号边通过 10/12 -> 6/16、完赛 2/11 -> 1/16。
        # 这个门按里程算，于是和它要解决的问题构成死锁——机体走不动正是因为
        # 没有修正，而门要等它走够 0.5 m 才放行：
        #   odometry progress 0.06/3.72 m, cmd=(0.450,0.000,0.000), entry-hold 0.06/0.50 m
        # 保留这个参数是为了让这条实测结论有地方写，不是为了给人打开。
        self.declare_parameter('p5_deck_lateral_engage_after_m', 0.0)
        # 横向积分项：把每段固定的 *_vy 开环横移量变成自校准量。
        # 倾斜面上的扰动是常量，纯 P 回路对常量扰动只能收敛到一个常值误差，
        # 这个误差在 race_physical 的环形边上实测约 0.04 m/m（见 deck_lateral.py）。
        # 0 表示关闭（默认，任何已有档位行为不变）。
        self.declare_parameter('p5_deck_lateral_k_i_vy', 0.0)
        self.declare_parameter('p5_deck_lateral_max_i_vy', 0.06)
        # 深度观测不可用时，用里程相对段入口基准线兜底。
        self.declare_parameter('p5_route_lateral_fallback', False)
        self.declare_parameter('p5_route_lateral_heading_only', False)
        self.declare_parameter('p5_route_lateral_snap_heading', True)
        # 路线航向栅格的基准方向（度）。NaN = 用第一段入口实测偏航兜底。
        #
        # 为什么必须能声明：栅格原来锚在“机体在第一段入口时的实测偏航”上。
        # 那个值每跑都不一样——2026-08-04 两批 20 跑里从 +1.11 到 +1.83 rad
        # （真值赛道方向是 +pi/2），完全取决于开局站姿怎么落定。栅格错 e 弧度，
        # 里程兜底基准线就跟着错 e：爬坡 3.4 m 之后凭空造出 3.4*sin(e) 的
        # 横向偏差（e=0.057 rad -> 0.19 m），已经接近半个桥宽 0.245 m，
        # 而 deck_lateral 会忠实地把机体横挪那么多——挪下桥。
        # 实测分界干净：栅格误差 <= 0.03 rad 的 8 跑全部爬上桥（0 失败），
        # > 0.03 rad 的 12 跑里 9 跑丢在爬坡段。见计划书第 29 条。
        #
        # 声明值同时让航向环变得有意义：机体真的歪 15 度时，误差是可见的
        # 0.26 rad，环会把它拧回赛道方向，而不是把 15 度歪当成“正确航向”。
        self.declare_parameter('p5_route_base_yaw_deg', float('nan'))
        # 上桥前的起始航向对齐（计划书第 30 条）。默认关闭：它只有在
        # p5_route_base_yaw_deg 已标定时才有意义，实体标定前打开等于按一个
        # 无意义的方向转机体。没有声明航向或里程不可用时本状态自动跳过。
        #
        # 为什么需要：实测 P5_RECOVERY_STAND 会把机体转歪 25~31 度
        # （2026-08-05 批 r07 +31.3、r12 +25.0，而站起前的起始位姿是干净的），
        # 之后到上桥之间没有任何环节把它转回来。25 度航向误差在 3.4 m 爬坡上
        # 就是 1.4 m 的横向位移——桥面半宽只有 0.245 m。
        self.declare_parameter('p5_start_align_enabled', False)
        self.declare_parameter('p5_start_align_tol_deg', 4.0)
        self.declare_parameter('p5_start_align_wz', 0.30)
        self.declare_parameter('p5_start_align_step_height', 0.04)
        self.declare_parameter('p5_start_align_timeout_s', 12.0)
        # 用最近一次深度定位锚定里程基准线，而不是锁死段入口位姿。
        # 默认关闭：锚定在概念上是对的（入口基准线会掩盖入段偏差），
        # 但 2026-08-04 实测没有改善完成率（见计划书第 20 条），
        # 所以默认保持与实测最好的那一档一致，需要时再显式打开。
        self.declare_parameter('p5_route_lateral_anchor_enabled', False)
        # 允许被 p5_deck_lateral_depth_states 压掉转向权的深度帧仍然参与锚定。
        # 压制的理由是“这个面上的深度不能直接转向”，不是“这个面上的深度是假的”：
        # race_physical 三跑实测，一号边入段那一帧深度报 -0.088/-0.069/-0.063，
        # 真值 -0.088/-0.066/-0.060 —— 误差 5 mm 以内，却被整帧丢弃，
        # 于是里程基准线锁在入口位姿上，把入段偏差当成零误差保持到走出赛道。
        # 默认关闭：在 race.world 上同一帧带 +0.126 m 偏置（计划书第 34 条），
        # 拿它锚定会把基准线拉歪，所以必须逐档显式打开。
        self.declare_parameter('p5_route_lateral_anchor_suppressed_depth', False)
        # 锚定量的合理上限（米）。默认沿用 route_lateral 的 0.60 m，即不额外收紧。
        # 环形边半宽只有 0.245 m，而机体侧倾 0.48 rad 时实测约 0.11 m 就开始
        # 掉下去，所以入段时报出 0.20 m 的深度定位一定是拟合坏了而不是姿态：
        # 2026-08-16 实测 r02 在三号边入段被一帧 -0.206 m 锚死，整段照着它走。
        self.declare_parameter('p5_route_lateral_anchor_max_m', 0.60)
        # 深度观测器单侧边缘兜底：环形赛道每条边外侧是抬起路缘而不是跌落，
        # 双边提取器在那里永远拿不到证据。宽度是已声明的赛道属性。
        self.declare_parameter('p5_bridge_single_sided_edges_enabled', False)
        self.declare_parameter('p5_bridge_declared_deck_width', 0.0)
        # 段进度用路径长度还是沿路线投影（path / along_track）。
        self.declare_parameter('p5_route_progress_measure', 'path')
        # 入段深度完整性门：爬坡段结束时深度观测器的有效帧数必须达标，
        # 否则机体不在赛道上（两次假完赛都是开局摔下桥后在地面走完全程）。
        # min_frames=0 关闭。
        self.declare_parameter('p5_route_entry_depth_segment', 'up_slope')
        self.declare_parameter('p5_route_entry_depth_min_frames', 0)
        # 段内横向偏离上限（0 = 关闭）：入段深度门只看爬坡段，走下**环形边**
        # 的机体照样满足每一个距离窗口和转角航向校核，然后在地面上走到
        # P5_DONE。2026-08-05 批六次“完赛”里三次是这样，终点落在三号边外缘
        # 之外 0.3~0.6 m 的地面上。批后审计能查出来，运行时没有任何东西拦它。
        #
        # 度量是相对段入口基准线的里程横向偏移，所以深度致盲时依然可用。
        # 它的盲区必须说清楚：基准线锚在段入口，带着偏差入段的机体在这里读
        # 数是零。它拦的是**段内漂移**——三号边正是这一类。
        self.declare_parameter('p5_route_max_cross_track_m', 0.0)
        self.declare_parameter('p5_route_cross_track_samples', 25)
        # 翻倒闩锁：机体侧倾/俯仰超过这个角度并持续若干采样就判定离开赛道。
        # 0 = 关闭（默认，行为不变）。计划书第 37 条：翻滚会毁掉腿式里程计的
        # 位置状态而不影响姿态，所以这里量的是“翻过去了”而不是“翻到哪儿”。
        # 停滞检测（计划书第 11 条，长期未做）：命令在走、里程不动。
        # 实测机体在桥面入口台阶前塌成劈叉（真值 z 从 0.30 掉到 0.13），
        # 命令 vx=0.45 而里程钉在 0.20 m，一直耗到 45 s 状态超时，十一跑三跑。
        # 判据必须是“命令速度 + 里程不动”这一对：原地转向时 vx 本来就是 0，
        # 只看计时器会把正常的转向也判成停滞。0 = 关闭（默认，行为不变）。
        self.declare_parameter('p5_route_stall_min_speed', 0.05)
        self.declare_parameter('p5_route_stall_timeout_s', 0.0)
        self.declare_parameter('p5_route_stall_min_progress_m', 0.05)
        self.declare_parameter('p5_route_stall_max_attempts', 1)
        self.declare_parameter('p5_route_topple_limit_rad', 0.0)
        self.declare_parameter('p5_route_topple_samples', 25)

        # 可选角度修正
        self.declare_parameter('p5_yellow_angle_align_enabled', True)
        self.declare_parameter('p5_yellow_angle_align_fixed_wz', 0.15)
        self.declare_parameter('p5_yellow_angle_align_deadband_deg', 0.5)

        # 路线模型（计划 §4.1/§4.2）。里程来自 LCM state_estimator，
        # 仿真与实体同源，不依赖只有仿真才有的 cyberdog_visual TF。
        self.declare_parameter('p5_route_model_enabled', True)
        # enforce：受控段的视觉触发必须落在里程窗口内（双证据）；
        # monitor：只观测记录窗口，不改变任何状态转移（用于标定窗口）。
        self.declare_parameter('p5_route_model_mode', 'enforce')
        # 超出窗口上限：fault = 停住报故障；degraded_advance = 声明式有界降级推进。
        self.declare_parameter('p5_route_overrun_action', 'fault')
        self.declare_parameter('p5_route_odometry_required', True)
        self.declare_parameter('p5_route_odom_max_age_s', 0.50)
        # 50 Hz 下正常步进约 1 cm；超过这个单步位移视为估计器跳变/跳跃不计入里程。
        self.declare_parameter('p5_route_odom_max_step_m', 0.20)
        self.declare_parameter('p5_route_speed_cap_enabled', True)
        self.declare_parameter('p5_route_turn_verify_enabled', True)
        self.declare_parameter('p5_route_realign_wz', 0.30)
        self.declare_parameter('p5_route_realign_step_height', 0.04)
        self.declare_parameter('p5_route_realign_timeout_s', 6.0)
        self.declare_parameter('p5_route_realign_max_attempts', 1)

        # 段尾证据来源：vision = 视觉触发 + 里程窗口双证据（默认）；
        # odometry = 只按里程走完声明长度（单证据，等价于 fallback 第 3 档，
        # 由 max_m 与状态超时封顶）。用于不依赖黄线的里程主导模式。
        self.declare_parameter('p5_route_exit_source', EXIT_VISION)
        # 黄线横向/航向闭环修正总开关。里程主导模式下关闭：黄线不再参与控制，
        # 只保留各段标定好的固定 vy/wz 基准值。
        self.declare_parameter('p5_yellow_lateral_correction_enabled', True)

        # 每段窗口/期望转角/限速都可按参数档覆盖；默认值来自图纸推导的段表，
        # 仿真档在 config/stage5_sim.yaml 里显式覆盖（不让仿真值成为默认值）。
        for segment in RouteModel().segments:
            prefix = 'p5_route_{}'.format(segment.name)
            self.declare_parameter(prefix + '_expected_m', segment.expected_m)
            self.declare_parameter(prefix + '_min_m', segment.min_m)
            self.declare_parameter(prefix + '_max_m', segment.max_m)
            self.declare_parameter(prefix + '_expected_yaw_deg', segment.expected_yaw_deg)
            self.declare_parameter(prefix + '_yaw_tol_deg', segment.yaw_tol_deg)
            self.declare_parameter(prefix + '_speed_cap_mps', segment.speed_cap_mps)

    # ============================================================
    # 参数读取
    # ============================================================
    def p5_load_params(self):
        gp = self.get_parameter

        self.p5_initial_state = self.resolve_stage_entry(
            self.p5_entry_table(), str(gp('p5_initial_state').value))
        self.control_period_s = float(gp('p5_control_period_s').value)

        self.p5_show_debug_vis = bool(gp('p5_show_debug_vis').value)
        self.p5_debug_vis_detail_level = int(gp('p5_debug_vis_detail_level').value)
        self.p5_show_yellow_mask = bool(gp('p5_show_yellow_mask').value)

        self.p5_body_normal_roll = float(gp('p5_body_normal_roll').value)
        self.p5_body_normal_height = float(gp('p5_body_normal_height').value)
        self.p5_body_normal_wait_s = float(gp('p5_body_normal_wait_s').value)

        self.p5_step_up_vx = float(gp('p5_step_up_vx').value)
        self.p5_step_up_vy = float(gp('p5_step_up_vy').value)
        self.p5_step_up_wz = float(gp('p5_step_up_wz').value)
        self.p5_step_up_step_height = float(gp('p5_step_up_step_height').value)
        self.p5_step_up_duration_s = float(gp('p5_step_up_duration_s').value)

        self.p5_up_slope_vx = float(gp('p5_up_slope_vx').value)
        self.p5_up_slope_vy = float(gp('p5_up_slope_vy').value)
        self.p5_up_slope_wz = float(gp('p5_up_slope_wz').value)
        self.p5_up_slope_step_height = float(gp('p5_up_slope_step_height').value)
        self.p5_up_slope_roll = float(gp('p5_up_slope_roll').value)
        self.p5_up_slope_pitch = float(gp('p5_up_slope_pitch').value)

        self.p5_after_up_slope_forward_duration_s = float(
            gp('p5_after_up_slope_forward_duration_s').value
        )
        self.p5_after_up_slope_forward_vx = float(
            gp('p5_after_up_slope_forward_vx').value
        )
        self.p5_after_up_slope_forward_vy = float(
            gp('p5_after_up_slope_forward_vy').value
        )
        self.p5_after_up_slope_forward_wz = float(
            gp('p5_after_up_slope_forward_wz').value
        )
        self.p5_after_up_slope_forward_step_height = float(
            gp('p5_after_up_slope_forward_step_height').value
        )

        self.p5_inner_edge_align_enabled = bool(gp('p5_inner_edge_align_enabled').value)
        self.p5_inner_edge_enable_vy = bool(gp('p5_inner_edge_enable_vy').value)
        self.p5_inner_edge_enable_wz = bool(gp('p5_inner_edge_enable_wz').value)

        self.p5_inner_edge_left_roi_x_min = float(gp('p5_inner_edge_left_roi_x_min').value)
        self.p5_inner_edge_left_roi_x_max = float(gp('p5_inner_edge_left_roi_x_max').value)
        self.p5_inner_edge_right_roi_x_min = float(gp('p5_inner_edge_right_roi_x_min').value)
        self.p5_inner_edge_right_roi_x_max = float(gp('p5_inner_edge_right_roi_x_max').value)
        self.p5_inner_edge_roi_y_min = float(gp('p5_inner_edge_roi_y_min').value)
        self.p5_inner_edge_roi_y_max = float(gp('p5_inner_edge_roi_y_max').value)

        self.p5_inner_edge_min_points = int(gp('p5_inner_edge_min_points').value)
        self.p5_inner_edge_min_y_span = float(gp('p5_inner_edge_min_y_span').value)
        self.p5_inner_edge_bottom_min_ratio = float(gp('p5_inner_edge_bottom_min_ratio').value)
        self.p5_inner_edge_x_std_max = float(gp('p5_inner_edge_x_std_max').value)
        self.p5_inner_edge_use_bottom_connected_segment = bool(
            gp('p5_inner_edge_use_bottom_connected_segment').value
        )
        self.p5_inner_edge_max_y_gap = int(gp('p5_inner_edge_max_y_gap').value)
        self.p5_inner_edge_min_common_y_span = float(gp('p5_inner_edge_min_common_y_span').value)
        self.p5_inner_edge_row_step = int(gp('p5_inner_edge_row_step').value)
        self.p5_inner_edge_top_bottom_band_ratio = float(
            gp('p5_inner_edge_top_bottom_band_ratio').value
        )

        self.p5_inner_edge_center_k_vy = float(gp('p5_inner_edge_center_k_vy').value)
        self.p5_inner_edge_heading_k_wz = float(gp('p5_inner_edge_heading_k_wz').value)
        self.p5_inner_edge_vy_max_correction = abs(
            float(gp('p5_inner_edge_vy_max_correction').value)
        )
        self.p5_inner_edge_wz_max_correction = abs(
            float(gp('p5_inner_edge_wz_max_correction').value)
        )
        self.p5_inner_edge_center_deadband_px = abs(
            float(gp('p5_inner_edge_center_deadband_px').value)
        )
        self.p5_inner_edge_heading_deadband_px = abs(
            float(gp('p5_inner_edge_heading_deadband_px').value)
        )

        self.p5_right_side_yellow_roi_x_min = float(gp('p5_right_side_yellow_roi_x_min').value)
        self.p5_right_side_yellow_roi_x_max = float(gp('p5_right_side_yellow_roi_x_max').value)
        self.p5_right_side_yellow_roi_y_min = float(gp('p5_right_side_yellow_roi_y_min').value)
        self.p5_right_side_yellow_roi_y_max = float(gp('p5_right_side_yellow_roi_y_max').value)

        self.p5_right_side_yellow_min_area = float(gp('p5_right_side_yellow_min_area').value)
        self.p5_right_side_yellow_min_height = int(gp('p5_right_side_yellow_min_height').value)
        self.p5_right_side_yellow_min_width = int(gp('p5_right_side_yellow_min_width').value)
        self.p5_right_side_yellow_bottom_valid_ratio = float(
            gp('p5_right_side_yellow_bottom_valid_ratio').value
        )

        self.p5_right_side_yellow_lost_confirm_count = int(
            gp('p5_right_side_yellow_lost_confirm_count').value
        )
        self.p5_right_side_yellow_ignore_after_enter_s = float(
            gp('p5_right_side_yellow_ignore_after_enter_s').value
        )

        self.p5_after_up_slope_control_duration_s = float(
            gp('p5_after_up_slope_control_duration_s').value
        )
        self.p5_after_up_slope_turn_method = str(
            gp('p5_after_up_slope_turn_method').value
        ).strip().lower()
        self.p5_after_up_slope_turn_jump_mode = int(
            gp('p5_after_up_slope_turn_jump_mode').value
        )
        self.p5_after_up_slope_turn_jump_gait = int(
            gp('p5_after_up_slope_turn_jump_gait').value
        )
        if self.p5_after_up_slope_turn_method not in ['velocity', 'right_jump']:
            self.get_logger().warn(
                f'[P5_PARAM] unknown p5_after_up_slope_turn_method='
                f'{self.p5_after_up_slope_turn_method}, fallback to velocity'
            )
            self.p5_after_up_slope_turn_method = 'velocity'
        self.p5_after_up_slope_control_vx = float(
            gp('p5_after_up_slope_control_vx').value
        )
        self.p5_after_up_slope_control_vy = float(
            gp('p5_after_up_slope_control_vy').value
        )
        self.p5_after_up_slope_control_wz = float(
            gp('p5_after_up_slope_control_wz').value
        )
        self.p5_after_up_slope_control_step_height = float(
            gp('p5_after_up_slope_control_step_height').value
        )

        self.p5_right_slope_roll = float(gp('p5_right_slope_roll').value)
        self.p5_right_slope_height = float(gp('p5_right_slope_height').value)
        self.p5_right_slope_body_wait_s = float(gp('p5_right_slope_body_wait_s').value)

        self.p5_right_slope_right_edge_vy_adjust_enabled = bool(
            gp('p5_right_slope_right_edge_vy_adjust_enabled').value
        )
        self.p5_right_slope_right_edge_roi_x_min = float(
            gp('p5_right_slope_right_edge_roi_x_min').value
        )
        self.p5_right_slope_right_edge_roi_x_max = float(
            gp('p5_right_slope_right_edge_roi_x_max').value
        )
        self.p5_right_slope_right_edge_roi_y_min = float(
            gp('p5_right_slope_right_edge_roi_y_min').value
        )
        self.p5_right_slope_right_edge_roi_y_max = float(
            gp('p5_right_slope_right_edge_roi_y_max').value
        )
        self.p5_right_slope_right_edge_row_step = int(
            gp('p5_right_slope_right_edge_row_step').value
        )
        self.p5_right_slope_right_edge_use_bottom_connected_segment = bool(
            gp('p5_right_slope_right_edge_use_bottom_connected_segment').value
        )
        self.p5_right_slope_right_edge_max_y_gap = int(
            gp('p5_right_slope_right_edge_max_y_gap').value
        )
        self.p5_right_slope_right_edge_min_points = int(
            gp('p5_right_slope_right_edge_min_points').value
        )
        self.p5_right_slope_right_edge_min_y_span = float(
            gp('p5_right_slope_right_edge_min_y_span').value
        )
        self.p5_right_slope_right_edge_x_std_max = float(
            gp('p5_right_slope_right_edge_x_std_max').value
        )
        self.p5_right_slope_right_edge_bottom_min_ratio = float(
            gp('p5_right_slope_right_edge_bottom_min_ratio').value
        )
        self.p5_right_slope_right_edge_bottom_band_ratio = float(
            gp('p5_right_slope_right_edge_bottom_band_ratio').value
        )
        self.p5_right_slope_right_edge_bottom_band_ratio = max(
            0.05, min(0.80, self.p5_right_slope_right_edge_bottom_band_ratio)
        )
        self.p5_right_slope_right_too_center_ratio = float(
            gp('p5_right_slope_right_too_center_ratio').value
        )
        self.p5_right_slope_right_too_right_ratio = float(
            gp('p5_right_slope_right_too_right_ratio').value
        )
        self.p5_right_slope_right_too_center_add_vy = abs(float(
            gp('p5_right_slope_right_too_center_add_vy').value
        ))
        self.p5_right_slope_right_too_right_reduce_vy = abs(float(
            gp('p5_right_slope_right_too_right_reduce_vy').value
        ))

        self.p5_right_slope_lost_extra_enabled = bool(
            gp('p5_right_slope_lost_extra_enabled').value
        )
        self.p5_right_slope_lost_extra_ignore_after_enter_s = max(
            0.0,
            float(gp('p5_right_slope_lost_extra_ignore_after_enter_s').value)
        )
        self.p5_right_slope_lost_extra_confirm_count = max(
            1,
            int(gp('p5_right_slope_lost_extra_confirm_count').value)
        )

        # 这里不要 abs，因为这两个参数需要允许正负号，方便直接调方向。
        self.p5_right_slope_lost_extra_too_center_vy = float(
            gp('p5_right_slope_lost_extra_too_center_vy').value
        )
        self.p5_right_slope_lost_extra_too_right_vy = float(
            gp('p5_right_slope_lost_extra_too_right_vy').value
        )


        self.p5_right_slope_1_vx = float(gp('p5_right_slope_1_vx').value)
        self.p5_right_slope_1_vy = float(gp('p5_right_slope_1_vy').value)
        self.p5_right_slope_1_wz = float(gp('p5_right_slope_1_wz').value)
        self.p5_right_slope_1_step_height = float(gp('p5_right_slope_1_step_height').value)
        self.p5_right_slope_1_after_center_lost_duration_s = float(
            gp('p5_right_slope_1_after_center_lost_duration_s').value
        )
        self.p5_right_slope_1_after_center_lost_vx = float(
            gp('p5_right_slope_1_after_center_lost_vx').value
        )
        self.p5_right_slope_1_after_center_lost_vy = float(
            gp('p5_right_slope_1_after_center_lost_vy').value
        )
        self.p5_right_slope_1_after_center_lost_wz = float(
            gp('p5_right_slope_1_after_center_lost_wz').value
        )
        self.p5_right_slope_1_after_center_lost_step_height = float(
            gp('p5_right_slope_1_after_center_lost_step_height').value
        )

        self.p5_right_slope_turn_method = str(
            gp('p5_right_slope_turn_method').value
        ).strip().lower()
        self.p5_right_slope_turn_1_jump_mode = int(
            gp('p5_right_slope_turn_1_jump_mode').value
        )
        self.p5_right_slope_turn_1_jump_gait = int(
            gp('p5_right_slope_turn_1_jump_gait').value
        )
        self.p5_right_slope_turn_2_jump_mode = int(
            gp('p5_right_slope_turn_2_jump_mode').value
        )
        self.p5_right_slope_turn_2_jump_gait = int(
            gp('p5_right_slope_turn_2_jump_gait').value
        )
        self.p5_right_slope_turn_jump_stop_after_finish = bool(
            gp('p5_right_slope_turn_jump_stop_after_finish').value
        )

        if self.p5_right_slope_turn_method not in ['velocity', 'right_jump']:
            self.get_logger().warn(
                f'[P5_PARAM] unknown p5_right_slope_turn_method='
                f'{self.p5_right_slope_turn_method}, fallback to velocity'
            )
            self.p5_right_slope_turn_method = 'velocity'

        self.p5_turn_1_duration_s = float(gp('p5_turn_1_duration_s').value)
        self.p5_turn_1_vx = float(gp('p5_turn_1_vx').value)
        self.p5_turn_1_vy = float(gp('p5_turn_1_vy').value)
        self.p5_turn_1_wz = float(gp('p5_turn_1_wz').value)
        self.p5_turn_1_step_height = float(gp('p5_turn_1_step_height').value)

        self.p5_right_slope_2_vx = float(gp('p5_right_slope_2_vx').value)
        self.p5_right_slope_2_entry_recenter_duration_s = max(
            0.0, float(gp('p5_right_slope_2_entry_recenter_duration_s').value)
        )
        self.p5_right_slope_2_entry_recenter_vy = float(
            gp('p5_right_slope_2_entry_recenter_vy').value
        )
        self.p5_right_slope_2_vy = float(gp('p5_right_slope_2_vy').value)
        self.p5_right_slope_2_right_edge_adjust_enabled = bool(
            gp('p5_right_slope_2_right_edge_adjust_enabled').value
        )
        self.p5_right_slope_2_wz = float(gp('p5_right_slope_2_wz').value)
        self.p5_right_slope_2_step_height = float(gp('p5_right_slope_2_step_height').value)
        self.p5_right_slope_2_after_center_lost_duration_s = float(
            gp('p5_right_slope_2_after_center_lost_duration_s').value
        )
        self.p5_right_slope_2_after_center_lost_vx = float(
            gp('p5_right_slope_2_after_center_lost_vx').value
        )
        self.p5_right_slope_2_after_center_lost_vy = float(
            gp('p5_right_slope_2_after_center_lost_vy').value
        )
        self.p5_right_slope_2_after_center_lost_wz = float(
            gp('p5_right_slope_2_after_center_lost_wz').value
        )
        self.p5_right_slope_2_after_center_lost_step_height = float(
            gp('p5_right_slope_2_after_center_lost_step_height').value
        )

        self.p5_right_slope_3_after_center_lost_duration_s = float(
            gp('p5_right_slope_3_after_center_lost_duration_s').value
        )
        self.p5_right_slope_3_after_center_lost_vx = float(
            gp('p5_right_slope_3_after_center_lost_vx').value
        )
        self.p5_right_slope_3_after_center_lost_vy = float(
            gp('p5_right_slope_3_after_center_lost_vy').value
        )
        self.p5_right_slope_3_after_center_lost_wz = float(
            gp('p5_right_slope_3_after_center_lost_wz').value
        )
        self.p5_right_slope_3_after_center_lost_step_height = float(
            gp('p5_right_slope_3_after_center_lost_step_height').value
        )

        self.p5_turn_2_duration_s = float(gp('p5_turn_2_duration_s').value)
        self.p5_turn_2_vx = float(gp('p5_turn_2_vx').value)
        self.p5_turn_2_vy = float(gp('p5_turn_2_vy').value)
        self.p5_turn_2_wz = float(gp('p5_turn_2_wz').value)
        self.p5_turn_2_step_height = float(gp('p5_turn_2_step_height').value)

        self.p5_right_slope_3_vx = float(gp('p5_right_slope_3_vx').value)
        self.p5_right_slope_3_vy = float(gp('p5_right_slope_3_vy').value)
        self.p5_right_slope_3_wz = float(gp('p5_right_slope_3_wz').value)
        self.p5_right_slope_3_step_height = float(gp('p5_right_slope_3_step_height').value)

        self.p5_right_shift_before_right_jump_duration_s = float(
            gp('p5_right_shift_before_right_jump_duration_s').value
        )
        self.p5_right_shift_before_right_jump_vx = float(
            gp('p5_right_shift_before_right_jump_vx').value
        )
        self.p5_right_shift_before_right_jump_vy = float(
            gp('p5_right_shift_before_right_jump_vy').value
        )
        self.p5_right_shift_before_right_jump_wz = float(
            gp('p5_right_shift_before_right_jump_wz').value
        )
        self.p5_right_shift_before_right_jump_step_height = float(
            gp('p5_right_shift_before_right_jump_step_height').value
        )

        self.p5_right_shift_before_right_jump_2_duration_s = float(
            gp('p5_right_shift_before_right_jump_2_duration_s').value
        )
        self.p5_right_shift_before_right_jump_2_vx = float(
            gp('p5_right_shift_before_right_jump_2_vx').value
        )
        self.p5_right_shift_before_right_jump_2_vy = float(
            gp('p5_right_shift_before_right_jump_2_vy').value
        )
        self.p5_right_shift_before_right_jump_2_wz = float(
            gp('p5_right_shift_before_right_jump_2_wz').value
        )
        self.p5_right_shift_before_right_jump_2_step_height = float(
            gp('p5_right_shift_before_right_jump_2_step_height').value
        )

        self.p5_right_jump_after_reset_body_mode = int(
            gp('p5_right_jump_after_reset_body_mode').value
        )
        self.p5_right_jump_after_reset_body_gait = int(
            gp('p5_right_jump_after_reset_body_gait').value
        )

        self.p5_align_after_right_jump_duration_s = float(
            gp('p5_align_after_right_jump_duration_s').value
        )
        self.p5_align_after_right_jump_vx = float(
            gp('p5_align_after_right_jump_vx').value
        )
        self.p5_align_after_right_jump_vy = float(
            gp('p5_align_after_right_jump_vy').value
        )
        self.p5_align_after_right_jump_wz = float(
            gp('p5_align_after_right_jump_wz').value
        )
        self.p5_align_after_right_jump_step_height = float(
            gp('p5_align_after_right_jump_step_height').value
        )

        self.p5_forward_after_reset_body_duration_s = float(
            gp('p5_forward_after_reset_body_duration_s').value
        )
        self.p5_forward_after_reset_body_vx = float(
            gp('p5_forward_after_reset_body_vx').value
        )
        self.p5_forward_after_reset_body_vy = float(
            gp('p5_forward_after_reset_body_vy').value
        )
        self.p5_forward_after_reset_body_wz = float(
            gp('p5_forward_after_reset_body_wz').value
        )
        self.p5_forward_after_reset_body_step_height = float(
            gp('p5_forward_after_reset_body_step_height').value
        )
        self.p5_forward_after_reset_body_hold_align_enabled = bool(
            gp('p5_forward_after_reset_body_hold_align_enabled').value
        )
        self.p5_forward_after_reset_body_align_center_done_px = abs(float(
            gp('p5_forward_after_reset_body_align_center_done_px').value
        ))
        self.p5_forward_after_reset_body_align_heading_done_px = abs(float(
            gp('p5_forward_after_reset_body_align_heading_done_px').value
        ))
        self.p5_forward_after_reset_body_align_stable_frames = int(
            gp('p5_forward_after_reset_body_align_stable_frames').value
        )
        self.p5_forward_after_reset_body_align_lost_confirm_frames = max(
            1,
            int(gp('p5_forward_after_reset_body_align_lost_confirm_frames').value)
        )
        self.p5_forward_after_reset_body_align_max_extra_s = float(
            gp('p5_forward_after_reset_body_align_max_extra_s').value
        )

        self.p5_forward_no_align_after_reset_body_duration_s = float(
            gp('p5_forward_no_align_after_reset_body_duration_s').value
        )
        self.p5_forward_no_align_after_reset_body_vx = float(
            gp('p5_forward_no_align_after_reset_body_vx').value
        )
        self.p5_forward_no_align_after_reset_body_vy = float(
            gp('p5_forward_no_align_after_reset_body_vy').value
        )
        self.p5_forward_no_align_after_reset_body_wz = float(
            gp('p5_forward_no_align_after_reset_body_wz').value
        )
        self.p5_forward_no_align_after_reset_body_step_height = float(
            gp('p5_forward_no_align_after_reset_body_step_height').value
        )

        self.p5_jump_exit_slope_mode = int(gp('p5_jump_exit_slope_mode').value)
        self.p5_jump_exit_slope_gait = int(gp('p5_jump_exit_slope_gait').value)

        self.p5_final_long_jump_mode = int(gp('p5_final_long_jump_mode').value)
        self.p5_final_long_jump_gait = int(gp('p5_final_long_jump_gait').value)

        self.p5_reset_roll = float(gp('p5_reset_roll').value)
        self.p5_reset_height = float(gp('p5_reset_height').value)
        self.p5_reset_body_wait_s = float(gp('p5_reset_body_wait_s').value)
        self.p5_reset_body_ramp_s = max(
            0.0, float(gp('p5_reset_body_ramp_s').value))

        self.p5_yellow_roi_top_ratio = float(gp('p5_yellow_roi_top_ratio').value)
        self.p5_yellow_roi_left_ratio = float(gp('p5_yellow_roi_left_ratio').value)
        self.p5_yellow_roi_right_ratio = float(gp('p5_yellow_roi_right_ratio').value)

        self.p5_yellow_h_min = int(gp('p5_yellow_h_min').value)
        self.p5_yellow_h_max = int(gp('p5_yellow_h_max').value)
        self.p5_yellow_s_min = int(gp('p5_yellow_s_min').value)
        self.p5_yellow_s_max = int(gp('p5_yellow_s_max').value)
        self.p5_yellow_v_min = int(gp('p5_yellow_v_min').value)
        self.p5_yellow_v_max = int(gp('p5_yellow_v_max').value)

        self.p5_yellow_min_contour_area = float(gp('p5_yellow_min_contour_area').value)
        self.p5_yellow_min_width_height_ratio = float(gp('p5_yellow_min_width_height_ratio').value)
        self.p5_yellow_max_tilt_deg = float(gp('p5_yellow_max_tilt_deg').value)
        self.p5_yellow_center_tolerance_ratio = float(gp('p5_yellow_center_tolerance_ratio').value)
        self.p5_yellow_min_width_ratio = float(gp('p5_yellow_min_width_ratio').value)

        self.p5_yellow_stop_line_y_ratio = float(gp('p5_yellow_stop_line_y_ratio').value)
        self.p5_yellow_stop_confirm_count = int(gp('p5_yellow_stop_confirm_count').value)
        self.p5_yellow_ignore_after_enter_s = float(gp('p5_yellow_ignore_after_enter_s').value)

        self.p5_center_yellow_roi_x_min = float(gp('p5_center_yellow_roi_x_min').value)
        self.p5_center_yellow_roi_x_max = float(gp('p5_center_yellow_roi_x_max').value)
        self.p5_center_yellow_roi_y_min = float(gp('p5_center_yellow_roi_y_min').value)
        self.p5_center_yellow_roi_y_max = float(gp('p5_center_yellow_roi_y_max').value)
        self.p5_center_yellow_min_pixels = int(gp('p5_center_yellow_min_pixels').value)
        self.p5_center_yellow_min_ratio = float(gp('p5_center_yellow_min_ratio').value)
        self.p5_center_yellow_absent_confirm_count = int(
            gp('p5_center_yellow_absent_confirm_count').value
        )
        self.p5_center_yellow_ignore_after_enter_s = float(
            gp('p5_center_yellow_ignore_after_enter_s').value
        )

        self.p5_keep_moving_when_no_image = bool(gp('p5_keep_moving_when_no_image').value)

        self.p5_sensor_watchdog_enabled = bool(gp('p5_sensor_watchdog_enabled').value)
        self.p5_sensor_max_frame_age_s = float(gp('p5_sensor_max_frame_age_s').value)
        self.p5_sensor_fault_grace_s = float(gp('p5_sensor_fault_grace_s').value)

        self.p5_up_slope_timeout_s = float(gp('p5_up_slope_timeout_s').value)
        self.p5_right_slope_1_timeout_s = float(gp('p5_right_slope_1_timeout_s').value)
        self.p5_right_slope_2_timeout_s = float(gp('p5_right_slope_2_timeout_s').value)
        self.p5_right_slope_3_timeout_s = float(gp('p5_right_slope_3_timeout_s').value)

        self.p5_right_slope_lost_extra_max_hold_s = max(
            0.0, float(gp('p5_right_slope_lost_extra_max_hold_s').value)
        )

        self.p5_evidence_log_dir = str(gp('p5_evidence_log_dir').value)

        self.p5_action_timeout_s = max(0.1, float(gp('p5_action_timeout_s').value))
        self.p5_stop_timeout_s = max(0.1, float(gp('p5_stop_timeout_s').value))
        self.p5_action_min_ack_delay_s = max(
            0.0, float(gp('p5_action_min_ack_delay_s').value))
        self.p5_action_require_progress = bool(
            gp('p5_action_require_progress').value)
        self.p5_action_resend_max = max(
            0, int(gp('p5_action_resend_max').value))
        self.p5_action_resend_after_s = max(
            0.5, float(gp('p5_action_resend_after_s').value))
        self.p5_action_stall_unwedge_after_s = max(
            0.0, float(gp('p5_action_stall_unwedge_after_s').value))
        self.p5_action_unwedge_release_timeout_s = max(
            0.5, float(gp('p5_action_unwedge_release_timeout_s').value))
        if self.p5_action_stall_unwedge_after_s > 0.0:
            self.get_logger().info(
                '[P5_PARAM] stalled-action unwedge armed at '
                f'{self.p5_action_stall_unwedge_after_s:.1f}s '
                '(kOff -> recovery stand)')
        self.p5_fall_recovery_enabled = bool(
            gp('p5_fall_recovery_enabled').value)
        self.p5_fall_recovery_max_attempts = max(
            0, int(gp('p5_fall_recovery_max_attempts').value))
        self.p5_fall_recovery_min_rp_rad = max(
            0.1, float(gp('p5_fall_recovery_min_rp_rad').value))
        self.p5_fall_recovery_release_timeout_s = max(
            0.5, float(gp('p5_fall_recovery_release_timeout_s').value))
        self.p5_fall_recovery_stand_timeout_s = max(
            1.0, float(gp('p5_fall_recovery_stand_timeout_s').value))
        self.p5_fall_recovery_total_timeout_s = max(
            1.0, float(gp('p5_fall_recovery_total_timeout_s').value))
        if self.p5_fall_recovery_enabled:
            self.get_logger().info(
                '[P5_PARAM] corner fall pick-up armed: '
                f'{self.p5_fall_recovery_max_attempts} attempt(s), '
                'stand up then hold (never resumes the route)')
        self.p5_action_post_complete_hold_s = max(
            0.0, float(gp('p5_action_post_complete_hold_s').value))
        self.p5_action_feedback_max_age_s = max(
            0.05, float(gp('p5_action_feedback_max_age_s').value))
        self.p5_timed_motion_timeout_factor = max(
            1.0, float(gp('p5_timed_motion_timeout_factor').value))
        self.p5_timed_motion_timeout_margin_s = max(
            0.1, float(gp('p5_timed_motion_timeout_margin_s').value))
        self.p5_forward_after_reset_body_timeout_s = max(
            0.1, float(gp('p5_forward_after_reset_body_timeout_s').value))

        self.p5_bridge_observer_enabled = bool(gp('p5_bridge_observer_enabled').value)
        self.p5_bridge_observer_period_s = max(
            0.0, float(gp('p5_bridge_observer_period_s').value))
        self.p5_depth_camera_info_topic = str(gp('p5_depth_camera_info_topic').value)
        self.p5_imu_topic = str(gp('p5_imu_topic').value)
        self.p5_depth_horizontal_fov = float(gp('p5_depth_horizontal_fov').value)
        self.p5_depth_camera_mount_roll = float(
            gp('p5_depth_camera_mount_roll').value)
        self.p5_depth_camera_mount_pitch = float(
            gp('p5_depth_camera_mount_pitch').value)
        self.p5_depth_camera_mount_x = float(
            gp('p5_depth_camera_mount_x').value)
        self.p5_depth_camera_mount_y = float(
            gp('p5_depth_camera_mount_y').value)
        self.p5_depth_camera_mount_z = float(
            gp('p5_depth_camera_mount_z').value)
        self.p5_bridge_max_imu_age_s = max(
            0.0, float(gp('p5_bridge_max_imu_age_s').value))
        self.p5_bridge_max_imu_depth_skew_s = max(
            0.0, float(gp('p5_bridge_max_imu_depth_skew_s').value))

        self.p5_deck_lateral_enabled = bool(gp('p5_deck_lateral_enabled').value)
        self.p5_deck_lateral = DeckLateralController(DeckLateralConfig(
            k_vy=float(gp('p5_deck_lateral_k_vy').value),
            k_wz=float(gp('p5_deck_lateral_k_wz').value),
            deadband_m=abs(float(gp('p5_deck_lateral_deadband_m').value)),
            heading_deadband_rad=abs(
                float(gp('p5_deck_lateral_heading_deadband_rad').value)),
            max_vy=abs(float(gp('p5_deck_lateral_max_vy').value)),
            max_wz=abs(float(gp('p5_deck_lateral_max_wz').value)),
            max_age_s=max(0.0, float(gp('p5_deck_lateral_max_age_s').value)),
            min_consecutive_valid=max(
                1, int(gp('p5_deck_lateral_min_consecutive_valid').value)),
            max_plausible_offset_m=abs(
                float(gp('p5_deck_lateral_max_plausible_offset_m').value)),
            blind_vx_scale=min(1.0, max(
                0.0, float(gp('p5_deck_lateral_blind_vx_scale').value))),
            centre_first_offset_m=max(
                0.0, float(gp('p5_deck_lateral_centre_first_offset_m').value)),
            centre_first_release_m=max(
                0.0, float(gp('p5_deck_lateral_centre_first_release_m').value)),
            centre_first_vx_scale=min(1.0, max(0.0, float(
                gp('p5_deck_lateral_centre_first_vx_scale').value))),
            k_i_vy=max(0.0, float(gp('p5_deck_lateral_k_i_vy').value)),
            max_i_vy=max(0.0, float(gp('p5_deck_lateral_max_i_vy').value)),
        ))
        self.p5_route_lateral_fallback = bool(
            gp('p5_route_lateral_fallback').value)
        self.p5_route_lateral_heading_only = bool(
            gp('p5_route_lateral_heading_only').value)
        self.p5_route_lateral_snap_heading = bool(
            gp('p5_route_lateral_snap_heading').value)
        declared_base_deg = float(gp('p5_route_base_yaw_deg').value)
        self.p5_route_base_yaw_declared = (
            None if declared_base_deg != declared_base_deg      # NaN = 不声明
            else wrap_rad(math.radians(declared_base_deg)))
        self.p5_start_align_enabled = bool(gp('p5_start_align_enabled').value)
        self.p5_start_align_tol_rad = abs(math.radians(
            float(gp('p5_start_align_tol_deg').value)))
        self.p5_start_align_wz = abs(float(gp('p5_start_align_wz').value))
        self.p5_start_align_step_height = abs(float(
            gp('p5_start_align_step_height').value))
        self.p5_start_align_timeout_s = max(
            0.0, float(gp('p5_start_align_timeout_s').value))
        self.p5_route_lateral_anchor_enabled = bool(
            gp('p5_route_lateral_anchor_enabled').value)
        self.p5_route_lateral_anchor_suppressed_depth = bool(
            gp('p5_route_lateral_anchor_suppressed_depth').value)
        self.p5_route_lateral_anchor_max_m = abs(
            float(gp('p5_route_lateral_anchor_max_m').value))
        self.p5_bridge_config.single_sided_edges_enabled = bool(
            gp('p5_bridge_single_sided_edges_enabled').value)
        self.p5_bridge_config.declared_deck_width = max(
            0.0, float(gp('p5_bridge_declared_deck_width').value))
        self.p5_route_progress_measure = str(
            gp('p5_route_progress_measure').value)
        if self.p5_route_progress_measure not in ('path', 'along_track'):
            self.get_logger().warn(
                f'[P5_PARAM] unknown p5_route_progress_measure='
                f'{self.p5_route_progress_measure}, fallback to path')
            self.p5_route_progress_measure = 'path'

        entry_depth_min_frames = max(
            0, int(gp('p5_route_entry_depth_min_frames').value))
        if entry_depth_min_frames > 0 and not self.p5_bridge_observer_enabled:
            self.get_logger().warn(
                '[P5_PARAM] p5_route_entry_depth_min_frames set but the '
                'bridge observer is disabled; entry depth gate stays off')
            entry_depth_min_frames = 0
        self.p5_route_entry_depth_gate = EntryDepthGate(
            segment_name=str(gp('p5_route_entry_depth_segment').value),
            min_valid_frames=entry_depth_min_frames)

        self.p5_route_cross_track_gate = CrossTrackGate(
            limit_m=max(0.0, float(gp('p5_route_max_cross_track_m').value)),
            consecutive_samples=max(
                1, int(gp('p5_route_cross_track_samples').value)))
        self.p5_route_topple_gate = ToppleGate(
            limit_rad=abs(float(gp('p5_route_topple_limit_rad').value)),
            consecutive_samples=max(
                1, int(gp('p5_route_topple_samples').value)))
        self.p5_route_stall_gate = StallGate(
            min_speed=abs(float(gp('p5_route_stall_min_speed').value)),
            timeout_s=max(0.0, float(gp('p5_route_stall_timeout_s').value)),
            min_progress_m=abs(
                float(gp('p5_route_stall_min_progress_m').value)))
        self.p5_route_stall_max_attempts = max(
            0, int(gp('p5_route_stall_max_attempts').value))
        self.p5_deck_lateral_engage_after_m = max(
            0.0, float(gp('p5_deck_lateral_engage_after_m').value))
        self.p5_deck_lateral_centre_first_states = frozenset(
            name.strip() for name
            in str(gp('p5_deck_lateral_centre_first_states').value).split(',')
            if name.strip())
        if self.p5_deck_lateral_centre_first_states:
            self.get_logger().info(
                '[P5_PARAM] centre-first vx hold restricted to '
                + ', '.join(sorted(self.p5_deck_lateral_centre_first_states)))
        self.p5_deck_lateral_depth_states = frozenset(
            name.strip() for name
            in str(gp('p5_deck_lateral_depth_states').value).split(',')
            if name.strip())
        if self.p5_deck_lateral_depth_states:
            self.get_logger().info(
                '[P5_PARAM] depth lateral observations restricted to '
                + ', '.join(sorted(self.p5_deck_lateral_depth_states)))

        self.p5_yellow_angle_align_enabled = bool(gp('p5_yellow_angle_align_enabled').value)
        self.p5_yellow_angle_align_fixed_wz = abs(float(gp('p5_yellow_angle_align_fixed_wz').value))
        self.p5_yellow_angle_align_deadband_deg = float(gp('p5_yellow_angle_align_deadband_deg').value)

        self.p5_route_model_enabled = bool(gp('p5_route_model_enabled').value)
        self.p5_route_model_mode = str(gp('p5_route_model_mode').value)
        self.p5_route_overrun_action = str(gp('p5_route_overrun_action').value)
        self.p5_route_odometry_required = bool(gp('p5_route_odometry_required').value)
        self.p5_route_odom_max_age_s = max(0.0, float(gp('p5_route_odom_max_age_s').value))
        self.p5_route_odom_max_step_m = max(1e-3, float(gp('p5_route_odom_max_step_m').value))
        self.p5_route_speed_cap_enabled = bool(gp('p5_route_speed_cap_enabled').value)
        self.p5_route_turn_verify_enabled = bool(gp('p5_route_turn_verify_enabled').value)
        self.p5_route_realign_wz = abs(float(gp('p5_route_realign_wz').value))
        self.p5_route_realign_step_height = abs(float(gp('p5_route_realign_step_height').value))
        self.p5_route_realign_timeout_s = max(0.0, float(gp('p5_route_realign_timeout_s').value))
        self.p5_route_realign_max_attempts = max(0, int(gp('p5_route_realign_max_attempts').value))
        self.p5_route_exit_source = str(gp('p5_route_exit_source').value)
        self.p5_yellow_lateral_correction_enabled = bool(
            gp('p5_yellow_lateral_correction_enabled').value)

        if self.p5_route_exit_source not in (EXIT_VISION, EXIT_ODOMETRY):
            self.get_logger().error(
                f'[P5_ROUTE] unknown p5_route_exit_source='
                f'{self.p5_route_exit_source!r}, fall back to {EXIT_VISION}')
            self.p5_route_exit_source = EXIT_VISION

        if self.p5_route_model_mode not in ('enforce', 'monitor'):
            self.get_logger().error(
                f'[P5_ROUTE] unknown p5_route_model_mode='
                f'{self.p5_route_model_mode!r}, fall back to monitor')
            self.p5_route_model_mode = 'monitor'
        if self.p5_route_overrun_action not in ('fault', 'degraded_advance'):
            self.get_logger().error(
                f'[P5_ROUTE] unknown p5_route_overrun_action='
                f'{self.p5_route_overrun_action!r}, fall back to fault')
            self.p5_route_overrun_action = 'fault'

        overrides = {}
        for segment in RouteModel().segments:
            prefix = 'p5_route_{}'.format(segment.name)
            fields = {
                'expected_m': float(gp(prefix + '_expected_m').value),
                'min_m': float(gp(prefix + '_min_m').value),
                'max_m': float(gp(prefix + '_max_m').value),
                'expected_yaw_deg': float(gp(prefix + '_expected_yaw_deg').value),
                'yaw_tol_deg': float(gp(prefix + '_yaw_tol_deg').value),
                'speed_cap_mps': float(gp(prefix + '_speed_cap_mps').value),
            }
            # 只有原本靠感知结束的段（四条直行段）会改用里程结束。
            # 定点转角保留航向校核，定时段保留各自的计时转移。
            if segment.enforced and segment.progress == PROGRESS_DISTANCE:
                fields['exit_source'] = self.p5_route_exit_source
                if self.p5_route_exit_source == EXIT_ODOMETRY:
                    # 单证据推进必须显式声明降级档，段表校验会检查这一点。
                    fields['fallback_tier'] = TIER_DEAD_RECKONING
                    fields['exit_evidence'] = (
                        'odometry distance only (declared dead reckoning, '
                        'bounded by max_m and the per-state timeout)')
            overrides[segment.name] = fields
        self.p5_route_model = RouteModel().with_overrides(overrides)
        problems = self.p5_route_model.validate()
        if problems:
            # 段表自洽性是配置错误，不是运行时状况：不静默降级。
            for problem in problems:
                self.get_logger().error(f'[P5_ROUTE] invalid route segment: {problem}')
            raise ValueError(
                'invalid Stage 5 route model configuration: {}'.format(problems))
        self.p5_route_progress = SegmentProgress(
            max_step_m=self.p5_route_odom_max_step_m)
        self.get_logger().info(
            f'[P5_ROUTE] route model enabled={self.p5_route_model_enabled}, '
            f'mode={self.p5_route_model_mode}, '
            f'exit_source={self.p5_route_exit_source}, '
            f'yellow_lateral_correction='
            f'{self.p5_yellow_lateral_correction_enabled}, '
            f'overrun={self.p5_route_overrun_action}, '
            f'turn_verify={self.p5_route_turn_verify_enabled}, '
            f'speed_cap={self.p5_route_speed_cap_enabled}, '
            f'segments={len(self.p5_route_model.segments)}')

    # ============================================================
    # 时间与状态工具
    # ============================================================
    def p5_now_sec(self) -> float:
        # use_sim_time=True 时，这里读取的是 Gazebo /clock 仿真时间。
        return self.get_clock().now().nanoseconds * 1e-9

    def p5_safety_elapsed_s(self) -> float:
        """Elapsed wall duration immune to missing/frozen ROS /clock."""
        return max(0.0, time.monotonic() - self.state_start_monotonic_s)

    def p5_state_elapsed_s(self) -> float:
        now = self.p5_now_sec()

        # 刚切到仿真时间时，如果 /clock 还没来，now 可能是 0。
        # 这时不要让 elapsed 变成异常大值，也不要推进状态机。
        if now <= 0.0:
            return 0.0

        # 如果进入状态时 /clock 尚未有效，state_start_time 可能是 0。
        # 第一次拿到有效 /clock 后，从当前仿真时刻重新开始计时。
        if getattr(self, 'state_start_time', 0.0) <= 0.0:
            self.state_start_time = now
            self.get_logger().info(
                f'[P5_TIME] start state timer after /clock valid: '
                f'state={self.state}, sim_time={now:.3f}',
                throttle_duration_sec=1.0
            )
            return 0.0

        self.state_start_time = self.align_motion_timer_start(
            self.state_start_time, now)
        return max(0.0, now - self.state_start_time)

    def p5_enter_state(self, new_state: str):
        # 单一转移收口：离开转角段时先做航向校核，失败则改道再对齐/故障保持，
        # 而不是让每个状态处理函数各自记得校核。
        new_state = self.p5_route_check_turn(new_state)

        now = self.p5_now_sec()
        self.get_logger().info(f'[P5] ENTER STATE -> {new_state}, sim_time={now:.3f}')
        self.p5_evidence_log({
            'event': 'state_transition',
            'from': str(getattr(self, 'state', '')),
            'to': str(new_state),
            'frame_seq': int(self.latest_frame_seq),
        })
        self.state = new_state
        self.state_start_time = now
        self.state_start_monotonic_s = time.monotonic()
        self.state_enter_frame_seq = self.latest_frame_seq
        self.action_sent = False
        self.p5_action_phase = 'idle'
        self.p5_action_response_seq = 0
        self.p5_action_sent_monotonic_s = None
        self.p5_action_target = None
        self.p5_action_progress_seen = False
        self.p5_action_completed_monotonic_s = None
        self.p5_action_resends_done = 0
        self.p5_action_last_send_monotonic_s = None
        self.p5_action_recovery_pending = False
        self.p5_stop_complete_seq = 0
        self.p5_stop_complete_rx_monotonic_s = None
        self.p5_action_stall_since_monotonic_s = None
        self.p5_action_stall_bar = None
        self.p5_action_unwedge_phase = ''
        self.p5_action_unwedge_done = False
        self.p5_action_unwedge_origin = None
        self.p5_action_unwedge_sent_monotonic_s = None
        # 扶起梯子的阶段状态跟着状态切换清零；p5_fall_recovery_begin() 在调用
        # 本函数之后才装入重试上下文，所以这里的清零不会吃掉它。
        self.p5_fall_recover_phase = ''
        self.p5_fall_recover_since_monotonic_s = None
        self.p5_fall_recover_retry_state = ''
        self.p5_yellow_stop_counter = 0
        self.p5_center_yellow_absent_counter = 0
        self.p5_center_yellow_last_eval_frame_seq = self.state_enter_frame_seq
        self.p5_right_side_yellow_lost_counter = 0
        self.p5_right_side_yellow_last_eval_frame_seq = self.state_enter_frame_seq
        self.p5_forward_align_stable_counter = 0
        self.p5_forward_align_lost_counter = 0
        self.p5_forward_align_last_eval_frame_seq = self.state_enter_frame_seq
        self.p5_right_slope_lost_extra_last_eval_frame_seq = self.state_enter_frame_seq

        if new_state in [
            self.P5_RIGHT_SLOPE_1,
            self.P5_RIGHT_SLOPE_2,
            self.P5_RIGHT_SLOPE_3,
        ]:
            self.reset_p5_right_slope_lost_extra_state()

        self.p5_route_on_state_entered(new_state)

    def p5_inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 1

    # ============================================================
    # 传感器健康 watchdog / 状态超时 / 证据日志
    # ============================================================
    def p5_timed_motion_guard(self, duration_s: float, log_name: str) -> bool:
        """Fail closed if a ROS-clock timed motion outlives its wall deadline."""
        hard_timeout_s = (
            max(0.0, float(duration_s)) * self.p5_timed_motion_timeout_factor
            + self.p5_timed_motion_timeout_margin_s
        )
        elapsed = self.p5_safety_elapsed_s()
        if elapsed < hard_timeout_s:
            return False
        self.get_logger().error(
            f'[{log_name}] timed motion wall timeout: elapsed={elapsed:.2f}s, '
            f'hard_timeout={hard_timeout_s:.2f}s')
        self.p5_evidence_log({
            'event': 'timed_motion_timeout',
            'state': str(self.state),
            'elapsed_s': float(elapsed),
            'hard_timeout_s': float(hard_timeout_s),
        })
        self.p5_send_velocity_command(
            vx=0.0, vy=0.0, wz=0.0, step_height=0.0)
        self.p5_enter_state(self.P5_SENSOR_FAULT_HOLD)
        return True

    def p5_state_timeout_guard(self, timeout_s: float, log_name: str) -> bool:
        """状态整体超时守护（与传感器种类无关）。

        触发时发一次零速度命令并切入 P5_SENSOR_FAULT_HOLD，返回 True。
        里程主导的段不消费 RGB，只需要这一层；视觉门控段在它之上再加
        RGB 新鲜度 watchdog。
        """
        elapsed = self.p5_safety_elapsed_s()

        if timeout_s > 0.0 and elapsed >= timeout_s:
            self.get_logger().error(
                f'[{log_name}] state timeout: elapsed={elapsed:.1f}s >= '
                f'{timeout_s:.1f}s, enter {self.P5_SENSOR_FAULT_HOLD}'
            )
            self.p5_evidence_log({
                'event': 'stage_timeout',
                'state': str(self.state),
                'elapsed_s': float(elapsed),
                'timeout_s': float(timeout_s),
            })
            self.p5_send_velocity_command(vx=0.0, vy=0.0, wz=0.0, step_height=0.0)
            self.p5_enter_state(self.P5_SENSOR_FAULT_HOLD)
            return True

        return False

    def p5_vision_state_guard(self, timeout_s: float, log_name: str) -> bool:
        """视觉门控状态的公共守护：状态超时 + RGB 流新鲜度 watchdog。

        触发时发一次零速度命令并切入 P5_SENSOR_FAULT_HOLD，返回 True，
        调用方应立即 return。未触发返回 False。
        """
        if self.p5_state_timeout_guard(timeout_s, log_name):
            return True

        elapsed = self.p5_safety_elapsed_s()
        if not self.p5_sensor_watchdog_enabled:
            return False
        if elapsed < self.p5_sensor_fault_grace_s:
            return False

        age = self.rgb_age_s()
        if age is None or age > self.p5_sensor_max_frame_age_s:
            age_text = 'never' if age is None else f'{age:.2f}s'
            self.get_logger().error(
                f'[{log_name}] rgb stream stale: age={age_text} > '
                f'{self.p5_sensor_max_frame_age_s:.2f}s, '
                f'enter {self.P5_SENSOR_FAULT_HOLD}'
            )
            self.p5_evidence_log({
                'event': 'sensor_fault',
                'state': str(self.state),
                'rgb_age_s': None if age is None else float(age),
                'max_frame_age_s': float(self.p5_sensor_max_frame_age_s),
            })
            self.p5_send_velocity_command(vx=0.0, vy=0.0, wz=0.0, step_height=0.0)
            self.p5_enter_state(self.P5_SENSOR_FAULT_HOLD)
            return True

        return False

    def p5_evidence_log(self, event: dict):
        """向 JSONL 证据日志追加一条事件；目录未配置或写失败时静默降级。"""
        if not getattr(self, 'p5_evidence_log_dir', ''):
            return
        if self._p5_evidence_failed:
            return
        try:
            if self._p5_evidence_fp is None:
                os.makedirs(self.p5_evidence_log_dir, exist_ok=True)
                path = os.path.join(
                    self.p5_evidence_log_dir,
                    f'stage5_evidence_{int(time.time())}.jsonl'
                )
                self._p5_evidence_fp = open(path, 'a', encoding='utf-8')
                self.get_logger().info(f'[P5_EVIDENCE] logging to {path}')
            record = {
                't_wall': time.time(),
                't_monotonic': time.monotonic(),
                't_sim': self.p5_now_sec(),
            }
            record.update(event)
            self._p5_evidence_fp.write(json.dumps(record, ensure_ascii=False) + '\n')
            self._p5_evidence_fp.flush()
        except OSError as e:
            self._p5_evidence_failed = True
            self.get_logger().warn(f'[P5_EVIDENCE] disabled after write failure: {e}')

    # ============================================================
    # 路线模型（STAGE5_PHYSICAL_REDESIGN_PLAN.md §4.1/§4.2）
    # ============================================================
    def p5_route_active(self) -> bool:
        return bool(self.p5_route_model_enabled and self.p5_route_model is not None)

    def p5_route_enforcing(self, segment) -> bool:
        """True when this segment's transitions must satisfy both sources."""
        return bool(
            self.p5_route_active()
            and self.p5_route_model_mode == 'enforce'
            and segment is not None
            and segment.enforced
        )

    def p5_route_read_odometry(self):
        """Integrate one fresh state-estimator sample into the current segment.

        Sets ``p5_route_odom_valid``.  A never-received or stale stream is
        invalid: it must not be mistaken for "no motion", which would freeze
        progress at zero and hold every gate closed silently.
        """
        self.p5_route_odom_valid = False
        self.p5_route_odom_age_s = None
        if not self.p5_route_active() or self.Odom is None:
            return
        progress = self.p5_route_progress
        snapshot = self.Odom.snapshot()
        self.p5_route_odom_seq = int(snapshot['seq'])
        if snapshot['seq'] <= 0 or snapshot['rx_monotonic_s'] is None:
            return
        age = max(0.0, time.monotonic() - float(snapshot['rx_monotonic_s']))
        self.p5_route_odom_age_s = age
        if age > self.p5_route_odom_max_age_s:
            return
        self.p5_route_odom_valid = True
        progress.update(
            seq=snapshot['seq'],
            x=snapshot['p'][0],
            y=snapshot['p'][1],
            yaw=snapshot['rpy'][2],
        )
        # Measure progress along the route's *declared* direction rather than
        # the entry yaw the last corner happened to leave behind.
        if (self.p5_route_lateral_snap_heading and progress.reference_yaw is None
                and progress.origin_yaw is not None):
            reference, _ = snap_reference_heading(
                progress.origin_yaw, self.p5_route_base_yaw)
            if reference is not None:
                progress.set_reference_yaw(reference)

    def p5_route_distance_m(self) -> float:
        """Return the active segment's progress under the configured measure."""
        if self.p5_route_progress_measure == 'along_track':
            return max(0.0, float(self.p5_route_progress.along_track_m))
        return float(self.p5_route_progress.distance_m)

    def p5_route_yaw_delta_deg(self) -> float:
        return float(self.p5_route_progress.yaw_delta_deg)

    def p5_route_speed_cap_mps(self) -> float:
        if not (self.p5_route_speed_cap_enabled and self.p5_route_active()):
            return 0.0
        if self.p5_route_segment is None:
            return 0.0
        return float(self.p5_route_segment.speed_cap_mps)

    def p5_route_fault(self, log_name: str, event: str, detail: dict):
        """Stop and hold after a route-model violation.

        Sends the real STOP command (mode 12) directly rather than a
        zero-velocity locomotion command, so the fault path cannot transit the
        locomotion FSM on its way to holding still.
        """
        payload = {'event': event, 'state': str(self.state)}
        payload.update(detail)
        self.get_logger().error(f'[{log_name}] route fault: {payload}')
        self.p5_evidence_log(payload)
        self.p5_send_stop_command()
        self.p5_enter_state(self.P5_SENSOR_FAULT_HOLD)

    def p5_route_monitor(self) -> bool:
        """Per-tick route bookkeeping. True when it took the state over.

        Runs before the state dispatch so that odometry integration, window
        logging, and overrun/odometry faults are independent of which state
        happens to be active.
        """
        if not self.p5_route_active():
            return False

        self.p5_route_read_odometry()
        segment = self.p5_route_segment
        if segment is None:
            return False

        if (
            self.p5_route_entry_depth_gate.tripped
            and not self.p5_route_entry_depth_faulted
            and self.state != self.P5_SENSOR_FAULT_HOLD
        ):
            self.p5_route_entry_depth_faulted = True
            self.p5_route_fault(
                'P5_ROUTE', 'route_entry_depth_fault',
                self.p5_route_entry_depth_gate.snapshot())
            return True

        enforcing = self.p5_route_enforcing(segment)

        if not self.p5_route_odom_valid:
            if (
                enforcing
                and self.p5_route_odometry_required
                and self.p5_safety_elapsed_s() >= self.p5_sensor_fault_grace_s
            ):
                self.p5_route_fault(
                    'P5_ROUTE', 'route_odometry_fault', {
                        'segment': segment.name,
                        'odom_seq': int(self.p5_route_odom_seq),
                        'odom_age_s': self.p5_route_odom_age_s,
                        'max_age_s': float(self.p5_route_odom_max_age_s),
                    })
                return True
            self.p5_route_log_status(segment, GATE_UNAVAILABLE)
            return False

        # 横向偏离检查放在段类型判断之前：走下赛道与本段是靠里程还是靠视觉
        # 收尾无关，而恰恰是视觉收尾的段最需要这一层。
        # 只对**有边可掉**的段生效（lateral_profile != flat）：转角跳跃、
        # 跳跃后对齐、下坡与终点区都是刻意的横向位移，在那里量横向偏离量的
        # 是动作本身。实测代价：不加这一条时连续三跑倒在 P5_FINAL_LONG_JUMP
        # ——P5_DONE 前的最后一个状态。
        # 翻倒判定放在最前面，而且不看 lateral_profile：机体已经躺下这件事
        # 与它在哪一段无关，而失败的转角跳跃恰恰把机体扔到段与段之间。
        if self.p5_route_topple_gate.record(*self.p5_odom_attitude_rp()):
            detail = {'segment': segment.name}
            detail.update(self.p5_route_topple_gate.snapshot())
            self.p5_route_fault('P5_ROUTE', 'route_topple_fault', detail)
            return True

        if CrossTrackGate.applies_to(segment) and self.p5_route_cross_track_gate.record(
                self.p5_route_progress.cross_track_m):
            detail = {'segment': segment.name}
            detail.update(self.p5_route_cross_track_gate.snapshot())
            detail['cross_track_m'] = float(
                self.p5_route_progress.cross_track_m)
            self.p5_route_fault('P5_ROUTE', 'route_cross_track_fault', detail)
            return True

        if segment.progress != PROGRESS_DISTANCE:
            return False

        distance = self.p5_route_distance_m()
        if segment.max_m > 0.0 and distance > segment.max_m:
            return self.p5_route_handle_overrun(segment, distance, enforcing)

        self.p5_route_log_status(
            segment,
            GATE_BELOW_MIN if distance < segment.min_m else GATE_IN_WINDOW)
        return False

    def p5_route_handle_overrun(self, segment, distance, enforcing) -> bool:
        """Apply the declared overrun policy once per segment."""
        detail = {
            'segment': segment.name,
            'distance_m': float(distance),
            'window_m': [segment.min_m, segment.max_m],
            'fallback_tier': int(segment.fallback_tier),
            'odom_rejected_steps': int(self.p5_route_progress.rejected_steps),
        }

        if not enforcing:
            self.p5_route_log_status(segment, GATE_OVERRUN, detail)
            return False

        if self.p5_route_overrun_handled:
            return False
        self.p5_route_overrun_handled = True

        if (
            self.p5_route_overrun_action == 'degraded_advance'
            and segment.degraded_next_state
        ):
            # 声明式有界降级（计划 §3 B3 第 3 档）：只用里程推进一次段尾，
            # 记录证据，绝不静默当成正常的双证据转移。
            detail['event'] = 'route_overrun_degraded_advance'
            detail['next_state'] = str(segment.degraded_next_state)
            self.get_logger().warn(
                f'[P5_ROUTE] segment {segment.name} overran its window '
                f'({distance:.2f} m > {segment.max_m:.2f} m) with no exit '
                f'evidence; declared degraded advance to '
                f'{segment.degraded_next_state}')
            self.p5_evidence_log(detail)
            self.p5_enter_state(segment.degraded_next_state)
            return True

        self.p5_route_fault('P5_ROUTE', 'route_overrun_fault', detail)
        return True

    def p5_route_log_status(self, segment, status, detail=None):
        """Log a segment's window status once per change, not per tick."""
        key = '{}:{}'.format(segment.name, status)
        if key == self.p5_route_logged_status:
            return
        self.p5_route_logged_status = key
        payload = {
            'event': 'route_window_status',
            'state': str(self.state),
            'segment': segment.name,
            'status': str(status),
            'distance_m': self.p5_route_distance_m(),
            'yaw_delta_deg': self.p5_route_yaw_delta_deg(),
            'window_m': [segment.min_m, segment.max_m],
            'enforced': bool(self.p5_route_enforcing(segment)),
            'odom_valid': bool(self.p5_route_odom_valid),
            'odom_samples': int(self.p5_route_progress.samples),
        }
        if detail:
            payload.update(detail)
        self.get_logger().info(f'[P5_ROUTE] {payload}')
        self.p5_evidence_log(payload)

    def p5_route_blocks_exit(self, log_name: str) -> bool:
        """True when a perception exit trigger must be suppressed.

        The caller has already confirmed its own (vision) evidence; this asks
        the odometry window whether the second source agrees.
        """
        segment = self.p5_route_segment
        if not self.p5_route_enforcing(segment):
            return False
        if segment.progress != PROGRESS_DISTANCE:
            return False

        decision = evaluate_gate(
            segment,
            progress_m=self.p5_route_distance_m(),
            exit_confirmed=True,
            odometry_valid=self.p5_route_odom_valid,
        )
        if decision.allow_exit:
            return False

        payload = {
            'event': 'route_exit_suppressed',
            'state': str(self.state),
            'segment': segment.name,
        }
        payload.update(decision.to_dict())
        self.get_logger().warn(
            f'[{log_name}] route gate suppresses exit evidence: '
            f'status={decision.status}, progress={decision.progress_m:.2f} m, '
            f'window={decision.window}',
            throttle_duration_sec=1.0)
        self.p5_route_log_status(segment, decision.status, payload)
        return True

    def p5_route_segment_uses_odometry_exit(self) -> bool:
        """Return True while the active segment ends on odometry distance alone."""
        segment = self.p5_route_segment
        return bool(
            self.p5_route_active()
            and segment is not None
            and segment.odometry_exit
            and segment.progress == PROGRESS_DISTANCE
        )

    def p5_route_lateral_observation(self):
        """Build the odometry fallback observation for the active segment.

        Returns ``None`` when the fallback is switched off, so the ladder
        collapses to depth-only and the previous behaviour is unchanged.
        """
        if not self.p5_route_lateral_fallback:
            return None
        progress = self.p5_route_progress
        if progress.last_xy is None or progress.origin_xy is None:
            return {'valid': False, 'source': SOURCE_ODOMETRY,
                    'reason': 'no_segment_origin',
                    'lateral_offset': None, 'heading_error': None}

        reference_yaw = progress.origin_yaw
        if self.p5_route_lateral_snap_heading:
            reference_yaw, snap_error = snap_reference_heading(
                progress.origin_yaw, self.p5_route_base_yaw)
            if reference_yaw is None:
                # Refusing to snap is the safe answer: an entry yaw this far
                # off the declared grid means the corner went wrong, and
                # holding a line drawn from it would steer into the failure.
                return {'valid': False, 'source': SOURCE_ODOMETRY,
                        'reason': 'heading_off_route_grid',
                        'snap_error_rad': snap_error,
                        'lateral_offset': None, 'heading_error': None}

        return route_frame_observation(
            progress.origin_xy, reference_yaw,
            progress.last_xy[0], progress.last_xy[1], progress.last_yaw,
            odometry_valid=self.p5_route_odom_valid,
            mode=(USE_HEADING_ONLY if self.p5_route_lateral_heading_only
                  else USE_FULL),
            line_offset_m=self.p5_route_lateral_anchor_m,
        )

    def p5_route_lateral_reanchor(self, depth_observation, odometry_observation,
                                  once_per_segment=False):
        """Re-anchor the odometry line whenever depth can still see the deck.

        Without this the fallback holds the segment's entry pose, which reports
        zero error for a body that entered off centre — masking exactly the
        deviation the centre-first hold exists to catch.

        Two guards, both learned the hard way (2026-08-16, ``race_physical``):

        ``frame_seq`` de-duplication is not an optimisation.  The loop runs at
        10 Hz and the observer at 5 Hz, so without it the anchor is recomputed
        several times against the *same* depth fix while odometry keeps moving
        underneath it — which solves ``anchor = depth - odometry(t)`` afresh
        every tick and pins the reported offset to the held depth number.  The
        odometry line then contributes nothing and the loop is steering on raw
        depth, allow-list or no allow-list.

        ``once_per_segment`` is what the allow-list means on a surface where
        depth may not steer: one fix to place the line, then dead reckoning.
        Measured on straight_3, where the observer is noisy: continuous
        anchoring swung the line +0.009 -> -0.040 -> +0.093 m within two
        seconds and the loop chased every swing.
        """
        if not self.p5_route_lateral_anchor_enabled:
            return
        if not (isinstance(depth_observation, dict)
                and depth_observation.get('valid')):
            return
        seq = depth_observation.get('frame_seq')
        if seq is not None and seq == self.p5_route_lateral_anchor_seq:
            return
        if once_per_segment and self.p5_route_lateral_anchor_source == str(self.state):
            return
        if not (isinstance(odometry_observation, dict)
                and odometry_observation.get('valid')):
            return
        raw = odometry_observation.get('lateral_offset')
        if raw is None:
            return
        # The stored observation already carries the current anchor; strip it
        # back out so the new anchor is computed against the raw line.
        raw_unanchored = raw - float(odometry_observation.get('line_offset_m', 0.0))
        anchor = anchor_from_depth(
            depth_observation.get('lateral_offset'), raw_unanchored,
            max_anchor_m=self.p5_route_lateral_anchor_max_m)
        if anchor is None:
            # Not silent: a rejected fix means the segment dead-reckons from its
            # entry pose, which is a materially weaker reference, and a failure
            # trace has to show which of the two the run was steering on.
            self.get_logger().warn(
                f'[{self.state}] depth fix rejected for anchoring '
                f'(offset={depth_observation.get("lateral_offset")}, '
                f'limit={self.p5_route_lateral_anchor_max_m:.2f} m); '
                f'odometry line stays on the segment entry pose',
                throttle_duration_sec=5.0)
            return
        self.p5_route_lateral_anchor_seq = seq
        self.p5_route_lateral_anchor_m = anchor
        if self.p5_route_lateral_anchor_source != str(self.state):
            # Log once per state: the first anchor of a segment is the one that
            # decides whether the fallback holds the course centreline or
            # whatever pose the corner left the body in.
            self.p5_route_lateral_anchor_source = str(self.state)
            self.get_logger().info(
                f'[{self.state}] odometry line anchored to depth: '
                f'offset={depth_observation.get("lateral_offset")}, '
                f'anchor={anchor:+.3f} m, once={once_per_segment}')
            self.p5_evidence_log({
                'event': 'route_lateral_anchor',
                'state': str(self.state),
                'anchor_m': anchor,
                'once_per_segment': bool(once_per_segment),
                'depth_lateral_offset':
                    depth_observation.get('lateral_offset'),
            })

    def p5_depth_observation_for_steering(self):
        """Return the depth observation only where it is allowed to steer.

        The observer's accuracy is per-surface, not global: it tracks ground
        truth to within 5 mm on the bridge deck and is uncorrelated noise with
        a 0.12 m outward bias on the banked ring rails (see the parameter's
        own comment for the numbers).  An empty allow-list keeps every state
        eligible, so the default behaviour is unchanged.

        The observation is still *published* and still counted by
        ``EntryDepthGate`` — this suppresses steering, not observation.
        """
        observation = self.latest_bridge_observation
        allowed = self.p5_deck_lateral_depth_states
        if not allowed or str(self.state) in allowed:
            return observation
        if (isinstance(observation, dict) and observation.get('valid')
                and self.p5_lateral_depth_suppressed_state != str(self.state)):
            # 只在每个状态里报一次：被压掉的是**有效**帧，这正是需要看见的
            # 那一条——否则日志里只剩“source -> odometry”，看不出深度其实在
            # 说话，只是没人听。
            self.p5_lateral_depth_suppressed_state = str(self.state)
            self.get_logger().info(
                f'[{self.state}] depth lateral observation suppressed here '
                f'(offset={observation.get("lateral_offset")}); '
                f'steering on odometry')
            self.p5_evidence_log({
                'event': 'depth_lateral_suppressed',
                'state': str(self.state),
                'lateral_offset': observation.get('lateral_offset'),
                'heading_error': observation.get('heading_error'),
            })
        return None

    def p5_deck_lateral_update(self, name: str):
        """Fold the best available lateral observation into a vy/wz correction.

        Returns ``(vy, wz, vx_scale)``.  When the loop is disabled this is a
        no-op, so the odometry-only behaviour is bit-for-bit unchanged.
        """
        if not self.p5_deck_lateral_enabled:
            return 0.0, 0.0, 1.0

        depth_observation = self.p5_depth_observation_for_steering()
        odometry_observation = self.p5_route_lateral_observation()
        # Anchoring and steering are separate privileges.  A surface where the
        # observer may not hold the wheel can still be a surface where it fixes
        # the line the dead-reckoned fallback holds — but only once, or the
        # anchor becomes a back door through which the same suppressed frames
        # steer anyway.
        suppressed = (depth_observation is None
                      and self.p5_route_lateral_anchor_suppressed_depth)
        self.p5_route_lateral_reanchor(
            self.latest_bridge_observation if suppressed else depth_observation,
            odometry_observation,
            once_per_segment=suppressed)
        observation, source = select_observation(
            depth_observation, odometry_observation)
        if source != self.p5_lateral_source_last:
            self.p5_lateral_source_last = source
            self.get_logger().info(
                f'[{name}] lateral observation source -> {source}')
            self.p5_evidence_log({
                'event': 'lateral_source_change',
                'state': str(self.state),
                'source': source,
            })

        command = self.p5_deck_lateral.update(observation, time.monotonic())
        previous = self.p5_deck_lateral_last
        self.p5_deck_lateral_last = command.state

        # Only log on a state change: at 10 Hz this would otherwise bury the log.
        if command.state != previous:
            payload = {'event': 'deck_lateral', 'state_name': str(self.state)}
            payload.update(command.to_dict())
            self.p5_evidence_log(payload)
            message = (
                f'[{name}] deck lateral {command.state}: '
                f'reason={command.reason}, offset={command.lateral_offset}, '
                f'heading={command.heading_error}, '
                f'vy={command.vy:+.3f}, wz={command.wz:+.3f}')
            # rclpy caches severity per call site, so these must stay two
            # separate statements — one site cannot alternate info/warn.
            if command.state == CONTROL_BLIND:
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)
        # Segment-entry transient: the observation is real but the body is
        # still landing, so steering on it means steering on the landing.
        held_m = self.p5_deck_lateral_engage_after_m
        if held_m > 0.0 and self.p5_route_active():
            travelled = self.p5_route_distance_m()
            if travelled is not None and travelled < held_m:
                self.p5_deck_lateral_debug_last = (
                    f'{source[:4]} entry-hold {travelled:.2f}/{held_m:.2f} m'
                    f' anchor={self.p5_route_lateral_anchor_m:+.3f}')
                return 0.0, 0.0, 1.0

        allowed = self.p5_deck_lateral_centre_first_states
        if allowed and str(self.state) not in allowed and command.vx_scale < 1.0:
            # Only the forward hold is withdrawn; the lateral correction the
            # loop computed is still applied, so this weakens the response to a
            # big offset rather than removing it.
            self.get_logger().info(
                f'[{name}] centre-first vx hold not allowed here '
                f'(offset={command.lateral_offset}); correcting while walking',
                throttle_duration_sec=5.0)
            command.vx_scale = 1.0

        self.p5_deck_lateral_debug_last = (
            f'{source[:4]} off='
            f'{"n/a" if command.lateral_offset is None else f"{command.lateral_offset:+.3f}"}'
            f' hdg='
            f'{"n/a" if command.heading_error is None else f"{command.heading_error:+.3f}"}'
            f' i={command.vy_integral:+.3f}'
            f' anchor={self.p5_route_lateral_anchor_m:+.3f}'
            f' {command.state}/{command.reason}')
        return command.vy, command.wz, command.vx_scale

    def p5_deck_lateral_debug(self) -> str:
        """Return the last lateral-loop inputs, for the segment progress line."""
        return self.p5_deck_lateral_debug_last or 'off'

    def run_odometry_distance_velocity_state(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
        timeout_s: float = 0.0,
    ):
        """Drive a straight until the route model's declared length is covered.

        This is the yellow-free segment runner.  It consumes no image at all —
        only the LCM state-estimator distance — so it is bounded by the route
        window (``max_m``), the per-state timeout, and the route model's
        odometry watchdog, in that order.  It is single-source by construction:
        the segment must declare ``fallback_tier >= TIER_DEAD_RECKONING``, which
        ``RouteModel.validate()`` enforces at start-up.
        """
        name = log_name or self.state
        if self.p5_state_timeout_guard(timeout_s, name):
            return

        segment = self.p5_route_segment
        decision = odometry_exit_reached(
            segment,
            progress_m=self.p5_route_distance_m(),
            odometry_valid=self.p5_route_odom_valid,
        )

        if decision.allow_exit:
            self.get_logger().info(
                f'[{name}] odometry length reached: '
                f'{decision.progress_m:.2f} m >= {segment.expected_m:.2f} m, '
                f'go {next_state}')
            payload = {
                'event': 'route_exit_odometry',
                'state': str(self.state),
                'segment': segment.name,
                'expected_m': float(segment.expected_m),
                'fallback_tier': int(segment.fallback_tier),
                'single_source': True,
            }
            payload.update(decision.to_dict())
            self.p5_evidence_log(payload)
            self.p5_enter_state(next_state)
            return

        if decision.status == GATE_UNAVAILABLE:
            # 没有里程就没有任何段尾证据：停住，不盲走。
            self.p5_route_fault(
                name, 'route_odometry_exit_unavailable', {
                    'segment': segment.name,
                    'reason': decision.reason,
                    'odom_seq': int(self.p5_route_odom_seq),
                    'odom_age_s': self.p5_route_odom_age_s,
                })
            return

        # Odometry gives progress along the route; it never gives offset from
        # the deck centreline.  This is the only loop that closes that.
        deck_vy, deck_wz, vx_scale = self.p5_deck_lateral_update(name)
        vx = vx * vx_scale
        vy = vy + deck_vy
        wz = wz + deck_wz

        if self.p5_route_stall_gate.record(
                vx, decision.progress_m, time.monotonic()):
            if not self.p5_stall_recovery_begin(name, decision.progress_m):
                detail = {'segment': segment.name,
                          'progress_m': float(decision.progress_m)}
                detail.update(self.p5_route_stall_gate.snapshot())
                self.p5_route_fault(name, 'route_stall_fault', detail)
            return

        self.p5_send_velocity_command(
            vx=vx, vy=vy, wz=wz, step_height=step_height,
            roll=roll, pitch=pitch, body_height=body_height,
        )
        # The lateral terms are printed alongside the command because a command
        # on its own cannot be diagnosed: the same vy can be the loop holding a
        # line or the loop chasing a bad estimate, and those need opposite fixes.
        self.get_logger().info(
            f'[{name}] odometry progress {decision.progress_m:.2f}/'
            f'{segment.expected_m:.2f} m, cmd=({vx:.3f},{vy:.3f},{wz:.3f}), '
            f'lat={self.p5_deck_lateral_debug()}',
            throttle_duration_sec=1.0)

    def p5_route_check_turn(self, next_state: str) -> str:
        """Verify a corner's rotation before leaving it; may redirect.

        Returns the state to actually enter: ``next_state`` when the measured
        odometry yaw delta matches the segment's declared rotation,
        ``P5_ROUTE_REALIGN`` for a bounded correction attempt, or
        ``P5_SENSOR_FAULT_HOLD`` when no correction attempt is left.
        """
        segment = self.p5_route_segment
        if segment is None or segment.progress != PROGRESS_YAW:
            return next_state
        if not (self.p5_route_active() and self.p5_route_turn_verify_enabled):
            return next_state
        if next_state in (self.P5_SENSOR_FAULT_HOLD, self.P5_ROUTE_REALIGN,
                          self.P5_FALL_RECOVER):
            return next_state
        if next_state in segment.states:
            return next_state
        if self.p5_route_verified_segment == segment.name:
            # A corner is verified once.  Off-route states (body presets,
            # re-alignment) leave the segment active, and re-checking on the way
            # out of them re-judges a corner that already passed - against a yaw
            # that has since drifted, and while spending the re-alignment budget.
            return next_state

        # In monitor mode the corner is still measured and logged (that is how
        # the windows get calibrated); only the redirect is withheld.
        enforcing = self.p5_route_enforcing(segment)

        if not self.p5_route_odom_valid:
            if not (enforcing and self.p5_route_odometry_required):
                return next_state
            self.get_logger().error(
                f'[P5_ROUTE] cannot verify corner {segment.name}: odometry '
                f'unavailable (seq={self.p5_route_odom_seq}, '
                f'age={self.p5_route_odom_age_s})')
            self.p5_evidence_log({
                'event': 'route_turn_verify_unavailable',
                'state': str(self.state),
                'segment': segment.name,
                'odom_seq': int(self.p5_route_odom_seq),
                'odom_age_s': self.p5_route_odom_age_s,
            })
            return self.P5_SENSOR_FAULT_HOLD

        yaw_delta_deg = self.p5_route_yaw_delta_deg()
        ok, error_deg = verify_yaw(segment, yaw_delta_deg)
        payload = {
            'event': 'route_turn_verify',
            'state': str(self.state),
            'segment': segment.name,
            'measured_yaw_deg': float(yaw_delta_deg),
            'expected_yaw_deg': float(segment.expected_yaw_deg),
            'tolerance_deg': float(segment.yaw_tol_deg),
            'error_deg': float(error_deg),
            'ok': bool(ok),
            'enforced': bool(enforcing),
            'odom_samples': int(self.p5_route_progress.samples),
        }
        self.p5_evidence_log(payload)

        if not enforcing:
            self.get_logger().info(
                f'[P5_ROUTE] corner {segment.name} measured (monitor only): '
                f'measured={yaw_delta_deg:+.1f} deg, '
                f'expected={segment.expected_yaw_deg:+.1f} deg, '
                f'error={error_deg:+.1f} deg, ok={ok}')
            return next_state

        if ok:
            self.p5_route_verified_segment = segment.name
            self.get_logger().info(
                f'[P5_ROUTE] corner {segment.name} verified: '
                f'measured={yaw_delta_deg:+.1f} deg, '
                f'expected={segment.expected_yaw_deg:+.1f} deg, '
                f'error={error_deg:+.1f} deg')
            return next_state

        if self.p5_route_realign_attempts >= self.p5_route_realign_max_attempts:
            self.get_logger().error(
                f'[P5_ROUTE] corner {segment.name} off by {error_deg:+.1f} deg '
                f'and no re-alignment attempt left '
                f'({self.p5_route_realign_attempts}/'
                f'{self.p5_route_realign_max_attempts})')
            self.p5_evidence_log({
                'event': 'route_turn_verify_exhausted',
                'state': str(self.state),
                'segment': segment.name,
                'error_deg': float(error_deg),
                'attempts': int(self.p5_route_realign_attempts),
            })
            return self.P5_SENSOR_FAULT_HOLD

        self.p5_route_realign_attempts += 1
        self.p5_route_realign_segment_name = segment.name
        self.p5_route_realign_resume_state = str(next_state)
        self.get_logger().warn(
            f'[P5_ROUTE] corner {segment.name} off by {error_deg:+.1f} deg, '
            f'enter {self.P5_ROUTE_REALIGN} (attempt '
            f'{self.p5_route_realign_attempts}/'
            f'{self.p5_route_realign_max_attempts}) before {next_state}')
        return self.P5_ROUTE_REALIGN

    def p5_route_on_state_entered(self, state: str):
        """Reset per-segment accumulation when the segment changes.

        Off-route states (fault hold, body presets, re-alignment) keep the
        current accumulator, so a corner's yaw delta survives the detour into
        ``P5_ROUTE_REALIGN`` and stays measured from the corner's entry.
        """
        if not self.p5_route_active():
            return
        segment = self.p5_route_model.segment_for_state(state)
        if segment is None:
            return
        previous = self.p5_route_segment
        self.p5_route_segment = segment
        if previous is not None and previous.name == segment.name:
            return
        if previous is not None:
            self.p5_evidence_log({
                'event': 'route_segment_exit',
                'segment': previous.name,
                'next_segment': segment.name,
                'progress': self.p5_route_progress.snapshot(),
                'odom_valid': bool(self.p5_route_odom_valid),
            })
            # 入段深度门在段收尾时判定；违规由 p5_route_monitor() 在下一拍
            # 走统一的 p5_route_fault() 路径，避免在状态切换中再嵌套切状态。
            if self.p5_route_entry_depth_gate.segment_closed(previous.name):
                self.get_logger().error(
                    f'[P5_ROUTE] entry depth gate tripped: '
                    f'{self.p5_route_entry_depth_gate.snapshot()}')
        self.p5_route_progress.reset(segment.name)
        # The route's heading grid: every segment's true heading is this base
        # plus a whole number of declared quarter turns.  Capturing it once
        # means a sloppy corner cannot drag the reference along with it.
        #
        # It must come from the *course* when the course is known.  Anchoring it
        # to the body's measured yaw at the first segment made the grid a
        # property of how the robot happened to settle on the start line, and
        # that varied by up to 0.46 rad across 20 measured runs.  The grid error
        # goes straight into every odometry-fallback reference line, so a climb
        # of 3.4 m fabricates 3.4*sin(error) of cross-track deviation and the
        # deck-centring loop side-steps the body that far — off a deck whose
        # half-width is 0.245 m.  Measured split (plan item 29): 0 of the 8 runs
        # with a grid error <= 0.03 rad lost the climb; 9 of the 12 above it did.
        if self.p5_route_base_yaw is None:
            snapshot = (self.Odom.snapshot() if self.Odom is not None else None)
            measured = (float(snapshot['rpy'][2])
                        if snapshot is not None and snapshot['seq'] > 0
                        else None)
            if self.p5_route_base_yaw_declared is not None:
                self.p5_route_base_yaw = self.p5_route_base_yaw_declared
                source = 'declared'
            elif self.p5_route_odom_valid and measured is not None:
                self.p5_route_base_yaw = measured
                source = 'measured'
            else:
                source = ''
            if source:
                # 声明值与实测的差就是开局站姿误差本身：它不该进基准线，
                # 但必须可见——摆得太歪时航向环要花时间拧回来。
                error = (None if measured is None
                         else wrap_rad(measured - self.p5_route_base_yaw))
                self.get_logger().info(
                    f'[P5_ROUTE] route heading grid anchored at '
                    f'{self.p5_route_base_yaw:+.4f} rad ({source}) on segment '
                    f'{segment.name}, measured='
                    f'{"n/a" if measured is None else f"{measured:+.4f}"}, '
                    f'error='
                    f'{"n/a" if error is None else f"{error:+.4f}"} rad')
                self.p5_evidence_log({
                    'event': 'route_heading_grid',
                    'state': str(self.state),
                    'segment': segment.name,
                    'base_yaw_rad': float(self.p5_route_base_yaw),
                    'source': source,
                    'measured_yaw_rad': measured,
                    'placement_error_rad': error,
                })
                if error is not None and abs(error) > ROUTE_START_YAW_WARN_RAD:
                    self.get_logger().warn(
                        f'[P5_ROUTE] start placement is {math.degrees(error):+.1f} '
                        f'deg off the declared route heading; the lateral loop '
                        f'will spend time turning onto the route')
        self.p5_route_overrun_handled = False
        self.p5_route_logged_status = ''
        self.p5_route_verified_segment = ''
        # A new segment means a new deck: re-confirm before steering on it,
        # and drop the previous segment's line anchor with it.
        self.p5_route_lateral_anchor_m = 0.0
        self.p5_route_lateral_anchor_source = ''
        self.p5_route_lateral_anchor_seq = None
        self.p5_deck_lateral.reset()
        self.p5_deck_lateral_last = None
        # 新段 = 新入口基准线，横向偏离预算跟着重置。
        self.p5_route_cross_track_gate.reset()
        self.p5_route_stall_gate.reset()
        # The re-alignment budget is per corner, not per stage run: one corner
        # spending it must not silently turn the next corner's first miss into
        # an immediate fault.
        self.p5_route_realign_attempts = 0
        self.p5_evidence_log({
            'event': 'route_segment_enter',
            'segment': segment.name,
            'state': str(state),
            'window_m': [segment.min_m, segment.max_m],
            'expected_m': float(segment.expected_m),
            'expected_yaw_deg': float(segment.expected_yaw_deg),
            'enforcement': segment.enforcement,
            'fallback_tier': int(segment.fallback_tier),
            'speed_cap_mps': float(segment.speed_cap_mps),
            'reference': segment.reference,
        })
        self.get_logger().info(
            f'[P5_ROUTE] enter segment {segment.name} '
            f'({segment.enforcement}, window={segment.window}, '
            f'ref={segment.reference})')

    def p5_fresh_odom_yaw(self):
        """Return the estimator's current yaw, or ``None`` when it is unusable.

        Deliberately independent of the route accumulator: this is needed
        *before* the first segment exists, and a helper that silently returned
        0.0 for a dead stream would turn the body to a fabricated heading.
        """
        if self.Odom is None:
            return None
        snapshot = self.Odom.snapshot()
        if snapshot['seq'] <= 0 or snapshot['rx_monotonic_s'] is None:
            return None
        age = max(0.0, time.monotonic() - float(snapshot['rx_monotonic_s']))
        if age > self.p5_route_odom_max_age_s:
            return None
        return float(snapshot['rpy'][2])

    def p5_start_align_skip(self, reason):
        """Leave the alignment state without turning, saying why."""
        self.get_logger().info(
            f'[P5_START_ALIGN] skipped ({reason}), go {self.P5_STEP_UP}')
        self.p5_evidence_log({
            'event': 'start_align_skipped',
            'state': str(self.state),
            'reason': str(reason),
        })
        self.p5_enter_state(self.P5_STEP_UP)

    def p5_run_start_align(self):
        """Turn the body onto the declared route heading before the climb.

        Measured (plan item 30): ``P5_RECOVERY_STAND`` leaves the body up to
        31 deg off course even from a clean start pose, and nothing between it
        and ``P5_STEP_UP`` corrects that.  A 25 deg heading error walks the body
        1.4 m sideways over the 3.4 m climb, off a deck 0.49 m wide.

        Bounded by construction: it turns in place, it never runs without a
        declared heading and live odometry, and on timeout it proceeds anyway
        rather than converting a bad start into a guaranteed stage stall — the
        error is logged loudly and the route model still gates what follows.
        """
        if not self.p5_start_align_enabled:
            self.p5_start_align_skip('disabled')
            return
        if self.p5_route_base_yaw_declared is None:
            # 没有声明航向就没有"正确方向"可对：实体标定前必须走到这里。
            self.p5_start_align_skip('no_declared_route_heading')
            return

        yaw = self.p5_fresh_odom_yaw()
        if yaw is None:
            self.p5_start_align_skip('odometry_unavailable')
            return

        error = wrap_rad(self.p5_route_base_yaw_declared - yaw)
        # 用 monotonic 而不是 /clock：这是一个安全上界，不能因为 /clock 停了
        # 就永远转下去。与 P5_ROUTE_REALIGN 的计时口径一致。
        elapsed = self.p5_safety_elapsed_s()

        if abs(error) <= self.p5_start_align_tol_rad:
            self.p5_send_velocity_command(
                vx=0.0, vy=0.0, wz=0.0,
                step_height=0.0,
                roll=self.p5_last_cmd_roll,
                pitch=self.p5_last_cmd_pitch,
                body_height=self.p5_last_cmd_body_height,
            )
            self.get_logger().info(
                f'[P5_START_ALIGN] aligned: yaw={yaw:+.4f}, '
                f'error={math.degrees(error):+.1f} deg after {elapsed:.1f}s, '
                f'go {self.P5_STEP_UP}')
            self.p5_evidence_log({
                'event': 'start_align_done',
                'state': str(self.state),
                'yaw_rad': float(yaw),
                'error_deg': float(math.degrees(error)),
                'elapsed_s': float(elapsed),
            })
            self.p5_enter_state(self.P5_STEP_UP)
            return

        if (self.p5_start_align_timeout_s > 0.0
                and elapsed >= self.p5_start_align_timeout_s):
            self.p5_send_velocity_command(
                vx=0.0, vy=0.0, wz=0.0,
                step_height=0.0,
                roll=self.p5_last_cmd_roll,
                pitch=self.p5_last_cmd_pitch,
                body_height=self.p5_last_cmd_body_height,
            )
            self.get_logger().warn(
                f'[P5_START_ALIGN] timeout after {elapsed:.1f}s with '
                f'error={math.degrees(error):+.1f} deg still outstanding; '
                f'starting the climb anyway')
            self.p5_evidence_log({
                'event': 'start_align_timeout',
                'state': str(self.state),
                'yaw_rad': float(yaw),
                'error_deg': float(math.degrees(error)),
                'elapsed_s': float(elapsed),
                'timeout_s': float(self.p5_start_align_timeout_s),
            })
            self.p5_enter_state(self.P5_STEP_UP)
            return

        self.p5_send_velocity_command(
            vx=0.0,
            vy=0.0,
            wz=math.copysign(self.p5_start_align_wz, error),
            step_height=self.p5_start_align_step_height,
            roll=self.p5_last_cmd_roll,
            pitch=self.p5_last_cmd_pitch,
            body_height=self.p5_last_cmd_body_height,
        )
        self.get_logger().info(
            f'[P5_START_ALIGN] turning onto the route: yaw={yaw:+.4f}, '
            f'error={math.degrees(error):+.1f} deg, elapsed={elapsed:.1f}s',
            throttle_duration_sec=1.0)

    def p5_run_route_realign(self):
        """Bounded in-place yaw correction after a failed corner verification."""
        segment = None
        if self.p5_route_model is not None:
            segment = self.p5_route_model.segment_by_name(
                self.p5_route_realign_segment_name)
        if segment is None or not self.p5_route_realign_resume_state:
            self.p5_route_fault(
                'P5_ROUTE_REALIGN', 'route_realign_no_context', {
                    'segment': str(self.p5_route_realign_segment_name),
                    'resume_state': str(self.p5_route_realign_resume_state),
                })
            return

        if not self.p5_route_odom_valid:
            self.p5_route_fault(
                'P5_ROUTE_REALIGN', 'route_realign_odometry_lost', {
                    'segment': segment.name,
                    'odom_seq': int(self.p5_route_odom_seq),
                    'odom_age_s': self.p5_route_odom_age_s,
                })
            return

        yaw_delta_deg = self.p5_route_yaw_delta_deg()
        ok, error_deg = verify_yaw(segment, yaw_delta_deg)
        if ok:
            resume_state = self.p5_route_realign_resume_state
            self.get_logger().info(
                f'[P5_ROUTE_REALIGN] corner {segment.name} realigned: '
                f'measured={yaw_delta_deg:+.1f} deg, '
                f'error={error_deg:+.1f} deg, resume {resume_state}')
            self.p5_evidence_log({
                'event': 'route_realign_done',
                'segment': segment.name,
                'measured_yaw_deg': float(yaw_delta_deg),
                'error_deg': float(error_deg),
                'elapsed_s': float(self.p5_safety_elapsed_s()),
                'resume_state': str(resume_state),
            })
            self.p5_send_velocity_command(
                vx=0.0, vy=0.0, wz=0.0, step_height=0.0,
                roll=self.p5_last_cmd_roll,
                pitch=self.p5_last_cmd_pitch,
                body_height=self.p5_last_cmd_body_height,
            )
            self.p5_route_realign_resume_state = ''
            self.p5_route_realign_segment_name = ''
            # The corner was just measured inside tolerance; entering the resume
            # state must not judge it a second time against a drifting yaw.
            self.p5_route_verified_segment = segment.name
            self.p5_enter_state(resume_state)
            return

        elapsed = self.p5_safety_elapsed_s()
        if self.p5_route_realign_timeout_s > 0.0 and elapsed >= self.p5_route_realign_timeout_s:
            self.p5_route_fault(
                'P5_ROUTE_REALIGN', 'route_realign_timeout', {
                    'segment': segment.name,
                    'measured_yaw_deg': float(yaw_delta_deg),
                    'error_deg': float(error_deg),
                    'elapsed_s': float(elapsed),
                    'timeout_s': float(self.p5_route_realign_timeout_s),
                })
            return

        wz = math.copysign(self.p5_route_realign_wz, error_deg)
        self.p5_send_velocity_command(
            vx=0.0,
            vy=0.0,
            wz=wz,
            step_height=self.p5_route_realign_step_height,
            roll=self.p5_last_cmd_roll,
            pitch=self.p5_last_cmd_pitch,
            body_height=self.p5_last_cmd_body_height,
        )
        self.get_logger().warn(
            f'[P5_ROUTE_REALIGN] correcting corner {segment.name}: '
            f'measured={yaw_delta_deg:+.1f} deg, error={error_deg:+.1f} deg, '
            f'wz={wz:+.2f}, elapsed={elapsed:.1f}/'
            f'{self.p5_route_realign_timeout_s:.1f}s',
            throttle_duration_sec=0.5)

    # ============================================================
    # 图像回调
    # ============================================================
    def p5_rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[P5_RGB] cv_bridge convert failed: {e}')
            return

        self.record_valid_rgb_frame(msg, frame)
        self.latest_frame_seq = self.latest_rgb_seq
        self.latest_p5_yellow_result = self.detect_p5_yellow_stop_line(frame)

        if self.p5_show_debug_vis:
            self.show_p5_debug_window(frame)

    def p5_run_bridge_observer(self, depth_img, msg: Image):
        """Evaluate one synchronized-enough D435 frame for read-only evidence."""
        stamp_s = None
        try:
            depth_m = depth_image_to_meters(depth_img, msg.encoding)
            stamp_s = None
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp_s = (
                    float(msg.header.stamp.sec)
                    + float(msg.header.stamp.nanosec) * 1e-9
                )

            imu_age_s = None
            if self.p5_last_imu_monotonic_s is not None:
                imu_age_s = max(
                    0.0, time.monotonic() - self.p5_last_imu_monotonic_s)
            imu_skew_s = None
            if (
                stamp_s is not None and stamp_s > 0.0
                and self.p5_last_imu_stamp_s is not None
                and self.p5_last_imu_stamp_s > 0.0
            ):
                imu_skew_s = abs(stamp_s - self.p5_last_imu_stamp_s)

            imu_reason = None
            if imu_age_s is None:
                imu_reason = 'imu_missing'
            elif (
                stamp_s is None or stamp_s <= 0.0
                or self.p5_last_imu_stamp_s is None
                or self.p5_last_imu_stamp_s <= 0.0
            ):
                imu_reason = 'sensor_timestamp_missing'
            elif imu_age_s > self.p5_bridge_max_imu_age_s:
                imu_reason = 'imu_stale'
            elif imu_skew_s > self.p5_bridge_max_imu_depth_skew_s:
                imu_reason = 'imu_depth_unsynchronized'

            if imu_reason is not None:
                obs = invalid_bridge_observation(
                    {
                        'frame_seq': int(self.latest_depth_seq),
                        'stamp_s': stamp_s,
                    },
                    imu_reason,
                )
                intrinsics_source = 'not_evaluated'
            else:
                intrinsics = self.p5_bridge_intrinsics
                shape = depth_m.shape[:2]
                if intrinsics is not None and self.p5_bridge_intrinsics_shape == shape:
                    intrinsics_source = 'camera_info'
                else:
                    height, width = shape
                    intrinsics = CameraIntrinsics.from_horizontal_fov(
                        width, height, self.p5_depth_horizontal_fov)
                    intrinsics_source = 'fov_fallback'

                control_point = rotation_gravity_from_body(
                    self.p5_imu_roll, self.p5_imu_pitch
                ) @ np.array([
                    -self.p5_depth_camera_mount_x,
                    -self.p5_depth_camera_mount_y,
                    -self.p5_depth_camera_mount_z,
                ])
                obs = bridge_observation(
                    depth_m,
                    intrinsics,
                    camera_roll=(
                        self.p5_imu_roll + self.p5_depth_camera_mount_roll),
                    camera_pitch=(
                        self.p5_imu_pitch + self.p5_depth_camera_mount_pitch),
                    config=self.p5_bridge_config,
                    control_point_x=float(control_point[0]),
                    control_point_y=float(control_point[1]),
                    frame_seq=self.latest_depth_seq,
                    stamp_s=stamp_s,
                )

            obs['imu_age_s'] = imu_age_s
            obs['imu_depth_skew_s'] = imu_skew_s
            obs['intrinsics_source'] = intrinsics_source
            obs['camera_info_frame_id'] = self.p5_bridge_camera_info_frame_id
            obs['reference_point'] = 'body_origin'
            obs['forward_distance_reference'] = 'body_origin'
            obs['control_use'] = (
                'deck_lateral_hold' if self.p5_deck_lateral_enabled
                else 'read_only')
            self.latest_bridge_observation = obs
            self.p5_route_entry_depth_gate.record_frame(
                bool(obs.get('valid')),
                self.p5_route_segment.name
                if self.p5_route_segment is not None else '')
            self.p5_evidence_log({
                'event': 'bridge_observation',
                'observation': obs,
            })
            self.get_logger().info(
                f'[P5_BRIDGE_OBSERVER] valid={obs.get("valid")}, '
                f'reason={obs.get("reason")}, seq={self.latest_depth_seq}, '
                f'offset={obs.get("lateral_offset")}, '
                f'heading={obs.get("heading_error")}, '
                f'forward_dropoff={obs.get("d_forward_dropoff")}',
                throttle_duration_sec=1.0,
            )
        except (TypeError, ValueError, FloatingPointError) as e:
            obs = invalid_bridge_observation(
                {
                    'frame_seq': int(self.latest_depth_seq),
                    'stamp_s': stamp_s,
                },
                f'exception:{type(e).__name__}',
            )
            obs['control_use'] = (
                'deck_lateral_hold' if self.p5_deck_lateral_enabled
                else 'read_only')
            obs['reference_point'] = 'body_origin'
            obs['forward_distance_reference'] = 'body_origin'
            self.latest_bridge_observation = obs
            self.get_logger().error(f'[P5_BRIDGE_OBSERVER] rejected frame: {e}')

    def p5_reset_body_ramp_tick(self):
        """Roll the body level in steps instead of in one command.

        The straights hold ``p5_right_slope_roll`` (-0.6 rad) to lean into the
        rail's camber; ``P5_RESET_BODY`` puts the body level again before the
        corner-4 jump.  Doing that in one command moves the centre of mass
        across the whole camber at once.  On ``race.world`` the border ledge at
        the rail edge absorbs that; in ``race_physical`` there is no ledge, and
        ground truth 2026-08-16 shows the body going roll -0.47 -> +0.19 ->
        +0.55 over two seconds and sliding off the low side of straight_3
        before the jump was ever commanded.

        Zero-velocity commands are re-sent every tick so the interpolated
        baseline is actually applied: ``set_body_roll_height`` only refreshes
        while the controller is already in locomotion, and a standing robot
        would otherwise sit at the old pose until the jump.
        """
        span = self.p5_reset_body_ramp_s
        fraction = min(1.0, self.p5_state_elapsed_s() / span) if span > 0 else 1.0
        roll = (self.p5_reset_body_roll_from
                + (self.p5_reset_roll - self.p5_reset_body_roll_from) * fraction)
        self.p5_send_velocity_command(
            vx=0.0, vy=0.0, wz=0.0,
            step_height=self.p5_right_slope_3_step_height,
            roll=roll, pitch=0.0, body_height=self.p5_reset_height,
        )
        self.get_logger().info(
            f'[P5_RESET_BODY] rolling level {self.p5_reset_body_roll_from:+.3f} '
            f'-> {self.p5_reset_roll:+.3f}: now {roll:+.3f} '
            f'({fraction * 100:.0f}%)',
            throttle_duration_sec=0.5)

    def set_body_roll_height(self, roll: float, height: float):
        """设定身体 roll / 高度基线，随控制命令直接下发（不再改控制器 YAML）。

        新控制器从 rpy_des[0]/pos_des[2] 读 roll/身体高度（实体机上没有
        YAML 通道），所以这里只更新姿态基线；之后每条速度命令都会带上它。
        如果机器人正处于运动控制（mode 11），立即用新姿态重发当前命令，
        让姿态在等待窗口内就开始变化，与旧的 YAML 生效时机一致。
        """
        self.p5_last_cmd_roll = float(roll)
        self.p5_last_cmd_body_height = float(height)

        refresh_sent = self.msg.mode == 11
        if refresh_sent:
            vel = list(self.msg.vel_des)
            step = list(self.msg.step_height)
            self.Ctrl.move(
                float(vel[0]), float(vel[1]), float(vel[2]),
                step_height=float(max(step) if step else 0.05),
                roll=float(roll),
                pitch=float(self.p5_last_cmd_pitch),
                body_height=float(height),
                legacy_gait_id=3,
            )

        self.get_logger().info(
            f'[P5_BODY] set body pose baseline: '
            f'roll={roll:.3f}, height={height:.3f}, '
            f'refresh_sent={refresh_sent}'
        )

    # ============================================================
    # 控制命令
    # ============================================================
    def p5_send_stop_command(self):
        # Semantic STOP: on the physical robot this is zero DATA + SERVO_END,
        # not RecoveryStand(111).  The simulator adapter intentionally keeps
        # the old mode=12 behaviour so Gazebo regression remains unchanged.
        if self.Ctrl is None:
            return
        # Fire-and-forget, as before the backend migration: this runs inside
        # the Stage-5 control loop, which polls action completion itself.
        self.Ctrl.stop_motion(wait_finish=False)
        self.get_logger().info('[P5_CMD] STOP sent', throttle_duration_sec=1.0)

    def p5_send_velocity_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: float = 0.0,
        body_height: Optional[float] = None,
    ):
        # None = 维持上一次命令的身体姿态。旧控制器里 roll/高度锁存在
        # YAML 参数中，不随命令回零；新控制器姿态逐命令下发，这里补上
        # 同样的锁存语义，避免省缺参数的调用（如故障停车）把姿态打回默认。
        if roll is None:
            roll = self.p5_last_cmd_roll
        if pitch is None:
            pitch = self.p5_last_cmd_pitch
        if body_height is None:
            body_height = self.p5_last_cmd_body_height

        # 路线段限速：只按段表上限等比缩小平面速度，不改变方向，零命令仍为零。
        cap = self.p5_route_speed_cap_mps()
        vx, vy, capped = clamp_speed(cap, vx, vy)
        if capped:
            self.get_logger().warn(
                f'[P5_ROUTE] segment {self.p5_route_segment.name} speed cap '
                f'{cap:.2f} m/s applied: cmd=({vx:.3f},{vy:.3f})',
                throttle_duration_sec=2.0)

        self.p5_last_cmd_roll = float(roll)
        self.p5_last_cmd_pitch = float(pitch)
        self.p5_last_cmd_body_height = float(body_height)

        # Keep the reusable legacy message in sync because a few Stage-5
        # helpers still inspect it, but send motion through the semantic API.
        self.msg.mode = 11
        self.msg.gait_id = 3
        self.msg.vel_des = [float(vx), float(vy), float(wz)]
        self.msg.step_height = [float(step_height), float(step_height)]
        self.msg.rpy_des = [float(roll), float(pitch), float(yaw)]
        self.msg.pos_des = [0.0, 0.0, float(body_height)]

        self.Ctrl.move(
            float(vx), float(vy), float(wz),
            step_height=float(step_height),
            roll=float(roll), pitch=float(pitch), yaw=float(yaw),
            body_height=float(body_height),
            legacy_gait_id=3,
        )

    def p5_send_action_once(self, mode: int, gait_id: int, response_barrier=False):
        self.msg.mode = int(mode)
        self.msg.gait_id = int(gait_id)
        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.msg.step_height = [0.0, 0.0]
        self.msg.rpy_des = [0.0, 0.0, 0.0]

        self.p5_inc_life_count()
        barrier = None
        if response_barrier:
            barrier = self.Ctrl.Send_cmd_with_response_barrier(self.msg)
        else:
            self.Ctrl.Send_cmd(self.msg)

        self.get_logger().info(
            f'[P5_ACTION] send once: '
            f'mode={mode}, gait_id={gait_id}, life_count={self.msg.life_count}'
        )
        return barrier

    # ============================================================
    # 严格前方黄线检测
    # ============================================================
    def is_p5_front_horizontal_yellow_line(self, cnt, roi_shape) -> bool:
        _, roi_w = roi_shape[:2]

        area = cv2.contourArea(cnt)
        if area < self.p5_yellow_min_contour_area:
            return False

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh <= 0:
            return False

        wh_ratio = bw / float(bh)
        if wh_ratio < self.p5_yellow_min_width_height_ratio:
            return False

        width_ratio = bw / float(max(roi_w, 1))
        if width_ratio < self.p5_yellow_min_width_ratio:
            return False

        cx = x + bw / 2.0
        roi_cx = roi_w / 2.0
        center_offset_ratio = abs(cx - roi_cx) / float(max(roi_w, 1))
        if center_offset_ratio > self.p5_yellow_center_tolerance_ratio:
            return False

        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), angle = rect

        if rw < rh:
            tilt_deg = abs(angle - 90.0)
        else:
            tilt_deg = abs(angle)

        if tilt_deg > 45.0:
            tilt_deg = abs(90.0 - tilt_deg)

        if tilt_deg > self.p5_yellow_max_tilt_deg:
            return False

        return True

    def get_signed_p5_yellow_line_angle_deg(self, cnt) -> float:
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

    def make_p5_yellow_mask_for_debug(self, frame):
        h, w = frame.shape[:2]

        roi_top = int(h * self.p5_yellow_roi_top_ratio)
        roi_left = int(w * self.p5_yellow_roi_left_ratio)
        roi_right = int(w * self.p5_yellow_roi_right_ratio)

        roi_top = max(0, min(h - 1, roi_top))
        roi_left = max(0, min(w - 1, roi_left))
        roi_right = max(roi_left + 1, min(w, roi_right))

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, (roi_left, roi_top, roi_right, h)

    def detect_p5_yellow_stop_line(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        roi_top = int(h * self.p5_yellow_roi_top_ratio)
        roi_left = int(w * self.p5_yellow_roi_left_ratio)
        roi_right = int(w * self.p5_yellow_roi_right_ratio)

        roi_top = max(0, min(h - 1, roi_top))
        roi_left = max(0, min(w - 1, roi_left))
        roi_right = max(roi_left + 1, min(w, roi_right))

        roi = frame[roi_top:h, roi_left:roi_right]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.latest_p5_yellow_mask = mask
        self.latest_p5_yellow_roi = (roi_left, roi_top, roi_right, h)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.p5_yellow_min_contour_area:
                continue

            if not self.is_p5_front_horizontal_yellow_line(cnt, roi.shape):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # 严格版：优先选最靠下的前方横线
            score = y + bh

            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return {
                'has_line': False,
                'line_bottom_y': None,
                'line_center': None,
                'img_shape': (h, w),
                'angle_deg': None,
                'abs_tilt_deg': None,
                'bbox': None,
                'width_ratio': None,
                'wh_ratio': None,
            }

        x, y, bw, bh = cv2.boundingRect(best_contour)

        line_bottom_y = roi_top + y + bh
        cx = roi_left + x + bw // 2
        cy = roi_top + y + bh // 2

        angle_deg = self.get_signed_p5_yellow_line_angle_deg(best_contour)
        abs_tilt_deg = abs(angle_deg)

        width_ratio = bw / float(max(roi_right - roi_left, 1))
        wh_ratio = bw / float(max(bh, 1))

        return {
            'has_line': True,
            'line_bottom_y': int(line_bottom_y),
            'line_center': (int(cx), int(cy)),
            'img_shape': (h, w),
            'angle_deg': float(angle_deg),
            'abs_tilt_deg': float(abs_tilt_deg),
            'bbox': (
                int(roi_left + x),
                int(roi_top + y),
                int(roi_left + x + bw),
                int(roi_top + y + bh)
            ),
            'width_ratio': float(width_ratio),
            'wh_ratio': float(wh_ratio),
        }

    def p5_yellow_reached_bottom(self, yellow_result: dict) -> bool:
        if yellow_result is None:
            self.p5_yellow_stop_counter = 0
            return False

        if yellow_result.get('img_shape') is None or not yellow_result.get('has_line', False):
            self.p5_yellow_stop_counter = 0
            return False

        h, _ = yellow_result['img_shape']
        stop_y_threshold = int(h * self.p5_yellow_stop_line_y_ratio)

        bottom_y = yellow_result.get('line_bottom_y')

        if bottom_y is not None and int(bottom_y) >= stop_y_threshold:
            self.p5_yellow_stop_counter += 1
        else:
            self.p5_yellow_stop_counter = 0

        self.get_logger().info(
            f'[P5_YELLOW] bottom={bottom_y}, '
            f'threshold={stop_y_threshold}, '
            f'counter={self.p5_yellow_stop_counter}/{self.p5_yellow_stop_confirm_count}',
            throttle_duration_sec=0.3
        )

        return self.p5_yellow_stop_counter >= self.p5_yellow_stop_confirm_count

    def compute_p5_yellow_angle_align_wz(self, yellow_result: dict) -> float:
        if not self.p5_yellow_angle_align_enabled:
            return 0.0

        if yellow_result is None or not yellow_result.get('has_line', False):
            return 0.0

        angle_deg = yellow_result.get('angle_deg', None)
        if angle_deg is None:
            return 0.0

        angle_deg = float(angle_deg)

        if abs(angle_deg) <= self.p5_yellow_angle_align_deadband_deg:
            return 0.0

        # 符号沿用前面赛段严格横线角度矫正的约定：
        # angle > 0 给负 wz，angle < 0 给正 wz。
        # 如果实测越修越歪，就把这里正负号对调。
        if angle_deg > 0.0:
            wz = -abs(self.p5_yellow_angle_align_fixed_wz)
        else:
            wz = abs(self.p5_yellow_angle_align_fixed_wz)

        self.get_logger().info(
            f'[P5_YELLOW_ALIGN] angle={angle_deg:.2f}deg, '
            f'deadband={self.p5_yellow_angle_align_deadband_deg:.2f}, '
            f'wz={wz:.3f}',
            throttle_duration_sec=0.3
        )

        return wz



    # ============================================================
    # P5_RIGHT_SLOPE：中间区域黄色消失检测
    # ============================================================
    def detect_p5_center_yellow_presence(self, frame: np.ndarray) -> dict:
        """
        右斜坡阶段专用：检测图像中间区域是否还有黄色。

        右斜坡上相机画面倾斜、抖动明显，所以这里不再要求黄线是横向、
        不计算角度，也不检查宽高比。只看中间 ROI 内黄色像素数量/比例。

        has_yellow=True  表示中间区域还有黄色，继续走；
        has_yellow=False 表示中间区域基本没有黄色，可以累计 absent counter。
        """
        h, w = frame.shape[:2]

        x1 = int(w * self.p5_center_yellow_roi_x_min)
        x2 = int(w * self.p5_center_yellow_roi_x_max)
        y1 = int(h * self.p5_center_yellow_roi_y_min)
        y2 = int(h * self.p5_center_yellow_roi_y_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        yellow_pixels = int(cv2.countNonZero(mask))
        roi_pixels = int(mask.shape[0] * mask.shape[1])
        yellow_ratio = yellow_pixels / float(max(roi_pixels, 1))

        has_yellow = (
            yellow_pixels >= self.p5_center_yellow_min_pixels and
            yellow_ratio >= self.p5_center_yellow_min_ratio
        )

        bbox = None
        if yellow_pixels > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 5.0]
            if valid_contours:
                all_pts = np.vstack(valid_contours)
                rx, ry, rw, rh = cv2.boundingRect(all_pts)
                bbox = (
                    int(x1 + rx),
                    int(y1 + ry),
                    int(x1 + rx + rw),
                    int(y1 + ry + rh)
                )

        if has_yellow:
            reason = 'center_yellow_present'
        else:
            reason = 'center_yellow_absent'

        return {
            'has_yellow': bool(has_yellow),
            'yellow_pixels': int(yellow_pixels),
            'roi_pixels': int(roi_pixels),
            'yellow_ratio': float(yellow_ratio),
            'bbox': bbox,
            'roi': (int(x1), int(y1), int(x2), int(y2)),
            'img_shape': (h, w),
            'reason': reason,
        }

    # ============================================================
    # P5_UP_SLOPE：右侧赛道黄线消失检测
    # ============================================================
    def detect_p5_right_side_yellow_line(self, frame: np.ndarray) -> dict:
        """
        P5_UP_SLOPE 专用：检测右侧赛道旁边的黄色边线。

        这里不做很严格的横线/竖线形状判断，只在右侧 ROI 内找黄色区域。
        但是有一个关键限制：检测到的黄色区域 bottom_ratio 必须接近图像底部，
        才算“右侧赛道黄线还存在”。

        这样可以避免上坡末尾时，把前方右侧黄线误认为当前右侧赛道黄线。
        """
        h, w = frame.shape[:2]

        x1 = int(w * self.p5_right_side_yellow_roi_x_min)
        x2 = int(w * self.p5_right_side_yellow_roi_x_max)
        y1 = int(h * self.p5_right_side_yellow_roi_y_min)
        y2 = int(h * self.p5_right_side_yellow_roi_y_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.p5_right_side_yellow_min_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)

            if rw < self.p5_right_side_yellow_min_width:
                continue
            if rh < self.p5_right_side_yellow_min_height:
                continue

            bx1 = x1 + rx
            by1 = y1 + ry
            bx2 = bx1 + rw
            by2 = by1 + rh
            cx = bx1 + rw // 2
            cy = by1 + rh // 2

            bottom_ratio = by2 / float(max(h, 1))

            candidates.append({
                'bbox': (int(bx1), int(by1), int(bx2), int(by2)),
                'center': (int(cx), int(cy)),
                'area': float(area),
                'height': int(rh),
                'width': int(rw),
                'bottom_y': int(by2),
                'bottom_ratio': float(bottom_ratio),
                # 右侧赛道黄线应贴近图像底部，所以优先选 bottom 最大的黄色区域
                'score': float(5.0 * by2 + 0.01 * area + rh),
            })

        if len(candidates) == 0:
            return {
                'has_line': False,
                'valid_bottom': False,
                'bbox': None,
                'center': None,
                'bottom_y': None,
                'bottom_ratio': None,
                'area': None,
                'height': None,
                'width': None,
                'roi': (int(x1), int(y1), int(x2), int(y2)),
                'candidates': candidates,
                'img_shape': (h, w),
                'reason': 'no_candidate',
            }

        best = max(candidates, key=lambda c: c['score'])
        valid_bottom = best['bottom_ratio'] >= self.p5_right_side_yellow_bottom_valid_ratio

        if not valid_bottom:
            # 检测到了黄线，但最低点不在图像底部附近。
            # 很可能是前方右侧黄线，不算右侧赛道黄线还存在。
            return {
                'has_line': False,
                'valid_bottom': False,
                'bbox': best['bbox'],
                'center': best['center'],
                'bottom_y': best['bottom_y'],
                'bottom_ratio': best['bottom_ratio'],
                'area': best['area'],
                'height': best['height'],
                'width': best['width'],
                'roi': (int(x1), int(y1), int(x2), int(y2)),
                'candidates': candidates,
                'img_shape': (h, w),
                'reason': 'bottom_not_near_image_bottom',
            }

        return {
            'has_line': True,
            'valid_bottom': True,
            'bbox': best['bbox'],
            'center': best['center'],
            'bottom_y': best['bottom_y'],
            'bottom_ratio': best['bottom_ratio'],
            'area': best['area'],
            'height': best['height'],
            'width': best['width'],
            'roi': (int(x1), int(y1), int(x2), int(y2)),
            'candidates': candidates,
            'img_shape': (h, w),
            'reason': 'valid_right_side_line',
        }


    # ============================================================
    # P5_UP_SLOPE：左右内侧黄线边缘检测与矫正
    # ============================================================
    def clamp_p5_roi(self, roi, w: int, h: int):
        x1, y1, x2, y2 = roi
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(y1 + 1, min(h, int(y2)))
        return x1, y1, x2, y2

    def get_p5_inner_edge_rois(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        y1 = int(h * self.p5_inner_edge_roi_y_min)
        y2 = int(h * self.p5_inner_edge_roi_y_max)

        left_roi = (
            int(w * self.p5_inner_edge_left_roi_x_min),
            y1,
            int(w * self.p5_inner_edge_left_roi_x_max),
            y2,
        )
        right_roi = (
            int(w * self.p5_inner_edge_right_roi_x_min),
            y1,
            int(w * self.p5_inner_edge_right_roi_x_max),
            y2,
        )

        return (
            self.clamp_p5_roi(left_roi, w, h),
            self.clamp_p5_roi(right_roi, w, h),
        )

    def make_p5_inner_edge_yellow_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.p5_yellow_h_min, self.p5_yellow_s_min, self.p5_yellow_v_min],
            dtype=np.uint8
        )
        upper_yellow = np.array(
            [self.p5_yellow_h_max, self.p5_yellow_s_max, self.p5_yellow_v_max],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def keep_p5_bottom_connected_segment(self, edge_points):
        """
        只保留从图像底部往上的连续边缘段。
        如果相邻两个边缘点的 y 间隔过大，认为上方远处黄线与近处黄线断开。
        """
        if not edge_points:
            return []

        if not self.p5_inner_edge_use_bottom_connected_segment:
            return sorted(edge_points, key=lambda p: p[1])

        max_y_gap = max(1, int(self.p5_inner_edge_max_y_gap))

        pts_bottom_to_top = sorted(edge_points, key=lambda p: p[1], reverse=True)

        kept = [pts_bottom_to_top[0]]
        prev_y = pts_bottom_to_top[0][1]

        for x, y in pts_bottom_to_top[1:]:
            y_gap = abs(prev_y - y)
            if y_gap > max_y_gap:
                break

            kept.append((x, y))
            prev_y = y

        return sorted(kept, key=lambda p: p[1])

    def mean_p5_x_in_y_band(self, points, y_low: float, y_high: float):
        if not points:
            return None, None, 0

        pts = np.array(points, dtype=np.float32)
        band = pts[(pts[:, 1] >= y_low) & (pts[:, 1] <= y_high)]

        if len(band) == 0:
            return None, None, 0

        return float(np.mean(band[:, 0])), float(np.mean(band[:, 1])), int(len(band))

    def extract_p5_inner_edge_points(
        self,
        mask: np.ndarray,
        roi,
        side: str,
        image_h: int,
    ) -> dict:
        """
        side='left'  ：左侧黄线取每一行最右侧黄色像素，作为左侧赛道内侧边缘。
        side='right' ：右侧黄线取每一行最左侧黄色像素，作为右侧赛道内侧边缘。
        """
        x1, y1, x2, y2 = roi
        roi_mask = mask[y1:y2, x1:x2]

        row_step = max(1, int(self.p5_inner_edge_row_step))

        edge_points = []
        for local_y in range(0, roi_mask.shape[0], row_step):
            row = roi_mask[local_y, :]
            xs = np.where(row > 0)[0]

            if xs.size == 0:
                continue

            if side == 'left':
                local_x = int(np.max(xs))
            else:
                local_x = int(np.min(xs))

            edge_points.append((int(x1 + local_x), int(y1 + local_y)))

        if len(edge_points) == 0:
            return {
                'valid': False,
                'reason': 'no_yellow_points',
                'points': [],
                'top_x': None,
                'bottom_x': None,
                'top_y': None,
                'bottom_y': None,
                'point_count': 0,
                'raw_point_count': 0,
                'y_span': 0.0,
                'bottom_ratio': 0.0,
                'x_std': 0.0,
            }

        raw_point_count = len(edge_points)
        edge_points = self.keep_p5_bottom_connected_segment(edge_points)

        if len(edge_points) == 0:
            return {
                'valid': False,
                'reason': 'no_bottom_connected_segment',
                'points': [],
                'top_x': None,
                'bottom_x': None,
                'top_y': None,
                'bottom_y': None,
                'point_count': 0,
                'raw_point_count': int(raw_point_count),
                'y_span': 0.0,
                'bottom_ratio': 0.0,
                'x_std': 0.0,
            }

        pts = np.array(edge_points, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        point_count = len(pts)
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        y_span = y_max - y_min
        bottom_ratio = y_max / float(max(image_h, 1))
        x_std = float(np.std(xs))

        band_ratio = max(0.05, min(0.50, float(self.p5_inner_edge_top_bottom_band_ratio)))
        top_thr = y_min + band_ratio * max(y_span, 1.0)
        bottom_thr = y_max - band_ratio * max(y_span, 1.0)

        top_band = pts[pts[:, 1] <= top_thr]
        bottom_band = pts[pts[:, 1] >= bottom_thr]

        if len(top_band) == 0:
            top_band = pts
        if len(bottom_band) == 0:
            bottom_band = pts

        top_x = float(np.mean(top_band[:, 0]))
        bottom_x = float(np.mean(bottom_band[:, 0]))

        fail_reasons = []

        if point_count < self.p5_inner_edge_min_points:
            fail_reasons.append(f'points<{self.p5_inner_edge_min_points}')

        if y_span < self.p5_inner_edge_min_y_span:
            fail_reasons.append(f'y_span<{self.p5_inner_edge_min_y_span:.0f}')

        if bottom_ratio < self.p5_inner_edge_bottom_min_ratio:
            fail_reasons.append(f'bottom<{self.p5_inner_edge_bottom_min_ratio:.2f}')

        if x_std > self.p5_inner_edge_x_std_max:
            fail_reasons.append(f'x_std>{self.p5_inner_edge_x_std_max:.0f}')

        valid = len(fail_reasons) == 0

        return {
            'valid': bool(valid),
            'reason': 'ok' if valid else ','.join(fail_reasons),
            'points': edge_points,
            'top_x': top_x,
            'bottom_x': bottom_x,
            'top_y': y_min,
            'bottom_y': y_max,
            'point_count': int(point_count),
            'raw_point_count': int(raw_point_count),
            'y_span': float(y_span),
            'bottom_ratio': float(bottom_ratio),
            'x_std': float(x_std),
        }

    def detect_p5_inner_edges(self, frame: np.ndarray) -> dict:
        """
        P5_UP_SLOPE 专用：检测左右两侧赛道内侧黄线边缘。
        输出 center_error 和 heading_error，供上坡过程的 vy / wz 修正使用。
        """
        h, w = frame.shape[:2]
        mask = self.make_p5_inner_edge_yellow_mask(frame)

        left_roi, right_roi = self.get_p5_inner_edge_rois(frame)

        left_edge = self.extract_p5_inner_edge_points(
            mask=mask,
            roi=left_roi,
            side='left',
            image_h=h,
        )
        right_edge = self.extract_p5_inner_edge_points(
            mask=mask,
            roi=right_roi,
            side='right',
            image_h=h,
        )

        has_left = bool(left_edge['valid'])
        has_right = bool(right_edge['valid'])
        has_both = has_left and has_right

        result = {
            'mask': mask,
            'left_roi': left_roi,
            'right_roi': right_roi,
            'left_edge': left_edge,
            'right_edge': right_edge,
            'has_left': has_left,
            'has_right': has_right,
            'has_both': has_both,
            'bottom_center_x': None,
            'top_center_x': None,
            'bottom_center_y': None,
            'top_center_y': None,
            'center_error': None,
            'heading_error': None,
            'common_top_y': None,
            'common_bottom_y': None,
            'common_y_span': None,
            'common_valid': False,
            'common_reason': 'need_both_edges',
            'cmd_vy_correction': 0.0,
            'cmd_wz_correction': 0.0,
        }

        if has_both:
            common_top_y = max(float(left_edge['top_y']), float(right_edge['top_y']))
            common_bottom_y = min(float(left_edge['bottom_y']), float(right_edge['bottom_y']))
            common_y_span = common_bottom_y - common_top_y

            result['common_top_y'] = float(common_top_y)
            result['common_bottom_y'] = float(common_bottom_y)
            result['common_y_span'] = float(common_y_span)

            if common_y_span >= self.p5_inner_edge_min_common_y_span:
                band_ratio = max(0.05, min(0.50, float(self.p5_inner_edge_top_bottom_band_ratio)))

                top_band_high = common_top_y + band_ratio * common_y_span
                bottom_band_low = common_bottom_y - band_ratio * common_y_span

                left_top_x, left_top_y, left_top_n = self.mean_p5_x_in_y_band(
                    left_edge['points'], common_top_y, top_band_high
                )
                right_top_x, right_top_y, right_top_n = self.mean_p5_x_in_y_band(
                    right_edge['points'], common_top_y, top_band_high
                )
                left_bottom_x, left_bottom_y, left_bottom_n = self.mean_p5_x_in_y_band(
                    left_edge['points'], bottom_band_low, common_bottom_y
                )
                right_bottom_x, right_bottom_y, right_bottom_n = self.mean_p5_x_in_y_band(
                    right_edge['points'], bottom_band_low, common_bottom_y
                )

                enough_band_points = (
                    left_top_n > 0 and right_top_n > 0 and
                    left_bottom_n > 0 and right_bottom_n > 0
                )

                if enough_band_points:
                    bottom_center_x = (left_bottom_x + right_bottom_x) / 2.0
                    top_center_x = (left_top_x + right_top_x) / 2.0
                    bottom_center_y = (left_bottom_y + right_bottom_y) / 2.0
                    top_center_y = (left_top_y + right_top_y) / 2.0

                    image_center_x = w / 2.0

                    result['bottom_center_x'] = float(bottom_center_x)
                    result['top_center_x'] = float(top_center_x)
                    result['bottom_center_y'] = float(bottom_center_y)
                    result['top_center_y'] = float(top_center_y)
                    result['center_error'] = float(bottom_center_x - image_center_x)
                    result['heading_error'] = float(top_center_x - bottom_center_x)
                    result['common_valid'] = True
                    result['common_reason'] = 'ok'

                    result['left_common_top'] = (float(left_top_x), float(left_top_y), int(left_top_n))
                    result['right_common_top'] = (float(right_top_x), float(right_top_y), int(right_top_n))
                    result['left_common_bottom'] = (float(left_bottom_x), float(left_bottom_y), int(left_bottom_n))
                    result['right_common_bottom'] = (float(right_bottom_x), float(right_bottom_y), int(right_bottom_n))
                else:
                    result['common_reason'] = 'empty_top_or_bottom_band'
            else:
                result['common_reason'] = (
                    f'common_y_span<{self.p5_inner_edge_min_common_y_span:.0f}'
                )

        return result

    @staticmethod
    def clamp_value(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def compute_p5_up_slope_inner_edge_corrected_cmd(
        self,
        base_vy: float,
        base_wz: float,
        frame: np.ndarray,
    ):
        """
        根据上坡左右内侧边缘结果修正 vy/wz。
        只在 common_valid=True 时修正；否则返回基础速度。
        """
        if not self.p5_yellow_lateral_correction_enabled:
            # 里程主导模式：黄线不参与控制，只保留本段标定的固定 vy/wz 基准。
            return base_vy, base_wz
        if not self.p5_inner_edge_align_enabled:
            return base_vy, base_wz

        edge_result = self.detect_p5_inner_edges(frame)
        self.latest_p5_inner_edge_result = edge_result

        if not edge_result.get('common_valid', False):
            self.get_logger().info(
                f'[P5_INNER_EDGE_ALIGN] no correction: '
                f'left={edge_result.get("has_left")}, '
                f'right={edge_result.get("has_right")}, '
                f'reason={edge_result.get("common_reason")}, '
                f'L_reason={edge_result.get("left_edge", {}).get("reason")}, '
                f'R_reason={edge_result.get("right_edge", {}).get("reason")}',
                throttle_duration_sec=0.5
            )
            return base_vy, base_wz

        center_error = float(edge_result.get('center_error', 0.0))
        heading_error = float(edge_result.get('heading_error', 0.0))

        vy_corr = 0.0
        wz_corr = 0.0

        if self.p5_inner_edge_enable_vy and abs(center_error) > self.p5_inner_edge_center_deadband_px:
            vy_corr = -self.p5_inner_edge_center_k_vy * center_error
            vy_corr = self.clamp_value(
                vy_corr,
                -self.p5_inner_edge_vy_max_correction,
                self.p5_inner_edge_vy_max_correction
            )

        if self.p5_inner_edge_enable_wz and abs(heading_error) > self.p5_inner_edge_heading_deadband_px:
            wz_corr = self.p5_inner_edge_heading_k_wz * heading_error
            wz_corr = self.clamp_value(
                wz_corr,
                -self.p5_inner_edge_wz_max_correction,
                self.p5_inner_edge_wz_max_correction
            )

        cmd_vy = base_vy + vy_corr
        cmd_wz = base_wz + wz_corr

        edge_result['cmd_vy_correction'] = float(vy_corr)
        edge_result['cmd_wz_correction'] = float(wz_corr)
        edge_result['cmd_vy'] = float(cmd_vy)
        edge_result['cmd_wz'] = float(cmd_wz)
        self.latest_p5_inner_edge_result = edge_result

        self.get_logger().info(
            f'[P5_INNER_EDGE_ALIGN] '
            f'center_error={center_error:.1f}px, heading_error={heading_error:.1f}px, '
            f'vy_corr={vy_corr:.3f}, wz_corr={wz_corr:.3f}, '
            f'cmd_vy={cmd_vy:.3f}, cmd_wz={cmd_wz:.3f}, '
            f'common_span={float(edge_result.get("common_y_span", 0.0)):.1f}',
            throttle_duration_sec=0.3
        )

        return cmd_vy, cmd_wz

    def draw_p5_inner_edge_debug(self, vis: np.ndarray, result: dict):
        if result is None:
            return

        h, w = vis.shape[:2]

        left_roi = result.get('left_roi')
        right_roi = result.get('right_roi')

        if left_roi is not None:
            x1, y1, x2, y2 = left_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                vis,
                'INNER LEFT ROI',
                (x1 + 3, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1
            )

        if right_roi is not None:
            x1, y1, x2, y2 = right_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                'INNER RIGHT ROI',
                (x1 + 3, max(45, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

        left_edge = result.get('left_edge')
        right_edge = result.get('right_edge')

        if left_edge is not None:
            self.draw_p5_inner_edge_line(vis, left_edge, (255, 0, 0), 'L_INNER')
        if right_edge is not None:
            self.draw_p5_inner_edge_line(vis, right_edge, (0, 0, 255), 'R_INNER')

        if result.get('common_top_y') is not None and result.get('common_bottom_y') is not None:
            common_top_y = int(result['common_top_y'])
            common_bottom_y = int(result['common_bottom_y'])
            cv2.line(vis, (0, common_top_y), (w - 1, common_top_y), (0, 180, 255), 1)
            cv2.line(vis, (0, common_bottom_y), (w - 1, common_bottom_y), (0, 180, 255), 1)

        if result.get('common_valid', False):
            bottom_center_x = int(result['bottom_center_x'])
            top_center_x = int(result['top_center_x'])
            bottom_center_y = int(result['bottom_center_y'])
            top_center_y = int(result['top_center_y'])

            cv2.circle(vis, (bottom_center_x, bottom_center_y), 7, (0, 255, 255), -1)
            cv2.circle(vis, (top_center_x, top_center_y), 7, (0, 255, 255), -1)
            cv2.line(
                vis,
                (bottom_center_x, bottom_center_y),
                (top_center_x, top_center_y),
                (0, 255, 255),
                2
            )

            cv2.putText(
                vis,
                f'INNER center={result["center_error"]:.1f}px '
                f'heading={result["heading_error"]:.1f}px '
                f'vy_c={result.get("cmd_vy_correction", 0.0):.3f} '
                f'wz_c={result.get("cmd_wz_correction", 0.0):.3f}',
                (10, h - 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )
        else:
            cv2.putText(
                vis,
                f'INNER no correction: L={result.get("has_left")} '
                f'R={result.get("has_right")} reason={result.get("common_reason")}',
                (10, h - 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

    def draw_p5_inner_edge_line(self, vis: np.ndarray, edge: dict, color, name: str):
        pts = edge.get('points', [])

        for i, (x, y) in enumerate(pts):
            if i % 2 == 0:
                cv2.circle(vis, (int(x), int(y)), 2, color, -1)

        if edge.get('top_x') is not None and edge.get('bottom_x') is not None:
            top_pt = (int(edge['top_x']), int(edge['top_y']))
            bottom_pt = (int(edge['bottom_x']), int(edge['bottom_y']))

            cv2.circle(vis, top_pt, 6, color, -1)
            cv2.circle(vis, bottom_pt, 6, color, -1)
            cv2.line(vis, top_pt, bottom_pt, color, 2)

        text = (
            f'{name} valid={edge.get("valid")} '
            f'pts={edge.get("point_count", 0)}/{edge.get("raw_point_count", 0)} '
            f'bot={edge.get("bottom_ratio", 0):.2f} '
            f'yspan={edge.get("y_span", 0):.0f} '
            f'reason={edge.get("reason")}'
        )
        text_y = 108 if name == 'L_INNER' else 134
        cv2.putText(
            vis,
            text,
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1
        )


    def draw_p5_right_slope_right_edge_debug(self, vis: np.ndarray, result: dict):
        if result is None:
            return

        h, w = vis.shape[:2]
        roi = result.get('roi')
        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                'RIGHT SLOPE RIGHT EDGE ROI',
                (x1 + 3, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

        center_thr = result.get('too_center_threshold_x')
        right_thr = result.get('too_right_threshold_x')
        if center_thr is not None:
            cx = int(center_thr)
            cv2.line(vis, (cx, 0), (cx, h - 1), (255, 255, 0), 1)
            cv2.putText(vis, 'too_center', (max(5, cx - 80), 154),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        if right_thr is not None:
            rx = int(right_thr)
            cv2.line(vis, (rx, 0), (rx, h - 1), (0, 255, 255), 2)
            cv2.putText(vis, 'too_right', (max(5, rx - 80), 178),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        for i, (x, y) in enumerate(result.get('raw_points', [])):
            if i % 3 == 0:
                cv2.circle(vis, (int(x), int(y)), 1, (120, 120, 120), -1)

        for i, (x, y) in enumerate(result.get('points', [])):
            if i % 2 == 0:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

        if result.get('top_x') is not None and result.get('bottom_x') is not None:
            top_pt = (int(result['top_x']), int(result['top_y']))
            bottom_pt = (int(result['bottom_x']), int(result['bottom_y']))
            cv2.circle(vis, top_pt, 5, (0, 0, 255), -1)
            cv2.circle(vis, bottom_pt, 5, (0, 0, 255), -1)
            cv2.line(vis, top_pt, bottom_pt, (0, 0, 255), 2)

        if result.get('bottom_band_low_y') is not None:
            by = int(result['bottom_band_low_y'])
            cv2.line(vis, (0, by), (w - 1, by), (0, 180, 255), 1)

        if result.get('valid') and result.get('right_inner_x') is not None:
            ix = int(result['right_inner_x'])
            cv2.line(vis, (ix, 0), (ix, h - 1), (0, 255, 0), 2)

        cmd_vx = result.get('cmd_vx')
        cmd_vy = result.get('cmd_vy')
        cmd_wz = result.get('cmd_wz')
        base_vy = result.get('base_vy')
        if cmd_vx is None or cmd_vy is None or cmd_wz is None:
            cmd_text = 'SEND cmd=(None,None,None)'
        else:
            cmd_text = f'SEND cmd=({float(cmd_vx):.3f},{float(cmd_vy):.3f},{float(cmd_wz):.3f})'

        cv2.putText(
            vis,
            f'R_SLOPE_EDGE valid={result.get("valid")} '
            f'too_center={result.get("too_center")} too_right={result.get("too_right")} '
            f'action={result.get("action")}',
            (10, h - 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'{cmd_text} base_vy={base_vy} final_vy={cmd_vy}',
            (10, h - 152),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'right_x={result.get("right_inner_x")} '
            f'pts={result.get("point_count", 0)}/{result.get("raw_point_count", 0)} '
            f'yspan={result.get("y_span", 0):.0f} '
            f'xstd={result.get("x_std", 0):.1f} '
            f'reason={result.get("reason")}',
            (10, h - 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f'lost_active={result.get("lost_extra_active")} '
            f'dir={result.get("lost_extra_direction")} '
            f'cntC={result.get("too_center_count", 0)}/'
            f'{result.get("lost_extra_confirm_count", 0)} '
            f'cntR={result.get("too_right_count", 0)}/'
            f'{result.get("lost_extra_confirm_count", 0)}',
            (10, h - 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            2
        )


    # ============================================================
    # 可视化
    # ============================================================
    def show_p5_compact_debug_window(self, frame: np.ndarray):
        """
        简洁可视化模式：只保留当前调试最需要看的信息。
        不画 raw_points / points / bottom band / 大量 reason，避免画面太乱。
        """
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]

            # 1. 图像中心线
            image_center_x = w // 2
            cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 255, 255), 1)

            # 2. 优先从右斜坡边缘结果里拿实际发送命令；如果还没有，就从 msg 里拿当前 vel_des。
            right_edge = getattr(self, 'latest_p5_right_slope_right_edge_result', None)
            cmd_vx = cmd_vy = cmd_wz = None
            if isinstance(right_edge, dict):
                cmd_vx = right_edge.get('cmd_vx')
                cmd_vy = right_edge.get('cmd_vy')
                cmd_wz = right_edge.get('cmd_wz')

            if cmd_vx is None or cmd_vy is None or cmd_wz is None:
                try:
                    vel = list(getattr(self.msg, 'vel_des', [0.0, 0.0, 0.0]))
                    cmd_vx, cmd_vy, cmd_wz = float(vel[0]), float(vel[1]), float(vel[2])
                except Exception:
                    cmd_vx, cmd_vy, cmd_wz = 0.0, 0.0, 0.0

            # 3. 右斜坡右侧边缘：只画 ROI、两个阈值、right_inner_x，不画点云。
            if isinstance(right_edge, dict):
                roi = right_edge.get('roi')
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

                center_thr = right_edge.get('too_center_threshold_x')
                right_thr = right_edge.get('too_right_threshold_x')
                if center_thr is not None:
                    cx = int(center_thr)
                    cv2.line(vis, (cx, 0), (cx, h - 1), (255, 255, 0), 2)
                    cv2.putText(vis, 'center_thr', (max(5, cx - 72), 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                if right_thr is not None:
                    rx = int(right_thr)
                    cv2.line(vis, (rx, 0), (rx, h - 1), (0, 255, 255), 2)
                    cv2.putText(vis, 'right_thr', (max(5, rx - 65), 88),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                if right_edge.get('valid') and right_edge.get('right_inner_x') is not None:
                    ix = int(right_edge['right_inner_x'])
                    cv2.line(vis, (ix, 0), (ix, h - 1), (0, 255, 0), 3)
                    cv2.putText(vis, 'right_inner', (max(5, ix - 70), 114),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 4. 上坡/右跳后前进阶段的内侧边缘矫正：只显示数值，不画左右大量点。
            inner_edge = getattr(self, 'latest_p5_inner_edge_result', None)
            inner_text = 'INNER: n/a'
            if isinstance(inner_edge, dict):
                ce = inner_edge.get('center_error')
                he = inner_edge.get('heading_error')
                vy_c = float(inner_edge.get('cmd_vy_correction', 0.0))
                wz_c = float(inner_edge.get('cmd_wz_correction', 0.0))
                if ce is None or he is None:
                    inner_text = f'INNER: valid={inner_edge.get("common_valid")} reason={inner_edge.get("common_reason")}'
                else:
                    inner_text = (
                        f'INNER: valid={inner_edge.get("common_valid")} '
                        f'ce={float(ce):.1f} he={float(he):.1f} '
                        f'vy_c={vy_c:.3f} wz_c={wz_c:.3f}'
                    )

            # 5. 右斜坡边缘状态文字：压缩为 1 行。
            if isinstance(right_edge, dict):
                ratio = right_edge.get('right_inner_x_ratio')
                ratio_text = 'None' if ratio is None else f'{float(ratio):.3f}'
                edge_text = (
                    f'R_EDGE: valid={right_edge.get("valid")} '
                    f'ratio={ratio_text} action={right_edge.get("action")} '
                    f'lost={right_edge.get("lost_extra_active")} '
                    f'dir={right_edge.get("lost_extra_direction")} '
                    f'C/R={right_edge.get("too_center_count", 0)}/'
                    f'{right_edge.get("too_right_count", 0)} '
                    f'reason={right_edge.get("reason")}'
                )
            else:
                edge_text = 'R_EDGE: n/a'

            # 6. 总文字区：固定只显示 4 行。
            cv2.putText(
                vis,
                f'STATE: {self.state}',
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f'CMD: vx={float(cmd_vx):.3f} vy={float(cmd_vy):.3f} wz={float(cmd_wz):.3f}',
                (10, h - 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )
            cv2.putText(
                vis,
                edge_text[:115],
                (10, h - 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2
            )
            cv2.putText(
                vis,
                inner_text[:115],
                (10, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                2
            )

            cv2.imshow('P5 Bridge Debug', vis)

            # 简洁模式默认不强制显示 mask；只有用户显式打开 p5_show_yellow_mask 才显示。
            if self.p5_show_yellow_mask:
                inner_edge = getattr(self, 'latest_p5_inner_edge_result', None)
                inner_mask = None if inner_edge is None else inner_edge.get('mask')
                if inner_mask is not None:
                    cv2.imshow('P5 Yellow Mask', inner_mask)
                elif self.latest_p5_yellow_mask is not None:
                    cv2.imshow('P5 Yellow Mask', self.latest_p5_yellow_mask)

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f'[P5_VIS_COMPACT] show debug failed: {repr(e)}')

    def show_p5_debug_window(self, frame: np.ndarray):
        try:
            detail_level = int(getattr(self, 'p5_debug_vis_detail_level', 1))
            if detail_level <= 0:
                return
            if detail_level == 1:
                self.show_p5_compact_debug_window(frame)
                return

            # detail_level >= 2：保留原来的完整可视化。
            vis = frame.copy()
            h, w = vis.shape[:2]

            yellow = self.latest_p5_yellow_result

            image_center_x = w // 2
            cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 255, 255), 1)

            roi_top = int(h * self.p5_yellow_roi_top_ratio)
            roi_left = int(w * self.p5_yellow_roi_left_ratio)
            roi_right = int(w * self.p5_yellow_roi_right_ratio)

            roi_top = max(0, min(h - 1, roi_top))
            roi_left = max(0, min(w - 1, roi_left))
            roi_right = max(roi_left + 1, min(w, roi_right))

            cv2.rectangle(vis, (roi_left, roi_top), (roi_right, h - 1), (0, 255, 255), 2)
            cv2.putText(
                vis,
                'P5 strict front yellow ROI',
                (roi_left + 3, max(20, roi_top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1
            )

            threshold_y = int(h * self.p5_yellow_stop_line_y_ratio)
            cv2.line(vis, (0, threshold_y), (w - 1, threshold_y), (0, 180, 255), 2)
            cv2.putText(
                vis,
                f'stop_y={threshold_y} ratio={self.p5_yellow_stop_line_y_ratio:.2f}',
                (10, max(25, threshold_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 180, 255),
                2
            )

            cv2.putText(
                vis,
                f'state={self.state}',
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.putText(
                vis,
                f'frame_seq={self.latest_frame_seq} state_enter_seq={self.state_enter_frame_seq}',
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            if yellow.get('has_line') and yellow.get('line_bottom_y') is not None:
                bottom_y = int(yellow['line_bottom_y'])
                bbox = yellow.get('bbox')
                center = yellow.get('line_center')
                angle = yellow.get('angle_deg')
                width_ratio = yellow.get('width_ratio')
                wh_ratio = yellow.get('wh_ratio')

                cv2.line(vis, (0, bottom_y), (w - 1, bottom_y), (0, 255, 255), 2)

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if center is not None:
                    cx, cy = center
                    cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

                    if angle is not None:
                        length = 80
                        rad = math.radians(float(angle))
                        dx = int(math.cos(rad) * length)
                        dy = int(math.sin(rad) * length)
                        cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)

                    angle_text = 'None' if angle is None else f'{float(angle):.1f}deg'
                    cv2.putText(
                        vis,
                        f'YELLOW bottom={bottom_y} angle={angle_text}',
                        (max(5, cx - 120), max(18, cy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 255),
                        2
                    )

                cv2.putText(
                    vis,
                    f'counter={self.p5_yellow_stop_counter}/{self.p5_yellow_stop_confirm_count} '
                    f'width_ratio={width_ratio:.2f} wh={wh_ratio:.1f}',
                    (10, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2
                )

            else:
                cv2.putText(
                    vis,
                    'P5 strict front yellow: NOT detected',
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2
                )

            # P5_RIGHT_SLOPE 专用：中间区域黄色存在/消失可视化
            center_yellow = getattr(self, 'latest_p5_center_yellow_presence_result', None)
            if center_yellow is not None:
                roi = center_yellow.get('roi')
                if roi is not None:
                    cx1, cy1, cx2, cy2 = roi
                    color = (0, 255, 0) if center_yellow.get('has_yellow', False) else (0, 0, 255)
                    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), color, 2)
                    cv2.putText(
                        vis,
                        'P5 center yellow presence ROI',
                        (cx1 + 3, max(20, cy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1
                    )

                bbox = center_yellow.get('bbox')
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                cv2.putText(
                    vis,
                    f'CENTER_YELLOW has={center_yellow.get("has_yellow")} '
                    f'pixels={center_yellow.get("yellow_pixels")}/'
                    f'{self.p5_center_yellow_min_pixels} '
                    f'ratio={float(center_yellow.get("yellow_ratio", 0.0)):.4f}/'
                    f'{self.p5_center_yellow_min_ratio:.4f} '
                    f'absent={self.p5_center_yellow_absent_counter}/'
                    f'{self.p5_center_yellow_absent_confirm_count}',
                    (10, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0) if center_yellow.get('has_yellow', False) else (0, 0, 255),
                    2
                )


            # P5_UP_SLOPE 专用：右侧赛道黄线可视化
            right_side = getattr(self, 'latest_p5_right_side_yellow_result', None)
            if right_side is not None:
                roi = right_side.get('roi')
                if roi is not None:
                    rx1, ry1, rx2, ry2 = roi
                    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
                    cv2.putText(
                        vis,
                        'P5 right-side yellow ROI',
                        (rx1 + 3, max(20, ry1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1
                    )

                for c in right_side.get('candidates', []):
                    bx1, by1, bx2, by2 = c['bbox']
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (120, 120, 120), 1)

                bbox = right_side.get('bbox')
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    color = (255, 0, 255) if right_side.get('has_line', False) else (0, 0, 255)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 3)

                    center = right_side.get('center')
                    if center is not None:
                        cx, cy = center
                        cv2.circle(vis, (cx, cy), 5, color, -1)

                ratio = right_side.get('bottom_ratio')
                ratio_text = 'None' if ratio is None else f'{float(ratio):.3f}'
                cv2.putText(
                    vis,
                    f'RIGHT_SIDE has={right_side.get("has_line")} '
                    f'reason={right_side.get("reason")} '
                    f'ratio={ratio_text}/{self.p5_right_side_yellow_bottom_valid_ratio:.2f} '
                    f'lost={self.p5_right_side_yellow_lost_counter}/'
                    f'{self.p5_right_side_yellow_lost_confirm_count}',
                    (10, h - 96),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    2
                )

            # P5_UP_SLOPE 专用：左右内侧黄线边缘矫正可视化
            self.draw_p5_inner_edge_debug(
                vis,
                getattr(self, 'latest_p5_inner_edge_result', None)
            )

            # P5_RIGHT_SLOPE_1/2/3 专用：右侧黄线内侧边缘 vy 修正可视化
            self.draw_p5_right_slope_right_edge_debug(
                vis,
                getattr(self, 'latest_p5_right_slope_right_edge_result', None)
            )

            cv2.imshow('P5 Bridge Debug', vis)

            if self.p5_show_yellow_mask:
                inner_edge = getattr(self, 'latest_p5_inner_edge_result', None)
                inner_mask = None if inner_edge is None else inner_edge.get('mask')
                if inner_mask is not None:
                    cv2.imshow('P5 Yellow Mask', inner_mask)
                elif self.latest_p5_yellow_mask is not None:
                    cv2.imshow('P5 Yellow Mask', self.latest_p5_yellow_mask)

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f'[P5_VIS] show debug failed: {repr(e)}')

    # ============================================================
    # 通用状态执行函数
    # ============================================================
    def run_timed_velocity_state(
        self,
        duration_s: float,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
        zero_velocity_when_done: bool = False,
    ):
        if self.p5_timed_motion_guard(duration_s, log_name or self.state):
            return

        elapsed = self.p5_state_elapsed_s()

        if elapsed < duration_s:
            self.p5_send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            self.get_logger().info(
                f'[{log_name or self.state}] moving: '
                f'elapsed={elapsed:.3f}/{duration_s:.3f}s, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f}), '
                f'step_height={step_height:.3f}, '
                f'roll={roll:.3f}, pitch={pitch:.3f}, body_height={body_height:.3f}',
                throttle_duration_sec=0.5
            )
            return

        if zero_velocity_when_done:
            # 非阻塞地覆盖 Robot_Ctrl 心跳缓存，避免下一状态设置 body 的等待期间
            # 继续重发上一段转向速度。
            self.p5_send_velocity_command(
                vx=0.0,
                vy=0.0,
                wz=0.0,
                step_height=0.0,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

        self.get_logger().info(
            f'[{log_name or self.state}] done, go {next_state}, '
            f'zero_velocity_when_done={zero_velocity_when_done}'
        )

        # 不在持续速度段之间自动 stop，保证 P5_STEP_UP -> P5_UP_SLOPE 连续衔接。
        self.p5_enter_state(next_state)

    def run_yellow_stop_velocity_state(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
    ):
        elapsed = self.p5_state_elapsed_s()

        # 刚进入状态时，先忽略旧帧/旧黄线，避免动作刚切完立刻用上一状态图像触发。
        if elapsed < self.p5_yellow_ignore_after_enter_s:
            self.p5_send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] ignore yellow after enter: '
                f'elapsed={elapsed:.3f}/{self.p5_yellow_ignore_after_enter_s:.3f}, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        # 如果当前状态还没有收到新图像，先继续走，避免使用旧帧误触发。
        if self.latest_frame_seq <= self.state_enter_frame_seq:
            if self.p5_keep_moving_when_no_image:
                self.p5_send_velocity_command(
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    step_height=step_height,
                    roll=roll,
                    pitch=pitch,
                    body_height=body_height,
                )
                self.get_logger().warn(
                    f'[{log_name or self.state}] no new image after state enter, keep moving: '
                    f'frame_seq={self.latest_frame_seq}, enter_seq={self.state_enter_frame_seq}',
                    throttle_duration_sec=0.5
                )
            else:
                # fail-safe：非阻塞零速度保持，避免每个 tick 重复阻塞 Wait_finish。
                self.p5_send_velocity_command(
                    vx=0.0,
                    vy=0.0,
                    wz=0.0,
                    step_height=0.0,
                    roll=roll,
                    pitch=pitch,
                    body_height=body_height,
                )
                self.get_logger().warn(
                    f'[{log_name or self.state}] no new image after state enter, '
                    f'hold zero velocity',
                    throttle_duration_sec=0.5
                )
            return

        yellow = self.latest_p5_yellow_result

        yellow_wz = self.compute_p5_yellow_angle_align_wz(yellow)
        cmd_wz = yellow_wz if abs(yellow_wz) > 1e-6 else wz

        if self.p5_yellow_reached_bottom(yellow):
            bottom_y = yellow.get('line_bottom_y')
            self.get_logger().info(
                f'[{log_name or self.state}] yellow reached bottom, stop and go {next_state}: '
                f'bottom={bottom_y}'
            )
            self.p5_send_stop_command()
            self.p5_enter_state(next_state)
            return

        self.p5_send_velocity_command(
            vx=vx,
            vy=vy,
            wz=cmd_wz,
            step_height=step_height,
            roll=roll,
            pitch=pitch,
            body_height=body_height,
        )

        if yellow is None or not yellow.get('has_line', False):
            self.get_logger().info(
                f'[{log_name or self.state}] no strict front yellow, keep moving: '
                f'cmd=({vx:.3f},{vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.5
            )
        else:
            angle = yellow.get('angle_deg')
            angle_text = 'None' if angle is None else f'{float(angle):.2f}'
            self.get_logger().info(
                f'[{log_name or self.state}] strict front yellow seen, keep moving: '
                f'bottom={yellow.get("line_bottom_y")}, '
                f'angle={angle_text}, '
                f'cmd=({vx:.3f},{vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.5
            )


    # ============================================================
    # P5_RIGHT_SLOPE：右侧黄线内侧边缘检测与 vy 三档修正
    # ============================================================
    def keep_p5_right_slope_bottom_connected_segment(self, edge_points):
        """只保留从图像底部往上的连续右侧内侧边缘段。"""
        if not edge_points:
            return []

        if not self.p5_right_slope_right_edge_use_bottom_connected_segment:
            return sorted(edge_points, key=lambda p: p[1])

        max_y_gap = max(1, int(self.p5_right_slope_right_edge_max_y_gap))
        pts_bottom_to_top = sorted(edge_points, key=lambda p: p[1], reverse=True)

        kept = [pts_bottom_to_top[0]]
        prev_y = pts_bottom_to_top[0][1]

        for x, y in pts_bottom_to_top[1:]:
            y_gap = abs(prev_y - y)
            if y_gap > max_y_gap:
                break
            kept.append((x, y))
            prev_y = y

        return sorted(kept, key=lambda p: p[1])

    def detect_p5_right_slope_right_inner_edge(self, frame: np.ndarray) -> dict:
        """
        右斜坡阶段专用：只检测右侧黄线的内侧边缘。

        在右侧 ROI 内，每一行取最左侧黄色像素，得到右侧黄线内侧边缘点串。
        然后用点数、纵向跨度、x_std 和 bottom_ratio 判断是否有效。
        最后用底部 band 的平均 x 作为 right_inner_x。
        """
        h, w = frame.shape[:2]

        mask = self.make_p5_inner_edge_yellow_mask(frame)

        x1 = int(w * self.p5_right_slope_right_edge_roi_x_min)
        x2 = int(w * self.p5_right_slope_right_edge_roi_x_max)
        y1 = int(h * self.p5_right_slope_right_edge_roi_y_min)
        y2 = int(h * self.p5_right_slope_right_edge_roi_y_max)
        x1, y1, x2, y2 = self.clamp_p5_roi((x1, y1, x2, y2), w, h)

        roi_mask = mask[y1:y2, x1:x2]
        row_step = max(1, int(self.p5_right_slope_right_edge_row_step))

        raw_points = []
        for local_y in range(0, roi_mask.shape[0], row_step):
            row = roi_mask[local_y, :]
            xs = np.where(row > 0)[0]
            if xs.size == 0:
                continue

            # 右侧黄线取内侧边缘：每行最左侧黄色像素
            local_x = int(np.min(xs))
            raw_points.append((int(x1 + local_x), int(y1 + local_y)))

        points = self.keep_p5_right_slope_bottom_connected_segment(raw_points)

        result = {
            'mask': mask,
            'roi': (int(x1), int(y1), int(x2), int(y2)),
            'raw_points': raw_points,
            'points': points,
            'valid': False,
            'reason': 'init',
            'point_count': len(points),
            'raw_point_count': len(raw_points),
            'y_span': 0.0,
            'x_std': 0.0,
            'bottom_ratio': 0.0,
            'top_x': None,
            'top_y': None,
            'bottom_x': None,
            'bottom_y': None,
            'right_inner_x': None,
            'right_inner_x_ratio': None,
            'too_center_threshold_x': float(w * self.p5_right_slope_right_too_center_ratio),
            'too_right_threshold_x': float(w * self.p5_right_slope_right_too_right_ratio),
            'too_center': False,
            'too_right': False,
            'bottom_band_low_y': None,
            'cmd_vy': None,
        }

        if len(raw_points) == 0:
            result['reason'] = 'no_yellow_points'
            return result

        if len(points) == 0:
            result['reason'] = 'no_bottom_connected_segment'
            return result

        pts = np.array(points, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        point_count = len(points)
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        y_span = y_max - y_min
        x_std = float(np.std(xs))
        bottom_ratio = y_max / float(max(h, 1))

        top_band_high = y_min + 0.20 * max(y_span, 1.0)
        bottom_band_low = y_max - 0.20 * max(y_span, 1.0)
        top_band = pts[pts[:, 1] <= top_band_high]
        bottom_band = pts[pts[:, 1] >= bottom_band_low]
        if len(top_band) == 0:
            top_band = pts
        if len(bottom_band) == 0:
            bottom_band = pts

        top_x = float(np.mean(top_band[:, 0]))
        top_y = float(np.mean(top_band[:, 1]))
        bottom_x = float(np.mean(bottom_band[:, 0]))
        bottom_y = float(np.mean(bottom_band[:, 1]))

        # 用底部 band 计算 right_inner_x，更接近机器狗当前近处位置。
        bottom_band_low_y = y_max - self.p5_right_slope_right_edge_bottom_band_ratio * max(y_span, 1.0)
        inner_band = pts[pts[:, 1] >= bottom_band_low_y]
        if len(inner_band) == 0:
            inner_band = pts

        right_inner_x = float(np.mean(inner_band[:, 0]))
        right_inner_x_ratio = right_inner_x / float(max(w, 1))

        fail_reasons = []
        if point_count < self.p5_right_slope_right_edge_min_points:
            fail_reasons.append(f'points<{self.p5_right_slope_right_edge_min_points}')
        if y_span < self.p5_right_slope_right_edge_min_y_span:
            fail_reasons.append(f'y_span<{self.p5_right_slope_right_edge_min_y_span:.0f}')
        if x_std > self.p5_right_slope_right_edge_x_std_max:
            fail_reasons.append(f'x_std>{self.p5_right_slope_right_edge_x_std_max:.0f}')
        if bottom_ratio < self.p5_right_slope_right_edge_bottom_min_ratio:
            fail_reasons.append(f'bottom<{self.p5_right_slope_right_edge_bottom_min_ratio:.2f}')

        valid = len(fail_reasons) == 0

        too_center_threshold_x = w * self.p5_right_slope_right_too_center_ratio
        too_right_threshold_x = w * self.p5_right_slope_right_too_right_ratio
        too_center = bool(valid and right_inner_x < too_center_threshold_x)
        too_right = bool(valid and right_inner_x > too_right_threshold_x)

        result.update({
            'valid': bool(valid),
            'reason': 'ok' if valid else ','.join(fail_reasons),
            'point_count': int(point_count),
            'raw_point_count': int(len(raw_points)),
            'y_span': float(y_span),
            'x_std': float(x_std),
            'bottom_ratio': float(bottom_ratio),
            'top_x': float(top_x),
            'top_y': float(top_y),
            'bottom_x': float(bottom_x),
            'bottom_y': float(bottom_y),
            'right_inner_x': float(right_inner_x),
            'right_inner_x_ratio': float(right_inner_x_ratio),
            'too_center_threshold_x': float(too_center_threshold_x),
            'too_right_threshold_x': float(too_right_threshold_x),
            'too_center': bool(too_center),
            'too_right': bool(too_right),
            'bottom_band_low_y': float(bottom_band_low_y),
        })

        return result

    def reset_p5_right_slope_lost_extra_state(self):
        """
        清空右斜坡“危险后丢线持续补偿”状态。

        每次进入新的 P5_RIGHT_SLOPE_1/2/3 时调用，
        避免上一段右斜坡的危险方向影响下一段。
        """
        self.p5_right_slope_too_center_count = 0
        self.p5_right_slope_too_right_count = 0
        self.p5_right_slope_lost_extra_active = False
        self.p5_right_slope_lost_extra_direction = 'none'
        self.p5_right_slope_lost_extra_hold_start_s = None

        if isinstance(getattr(self, 'latest_p5_right_slope_right_edge_result', None), dict):
            self.latest_p5_right_slope_right_edge_result['lost_extra_active'] = False
            self.latest_p5_right_slope_right_edge_result['lost_extra_direction'] = 'none'
            self.latest_p5_right_slope_right_edge_result['too_center_count'] = 0
            self.latest_p5_right_slope_right_edge_result['too_right_count'] = 0
            self.latest_p5_right_slope_right_edge_result['record_ignore_active'] = True
            self.latest_p5_right_slope_right_edge_result['record_ignore_elapsed_s'] = 0.0
            self.latest_p5_right_slope_right_edge_result['record_ignore_duration_s'] = float(
                getattr(self, 'p5_right_slope_lost_extra_ignore_after_enter_s', 0.0)
            )
            self.latest_p5_right_slope_right_edge_result['action'] = 'reset_on_enter_right_slope'

        self.get_logger().info(
            '[P5_RIGHT_SLOPE_RIGHT_EDGE_VY] reset lost-extra state for new right-slope segment'
        )


    def compute_p5_right_slope_right_edge_corrected_vy(self, base_vy: float, frame: np.ndarray) -> float:
        """
        右斜坡阶段 vy 三档修正 + 危险后丢线持续补偿。

        当前看得到右侧黄线时：
        - right_inner_x 太靠中间：固定加大 vy；
        - right_inner_x 太靠右：固定减小 vy；
        - too_center / too_right 连续达到确认次数后，记录危险方向。

        后续当前右侧黄线无效 / 识别不到时：
        - 如果本段右斜坡前面已经确认过 too_center，则持续叠加 lost_extra_too_center_vy；
        - 如果本段右斜坡前面已经确认过 too_right，则持续叠加 lost_extra_too_right_vy；
        - 如果本段右斜坡前面没有确认过危险，则保持 base_vy。

        注意：lost_extra_active 一旦触发，会保持到当前 P5_RIGHT_SLOPE_x 状态结束；
        进入下一段 P5_RIGHT_SLOPE_1/2/3 时由 enter_state() 清零。
        """
        if not self.p5_yellow_lateral_correction_enabled:
            # 里程主导模式：黄线不参与控制，只保留本段标定的固定 vy 基准。
            return base_vy
        if not self.p5_right_slope_right_edge_vy_adjust_enabled:
            return base_vy
        if frame is None:
            return base_vy

        result = self.detect_p5_right_slope_right_inner_edge(frame)

        cmd_vy = base_vy
        action = 'base'

        valid = bool(result.get('valid', False))
        too_center = bool(result.get('too_center', False))
        too_right = bool(result.get('too_right', False))

        elapsed = self.p5_safety_elapsed_s()
        ignore_duration = float(getattr(self, 'p5_right_slope_lost_extra_ignore_after_enter_s', 0.0))
        record_ignore_active = elapsed < ignore_duration
        is_new_frame = (
            self.latest_frame_seq
            != self.p5_right_slope_lost_extra_last_eval_frame_seq
        )
        if is_new_frame:
            self.p5_right_slope_lost_extra_last_eval_frame_seq = self.latest_frame_seq

        if valid:
            # 重新看到有效右侧黄线：结束当前连续丢线补偿计时段。
            self.p5_right_slope_lost_extra_hold_start_s = None
            # 当前看得到右侧黄线：使用原来的三档修正，同时更新危险趋势计数。
            if too_center:
                cmd_vy = base_vy + self.p5_right_slope_right_too_center_add_vy

                if record_ignore_active:
                    # 刚进入当前右斜坡段的一小段时间：只修正，不记录危险次数，
                    # 避免切状态瞬间、机身未稳定或旧帧导致误触发 lost-extra。
                    if is_new_frame:
                        self.p5_right_slope_too_center_count = 0
                        self.p5_right_slope_too_right_count = 0
                    action = 'visible_too_center_add_vy_ignore_record'
                else:
                    if is_new_frame:
                        self.p5_right_slope_too_center_count += 1
                        self.p5_right_slope_too_right_count = 0
                    action = 'visible_too_center_add_vy'

                    if (
                        self.p5_right_slope_lost_extra_enabled
                        and self.p5_right_slope_too_center_count >= self.p5_right_slope_lost_extra_confirm_count
                    ):
                        self.p5_right_slope_lost_extra_active = True
                        self.p5_right_slope_lost_extra_direction = 'too_center'

            elif too_right:
                cmd_vy = base_vy - self.p5_right_slope_right_too_right_reduce_vy

                if record_ignore_active:
                    # 刚进入当前右斜坡段的一小段时间：只修正，不记录危险次数。
                    if is_new_frame:
                        self.p5_right_slope_too_center_count = 0
                        self.p5_right_slope_too_right_count = 0
                    action = 'visible_too_right_reduce_vy_ignore_record'
                else:
                    if is_new_frame:
                        self.p5_right_slope_too_right_count += 1
                        self.p5_right_slope_too_center_count = 0
                    action = 'visible_too_right_reduce_vy'

                    if (
                        self.p5_right_slope_lost_extra_enabled
                        and self.p5_right_slope_too_right_count >= self.p5_right_slope_lost_extra_confirm_count
                    ):
                        self.p5_right_slope_lost_extra_active = True
                        self.p5_right_slope_lost_extra_direction = 'too_right'

            else:
                # 看得到黄线且处在安全范围：只清当前连续计数。
                # 不清 lost_extra_active，因为用户需求是：
                # “本段右斜坡一旦前面超过危险次数，后面识别不到就一直补偿到本段结束”。
                if is_new_frame:
                    self.p5_right_slope_too_center_count = 0
                    self.p5_right_slope_too_right_count = 0
                cmd_vy = base_vy
                action = 'visible_safe_base'

        else:
            # 当前看不到/无效：只有“本段前面已经确认过危险方向”才继续补偿。
            if self.p5_right_slope_lost_extra_enabled and self.p5_right_slope_lost_extra_active:
                # 硬时间上限：一次误判不允许无限放大。到达上限后撤销危险记忆，
                # 需要黄线重新可见并重新连续确认后才会再次补偿。
                max_hold = self.p5_right_slope_lost_extra_max_hold_s
                if self.p5_right_slope_lost_extra_hold_start_s is None:
                    self.p5_right_slope_lost_extra_hold_start_s = elapsed
                held_s = max(0.0, elapsed - self.p5_right_slope_lost_extra_hold_start_s)

                if max_hold > 0.0 and held_s >= max_hold:
                    self.get_logger().warn(
                        f'[P5_RIGHT_SLOPE_RIGHT_EDGE_VY] lost-extra hold expired: '
                        f'held={held_s:.2f}s >= max_hold={max_hold:.2f}s, '
                        f'direction={self.p5_right_slope_lost_extra_direction}, '
                        f'revert to base_vy and require re-confirmation'
                    )
                    self.p5_right_slope_lost_extra_active = False
                    self.p5_right_slope_lost_extra_direction = 'none'
                    self.p5_right_slope_too_center_count = 0
                    self.p5_right_slope_too_right_count = 0
                    self.p5_right_slope_lost_extra_hold_start_s = None
                    cmd_vy = base_vy
                    action = 'lost_hold_expired'
                elif self.p5_right_slope_lost_extra_direction == 'too_center':
                    cmd_vy = base_vy + self.p5_right_slope_lost_extra_too_center_vy
                    action = 'lost_hold_too_center_extra_vy'
                elif self.p5_right_slope_lost_extra_direction == 'too_right':
                    cmd_vy = base_vy + self.p5_right_slope_lost_extra_too_right_vy
                    action = 'lost_hold_too_right_extra_vy'
                else:
                    cmd_vy = base_vy
                    action = 'lost_active_but_no_direction'
            else:
                cmd_vy = base_vy
                action = 'lost_no_extra_base'

        result['cmd_vy'] = float(cmd_vy)
        result['base_vy'] = float(base_vy)
        result['action'] = action

        result['lost_extra_enabled'] = bool(self.p5_right_slope_lost_extra_enabled)
        result['lost_extra_active'] = bool(self.p5_right_slope_lost_extra_active)
        result['lost_extra_direction'] = str(self.p5_right_slope_lost_extra_direction)
        result['too_center_count'] = int(self.p5_right_slope_too_center_count)
        result['too_right_count'] = int(self.p5_right_slope_too_right_count)
        result['lost_extra_confirm_count'] = int(self.p5_right_slope_lost_extra_confirm_count)
        result['record_ignore_active'] = bool(record_ignore_active)
        result['record_ignore_elapsed_s'] = float(elapsed)
        result['record_ignore_duration_s'] = float(ignore_duration)
        result['lost_extra_too_center_vy'] = float(self.p5_right_slope_lost_extra_too_center_vy)
        result['lost_extra_too_right_vy'] = float(self.p5_right_slope_lost_extra_too_right_vy)

        self.latest_p5_right_slope_right_edge_result = result

        self.get_logger().info(
            f'[P5_RIGHT_SLOPE_RIGHT_EDGE_VY] '
            f'valid={valid}, reason={result.get("reason")}, '
            f'right_x={result.get("right_inner_x")}, '
            f'ratio={result.get("right_inner_x_ratio")}, '
            f'center_thr={result.get("too_center_threshold_x")}, '
            f'right_thr={result.get("too_right_threshold_x")}, '
            f'too_center={too_center}, too_right={too_right}, '
            f'cnt_center={self.p5_right_slope_too_center_count}/'
            f'{self.p5_right_slope_lost_extra_confirm_count}, '
            f'cnt_right={self.p5_right_slope_too_right_count}/'
            f'{self.p5_right_slope_lost_extra_confirm_count}, '
            f'ignore_record={record_ignore_active}({elapsed:.2f}/{ignore_duration:.2f}s), '
            f'new_frame={is_new_frame}, '
            f'lost_active={self.p5_right_slope_lost_extra_active}, '
            f'lost_dir={self.p5_right_slope_lost_extra_direction}, '
            f'action={action}, base_vy={base_vy:.3f}, cmd_vy={cmd_vy:.3f}',
            throttle_duration_sec=0.3
        )

        return cmd_vy

    def p5_apply_no_image_policy(
        self, vx, vy, wz, step_height, roll, pitch, body_height, log_name,
    ):
        """Apply the bounded policy before a valid state-local RGB frame."""
        if self.p5_keep_moving_when_no_image:
            command = (vx, vy, wz, step_height)
            action = 'keep moving (profile override)'
        else:
            command = (0.0, 0.0, 0.0, 0.0)
            action = 'hold zero velocity'
        self.p5_send_velocity_command(
            vx=command[0], vy=command[1], wz=command[2],
            step_height=command[3], roll=roll, pitch=pitch,
            body_height=body_height,
        )
        self.get_logger().warn(
            f'[{log_name or self.state}] no valid new RGB after state enter, '
            f'{action}: frame_seq={self.latest_frame_seq}, '
            f'enter_seq={self.state_enter_frame_seq}',
            throttle_duration_sec=0.5,
        )

    def run_center_yellow_absence_velocity_state(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
        stop_before_next: bool = False,
        timeout_s: float = 0.0,
        right_edge_adjust_enabled: bool = True,
    ):
        """
        P5_RIGHT_SLOPE_1/2/3 专用：
        中间 ROI 内还有黄色 -> 继续走；
        中间 ROI 内连续 N 个新图像帧没有黄色 -> 进入 next_state。
        （计数按新帧推进，不按控制 tick，与上坡丢线计数一致。）

        默认不主动 stop，保持 1/2 段连续衔接转向；
        如果 stop_before_next=True，则在切入 next_state 前先发送一次速度为 0 的速度命令，
        主要用于 P5_RIGHT_SLOPE_3 结束后清掉上一段运动速度，再 reset body。
        注意：这里不调用 send_stop_command()，保持控制循环非阻塞。

        如果本段声明用里程结束（不依赖黄线），直接转交里程段执行器。
        """
        if self.p5_route_segment_uses_odometry_exit():
            self.run_odometry_distance_velocity_state(
                vx=vx, vy=vy, wz=wz, step_height=step_height,
                next_state=next_state, roll=roll, pitch=pitch,
                body_height=body_height, log_name=log_name, timeout_s=timeout_s,
            )
            return

        if self.p5_vision_state_guard(timeout_s, log_name or self.state):
            return

        elapsed = self.p5_state_elapsed_s()

        if self.latest_frame_seq <= self.state_enter_frame_seq or self.latest_bgr is None:
            self.p5_apply_no_image_policy(
                vx, vy, wz, step_height, roll, pitch, body_height,
                log_name or self.state,
            )
            return

        if elapsed < self.p5_center_yellow_ignore_after_enter_s:
            self.p5_send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] ignore center yellow after enter: '
                f'elapsed={elapsed:.3f}/{self.p5_center_yellow_ignore_after_enter_s:.3f}, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        result = self.detect_p5_center_yellow_presence(self.latest_bgr)
        self.latest_p5_center_yellow_presence_result = result

        # 控制循环频率高于相机帧率；同一张图只允许计一次数，
        # 保证 absent_confirm_count 代表连续的新图像帧而不是控制 tick。
        is_new_frame = (
            self.latest_frame_seq != self.p5_center_yellow_last_eval_frame_seq
        )
        if is_new_frame:
            self.p5_center_yellow_last_eval_frame_seq = self.latest_frame_seq

        # 右斜坡主运动阶段专用：根据右侧黄线内侧边缘，对 vy 做三档固定修正。
        # 只影响 P5_RIGHT_SLOPE_1/2/3，不影响转向和额外前进阶段。
        if right_edge_adjust_enabled:
            cmd_vy = self.compute_p5_right_slope_right_edge_corrected_vy(
                base_vy=vy,
                frame=self.latest_bgr,
            )
        else:
            cmd_vy = vy

        # 给可视化保存当前右斜坡阶段实际准备发送的控制命令。
        # draw_p5_right_slope_right_edge_debug() 会把这个 cmd=(vx,vy,wz) 直接画出来。
        if isinstance(getattr(self, 'latest_p5_right_slope_right_edge_result', None), dict):
            self.latest_p5_right_slope_right_edge_result['cmd_vx'] = float(vx)
            self.latest_p5_right_slope_right_edge_result['cmd_vy'] = float(cmd_vy)
            self.latest_p5_right_slope_right_edge_result['cmd_wz'] = float(wz)
            self.latest_p5_right_slope_right_edge_result['cmd_step_height'] = float(step_height)
            self.latest_p5_right_slope_right_edge_result['control_state'] = str(self.state)

        if result.get('has_yellow', False):
            self.p5_center_yellow_absent_counter = 0
            self.p5_send_velocity_command(
                vx=vx,
                vy=cmd_vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] center yellow present, keep moving: '
                f'pixels={result.get("yellow_pixels")}/{self.p5_center_yellow_min_pixels}, '
                f'ratio={float(result.get("yellow_ratio", 0.0)):.4f}/'
                f'{self.p5_center_yellow_min_ratio:.4f}, '
                f'cmd=({vx:.3f},{cmd_vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.5
            )
            return

        if is_new_frame:
            self.p5_center_yellow_absent_counter += 1

        self.get_logger().info(
            f'[{log_name or self.state}] center yellow absent: '
            f'pixels={result.get("yellow_pixels")}/{self.p5_center_yellow_min_pixels}, '
            f'ratio={float(result.get("yellow_ratio", 0.0)):.4f}/'
            f'{self.p5_center_yellow_min_ratio:.4f}, '
            f'counter={self.p5_center_yellow_absent_counter}/'
            f'{self.p5_center_yellow_absent_confirm_count}, '
            f'new_frame={is_new_frame}, frame_seq={self.latest_frame_seq}',
            throttle_duration_sec=0.3
        )

        if (
            self.p5_center_yellow_absent_counter
            >= self.p5_center_yellow_absent_confirm_count
            and not self.p5_route_blocks_exit(log_name or self.state)
        ):
            if stop_before_next:
                self.get_logger().info(
                    f'[{log_name or self.state}] center yellow disappeared, '
                    f'send one zero-velocity command before entering {next_state}'
                )

                # 这里只发送一次速度为 0 的 mode=11/gait=3 命令，清掉上一段右斜坡速度。
                # 不调用 send_stop_command()，保持控制循环非阻塞，
                # 也避免阻塞后 /clock 刷新导致下一个计时状态 elapsed 异常。
                self.p5_send_velocity_command(
                    vx=0.0,
                    vy=0.0,
                    wz=0.0,
                    step_height=0.0,
                    roll=roll,
                    pitch=pitch,
                    body_height=body_height,
                )
            else:
                self.get_logger().info(
                    f'[{log_name or self.state}] center yellow disappeared, '
                    f'go {next_state}'
                )

            # 这里没有阻塞等待，所以直接 enter_state，不走 enter_state_after_blocking_wait。
            self.p5_enter_state(next_state)
            return

        self.p5_send_velocity_command(
            vx=vx,
            vy=cmd_vy,
            wz=wz,
            step_height=step_height,
            roll=roll,
            pitch=pitch,
            body_height=body_height,
        )

    def run_right_side_yellow_lost_velocity_state(
        self,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
        timeout_s: float = 0.0,
    ):
        """
        P5_UP_SLOPE 专用：
        1. 右侧赛道黄线还在图像底部附近 -> 继续上坡；
        2. 上坡过程中，如果左右内侧边缘都有效，则用 center_error 修正 vy，用 heading_error 修正 wz；
        3. 右侧赛道黄线连续 lost N 帧 -> 进入 next_state。

        如果本段声明用里程结束（不依赖黄线），直接转交里程段执行器。
        """
        if self.p5_route_segment_uses_odometry_exit():
            self.run_odometry_distance_velocity_state(
                vx=vx, vy=vy, wz=wz, step_height=step_height,
                next_state=next_state, roll=roll, pitch=pitch,
                body_height=body_height, log_name=log_name, timeout_s=timeout_s,
            )
            return

        if self.p5_vision_state_guard(timeout_s, log_name or self.state):
            return

        elapsed = self.p5_state_elapsed_s()

        if self.latest_frame_seq <= self.state_enter_frame_seq or self.latest_bgr is None:
            self.p5_apply_no_image_policy(
                vx, vy, wz, step_height, roll, pitch, body_height,
                log_name or self.state,
            )
            return

        if elapsed < self.p5_right_side_yellow_ignore_after_enter_s:
            self.p5_send_velocity_command(
                vx=vx,
                vy=vy,
                wz=wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )
            self.get_logger().info(
                f'[{log_name or self.state}] ignore right-side yellow after enter: '
                f'elapsed={elapsed:.3f}/{self.p5_right_side_yellow_ignore_after_enter_s:.3f}, '
                f'cmd=({vx:.3f},{vy:.3f},{wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        result = self.detect_p5_right_side_yellow_line(self.latest_bgr)
        self.latest_p5_right_side_yellow_result = result
        is_new_frame = (
            self.latest_frame_seq != self.p5_right_side_yellow_last_eval_frame_seq
        )
        if is_new_frame:
            self.p5_right_side_yellow_last_eval_frame_seq = self.latest_frame_seq

        # 上坡内侧边缘修正：不负责结束，只负责改 vy / wz。
        cmd_vy, cmd_wz = self.compute_p5_up_slope_inner_edge_corrected_cmd(
            base_vy=vy,
            base_wz=wz,
            frame=self.latest_bgr,
        )

        if result.get('has_line', False):
            self.p5_right_side_yellow_lost_counter = 0

            self.p5_send_velocity_command(
                vx=vx,
                vy=cmd_vy,
                wz=cmd_wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            self.get_logger().info(
                f'[{log_name or self.state}] right-side yellow valid, keep moving: '
                f'bottom={result.get("bottom_y")}, '
                f'ratio={float(result.get("bottom_ratio", 0.0)):.3f}/'
                f'{self.p5_right_side_yellow_bottom_valid_ratio:.3f}, '
                f'bbox={result.get("bbox")}, '
                f'cmd=({vx:.3f},{cmd_vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.5
            )
            return

        # 控制循环频率高于相机帧率；同一张图只能计数一次，确保这里的
        # lost_confirm_count 确实代表连续的新图像，而不是连续控制 tick。
        if is_new_frame:
            self.p5_right_side_yellow_lost_counter += 1

        ratio = result.get('bottom_ratio')
        ratio_text = 'None' if ratio is None else f'{float(ratio):.3f}'

        self.get_logger().info(
            f'[{log_name or self.state}] right-side yellow invalid/lost: '
            f'reason={result.get("reason")}, '
            f'bottom={result.get("bottom_y")}, '
            f'ratio={ratio_text}, '
            f'counter={self.p5_right_side_yellow_lost_counter}/'
            f'{self.p5_right_side_yellow_lost_confirm_count}, '
            f'new_frame={is_new_frame}, frame_seq={self.latest_frame_seq}, '
            f'candidates={len(result.get("candidates", []))}',
            throttle_duration_sec=0.3
        )

        if (
            self.p5_right_side_yellow_lost_counter
            >= self.p5_right_side_yellow_lost_confirm_count
            and not self.p5_route_blocks_exit(log_name or self.state)
        ):
            self.get_logger().info(
                f'[{log_name or self.state}] right-side yellow disappeared, '
                f'go {next_state} directly'
            )
            self.p5_enter_state(next_state)
            return

        self.p5_send_velocity_command(
            vx=vx,
            vy=cmd_vy,
            wz=cmd_wz,
            step_height=step_height,
            roll=roll,
            pitch=pitch,
            body_height=body_height,
        )

    def p5_forward_inner_edge_aligned(self) -> bool:
        """
        判断右跳后平地前进阶段是否已经居中和角度对齐。
        注意：这里不重新检测，只读取最近一次 compute_p5_up_slope_inner_edge_corrected_cmd()
        更新后的 latest_p5_inner_edge_result。
        """
        if not self.p5_yellow_lateral_correction_enabled:
            # 里程主导模式下没有黄线内侧边缘结果可读；
            # 该段退化为纯定时前进（hold_after_duration_until_aligned 需为 False）。
            return True

        result = self.latest_p5_inner_edge_result
        if not result or not result.get('common_valid', False):
            return False

        center_error = result.get('center_error', None)
        heading_error = result.get('heading_error', None)
        if center_error is None or heading_error is None:
            return False

        center_ok = abs(float(center_error)) <= self.p5_forward_after_reset_body_align_center_done_px
        heading_ok = abs(float(heading_error)) <= self.p5_forward_after_reset_body_align_heading_done_px
        return bool(center_ok and heading_ok)

    def run_timed_velocity_then_stop_state(
        self,
        duration_s: float,
        vx: float,
        vy: float,
        wz: float,
        step_height: float,
        next_state: str,
        roll: float = 0.0,
        pitch: float = 0.0,
        body_height: float = 0.25,
        log_name: str = '',
        use_inner_edge_align: bool = False,
        hold_after_duration_until_aligned: bool = False,
        stop_when_done: bool = True,
    ):
        """
        持续发送速度命令 duration_s 秒。

        普通模式：duration_s 到时后根据 stop_when_done 决定是否 stop，然后进入 next_state。

        hold_after_duration_until_aligned=True 时：
        - duration_s 之前：正常 vx 前进，同时可叠加内侧边缘 vy/wz 矫正；
        - duration_s 之后：vx 置 0，只保留 vy/wz 矫正；
        - 连续稳定若干帧满足 center_error / heading_error 阈值后，根据 stop_when_done 决定是否 stop，然后进入 next_state。
        """
        if not use_inner_edge_align and self.p5_timed_motion_guard(
            duration_s, log_name or self.state
        ):
            return
        if use_inner_edge_align:
            if self.p5_vision_state_guard(
                self.p5_forward_after_reset_body_timeout_s,
                log_name or self.state,
            ):
                return
            if self.latest_frame_seq <= self.state_enter_frame_seq or self.latest_bgr is None:
                self.p5_apply_no_image_policy(
                    vx, vy, wz, step_height, roll, pitch, body_height,
                    log_name or self.state,
                )
                return

        elapsed = self.p5_state_elapsed_s()

        cmd_vy = vy
        cmd_wz = wz

        if use_inner_edge_align and self.latest_bgr is not None:
            cmd_vy, cmd_wz = self.compute_p5_up_slope_inner_edge_corrected_cmd(
                base_vy=vy,
                base_wz=wz,
                frame=self.latest_bgr,
            )
        elif use_inner_edge_align and self.latest_bgr is None:
            self.get_logger().warn(
                f'[{log_name or self.state}] no image for inner-edge align, use base vy/wz',
                throttle_duration_sec=0.5
            )

        if elapsed < duration_s:
            self.p5_send_velocity_command(
                vx=vx,
                vy=cmd_vy,
                wz=cmd_wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            self.get_logger().info(
                f'[{log_name or self.state}] timed velocity: '
                f'elapsed={elapsed:.3f}/{duration_s:.3f}s, '
                f'cmd=({vx:.3f},{cmd_vy:.3f},{cmd_wz:.3f}), '
                f'align={use_inner_edge_align}, '
                f'step_height={step_height:.3f}, '
                f'roll={roll:.3f}, pitch={pitch:.3f}, body_height={body_height:.3f}',
                throttle_duration_sec=0.3
            )
            return

        if hold_after_duration_until_aligned and use_inner_edge_align:
            # 固定前进结束后，不再继续向前冲。只用 vy/wz 做居中和角度矫正。
            edge_result = self.latest_p5_inner_edge_result or {}
            common_valid = bool(edge_result.get('common_valid', False))
            is_new_frame = (
                self.latest_frame_seq != self.p5_forward_align_last_eval_frame_seq
            )
            if is_new_frame:
                self.p5_forward_align_last_eval_frame_seq = self.latest_frame_seq
                if common_valid:
                    self.p5_forward_align_lost_counter = 0
                else:
                    self.p5_forward_align_lost_counter += 1

            lost_confirmed = (
                self.latest_bgr is None or
                self.p5_forward_align_lost_counter >=
                self.p5_forward_after_reset_body_align_lost_confirm_frames
            )
            if lost_confirmed:
                self.get_logger().warn(
                    f'[{log_name or self.state}] inner edges lost after forward duration, '
                    f'go {next_state}: '
                    f'lost={self.p5_forward_align_lost_counter}/'
                    f'{self.p5_forward_after_reset_body_align_lost_confirm_frames}, '
                    f'new_frame={is_new_frame}, '
                    f'reason={edge_result.get("common_reason")}, '
                    f'stop_when_done={stop_when_done}'
                )
                self.p5_finish_after_optional_stop(
                    next_state, log_name or self.state, stop_when_done)
                return

            aligned = self.p5_forward_inner_edge_aligned()
            if is_new_frame:
                if aligned:
                    self.p5_forward_align_stable_counter += 1
                else:
                    self.p5_forward_align_stable_counter = 0

            max_extra_s = self.p5_forward_after_reset_body_align_max_extra_s
            max_extra_reached = (
                max_extra_s > 0.0 and
                elapsed >= duration_s + max_extra_s
            )

            if self.p5_forward_align_stable_counter >= self.p5_forward_after_reset_body_align_stable_frames:
                self.get_logger().info(
                    f'[{log_name or self.state}] forward duration done and align stable, '
                    f'go {next_state}: '
                    f'stable={self.p5_forward_align_stable_counter}/'
                    f'{self.p5_forward_after_reset_body_align_stable_frames}, '
                    f'stop_when_done={stop_when_done}'
                )
                self.p5_finish_after_optional_stop(
                    next_state, log_name or self.state, stop_when_done)
                return

            if max_extra_reached:
                self.get_logger().warn(
                    f'[{log_name or self.state}] align max extra time reached, '
                    f'go {next_state}: elapsed={elapsed:.3f}, '
                    f'duration={duration_s:.3f}, max_extra={max_extra_s:.3f}, '
                    f'stop_when_done={stop_when_done}'
                )
                self.p5_finish_after_optional_stop(
                    next_state, log_name or self.state, stop_when_done)
                return

            self.p5_send_velocity_command(
                vx=0.0,
                vy=cmd_vy,
                wz=cmd_wz,
                step_height=step_height,
                roll=roll,
                pitch=pitch,
                body_height=body_height,
            )

            self.get_logger().info(
                f'[{log_name or self.state}] duration done, hold vx=0 for align: '
                f'elapsed={elapsed:.3f}/{duration_s:.3f}s, '
                f'center_error={edge_result.get("center_error")}, '
                f'heading_error={edge_result.get("heading_error")}, '
                f'common_valid={edge_result.get("common_valid")}, '
                f'reason={edge_result.get("common_reason")}, '
                f'stable={self.p5_forward_align_stable_counter}/'
                f'{self.p5_forward_after_reset_body_align_stable_frames}, '
                f'cmd=(0.000,{cmd_vy:.3f},{cmd_wz:.3f})',
                throttle_duration_sec=0.3
            )
            return

        self.get_logger().info(
            f'[{log_name or self.state}] timed velocity done, '
            f'go {next_state}, stop_when_done={stop_when_done}'
        )
        self.p5_finish_after_optional_stop(
            next_state, log_name or self.state, stop_when_done)

    def p5_finish_after_optional_stop(
        self, next_state: str, log_name: str, stop_when_done: bool
    ):
        """Enter next_state only after a fresh STOP ACK when requested."""
        if not stop_when_done:
            self.p5_enter_state(next_state)
            return
        if self.p5_action_phase == 'idle':
            self.get_logger().info(
                f'[{log_name}] begin non-blocking STOP before {next_state}')
            self.p5_begin_action_poll(12, 0, 'timed_stop')
            return
        if self.p5_action_phase != 'timed_stop':
            self.get_logger().error(
                f'[{log_name}] invalid STOP phase={self.p5_action_phase}')
            self.p5_action_timeout_fault(
                log_name, self.p5_safety_elapsed_s(),
                self.Ctrl.response_snapshot())
            return
        status, elapsed, snapshot = self.p5_poll_action(self.p5_stop_timeout_s)
        if status == 'timeout':
            self.p5_action_timeout_fault(log_name, elapsed, snapshot)
            return
        if status == 'complete':
            self.get_logger().info(
                f'[{log_name}] STOP acknowledged, go {next_state}')
            self.p5_enter_state(next_state)

    def p5_begin_action_poll(self, mode: int, gait_id: int, phase: str):
        """Atomically send once and capture the post-publish response barrier."""
        self.p5_action_target = (int(mode), int(gait_id))
        self.p5_action_phase = str(phase)
        self.p5_action_progress_seen = False
        self.p5_action_completed_monotonic_s = None
        self.p5_stop_complete_seq = 0
        self.p5_stop_complete_rx_monotonic_s = None
        # In the legacy LCM controller (12,0) was overloaded as both STOP
        # and RecoveryStand.  Keep RecoveryStand for phase='action', but when
        # the state machine explicitly requests a stop, use SERVO_END on the
        # physical robot and synthesize the legacy ACK only for this poller.
        if (
            getattr(self.Ctrl, 'is_real', False)
            and (int(mode), int(gait_id)) == (12, 0)
            and str(phase) in ('timed_stop', 'stop')
        ):
            barrier = self.Ctrl.stop_motion_with_response_barrier((12, 0))
        else:
            barrier = self.p5_send_action_once(
                mode, gait_id, response_barrier=True)
        self.p5_action_response_seq = int(barrier[0])
        self.p5_action_sent_monotonic_s = float(barrier[1])
        self.p5_action_resends_done = 0
        self.p5_action_last_send_monotonic_s = self.p5_action_sent_monotonic_s
        self.p5_action_recovery_pending = False
        self.p5_action_stall_since_monotonic_s = None
        self.p5_action_stall_bar = None
        self.p5_action_unwedge_phase = ''
        self.p5_action_unwedge_done = False
        self.p5_action_unwedge_origin = None
        self.p5_action_unwedge_sent_monotonic_s = None
        self.action_sent = True

    def p5_poll_action(self, timeout_s: float):
        """Return pending/complete/timeout without blocking the ROS executor."""
        now = time.monotonic()
        elapsed = max(0.0, now - self.p5_action_sent_monotonic_s)
        snapshot = self.Ctrl.response_snapshot()
        if elapsed >= timeout_s:
            return "timeout", elapsed, snapshot
        mode, gait_id = self.p5_action_target
        rx_time = snapshot.get("rx_monotonic_s")
        fresh = (
            int(snapshot["seq"]) > self.p5_action_response_seq
            and rx_time is not None
            and float(rx_time) > self.p5_action_sent_monotonic_s
        )
        progress_rx_time = snapshot.get("last_incomplete_rx_monotonic_s")
        progress_fresh = (
            int(snapshot.get("last_incomplete_seq", 0))
            > self.p5_action_response_seq
            and progress_rx_time is not None
            and float(progress_rx_time) > self.p5_action_sent_monotonic_s
            and int(snapshot.get("last_incomplete_mode", -1)) == mode
            and int(snapshot.get("last_incomplete_gait_id", -1)) == gait_id
        )
        if progress_fresh:
            self.p5_action_progress_seen = True
        matching_complete = (
            fresh
            and elapsed >= self.p5_action_min_ack_delay_s
            and int(snapshot["mode"]) == mode
            and int(snapshot["gait_id"]) == gait_id
            and int(snapshot["order_process_bar"]) >= 95
        )
        if matching_complete and (
            not self.p5_action_require_progress
            or self.p5_action_progress_seen
        ):
            return "complete", elapsed, snapshot

        # STOP 是幂等安全目标。控制器若发送命令前已经处于完成的 STOP，
        # 不会产生新的 incomplete 边沿；此时要求两个 barrier 后持续发布的
        # 完成包，既避免单个延迟包误确认，也不会让同目标 STOP 永久超时。
        if matching_complete and (mode, gait_id) == (12, 0):
            snapshot_seq = int(snapshot["seq"])
            snapshot_rx_time = float(snapshot["rx_monotonic_s"])
            if self.p5_stop_complete_seq <= self.p5_action_response_seq:
                self.p5_stop_complete_seq = snapshot_seq
                self.p5_stop_complete_rx_monotonic_s = snapshot_rx_time
                return "pending", elapsed, snapshot
            if (
                snapshot_seq > self.p5_stop_complete_seq
                and snapshot_rx_time - self.p5_stop_complete_rx_monotonic_s
                >= self.p5_action_min_ack_delay_s
            ):
                return "complete", elapsed, snapshot

        # 卡死解楔（计划 33 条）。与下面的拒单救援互斥：救援处理"回显不是本
        # 目标"，解楔处理"回显正是本目标却永远不完成"。必须排在救援之前——
        # 解楔发出的 kOff 会让控制器回显 mode=0，落到救援的判据里就会把跳跃
        # 重发一次。
        if self.p5_action_unwedge_poll(now, elapsed, snapshot, fresh):
            return "pending", elapsed, snapshot

        # 拒单救援（计划 19 条）。前提判据必须同时成立：
        # - 响应流在发送之后仍在前进（fresh）——控制器活着、链路通畅
        #   （链路死了是超时的职责，救援反而会在链路恢复时重复排队）；
        # - 回显 mode 不是本目标——动作没有开始执行（执行中绝不重发，
        #   跳跃动作重发会二次起跳）；
        # - 从未见过本目标的 incomplete 边沿——排除"开始过但回报滞后"。
        #
        # 实测（三批 + 冒烟共 6 次超时）回显分两类：
        # - mode 7/9（kPureDamper/kLifted，切换状态 3/4）共 5 次：保护态。
        #   FSM 源码（fsm_state_pure_damper/lifted.cpp CheckTransition）只
        #   接受 12（恢复站立）等少数目标，跳跃 16 是 "Bad Request"，直接
        #   重发已实测无效（life_count 更新后依旧拒收）。正确的梯子是：
        #   先发 12，等恢复站立完成（回显 12 且 bar>=95），再重发原动作
        #   （RecoveryStand 的 CheckTransition 接受 kJump3d）。
        # - 其他 mode 不匹配（切换竞态/真丢单）：直接重发一次。
        if (
            self.p5_action_resend_max > 0
            and not self.p5_action_progress_seen
            and fresh
            and int(snapshot["mode"]) != mode
            and self.p5_action_last_send_monotonic_s is not None
        ):
            echoed_mode = int(snapshot["mode"])
            since_send = now - self.p5_action_last_send_monotonic_s
            if self.p5_action_recovery_pending:
                if (
                    echoed_mode == 12
                    and int(snapshot["order_process_bar"]) >= 95
                ):
                    self.p5_action_recovery_pending = False
                    self.p5_action_resends_done += 1
                    self.p5_action_last_send_monotonic_s = now
                    self.get_logger().warn(
                        f'[P5_ACTION] resend {self.p5_action_resends_done}/'
                        f'{self.p5_action_resend_max} after recovery stand: '
                        f'target=({mode}, {gait_id}), elapsed={elapsed:.2f}s')
                    self.p5_evidence_log({
                        'event': 'action_resend',
                        'state': str(self.state),
                        'target': [int(mode), int(gait_id)],
                        'resend': int(self.p5_action_resends_done),
                        'after_recovery': True,
                        'elapsed_s': float(elapsed),
                        'response': snapshot,
                    })
                    self.p5_send_action_once(mode, gait_id)
            elif (
                self.p5_action_resends_done < self.p5_action_resend_max
                and since_send >= self.p5_action_resend_after_s
            ):
                if echoed_mode in (7, 9) and (mode, gait_id) != (12, 0):
                    # kPureDamper / kLifted：先解保护，不直接重发。
                    self.p5_action_recovery_pending = True
                    self.p5_action_last_send_monotonic_s = now
                    self.get_logger().warn(
                        f'[P5_ACTION] recovery preface for target='
                        f'({mode}, {gait_id}): controller in protection '
                        f'mode={echoed_mode} '
                        f'(switch_status={snapshot.get("switch_status")}), '
                        f'sending recovery stand first, '
                        f'elapsed={elapsed:.2f}s')
                    self.p5_evidence_log({
                        'event': 'action_recovery_preface',
                        'state': str(self.state),
                        'target': [int(mode), int(gait_id)],
                        'echoed_mode': echoed_mode,
                        'elapsed_s': float(elapsed),
                        'response': snapshot,
                    })
                    self.p5_send_action_once(12, 0)
                else:
                    self.p5_action_resends_done += 1
                    self.p5_action_last_send_monotonic_s = now
                    self.get_logger().warn(
                        f'[P5_ACTION] resend {self.p5_action_resends_done}/'
                        f'{self.p5_action_resend_max}: '
                        f'target=({mode}, {gait_id}), controller echoes '
                        f'mode={echoed_mode} {elapsed:.2f}s after send')
                    self.p5_evidence_log({
                        'event': 'action_resend',
                        'state': str(self.state),
                        'target': [int(mode), int(gait_id)],
                        'resend': int(self.p5_action_resends_done),
                        'after_recovery': False,
                        'elapsed_s': float(elapsed),
                        'response': snapshot,
                    })
                    self.p5_send_action_once(mode, gait_id)
        return "pending", elapsed, snapshot

    @staticmethod
    def p5_action_is_stalled(snapshot: dict, target, progress_seen: bool,
                             fresh: bool) -> bool:
        """True when the controller echoes our own target and never finishes.

        This is the accepted-but-stalled signature of plan item 33, and it is
        the exact complement of the refusal the resend ladder handles: the
        action was received and entered (a fresh incomplete edge carried our
        mode and gait), the controller is still publishing, it is still
        reporting our target — and `order_process_bar` never leaves 0.
        """
        if target is None or not fresh or not progress_seen:
            return False
        mode, gait_id = target
        return (
            int(snapshot.get('mode', -1)) == mode
            and int(snapshot.get('gait_id', -1)) == gait_id
            and int(snapshot.get('order_process_bar', 0)) < 95
        )

    def p5_action_unwedge_poll(self, now: float, elapsed: float,
                               snapshot: dict, fresh: bool) -> bool:
        """Release a controller wedged inside an accepted-but-stalled action.

        Returns True when this tick belongs to the unwedge ladder and the
        caller must report `pending` without running the resend ladder.

        Why kOff and nothing else: `FsmStateJump3d::Transition()` gates every
        exit except `kOff` on `data_end_ && touch_down_ &&
        height_good_for_trans_`, the same conditions that produce the progress
        bar. A jump whose landing does not satisfy them can therefore never be
        left — kPureDamper and kRecoveryStand are accepted by CheckTransition
        and then hang in transition forever. The `kOff` arm sets
        `transition_data_.done = true` unconditionally, so mode 0 is the only
        command that can free the FSM. It is the same escape the gamepad's
        back button maps to.

        Going limp is a real cost, so the ladder immediately stands the robot
        back up and only then reports the action complete — and it reports it
        as the recovery stand it actually is, by rewriting the action target.
        Nothing is declared complete unless the robot got back on its feet.
        """
        if self.p5_action_stall_unwedge_after_s <= 0.0:
            return False

        if self.p5_action_unwedge_phase == 'off_sent':
            if int(snapshot.get('mode', -1)) == 0:
                self.p5_action_unwedge_retarget_to_recovery(snapshot)
                return True
            waited_s = max(
                0.0, now - float(self.p5_action_unwedge_sent_monotonic_s))
            if waited_s >= self.p5_action_unwedge_release_timeout_s:
                # kOff 也没被接住：不再干预，把判决交给动作超时。
                self.p5_action_unwedge_phase = 'off_refused'
                self.get_logger().error(
                    '[P5_ACTION] unwedge release refused: controller still '
                    f'echoes mode={snapshot.get("mode")} '
                    f'{waited_s:.2f}s after kOff')
                self.p5_evidence_log({
                    'event': 'action_unwedge_refused',
                    'state': str(self.state),
                    'origin_target': list(self.p5_action_unwedge_origin or []),
                    'waited_s': float(waited_s),
                    'response': snapshot,
                })
                return False
            return True

        if self.p5_action_unwedge_phase:
            # 'off_refused' 是终态；不再重复发 kOff。
            return False

        if not self.p5_action_is_stalled(
                snapshot, self.p5_action_target,
                self.p5_action_progress_seen, fresh):
            self.p5_action_stall_since_monotonic_s = None
            self.p5_action_stall_bar = None
            return False

        # A wedge is a *pinned* bar, not merely an unfinished one. Restart the
        # clock whenever the bar moves: 2026-08-06 measured the ladder firing
        # twice on a recovery stand sitting below 95, and RecoveryStand from
        # kLifted legitimately takes 17.7 s while climbing 20/40/50/60/80 --
        # far past a window calibrated on jump acknowledgements (4.26 s worst).
        # A genuine wedge holds one value forever, so this can only suppress
        # firings on actions that are still making progress.
        bar = int(snapshot.get('order_process_bar', 0))
        if self.p5_action_stall_bar != bar:
            self.p5_action_stall_bar = bar
            self.p5_action_stall_since_monotonic_s = now
            return False

        if self.p5_action_stall_since_monotonic_s is None:
            self.p5_action_stall_since_monotonic_s = now
            return False
        stalled_s = now - self.p5_action_stall_since_monotonic_s
        if stalled_s < self.p5_action_stall_unwedge_after_s:
            return False
        if self.p5_action_unwedge_done:
            # 每个动作只解楔一次。第二次卡死不是握手问题。
            return False

        self.p5_action_unwedge_origin = tuple(self.p5_action_target)
        self.p5_action_unwedge_phase = 'off_sent'
        self.p5_action_unwedge_sent_monotonic_s = now
        self.get_logger().warn(
            '[P5_ACTION] stalled action unwedge: target='
            f'{self.p5_action_unwedge_origin} echoed back with '
            f'order_process_bar={snapshot.get("order_process_bar")} for '
            f'{stalled_s:.2f}s, releasing with kOff, elapsed={elapsed:.2f}s')
        self.p5_evidence_log({
            'event': 'action_unwedge_release',
            'state': str(self.state),
            'origin_target': list(self.p5_action_unwedge_origin),
            'stalled_s': float(stalled_s),
            'elapsed_s': float(elapsed),
            'response': snapshot,
        })
        self.p5_send_action_once(0, 0)
        return True

    def p5_action_unwedge_retarget_to_recovery(self, snapshot: dict):
        """Stand up after a kOff release and finish as that recovery stand."""
        origin = self.p5_action_unwedge_origin
        self.get_logger().warn(
            f'[P5_ACTION] unwedge released target={origin}; '
            'standing up and completing as recovery stand')
        self.p5_evidence_log({
            'event': 'action_unwedge_recover',
            'state': str(self.state),
            'origin_target': list(origin or []),
            'response': snapshot,
        })
        barrier = self.p5_send_action_once(12, 0, response_barrier=True)
        self.p5_action_target = (12, 0)
        self.p5_action_response_seq = int(barrier[0])
        self.p5_action_sent_monotonic_s = float(barrier[1])
        self.p5_action_progress_seen = False
        self.p5_action_completed_monotonic_s = None
        self.p5_stop_complete_seq = 0
        self.p5_stop_complete_rx_monotonic_s = None
        self.p5_action_resends_done = 0
        self.p5_action_last_send_monotonic_s = self.p5_action_sent_monotonic_s
        self.p5_action_recovery_pending = False
        self.p5_action_stall_since_monotonic_s = None
        self.p5_action_stall_bar = None
        self.p5_action_unwedge_phase = ''
        self.p5_action_unwedge_done = True
        self.action_sent = True

    def p5_action_release_wedged_controller(self, snapshot: dict) -> bool:
        """Send kOff when the timeout leaves the controller inside our action.

        Unlike the unwedge ladder above, this is not gated on a parameter and
        never resumes the run. The run is already lost when it is called; the
        only remaining question is whether we hand back a robot that can still
        be commanded. A controller wedged in `kJump3d` ignores STOP,
        kPureDamper and kRecoveryStand alike and holds the body rigid, which
        in the simulator costs every remaining run of a batch and on the
        physical robot leaves nothing short of a power cycle. Sending mode 0
        costs a limp drop from whatever pose the stall froze; the STOP that
        follows immediately stands the robot back up.
        """
        # A stale snapshot cannot be distinguished from a live wedge here and
        # does not need to be: if the link is dead the kOff goes nowhere.
        if not self.p5_action_is_stalled(
                snapshot, self.p5_action_target,
                progress_seen=True, fresh=True):
            return False
        self.p5_send_action_once(0, 0)
        self.p5_evidence_log({
            'event': 'action_timeout_release',
            'state': str(self.state),
            'target': list(self.p5_action_target or []),
            'response': snapshot,
        })
        return True

    @staticmethod
    def p5_response_is_damped(snapshot: dict) -> bool:
        """True for the controller's post-fall protection signature.

        ``mode 7`` (kPureDamper) with ``switch_status 3`` (kEdamp) is what
        ``control_fsm.cpp`` publishes once ``SafetyPostCheck()`` has rejected
        the commanded foot positions: it forces ``current_state_`` to the pure
        damper and refuses to read ``CheckTransition()`` again until
        ``edamp_iter > 1550``, which a still-fallen robot resets on every
        attempted stand.
        """
        return (int(snapshot.get('mode', -1)) == 7
                and int(snapshot.get('switch_status', 0)) == 3)

    def p5_fall_recovery_body_down_rp(self):
        """Return (roll, pitch) when odometry proves the body is down.

        The kOff that starts the ladder is only safe because the body is
        already on the ground.  Echo alone cannot establish that -- kEdamp can
        also latch on a workspace violation with the robot still standing, and
        dropping *that* robot would cause the fall it is meant to recover.
        """
        if self.Odom is None:
            return None
        snapshot = self.Odom.snapshot()
        if snapshot['seq'] <= 0 or snapshot['rx_monotonic_s'] is None:
            return None
        age = max(0.0, time.monotonic() - float(snapshot['rx_monotonic_s']))
        if age > self.p5_route_odom_max_age_s:
            return None
        roll = float(snapshot['rpy'][0])
        pitch = float(snapshot['rpy'][1])
        if max(abs(roll), abs(pitch)) < self.p5_fall_recovery_min_rp_rad:
            return None
        return roll, pitch

    def p5_odom_attitude_rp(self):
        """Return the latest odometry (roll, pitch), or (nan, nan) when stale.

        NaN rather than zero: a missing attitude must not read as "upright" to
        a gate whose whole job is to notice that the body is not.
        """
        nan = float('nan')
        if self.Odom is None:
            return nan, nan
        snapshot = self.Odom.snapshot()
        if snapshot['seq'] <= 0 or snapshot['rx_monotonic_s'] is None:
            return nan, nan
        age = max(0.0, time.monotonic() - float(snapshot['rx_monotonic_s']))
        if age > self.p5_route_odom_max_age_s:
            return nan, nan
        return float(snapshot['rpy'][0]), float(snapshot['rpy'][1])

    def p5_fall_recovery_begin(self, log_name: str, snapshot: dict) -> bool:
        """Divert a fallen-robot timeout into the bounded pick-up ladder."""
        if not self.p5_fall_recovery_enabled:
            return False
        if self.p5_fall_recovery_attempts >= self.p5_fall_recovery_max_attempts:
            return False
        if not self.p5_response_is_damped(snapshot):
            return False
        down = self.p5_fall_recovery_body_down_rp()
        if down is None:
            return False
        roll, pitch = down
        self.p5_fall_recovery_attempts += 1
        retry_state = str(self.state)
        self.get_logger().warn(
            f'[{log_name}] fallen at {retry_state} '
            f'(roll={roll:+.2f} pitch={pitch:+.2f} rad, controller in kEdamp); '
            f'attempt {self.p5_fall_recovery_attempts}/'
            f'{self.p5_fall_recovery_max_attempts} to stand it back up')
        self.p5_evidence_log({
            'event': 'fall_recovery_begin',
            'state': retry_state,
            'target': list(self.p5_action_target or []),
            'roll_rad': float(roll),
            'pitch_rad': float(pitch),
            'attempt': int(self.p5_fall_recovery_attempts),
            'response': snapshot,
        })
        self.p5_enter_state(self.P5_FALL_RECOVER)
        # p5_enter_state() clears per-state action fields, so the retry context
        # is installed after it, not before.
        self.p5_fall_recover_retry_state = retry_state
        self.p5_fall_recover_phase = ''
        self.p5_fall_recover_since_monotonic_s = None
        return True

    def p5_fall_recovery_fault(self, reason: str):
        """End the ladder without resuming; the robot is standing by then."""
        self.p5_send_stop_command()
        self.action_sent = True
        self.get_logger().error(
            f'[P5_FALL_RECOVER] giving up: {reason}, '
            f'retry_state={self.p5_fall_recover_retry_state}')
        self.p5_evidence_log({
            'event': 'fall_recovery_fault',
            'state': str(self.p5_fall_recover_retry_state),
            'reason': str(reason),
        })
        self.p5_enter_state(self.P5_SENSOR_FAULT_HOLD)
        self.action_sent = True

    def p5_run_fall_recover(self):
        """kOff -> recovery stand -> retry the corner once, or fail closed."""
        now = time.monotonic()
        snapshot = self.Ctrl.response_snapshot()

        if (self.p5_fall_recover_phase != ''
                and self.p5_safety_elapsed_s()
                >= self.p5_fall_recovery_total_timeout_s):
            self.p5_fall_recovery_fault(
                f'pick-up budget spent '
                f'({self.p5_safety_elapsed_s():.1f}s >= '
                f'{self.p5_fall_recovery_total_timeout_s:.1f}s)')
            return

        if self.p5_fall_recover_phase == '':
            # kOff is the only exit control_fsm.cpp leaves open from kEdamp.
            self.p5_send_action_once(0, 0)
            self.p5_fall_recover_phase = 'off_sent'
            self.p5_fall_recover_since_monotonic_s = now
            return

        elapsed = max(0.0, now - self.p5_fall_recover_since_monotonic_s)

        if self.p5_fall_recover_phase == 'off_sent':
            if not self.p5_response_is_damped(snapshot):
                self.get_logger().info(
                    f'[P5_FALL_RECOVER] released from kEdamp after '
                    f'{elapsed:.2f}s, standing up')
                self.p5_begin_action_poll(12, 0, 'action')
                self.p5_fall_recover_phase = 'standing'
                self.p5_fall_recover_since_monotonic_s = now
                return
            if elapsed >= self.p5_fall_recovery_release_timeout_s:
                self.p5_fall_recovery_fault(
                    f'kOff refused, still damped after {elapsed:.2f}s')
            return

        if self.p5_fall_recover_phase == 'standing':
            status, _, stand_snapshot = self.p5_poll_action(
                self.p5_fall_recovery_stand_timeout_s)
            if status == 'timeout':
                self.p5_fall_recovery_fault(
                    f'recovery stand timed out after '
                    f'{self.p5_fall_recovery_stand_timeout_s:.1f}s '
                    f'({elapsed:.1f}s into the pick-up)')
                return
            if status != 'complete':
                return
            self.p5_fall_recover_finish(stand_snapshot)

    def p5_stall_recovery_begin(self, log_name: str, progress_m: float) -> bool:
        """Divert a stalled segment into a bounded stand-up-and-resume.

        This is deliberately *not* ``P5_FALL_RECOVER``.  That path holds rather
        than resuming because a tumble destroys the leg odometry's position
        state (plan item 37), so nothing on board can answer "am I still on the
        course".  A stall is the opposite case: the robot never went over, its
        attitude is level and its position estimate is exactly as good as it
        was a moment ago — it is simply splayed and pushing against a step.
        Resuming is sound *because* of that, so the check is enforced here
        rather than assumed: a body that is actually down is refused, and falls
        through to the caller's fault.
        """
        if self.p5_stall_recover_attempts >= self.p5_route_stall_max_attempts:
            return False
        roll, pitch = self.p5_odom_attitude_rp()
        if roll != roll or pitch != pitch:
            return False                       # no attitude: do not guess
        if max(abs(roll), abs(pitch)) >= self.p5_fall_recovery_min_rp_rad:
            return False                       # actually down: not this ladder
        self.p5_stall_recover_attempts += 1
        self.p5_stall_recover_resume_state = str(self.state)
        self.get_logger().warn(
            f'[{log_name}] stalled at {progress_m:.2f} m with vx commanded '
            f'(roll={roll:+.2f} pitch={pitch:+.2f} rad, body is level); '
            f'attempt {self.p5_stall_recover_attempts}/'
            f'{self.p5_route_stall_max_attempts} to stand and resume')
        self.p5_evidence_log({
            'event': 'stall_recovery_begin',
            'state': str(self.state),
            'progress_m': float(progress_m),
            'roll_rad': roll,
            'pitch_rad': pitch,
        })
        self.p5_enter_state(self.P5_STALL_RECOVER)
        return True

    def p5_run_stall_recover(self):
        """Recovery stand, then re-enter the stalled state, or fail closed."""
        if not self.action_sent:
            self.p5_begin_action_poll(12, 0, 'action')
            self.action_sent = True
            return
        status, _, snapshot = self.p5_poll_action(
            self.p5_fall_recovery_stand_timeout_s)
        if status == 'timeout':
            self.p5_route_fault('P5_STALL_RECOVER', 'route_stall_fault', {
                'reason': 'recovery stand timed out',
                'resume_state': self.p5_stall_recover_resume_state,
            })
            return
        if status != 'complete':
            return
        resume = self.p5_stall_recover_resume_state or self.P5_SENSOR_FAULT_HOLD
        self.get_logger().warn(
            f'[P5_STALL_RECOVER] stood up, resuming {resume}')
        self.p5_evidence_log({
            'event': 'stall_recovery_resumed',
            'state': resume,
            'response': snapshot,
        })
        self.p5_enter_state(resume)

    def p5_fall_recover_finish(self, snapshot: dict):
        """Report the outcome of the pick-up and hold; never resume the route.

        The run is over either way.  What this buys is a robot left standing
        and commandable instead of rigid on its side inside a latched kEdamp,
        which on the physical robot is the difference between a reset and a
        power cycle.

        It deliberately does *not* retry the interrupted corner.  Measured
        2026-08-07: after ~30 s of tumbling and righting the leg odometry's
        position state is meaningless, so nothing on board can answer "am I
        still on the rail". Both pick-ups in that batch stood the robot up
        0.29 m outside the rail's inner edge while odometry reported 0.024 and
        0.025 m of cross-rail displacement, and the one that resumed walked the
        remaining route across the floor to a false ``P5_DONE``. A false
        completion publishes ``stage_complete`` and advances the mission; a
        fault does not.
        """
        down = self.p5_fall_recovery_body_down_rp()
        if down is not None:
            self.p5_fall_recovery_fault(
                'stand reported complete but the body is still down '
                f'(roll={down[0]:+.2f} pitch={down[1]:+.2f} rad)')
            return
        self.p5_evidence_log({
            'event': 'fall_recovery_stood_up',
            'state': str(self.p5_fall_recover_retry_state),
            'response': snapshot,
        })
        self.p5_fall_recovery_fault(
            'robot is standing again; holding rather than resuming, because '
            'odometry cannot establish that it is still on the course')

    def p5_action_timeout_fault(self, log_name: str, elapsed: float, snapshot: dict):
        """Fail closed when an action/STOP acknowledgement never arrives."""
        target = self.p5_action_target
        phase = self.p5_action_phase
        # 先把控制器从卡死的动作里放出来，否则它连 STOP 都收不进去，机体
        # 会一直僵在原姿态：实测这会毁掉整批测试，实体上则毫无退路。
        self.p5_action_release_wedged_controller(snapshot)
        # Safety command must precede synchronous logging/evidence I/O.
        self.p5_send_stop_command()
        self.action_sent = True
        self.get_logger().error(
            f'[{log_name}] action timeout: phase={phase}, '
            f'target={target}, elapsed={elapsed:.2f}s, response={snapshot}'
        )
        self.p5_evidence_log({
            'event': 'action_timeout',
            'state': str(self.state),
            'phase': str(phase),
            'target': list(target),
            'elapsed_s': float(elapsed),
            'response': snapshot,
        })
        # 摔倒是可扶起的，卡死不是：只有回显 kEdamp 且里程计确认机体已躺倒时
        # 才改道扶起梯子，其余一律故障保持。
        if self.p5_fall_recovery_begin(log_name, snapshot):
            return
        self.p5_enter_state(self.P5_SENSOR_FAULT_HOLD)
        self.action_sent = True

    def p5_action_hold_feedback_fault(
        self, log_name: str, reason: str, snapshot: dict
    ):
        """Stop immediately when post-complete hold feedback becomes unsafe."""
        target = self.p5_action_target
        # The safety command must precede logging/evidence I/O.
        self.p5_send_stop_command()
        self.action_sent = True
        self.get_logger().error(
            f'[{log_name}] unsafe action hold feedback: reason={reason}, '
            f'target={target}, response={snapshot}'
        )
        self.p5_evidence_log({
            'event': 'action_hold_feedback_fault',
            'state': str(self.state),
            'target': list(target),
            'reason': str(reason),
            'response': snapshot,
        })
        self.p5_enter_state(self.P5_SENSOR_FAULT_HOLD)
        self.action_sent = True

    def p5_action_hold_feedback_fault_reason(
        self, snapshot: dict, now_monotonic_s: float
    ) -> str:
        """Return empty for healthy hold feedback, otherwise a fault reason."""
        rx_time = snapshot.get('rx_monotonic_s')
        if rx_time is None:
            return 'feedback_missing'
        feedback_age_s = max(0.0, now_monotonic_s - float(rx_time))
        if feedback_age_s > self.p5_action_feedback_max_age_s:
            return (
                f'feedback_stale:{feedback_age_s:.3f}s>'
                f'{self.p5_action_feedback_max_age_s:.3f}s'
            )

        mode, gait_id = self.p5_action_target
        if (
            int(snapshot.get('mode', -1)) != mode
            or int(snapshot.get('gait_id', -1)) != gait_id
            or int(snapshot.get('order_process_bar', 0)) < 95
        ):
            return 'target_complete_state_lost'

        error_fields = {
            'switch_status': int(snapshot.get('switch_status', 0)),
            'ori_error': int(snapshot.get('ori_error', 0)),
            'footpos_error': int(snapshot.get('footpos_error', 0)),
        }
        for field, value in error_fields.items():
            if value != 0:
                return f'{field}:{value}'
        motor_error = [int(value) for value in snapshot.get('motor_error', [])]
        if any(value != 0 for value in motor_error):
            return 'motor_error'
        return ''

    def run_action_state(
        self,
        mode: int,
        gait_id: int,
        next_state: str,
        log_name: str = "",
        stop_after_finish: bool = False,
    ):
        """Run an action and optional STOP with bounded, non-blocking polls."""
        name = log_name or self.state
        if self.p5_action_phase == "idle":
            self.p5_begin_action_poll(mode, gait_id, "action")
            self.get_logger().info(
                f"[{name}] non-blocking action poll: "
                f"mode={mode}, gait_id={gait_id}"
            )
            return

        if self.p5_action_completed_monotonic_s is None:
            timeout_s = (
                self.p5_stop_timeout_s
                if self.p5_action_phase == "stop"
                else self.p5_action_timeout_s
            )
            status, elapsed, snapshot = self.p5_poll_action(timeout_s)
            if status == "timeout":
                self.p5_action_timeout_fault(name, elapsed, snapshot)
                return
            if status != "complete":
                return
            if self.p5_action_phase == "action":
                self.p5_action_completed_monotonic_s = time.monotonic()
            if (
                self.p5_action_phase == "action"
                and self.p5_action_target != (12, 0)
            ):
                self.get_logger().info(
                    f"[{name}] action acknowledged, begin empirical "
                    f"post-complete hold: "
                    f"{self.p5_action_post_complete_hold_s:.2f}s"
                )

        if (
            self.p5_action_phase == "action"
            and self.p5_action_target != (12, 0)
        ):
            now = time.monotonic()
            settle_elapsed = max(
                0.0, now - self.p5_action_completed_monotonic_s)
            if settle_elapsed < self.p5_action_post_complete_hold_s:
                # MotionResultCmd returns a terminal service result rather
                # than the simulator's continuously refreshed LCM completion
                # heartbeat.  On the physical backend the successful service
                # result is the completion proof; keep the mechanical settle
                # delay but do not manufacture a fake feedback-freshness test.
                if not getattr(self.Ctrl, 'is_real', False):
                    snapshot = self.Ctrl.response_snapshot()
                    reason = self.p5_action_hold_feedback_fault_reason(snapshot, now)
                    if reason:
                        self.p5_action_hold_feedback_fault(name, reason, snapshot)
                return

        if self.p5_action_phase == "action" and stop_after_finish:
            if self.p5_action_target == (12, 0):
                self.get_logger().info(
                    f"[{name}] STOP action already acknowledged; "
                    f"skip redundant same-target STOP, go {next_state}"
                )
                self.p5_enter_state(next_state)
                return
            self.get_logger().info(
                f"[{name}] action acknowledged, begin non-blocking STOP "
                f"before {next_state}"
            )
            self.p5_begin_action_poll(12, 0, "stop")
            return

        self.get_logger().info(
            f"[{name}] {self.p5_action_phase} acknowledged, go {next_state}"
        )
        self.p5_enter_state(next_state)

    # ============================================================
    # 主状态机
    # ============================================================
    def p5_control_loop(self):
        try:
            # 里程积分与段窗口守护先于状态分发：任何状态里的推进都在
            # 同一份路线证据之上做判断。
            if self.p5_route_monitor():
                return

            if self.state == self.P5_RECOVERY_STAND:
                self.run_action_state(
                    mode=12,
                    gait_id=0,
                    next_state=self.P5_SET_BODY_NORMAL,
                    log_name='P5_RECOVERY_STAND'
                )

            elif self.state == self.P5_SET_BODY_NORMAL:
                if not self.action_sent:
                    self.set_body_roll_height(
                        roll=self.p5_body_normal_roll,
                        height=self.p5_body_normal_height
                    )
                    self.action_sent = True

                if self.p5_state_elapsed_s() >= self.p5_body_normal_wait_s:
                    self.p5_enter_state(self.P5_START_ALIGN)

            elif self.state == self.P5_START_ALIGN:
                self.p5_run_start_align()

            elif self.state == self.P5_STEP_UP:
                self.run_timed_velocity_state(
                    duration_s=self.p5_step_up_duration_s,
                    vx=self.p5_step_up_vx,
                    vy=self.p5_step_up_vy,
                    wz=self.p5_step_up_wz,
                    step_height=self.p5_step_up_step_height,
                    next_state=self.P5_UP_SLOPE,
                    body_height=self.p5_body_normal_height,
                    log_name='P5_STEP_UP'
                )

            elif self.state == self.P5_UP_SLOPE:
                self.run_right_side_yellow_lost_velocity_state(
                    vx=self.p5_up_slope_vx,
                    vy=self.p5_up_slope_vy,
                    wz=self.p5_up_slope_wz,
                    step_height=self.p5_up_slope_step_height,
                    next_state=self.P5_AFTER_UP_SLOPE_FORWARD,
                    roll=self.p5_up_slope_roll,
                    pitch=self.p5_up_slope_pitch,
                    body_height=self.p5_body_normal_height,
                    log_name='P5_UP_SLOPE',
                    timeout_s=self.p5_up_slope_timeout_s,
                )

            elif self.state == self.P5_AFTER_UP_SLOPE_FORWARD:
                self.run_timed_velocity_state(
                    duration_s=self.p5_after_up_slope_forward_duration_s,
                    vx=self.p5_after_up_slope_forward_vx,
                    vy=self.p5_after_up_slope_forward_vy,
                    wz=self.p5_after_up_slope_forward_wz,
                    step_height=self.p5_after_up_slope_forward_step_height,
                    next_state=self.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL,
                    body_height=self.p5_body_normal_height,
                    log_name='P5_AFTER_UP_SLOPE_FORWARD'
                )

            elif self.state == self.P5_AFTER_UP_SLOPE_VELOCITY_CONTROL:
                if self.p5_after_up_slope_turn_method == 'right_jump':
                    self.run_action_state(
                        mode=self.p5_after_up_slope_turn_jump_mode,
                        gait_id=self.p5_after_up_slope_turn_jump_gait,
                        next_state=self.P5_SET_RIGHT_SLOPE_BODY,
                        log_name='P5_AFTER_UP_SLOPE_RIGHT_JUMP',
                        stop_after_finish=True,
                    )
                else:
                    self.run_timed_velocity_state(
                        duration_s=self.p5_after_up_slope_control_duration_s,
                        vx=self.p5_after_up_slope_control_vx,
                        vy=self.p5_after_up_slope_control_vy,
                        wz=self.p5_after_up_slope_control_wz,
                        step_height=self.p5_after_up_slope_control_step_height,
                        next_state=self.P5_SET_RIGHT_SLOPE_BODY,
                        body_height=self.p5_body_normal_height,
                        log_name='P5_AFTER_UP_SLOPE_VELOCITY_CONTROL',
                        zero_velocity_when_done=True,
                    )

            elif self.state == self.P5_SET_RIGHT_SLOPE_BODY:
                if not self.action_sent:
                    self.set_body_roll_height(
                        roll=self.p5_right_slope_roll,
                        height=self.p5_right_slope_height
                    )
                    self.action_sent = True

                if self.p5_state_elapsed_s() >= self.p5_right_slope_body_wait_s:
                    self.p5_enter_state(self.P5_RIGHT_SLOPE_1)

            elif self.state == self.P5_RIGHT_SLOPE_1:
                self.run_center_yellow_absence_velocity_state(
                    vx=self.p5_right_slope_1_vx,
                    vy=self.p5_right_slope_1_vy,
                    wz=self.p5_right_slope_1_wz,
                    step_height=self.p5_right_slope_1_step_height,
                    next_state=self.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_1',
                    timeout_s=self.p5_right_slope_1_timeout_s,
                )

            elif self.state == self.P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST:
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_slope_1_after_center_lost_duration_s,
                    vx=self.p5_right_slope_1_after_center_lost_vx,
                    vy=self.p5_right_slope_1_after_center_lost_vy,
                    wz=self.p5_right_slope_1_after_center_lost_wz,
                    step_height=self.p5_right_slope_1_after_center_lost_step_height,
                    next_state=self.P5_TURN_1,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST'
                )  # no stop: extra forward -> turn directly

            elif self.state == self.P5_TURN_1:
                if self.p5_right_slope_turn_method == 'right_jump':
                    self.run_action_state(
                        mode=self.p5_right_slope_turn_1_jump_mode,
                        gait_id=self.p5_right_slope_turn_1_jump_gait,
                        next_state=self.P5_RIGHT_SLOPE_2,
                        log_name='P5_TURN_1_RIGHT_JUMP',
                        stop_after_finish=self.p5_right_slope_turn_jump_stop_after_finish
                    )
                else:
                    self.run_timed_velocity_state(
                        duration_s=self.p5_turn_1_duration_s,
                        vx=self.p5_turn_1_vx,
                        vy=self.p5_turn_1_vy,
                        wz=self.p5_turn_1_wz,
                        step_height=self.p5_turn_1_step_height,
                        next_state=self.P5_RIGHT_SLOPE_2,
                        roll=self.p5_right_slope_roll,
                        body_height=self.p5_right_slope_height,
                        log_name='P5_TURN_1'
                    )  # no stop: turn -> right slope 2 directly

            elif self.state == self.P5_RECOVERY_AFTER_TURN_1:
                # 保留这个状态名是为了兼容旧流程；当前速度控制转向不会进入这里。
                self.p5_enter_state(self.P5_RIGHT_SLOPE_2)

            elif self.state == self.P5_RIGHT_SLOPE_2:
                slope2_vy = self.p5_right_slope_2_vy
                if (
                    self.p5_safety_elapsed_s()
                    < self.p5_right_slope_2_entry_recenter_duration_s
                ):
                    slope2_vy = self.p5_right_slope_2_entry_recenter_vy
                self.run_center_yellow_absence_velocity_state(
                    vx=self.p5_right_slope_2_vx,
                    vy=slope2_vy,
                    wz=self.p5_right_slope_2_wz,
                    step_height=self.p5_right_slope_2_step_height,
                    next_state=self.P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_2',
                    timeout_s=self.p5_right_slope_2_timeout_s,
                    right_edge_adjust_enabled=(
                        self.p5_right_slope_2_right_edge_adjust_enabled
                    ),
                )

            elif self.state == self.P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST:
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_slope_2_after_center_lost_duration_s,
                    vx=self.p5_right_slope_2_after_center_lost_vx,
                    vy=self.p5_right_slope_2_after_center_lost_vy,
                    wz=self.p5_right_slope_2_after_center_lost_wz,
                    step_height=self.p5_right_slope_2_after_center_lost_step_height,
                    next_state=self.P5_TURN_2,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST'
                )  # no stop: extra forward -> turn directly

            elif self.state == self.P5_TURN_2:
                if self.p5_right_slope_turn_method == 'right_jump':
                    self.run_action_state(
                        mode=self.p5_right_slope_turn_2_jump_mode,
                        gait_id=self.p5_right_slope_turn_2_jump_gait,
                        next_state=self.P5_RIGHT_SLOPE_3,
                        log_name='P5_TURN_2_RIGHT_JUMP',
                        stop_after_finish=self.p5_right_slope_turn_jump_stop_after_finish
                    )
                else:
                    self.run_timed_velocity_state(
                        duration_s=self.p5_turn_2_duration_s,
                        vx=self.p5_turn_2_vx,
                        vy=self.p5_turn_2_vy,
                        wz=self.p5_turn_2_wz,
                        step_height=self.p5_turn_2_step_height,
                        next_state=self.P5_RIGHT_SLOPE_3,
                        roll=self.p5_right_slope_roll,
                        body_height=self.p5_right_slope_height,
                        log_name='P5_TURN_2'
                    )  # no stop: turn -> right slope 3 directly

            elif self.state == self.P5_RECOVERY_AFTER_TURN_2:
                # 保留这个状态名是为了兼容旧流程；当前速度控制转向不会进入这里。
                self.p5_enter_state(self.P5_RIGHT_SLOPE_3)

            elif self.state == self.P5_RIGHT_SLOPE_3:
                self.run_center_yellow_absence_velocity_state(
                    vx=self.p5_right_slope_3_vx,
                    vy=self.p5_right_slope_3_vy,
                    wz=self.p5_right_slope_3_wz,
                    step_height=self.p5_right_slope_3_step_height,
                    next_state=self.P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_3',
                    timeout_s=self.p5_right_slope_3_timeout_s,
                )

            elif self.state == self.P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST:
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_slope_3_after_center_lost_duration_s,
                    vx=self.p5_right_slope_3_after_center_lost_vx,
                    vy=self.p5_right_slope_3_after_center_lost_vy,
                    wz=self.p5_right_slope_3_after_center_lost_wz,
                    step_height=self.p5_right_slope_3_after_center_lost_step_height,
                    next_state=self.P5_RESET_BODY,
                    roll=self.p5_right_slope_roll,
                    body_height=self.p5_right_slope_height,
                    log_name='P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST',
                    zero_velocity_when_done=True,
                )

            elif self.state == self.P5_RESET_BODY:
                if not self.action_sent:
                    self.p5_reset_body_roll_from = float(self.p5_last_cmd_roll)
                    self.set_body_roll_height(
                        roll=(self.p5_reset_body_roll_from
                              if self.p5_reset_body_ramp_s > 0.0
                              else self.p5_reset_roll),
                        height=self.p5_reset_height
                    )
                    self.action_sent = True

                if self.p5_reset_body_ramp_s > 0.0:
                    self.p5_reset_body_ramp_tick()

                if self.p5_state_elapsed_s() >= max(
                        self.p5_reset_body_wait_s, self.p5_reset_body_ramp_s):
                    self.get_logger().info(
                        '[P5_RESET_BODY] reset body done, go P5_RIGHT_JUMP_AFTER_RESET_BODY'
                    )
                    self.p5_enter_state(self.P5_RIGHT_JUMP_AFTER_RESET_BODY)

            elif self.state == self.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP:
                # reset body 后，先执行第一段右移固定时间，再进入第二段右移。
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_shift_before_right_jump_duration_s,
                    vx=self.p5_right_shift_before_right_jump_vx,
                    vy=self.p5_right_shift_before_right_jump_vy,
                    wz=self.p5_right_shift_before_right_jump_wz,
                    step_height=self.p5_right_shift_before_right_jump_step_height,
                    next_state=self.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP'
                )

            elif self.state == self.P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2:
                # 第一段右移后，继续右移一段固定时间，然后再执行右跳动作。
                self.run_timed_velocity_state(
                    duration_s=self.p5_right_shift_before_right_jump_2_duration_s,
                    vx=self.p5_right_shift_before_right_jump_2_vx,
                    vy=self.p5_right_shift_before_right_jump_2_vy,
                    wz=self.p5_right_shift_before_right_jump_2_wz,
                    step_height=self.p5_right_shift_before_right_jump_2_step_height,
                    next_state=self.P5_RIGHT_JUMP_AFTER_RESET_BODY,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_RIGHT_SHIFT_BEFORE_RIGHT_JUMP_2'
                )

            elif self.state == self.P5_RIGHT_JUMP_AFTER_RESET_BODY:
                # 到达第三段中心线终点后执行第四个转角右跳；不再在斜坡上侧移。
                self.run_action_state(
                    mode=self.p5_right_jump_after_reset_body_mode,
                    gait_id=self.p5_right_jump_after_reset_body_gait,
                    next_state=self.P5_ALIGN_AFTER_RIGHT_JUMP,
                    log_name='P5_RIGHT_JUMP_AFTER_RESET_BODY',
                    stop_after_finish=True
                )

            elif self.state == self.P5_ALIGN_AFTER_RIGHT_JUMP:
                self.run_timed_velocity_state(
                    duration_s=self.p5_align_after_right_jump_duration_s,
                    vx=self.p5_align_after_right_jump_vx,
                    vy=self.p5_align_after_right_jump_vy,
                    wz=self.p5_align_after_right_jump_wz,
                    step_height=self.p5_align_after_right_jump_step_height,
                    next_state=self.P5_FORWARD_AFTER_RESET_BODY,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_ALIGN_AFTER_RIGHT_JUMP',
                    zero_velocity_when_done=True,
                )

            elif self.state == self.P5_FORWARD_AFTER_RESET_BODY:
                # 第三转角右跳后第一段：先固定时间前进，同时根据内侧边缘做 vy/wz 矫正；
                # 如果时间到了还没对齐，则 vx=0 原地横移/转向继续矫正。
                # 矫正完成后不 stop，直接进入下一段“不矫正固定前进”。
                self.run_timed_velocity_then_stop_state(
                    duration_s=self.p5_forward_after_reset_body_duration_s,
                    vx=self.p5_forward_after_reset_body_vx,
                    vy=self.p5_forward_after_reset_body_vy,
                    wz=self.p5_forward_after_reset_body_wz,
                    step_height=self.p5_forward_after_reset_body_step_height,
                    next_state=self.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_FORWARD_AFTER_RESET_BODY',
                    # 里程主导模式下没有黄线内侧边缘可用，这一段退化为纯定时前进，
                    # 也不再等待 RGB 帧（否则会因不再消费的图像流而卡住）。
                    use_inner_edge_align=self.p5_yellow_lateral_correction_enabled,
                    hold_after_duration_until_aligned=(
                        self.p5_forward_after_reset_body_hold_align_enabled
                        and self.p5_yellow_lateral_correction_enabled
                    ),
                    stop_when_done=False
                )

            elif self.state == self.P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY:
                # 矫正完成后第二段：固定时间前进，不再叠加视觉矫正；
                # 结束后 stop，再进入离坡右跳/跳远流程。
                self.run_timed_velocity_then_stop_state(
                    duration_s=self.p5_forward_no_align_after_reset_body_duration_s,
                    vx=self.p5_forward_no_align_after_reset_body_vx,
                    vy=self.p5_forward_no_align_after_reset_body_vy,
                    wz=self.p5_forward_no_align_after_reset_body_wz,
                    step_height=self.p5_forward_no_align_after_reset_body_step_height,
                    next_state=self.P5_JUMP_EXIT_SLOPE,
                    roll=self.p5_reset_roll,
                    body_height=self.p5_reset_height,
                    log_name='P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY',
                    use_inner_edge_align=False,
                    hold_after_duration_until_aligned=False,
                    stop_when_done=True
                )

            elif self.state == self.P5_JUMP_EXIT_SLOPE:
                self.run_action_state(
                    mode=self.p5_jump_exit_slope_mode,
                    gait_id=self.p5_jump_exit_slope_gait,
                    next_state=self.P5_RECOVERY_AFTER_JUMP_2,
                    log_name='P5_JUMP_EXIT_SLOPE'
                )

            elif self.state == self.P5_RECOVERY_AFTER_JUMP_2:
                self.run_action_state(
                    mode=12,
                    gait_id=0,
                    next_state=self.P5_FINAL_LONG_JUMP,
                    log_name='P5_RECOVERY_AFTER_JUMP_2',
                    stop_after_finish=True
                )

            elif self.state == self.P5_FINAL_LONG_JUMP:
                # 最后跳远动作完成后，先发送一次 STOP，清掉跳远后的残余动作，
                # 再进入 P5_DONE / 后续第六赛段衔接。
                self.run_action_state(
                    mode=self.p5_final_long_jump_mode,
                    gait_id=self.p5_final_long_jump_gait,
                    next_state=self.P5_DONE,
                    log_name='P5_FINAL_LONG_JUMP',
                    stop_after_finish=True
                )

            elif self.state == self.P5_DONE:
                if not self.action_sent:
                    self.p5_send_stop_command()
                    self.action_sent = True

                self.get_logger().info(
                    '[P5_DONE] fifth stage done, keep stop',
                    throttle_duration_sec=1.0
                )

            elif self.state == self.P5_ROUTE_REALIGN:
                self.p5_run_route_realign()

            elif self.state == self.P5_FALL_RECOVER:
                self.p5_run_fall_recover()

            elif self.state == self.P5_STALL_RECOVER:
                self.p5_run_stall_recover()

            elif self.state == self.P5_SENSOR_FAULT_HOLD:
                # 传感器故障 / 状态超时：站定保持，等待人工恢复。
                # 不上报 stage_complete；恢复需按分段重启流程重新激活。
                if not self.action_sent:
                    self.p5_send_stop_command()
                    self.action_sent = True

                self.get_logger().error(
                    '[P5_SENSOR_FAULT_HOLD] holding stop after sensor fault / '
                    'stage timeout; manual recovery required '
                    '(re-activate stage with an explicit p5_initial_state)',
                    throttle_duration_sec=2.0
                )

            else:
                self.get_logger().error(
                    f'[P5] unknown state={self.state}, send stop'
                )
                self.p5_send_stop_command()

        except Exception as e:
            self.get_logger().error(
                f'[P5] control_loop exception: {repr(e)}'
            )
            self.p5_send_stop_command()
            raise

    def p5_destroy_node(self):
        try:
            self.p5_send_stop_command()
            self.Ctrl.quit()
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if self.Odom is not None:
                self.Odom.quit()
                self.Odom = None
        except Exception:
            pass

        try:
            if self._p5_evidence_fp is not None:
                self._p5_evidence_fp.close()
                self._p5_evidence_fp = None
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Stage5Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        try:
            if node.Ctrl is not None:
                node.p5_send_stop_command()
        except Exception:
            pass
        try:
            if node.Ctrl is not None:
                node.Ctrl.quit()
        except Exception:
            pass
        try:
            if node.Odom is not None:
                node.Odom.quit()
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
