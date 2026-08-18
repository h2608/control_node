#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二赛段节点：激活后先恢复站立，保留 RGB+Depth 巡航居中，使用左右鱼眼完成橙球侧撞，然后找到出口。

原 control_node_123456.py 的第二赛段状态机（内部状态名保持不变：
STAGE1_*/STAGE2_*/STAGE3_* 是第二赛段内部的三个巡航子阶段，BALL_* 是撞球子链）。
STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT 结束后向任务控制节点上报完成
（原来是直接切入第三赛段 P3_S_CURVE_CRUISE）。
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from control_node.stage_common import StageNodeBase, clamp, find_contours
from control_node.stage_entry import (
    EntryPoint,
    StageEntryTable,
    is_default_request,
)


# 撞球子链：这些状态打完球后回到 ball_return_state，直接从它们启动时必须
# 用 p2_ball_return_state 指定回哪条巡航，否则会回到自己形成死循环。
P2_BALL_SUBCHAIN_STATES = (
    'BALL_LATERAL_ALIGN',
    'BALL_HIT_CONFIRM_FORWARD',
    'BALL_POST_HIT_SIDE_SHIFT',
)

P2_CRUISE_STATES = (
    'STAGE1_CRUISE_BALL_AND_YELLOW',
    'STAGE2_CRUISE_YELLOW_ONLY',
    'STAGE3_GO_FINAL',
)


def p2_entry_table():
    """第二赛段调试入口表。

    内部状态名里的 STAGE1_/STAGE2_/STAGE3_ 是第二赛段自己的三条巡航赛道，
    不是比赛第一/二/三赛段；入口名用 track1/track2/track3 消歧。
    """
    states = (
        # 赛道 1
        'STAGE1_CRUISE_BALL_AND_YELLOW',
        'STAGE1_FORWARD_BEFORE_ROTATE',
        'STAGE1_ROTATE_LEFT_90',
        'STAGE1_MOVE_RIGHT_FIXED_DISTANCE',
        # 赛道 2
        'STAGE2_CRUISE_YELLOW_ONLY',
        'STAGE2_ROTATE_LEFT_90',
        # 赛道 2 -> 3：第二次90°前先定时左移
        'STAGE2_LEFT_SHIFT_BEFORE_SECOND_90',
        # 赛道 3：第二次90°后先不撞球，RGB居中找黄线
        'STAGE3_PRE180_CENTER_YELLOW',
        'STAGE3_ROTATE_BACK_180',
        'STAGE3_GO_FINAL',
        'STAGE3_FORWARD_AFTER_GO_FINAL',
        'STAGE3_ROTATE_LEFT_30',
        'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT',
        # 撞球子链
    ) + P2_BALL_SUBCHAIN_STATES + ('DONE',)
    ball_note = 'ball_return_state 必须用 p2_ball_return_state 指定'
    return StageEntryTable(2, 'STAGE1_CRUISE_BALL_AND_YELLOW', states, (
        EntryPoint('start', 'STAGE1_CRUISE_BALL_AND_YELLOW',
                   '赛道 1 巡航（橙球 + 黄线），完整第二赛段'),
        EntryPoint('track1', 'STAGE1_CRUISE_BALL_AND_YELLOW', '赛道 1 巡航'),
        EntryPoint('track1_exit', 'STAGE1_FORWARD_BEFORE_ROTATE',
                   '赛道 1 结束，转向前的定时前进'),
        EntryPoint('track1_turn', 'STAGE1_ROTATE_LEFT_90', '赛道 1 -> 2 左转 90°'),
        EntryPoint('track1_shift', 'STAGE1_MOVE_RIGHT_FIXED_DISTANCE',
                   '转向后的定距右横移'),
        EntryPoint('track2', 'STAGE2_CRUISE_YELLOW_ONLY', '赛道 2 巡航（只看黄线）'),
        EntryPoint('track2_shift', 'STAGE2_LEFT_SHIFT_BEFORE_SECOND_90',
                   '第二次90°之前：先定时左移'),
        EntryPoint('track2_turn', 'STAGE2_ROTATE_LEFT_90',
                   '左移完成后：原地踏步恢复姿态，再左转90°'),
        EntryPoint('track3', 'STAGE3_PRE180_CENTER_YELLOW',
                   '第二次90°后：禁用鱼眼撞球，RGB居中前进并寻找黄线'),
        EntryPoint('turn_back', 'STAGE3_ROTATE_BACK_180',
                   '第三条黄线到达后，原地踏步恢复姿态并掉头180°'),
        EntryPoint('final', 'STAGE3_GO_FINAL',
                   '180°后：启用鱼眼橙球，同时前进寻找最终黄线'),
        EntryPoint('final_forward', 'STAGE3_FORWARD_AFTER_GO_FINAL', '出口前定时前进'),
        EntryPoint('final_turn', 'STAGE3_ROTATE_LEFT_30', '出口前左转 30°'),
        EntryPoint('final_align', 'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT',
                   '收尾转向，之后上报完成'),
        EntryPoint('ball_align', 'BALL_LATERAL_ALIGN', '撞球子链：横向对准球',
                   requires=(ball_note, '橙球必须在鱼眼视野内')),
        EntryPoint('ball_hit', 'BALL_HIT_CONFIRM_FORWARD', '撞球子链：撞击确认前进',
                   requires=(ball_note,)),
        EntryPoint('ball_shift', 'BALL_POST_HIT_SIDE_SHIFT', '撞球子链：撞后侧移',
                   requires=(ball_note,)),
    ))


class Stage2Node(StageNodeBase):

    STAGE_ID = 2

    def __init__(self):
        super().__init__('stage2_node', self.STAGE_ID)

        # =========================
        # 第二赛段双频控制
        # =========================
        # 普通状态只按 5 Hz 执行 stage_control_loop，降低 CPU 压力。
        # 固定时间原地转向仍需要更细的时间分辨率，因此底层定时器保持 30 Hz；
        # 进入定时转向状态后不再做 RGB / Depth / 双鱼眼图像转换与检测。
        self.declare_parameter('stage2_control_hz', 5.0)
        self.declare_parameter('stage2_timed_turn_control_hz', 30.0)
        # 需要 RGB 的巡航状态如果连续一段时间没有成功处理新帧，就保持
        # 连续步态但速度清零，避免相机卡住时继续盲走。
        self.declare_parameter('stage2_rgb_stale_timeout_s', 0.60)
        # 5 Hz 普通状态速度已按低频控制下调；连续帧确认阈值也同步缩短。
        self.stage2_control_hz = max(
            0.1, float(self.get_parameter('stage2_control_hz').value))
        self.stage2_timed_turn_control_hz = max(
            self.stage2_control_hz,
            float(self.get_parameter('stage2_timed_turn_control_hz').value))
        self.stage2_control_period_s = 1.0 / self.stage2_control_hz
        self.stage2_rgb_stale_timeout_s = max(
            self.stage2_control_period_s,
            float(self.get_parameter('stage2_rgb_stale_timeout_s').value)
        )
        self._stage2_next_low_rate_tick_monotonic: Optional[float] = None

        # StageNodeBase 默认创建 control_timer。第二赛段自己固定以 30 Hz 唤醒，
        # 再在 stage_control_loop 内把非转向状态节流到 5 Hz。
        # 这样不会修改公共 StageNodeBase，也不会影响其他赛段。
        if abs(self.control_hz - self.stage2_timed_turn_control_hz) > 1e-6:
            self.control_timer.cancel()
            self.destroy_timer(self.control_timer)
            self.control_timer = self.create_timer(
                1.0 / self.stage2_timed_turn_control_hz,
                self._control_timer_cb
            )
        self.control_hz = self.stage2_timed_turn_control_hz

        # 调试入口：可写入口名（track2、final…）或直接写状态名。
        # 统一的 entry_point 参数（launch 里的 stage2_entry）优先于这一个。
        self.declare_parameter('second_stage_initial_state', 'default')
        # 从撞球子链入口启动时，指定打完球回到哪条巡航赛道。
        self.declare_parameter('p2_ball_return_state', 'default')

        # =========================
        # 状态机状态说明
        # =========================
        # STAGE1_CRUISE_BALL_AND_YELLOW:
        #   第一阶段巡航；沿两排球中间前进，同时看橙球和黄线。
        #   发现满足条件的橙球时转入 BALL_LATERAL_ALIGN；
        #   黄线达到第一阶段阈值时转入 STAGE1_FORWARD_BEFORE_ROTATE。
        #
        # STAGE1_FORWARD_BEFORE_ROTATE:
        #   第一次左转前的固定前进状态。
        #   保持 stage2_body_pitch 前倾，以固定 vx 前进固定仿真时间，
        #   到时后进入 STAGE1_ROTATE_LEFT_90。
        #
        # STAGE1_ROTATE_LEFT_90:
        #   第一阶段结束后的左转状态。
        #   使用固定角速度 + 固定仿真时间代替原地左跳，完成后直接进入 STAGE2_CRUISE_YELLOW_ONLY。
        #
        # STAGE2_CRUISE_YELLOW_ONLY:
        #   第二阶段巡航；主要看黄线，不进入撞球子状态。
        #   RGB-only 检测危险小球并执行右移避障。
        #   黄线达到第二阶段阈值后，不立即转向；
        #   先进入 STAGE2_LEFT_SHIFT_BEFORE_SECOND_90 定时左移。
        #
        # STAGE2_LEFT_SHIFT_BEFORE_SECOND_90:
        #   第二次 90° 左转之前的定时左移。
        #   该阶段关闭视觉并使用 30 Hz 控制；左移完成后进入
        #   STAGE2_ROTATE_LEFT_90。
        #
        # STAGE2_ROTATE_LEFT_90:
        #   第二阶段结束后的第二次 90° 左转。
        #   状态内部先原地踏步 timed_turn_pre_stand_duration_sec，
        #   再按固定角速度 + 固定时间完成 90°。
        #   完成后进入 STAGE3_PRE180_CENTER_YELLOW。
        #
        # STAGE3_PRE180_CENTER_YELLOW:
        #   第二次 90° 后的第三条直道前半段。
        #   此阶段不启用鱼眼橙球撞击；RGB 仍检测左右参考球用于中线居中，
        #   同时检测黄线。这里直接复用旧 STAGE3_CRUISE_BALL_ONLY
        #   原本使用的黄线参数：
        #   yellow_stop_line_y_ratio_stage3 / yellow_slowdown_ratio_stage3。
        #   达到黄线后直接进入 STAGE3_ROTATE_BACK_180。
        #
        # STAGE3_ROTATE_BACK_180:
        #   先原地踏步 timed_turn_pre_stand_duration_sec 恢复水平姿态，
        #   再按固定时间掉头 180°。
        #   完成后进入 STAGE3_GO_FINAL。
        #
        # STAGE3_GO_FINAL:
        #   180° 后开始启用鱼眼橙球撞击，同时继续向前寻找最终黄线。
        #   撞球完成后回到本状态继续前进；最终黄线触发阈值保持 yellow_ratio_final。
        #
        # STAGE3_FORWARD_AFTER_GO_FINAL:
        #   最终出口线后的固定前进状态。
        #   按独立参数保持直行固定时间，到时后进入 STAGE3_ROTATE_LEFT_30。
        #
        # STAGE3_ROTATE_LEFT_30:
        #   最终出口前的收尾状态。
        #   不再使用 TF 判断结束，而是按仿真时间发送移动速度；
        #   到时后直接进入 STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT。
        #
        # STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT:
        #   移动完成后，不再使用 TF yaw 闭环，而是按仿真时间发送转向角速度；
        #   到时后进入 DONE。
        #
        # BALL_LATERAL_ALIGN:
        #   撞球子状态 1：横向对齐球。
        #   采用“小 vx + 主 vy”边前进边横移，把目标橙球送到机器狗正前方。
        #
        # BALL_HIT_CONFIRM_FORWARD:
        #   撞球子状态 2：确认后直接前冲撞击。
        #   进入时记录目标球深度，用“球深 + 额外前冲距离”生成撞击距离，
        #   并用仿真时间判断撞击是否完成。
        #
        # BALL_POST_HIT_SIDE_SHIFT:
        #   撞球子状态 3：撞完后只做左右横移。
        #   撞左球则向右移，撞右球则向左移。
        #   使用固定速度，持续指定仿真时间后直接回到保存的巡航状态 ball_return_state。
        #
        # DONE:
        #   全流程结束状态。
        #   持续发送停止命令，任务完成。

        # =========================
        # 橙球检测
        # =========================
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 25)
        self.declare_parameter('orange_s_min', 100)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 80)
        self.declare_parameter('orange_v_max', 255)
        self.declare_parameter('orange_min_contour_area', 400.0)

        # =========================
        # 蓝球检测（只用于辅助中线）
        # =========================
        self.declare_parameter('blue_h_min', 90)
        self.declare_parameter('blue_h_max', 130)
        self.declare_parameter('blue_s_min', 80)
        self.declare_parameter('blue_s_max', 255)
        self.declare_parameter('blue_v_min', 50)
        self.declare_parameter('blue_v_max', 255)
        self.declare_parameter('blue_min_contour_area', 400.0)

        self.declare_parameter('prefer_nearest_ball', True)
        self.declare_parameter('min_ball_radius_to_trigger', 40.0)

        # =========================
        # 深度搜索
        # =========================
        self.declare_parameter('depth_search_half', 12)
        self.declare_parameter('valid_min_depth_m', 0.05)
        self.declare_parameter('valid_max_depth_m', 10.0)

        # =========================
        # 左右鱼眼侧撞（替换原前方相机撞球子链）
        # =========================
        default_fisheye_left_topic = (
            '/mi_desktop_48_b0_2d_7b_00_e2/image_left'
            if self.platform == 'real' else '/image_left'
        )
        default_fisheye_right_topic = (
            '/mi_desktop_48_b0_2d_7b_00_e2/image_right'
            if self.platform == 'real' else '/image_right'
        )
        self.declare_parameter('fisheye_left_topic', default_fisheye_left_topic)
        self.declare_parameter('fisheye_right_topic', default_fisheye_right_topic)

        self.declare_parameter('fisheye_orange_min_contour_area', 1000.0)
        self.declare_parameter('fisheye_morph_kernel_size', 5)
        self.declare_parameter('fisheye_min_circularity', 0.60)
        self.declare_parameter('fisheye_min_aspect_ratio', 0.65)
        self.declare_parameter('fisheye_max_aspect_ratio', 1.45)
        self.declare_parameter('fisheye_min_circle_fill_ratio', 0.50)
        self.declare_parameter('fisheye_min_bbox_fill_ratio', 0.45)

        # 鱼眼橙球允许进入撞球子链的横向范围，使用显式 min/max 更直观。
        self.declare_parameter('fisheye_entry_x_min_ratio', 0.20)
        self.declare_parameter('fisheye_entry_x_max_ratio', 0.70)
        self.declare_parameter('fisheye_entry_center_y_ratio', 0.50)
        self.declare_parameter('fisheye_entry_y_tolerance', 0.25)
        self.declare_parameter('fisheye_entry_confirm_frames', 1)

        self.declare_parameter('fisheye_approach_vy', 0.12)
        self.declare_parameter('fisheye_approach_lost_frames', 2)
        self.declare_parameter('fisheye_approach_target_x_ratio', 0.50)
        self.declare_parameter('fisheye_approach_x_deadband_ratio', 0.05)
        self.declare_parameter('fisheye_approach_vx_k', 0.60)
        self.declare_parameter('fisheye_approach_vx_max', 0.10)
        self.declare_parameter('left_fisheye_x_to_vx_sign', 1.0)
        self.declare_parameter('right_fisheye_x_to_vx_sign', -1.0)

        self.declare_parameter('fisheye_hit_radius', 45.0)
        self.declare_parameter('fisheye_hit_radius_confirm_frames', 1)
        self.declare_parameter('fisheye_hit_vy', 0.20)
        self.declare_parameter('fisheye_hit_duration_sec', 1.0)

        self.declare_parameter('fisheye_recover_forward_vx', 0.0)
        self.declare_parameter('fisheye_recover_vy', 0.15)
        self.declare_parameter('fisheye_recover_duration_sec', 2.0)

        # =========================
        # 黄线检测
        # =========================
        self.declare_parameter('yellow_roi_top_ratio', 0.65)
        self.declare_parameter('yellow_roi_left_ratio', 0.4)
        self.declare_parameter('yellow_roi_right_ratio', 0.6)

        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('yellow_s_min', 80)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 80)
        self.declare_parameter('yellow_v_max', 255)
        self.declare_parameter('yellow_min_contour_area', 100.0)

        self.declare_parameter('yellow_min_width_height_ratio', 2.5)
        self.declare_parameter('yellow_max_tilt_deg', 30.0)
        self.declare_parameter('yellow_center_tolerance_ratio', 0.15)
        self.declare_parameter('yellow_min_width_ratio', 0.45)

        self.declare_parameter('yellow_stop_line_y_ratio_stage1', 1.0)
        self.declare_parameter('yellow_stop_line_y_ratio_stage2', 0.75)
        self.declare_parameter('yellow_stop_line_y_ratio_stage3', 1.0)
        self.declare_parameter('yellow_stop_confirm_count', 1)

        self.declare_parameter('yellow_ratio_final', 1.0)

        # =========================
        # 巡航中黄线角度矫正
        # =========================
        self.declare_parameter('yellow_angle_align_enabled', True)
        # 黄线角度矫正使用固定角速度：只根据 angle_deg 正负决定转向方向。
        self.declare_parameter('yellow_angle_align_fixed_wz', 0.10)
        self.declare_parameter('yellow_angle_align_deadband_deg', 0.5)

        # =========================
        # 巡航 / 中线
        # =========================
        self.declare_parameter('stage1_cruise_forward_speed', 0.20)
        self.declare_parameter('stage2_cruise_forward_speed', 0.30)

        # 第一次 90° 完成后的 STAGE2_CRUISE_YELLOW_ONLY：
        # 前 3 秒保持原巡航；3 秒后在无危险小球时加速并小幅向左。
        self.declare_parameter('stage2_after_turn_left_bias_delay_sec', 3.0)
        self.declare_parameter('stage2_cruise_forward_speed_after_delay', 0.30)
        self.declare_parameter('stage2_after_turn_left_bias_vy', 0.02)

        self.declare_parameter('stage3_cruise_ball_only_speed', 0.20)

        # =========================
        # 第二赛段左侧近球固定右避让
        # =========================
        # STAGE2_CRUISE_YELLOW_ONLY 本来只看黄线向前走，容易蹭到左侧靠近中线的蓝球/橙球。
        # 这里不进入撞球子状态，只在危险球连续出现时给固定右移 vy。
        self.declare_parameter('stage2_left_ball_avoid_enabled', True)
        self.declare_parameter('stage2_left_ball_avoid_center_px', 130)
        self.declare_parameter('stage2_left_ball_avoid_vy', 0.08)
        self.declare_parameter('stage2_left_ball_avoid_confirm_frames', 1)
        # RGB-only 避障判定，不使用 Depth。
        self.declare_parameter('stage2_left_ball_avoid_min_area', 5000.0)
        self.declare_parameter('stage2_left_ball_avoid_min_radius', 50.0)

        self.declare_parameter('stage3_go_final_speed', 0.25)

        # STAGE3_GO_FINAL 到达最终黄线后，再固定向前走一段。
        self.declare_parameter('stage3_forward_after_go_final_vx', 0.15)
        self.declare_parameter('stage3_forward_after_go_final_duration_s', 1.0)

        # 黄线预触发减速区：先减速，再真正触发切状态
        self.declare_parameter('yellow_slowdown_ratio_stage1', 0.90)
        self.declare_parameter('yellow_slowdown_ratio_stage2', 0.7)
        self.declare_parameter('yellow_slowdown_ratio_stage3', 0.90)
        self.declare_parameter('yellow_slowdown_ratio_final', 0.90)

        self.declare_parameter('stage1_yellow_slow_speed', 0.10)
        self.declare_parameter('stage2_yellow_slow_speed', 0.10)
        self.declare_parameter('stage3_yellow_slow_speed', 0.10)
        self.declare_parameter('stage3_go_final_yellow_slow_speed', 0.10)

        self.declare_parameter('turn_trigger_distance_m', 0.45)

        # 中线对齐：使用固定 vy 横向平移修正。
        self.declare_parameter('center_cruise_vy_gain', 0.25)  # 保留兼容，当前不再使用
        self.declare_parameter('center_cruise_vy_max', 0.3)  # 保留兼容，当前不再使用
        self.declare_parameter('center_ok_px', 10.0)
        self.declare_parameter('center_cruise_fixed_vy', 0.06)
        # 左右参考球深度差太大时，不再按两球图像中点做中线对齐，
        # 而是向距离更远的小球一侧给一个较小固定 vy。
        self.declare_parameter('center_depth_diff_disable_align_m', 0.75)
        self.declare_parameter('center_far_side_fixed_vy', 0.02)
        # 只看到一侧参考球时，用小速度横移，尝试把另一侧参考球重新带回画面。
        # 当前方向约定：vy>0 向左，vy<0 向右。
        self.declare_parameter('center_single_side_fixed_vy', 0.02)

        # =========================
        # 对齐球阶段：小 vx + 主 vy
        # =========================
        self.declare_parameter('lateral_align_forward_speed', 0.08)
        self.declare_parameter('lateral_align_vy_gain', 0.20)
        self.declare_parameter('lateral_align_vy_max', 0.18)
        self.declare_parameter('lateral_align_vy_min', 0.06)
        self.declare_parameter('lateral_align_px_tol', 20.0)
        self.declare_parameter('lateral_align_confirm_count', 1)

        # =========================
        # 对齐球阶段：目标丢失 / 深度突然变远保护
        # =========================
        # 对齐过程中如果目标球突然识别不到：
        # 认为机器狗已经离球很近，球进入相机盲区/穿模，直接开始撞击。
        self.declare_parameter('ball_align_lost_go_hit', True)

        # 对齐过程中如果 best_target_ball 深度突然变大：
        # 认为近处 A 球丢失，当前识别到的是远处 B 球，不继续对齐，直接撞击。
        self.declare_parameter('ball_align_depth_jump_enabled', True)

        # 深度增加超过这个值，认为是跳变。
        # 例如上一帧 0.30m，下一帧 0.70m，增加 0.40m，就触发。
        self.declare_parameter('ball_align_depth_jump_threshold_m', 0.25)

        # 只有曾经看到目标小于这个距离，才启用“突然变远 -> 直接撞击”。
        # 避免远距离正常识别波动时误触发。
        self.declare_parameter('ball_align_near_depth_for_jump_m', 0.45)

        # =========================
        # 撞击 / 撞后移动
        # =========================
        # 撞击前冲：按仿真时间结束，不再用 TF 距离和 hit_extra_distance_m。
        self.declare_parameter('hit_forward_speed', 0.15)
        self.declare_parameter('hit_forward_duration_sec', 0.7)

        # 撞完后左右移动：固定速度 + 固定仿真时间。
        # 不再使用 post_hit_side_shift_distance_m / fast / slow / slowdown_ratio。
        self.declare_parameter('post_hit_side_shift_speed', 0.20)
        self.declare_parameter('post_hit_side_shift_duration_sec', 1.5)

        # =========================
        # 防重复撞同一颗球
        # =========================
        # 防重复触发现在只按仿真时间冷却判断，不再用 TF 位移。
        self.declare_parameter('ball_retrigger_cooldown_sec', 0.5)

        # 先按仿真时间移动一段，再按仿真时间转向一段。
        self.declare_parameter('stage3_final_left_shift_speed', 0.20)
        self.declare_parameter('stage3_final_left_shift_duration_sec', 1.0)
        self.declare_parameter('stage3_final_rotate_wz', 0.60)
        self.declare_parameter('stage3_final_rotate_duration_sec', 1.75)

        # 第三条黄线触发后、180° 掉头前，先固定向左横移一段。
        # 与最终出口左移参数分开，避免两处互相影响。
        self.declare_parameter('stage2_before_second_90_left_shift_speed', 0.20)
        self.declare_parameter('stage2_before_second_90_left_shift_duration_sec', 1.0)

        # =========================
        # 第二段左跳后按仿真时间前进
        # =========================
        self.declare_parameter('stage2_forward_after_left_jump_speed', 0.20)
        self.declare_parameter('stage2_forward_after_left_jump_duration_sec', 0.3)

        # =========================
        # 用固定角速度 + 固定仿真时间代替原地左跳转向
        # =========================
        # 实际值需要按仿真里机器狗的真实转角微调。
        self.declare_parameter('timed_turn_wz_90', 0.60)

        # 第一次 90° 左转专用时间。
        # 第二次 90° 仍继续使用 timed_turn_duration_90_sec = 3.85s。
        self.declare_parameter('stage1_timed_turn_duration_90_sec', 3.95)
        self.declare_parameter('timed_turn_duration_90_sec', 3.85)
        self.declare_parameter('timed_turn_wz_180', 0.60)
        self.declare_parameter('timed_turn_duration_180_sec', 7.7)
        self.declare_parameter('timed_turn_step_height', 0.02)
        # 所有固定时间转向开始前，先原地踏步并恢复水平机身姿态。
        self.declare_parameter('timed_turn_pre_stand_duration_sec', 0.5)

        # 第二赛段机身前倾：
        # 默认所有速度动作（巡航、靠近球、侧撞、撞后恢复、平移）均保持前倾；
        # 只有按时间执行的原地转向状态恢复为水平姿态。
        # 若实测前倾方向相反，把负值改成正值。
        self.declare_parameter('stage2_body_pitch', 0.15)
        self.declare_parameter('stage2_velocity_step_height', 0.05)

        # 第一次 90° 转向前固定向前走一段。
        # 该动作保持 stage2_body_pitch；进入原转向状态后仍由 send_velocity_command 自动恢复 pitch=0。
        self.declare_parameter('stage1_before_turn_forward_vx', 0.15)
        self.declare_parameter('stage1_before_turn_forward_duration_s', 1.0)

        self.p2_entry_table = p2_entry_table()
        self.second_stage_initial_state = self.resolve_stage_entry(
            self.p2_entry_table,
            str(self.get_parameter('second_stage_initial_state').value))
        self.p2_ball_return_state = self.resolve_p2_ball_return_state()

        self.orange_h_min = int(self.get_parameter('orange_h_min').value)
        self.orange_h_max = int(self.get_parameter('orange_h_max').value)
        self.orange_s_min = int(self.get_parameter('orange_s_min').value)
        self.orange_s_max = int(self.get_parameter('orange_s_max').value)
        self.orange_v_min = int(self.get_parameter('orange_v_min').value)
        self.orange_v_max = int(self.get_parameter('orange_v_max').value)
        self.orange_min_contour_area = float(self.get_parameter('orange_min_contour_area').value)

        self.blue_h_min = int(self.get_parameter('blue_h_min').value)
        self.blue_h_max = int(self.get_parameter('blue_h_max').value)
        self.blue_s_min = int(self.get_parameter('blue_s_min').value)
        self.blue_s_max = int(self.get_parameter('blue_s_max').value)
        self.blue_v_min = int(self.get_parameter('blue_v_min').value)
        self.blue_v_max = int(self.get_parameter('blue_v_max').value)
        self.blue_min_contour_area = float(self.get_parameter('blue_min_contour_area').value)

        self.prefer_nearest_ball = bool(self.get_parameter('prefer_nearest_ball').value)
        self.min_ball_radius_to_trigger = float(self.get_parameter('min_ball_radius_to_trigger').value)

        self.depth_search_half = int(self.get_parameter('depth_search_half').value)
        self.valid_min_depth_m = float(self.get_parameter('valid_min_depth_m').value)
        self.valid_max_depth_m = float(self.get_parameter('valid_max_depth_m').value)

        self.fisheye_left_topic = str(self.get_parameter('fisheye_left_topic').value)
        self.fisheye_right_topic = str(self.get_parameter('fisheye_right_topic').value)
        self.fisheye_orange_min_contour_area = float(self.get_parameter('fisheye_orange_min_contour_area').value)
        self.fisheye_morph_kernel_size = max(1, int(self.get_parameter('fisheye_morph_kernel_size').value))
        if self.fisheye_morph_kernel_size % 2 == 0:
            self.fisheye_morph_kernel_size += 1
        self.fisheye_min_circularity = float(self.get_parameter('fisheye_min_circularity').value)
        self.fisheye_min_aspect_ratio = float(self.get_parameter('fisheye_min_aspect_ratio').value)
        self.fisheye_max_aspect_ratio = float(self.get_parameter('fisheye_max_aspect_ratio').value)
        self.fisheye_min_circle_fill_ratio = float(self.get_parameter('fisheye_min_circle_fill_ratio').value)
        self.fisheye_min_bbox_fill_ratio = float(self.get_parameter('fisheye_min_bbox_fill_ratio').value)
        self.fisheye_entry_x_min_ratio = float(
            self.get_parameter('fisheye_entry_x_min_ratio').value)
        self.fisheye_entry_x_max_ratio = float(
            self.get_parameter('fisheye_entry_x_max_ratio').value)
        self.fisheye_entry_center_y_ratio = float(
            self.get_parameter('fisheye_entry_center_y_ratio').value)
        self.fisheye_entry_y_tolerance = float(
            self.get_parameter('fisheye_entry_y_tolerance').value)
        self.fisheye_entry_confirm_frames = max(1, int(self.get_parameter('fisheye_entry_confirm_frames').value))
        self.fisheye_approach_vy = abs(float(self.get_parameter('fisheye_approach_vy').value))
        self.fisheye_approach_lost_frames = max(1, int(self.get_parameter('fisheye_approach_lost_frames').value))
        self.fisheye_approach_target_x_ratio = float(self.get_parameter('fisheye_approach_target_x_ratio').value)
        self.fisheye_approach_x_deadband_ratio = float(self.get_parameter('fisheye_approach_x_deadband_ratio').value)
        self.fisheye_approach_vx_k = float(self.get_parameter('fisheye_approach_vx_k').value)
        self.fisheye_approach_vx_max = abs(float(self.get_parameter('fisheye_approach_vx_max').value))
        self.left_fisheye_x_to_vx_sign = float(self.get_parameter('left_fisheye_x_to_vx_sign').value)
        self.right_fisheye_x_to_vx_sign = float(self.get_parameter('right_fisheye_x_to_vx_sign').value)
        self.fisheye_hit_radius = float(self.get_parameter('fisheye_hit_radius').value)
        self.fisheye_hit_radius_confirm_frames = max(1, int(self.get_parameter('fisheye_hit_radius_confirm_frames').value))
        self.fisheye_hit_vy = abs(float(self.get_parameter('fisheye_hit_vy').value))
        self.fisheye_hit_duration_sec = float(self.get_parameter('fisheye_hit_duration_sec').value)
        self.fisheye_recover_forward_vx = float(self.get_parameter('fisheye_recover_forward_vx').value)
        self.fisheye_recover_vy = abs(float(self.get_parameter('fisheye_recover_vy').value))
        self.fisheye_recover_duration_sec = float(self.get_parameter('fisheye_recover_duration_sec').value)

        self.yellow_roi_top_ratio = float(self.get_parameter('yellow_roi_top_ratio').value)
        self.yellow_roi_left_ratio = float(self.get_parameter('yellow_roi_left_ratio').value)
        self.yellow_roi_right_ratio = float(self.get_parameter('yellow_roi_right_ratio').value)

        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.yellow_s_min = int(self.get_parameter('yellow_s_min').value)
        self.yellow_s_max = int(self.get_parameter('yellow_s_max').value)
        self.yellow_v_min = int(self.get_parameter('yellow_v_min').value)
        self.yellow_v_max = int(self.get_parameter('yellow_v_max').value)
        self.yellow_min_contour_area = float(self.get_parameter('yellow_min_contour_area').value)

        self.yellow_min_width_height_ratio = float(self.get_parameter('yellow_min_width_height_ratio').value)
        self.yellow_max_tilt_deg = float(self.get_parameter('yellow_max_tilt_deg').value)
        self.yellow_center_tolerance_ratio = float(self.get_parameter('yellow_center_tolerance_ratio').value)
        self.yellow_min_width_ratio = float(self.get_parameter('yellow_min_width_ratio').value)

        self.yellow_stop_line_y_ratio_stage1 = float(self.get_parameter('yellow_stop_line_y_ratio_stage1').value)
        self.yellow_stop_line_y_ratio_stage2 = float(self.get_parameter('yellow_stop_line_y_ratio_stage2').value)
        self.yellow_stop_line_y_ratio_stage3 = float(self.get_parameter('yellow_stop_line_y_ratio_stage3').value)
        self.yellow_stop_confirm_count = int(self.get_parameter('yellow_stop_confirm_count').value)

        self.yellow_ratio_final = float(self.get_parameter('yellow_ratio_final').value)

        self.yellow_angle_align_enabled = bool(self.get_parameter('yellow_angle_align_enabled').value)
        self.yellow_angle_align_fixed_wz = abs(float(self.get_parameter('yellow_angle_align_fixed_wz').value))
        self.yellow_angle_align_deadband_deg = float(self.get_parameter('yellow_angle_align_deadband_deg').value)

        self.stage1_cruise_forward_speed = float(self.get_parameter('stage1_cruise_forward_speed').value)
        self.stage2_cruise_forward_speed = float(self.get_parameter('stage2_cruise_forward_speed').value)

        self.stage2_after_turn_left_bias_delay_sec = max(
            0.0,
            float(self.get_parameter('stage2_after_turn_left_bias_delay_sec').value)
        )
        self.stage2_cruise_forward_speed_after_delay = float(
            self.get_parameter('stage2_cruise_forward_speed_after_delay').value
        )
        self.stage2_after_turn_left_bias_vy = abs(float(
            self.get_parameter('stage2_after_turn_left_bias_vy').value
        ))

        self.stage3_cruise_ball_only_speed = float(self.get_parameter('stage3_cruise_ball_only_speed').value)

        self.stage2_left_ball_avoid_enabled = bool(self.get_parameter('stage2_left_ball_avoid_enabled').value)
        self.stage2_left_ball_avoid_center_px = float(self.get_parameter('stage2_left_ball_avoid_center_px').value)
        self.stage2_left_ball_avoid_vy = abs(float(self.get_parameter('stage2_left_ball_avoid_vy').value))
        self.stage2_left_ball_avoid_confirm_frames = int(self.get_parameter('stage2_left_ball_avoid_confirm_frames').value)
        self.stage2_left_ball_avoid_min_area = float(
            self.get_parameter('stage2_left_ball_avoid_min_area').value)
        self.stage2_left_ball_avoid_min_radius = float(
            self.get_parameter('stage2_left_ball_avoid_min_radius').value)
        self.stage3_go_final_speed = float(self.get_parameter('stage3_go_final_speed').value)
        self.stage3_forward_after_go_final_vx = float(
            self.get_parameter('stage3_forward_after_go_final_vx').value
        )
        self.stage3_forward_after_go_final_duration_s = max(
            0.0,
            float(self.get_parameter('stage3_forward_after_go_final_duration_s').value)
        )

        self.yellow_slowdown_ratio_stage1 = float(self.get_parameter('yellow_slowdown_ratio_stage1').value)
        self.yellow_slowdown_ratio_stage2 = float(self.get_parameter('yellow_slowdown_ratio_stage2').value)
        self.yellow_slowdown_ratio_stage3 = float(self.get_parameter('yellow_slowdown_ratio_stage3').value)
        self.yellow_slowdown_ratio_final = float(self.get_parameter('yellow_slowdown_ratio_final').value)

        self.stage1_yellow_slow_speed = float(self.get_parameter('stage1_yellow_slow_speed').value)
        self.stage2_yellow_slow_speed = float(self.get_parameter('stage2_yellow_slow_speed').value)
        self.stage3_yellow_slow_speed = float(self.get_parameter('stage3_yellow_slow_speed').value)
        self.stage3_go_final_yellow_slow_speed = float(self.get_parameter('stage3_go_final_yellow_slow_speed').value)

        self.turn_trigger_distance_m = float(self.get_parameter('turn_trigger_distance_m').value)

        self.center_cruise_vy_gain = float(self.get_parameter('center_cruise_vy_gain').value)
        self.center_cruise_vy_max = float(self.get_parameter('center_cruise_vy_max').value)
        self.center_ok_px = float(self.get_parameter('center_ok_px').value)
        self.center_cruise_fixed_vy = abs(float(self.get_parameter('center_cruise_fixed_vy').value))
        self.center_depth_diff_disable_align_m = float(
            self.get_parameter('center_depth_diff_disable_align_m').value
        )
        self.center_far_side_fixed_vy = abs(float(self.get_parameter('center_far_side_fixed_vy').value))
        self.center_single_side_fixed_vy = abs(float(
            self.get_parameter('center_single_side_fixed_vy').value))

        self.lateral_align_forward_speed = float(self.get_parameter('lateral_align_forward_speed').value)
        self.lateral_align_vy_gain = float(self.get_parameter('lateral_align_vy_gain').value)
        self.lateral_align_vy_max = float(self.get_parameter('lateral_align_vy_max').value)
        self.lateral_align_vy_min = float(self.get_parameter('lateral_align_vy_min').value)
        self.lateral_align_px_tol = float(self.get_parameter('lateral_align_px_tol').value)
        self.lateral_align_confirm_count = int(self.get_parameter('lateral_align_confirm_count').value)

        self.ball_align_lost_go_hit = bool(self.get_parameter('ball_align_lost_go_hit').value)
        self.ball_align_depth_jump_enabled = bool(self.get_parameter('ball_align_depth_jump_enabled').value)
        self.ball_align_depth_jump_threshold_m = float(
            self.get_parameter('ball_align_depth_jump_threshold_m').value
        )
        self.ball_align_near_depth_for_jump_m = float(
            self.get_parameter('ball_align_near_depth_for_jump_m').value
        )

        self.hit_forward_speed = float(self.get_parameter('hit_forward_speed').value)
        self.hit_forward_duration_sec = float(self.get_parameter('hit_forward_duration_sec').value)

        self.post_hit_side_shift_speed = float(self.get_parameter('post_hit_side_shift_speed').value)
        self.post_hit_side_shift_duration_sec = float(self.get_parameter('post_hit_side_shift_duration_sec').value)

        self.ball_retrigger_cooldown_sec = float(self.get_parameter('ball_retrigger_cooldown_sec').value)

        self.stage3_final_left_shift_speed = float(self.get_parameter('stage3_final_left_shift_speed').value)
        self.stage3_final_left_shift_duration_sec = float(
            self.get_parameter('stage3_final_left_shift_duration_sec').value
        )
        self.stage3_final_rotate_wz = float(self.get_parameter('stage3_final_rotate_wz').value)
        self.stage3_final_rotate_duration_sec = float(
            self.get_parameter('stage3_final_rotate_duration_sec').value
        )
        self.stage2_before_second_90_left_shift_speed = abs(float(
            self.get_parameter('stage2_before_second_90_left_shift_speed').value
        ))
        self.stage2_before_second_90_left_shift_duration_sec = max(
            0.0,
            float(self.get_parameter('stage2_before_second_90_left_shift_duration_sec').value)
        )

        self.stage2_forward_after_left_jump_speed = float(
            self.get_parameter('stage2_forward_after_left_jump_speed').value)
        self.stage2_forward_after_left_jump_duration_sec = float(
            self.get_parameter('stage2_forward_after_left_jump_duration_sec').value)

        self.timed_turn_wz_90 = float(self.get_parameter('timed_turn_wz_90').value)

        self.stage1_timed_turn_duration_90_sec = float(
            self.get_parameter('stage1_timed_turn_duration_90_sec').value
        )

        # 第二次 90° 继续使用原来的 3.85s。
        self.timed_turn_duration_90_sec = float(
            self.get_parameter('timed_turn_duration_90_sec').value
        )
        self.timed_turn_wz_180 = float(self.get_parameter('timed_turn_wz_180').value)
        self.timed_turn_duration_180_sec = float(self.get_parameter('timed_turn_duration_180_sec').value)
        self.timed_turn_step_height = float(self.get_parameter('timed_turn_step_height').value)
        self.timed_turn_pre_stand_duration_sec = max(
            0.0,
            float(self.get_parameter('timed_turn_pre_stand_duration_sec').value)
        )

        self.stage2_body_pitch = float(
            self.get_parameter('stage2_body_pitch').value
        )
        self.stage2_velocity_step_height = float(
            self.get_parameter('stage2_velocity_step_height').value
        )
        self.stage1_before_turn_forward_vx = float(
            self.get_parameter('stage1_before_turn_forward_vx').value
        )
        self.stage1_before_turn_forward_duration_s = max(
            0.0,
            float(self.get_parameter('stage1_before_turn_forward_duration_s').value)
        )
        # 这些状态属于固定时间原地转向：
        # - 控制循环使用 30 Hz；
        # - RGB / Depth / 双鱼眼视觉处理全部暂停；
        # - 发送速度命令时恢复 pitch=0。
        # 这些“固定时间动作”使用 30 Hz，并暂停全部视觉处理。
        # STAGE2_LEFT_SHIFT_BEFORE_SECOND_90 虽然不是转向，也放在这里，
        # 这样第二次90°前的固定横移计时更精确，同时减轻视觉算力。
        self.stage2_timed_turn_states = {
            'STAGE1_ROTATE_LEFT_90',
            'STAGE2_ROTATE_LEFT_90',
            'STAGE2_LEFT_SHIFT_BEFORE_SECOND_90',
            'STAGE3_ROTATE_BACK_180',
            'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT',
        }
        # 只有真正的原地转向/转向前踏步阶段恢复水平机身。
        # 第二次90°前的左移仍保持第二赛段正常前倾姿态。
        self.stage2_level_body_states = {
            'STAGE1_ROTATE_LEFT_90',
            'STAGE2_ROTATE_LEFT_90',
            'STAGE3_ROTATE_BACK_180',
            'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT',
        }
        self.stage2_rgb_required_states = {
            'STAGE1_CRUISE_BALL_AND_YELLOW',
            'STAGE2_CRUISE_YELLOW_ONLY',
            'STAGE3_PRE180_CENTER_YELLOW',
            'STAGE3_GO_FINAL',
        }

        self.rgb_w = 640
        self.rgb_h = 480

        self.latest_ball_result = {
            'has_ball': False,
            'ball_center': None,
            'ball_radius': None,
            'ball_depth_m': None,
            'img_shape': None,
            'error_x': None,
            'aligned': False,
            'depth_center': None,
            'depth_box': None,

            'orange_balls': [],
            'blue_balls': [],
            # STAGE2_CRUISE_YELLOW_ONLY 专用 RGB-only 避让候选；
            # 不要求 depth 有效，因此不会受深度抖动影响。
            'stage2_avoid_rgb_balls': [],
            'left_balls': [],
            'right_balls': [],
            'has_center_reference': False,
            'center_error_px': None,
            'left_ref': None,
            'right_ref': None,
            'best_target_ball': None,
        }

        # 第二赛段左侧近球避让缓存：用于连续帧确认和可视化。
        self.stage2_left_ball_avoid_counter = 0
        self.stage2_left_ball_avoid_active = False
        self.stage2_left_ball_avoid_debug = {
            'enabled': self.stage2_left_ball_avoid_enabled,
            'active': False,
            'counter': 0,
            'danger_ball': None,
            'candidate_count': 0,
            'vy': 0.0,
            'reason': 'init',
        }

        self.latest_yellow_result = {
            'has_line': False,
            'line_bottom_y': None,
            'line_center': None,
            'img_shape': None,
            'angle_deg': None,
            'abs_tilt_deg': None,
            'bbox': None,
            'width_ratio': None,
            'wh_ratio': None,
            'require_front_horizontal': None,
        }

        # self.state 由 StageNodeBase 管理；赛段入口状态在 on_activated() 里设置。
        # 整合后 initial_state 是 P1_STAND_WAIT；撞球子链回退状态必须默认指向第二赛段入口，
        # 避免异常路径下撞球结束后跳回第一赛段。
        self.ball_return_state = self.p2_ball_return_state

        self.yellow_stop_counter = 0

        # 第一阶段黄线“到底后出图”逻辑
        self.stage1_yellow_touched_bottom = False
        self.stage1_yellow_disappear_counter = 0

        self.pre_turn_pose: Optional[Tuple[float, float, float]] = None
        self.last_ball_done_time_sec: Optional[float] = None
        self.last_ball_done_pose: Optional[Tuple[float, float, float]] = None

        # 第一次 90° 后巡航的计时起点。
        self.stage2_cruise_after_first_turn_start_time_sec: Optional[float] = None

        # 只要 STAGE2_CRUISE_YELLOW_ONLY 中曾经真正触发过一次危险小球避障，
        # 就锁存为 True。之后即使小球消失，也不再给主动左移偏置。
        self.stage2_danger_ball_seen_once = False

        self.stage2_forward_after_left_jump_start_time_sec: Optional[float] = None
        self.stage2_before_second_90_left_shift_start_time_sec: Optional[float] = None
        self.stage3_final_left_shift_start_time_sec: Optional[float] = None
        self.stage3_final_rotate_start_time_sec: Optional[float] = None
        self.stage3_forward_after_go_final_start_time_sec: Optional[float] = None
        self.timed_turn_start_time_sec: Optional[float] = None
        self.timed_turn_pre_stand_start_time_sec: Optional[float] = None
        self.before_turn_forward_start_time_sec: Optional[float] = None

        self.lateral_align_counter = 0

        # BALL_LATERAL_ALIGN 阶段记录目标球深度变化。
        # 用于判断：近处球是否丢失、是否误切到远处其他球。
        self.ball_align_last_depth_m: Optional[float] = None
        self.ball_align_min_seen_depth_m: Optional[float] = None

        self.hit_start_pose: Optional[Tuple[float, float, float]] = None
        self.hit_start_depth_m: Optional[float] = None
        self.hit_start_time_sec: Optional[float] = None

        self.post_hit_side_shift_start_pose: Optional[Tuple[float, float, float]] = None
        self.post_hit_side_shift_start_time_sec: Optional[float] = None
        self.last_hit_side: Optional[str] = None
        self.side_shift_done: bool = False

        # 第一、第三巡航阶段分别记录已经成功撞过橙球的鱼眼侧。
        # 同一阶段内每侧最多只有一个橙球；撞完后不再处理该侧，
        # 但第一阶段的记录不会影响第三阶段。
        self.fisheye_hit_sides_by_stage = {
            'stage1': set(),
            'stage3': set(),
        }

        self.center_cruise_debug_info = {
            'mode': 'INIT',
            'left_depth': None,
            'right_depth': None,
            'depth_diff': None,
            'center_error_px': None,
            'vy': 0.0,
        }

        # 视觉消息只缓存最新一帧；真正的 cv_bridge + OpenCV 检测统一在
        # 5 Hz 控制拍开始时执行。这样不会让 RGB/Depth/鱼眼回调把单线程
        # executor 挤满，也不会排队处理旧图像。定时转向状态仍完全不处理视觉。
        self._pending_rgb_msg: Optional[Image] = None
        self._pending_depth_msg: Optional[Image] = None
        self._pending_fisheye_left_msg: Optional[Image] = None
        self._pending_fisheye_right_msg: Optional[Image] = None

        # 双鱼眼检测结果缓存。巡航居中仍使用原 RGB + Depth 逻辑。
        self.latest_fisheye_left_target: Optional[Dict] = None
        self.latest_fisheye_right_target: Optional[Dict] = None
        self.fisheye_left_entry_counter = 0
        self.fisheye_right_entry_counter = 0
        self.fisheye_target_side: Optional[str] = None
        self.fisheye_hit_radius_counter = 0
        self.fisheye_approach_lost_counter = 0
        self.fisheye_hit_start_time_sec: Optional[float] = None
        self.fisheye_recover_start_time_sec: Optional[float] = None

        self.fisheye_left_sub = self.create_subscription(
            Image, self.fisheye_left_topic,
            self.fisheye_left_callback, qos_profile_sensor_data
        )
        self.fisheye_right_sub = self.create_subscription(
            Image, self.fisheye_right_topic,
            self.fisheye_right_callback, qos_profile_sensor_data
        )

        self.get_logger().info(
            f'Stage2Node ready. fisheye_left={self.fisheye_left_topic}, '
            f'fisheye_right={self.fisheye_right_topic}'
        )
        self.get_logger().info(
            f'[P2_RATE] normal={self.stage2_control_hz:.1f} Hz, '
            f'timed_turn={self.stage2_timed_turn_control_hz:.1f} Hz, '
            f'timed_turn_vision=OFF'
        )

    def send_velocity_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        pitch: Optional[float] = None,
        step_height: Optional[float] = None
    ):
        """
        第二赛段专用速度命令。

        默认行为：
        - 巡航、鱼眼靠近、侧撞、撞后恢复、直行和平移：保持 stage2_body_pitch；
        - 定时原地转向状态：自动恢复 pitch=0；
        - 可通过 pitch 参数显式覆盖自动选择结果。

        该函数在 Stage2Node 内覆盖 StageNodeBase.send_velocity_command，
        因此第二赛段现有所有 self.send_velocity_command(...) 调用都会自动生效。
        """
        if getattr(self, 'Ctrl', None) is None:
            self.get_logger().warning(
                '[STAGE2_CMD] Robot_Ctrl is not active; velocity command ignored',
                throttle_duration_sec=1.0
            )
            return

        # 显式传入 pitch 时优先使用；否则按当前状态自动决定。
        if pitch is None:
            if self.state in self.stage2_level_body_states:
                command_pitch = 0.0
                pitch_mode = 'LEVEL_FOR_TIMED_TURN'
            else:
                command_pitch = self.stage2_body_pitch
                pitch_mode = 'FORWARD_LEAN'
        else:
            command_pitch = float(pitch)
            pitch_mode = 'EXPLICIT'

        if step_height is None:
            if self.state in self.stage2_level_body_states:
                command_step_height = self.timed_turn_step_height
            else:
                command_step_height = self.stage2_velocity_step_height
        else:
            command_step_height = float(step_height)

        # CyberDog continuous motion.  Stage 2 keeps its old gait=27
        # semantic tag; the real adapter maps it to a configurable motion_id.
        self.Ctrl.move(
            float(vx), float(vy), float(wz),
            step_height=float(command_step_height),
            pitch=float(command_pitch),
            body_height=0.0,
            legacy_gait_id=27,
        )

        self.get_logger().info(
            f'[STAGE2_CMD] state={self.state}, '
            f'vel=[{float(vx):+.3f}, {float(vy):+.3f}, {float(wz):+.3f}], '
            f'pitch={command_pitch:+.3f} ({pitch_mode}), '
            f'step_height={command_step_height:.3f}',
            throttle_duration_sec=0.5
        )

    def send_recovery_stand_and_wait(self) -> bool:
        """Execute RecoveryStand using the selected robot backend."""
        if getattr(self, 'Ctrl', None) is None:
            self.get_logger().warning(
                '[RECOVERY] robot backend is not active; stand command ignored'
            )
            return False

        self.get_logger().info('[RECOVERY] send recovery stand')
        finished = bool(self.Ctrl.recovery_stand(wait_finish=True))
        if finished:
            self.get_logger().info('[RECOVERY] recovery stand finished')
        else:
            self.get_logger().warning('[RECOVERY] recovery stand failed or timed out')
        return finished

    def resolve_p2_ball_return_state(self) -> str:
        """撞球子链打完球后回到的巡航状态。

        正常情况就是本次的入口状态。但从撞球子链本身启动调试时，那会让子链
        回到自己形成死循环，所以要么用 ``p2_ball_return_state`` 指定一条巡航
        赛道，要么退回赛道 1 并明确告警。
        """
        requested = str(self.get_parameter('p2_ball_return_state').value)
        if not is_default_request(requested):
            resolution = self.p2_entry_table.resolve(requested)
            if not resolution.ok:
                self.get_logger().error(
                    '[ENTRY] p2_ball_return_state: ' + resolution.message)
            elif resolution.state in P2_BALL_SUBCHAIN_STATES:
                self.get_logger().error(
                    f'[ENTRY] p2_ball_return_state={resolution.state} is itself a '
                    'ball sub-chain state; ignoring it')
            else:
                self.get_logger().warn(
                    f'[ENTRY] ball sub-chain returns to {resolution.state}')
                return resolution.state

        entry = self.second_stage_initial_state
        if entry in P2_BALL_SUBCHAIN_STATES:
            fallback = P2_CRUISE_STATES[0]
            self.get_logger().warn(
                f'[ENTRY] stage 2 starts inside the ball sub-chain ({entry}) '
                'without p2_ball_return_state; the sub-chain will return to '
                f'{fallback}')
            return fallback
        return entry

    def on_activated(self):
        # 每次激活都让普通 5 Hz 控制立即执行第一拍。
        self._stage2_next_low_rate_tick_monotonic = None

        # Full mission startup already performs the single RecoveryStand(111).
        # Do not inject another preset action merely because Stage2 becomes active.

        # 清理第二赛段内部缓存。
        self.yellow_stop_counter = 0
        self.stage1_yellow_touched_bottom = False
        self.stage1_yellow_disappear_counter = 0

        self.lateral_align_counter = 0
        self.ball_align_last_depth_m = None
        self.ball_align_min_seen_depth_m = None

        self.hit_start_pose = None
        self.hit_start_depth_m = None
        self.hit_start_time_sec = None
        self.post_hit_side_shift_start_pose = None
        self.post_hit_side_shift_start_time_sec = None
        self.last_hit_side = None
        self.side_shift_done = False
        self.ball_return_state = self.p2_ball_return_state

        self.stage2_forward_after_left_jump_start_time_sec = None
        self.stage2_before_second_90_left_shift_start_time_sec = None
        self.stage3_final_left_shift_start_time_sec = None
        self.stage3_final_rotate_start_time_sec = None
        self.stage3_forward_after_go_final_start_time_sec = None
        self.timed_turn_start_time_sec = None
        self.timed_turn_pre_stand_start_time_sec = None
        self.before_turn_forward_start_time_sec = None

        self.last_ball_done_time_sec = None
        self.last_ball_done_pose = None

        # 每次重新激活第二赛段都从全新的赛道目标记录开始。
        for hit_sides in self.fisheye_hit_sides_by_stage.values():
            hit_sides.clear()

        self.reset_fisheye_hit_context(clear_entry=True)

        self.set_state(self.second_stage_initial_state)
        # Robot_Ctrl starts with heartbeat disabled until the first intentional
        # command.  For the normal entry state, immediately continue forward
        # so P1 -> P2 does not expose a mode=0/kOff window while vision is
        # converted below.  Debug entry states get a safe zero-velocity
        # locomotion command until their first control tick chooses the exact
        # action.
        initial_vx = (
            self.stage1_cruise_forward_speed
            if self.state == 'STAGE1_CRUISE_BALL_AND_YELLOW'
            else 0.0
        )
        self.send_velocity_command(initial_vx, 0.0, 0.0)

        # 激活前公共节点已经缓存了最新原始 RGB/Depth 消息。这里只把它们作为
        # 第一拍的 pending 输入，真正的转换/检测仍统一放到 5 Hz 控制拍里。
        self._pending_rgb_msg = None
        self._pending_depth_msg = None
        self._pending_fisheye_left_msg = None
        self._pending_fisheye_right_msg = None
        if not self.is_stage2_timed_turn_state():
            if self.latest_rgb_msg is not None:
                self._pending_rgb_msg = self.latest_rgb_msg
            if self.latest_depth_msg is not None:
                self._pending_depth_msg = self.latest_depth_msg

    def is_stage2_timed_turn_state(self, state: Optional[str] = None) -> bool:
        """固定时间高精度动作使用 30 Hz，并暂停全部视觉处理。"""
        target_state = self.state if state is None else state
        return target_state in self.stage2_timed_turn_states

    def stage2_control_tick_due(self) -> bool:
        """30 Hz 底层定时器之上实现：普通状态 5 Hz，固定时间动作状态 30 Hz。"""
        if self.is_stage2_timed_turn_state():
            return True

        now = time.monotonic()
        if self._stage2_next_low_rate_tick_monotonic is None:
            self._stage2_next_low_rate_tick_monotonic = now + self.stage2_control_period_s
            return True

        if now < self._stage2_next_low_rate_tick_monotonic:
            return False

        # 用固定周期向前推进 deadline，避免直接用 now 重置造成长期频率漂移。
        while self._stage2_next_low_rate_tick_monotonic <= now:
            self._stage2_next_low_rate_tick_monotonic += self.stage2_control_period_s
        return True

    def handle_rgb_msg(self, msg: Image):
        """RGB 回调只覆盖保存最新消息；5 Hz 控制拍再转换和检测。"""
        if self.is_stage2_timed_turn_state():
            self._pending_rgb_msg = None
            return
        self._pending_rgb_msg = msg

    def depth_callback(self, msg: Image):
        """Depth 回调只覆盖保存最新消息；避免全帧率做 cv_bridge 转换。"""
        self.latest_depth_msg = msg
        if not self.active or self.finished or self.is_stage2_timed_turn_state():
            self._pending_depth_msg = None
            return
        self._pending_depth_msg = msg

    def process_stage2_latest_vision(self):
        """每个普通 5 Hz 控制拍只消费各相机最新一帧。

        顺序先 Depth、后 RGB，保证 RGB 球检测读取到尽可能新的深度；
        两路鱼眼也各最多处理一帧。中间到来的旧消息会被直接覆盖。
        """
        if self.is_stage2_timed_turn_state():
            self._pending_rgb_msg = None
            self._pending_depth_msg = None
            self._pending_fisheye_left_msg = None
            self._pending_fisheye_right_msg = None
            return

        ball_subchain_active = self.state in P2_BALL_SUBCHAIN_STATES

        # 撞球/回退期间 RGB+Depth 不做转换和识别。
        # 黄线必须等 BALL_POST_HIT_SIDE_SHIFT 完成并返回巡航后再处理。
        if ball_subchain_active:
            rgb_processed = False
            self.get_logger().info(
                f'[P2_VISION] state={self.state}: defer RGB/Depth until ball recovery completes',
                throttle_duration_sec=1.0
            )
        else:
            depth_msg = self._pending_depth_msg
            self._pending_depth_msg = None
            if depth_msg is not None:
                StageNodeBase.depth_callback(self, depth_msg)

            rgb_msg = self._pending_rgb_msg
            self._pending_rgb_msg = None
            rgb_processed = rgb_msg is not None
            if rgb_processed:
                StageNodeBase.handle_rgb_msg(self, rgb_msg)

            rgb_age = self.rgb_age_s()
            self.get_logger().info(
                f'[P2_VISION] rgb_new={rgb_processed} rgb_seq={self.latest_rgb_seq} '
                f'rgb_age={rgb_age if rgb_age is not None else -1.0:.3f}s '
                f'yellow={self.latest_yellow_result.get("has_line", False)}',
                throttle_duration_sec=1.0
            )

        fisheye_detection_enabled = (
            ball_subchain_active
            or self.state in (
                'STAGE1_CRUISE_BALL_AND_YELLOW',
                'STAGE3_GO_FINAL',
            )
        )

        if not fisheye_detection_enabled:
            # 第二次90°后到180°之前不做鱼眼图像转换/橙球检测，节省算力。
            self._pending_fisheye_left_msg = None
            self._pending_fisheye_right_msg = None
            self.latest_fisheye_left_target = None
            self.latest_fisheye_right_target = None
            self.fisheye_left_entry_counter = 0
            self.fisheye_right_entry_counter = 0
        else:
            left_msg = self._pending_fisheye_left_msg
            self._pending_fisheye_left_msg = None
            if left_msg is not None:
                if self.is_fisheye_side_ignored('left'):
                    self.latest_fisheye_left_target = None
                else:
                    try:
                        frame = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
                        self.latest_fisheye_left_target = self.detect_fisheye_orange_ball(frame)
                    except Exception as exc:
                        self.get_logger().error(f'left fisheye processing failed: {exc}')

            right_msg = self._pending_fisheye_right_msg
            self._pending_fisheye_right_msg = None
            if right_msg is not None:
                if self.is_fisheye_side_ignored('right'):
                    self.latest_fisheye_right_target = None
                else:
                    try:
                        frame = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
                        self.latest_fisheye_right_target = self.detect_fisheye_orange_ball(frame)
                    except Exception as exc:
                        self.get_logger().error(f'right fisheye processing failed: {exc}')

    def on_rgb_frame(self, frame: np.ndarray):
        self.latest_ball_result = self.detect_ball_scene(frame)
        self.latest_yellow_result = self.detect_yellow_stop_line(frame)
        if self.show_debug_vis:
            self.show_debug_window(frame)

    def can_trigger_ball_again(self, current_pose: Tuple[float, float, float]) -> bool:
        """
        防重复撞球：现在只按仿真时间冷却判断，不再依赖 TF 位移。
        current_pose 参数保留是为了兼容原调用位置。
        """
        if self.last_ball_done_time_sec is None:
            return True

        now_sec = self.now_sec()
        dt = now_sec - self.last_ball_done_time_sec
        cooldown_ok = dt >= self.ball_retrigger_cooldown_sec

        self.get_logger().info(
            f'ball retrigger check by sim time only: '
            f'dt={dt:.2f}s/{self.ball_retrigger_cooldown_sec:.2f}s, '
            f'cooldown_ok={cooldown_ok}',
            throttle_duration_sec=1.0
        )
        return cooldown_ok

    # ============================================================
    # 双鱼眼橙球检测与触发
    # ============================================================
    def get_fisheye_stage_key(self, state: Optional[str] = None) -> Optional[str]:
        """把第一/第三阶段的所有子状态映射到各自的鱼眼屏蔽记录。"""
        target_state = self.state if state is None else state
        if target_state.startswith('BALL_'):
            target_state = self.ball_return_state
        if target_state.startswith('STAGE1_'):
            return 'stage1'
        if target_state.startswith('STAGE3_'):
            return 'stage3'
        return None

    def is_fisheye_side_ignored(
            self,
            side: str,
            state: Optional[str] = None
    ) -> bool:
        """当前第一/第三阶段是否已经成功撞过指定侧的橙球。"""
        stage_key = self.get_fisheye_stage_key(state)
        if stage_key is None:
            return False
        return side in self.fisheye_hit_sides_by_stage[stage_key]

    def mark_fisheye_side_hit(self, return_state: str, side: Optional[str]):
        """撞球子链成功结束后，在对应阶段永久屏蔽这一侧。"""
        if side not in ('left', 'right'):
            self.get_logger().warning(
                f'cannot mark fisheye side as hit: invalid side={side}'
            )
            return

        stage_key = self.get_fisheye_stage_key(return_state)
        if stage_key is None:
            return

        hit_sides = self.fisheye_hit_sides_by_stage[stage_key]
        hit_sides.add(side)

        # 立即清掉该侧缓存和确认帧，避免异步图像回调留下旧目标。
        if side == 'left':
            self.latest_fisheye_left_target = None
            self.fisheye_left_entry_counter = 0
        else:
            self.latest_fisheye_right_target = None
            self.fisheye_right_entry_counter = 0

        self.get_logger().info(
            f'fisheye side disabled after successful hit: '
            f'stage={stage_key}, side={side}, disabled_sides={sorted(hit_sides)}'
        )

    def fisheye_left_callback(self, msg: Image):
        if self.is_stage2_timed_turn_state():
            self._pending_fisheye_left_msg = None
            self.latest_fisheye_left_target = None
            self.fisheye_left_entry_counter = 0
            return
        # 只保留最新消息，不在相机 callback 里做 OpenCV。
        self._pending_fisheye_left_msg = msg

    def fisheye_right_callback(self, msg: Image):
        if self.is_stage2_timed_turn_state():
            self._pending_fisheye_right_msg = None
            self.latest_fisheye_right_target = None
            self.fisheye_right_entry_counter = 0
            return
        # 只保留最新消息，不在相机 callback 里做 OpenCV。
        self._pending_fisheye_right_msg = msg

    def detect_fisheye_orange_ball(self, frame: np.ndarray) -> Optional[Dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.orange_h_min, self.orange_s_min, self.orange_v_min], dtype=np.uint8)
        upper = np.array([self.orange_h_max, self.orange_s_max, self.orange_v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((self.fisheye_morph_kernel_size, self.fisheye_morph_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours = find_contours(mask)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.fisheye_orange_min_contour_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            (cx, cy), enclosing_radius = cv2.minEnclosingCircle(contour)
            if enclosing_radius <= 1e-6:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = bw / float(max(bh, 1))
            circle_fill_ratio = area / max(math.pi * enclosing_radius * enclosing_radius, 1e-6)
            bbox_fill_ratio = area / float(max(bw * bh, 1))
            if circularity < self.fisheye_min_circularity:
                continue
            if not self.fisheye_min_aspect_ratio <= aspect_ratio <= self.fisheye_max_aspect_ratio:
                continue
            if circle_fill_ratio < self.fisheye_min_circle_fill_ratio:
                continue
            if bbox_fill_ratio < self.fisheye_min_bbox_fill_ratio:
                continue
            radius = min(float(enclosing_radius), math.sqrt(area / math.pi))
            candidates.append({
                'center': (int(cx), int(cy)), 'radius': radius, 'area': area,
                'image_shape': frame.shape[:2], 'circularity': circularity,
                'aspect_ratio': aspect_ratio, 'circle_fill_ratio': circle_fill_ratio,
                'bbox_fill_ratio': bbox_fill_ratio,
            })
        return max(candidates, key=lambda item: item['radius']) if candidates else None

    def fisheye_target_near_center(self, target: Optional[Dict]) -> bool:
        if target is None:
            return False
        h, w = target['image_shape']
        cx, cy = target['center']
        x_ratio = cx / float(max(w, 1))
        y_ratio = cy / float(max(h, 1))
        return (
            self.fisheye_entry_x_min_ratio <= x_ratio <= self.fisheye_entry_x_max_ratio
            and abs(y_ratio - self.fisheye_entry_center_y_ratio) <= self.fisheye_entry_y_tolerance
        )

    def update_fisheye_entry_counters(self):
        if self.is_fisheye_side_ignored('left'):
            self.fisheye_left_entry_counter = 0
        elif self.fisheye_target_near_center(self.latest_fisheye_left_target):
            self.fisheye_left_entry_counter += 1
        else:
            self.fisheye_left_entry_counter = 0
        if self.is_fisheye_side_ignored('right'):
            self.fisheye_right_entry_counter = 0
        elif self.fisheye_target_near_center(self.latest_fisheye_right_target):
            self.fisheye_right_entry_counter += 1
        else:
            self.fisheye_right_entry_counter = 0

    def choose_fisheye_entry_side(self) -> Optional[str]:
        left_ready = (
            not self.is_fisheye_side_ignored('left')
            and self.fisheye_left_entry_counter >= self.fisheye_entry_confirm_frames
        )
        right_ready = (
            not self.is_fisheye_side_ignored('right')
            and self.fisheye_right_entry_counter >= self.fisheye_entry_confirm_frames
        )
        if left_ready and not right_ready:
            return 'left'
        if right_ready and not left_ready:
            return 'right'
        if left_ready and right_ready:
            lr = self.latest_fisheye_left_target['radius'] if self.latest_fisheye_left_target else -1.0
            rr = self.latest_fisheye_right_target['radius'] if self.latest_fisheye_right_target else -1.0
            return 'left' if lr >= rr else 'right'
        return None

    def get_locked_fisheye_target(self) -> Optional[Dict]:
        if self.fisheye_target_side == 'left':
            return self.latest_fisheye_left_target
        if self.fisheye_target_side == 'right':
            return self.latest_fisheye_right_target
        return None

    def fisheye_side_sign(self) -> float:
        if self.fisheye_target_side == 'left':
            return 1.0
        if self.fisheye_target_side == 'right':
            return -1.0
        return 0.0

    def compute_fisheye_approach_vx(self, target: Optional[Dict]) -> float:
        if target is None:
            return 0.0
        _, w = target['image_shape']
        cx, _ = target['center']
        error = cx / float(max(w, 1)) - self.fisheye_approach_target_x_ratio
        if abs(error) <= self.fisheye_approach_x_deadband_ratio:
            return 0.0
        sign = self.left_fisheye_x_to_vx_sign if self.fisheye_target_side == 'left' else self.right_fisheye_x_to_vx_sign
        return clamp(sign * self.fisheye_approach_vx_k * error,
                     -self.fisheye_approach_vx_max, self.fisheye_approach_vx_max)

    def reset_fisheye_hit_context(self, clear_entry: bool = False):
        self.fisheye_target_side = None
        self.fisheye_hit_radius_counter = 0
        self.fisheye_approach_lost_counter = 0
        self.fisheye_hit_start_time_sec = None
        self.fisheye_recover_start_time_sec = None
        if clear_entry:
            self.fisheye_left_entry_counter = 0
            self.fisheye_right_entry_counter = 0

    def try_start_fisheye_hit(self, return_state: str, pose: Tuple[float, float, float]) -> bool:
        side = self.choose_fisheye_entry_side()
        if side is None or not self.can_trigger_ball_again(pose):
            return False
        # 最后一层保护：即使图像回调与控制循环恰好并发，也不能再次锁定本阶段已撞侧。
        if self.is_fisheye_side_ignored(side, return_state):
            return False
        self.fisheye_target_side = side
        self.ball_return_state = return_state
        self.last_hit_side = side
        self.fisheye_hit_radius_counter = 0
        self.fisheye_approach_lost_counter = 0
        self.fisheye_left_entry_counter = 0
        self.fisheye_right_entry_counter = 0
        self.get_logger().info(f'fisheye ball locked: side={side}, return_state={return_state}')
        self.set_state('BALL_LATERAL_ALIGN')
        return True

    # ============================================================
    # 深度查值
    # ============================================================
    def get_depth_for_rgb_point(self, rgb_cx: int, rgb_cy: int):
        if self.latest_depth is None or self.latest_depth_encoding is None:
            return None, None, None

        depth = self.latest_depth
        encoding = self.latest_depth_encoding
        dh, dw = depth.shape[:2]

        depth_cx = int(rgb_cx * dw / max(self.rgb_w, 1))
        depth_cy = int(rgb_cy * dh / max(self.rgb_h, 1))

        x1 = max(0, depth_cx - self.depth_search_half)
        x2 = min(dw, depth_cx + self.depth_search_half + 1)
        y1 = max(0, depth_cy - self.depth_search_half)
        y2 = min(dh, depth_cy + self.depth_search_half + 1)

        patch = depth[y1:y2, x1:x2]

        if encoding == '16UC1':
            patch_m = patch.astype(np.float32) / 1000.0
        elif encoding == '32FC1':
            patch_m = patch.astype(np.float32)
        else:
            return None, (depth_cx, depth_cy), (x1, y1, x2, y2)

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid > self.valid_min_depth_m) & (valid < self.valid_max_depth_m)]

        if valid.size == 0:
            return None, (depth_cx, depth_cy), (x1, y1, x2, y2)

        depth_m = float(np.percentile(valid, 20))
        return depth_m, (depth_cx, depth_cy), (x1, y1, x2, y2)

    # ============================================================
    # 球检测
    # ============================================================
    def detect_stage2_avoid_rgb_candidates(self, frame: np.ndarray) -> List[Dict]:
        """
        STAGE2_CRUISE_YELLOW_ONLY 专用 RGB-only 小球候选。

        这一份检测故意不读取 Depth。它只复用当前蓝球/橙球 HSV、形态学和
        min_contour_area 参数，用于第一次 90° 后的左侧小球规避。

        真正避让条件仍在 choose_stage2_left_danger_ball() 中判断：
          1) 球中心在图像中心左侧；
          2) image_center_x - cx <= stage2_left_ball_avoid_center_px；
          3) area >= stage2_left_ball_avoid_min_area；
          4) radius >= stage2_left_ball_avoid_min_radius。
        """
        h, w = frame.shape[:2]
        image_center_x = w // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        configs = (
            (
                'orange',
                self.orange_h_min, self.orange_h_max,
                self.orange_s_min, self.orange_s_max,
                self.orange_v_min, self.orange_v_max,
                self.orange_min_contour_area,
            ),
            (
                'blue',
                self.blue_h_min, self.blue_h_max,
                self.blue_s_min, self.blue_s_max,
                self.blue_v_min, self.blue_v_max,
                self.blue_min_contour_area,
            ),
        )

        candidates: List[Dict] = []

        for (
                color_name,
                h_min, h_max,
                s_min, s_max,
                v_min, v_max,
                min_contour_area,
        ) in configs:
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours = find_contours(mask)

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < float(min_contour_area):
                    continue

                (cx_f, cy_f), r_circle = cv2.minEnclosingCircle(cnt)
                cx = int(cx_f)
                cy = int(cy_f)
                r_circle = float(r_circle)
                r_eq = math.sqrt(area / math.pi)
                radius = min(r_circle, r_eq)

                candidates.append({
                    'color': color_name,
                    'center': (cx, cy),
                    'radius': float(radius),
                    'radius_circle': r_circle,
                    'radius_eq': r_eq,
                    'area': area,
                    # 明确标记：这一份候选没有使用深度。
                    'depth_m': None,
                    'error_x': int(cx - image_center_x),
                    'side': 'left' if cx < image_center_x else 'right',
                    'rgb_only': True,
                })

        return candidates

    def detect_color_ball_candidates(
            self,
            frame: np.ndarray,
            h_min: int, h_max: int,
            s_min: int, s_max: int,
            v_min: int, v_max: int,
            min_contour_area: float,
            color_name: str
    ) -> List[Dict]:
        h, w = frame.shape[:2]
        self.rgb_w = w
        self.rgb_h = h

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours = find_contours(mask)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue

            # 圆心仍然用最小外接圆
            (cx_f, cy_f), r_circle = cv2.minEnclosingCircle(cnt)
            cx = int(cx_f)
            cy = int(cy_f)
            r_circle = float(r_circle)

            # 面积等效半径
            r_eq = math.sqrt(area / math.pi)

            # 最终半径：只信较小的那个
            radius = min(r_circle, r_eq)

            depth_m, depth_center, depth_box = self.get_depth_for_rgb_point(cx, cy)
            if depth_m is None:
                continue

            image_center_x = w // 2
            error_x = cx - image_center_x
            side = 'left' if cx < image_center_x else 'right'

            candidates.append({
                'color': color_name,
                'center': (cx, cy),
                'radius': radius,
                'radius_circle': r_circle,
                'radius_eq': r_eq,
                'area': float(area),
                'depth_m': depth_m,
                'error_x': int(error_x),
                'depth_center': depth_center,
                'depth_box': depth_box,
                'side': side,
            })

        return candidates

    def choose_side_reference_ball(self, balls: List[Dict]) -> Optional[Dict]:
        if len(balls) == 0:
            return None
        return min(balls, key=lambda b: b['depth_m'])

    def choose_best_target_orange_ball(self, orange_balls: List[Dict]) -> Optional[Dict]:
        if len(orange_balls) == 0:
            return None
        if self.prefer_nearest_ball:
            return min(orange_balls, key=lambda b: b['depth_m'])
        return min(orange_balls, key=lambda b: b['depth_m'] + 0.002 * abs(b['error_x']))

    def detect_ball_scene(self, frame: np.ndarray) -> Dict:
        h, w = frame.shape[:2]
        self.rgb_w = w
        self.rgb_h = h

        orange_balls = self.detect_color_ball_candidates(
            frame,
            self.orange_h_min, self.orange_h_max,
            self.orange_s_min, self.orange_s_max,
            self.orange_v_min, self.orange_v_max,
            self.orange_min_contour_area,
            'orange'
        )

        blue_balls = self.detect_color_ball_candidates(
            frame,
            self.blue_h_min, self.blue_h_max,
            self.blue_s_min, self.blue_s_max,
            self.blue_v_min, self.blue_v_max,
            self.blue_min_contour_area,
            'blue'
        )

        # 第一次 90° 后避让使用独立 RGB-only 候选。
        # 即使 Depth 没值，这里依然能够识别并触发横移。
        stage2_avoid_rgb_balls = self.detect_stage2_avoid_rgb_candidates(frame)

        all_balls = orange_balls + blue_balls
        image_center_x = w // 2

        left_balls = [b for b in all_balls if b['center'][0] < image_center_x]
        right_balls = [b for b in all_balls if b['center'][0] >= image_center_x]

        left_ref = self.choose_side_reference_ball(left_balls)
        right_ref = self.choose_side_reference_ball(right_balls)

        has_center_reference = (left_ref is not None and right_ref is not None)
        center_error_px = None
        if has_center_reference:
            left_cx = left_ref['center'][0]
            right_cx = right_ref['center'][0]
            lane_mid_x = 0.5 * (left_cx + right_cx)
            center_error_px = lane_mid_x - image_center_x

        best_target_ball = self.choose_best_target_orange_ball(orange_balls)

        if best_target_ball is None:
            return {
                'has_ball': False,
                'ball_center': None,
                'ball_radius': None,
                'ball_depth_m': None,
                'img_shape': (h, w),
                'error_x': None,
                'aligned': False,
                'depth_center': None,
                'depth_box': None,

                'orange_balls': orange_balls,
                'blue_balls': blue_balls,
                'stage2_avoid_rgb_balls': stage2_avoid_rgb_balls,
                'left_balls': left_balls,
                'right_balls': right_balls,
                'has_center_reference': has_center_reference,
                'center_error_px': center_error_px,
                'left_ref': left_ref,
                'right_ref': right_ref,
                'best_target_ball': None,
            }

        return {
            'has_ball': True,
            'ball_center': best_target_ball['center'],
            'ball_radius': best_target_ball['radius'],
            'ball_depth_m': best_target_ball['depth_m'],
            'img_shape': (h, w),
            'error_x': best_target_ball['error_x'],
            'aligned': abs(best_target_ball['error_x']) <= self.lateral_align_px_tol,
            'depth_center': best_target_ball['depth_center'],
            'depth_box': best_target_ball['depth_box'],

            'orange_balls': orange_balls,
            'blue_balls': blue_balls,
            'stage2_avoid_rgb_balls': stage2_avoid_rgb_balls,
            'left_balls': left_balls,
            'right_balls': right_balls,
            'has_center_reference': has_center_reference,
            'center_error_px': center_error_px,
            'left_ref': left_ref,
            'right_ref': right_ref,
            'best_target_ball': best_target_ball,
        }

    # ============================================================
    # 黄线检测
    # ============================================================
    def is_front_horizontal_yellow_line(self, cnt, roi_shape) -> bool:
        """
        判断黄色轮廓是否为前方横向停止线。

        改进版：不再使用 minAreaRect / fitLine 的角度作为过滤条件，
        避免同一条横线在 0° 和 90° 之间跳变导致误拒绝。

        只使用更严格的 bbox 条件：
        1. wh_ratio = bbox_width / bbox_height 足够大，必须像横向长条；
        2. width_ratio = bbox_width / roi_width 足够大，必须横跨较大前方区域；
        3. center_offset_ratio 足够小，必须靠近 ROI 中心，避免旁边黄线误判。
        """
        _, roi_w = roi_shape[:2]

        area = cv2.contourArea(cnt)
        if area < self.yellow_min_contour_area:
            return False

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh <= 0:
            return False

        wh_ratio = bw / float(bh)
        if wh_ratio < self.yellow_min_width_height_ratio:
            return False

        width_ratio = bw / float(max(roi_w, 1))
        if width_ratio < self.yellow_min_width_ratio:
            return False

        cx = x + bw / 2.0
        roi_cx = roi_w / 2.0
        center_offset_ratio = abs(cx - roi_cx) / float(max(roi_w, 1))
        if center_offset_ratio > self.yellow_center_tolerance_ratio:
            return False

        return True

    def get_signed_yellow_line_angle_deg(self, cnt) -> float:
        """
        复用原黄线轮廓，估计其相对图像水平线的有符号角度。
        0 度表示基本水平；正负号只用于后面的 wz 矫正。
        """
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

    def detect_yellow_stop_line(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        roi_top = int(h * self.yellow_roi_top_ratio)
        roi_left = int(w * self.yellow_roi_left_ratio)
        roi_right = int(w * self.yellow_roi_right_ratio)

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours = find_contours(mask)

        best_contour = None
        best_score = -1.0

        # 第一赛道和最终黄线阶段放宽黄线形状约束。
        # 第二次90°后的 PRE180 阶段保持和 STAGE2_CRUISE_YELLOW_ONLY 一样，
        # 要求候选是前方横线。
        require_front_horizontal = self.state not in (
            'STAGE1_CRUISE_BALL_AND_YELLOW',
            'STAGE3_GO_FINAL',
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.yellow_min_contour_area:
                continue

            if require_front_horizontal and not self.is_front_horizontal_yellow_line(cnt, roi.shape):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
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
                'require_front_horizontal': bool(require_front_horizontal),
            }

        x, y, bw, bh = cv2.boundingRect(best_contour)
        line_bottom_y = roi_top + y + bh
        cx = roi_left + x + bw // 2
        cy = roi_top + y + bh // 2
        angle_deg = self.get_signed_yellow_line_angle_deg(best_contour)
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
            'bbox': (int(roi_left + x), int(roi_top + y), int(roi_left + x + bw), int(roi_top + y + bh)),
            'width_ratio': float(width_ratio),
            'wh_ratio': float(wh_ratio),
            'require_front_horizontal': bool(require_front_horizontal),
        }

    # ============================================================
    # 状态切换
    # ============================================================
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f'STATE: {self.state} -> {new_state}')
            self.state = new_state

            if new_state in (
                    'STAGE1_CRUISE_BALL_AND_YELLOW',
                    'STAGE2_CRUISE_YELLOW_ONLY',
                    'STAGE3_PRE180_CENTER_YELLOW',
                    'STAGE3_GO_FINAL'
            ):
                self.yellow_stop_counter = 0

            if new_state == 'STAGE2_CRUISE_YELLOW_ONLY':
                self.stage2_cruise_after_first_turn_start_time_sec = None
                self.stage2_left_ball_avoid_counter = 0
                self.stage2_left_ball_avoid_active = False
                self.stage2_danger_ball_seen_once = False

            if new_state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
                self.stage1_yellow_touched_bottom = False
                self.stage1_yellow_disappear_counter = 0

            if new_state == 'STAGE3_PRE180_CENTER_YELLOW':
                # 第二次90°后明确关闭鱼眼撞球入口，并清除之前的鱼眼目标缓存。
                self.latest_fisheye_left_target = None
                self.latest_fisheye_right_target = None
                self.fisheye_left_entry_counter = 0
                self.fisheye_right_entry_counter = 0
                self._pending_fisheye_left_msg = None
                self._pending_fisheye_right_msg = None

            if new_state in (
                    'STAGE1_FORWARD_BEFORE_ROTATE',
            ):
                self.before_turn_forward_start_time_sec = None

            if new_state == 'STAGE2_LEFT_SHIFT_BEFORE_SECOND_90':
                self.stage2_before_second_90_left_shift_start_time_sec = None

            if new_state in (
                    'STAGE1_ROTATE_LEFT_90',
                    'STAGE2_ROTATE_LEFT_90',
                    'STAGE3_ROTATE_BACK_180',
            ):
                self.timed_turn_start_time_sec = None
                self.timed_turn_pre_stand_start_time_sec = None

            if new_state == 'STAGE3_ROTATE_LEFT_30':
                self.stage3_final_left_shift_start_time_sec = None

            if new_state == 'STAGE3_FORWARD_AFTER_GO_FINAL':
                self.stage3_forward_after_go_final_start_time_sec = None

            if new_state == 'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT':
                self.stage3_final_rotate_start_time_sec = None
                self.timed_turn_pre_stand_start_time_sec = None

            if new_state == 'STAGE2_MOVE_FORWARD_AFTER_LEFT_JUMP_TIME':
                self.stage2_forward_after_left_jump_start_time_sec = None

            if new_state == 'BALL_LATERAL_ALIGN':
                self.lateral_align_counter = 0

                # 每次进入对齐球阶段，都重新记录深度变化。
                self.ball_align_last_depth_m = None
                self.ball_align_min_seen_depth_m = None

                # 锁定“开始对齐时”的目标球所在侧。
                # 后面对齐过程中目标球可能因为机器人横移跑到画面另一边，
                # 撞后横移方向仍然使用这里锁定的初始 side，不再在撞击前冲时覆盖。
                target = self.latest_ball_result.get('best_target_ball') if isinstance(self.latest_ball_result, dict) else None

                if target is not None:
                    self.last_hit_side = target.get('side')

                    depth = target.get('depth_m', None)
                    if depth is not None:
                        self.ball_align_last_depth_m = float(depth)
                        self.ball_align_min_seen_depth_m = float(depth)

                    self.get_logger().info(
                        f'BALL_LATERAL_ALIGN lock hit side at align start: '
                        f'last_hit_side={self.last_hit_side}, '
                        f'target_center={target.get("center")}, '
                        f'error_x={target.get("error_x")}, '
                        f'depth={target.get("depth_m")}, '
                        f'radius={target.get("radius")}'
                    )
                else:
                    self.last_hit_side = None
                    self.get_logger().warn(
                        'BALL_LATERAL_ALIGN start but target is None; '
                        'last_hit_side=None, depth cache cleared'
                    )
            if new_state == 'BALL_HIT_CONFIRM_FORWARD':
                self.hit_start_pose = None
                self.hit_start_depth_m = None
                self.hit_start_time_sec = None
                self.fisheye_hit_start_time_sec = None

            if new_state == 'BALL_POST_HIT_SIDE_SHIFT':
                self.post_hit_side_shift_start_pose = None
                self.post_hit_side_shift_start_time_sec = None
                self.side_shift_done = False
                self.fisheye_recover_start_time_sec = None

    # ============================================================
    # 判定
    # ============================================================
    def yellow_reached(self, yellow_result: dict, ratio: float) -> bool:
        if yellow_result['img_shape'] is None or not yellow_result['has_line']:
            self.yellow_stop_counter = 0
            return False

        h, _ = yellow_result['img_shape']
        stop_y_threshold = int(h * ratio)

        if yellow_result['line_bottom_y'] is not None and yellow_result['line_bottom_y'] >= stop_y_threshold:
            self.yellow_stop_counter += 1
        else:
            self.yellow_stop_counter = 0

        self.get_logger().info(
            f'yellow line check: bottom={yellow_result["line_bottom_y"]}, '
            f'threshold={stop_y_threshold}, counter={self.yellow_stop_counter}/{self.yellow_stop_confirm_count}',
            throttle_duration_sec=0.5
        )
        return self.yellow_stop_counter >= self.yellow_stop_confirm_count

    def stage1_yellow_passed(self, yellow_result: dict) -> bool:
        if yellow_result['img_shape'] is None:
            self.stage1_yellow_disappear_counter = 0
            return False

        h, _ = yellow_result['img_shape']
        bottom_threshold = int(h * self.yellow_stop_line_y_ratio_stage1)

        # 还看得到黄线
        if yellow_result['has_line'] and yellow_result['line_bottom_y'] is not None:
            current_bottom = yellow_result['line_bottom_y']

            # 先记录：黄线已经到过图像最底下
            if current_bottom >= bottom_threshold:
                if not self.stage1_yellow_touched_bottom:
                    self.get_logger().info(
                        f'STAGE1 yellow touched bottom: bottom={current_bottom}, '
                        f'threshold={bottom_threshold}'
                    )
                self.stage1_yellow_touched_bottom = True
                self.stage1_yellow_disappear_counter = 0
                return False

            # 如果之前已经到底过，现在又重新低于阈值
            # 就认为近处这条黄线已经过去了
            if self.stage1_yellow_touched_bottom and current_bottom < bottom_threshold:
                self.stage1_yellow_disappear_counter += 1
                self.get_logger().info(
                    f'STAGE1 yellow dropped below threshold after touching bottom: '
                    f'bottom={current_bottom}, threshold={bottom_threshold}, '
                    f'counter={self.stage1_yellow_disappear_counter}/{self.yellow_stop_confirm_count}',
                    throttle_duration_sec=0.2
                )
                return self.stage1_yellow_disappear_counter >= self.yellow_stop_confirm_count

            self.stage1_yellow_disappear_counter = 0
            return False

        # 如果已经到底过，并且现在彻底看不到黄线，也算通过
        if self.stage1_yellow_touched_bottom:
            self.stage1_yellow_disappear_counter += 1
            self.get_logger().info(
                f'STAGE1 yellow disappeared after touching bottom: '
                f'counter={self.stage1_yellow_disappear_counter}/{self.yellow_stop_confirm_count}',
                throttle_duration_sec=0.2
            )
            return self.stage1_yellow_disappear_counter >= self.yellow_stop_confirm_count

        self.stage1_yellow_disappear_counter = 0
        return False

    # ============================================================
    # 固定时间转向：转向前先原地踏步恢复姿态
    # ============================================================
    def execute_timed_turn(self, wz: float, duration_sec: float, next_state: str) -> bool:
        """
        第二赛段定时转向专用版本。

        当前状态一进入定时转向：
        1) 先以 30 Hz 发送 vx=vy=wz=0，持续 timed_turn_pre_stand_duration_sec；
           由于当前状态属于 stage2_level_body_states，send_velocity_command 会自动使用
           pitch=0 和 timed_turn_step_height，让机器狗原地踏步恢复水平机身姿态。
        2) 预恢复完成后才开始计算真正的转向 duration_sec。
        3) 整个预恢复 + 转向期间都属于 timed-turn state，因此 RGB/Depth/鱼眼视觉均关闭。
        """
        now = self.now_sec()

        # A. 转向前原地踏步恢复姿态
        if self.timed_turn_pre_stand_start_time_sec is None:
            self.timed_turn_pre_stand_start_time_sec = now
            self.get_logger().info(
                f'[PRE_TURN_STAND] start: duration='
                f'{self.timed_turn_pre_stand_duration_sec:.2f}s, next_turn_wz={wz:.3f}'
            )

        self.timed_turn_pre_stand_start_time_sec = self.align_motion_timer_start(
            self.timed_turn_pre_stand_start_time_sec, now)
        prep_elapsed = max(
            0.0, now - self.timed_turn_pre_stand_start_time_sec)

        if prep_elapsed < self.timed_turn_pre_stand_duration_sec:
            self.send_velocity_command(
                0.0, 0.0, 0.0,
                step_height=self.timed_turn_step_height
            )
            return True

        # B. 0.5s 恢复完成后，再开始真正的定时转向。
        if self.timed_turn_start_time_sec is None:
            self.timed_turn_start_time_sec = now
            self.get_logger().info(
                f'[PRE_TURN_STAND] done: elapsed={prep_elapsed:.2f}s'
            )
            self.get_logger().info(
                f'[TIMED_TURN] start: wz={wz:.3f}, '
                f'duration={duration_sec:.2f}s, next={next_state}'
            )

        self.timed_turn_start_time_sec = self.align_motion_timer_start(
            self.timed_turn_start_time_sec, now)
        elapsed = max(0.0, now - self.timed_turn_start_time_sec)

        if elapsed >= duration_sec:
            self.get_logger().info(
                f'[TIMED_TURN] done: elapsed={elapsed:.2f}s, next={next_state}'
            )
            self.timed_turn_start_time_sec = None
            self.timed_turn_pre_stand_start_time_sec = None
            self.set_state(next_state)
            return True

        self.send_velocity_command(
            0.0,
            0.0,
            wz,
            step_height=self.timed_turn_step_height
        )
        return True

    # ============================================================
    # 巡航中线
    # ============================================================
    def send_center_cruise_command(self, ball: Dict, vx: float):
        self.send_center_cruise_command_with_wz(ball, vx, 0.0)

    def compute_center_cruise_vy(self, ball: Dict) -> float:
        """
        RGB 中线横移修正。

        1. 只看到左侧参考球：
           认为右侧参考球可能已经跑出画面，向右小幅横移（vy < 0），
           直到重新同时看到左右两侧参考球。

        2. 只看到右侧参考球：
           认为左侧参考球可能已经跑出画面，向左小幅横移（vy > 0），
           直到重新同时看到左右两侧参考球。

        3. 左右参考球都有，但深度差过大：
           暂停使用图像中点，向距离更远的一侧给小幅 far-side bias。

        4. 左右参考球都有且深度差不大：
           使用 center_error_px 做正常固定 vy 中点矫正。

        如果真机横移方向与注释相反，只需要对调单侧分支的正负号。
        """
        self.center_cruise_debug_info = {
            'mode': 'NO_CENTER_REF',
            'left_depth': None,
            'right_depth': None,
            'depth_diff': None,
            'center_error_px': ball.get('center_error_px', None),
            'vy': 0.0,
        }

        left_ref = ball.get('left_ref')
        right_ref = ball.get('right_ref')
        err_px = ball.get('center_error_px', None)

        # 单侧参考球恢复：
        # 只见左球 -> 向右找回右球 -> vy < 0
        # 只见右球 -> 向左找回左球 -> vy > 0
        if left_ref is not None and right_ref is None:
            vy = -abs(self.center_single_side_fixed_vy)
            self.center_cruise_debug_info.update({
                'mode': 'SINGLE_LEFT_RECOVER_RIGHT',
                'left_depth': left_ref.get('depth_m', None),
                'vy': vy,
            })
            self.get_logger().info(
                f'center single-side recover: only_left=Y, only_right=N, '
                f'move=right, vy={vy:.3f}',
                throttle_duration_sec=0.3
            )
            return vy

        if left_ref is None and right_ref is not None:
            vy = abs(self.center_single_side_fixed_vy)
            self.center_cruise_debug_info.update({
                'mode': 'SINGLE_RIGHT_RECOVER_LEFT',
                'right_depth': right_ref.get('depth_m', None),
                'vy': vy,
            })
            self.get_logger().info(
                f'center single-side recover: only_left=N, only_right=Y, '
                f'move=left, vy={vy:.3f}',
                throttle_duration_sec=0.3
            )
            return vy

        # 两侧都没有参考球，无法判断横移方向。
        if left_ref is None and right_ref is None:
            return 0.0

        # 到这里说明左右球都重新出现，自动恢复原有双球居中逻辑。

        left_depth = left_ref.get('depth_m', None)
        right_depth = right_ref.get('depth_m', None)

        self.center_cruise_debug_info.update({
            'left_depth': left_depth,
            'right_depth': right_depth,
            'center_error_px': err_px,
        })

        # 先判断左右参考球深度差。
        # 如果深度差太大，说明这两个球不太适合直接拿来做“中点对齐”。
        if left_depth is not None and right_depth is not None:
            depth_diff = abs(float(left_depth) - float(right_depth))
            self.center_cruise_debug_info['depth_diff'] = depth_diff

            if depth_diff >= self.center_depth_diff_disable_align_m:
                if float(left_depth) > float(right_depth):
                    # 左边球更远：向左边给一个小 vy
                    vy = abs(self.center_far_side_fixed_vy)
                    far_side = 'left'
                else:
                    # 右边球更远：向右边给一个小 vy
                    vy = -abs(self.center_far_side_fixed_vy)
                    far_side = 'right'

                self.center_cruise_debug_info.update({
                    'mode': 'FAR_SIDE_BIAS',
                    'far_side': far_side,
                    'vy': vy,
                })

                self.get_logger().info(
                    f'center far-side bias: left_depth={float(left_depth):.3f}, '
                    f'right_depth={float(right_depth):.3f}, '
                    f'diff={depth_diff:.3f}/{self.center_depth_diff_disable_align_m:.3f}, '
                    f'far_side={far_side}, vy={vy:.3f}',
                    throttle_duration_sec=0.3
                )
                return vy

        # 深度差不大，或者深度无效时，退回原来的图像中线对齐。
        if err_px is None:
            self.center_cruise_debug_info['mode'] = 'NO_CENTER_ERR'
            return 0.0

        err_px = float(err_px)
        self.center_cruise_debug_info['center_error_px'] = err_px

        if abs(err_px) <= self.center_ok_px:
            self.center_cruise_debug_info.update({
                'mode': 'CENTER_OK',
                'vy': 0.0,
            })
            return 0.0

        # 沿用原来 vy 横向平移对齐的方向约定：
        # center_error_px > 0 -> vy < 0；center_error_px < 0 -> vy > 0。
        if err_px > 0.0:
            vy = -abs(self.center_cruise_fixed_vy)
        else:
            vy = abs(self.center_cruise_fixed_vy)

        self.center_cruise_debug_info.update({
            'mode': 'NORMAL_ALIGN',
            'vy': vy,
        })

        self.get_logger().info(
            f'center lateral align fixed: center_error_px={err_px:.1f}, '
            f'deadband={self.center_ok_px:.1f}, vy={vy:.3f}',
            throttle_duration_sec=0.3
        )
        return vy

    def send_center_cruise_command_with_wz(self, ball: Dict, vx: float, wz: float):
        center_vy = self.compute_center_cruise_vy(ball)

        self.get_logger().info(
            f'cruise correction: center_vy={center_vy:.3f}, yellow_wz={wz:.3f}',
            throttle_duration_sec=0.3
        )

        # 中线对齐使用固定 vy 横向平移；黄线角度矫正仍然使用 wz。
        # 两者不冲突，可以同时发送。
        self.send_velocity_command(vx, center_vy, wz)

    def choose_stage2_left_danger_ball(self, ball: Dict) -> Optional[Dict]:
        """
        第二赛段第一次 90° 后的左侧小球避让目标选择（RGB-only）。

        不再使用任何深度条件。

        触发条件：
        1. RGB-only 蓝球/橙球候选通过原有 HSV + min_contour_area；
        2. 球位于图像中心左侧；
        3. image_center_x - cx <= stage2_left_ball_avoid_center_px；
        4. area >= stage2_left_ball_avoid_min_area；
        5. radius >= stage2_left_ball_avoid_min_radius。

        满足后固定向右规避。
        """
        if not self.stage2_left_ball_avoid_enabled:
            return None
        if ball is None or ball.get('img_shape') is None:
            return None

        _, w = ball['img_shape']
        image_center_x = w / 2.0

        candidates = []
        for b in ball.get('stage2_avoid_rgb_balls', []):
            center = b.get('center')
            area = float(b.get('area', 0.0))
            radius = float(b.get('radius', 0.0))
            if center is None:
                continue

            cx = float(center[0])
            if cx >= image_center_x:
                continue

            dist_to_center_px = image_center_x - cx
            if dist_to_center_px > self.stage2_left_ball_avoid_center_px:
                continue

            if area < self.stage2_left_ball_avoid_min_area:
                continue

            if radius < self.stage2_left_ball_avoid_min_radius:
                continue

            item = dict(b)
            item['stage2_avoid_dist_to_center_px'] = float(dist_to_center_px)
            candidates.append(item)

        self.stage2_left_ball_avoid_debug['candidate_count'] = len(candidates)

        if not candidates:
            return None

        # 不再有深度可用于“最近”排序。
        # 优先选择最靠近机器狗前进中线的球；距离相同时优先半径更大的球。
        return min(
            candidates,
            key=lambda b: (
                float(b.get('stage2_avoid_dist_to_center_px', 9999.0)),
                -float(b.get('radius', 0.0)),
            )
        )

    def compute_stage2_left_ball_avoid_vy(self, ball: Dict) -> float:
        """
        STAGE2_CRUISE_YELLOW_ONLY 专用：左侧蓝球/橙球靠近路线时，固定向右偏移。

        当前代码约定：vy < 0 通常表示向右移动；如果实测方向反了，
        只需要把下面 return 的 -abs(...) 改成 +abs(...)，或者把参数值改负后自行扩展。
        """
        debug = {
            'enabled': self.stage2_left_ball_avoid_enabled,
            'active': False,
            'counter': self.stage2_left_ball_avoid_counter,
            'danger_ball': None,
            'candidate_count': 0,
            'vy': 0.0,
            'reason': 'disabled' if not self.stage2_left_ball_avoid_enabled else 'no_danger_ball',
        }
        self.stage2_left_ball_avoid_debug = debug

        if not self.stage2_left_ball_avoid_enabled:
            self.stage2_left_ball_avoid_counter = 0
            self.stage2_left_ball_avoid_active = False
            return 0.0

        danger = self.choose_stage2_left_danger_ball(ball)
        debug['candidate_count'] = self.stage2_left_ball_avoid_debug.get('candidate_count', 0)

        if danger is None:
            self.stage2_left_ball_avoid_counter = 0
            self.stage2_left_ball_avoid_active = False
            debug.update({
                'active': False,
                'counter': 0,
                'danger_ball': None,
                'vy': 0.0,
                'reason': 'no_danger_ball',
            })
            self.stage2_left_ball_avoid_debug = debug
            return 0.0

        self.stage2_left_ball_avoid_counter += 1
        confirm_frames = max(1, int(self.stage2_left_ball_avoid_confirm_frames))
        active = self.stage2_left_ball_avoid_counter >= confirm_frames
        self.stage2_left_ball_avoid_active = active

        vy = -abs(self.stage2_left_ball_avoid_vy) if active else 0.0
        debug.update({
            'active': active,
            'counter': self.stage2_left_ball_avoid_counter,
            'danger_ball': danger,
            'vy': vy,
            'reason': 'avoid_right' if active else 'confirming',
        })
        self.stage2_left_ball_avoid_debug = debug

        self.get_logger().info(
            f'[STAGE2_LEFT_BALL_AVOID] RGB_ONLY danger={danger.get("color")} '
            f'center={danger.get("center")}, '
            f'area={danger.get("area", 0.0):.0f}/{self.stage2_left_ball_avoid_min_area:.0f}, '
            f'radius={danger.get("radius", 0.0):.1f}/{self.stage2_left_ball_avoid_min_radius:.1f}, '
            f'dist_to_center={danger.get("stage2_avoid_dist_to_center_px", 0.0):.1f}/'
            f'{self.stage2_left_ball_avoid_center_px:.1f}px, '
            f'counter={self.stage2_left_ball_avoid_counter}/{confirm_frames}, '
            f'active={active}, vy={vy:.3f}',
            throttle_duration_sec=0.2
        )
        return vy

    def compute_yellow_angle_align_wz(self, yellow_result: dict) -> float:
        """
        使用原 detect_yellow_stop_line() 的检测结果做角度矫正。
        不改变原来的黄线筛选逻辑，只把 angle_deg 的正负转换成固定 wz。
        """
        if not self.yellow_angle_align_enabled:
            return 0.0
        if yellow_result is None or not yellow_result.get('has_line', False):
            return 0.0

        angle_deg = yellow_result.get('angle_deg', None)
        if angle_deg is None:
            return 0.0

        angle_deg = float(angle_deg)
        if abs(angle_deg) <= self.yellow_angle_align_deadband_deg:
            return 0.0

        # 固定角速度版本：只看 angle_deg 正负，不按角度大小改变速度。
        # 当前符号：黄线角度为正时给负 wz。
        # 如果实测发现越修越歪，把下面 if/else 的正负号对调。
        if angle_deg > 0.0:
            wz = -abs(self.yellow_angle_align_fixed_wz)
        else:
            wz = abs(self.yellow_angle_align_fixed_wz)

        self.get_logger().info(
            f'yellow angle align fixed: angle={angle_deg:.2f}deg, '
            f'deadband={self.yellow_angle_align_deadband_deg:.2f}deg, '
            f'wz={wz:.3f}, '
            f'require_front_horizontal={yellow_result.get("require_front_horizontal")}',
            throttle_duration_sec=0.3
        )
        return wz

    def get_yellow_slowdown_speed(
            self,
            yellow_result: dict,
            normal_speed: float,
            slow_speed: float,
            slowdown_ratio: float
    ) -> float:
        if yellow_result['img_shape'] is None or not yellow_result['has_line']:
            return normal_speed

        h, _ = yellow_result['img_shape']
        slow_threshold = int(h * slowdown_ratio)
        bottom = yellow_result['line_bottom_y']

        if bottom is not None and bottom >= slow_threshold:
            return min(normal_speed, slow_speed)
        return normal_speed

    # ============================================================
    # 可视化调试窗口
    # ============================================================
    def _make_yellow_mask_for_debug(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        roi_top = int(h * self.yellow_roi_top_ratio)
        roi_left = int(w * self.yellow_roi_left_ratio)
        roi_right = int(w * self.yellow_roi_right_ratio)

        roi_top = max(0, min(h - 1, roi_top))
        roi_left = max(0, min(w - 1, roi_left))
        roi_right = max(roi_left + 1, min(w, roi_right))

        roi = frame[roi_top:h, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, (roi_left, roi_top, roi_right, h)

    def get_current_yellow_ratio_for_debug(self):
        if self.state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
            return self.yellow_stop_line_y_ratio_stage1
        if self.state == 'STAGE2_CRUISE_YELLOW_ONLY':
            return self.yellow_stop_line_y_ratio_stage2
        if self.state == 'STAGE3_PRE180_CENTER_YELLOW':
            return self.yellow_stop_line_y_ratio_stage3
        if self.state == 'STAGE3_GO_FINAL':
            return self.yellow_ratio_final
        return None

    def show_debug_window(self, frame: np.ndarray):
        """
        第二赛段调试窗口。
        只显示当前识别结果，不改变状态机逻辑。
        """
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            image_center_x = w // 2
            image_center_y = h // 2

            ball = self.latest_ball_result
            yellow = self.latest_yellow_result

            # 画图像中心线
            cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 255, 255), 1)
            cv2.line(vis, (0, image_center_y), (w - 1, image_center_y), (80, 80, 80), 1)

            cv2.putText(vis, f'state={self.state}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            # 画黄色 ROI
            roi_top = int(h * self.yellow_roi_top_ratio)
            roi_left = int(w * self.yellow_roi_left_ratio)
            roi_right = int(w * self.yellow_roi_right_ratio)
            roi_top = max(0, min(h - 1, roi_top))
            roi_left = max(0, min(w - 1, roi_left))
            roi_right = max(roi_left + 1, min(w, roi_right))
            cv2.rectangle(vis, (roi_left, roi_top), (roi_right, h - 1), (0, 255, 255), 1)
            cv2.putText(vis, 'yellow ROI', (roi_left + 3, max(18, roi_top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)

            # 当前状态的黄线触发阈值线
            ratio = self.get_current_yellow_ratio_for_debug()
            if ratio is not None:
                threshold_y = int(h * ratio)
                cv2.line(vis, (0, threshold_y), (w - 1, threshold_y), (0, 180, 255), 2)
                cv2.putText(
                    vis,
                    f'th={threshold_y} ratio={ratio:.2f}',
                    (10, max(78, threshold_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 180, 255),
                    2
                )

            # 画所有蓝球/橙球候选
            for color_name, balls, draw_color in (
                    ('B', ball.get('blue_balls', []), (255, 0, 0)),
                    ('O', ball.get('orange_balls', []), (0, 140, 255)),
            ):
                for idx, b in enumerate(balls):
                    cx, cy = b['center']
                    radius = int(max(2, round(b.get('radius', 2))))
                    depth_m = b.get('depth_m')
                    error_x = b.get('error_x')
                    cv2.circle(vis, (cx, cy), radius, draw_color, 2)
                    cv2.circle(vis, (cx, cy), 4, draw_color, -1)
                    depth_text = 'None' if depth_m is None else f'{depth_m:.2f}m'
                    cv2.putText(
                        vis,
                        f'{color_name}{idx} r={b.get("radius", 0):.1f} d={depth_text} ex={error_x}',
                        (max(5, cx - 45), max(18, cy - radius - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        draw_color,
                        2
                    )

                    # 深度取样窗口映射到 RGB 上的大致位置
                    box = b.get('depth_box')
                    if box is not None and self.latest_depth is not None:
                        dh, dw = self.latest_depth.shape[:2]
                        x1, y1, x2, y2 = box
                        rx1 = int(x1 * w / max(dw, 1))
                        rx2 = int(x2 * w / max(dw, 1))
                        ry1 = int(y1 * h / max(dh, 1))
                        ry2 = int(y2 * h / max(dh, 1))
                        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), draw_color, 1)

            # 画左右参考球和中线
            left_ref = ball.get('left_ref')
            right_ref = ball.get('right_ref')
            if left_ref is not None:
                cx, cy = left_ref['center']
                cv2.circle(vis, (cx, cy), 8, (255, 255, 0), 3)
                cv2.putText(vis, 'LEFT_REF', (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 2)
            if right_ref is not None:
                cx, cy = right_ref['center']
                cv2.circle(vis, (cx, cy), 8, (255, 255, 0), 3)
                cv2.putText(vis, 'RIGHT_REF', (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 2)

            if left_ref is not None and right_ref is not None:
                lx = left_ref['center'][0]
                rx = right_ref['center'][0]
                lane_mid_x = int(0.5 * (lx + rx))
                cv2.line(vis, (lane_mid_x, 0), (lane_mid_x, h - 1), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f'lane_mid={lane_mid_x}, center_err={ball.get("center_error_px")}',
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    2
                )

            # 画当前最佳目标球
            target = ball.get('best_target_ball')
            if target is not None:
                cx, cy = target['center']
                radius = int(max(8, round(target.get('radius', 8))))
                cv2.circle(vis, (cx, cy), radius + 4, (0, 0, 255), 3)
                cv2.putText(
                    vis,
                    f'TARGET {target.get("color")} side={target.get("side")} d={target.get("depth_m", -1):.2f}',
                    (max(5, cx - 70), min(h - 10, cy + radius + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255),
                    2
                )

            # 第二赛段左侧近球避让可视化
            avoid_debug = getattr(self, 'stage2_left_ball_avoid_debug', {})
            danger = avoid_debug.get('danger_ball')
            if danger is not None and danger.get('center') is not None:
                cx, cy = danger['center']
                radius = int(max(8, round(float(danger.get('radius', 8)))))
                color = (0, 0, 255) if avoid_debug.get('active') else (0, 180, 255)
                cv2.circle(vis, (int(cx), int(cy)), radius + 10, color, 3)
                cv2.putText(
                    vis,
                    f'S2_AVOID {danger.get("color")} active={avoid_debug.get("active")} vy={avoid_debug.get("vy", 0.0):.2f}',
                    (max(5, int(cx) - 90), min(h - 10, int(cy) + radius + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2
                )

            if self.state == 'STAGE2_CRUISE_YELLOW_ONLY':
                cv2.putText(
                    vis,
                    f'S2 left-ball avoid: {avoid_debug.get("reason", "none")} '
                    f'cnt={avoid_debug.get("counter", 0)} vy={avoid_debug.get("vy", 0.0):.2f}',
                    (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255) if avoid_debug.get('active') else (0, 180, 255),
                    2
                )

            # 画黄线检测结果：底部线、中心点、bbox、角度
            if yellow.get('has_line') and yellow.get('line_bottom_y') is not None:
                bottom_y = int(yellow['line_bottom_y'])
                line_center = yellow.get('line_center')
                cv2.line(vis, (0, bottom_y), (w - 1, bottom_y), (0, 255, 255), 2)

                bbox = yellow.get('bbox')
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if line_center is not None:
                    cx, cy = line_center
                    cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
                    angle = yellow.get('angle_deg')
                    angle_text = 'None' if angle is None else f'{float(angle):.1f}deg'
                    cv2.putText(
                        vis,
                        f'YELLOW bottom={bottom_y} angle={angle_text}',
                        (max(5, cx - 100), max(18, cy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 255),
                        2
                    )

                # 画拟合角度方向线，方便看矫正方向
                angle = yellow.get('angle_deg')
                if line_center is not None and angle is not None:
                    cx, cy = line_center
                    length = 80
                    rad = math.radians(float(angle))
                    dx = int(math.cos(rad) * length)
                    dy = int(math.sin(rad) * length)
                    cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)

            else:
                cv2.putText(vis, 'YELLOW not detected', (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # 画当前修正量信息
            yellow_wz = self.compute_yellow_angle_align_wz(yellow)
            center_vy = self.compute_center_cruise_vy(ball)
            center_dbg = getattr(self, 'center_cruise_debug_info', {})
            mode = center_dbg.get('mode', 'NA')
            ld = center_dbg.get('left_depth')
            rd = center_dbg.get('right_depth')
            dd = center_dbg.get('depth_diff')
            ld_txt = 'None' if ld is None else f'{float(ld):.2f}'
            rd_txt = 'None' if rd is None else f'{float(rd):.2f}'
            dd_txt = 'None' if dd is None else f'{float(dd):.2f}'
            cv2.putText(
                vis,
                f'center_vy={center_vy:.2f} mode={mode} Ld={ld_txt} Rd={rd_txt} diff={dd_txt} yellow_wz={yellow_wz:.2f}',
                (10, h - 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f'orange_cnt={len(ball.get("orange_balls", []))} blue_cnt={len(ball.get("blue_balls", []))} '
                f'center_ref={ball.get("has_center_reference")}',
                (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f'yellow_has={yellow.get("has_line")} bottom={yellow.get("line_bottom_y")} '
                f'angle={yellow.get("angle_deg")} require_front={yellow.get("require_front_horizontal")}',
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2
            )

            cv2.imshow('second_stage_orange_yellow_debug', vis)

            if self.show_yellow_mask:
                mask, _ = self._make_yellow_mask_for_debug(frame)
                cv2.imshow('second_stage_yellow_mask', mask)

            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'show_debug_window failed: {e}', throttle_duration_sec=1.0)

    # ============================================================
    # 球子链
    # ============================================================
    def ball_align_should_go_hit(self, target: Optional[Dict]) -> bool:
        """
        BALL_LATERAL_ALIGN 阶段保护逻辑。

        目的：
        对齐 A 球时，如果 A 球太近导致识别不到，
        或者 A 球丢失后 best_target_ball 突然变成远处 B 球，
        不继续对齐 B 球，而是直接进入 BALL_HIT_CONFIRM_FORWARD。

        触发条件：
        1. target is None：
        认为球已经太近 / 进入盲区 / 穿模，直接撞击。

        2. 当前 target 深度比上一帧或历史最近深度突然变大：
        认为当前识别到的不是原来的近处球，而是远处其他球，直接撞击。
        """
        if target is None:
            if self.ball_align_lost_go_hit:
                self.get_logger().warn(
                    '[BALL_ALIGN_PROTECT] target=None during BALL_LATERAL_ALIGN, '
                    'assume ball is too close/lost, go BALL_HIT_CONFIRM_FORWARD'
                )
                return True

            return False

        if not self.ball_align_depth_jump_enabled:
            return False

        cur_depth = target.get('depth_m', None)

        if cur_depth is None:
            self.get_logger().warn(
                '[BALL_ALIGN_PROTECT] target depth=None during BALL_LATERAL_ALIGN, '
                'assume ball is too close/lost, go BALL_HIT_CONFIRM_FORWARD'
            )
            return True

        cur_depth = float(cur_depth)

        # 初始化历史深度
        if self.ball_align_last_depth_m is None:
            self.ball_align_last_depth_m = cur_depth

        if self.ball_align_min_seen_depth_m is None:
            self.ball_align_min_seen_depth_m = cur_depth
        else:
            self.ball_align_min_seen_depth_m = min(self.ball_align_min_seen_depth_m, cur_depth)

        last_depth = float(self.ball_align_last_depth_m)
        min_seen = float(self.ball_align_min_seen_depth_m)

        jump_from_last = cur_depth - last_depth
        jump_from_min = cur_depth - min_seen

        near_enough_before = min_seen <= self.ball_align_near_depth_for_jump_m

        depth_jump = (
            jump_from_last >= self.ball_align_depth_jump_threshold_m
            or jump_from_min >= self.ball_align_depth_jump_threshold_m
        )

        if near_enough_before and depth_jump:
            self.get_logger().warn(
                f'[BALL_ALIGN_PROTECT] target depth suddenly increased, '
                f'treat current target as another far ball and go hit: '
                f'cur={cur_depth:.3f}, last={last_depth:.3f}, min_seen={min_seen:.3f}, '
                f'jump_last={jump_from_last:.3f}, jump_min={jump_from_min:.3f}, '
                f'jump_th={self.ball_align_depth_jump_threshold_m:.3f}, '
                f'near_th={self.ball_align_near_depth_for_jump_m:.3f}, '
                f'center={target.get("center")}, side={target.get("side")}, '
                f'error_x={target.get("error_x")}, radius={target.get("radius")}'
            )
            return True

        self.ball_align_last_depth_m = cur_depth

        self.get_logger().info(
            f'[BALL_ALIGN_PROTECT] normal target depth: '
            f'cur={cur_depth:.3f}, last={last_depth:.3f}, min_seen={min_seen:.3f}, '
            f'center={target.get("center")}, side={target.get("side")}',
            throttle_duration_sec=0.3
        )

        return False

    def finish_ball_task_and_return(self, x: float, y: float, yaw: float):
        self.last_ball_done_time_sec = self.now_sec()
        self.last_ball_done_pose = (x, y, yaw)
        self.mark_fisheye_side_hit(
            return_state=self.ball_return_state,
            side=self.fisheye_target_side
        )
        self.get_logger().info(
            f'Ball task finished | '
            f'last_ball_done_time_sec={self.last_ball_done_time_sec:.2f} | '
            f'last_ball_done_pose=({x:.3f}, {y:.3f}, {yaw:.3f})'
        )
        self.set_state(self.ball_return_state)

    def handle_ball_subchain(self, x: float, y: float, yaw: float) -> bool:
        """双鱼眼侧撞子链；原巡航 RGB+Depth 居中逻辑保持不变。"""
        target = self.get_locked_fisheye_target()

        # 1) 向目标侧靠近，并根据鱼眼横坐标做前后微调。
        if self.state == 'BALL_LATERAL_ALIGN':
            if target is None:
                self.fisheye_approach_lost_counter += 1
                if self.fisheye_approach_lost_counter >= self.fisheye_approach_lost_frames:
                    self.get_logger().warning('fisheye target lost; return to cruise')
                    return_state = self.ball_return_state
                    self.reset_fisheye_hit_context(clear_entry=True)
                    self.set_state(return_state)
                    return True
            else:
                self.fisheye_approach_lost_counter = 0
                if target['radius'] >= self.fisheye_hit_radius:
                    self.fisheye_hit_radius_counter += 1
                else:
                    self.fisheye_hit_radius_counter = 0
                if self.fisheye_hit_radius_counter >= self.fisheye_hit_radius_confirm_frames:
                    self.set_state('BALL_HIT_CONFIRM_FORWARD')
                    return True

            vx = self.compute_fisheye_approach_vx(target)
            vy = self.fisheye_side_sign() * self.fisheye_approach_vy
            self.send_velocity_command(vx, vy, 0.0)
            return True

        # 2) 侧向快速撞击。
        if self.state == 'BALL_HIT_CONFIRM_FORWARD':
            now_sec = self.now_sec()
            if self.fisheye_hit_start_time_sec is None:
                self.fisheye_hit_start_time_sec = now_sec
                self.get_logger().info(
                    f'fisheye lateral hit start: side={self.fisheye_target_side}, '
                    f'duration={self.fisheye_hit_duration_sec:.2f}s'
                )
            self.fisheye_hit_start_time_sec = self.align_motion_timer_start(
                self.fisheye_hit_start_time_sec, now_sec)
            elapsed = max(0.0, now_sec - self.fisheye_hit_start_time_sec)
            if elapsed >= self.fisheye_hit_duration_sec:
                self.set_state('BALL_POST_HIT_SIDE_SHIFT')
                return True
            hit_vy = self.fisheye_side_sign() * self.fisheye_hit_vy
            self.send_velocity_command(0.0, hit_vy, 0.0)
            return True

        # 3) 撞后向相反方向横移，同时向前，持续 3 秒。
        if self.state == 'BALL_POST_HIT_SIDE_SHIFT':
            now_sec = self.now_sec()
            if self.fisheye_recover_start_time_sec is None:
                self.fisheye_recover_start_time_sec = now_sec
                self.get_logger().info(
                    f'fisheye recover start: duration={self.fisheye_recover_duration_sec:.2f}s'
                )
            self.fisheye_recover_start_time_sec = self.align_motion_timer_start(
                self.fisheye_recover_start_time_sec, now_sec)
            elapsed = max(0.0, now_sec - self.fisheye_recover_start_time_sec)
            if elapsed >= self.fisheye_recover_duration_sec:
                self.finish_ball_task_and_return(x, y, yaw)
                self.reset_fisheye_hit_context(clear_entry=True)
                return True
            recover_vy = -self.fisheye_side_sign() * self.fisheye_recover_vy
            self.send_velocity_command(
                self.fisheye_recover_forward_vx, recover_vy, 0.0
            )
            return True

        return False


    def execute_before_turn_forward(
            self,
            next_state: str,
            forward_vx: float,
            duration_s: float
    ):
        """在 90° 转向前保持前倾并按该段独立参数固定向前。"""
        now_sec = self.now_sec()

        if self.before_turn_forward_start_time_sec is None:
            self.before_turn_forward_start_time_sec = now_sec
            self.get_logger().info(
                f'{self.state} start: sim_time_start={now_sec:.3f}s, '
                f'duration={duration_s:.3f}s, '
                f'vx={forward_vx:.3f}, '
                f'pitch={self.stage2_body_pitch:.3f}, next_state={next_state}'
            )

        self.before_turn_forward_start_time_sec = self.align_motion_timer_start(
            self.before_turn_forward_start_time_sec, now_sec)
        elapsed = max(0.0, now_sec - self.before_turn_forward_start_time_sec)
        self.get_logger().info(
            f'{self.state}: elapsed='
            f'{elapsed:.3f}/{duration_s:.3f}s, '
            f'vx={forward_vx:.3f}, '
            f'pitch={self.stage2_body_pitch:.3f}',
            throttle_duration_sec=0.2
        )

        if elapsed >= duration_s:
            # 不插入 STOP，直接切到原来的定时转向状态。
            # 下一控制周期由转向状态发送 wz，并自动使用 pitch=0。
            self.set_state(next_state)
            return

        self.send_velocity_command(
            forward_vx,
            0.0,
            0.0,
            pitch=self.stage2_body_pitch
        )

    def execute_stage3_forward_after_go_final(self):
        """最终黄线后保持前倾并按仿真时间固定向前。"""
        now_sec = self.now_sec()

        if self.stage3_forward_after_go_final_start_time_sec is None:
            self.stage3_forward_after_go_final_start_time_sec = now_sec
            self.get_logger().info(
                f'STAGE3_FORWARD_AFTER_GO_FINAL start: '
                f'sim_time_start={now_sec:.3f}s, '
                f'duration={self.stage3_forward_after_go_final_duration_s:.3f}s, '
                f'vx={self.stage3_forward_after_go_final_vx:.3f}, '
                f'pitch={self.stage2_body_pitch:.3f}'
            )

        self.stage3_forward_after_go_final_start_time_sec = self.align_motion_timer_start(
            self.stage3_forward_after_go_final_start_time_sec, now_sec)
        elapsed = max(0.0, now_sec - self.stage3_forward_after_go_final_start_time_sec)
        self.get_logger().info(
            f'STAGE3_FORWARD_AFTER_GO_FINAL: elapsed='
            f'{elapsed:.3f}/{self.stage3_forward_after_go_final_duration_s:.3f}s, '
            f'vx={self.stage3_forward_after_go_final_vx:.3f}, '
            f'pitch={self.stage2_body_pitch:.3f}',
            throttle_duration_sec=0.2
        )

        if elapsed >= self.stage3_forward_after_go_final_duration_s:
            # 不插入 STOP，直接进入原来的最终横移状态。
            self.set_state('STAGE3_ROTATE_LEFT_30')
            return

        self.send_velocity_command(
            self.stage3_forward_after_go_final_vx,
            0.0,
            0.0,
            pitch=self.stage2_body_pitch
        )

    # ============================================================
    # 主循环（原 control_loop 的第二赛段部分；P1/P3 分发已拆到各自节点）
    # ============================================================
    def stage_control_loop(self):
        # StageNodeBase 的定时器以 30 Hz 唤醒本函数：
        # 普通状态只放行 5 Hz；固定时间转向状态每一拍都放行，即 30 Hz。
        if not self.stage2_control_tick_due():
            return

        timed_turn_state = self.is_stage2_timed_turn_state()
        if not timed_turn_state:
            # 先处理这一拍到来前缓存的最新视觉，再用结果做状态机判断。
            # 普通状态因此真正做到控制 5 Hz + 视觉最多 5 Hz。
            self.process_stage2_latest_vision()
            if self.state in (
                    'STAGE1_CRUISE_BALL_AND_YELLOW',
                    'STAGE3_GO_FINAL'):
                self.update_fisheye_entry_counters()
            else:
                self.fisheye_left_entry_counter = 0
                self.fisheye_right_entry_counter = 0

        pose = self.get_current_pose()

        # TF 现在不是状态机运行的必要条件。
        # 可用时记录用于日志/兼容；不可用时使用上一次位姿或 0 值占位，继续按图像和仿真时间运行。
        if pose is not None:
            self.last_known_pose = pose
            x, y, yaw = pose
        elif self.last_known_pose is not None:
            x, y, yaw = self.last_known_pose
            self.get_logger().warn(
                'TF pose unavailable, use last_known_pose and continue sim-time/image control.',
                throttle_duration_sec=1.0
            )
        else:
            x, y, yaw = 0.0, 0.0, 0.0
            self.get_logger().warn(
                'TF pose unavailable and no last_known_pose, use zero pose and continue sim-time/image control.',
                throttle_duration_sec=1.0
            )

        ball = self.latest_ball_result
        yellow = self.latest_yellow_result

        # RGB 依赖状态必须基于持续到来的新图像运行。若相机/回调卡住，
        # 保持 gait 心跳但把速度清零（原地踏步），直到新 RGB 恢复。
        if (not timed_turn_state) and self.state in self.stage2_rgb_required_states:
            rgb_age = self.rgb_age_s()
            if rgb_age is None or rgb_age > self.stage2_rgb_stale_timeout_s:
                age_text = 'none' if rgb_age is None else f'{rgb_age:.3f}s'
                self.get_logger().warning(
                    f'[P2_RGB_STALE] state={self.state}, rgb_age={age_text}, '
                    f'timeout={self.stage2_rgb_stale_timeout_s:.3f}s -> zero velocity',
                    throttle_duration_sec=0.5
                )
                self.send_velocity_command(0.0, 0.0, 0.0)
                return

        if (not timed_turn_state) and ball['img_shape'] is not None:
            self.get_logger().info(
                f"state={self.state} | "
                f"has_ball={ball['has_ball']} | ball_center={ball['ball_center']} | "
                f"ball_depth={ball['ball_depth_m']} | ball_radius={ball['ball_radius']} | "
                f"ball_error_x={ball['error_x']} | has_center_ref={ball['has_center_reference']} | "
                f"center_error_px={ball['center_error_px']} | "
                f"left_ref={'Y' if ball['left_ref'] is not None else 'N'} | "
                f"right_ref={'Y' if ball['right_ref'] is not None else 'N'} | "
                f"orange_cnt={len(ball['orange_balls'])} | blue_cnt={len(ball['blue_balls'])} | "
                f"yellow_has_line={yellow['has_line']} | yellow_bottom={yellow['line_bottom_y']} | "
                f"yellow_angle={yellow.get('angle_deg')}",
                throttle_duration_sec=0.6
            )

        if self.handle_ball_subchain(x, y, yaw):
            return

        if self.state == 'STAGE1_FORWARD_BEFORE_ROTATE':
            self.execute_before_turn_forward(
                next_state='STAGE1_ROTATE_LEFT_90',
                forward_vx=self.stage1_before_turn_forward_vx,
                duration_s=self.stage1_before_turn_forward_duration_s
            )
            return

        if self.state == 'STAGE1_ROTATE_LEFT_90':
            self.execute_timed_turn(
                wz=self.timed_turn_wz_90,
                # 仅第一次 90° 使用 3.95s。
                duration_sec=self.stage1_timed_turn_duration_90_sec,
                next_state='STAGE2_CRUISE_YELLOW_ONLY'
            )
            return

        if self.state == 'STAGE2_ROTATE_LEFT_90':
            self.execute_timed_turn(
                wz=self.timed_turn_wz_90,
                duration_sec=self.timed_turn_duration_90_sec,
                # 第二次90°前的左移已经完成；转向后直接进入第三条直道。
                next_state='STAGE3_PRE180_CENTER_YELLOW'
            )
            return

        if self.state == 'STAGE2_LEFT_SHIFT_BEFORE_SECOND_90':
            now_sec = self.now_sec()

            if self.stage2_before_second_90_left_shift_start_time_sec is None:
                self.stage2_before_second_90_left_shift_start_time_sec = now_sec
                self.get_logger().info(
                    f'[PRE_SECOND_90_LEFT_SHIFT] start: '
                    f'duration={self.stage2_before_second_90_left_shift_duration_sec:.2f}s, '
                    f'vy=+{self.stage2_before_second_90_left_shift_speed:.3f}'
                )

            self.stage2_before_second_90_left_shift_start_time_sec = self.align_motion_timer_start(
                self.stage2_before_second_90_left_shift_start_time_sec, now_sec)
            elapsed = max(
                0.0,
                now_sec - self.stage2_before_second_90_left_shift_start_time_sec
            )

            if elapsed >= self.stage2_before_second_90_left_shift_duration_sec:
                self.get_logger().info(
                    f'[PRE_SECOND_90_LEFT_SHIFT] done: elapsed={elapsed:.2f}s '
                    f'-> STAGE2_ROTATE_LEFT_90'
                )
                self.stage2_before_second_90_left_shift_start_time_sec = None

                # 左移完成后进入第二次90°。
                # execute_timed_turn() 会先原地踏步0.5s，再开始3.85s转向。
                self.set_state('STAGE2_ROTATE_LEFT_90')
                return

            # vy > 0 为向左。
            # 保持第二赛段正常前倾姿态；该固定动作期间视觉关闭。
            self.send_velocity_command(
                0.0,
                self.stage2_before_second_90_left_shift_speed,
                0.0,
                step_height=self.stage2_velocity_step_height
            )
            return

        if self.state == 'STAGE3_ROTATE_BACK_180':
            self.execute_timed_turn(
                wz=self.timed_turn_wz_180,
                duration_sec=self.timed_turn_duration_180_sec,
                next_state='STAGE3_GO_FINAL'
            )
            return

        if self.state == 'STAGE3_FORWARD_AFTER_GO_FINAL':
            self.execute_stage3_forward_after_go_final()
            return

        if self.state == 'STAGE3_ROTATE_LEFT_30':
            # 最终出口前的移动阶段：不再使用 TF 距离判断。
            # 使用 ROS2 节点时钟 now_sec() 计时；启用 use_sim_time 后就是仿真时间。
            now_sec = self.now_sec()
            if self.stage3_final_left_shift_start_time_sec is None:
                self.stage3_final_left_shift_start_time_sec = now_sec
                self.get_logger().info(
                    f'STAGE3_ROTATE_LEFT_30 start time-based shift: '
                    f'sim_time_start={now_sec:.3f}s, '
                    f'duration={self.stage3_final_left_shift_duration_sec:.3f}s, '
                    f'vy={self.stage3_final_left_shift_speed:.3f}'
                )

            self.stage3_final_left_shift_start_time_sec = self.align_motion_timer_start(
                self.stage3_final_left_shift_start_time_sec, now_sec)
            elapsed = max(0.0, now_sec - self.stage3_final_left_shift_start_time_sec)
            self.get_logger().info(
                f'STAGE3_ROTATE_LEFT_30 time shift: '
                f'elapsed={elapsed:.3f}/{self.stage3_final_left_shift_duration_sec:.3f}s, '
                f'vy={self.stage3_final_left_shift_speed:.3f}',
                throttle_duration_sec=0.2
            )

            if elapsed >= self.stage3_final_left_shift_duration_sec:
                # 不在移动和转向之间调用 STOP，避免 mode=12 + Wait_finish 带来的停顿。
                self.set_state('STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT')
                return

            # 默认发送 vy > 0 作为移动命令。
            # 如果实测方向反了，把 abs(...) 改成 -abs(...)。
            self.send_velocity_command(0.0, abs(self.stage3_final_left_shift_speed), 0.0)
            return

        if self.state == 'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT':
            # 最终出口前的固定转向同样先原地踏步 0.5s 恢复水平姿态。
            now_sec = self.now_sec()

            if self.timed_turn_pre_stand_start_time_sec is None:
                self.timed_turn_pre_stand_start_time_sec = now_sec
                self.get_logger().info(
                    f'[PRE_TURN_STAND] final rotate start: '
                    f'duration={self.timed_turn_pre_stand_duration_sec:.2f}s'
                )

            self.timed_turn_pre_stand_start_time_sec = self.align_motion_timer_start(
                self.timed_turn_pre_stand_start_time_sec, now_sec)
            prep_elapsed = max(
                0.0, now_sec - self.timed_turn_pre_stand_start_time_sec)

            if prep_elapsed < self.timed_turn_pre_stand_duration_sec:
                self.send_velocity_command(
                    0.0, 0.0, 0.0,
                    step_height=self.timed_turn_step_height
                )
                return

            # 预恢复结束后才开始统计最终转向的 1.5s。
            if self.stage3_final_rotate_start_time_sec is None:
                self.stage3_final_rotate_start_time_sec = now_sec
                self.get_logger().info(
                    f'[PRE_TURN_STAND] final rotate done: elapsed={prep_elapsed:.2f}s'
                )
                self.get_logger().info(
                    f'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT start time-based rotate: '
                    f'sim_time_start={now_sec:.3f}s, '
                    f'duration={self.stage3_final_rotate_duration_sec:.3f}s, '
                    f'wz={self.stage3_final_rotate_wz:.3f}'
                )

            self.stage3_final_rotate_start_time_sec = self.align_motion_timer_start(
                self.stage3_final_rotate_start_time_sec, now_sec)
            elapsed = max(0.0, now_sec - self.stage3_final_rotate_start_time_sec)
            self.get_logger().info(
                f'STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT time rotate: '
                f'elapsed={elapsed:.3f}/{self.stage3_final_rotate_duration_sec:.3f}s, '
                f'wz={self.stage3_final_rotate_wz:.3f}',
                throttle_duration_sec=0.2
            )

            if elapsed >= self.stage3_final_rotate_duration_sec:
                self.timed_turn_pre_stand_start_time_sec = None
                # 第二赛段结束后直接进入第三赛段入口；不先进入 DONE，避免提前全流程停止。
                self.complete_stage('STAGE3_FINAL_ROTATE_AFTER_LEFT_SHIFT finished')
                return

            # 默认 wz > 0 为左转。
            self.send_velocity_command(0.0, 0.0, abs(self.stage3_final_rotate_wz))
            return

        if self.state == 'STAGE1_CRUISE_BALL_AND_YELLOW':
            # 同一拍同时看到橙球和黄线时，橙球优先。
            # 完整撞击 + 回退结束返回巡航后，再重新处理最新黄线。
            if self.try_start_fisheye_hit(
                    'STAGE1_CRUISE_BALL_AND_YELLOW', (x, y, yaw)):
                return

            if self.stage1_yellow_passed(yellow):
                self.set_state('STAGE1_FORWARD_BEFORE_ROTATE')
                return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage1_cruise_forward_speed,
                self.stage1_yellow_slow_speed,
                self.yellow_slowdown_ratio_stage1
            )
            wz = self.compute_yellow_angle_align_wz(yellow)
            self.send_center_cruise_command_with_wz(ball, vx, wz)
            return

        if self.state == 'STAGE1_MOVE_RIGHT_FIXED_DISTANCE':
            if self.stage1_right_shift_start_pose is None:
                time.sleep(2)
                self.stage1_right_shift_start_pose = (x, y, yaw)
                self.get_logger().info(
                    f"Saved stage1_right_shift_start_pose = ({x:.3f}, {y:.3f}, {yaw:.3f})"
                )

            x0, y0, _ = self.stage1_right_shift_start_pose
            dist = math.hypot(x - x0, y - y0)

            self.get_logger().info(
                f"Stage1 fixed right shift: dist={dist:.3f} / {self.stage1_right_shift_distance_m:.3f}",
                throttle_duration_sec=0.5
            )

            if dist >= self.stage1_right_shift_distance_m - self.stage1_right_shift_tolerance_m:
                self.set_state('STAGE2_CRUISE_YELLOW_ONLY')
                return

            self.send_move_right_command(self.right_speed)
            return

        if self.state == 'STAGE2_CRUISE_YELLOW_ONLY':
            if self.yellow_reached(yellow, self.yellow_stop_line_y_ratio_stage2):
                # 第二次90°之前先执行定时左移。
                self.set_state('STAGE2_LEFT_SHIFT_BEFORE_SECOND_90')
                return

            now_sec = self.now_sec()

            if self.stage2_cruise_after_first_turn_start_time_sec is None:
                self.stage2_cruise_after_first_turn_start_time_sec = now_sec
                self.get_logger().info(
                    f'[STAGE2_AFTER_TURN_CRUISE] start: '
                    f'delay={self.stage2_after_turn_left_bias_delay_sec:.2f}s, '
                    f'vx_before={self.stage2_cruise_forward_speed:.3f}, '
                    f'vx_after={self.stage2_cruise_forward_speed_after_delay:.3f}, '
                    f'left_bias_vy=+{self.stage2_after_turn_left_bias_vy:.3f}'
                )

            self.stage2_cruise_after_first_turn_start_time_sec = self.align_motion_timer_start(
                self.stage2_cruise_after_first_turn_start_time_sec,
                now_sec
            )
            cruise_elapsed = max(
                0.0,
                now_sec - self.stage2_cruise_after_first_turn_start_time_sec
            )

            after_delay = (
                cruise_elapsed >= self.stage2_after_turn_left_bias_delay_sec
            )

            base_vx = (
                self.stage2_cruise_forward_speed_after_delay
                if after_delay
                else self.stage2_cruise_forward_speed
            )

            # 黄线接近时仍保留原本的减速机制。
            vx = self.get_yellow_slowdown_speed(
                yellow,
                base_vx,
                self.stage2_yellow_slow_speed,
                self.yellow_slowdown_ratio_stage2
            )
            wz = self.compute_yellow_angle_align_wz(yellow)

            # RGB-only 小球避障优先级最高。
            avoid_vy = self.compute_stage2_left_ball_avoid_vy(ball)

            if abs(avoid_vy) > 1e-6:
                # 只要真正触发过一次危险小球，就永久锁存。
                # 之后即使小球消失，本阶段也不再给主动左移偏置。
                if not self.stage2_danger_ball_seen_once:
                    self.stage2_danger_ball_seen_once = True
                    self.get_logger().info(
                        '[STAGE2_AFTER_TURN_CRUISE] danger ball seen once -> '
                        'disable left-bias for the rest of this stage'
                    )

                # 当前正在避球：停止前进，只向右横移。
                cmd_vx = 0.0
                cmd_vy = avoid_vy
                mode = 'BALL_AVOID_STOP_FORWARD'

            else:
                cmd_vx = vx

                if after_delay and not self.stage2_danger_ball_seen_once:
                    # 只有 3 秒后、而且整个阶段从未触发过危险小球，
                    # 才允许主动向左小幅偏移。
                    cmd_vy = abs(self.stage2_after_turn_left_bias_vy)
                    mode = 'AFTER_3S_FORWARD_PLUS_LEFT'

                elif after_delay and self.stage2_danger_ball_seen_once:
                    # 曾经避过一次球：后续只向前，不再主动向左。
                    cmd_vy = 0.0
                    mode = 'AFTER_BALL_SEEN_FORWARD_ONLY'

                else:
                    cmd_vy = 0.0
                    mode = 'FIRST_3S_FORWARD_ONLY'

            self.get_logger().info(
                f'[STAGE2_AFTER_TURN_CRUISE] '
                f'elapsed={cruise_elapsed:.2f}s, '
                f'mode={mode}, '
                f'avoid_active={abs(avoid_vy) > 1e-6}, '
                f'danger_seen_once={self.stage2_danger_ball_seen_once}, '
                f'cmd=({cmd_vx:+.3f},{cmd_vy:+.3f},{wz:+.3f})',
                throttle_duration_sec=0.3
            )

            self.send_velocity_command(
                cmd_vx,
                cmd_vy,
                wz
            )
            return

        if self.state == 'STAGE3_PRE180_CENTER_YELLOW':
            # 第二次90°后：不启用鱼眼橙球撞击。
            # RGB 用于左右参考球居中，同时识别前方黄线。
            # 黄线判定完全复用旧 STAGE3_CRUISE_BALL_ONLY 的参数。
            if self.yellow_reached(
                    yellow, self.yellow_stop_line_y_ratio_stage3):
                # 第二次90°前的定时左移已经完成。
                # 这里黄线到达后直接进入180°；180°状态内部仍会先踏步0.5s。
                self.set_state('STAGE3_ROTATE_BACK_180')
                return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage3_cruise_ball_only_speed,
                self.stage3_yellow_slow_speed,
                self.yellow_slowdown_ratio_stage3
            )
            wz = self.compute_yellow_angle_align_wz(yellow)
            self.send_center_cruise_command_with_wz(ball, vx, wz)
            return

        if self.state == 'STAGE3_GO_FINAL':
            # 180° 后开始启用鱼眼橙球。
            # 如果橙球和最终黄线同一拍同时满足，仍然让橙球优先；
            # 撞击和回退完成后返回 STAGE3_GO_FINAL，再继续处理最终黄线。
            if self.try_start_fisheye_hit(
                    'STAGE3_GO_FINAL', (x, y, yaw)):
                return

            if self.yellow_reached(yellow, self.yellow_ratio_final):
                self.set_state('STAGE3_FORWARD_AFTER_GO_FINAL')
                return

            vx = self.get_yellow_slowdown_speed(
                yellow, self.stage3_go_final_speed,
                self.stage3_go_final_yellow_slow_speed,
                self.yellow_slowdown_ratio_final
            )

            # GO_FINAL 恢复 RGB 中线居中，但不做黄线角度 wz 矫正。
            # vy 由左右 RGB 参考球计算，wz 固定为 0。
            self.send_center_cruise_command_with_wz(ball, vx, 0.0)
            return

        if self.state == 'DONE':
            self.send_stop_command()
            return



def main(args=None):
    rclpy.init(args=args)
    node = Stage2Node()
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
