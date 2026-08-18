#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第六赛段节点：把足球推出出口区域，到达终点圆圈并趴下。

原 control_node_123456.py 的 SixthStageMixin（状态机与视觉逻辑原样搬移）。
MISSION_COMPLETE 持续发送趴下命令 p6_mission_complete_grace_sec 秒后，
向任务控制节点上报完成。
"""

import math
import traceback

import cv2
import numpy as np

import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from control_node.stage_common import StageNodeBase
from control_node.football_vision import NEAR_BALL, detect_football
from control_node.stage6_perception import annotate_exit
from control_node.stage_entry import EntryPoint, StageEntryTable


def p6_entry_table():
    """第六赛段调试入口表（顺序即流程顺序）。

    状态集合取自机器人上实测跑通的第六赛段状态机（2026-08-16 版本）：起点是
    NORTHWARD_MARCH，BLIND_MARCH / CLEAR_BALL_TURN / P6_TEST_HOLD 已经不存在，
    新增 POST_TURN_CRAB / EASTWARD_BLIND_MARCH / PUSH_SETUP_TURN。
    """
    states = (
        'P6_START',
        'NORTHWARD_MARCH',
        'FINAL_ALIGN',
        'OPEN_LOOP_TURN',
        'POST_TURN_CRAB',
        'EASTWARD_MARCH',
        'EASTWARD_BLIND_MARCH',
        'PUSH_SETUP_TURN',
        'BUFFER_CRAB',
        'LOWER_BODY_FOR_CRAB',
        'CLEAR_BALL_CRAB',
        'RESTORE_POSTURE',
        'TURN_TO_WEST_WALL',
        'WESTWARD_VISUAL_MARCH',
        'ALIGN_WEST_WALL',
        'TURN_TO_EXIT',
        'LOWER_BODY_FINAL',
        'PUSH_TO_EXIT',
        'CROSS_FINISH_LINE',
        'MISSION_COMPLETE',
    )
    wall = '对应的墙面必须在前向视野内'
    return StageEntryTable(6, 'NORTHWARD_MARCH', states, (
        EntryPoint('start', 'NORTHWARD_MARCH', '起步北进贴墙，完整第六赛段',
                   requires=(wall,)),
        EntryPoint('north', 'NORTHWARD_MARCH', '北进贴墙', requires=(wall,)),
        EntryPoint('north_align', 'FINAL_ALIGN', '对北墙校准', requires=(wall,)),
        EntryPoint('turn', 'OPEN_LOOP_TURN', '开环右转 90°'),
        EntryPoint('post_turn_crab', 'POST_TURN_CRAB', '转弯后横移就位'),
        EntryPoint('east', 'EASTWARD_MARCH', '东进潜入，绕过球'),
        EntryPoint('east_blind', 'EASTWARD_BLIND_MARCH', '东进盲走段'),
        EntryPoint('push_setup', 'PUSH_SETUP_TURN', '推球前的对位自转'),
        EntryPoint('crab', 'CLEAR_BALL_CRAB', '解围：右横移推球'),
        EntryPoint('west', 'TURN_TO_WEST_WALL', '转向西墙'),
        EntryPoint('west_march', 'WESTWARD_VISUAL_MARCH', '视觉靠近西墙',
                   requires=(wall,)),
        EntryPoint('west_align', 'ALIGN_WEST_WALL', '对西墙校准', requires=(wall,)),
        EntryPoint('exit', 'TURN_TO_EXIT', '转向出口'),
        EntryPoint('push', 'PUSH_TO_EXIT', '推向出口'),
        EntryPoint('finish', 'CROSS_FINISH_LINE', '冲过终点线'),
    ))


class Stage6Node(StageNodeBase):

    STAGE_ID = 6

    def __init__(self):
        super().__init__('stage6_node', self.STAGE_ID)
        # The browser overlay and motion control consume the same perception
        # result.  This removes the former split-brain situation where the web
        # detector saw a line while this node's private detector did not.
        self.external_wall_last_rx_s = None
        # The external detector is the Stage 6 control source.  Allow a short
        # scheduling hiccup without alternating WALK/STOP, while still
        # failing safe well before the robot can travel a meaningful distance.
        self.external_wall_timeout_s = 0.75
        self.external_wall_frame_seq = -1
        self.wall_alignment_valid = False
        self.wall_alignment_confidence = 0.0
        self.wall_alignment_stable = False
        self.wall_alignment_raw_rad = 0.0
        self.wall_alignment_mad_deg = 999.0
        self.external_wall_sub = self.create_subscription(
            Float32MultiArray,
            '/stage6/perception/yellow_line',
            self._external_wall_cb,
            10,
        )
        # Publish the exact annotated frame used by this control node.  The
        # browser must consume this topic so its line/depth values cannot
        # diverge from the values that trigger slowing and stopping.
        self.control_overlay_pub = self.create_publisher(
            Image, '/stage6/debug/control_overlay', 1)
        # MISSION_COMPLETE（趴下）持续多久后向任务控制节点上报完成。
        self.declare_parameter('p6_mission_complete_grace_sec', 3.0)
        self.p6_mission_complete_grace_sec = float(
            self.get_parameter('p6_mission_complete_grace_sec').value)
        # 调试入口：可写具名入口（north、east、west_march …）或直接写状态名。
        # 统一的 entry_point 参数（launch 里的 stage6_entry）优先于这一个。
        # p6_north_align_test_only 及其 p6_north_test_* 参数已随机器人上的
        # 状态机改版一起删除：那条调试路径的终点 P6_TEST_HOLD 不再存在。
        self.declare_parameter('p6_initial_state', 'default')
        self.sixth_stage_init()
        # 原代码 show_vision 恒为 True；拆分后跟随 show_debug_vis 参数。
        self.show_vision = bool(self.show_debug_vis)
        self._mission_complete_since = None

    def _external_wall_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return
        self.wall_line_visible = bool(msg.data[0] >= 0.5)
        self.wall_dist = float(msg.data[1])
        self.wall_angle_rad = float(msg.data[2])
        if len(msg.data) >= 9:
            self.wall_alignment_valid = bool(msg.data[3] >= 0.5)
            self.wall_alignment_confidence = float(msg.data[4])
            self.external_wall_frame_seq = int(msg.data[5])
            self.wall_alignment_stable = bool(msg.data[6] >= 0.5)
            self.wall_alignment_raw_rad = float(msg.data[7])
            self.wall_alignment_mad_deg = float(msg.data[8])
        else:
            # Backwards compatibility while the always-on perception process
            # is being upgraded.  It may steer, but cannot pass the new
            # multi-frame stability gate.
            self.external_wall_frame_seq += 1
            self.wall_alignment_valid = self.wall_line_visible
            self.wall_alignment_confidence = 0.0
            self.wall_alignment_stable = False
        self.external_wall_last_rx_s = self.now_sec()

    def _external_wall_fresh(self) -> bool:
        if self.external_wall_last_rx_s is None:
            return False
        age = self.now_sec() - self.external_wall_last_rx_s
        return 0.0 <= age <= self.external_wall_timeout_s

    def _precision_alignment_step(self, msg, state_elapsed,
                                  settle_s=None, tolerance_deg=None):
        """Align only from the filtered green vivid-yellow line."""
        settle_s = self.align_settle_s if settle_s is None else float(settle_s)
        tolerance_deg = (
            self.align_tolerance_deg
            if tolerance_deg is None else float(tolerance_deg))
        msg.mode = 11
        # Waiting for a stable green line must be a true four-leg stand.  Trot
        # gait with zero velocity still shifts the feet and drifts on an uneven
        # floor.  Switch to gait 3 only for an actual yaw correction below.
        msg.gait_id = 0
        msg.step_height = [0.04, 0.04]
        msg.vel_des = [0.0, 0.0, 0.0]

        if state_elapsed < settle_s:
            return False

        # No cyan contour, no colour-blob fallback.  The same green line shown
        # in the browser is the sole control and pass/fail measurement.
        measurement_ok = bool(
            self._external_wall_fresh()
            and self.wall_line_visible
            and self.wall_alignment_valid
            and self.wall_alignment_stable
            and self.wall_alignment_confidence >= self.align_confidence_min)
        if not measurement_ok:
            return False

        angle_deg = math.degrees(self.wall_angle_rad)
        error_deg = angle_deg - self.align_target_angle_deg
        if abs(error_deg) <= tolerance_deg:
            self.get_logger().info(
                '[P6_FINAL_ALIGN] green line angle={:.2f}deg within '
                '+/-{:.2f}deg; done'.format(angle_deg, tolerance_deg))
            return True

        vyaw = -math.radians(error_deg) * self.align_vyaw_kp
        vyaw = max(min(vyaw, self.align_max_vyaw), -self.align_max_vyaw)
        if abs(vyaw) < self.align_min_vyaw:
            vyaw = math.copysign(self.align_min_vyaw, vyaw)
        msg.gait_id = 3
        msg.vel_des = [0.0, 0.0, vyaw]
        if self.state_ticks == 1 or self.state_ticks % 5 == 0:
            self.get_logger().info(
                '[P6_FINAL_ALIGN] green line={:.2f}deg conf={:.2f} '
                'cmd_wz={:.3f}'.format(
                    angle_deg, self.wall_alignment_confidence, vyaw))
        return False

    def handle_rgb_msg(self, msg: Image):
        # 黄线/墙体校准统一消费常驻感知节点发布的结果；在这些阶段再次
        # 跑整套 OpenCV 会抢占感知与 20 Hz 真机运动心跳。只有最终推球
        # 仍需要本节点从原图计算足球和出口。
        if self.state != 'PUSH_TO_EXIT':
            return
        self.p6_image_callback(msg)

    def on_activated(self):
        self._mission_complete_since = None
        self.enter_sixth_stage('mission control activation')
        self.p6_behavior_loop()

    def stage_control_loop(self):
        self.p6_control_loop()
        if self.state == 'MISSION_COMPLETE':
            now = self.now_sec()
            if self._mission_complete_since is None:
                self._mission_complete_since = now
            elif now - self._mission_complete_since >= self.p6_mission_complete_grace_sec:
                self.complete_stage('MISSION_COMPLETE (lie-down grace elapsed)')

    def sixth_stage_init(self):
        # 👇👇👇 [核心修改]：强制当前节点使用 Gazebo 的仿真时间
        # 👆👆👆

        # ========================================================
        # 👑 战术调参区 (重点关注这里)
        # ========================================================
        # 北向黄线视觉接近：只直线前进并按照距离逐级减速，最终停在
        # 黄线前 0.70 m；偏航校准只允许在停车后的 FINAL_ALIGN 执行。
        self.north_stop_dist_m = 0.70
        self.north_search_vx = 0.10
        self.north_cruise_vx = 0.20
        self.north_slow_dist_m = 1.20
        self.north_crawl_dist_m = 0.75
        self.north_min_vx = 0.05
        self.north_approach_timeout_s = 30.0
        # 不假设相机存在固定偏置；只使用绿色严格黄线拟合结果。
        self.align_target_angle_deg = 0.0
        self.align_tolerance_deg = 0.40   # 【完成阈值】绿色角度进入 ±0.40°立即结束
        self.align_settle_s = 0.40        # 【停车稳定】只等 0.4 秒
        self.align_confidence_min = 0.60  # 【绿色长黄线最低置信度】
        self.align_vyaw_kp = 0.80         # 【精校角速度比例】
        self.align_max_vyaw = 0.08        # 【精校最大角速度】rad/s
        self.align_min_vyaw = 0.04        # 【细调最小角速度】防止小命令不产生动作

        # 👇👇👇 转身90度调参区 👇👇👇
        self.turn_vyaw = -0.60               # 自转角速度 (负数是右转，保持不变即可)
        self.turn_time_s = 3.35              # 【调这里！】转身时间(秒)。慢慢改这个数字直到刚好转90度！
        # 👆👆👆 转身90度调参区 👆👆👆

        # 👇👇👇 转身后贴北墙横移调参区 👇👇👇
        self.post_turn_crab_vy = 0.15       # 【向左横移】速度（正数向左/向北）
        self.post_turn_crab_time_s = 3.7    # 【横移时长】决定离北墙有多近

        # 👇👇👇 东进：绿线到 0.60m 后无视觉盲走调参区 👇👇👇
        self.eastward_vx = 0.12              # 【绿线接近速度】始终保持稳定慢走
        self.east_line_switch_dist_m = 0.60  # 【切盲走距离】绿色拟合线沿线深度
        self.east_line_confirm_frames = 3    # 连续 3 个不同绿线帧达到 0.60m
        self.eastward_timeout_s = 20.0        # 绿线阶段超时只停车
        self.east_blind_vx = 0.12            # 【调参1】切换后盲走速度
        self.east_blind_time_s = 2         # 【调参2】切换后盲走时间（主要调这个）

        # 👇👇👇 解围推球调参区（无收腿，保留推球前必要转身）👇👇👇
        self.push_setup_turn_vyaw = -0.40   # 1.【推球前转身角速度】负数右转
        self.push_setup_turn_time_s = 2.30  # 2.【推球前转身时间】
        self.buffer_crab_vy = -0.05         # 3.【降身前缓冲横移速度】
        self.buffer_crab_time_s = 0.2       # 4.【缓冲横移时间】
        self.normal_body_height_m = 0.25    # 5.【正常机身高度】
        self.crab_body_height_m = 0.12      # 6.【侧身推球机身高度】
        self.lower_body_settle_s = 1.50     # 7.【等待机身降低时间】
        self.clear_crab_vy = -0.40          # 8.【侧身推球横移速度】中速步态
        self.clear_crab_roll_rad = math.radians(6.0)  # 9.【推球侧倾】正值降低右侧，6度
        self.clear_crab_time_s = 3.00       # 9.【侧身推球时间】
        self.restore_body_settle_s = 2.50   # 10.【恢复正常高度等待时间】
        # 👆👆👆 解围推球调参区 👆👆👆

      # 👇👇👇 终极解围后新战术：西墙校准与冲线调参区 👇👇👇
        # 1. 转西墙
        self.west_turn_vyaw = -0.6           # 【转西墙】角速度 (正数左转，负数右转，视狗当前姿态而定)
        self.west_turn_time_s = 5.2         # 【转西墙】时间(秒)。调这里让它大概正对西墙！



        self.west_visual_march_vx = 0.1    # 【视觉靠近西墙】边走边调姿态的速度
        self.west_stop_dist_m = 0.45        # 🎯【调这里！】距离西墙多近时刹车 (米)
        self.west_align_tolerance_deg = 1 # 【西墙校准】偏角小于几度算对齐西墙

        # 2. 转出口
        self.exit_turn_vyaw = 0.6           # 【转出口】角速度 (反向转回去面朝南方出口)
        self.exit_turn_time_s = 2.7         # 【转出口】时间(秒)。调这里让它刚好正对出口！

        # 👇👇👇 终极冲刺：三点一线推球调参区 👇👇👇
        self.push_vx = 0.30                 # 【推球】前进速度 (稳稳向前推)
        self.push_vy_kp = 0.30              # 【推球】横向追球敏锐度 (球偏了，狗往侧面滑去追的力度)
        self.push_vyaw_kp = 0.6             # 【推球】转头瞄准出口敏锐度 (头永远正对大门)


        self.exit_lost_confirm_s = 0.50      # 【出口消失确认】防止单帧漏检导致提前停车
        self.cross_line_time_s = 0.01        # 【确认出门后盲走】当前基本立即停车
        self.push_timeout_s = 11.0           # 【总时限保底】从开始最终推球计时；本场约停在出口外一机身
        # 👆👆👆 终极冲刺：三点一线推球调参区 👆👆👆

# 🆕 新增：出口冲线与趴下参数


        self.life_count_val = 0
        self.p6_body_height_m = self.normal_body_height_m
        self.wall_angle_rad = 0.0
        self.wall_dist = -1.0
        self.wall_line_visible = False
        self.wall_alignment_valid = False
        self.wall_alignment_confidence = 0.0
        self.wall_alignment_stable = False
        self.wall_alignment_raw_rad = 0.0
        self.wall_alignment_mad_deg = 999.0

        # ========================================================
        # 视觉模块整合区：原 ball_vision_tracker + wall_vision_tracker
        # 不再发布/订阅 /vision/ball_info、/vision/wall_info、/vision/exit_info，
        # 直接在同一个节点内更新 self.ball_* / self.wall_* / self.exit_*。
        # ========================================================

        self.wall_angle_rad = 0.0
        self.wall_dist = -1.0

        self.ball_offset_x = -999.0
        self.ball_dist = -1.0
        self.ball_detection_miss_frames = 0
        self.ball_detection_hold_frames = 5

        self.exit_offset_norm = -999.0
        self.exit_dist = -1.0

        self.show_vision = True
        self.vision_window_name = 'Sixth Stage Integrated Vision'
        self.vision_window_ready = False



        # 将初始状态设为盲走
        self.state_ticks = 0
        self.stable_counter = 0
        # 进入状态时记录 ROS clock 时间；use_sim_time=True 时这里就是 Gazebo 仿真时间。
        self.p6_state_start_time = self.now_sec()
        self.eastward_vy_done_logged = False
        self.east_line_last_seq = -1
        self.east_line_confirm_count = 0
        self.ball_lost_start_time = None
        self.exit_lost_start_time = None

        # 第六赛段状态集合与调试入口。注意：这些必须在 sixth_stage_init() 里执行，
        # 不能放到 now_sec() 的 return 后面，否则 P6 状态分发和直接调试入口会失效。
        # 状态集合取自入口表，两处不会各自漂移。
        entry_table = p6_entry_table()
        self.p6_states = set(entry_table.state_names())
        self.p6_initial_state = self.resolve_stage_entry(
            entry_table, str(self.get_parameter('p6_initial_state').value))
        self.p6_control_period_s = 0.10
        self.p6_last_control_time = None
        self.p6_state_start_time = self.now_sec()
        self.get_logger().info(
            '[SixthStageMixin] ready: visual approach enabled, blind march disabled')


    def is_sixth_stage_state(self, state: str) -> bool:
        return isinstance(state, str) and state in getattr(self, 'p6_states', set())

    def enter_sixth_stage(self, reason: str = ''):
        self.get_logger().info(f'[HANDOFF] P5 -> P6: {reason}')

        # 进入第六赛段时，必须把 P6 自己的视觉缓存、状态计时和限速计时全部清掉。
        # 否则 P5_DONE 后第一次进入起始状态（NORTHWARD_MARCH）时，可能沿用旧的
        # p6_state_start_time，导致该状态的计时被瞬间判定为完成。
        self.clear_pre_sixth_vision_caches()
        self.p6_last_control_time = None
        self.p6_force_zero_elapsed_once = True

        # 先恢复第六赛段默认机身参数，再进入黄线视觉接近。
        # p6_set_state 内部会记录当前 Gazebo 仿真时间作为状态起点。
        self.set_body_height(self.normal_body_height_m)
        self.p6_set_state(self.p6_initial_state)
        self.get_logger().info('=== 第六赛段启动：幽灵潜入 ===')

    def clear_pre_sixth_vision_caches(self):
        self.wall_angle_rad = 0.0
        self.wall_dist = -1.0
        self.wall_line_visible = False
        self.wall_alignment_valid = False
        self.wall_alignment_confidence = 0.0
        self.wall_alignment_stable = False
        self.wall_alignment_raw_rad = 0.0
        self.wall_alignment_mad_deg = 999.0
        self.ball_offset_x = -999.0
        self.ball_dist = -1.0
        self.ball_detection_miss_frames = 0
        self.exit_offset_norm = -999.0
        self.exit_dist = -1.0
        self.stable_counter = 0
        self.ball_lost_start_time = None
        self.exit_lost_start_time = None
        self.eastward_vy_done_logged = False
        self.east_line_last_seq = -1
        self.east_line_confirm_count = 0
        self.has_seen_exit = False
        self.exit_lost_ticks = 0
        self.ball_lost_ticks = 0

    def p6_send_cmd(self, msg):
        self.Ctrl.Send_cmd(msg)
        # Robot_Ctrl may advance the very first command once more to guarantee
        # that its life_count differs from the previous stage.
        self.life_count_val = int(msg.life_count)

    def p6_set_state(self, new_state: str):
        """第六赛段统一切换状态：清 tick，并记录进入状态的仿真时间。"""
        old_state = getattr(self, 'state', None)
        now = self.now_sec()

        self.state = new_state
        self.state_ticks = 0
        self.stable_counter = 0

        # 记录当前 Gazebo /clock 时间作为新状态起点。
        # 同时要求下一次 state_elapsed_s() 强制返回 0，避免刚切状态的同一轮回调
        # 由于旧时间戳/阻塞等待造成 elapsed 异常偏大。
        self.p6_state_start_time = now
        self.p6_force_zero_elapsed_once = True
        self.p6_last_control_time = None

        if new_state == 'EASTWARD_MARCH':
            self.eastward_vy_done_logged = False
            # 丢弃转身/横移阶段遗留的旧绿线，只接受东进后的新结果帧。
            self.east_line_last_seq = int(self.external_wall_frame_seq)
            self.east_line_confirm_count = 0
            self.ball_lost_ticks = 0
            self.ball_lost_start_time = None
        if new_state == 'PUSH_TO_EXIT':
            self.has_seen_exit = False
            self.exit_lost_ticks = 0
            self.exit_lost_start_time = None
        self.get_logger().info(f'[P6] ENTER STATE: {old_state} -> {new_state}, sim_time={now:.3f}')

    def p6_state_elapsed_s(self) -> float:
        """第六赛段当前状态已经持续的仿真时间。"""
        now = self.now_sec()

        if not hasattr(self, 'p6_state_start_time') or self.p6_state_start_time is None:
            self.p6_state_start_time = now
            self.p6_force_zero_elapsed_once = False
            return 0.0

        # 刚进入新状态后的第一轮控制强制认为 elapsed=0。
        # 这样 P5_DONE -> P6 的交接不会把起始状态直接跳过。
        if getattr(self, 'p6_force_zero_elapsed_once', False):
            self.p6_force_zero_elapsed_once = False
            self.p6_state_start_time = now
            return 0.0

        self.p6_state_start_time = self.align_motion_timer_start(
            self.p6_state_start_time, now)
        elapsed = now - self.p6_state_start_time
        if elapsed < 0.0:
            self.p6_state_start_time = now
            return 0.0
        return elapsed

    def p6_control_loop(self):
        now = self.now_sec()
        if self.p6_last_control_time is not None and (now - self.p6_last_control_time) < self.p6_control_period_s:
            return
        self.p6_last_control_time = now
        self.p6_behavior_loop()

    def set_body_height(self, target_height: float):
        """只调整机身高度，绝不发送任何腿宽/剪刀腿参数。"""
        # 仿真通过高度 YAML；真机由每帧 MotionServoCmd 的 pos_des[2]
        # 持续携带高度。这里有意不再发布任何 y_offset_trot 参数。
        self.p6_body_height_m = float(target_height)
        values_h = [0.0] * 12
        values_h[2] = float(target_height)
        self.publish_yaml_vecxd("des_roll_pitch_height_motion", values_h)
        self.publish_yaml_vecxd("des_roll_pitch_height", values_h)


    def _median_depth_m(self, cx, cy, patch_radius=5, max_depth=5.0):
        """从深度图局部 patch 取中位数，返回米；无效时返回 -1.0。"""
        if self.latest_depth is None:
            return -1.0

        h, w = self.latest_depth.shape[:2]
        x = int(max(patch_radius, min(w - patch_radius - 1, cx)))
        y = int(max(patch_radius, min(h - patch_radius - 1, cy)))
        patch = self.latest_depth[y-patch_radius:y+patch_radius, x-patch_radius:x+patch_radius]

        if patch.dtype == np.uint16:
            patch = patch.astype(np.float32) / 1000.0
        else:
            patch = patch.astype(np.float32)

        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < max_depth)]
        if valid.size == 0:
            return -1.0
        return float(np.median(valid))

    def _update_wall_vision(self, cv_image, hsv):
        """黄线结果由常驻视觉节点统一发布；本节点不再重复计算。"""
        return

        height, width = cv_image.shape[:2]

        lower_yellow = np.array([10, 50, 40])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 与已经过真机验证的绿色黄线调试器保持一致：从画面高度 52%
        # 开始检测。原来的 70% 会漏掉尚在画面中部的目标黄线。
        roi_x_min = int(width * 0.20)
        roi_x_max = int(width * 0.80)
        roi_y_min = int(height * 0.52)
        roi_y_max = height

        mask[:roi_y_min, :] = 0
        mask[:, :roi_x_min] = 0
        mask[:, roi_x_max:] = 0

        cv2.rectangle(cv_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 0, 255), 2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        self.wall_angle_rad = 0.0
        self.wall_dist = -1.0
        self.wall_line_visible = False
        valid_walls = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue

            vx, vy, x0, y0 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)
            if vx < 0:
                vx, vy = -vx, -vy

            angle_rad = math.atan2(vy, vx)
            if abs(math.degrees(angle_rad)) < 30.0:
                valid_walls.append({
                    'area': area,
                    'angle_rad': angle_rad,
                    'cx': float(x0),
                    'cy': float(y0),
                    'vx': vx,
                    'vy': vy,
                })

        if not valid_walls:
            cv2.putText(cv_image, 'WALL: SEARCHING', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        # 选画面里位置最低的黄线，也就是最近内墙。
        best_wall = max(valid_walls, key=lambda item: item['cy'])
        self.wall_line_visible = True
        angle_rad = best_wall['angle_rad']
        cx_contour, cy_contour = best_wall['cx'], best_wall['cy']
        vx, vy = best_wall['vx'], best_wall['vy']

        line_length = 500
        pt1 = (int(cx_contour - vx * line_length), int(cy_contour - vy * line_length))
        pt2 = (int(cx_contour + vx * line_length), int(cy_contour + vy * line_length))
        cv2.line(cv_image, pt1, pt2, (0, 255, 0), 2)

        center_x = width // 2
        if abs(vx) > 1e-5:
            center_y = int(cy_contour + (vy / vx) * (center_x - cx_contour))
        else:
            center_y = int(cy_contour)

        center_x = max(10, min(width - 11, center_x))
        center_y = max(10, min(height - 11, center_y))

        cv2.circle(cv_image, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.line(cv_image, (center_x, 0), (center_x, height), (0, 255, 255), 1)

        real_dist = self._median_depth_m(center_x, center_y, patch_radius=5, max_depth=5.0)
        self.wall_angle_rad = float(angle_rad)
        self.wall_dist = float(real_dist)

        if real_dist > 0:
            cv2.putText(cv_image, f'WALL Dist:{real_dist:.2f}m ANG:{math.degrees(angle_rad):.1f}deg',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(cv_image, 'WALL DEPTH INVALID', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    def _update_exit_vision(self, cv_image, hsv):
        """整合 ball_vision_tracker 的出口检测：检测黄色墙壁断口，更新 exit_offset_norm / exit_dist。"""
        result = annotate_exit(cv_image, hsv, self._median_depth_m)
        self.exit_offset_norm = float(result['offset_norm'])
        self.exit_dist = float(result['distance_m'])
        return

        height, width = cv_image.shape[:2]

        self.exit_offset_norm = -999.0
        self.exit_dist = -1.0

        lower_yellow = np.array([10, 50, 40])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel_y = np.ones((9, 9), np.uint8)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel_y)
        yellow_contours = cv2.findContours(
            mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        valid_walls = []
        for c in yellow_contours:
            if cv2.contourArea(c) > 1500:
                x, y, w, h = cv2.boundingRect(c)
                valid_walls.append({'x': x, 'y': y, 'w': w, 'h': h, 'bottom_y': y + h, 'contour': c})

        if not valid_walls:
            cv2.putText(cv_image, 'EXIT: SEARCHING', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        valid_walls.sort(key=lambda item: item['bottom_y'], reverse=True)
        nearest_bottom_y = valid_walls[0]['bottom_y']
        front_walls = [w for w in valid_walls if abs(w['bottom_y'] - nearest_bottom_y) < 350]
        front_walls.sort(key=lambda item: item['x'])

        for i, w in enumerate(front_walls):
            cv2.drawContours(cv_image, [w['contour']], -1, (0, 255, 255), 2)
            cv2.putText(cv_image, f'EW{i+1}', (w['x'], max(20, w['y'] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if len(front_walls) < 2:
            cv2.putText(cv_image, 'EXIT: NEED 2 WALLS', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        max_gap = 0
        gap_cx = -1
        best_left_edge = None
        best_right_edge = None
        for i in range(len(front_walls) - 1):
            left_edge = front_walls[i]['x'] + front_walls[i]['w']
            right_edge = front_walls[i + 1]['x']
            gap = right_edge - left_edge
            if gap > max_gap:
                max_gap = gap
                gap_cx = left_edge + gap / 2.0
                best_left_edge = left_edge
                best_right_edge = right_edge

        if max_gap <= 20 or best_left_edge is None or best_right_edge is None:
            cv2.putText(cv_image, 'EXIT: GAP INVALID', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        cv2.line(cv_image, (int(gap_cx), 0), (int(gap_cx), height), (255, 0, 255), 3)
        self.exit_offset_norm = float((gap_cx - width / 2.0) / (width / 2.0))

        exit_dist = -1.0
        px_l = int(max(5, min(width - 6, best_left_edge - 5)))
        px_r = int(max(5, min(width - 6, best_right_edge + 5)))
        py = int(max(5, min(height - 6, nearest_bottom_y - 10)))

        cv2.circle(cv_image, (px_l, py), 5, (255, 0, 0), -1)
        cv2.circle(cv_image, (px_r, py), 5, (255, 0, 0), -1)

        d_l = self._median_depth_m(px_l, py, patch_radius=5, max_depth=5.0)
        d_r = self._median_depth_m(px_r, py, patch_radius=5, max_depth=5.0)
        valid_dists = [d for d in (d_l, d_r) if d > 0]
        if valid_dists:
            exit_dist = float(min(valid_dists))
        self.exit_dist = exit_dist

        if exit_dist > 0:
            cv2.putText(cv_image, f'EXIT off:{self.exit_offset_norm:.2f} dist:{exit_dist:.2f}m',
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            cv2.putText(cv_image, f'EXIT off:{self.exit_offset_norm:.2f} depth invalid',
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    def _update_ball_vision(self, cv_image, hsv):
        """Detect the official 20 cm black-and-white size-4 football."""
        height, width = cv_image.shape[:2]

        previous_offset = self.ball_offset_x
        previous_dist = self.ball_dist
        # Stage 6 only ever pushes a ball that is already close, so it opts
        # into the near-field gates.  Stage 4 keeps the far-field defaults.
        candidate = detect_football(
            cv_image,
            depth_at=lambda x, y: self._median_depth_m(
                x, y, patch_radius=7, max_depth=5.0),
            focal_px=400.0,
            **NEAR_BALL
        )
        if candidate is None:
            self.ball_detection_miss_frames += 1
            if (self.ball_detection_miss_frames <= self.ball_detection_hold_frames and
                    previous_offset != -999.0):
                self.ball_offset_x = previous_offset
                self.ball_dist = previous_dist
                cv2.putText(cv_image, 'BALL: SHORT HOLD', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return
            self.ball_offset_x = -999.0
            self.ball_dist = -1.0
            cv2.putText(cv_image, 'BALL: SEARCHING', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        self.ball_detection_miss_frames = 0

        x = candidate['x']
        y = candidate['y']
        radius = candidate['radius']

        cv2.circle(cv_image, (int(x), int(y)), int(max(radius, 2)), (0, 255, 0), 2)
        cv2.circle(cv_image, (int(x), int(y)), 3, (0, 0, 255), -1)

        measured_offset = float((x - width / 2.0) / (width / 2.0))
        measured_dist = float(candidate.get('depth_m', -1.0))
        if measured_dist <= 0.0:
            measured_dist = self._median_depth_m(
                x, y, patch_radius=7, max_depth=5.0)
        if previous_offset != -999.0:
            self.ball_offset_x = 0.65 * previous_offset + 0.35 * measured_offset
        else:
            self.ball_offset_x = measured_offset
        if measured_dist > 0.0 and previous_dist > 0.0:
            self.ball_dist = 0.65 * previous_dist + 0.35 * measured_dist
        elif measured_dist > 0.0:
            self.ball_dist = measured_dist
        else:
            self.ball_dist = previous_dist

        diameter_m = (-1.0 if self.ball_dist <= 0.0 else
                      2.0 * radius * self.ball_dist / 400.0)

        if self.ball_dist > 0:
            cv2.putText(cv_image, f'BALL off:{self.ball_offset_x:.2f} dist:{self.ball_dist:.2f}m '
                        f'diam:{diameter_m:.2f}m',
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(cv_image, f'BALL off:{self.ball_offset_x:.2f} depth invalid',
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    def p6_image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # 每个状态只运行实际参与该阶段控制的视觉，避免北向黄线调试
            # 画面被黄色出口轮廓和足球标注覆盖。
            wall_states = {
                'NORTHWARD_MARCH', 'FINAL_ALIGN', 'EASTWARD_MARCH',
                'WESTWARD_VISUAL_MARCH', 'ALIGN_WEST_WALL',
            }
            if self.state in wall_states:
                self._update_wall_vision(cv_image, hsv)
            if self.state == 'PUSH_TO_EXIT':
                # 足球必须先在未经出口轮廓污染的原图上检测。
                self._update_ball_vision(cv_image, hsv)
                self._update_exit_vision(cv_image, hsv)

            cv2.putText(cv_image, f'STATE: {self.state}', (20, cv_image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            overlay_msg.header = msg.header
            self.control_overlay_pub.publish(overlay_msg)

            if self.show_vision:
                if not self.vision_window_ready:
                    try:
                        cv2.namedWindow(self.vision_window_name, cv2.WINDOW_NORMAL)
                        self.vision_window_ready = True
                    except Exception:
                        self.show_vision = False
                if self.show_vision:
                    cv2.imshow(self.vision_window_name, cv_image)
                    cv2.waitKey(1)

        except Exception:
            self.get_logger().error(f'Integrated vision failed:\n{traceback.format_exc()}')


    def destroy_vision_windows(self):
        try:
            if self.vision_window_ready:
                cv2.destroyWindow(self.vision_window_name)
            cv2.destroyAllWindows()
        except Exception:
            pass


    def p6_behavior_loop(self):
        if self.state == 'P6_START':
            self.p6_set_state(self.p6_initial_state)
            return
        self.life_count_val += 1
        if self.life_count_val > 127:
            self.life_count_val = 1
        msg = robot_control_cmd_lcmt()
        msg.life_count = self.life_count_val
        msg.duration = 0
        msg.pos_des = [0.0, 0.0, float(self.p6_body_height_m)]
        msg.rpy_des = [0.0, 0.0, 0.0]
        self.state_ticks += 1
        state_elapsed = self.p6_state_elapsed_s()

        # 黄线可见与深度有效是两件事：接近阶段仅用可靠深度减速、停车；
        # 黄线角度留到停车后的 FINAL_ALIGN 阶段再用于原地校准。
        wall_visible = bool(self.wall_line_visible)

        # ==========================================
        # 0. 黄线视觉接近阶段
        # ==========================================
        if self.state == 'NORTHWARD_MARCH':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.05, 0.05]

            # 未见黄线时只允许低速搜索，绝不使用原来的 0.5 m/s。
            vx = self.north_search_vx
            vyaw = 0.0

            # No fresh perception means no motion.  Never walk blindly toward
            # the line when the detector/DDS link has stopped updating.
            if not self._external_wall_fresh():
                msg.vel_des = [0.0, 0.0, 0.0]
                self.p6_send_cmd(msg)
                if self.state_ticks == 1 or self.state_ticks % 10 == 0:
                    self.get_logger().error(
                        '[P6_VISUAL_APPROACH] perception stale/missing; HOLD STOP')
                return

            if wall_visible:
                if self.wall_dist > 0.0:
                    if self.wall_dist <= self.north_crawl_dist_m:
                        vx = self.north_min_vx
                    elif self.wall_dist < self.north_slow_dist_m:
                        span = self.north_slow_dist_m - self.north_crawl_dist_m
                        ratio = ((self.wall_dist - self.north_crawl_dist_m) /
                                 max(span, 1e-3))
                        vx = self.north_min_vx + ratio * (
                            self.north_cruise_vx - self.north_min_vx)
                    else:
                        vx = self.north_cruise_vx

                if 0.0 < self.wall_dist <= self.north_stop_dist_m:
                    self.stable_counter += 1
                    if self.stable_counter >= 3:
                        self.get_logger().info(
                            f"🎯 抵达黄线前 {self.wall_dist:.2f}m，停车并精确校准。")
                        msg.vel_des = [0.0, 0.0, 0.0]
                        self.p6_send_cmd(msg)
                        self.p6_set_state('FINAL_ALIGN')
                        self.state_ticks = 0
                        self.stable_counter = 0
                        return
                else:
                    self.stable_counter = 0

            if self.state_ticks == 1 or self.state_ticks % 10 == 0:
                self.get_logger().info(
                    '[P6_VISUAL_APPROACH] line={} dist={:.2f}m angle={:.1f}deg '
                    'cmd_vx={:.2f} cmd_wz={:.2f}'.format(
                        wall_visible, self.wall_dist,
                        math.degrees(self.wall_angle_rad), vx, vyaw))

            if state_elapsed >= self.north_approach_timeout_s:
                self.get_logger().error(
                    '黄线视觉接近超时，安全停车；不会继续后续动作。')
                msg.vel_des = [0.0, 0.0, 0.0]
                self.p6_send_cmd(msg)
                return

            msg.vel_des = [vx, 0.0, vyaw]
            self.p6_send_cmd(msg)

        # ==========================================
        # 2. 原地绝对垂直校准
        # ==========================================
        elif self.state == 'FINAL_ALIGN':
            aligned = self._precision_alignment_step(msg, state_elapsed)
            self.p6_send_cmd(msg)
            if aligned:
                angle_deg = math.degrees(self.wall_angle_rad)
                self.get_logger().info(
                    '📐 北墙绿色黄线校准结束：angle={:.2f}deg '
                    'conf={:.2f}，开始自转90度！'.format(
                        angle_deg, self.wall_alignment_confidence))
                self.p6_set_state('OPEN_LOOP_TURN')
                self.state_ticks = 0
                self.stable_counter = 0
                return

        # ==========================================
        # 3. 参数开环转身
        # ==========================================
        elif self.state == 'OPEN_LOOP_TURN':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if state_elapsed >= self.turn_time_s:
                self.get_logger().info("✅ 转向东侧结束！开始向左横移贴近北墙！")
                self.p6_set_state('POST_TURN_CRAB')
                return

            msg.vel_des = [0.0, 0.0, self.turn_vyaw]
            self.p6_send_cmd(msg)

        # ==========================================
        # 3.5. 纯横移靠拢北墙
        # ==========================================
        elif self.state == 'POST_TURN_CRAB':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if state_elapsed >= self.post_turn_crab_time_s:
                self.get_logger().info("🛑 横移贴墙完成！开启视觉测距，全速东征！")
                self.p6_set_state('EASTWARD_MARCH')
                return

            msg.vel_des = [0.0, self.post_turn_crab_vy, 0.0]
            self.p6_send_cmd(msg)

        # ==========================================
        # 4. 东进视觉段：绿色拟合线到 0.60m
        # ==========================================
        elif self.state == 'EASTWARD_MARCH':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if state_elapsed >= self.eastward_timeout_s:
                self.get_logger().error(
                    "⚠️ 东进绿线接近达到20秒上限：停车等待，不进入盲走！")
                msg.vel_des = [0.0, 0.0, 0.0]
                self.p6_send_cmd(msg)
                return

            # 与北进相同：只消费网页绿色拟合线对应的统一感知结果。
            # DDS/感知停止更新时必须停车，不能误切盲走。
            if not self._external_wall_fresh():
                msg.vel_des = [0.0, 0.0, 0.0]
                self.p6_send_cmd(msg)
                if self.state_ticks == 1 or self.state_ticks % 10 == 0:
                    self.get_logger().error(
                        '[P6_EAST_LINE] perception stale/missing; HOLD STOP')
                return

            # 只在不同的感知 frame_seq 上累计，控制循环不能重复凑满3帧。
            if int(self.external_wall_frame_seq) != self.east_line_last_seq:
                self.east_line_last_seq = int(self.external_wall_frame_seq)
                if (wall_visible and
                        0.0 < self.wall_dist <= self.east_line_switch_dist_m):
                    self.east_line_confirm_count += 1
                else:
                    self.east_line_confirm_count = 0

            if self.east_line_confirm_count >= self.east_line_confirm_frames:
                self.get_logger().info(
                    '🎯 东进绿色拟合线 {:.2f}m，连续 {} 帧；'
                    '无停顿切换定时盲走 {:.2f}s @ {:.2f}m/s'.format(
                        self.wall_dist, self.east_line_confirm_count,
                        self.east_blind_time_s, self.east_blind_vx))
                # 切换前后发送相同方向的前进命令，避免先刹停再起步。
                msg.vel_des = [self.east_blind_vx, 0.0, 0.0]
                self.p6_send_cmd(msg)
                self.p6_set_state('EASTWARD_BLIND_MARCH')
                return

            if self.state_ticks == 1 or self.state_ticks % 10 == 0:
                self.get_logger().info(
                    '[P6_EAST_LINE] line={} dist={:.2f}m confirm={}/{} '
                    'cmd_vx={:.2f}'.format(
                        wall_visible, self.wall_dist,
                        self.east_line_confirm_count,
                        self.east_line_confirm_frames, self.eastward_vx))

            msg.vel_des = [self.eastward_vx, 0.0, 0.0]
            self.p6_send_cmd(msg)

        # ==========================================
        # 4.5. 过 0.60m 阈值后的纯定时盲走（完全不读视觉/深度）
        # ==========================================
        elif self.state == 'EASTWARD_BLIND_MARCH':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if state_elapsed >= self.east_blind_time_s:
                self.get_logger().info(
                    '✅ 东进盲走 {:.2f}s 完成，开始推球前转身！'.format(
                        self.east_blind_time_s))
                msg.vel_des = [0.0, 0.0, 0.0]
                self.p6_send_cmd(msg)
                self.p6_set_state('PUSH_SETUP_TURN')
                return

            msg.vel_des = [self.east_blind_vx, 0.0, 0.0]
            self.p6_send_cmd(msg)

        # ==========================================
        # 5. 推球前必要转身（不涉及腿宽变化）
        # ==========================================
        elif self.state == 'PUSH_SETUP_TURN':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if state_elapsed >= self.push_setup_turn_time_s:
                msg.vel_des = [0.0, 0.0, 0.0]
                self.p6_send_cmd(msg)
                self.get_logger().info(
                    f"🔄 推球前转身 {state_elapsed:.3f}/"
                    f"{self.push_setup_turn_time_s:.3f}s 完成！")
                self.p6_set_state('BUFFER_CRAB')
                self.state_ticks = 0
                return

            msg.vel_des = [0.0, 0.0, self.push_setup_turn_vyaw]
            self.p6_send_cmd(msg)

        # ==========================================
        # 6. 极慢速横移，腾出降底盘空间
        # ==========================================
        elif self.state == 'BUFFER_CRAB':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.03, 0.03]

            if state_elapsed >= self.buffer_crab_time_s:
                self.get_logger().info("🛡️ 缓冲空间已拉开，准备降低底盘贴地推球！")
                self.p6_set_state('LOWER_BODY_FOR_CRAB')
                self.state_ticks = 0
                msg.mode = 11
                self.p6_send_cmd(msg)
                return

            # 注意：仅做 vy 的横移，vx 为 0
            msg.vel_des = [0.0, self.buffer_crab_vy, 0.0]
            self.p6_send_cmd(msg)

        # ==========================================
        # 6.5. 插入新阶段：极限压低底盘
        # ==========================================
        elif self.state == 'LOWER_BODY_FOR_CRAB':
            if self.state_ticks == 1:
                self.set_body_height(self.crab_body_height_m)

            msg.mode = 11  # 保持原地站立，等待身体降下去
            msg.gait_id = 0
            msg.vel_des = [0.0, 0.0, 0.0]
            msg.pos_des = [0.0, 0.0, float(self.crab_body_height_m)]
            # 在接触足球前完成侧倾并稳定；推球阶段只保持姿态，不再突然下压。
            msg.rpy_des = [float(self.clear_crab_roll_rad), 0.0, 0.0]
            self.p6_send_cmd(msg)

            if state_elapsed > self.lower_body_settle_s:
                self.get_logger().info(
                    f"⬇️ 真机机身高度命令 {self.crab_body_height_m:.2f}m 已持续 "
                    f"{self.lower_body_settle_s:.2f}s，开始侧身推球！")
                self.p6_set_state('CLEAR_BALL_CRAB')
                self.state_ticks = 0

        # ==========================================
        # 7. 解围第二步：加速右侧横移把球扫出来
        # ==========================================
        elif self.state == 'CLEAR_BALL_CRAB':
            msg.mode = 11
            msg.gait_id = 28  # 推球专用映射：真机中速 Servo 308
            msg.step_height = [0.04, 0.04]
            msg.pos_des = [0.0, 0.0, float(self.crab_body_height_m)]
            msg.rpy_des = [float(self.clear_crab_roll_rad), 0.0, 0.0]

            if state_elapsed >= self.clear_crab_time_s:
                self.get_logger().info("🎉 解围完成！起立，准备找球！")
                self.p6_set_state('RESTORE_POSTURE')    # <--- 修改这里
                self.state_ticks = 0
                msg.mode = 11
                self.p6_send_cmd(msg)
                return

            # 注意：向右横移，所以 vx=0，vy 为负数
            msg.vel_des = [0.0, self.clear_crab_vy, 0.0]
            self.p6_send_cmd(msg)


        # ==========================================
        # 8. 起立恢复正常姿态
        # ==========================================
        elif self.state == 'RESTORE_POSTURE':
            if self.state_ticks == 1:
                self.set_body_height(self.normal_body_height_m)

            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]
            msg.vel_des = [0.0, 0.0, 0.0]
            msg.pos_des = [0.0, 0.0, float(self.normal_body_height_m)]
            self.p6_send_cmd(msg)

            if state_elapsed > self.restore_body_settle_s:
                self.get_logger().info("🐕 姿态恢复完毕！开始开环定时转身面向西墙！")
                self.p6_set_state('TURN_TO_WEST_WALL')  # <--- 修改这里
                self.state_ticks = 0
   # ==========================================
        # 9. 转向西墙 (纯调参开环)
        # ==========================================
        elif self.state == 'TURN_TO_WEST_WALL':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if state_elapsed >= self.west_turn_time_s:
                self.get_logger().info("✅ 转西墙结束！开始向前盲走靠近西墙！")
                self.p6_set_state('WESTWARD_VISUAL_MARCH')  # <--- 修改这里：去盲走！
                self.state_ticks = 0
                return

            msg.vel_des = [0.0, 0.0, self.west_turn_vyaw]
            self.p6_send_cmd(msg)


        # ==========================================
        # 9.6. 视觉靠近西墙 (直到满足距离才停车！)
        # ==========================================
        elif self.state == 'WESTWARD_VISUAL_MARCH':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            vx = self.west_visual_march_vx
            vyaw = 0.0

            if wall_visible:
                # 边走边粗调朝向
                vyaw = - (self.wall_angle_rad * 0.8)
                vyaw = max(min(vyaw, 0.3), -0.3)

                # 👑 核心逻辑：离西墙达到指定距离才刹车！
                if 0.0 < self.wall_dist < self.west_stop_dist_m:
                    self.stable_counter += 1
                    if self.stable_counter >= 2:
                        self.get_logger().info(f"🎯 抵达西墙极近距离 ({self.wall_dist:.2f}m)！停车准备原地垂直校准。")
                        self.p6_set_state('ALIGN_WEST_WALL')
                        self.state_ticks = 0
                        self.stable_counter = 0
                        msg.mode = 11
                        msg.vel_des = [0.0, 0.0, 0.0]
                        self.p6_send_cmd(msg)
                        return
                else:
                    self.stable_counter = 0

            msg.vel_des = [vx, 0.0, vyaw]
            self.p6_send_cmd(msg)

        # ==========================================
        # 10. 原地垂直校准西墙
        # ==========================================
        elif self.state == 'ALIGN_WEST_WALL':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if wall_visible:
                angle_deg = math.degrees(self.wall_angle_rad)

                # 对齐了！进入下一步
                if abs(angle_deg) < self.west_align_tolerance_deg:
                    self.stable_counter += 1
                    if self.stable_counter >= 3:
                        self.get_logger().info(f"📐 西墙已绝对垂直 (偏角: {angle_deg:.1f}度)！开始转身面朝出口！")
                        self.p6_set_state('TURN_TO_EXIT')
                        self.state_ticks = 0
                        self.stable_counter = 0
                        msg.mode = 11
                        self.p6_send_cmd(msg)
                        return
                else:
                    self.stable_counter = 0

                # 原地强扭 (P控制)
                vyaw = - (self.wall_angle_rad * 1.5)
                vyaw = max(min(vyaw, 0.4), -0.4)
                msg.vel_des = [0.0, 0.0, vyaw]
            else:
                msg.vel_des = [0.0, 0.0, 0.0]

            # 超时保护(最多对齐4秒)
            if state_elapsed > 3.5:
                self.get_logger().warn("⚠️ 西墙校准超时，强制进入转身出口阶段！")
                self.p6_set_state('TURN_TO_EXIT')
                self.state_ticks = 0

            self.p6_send_cmd(msg)

        # ==========================================
        # 11. 转向出口 (纯调参开环)
        # ==========================================
        elif self.state == 'TURN_TO_EXIT':
            msg.mode = 11
            msg.gait_id = 3
            msg.step_height = [0.04, 0.04]

            if state_elapsed >= self.exit_turn_time_s:
                self.get_logger().info("✅ 面朝出口！准备降底盘！")
                self.p6_set_state('LOWER_BODY_FINAL')
                self.state_ticks = 0
                msg.mode = 11
                self.p6_send_cmd(msg)
                return

            msg.vel_des = [0.0, 0.0, self.exit_turn_vyaw]
            self.p6_send_cmd(msg)

        # ==========================================
        # 12. 降底盘准备冲刺
        # ==========================================
        elif self.state == 'LOWER_BODY_FINAL':
            if self.state_ticks == 1:
                # 依然保持腿距 0.04，只降低高度，保证平稳行走
                self.set_body_height(0.14)

            msg.mode = 11
            msg.gait_id = 1
            msg.vel_des = [0.0, 0.0, 0.0]
            self.p6_send_cmd(msg)

            if state_elapsed > 1.5: # 1.5 秒丝滑降落
                self.get_logger().info("🚜 最终推土机形态就绪！开启三点一线推球冲刺！")
                # 👇 改这里：去执行三点一线推球！
                self.p6_set_state('PUSH_TO_EXIT')
                self.state_ticks = 0

      # ==========================================
        # 12. 终极冲刺：纯视觉三点一线推球入洞
        # ==========================================
        elif self.state == 'PUSH_TO_EXIT':
            if self.state_ticks == 1:
                self.has_seen_exit = False
                self.exit_lost_ticks = 0

            msg.mode = 11
            msg.gait_id = 27  # 用 Trot_Slow 推球最稳
            msg.step_height = [0.02, 0.02]

            exit_visible = (self.exit_offset_norm != -999.0)
            ball_visible = (self.ball_offset_x != -999.0 and self.ball_dist > 0.0)

            vx = self.push_vx
            vy = 0.0
            vyaw = 0.0

            # 【绝技 1：只要看见紫线，就死磕到底绝不停车！】
            if exit_visible:
                self.has_seen_exit = True
                self.exit_lost_ticks = 0
                self.exit_lost_start_time = None
                # 视觉巡线：让狗的头永远对着缺口的中心
                vyaw = - (self.exit_offset_norm * self.push_vyaw_kp)
                vyaw = max(min(vyaw, 0.3), -0.3)

                # 每秒打印一次状态，告诉您它还在死磕紫线
                if self.state_ticks % 10 == 0:
                    self.get_logger().info(f"⛳ 看到紫线！坚决推进中！vyaw: {vyaw:.2f}")
            else:
                # 【消失判定】：如果曾经看到过大门，现在紫线消失了，说明狗头已经穿过大门了！
                if self.has_seen_exit:
                    self.exit_lost_ticks += 1
                    if self.exit_lost_start_time is None:
                        self.exit_lost_start_time = self.now_sec()
                    exit_lost_elapsed = self.now_sec() - self.exit_lost_start_time
                    self.get_logger().info(
                        '⛩️ 紫线消失！冲线判定中... '
                        'elapsed={:.3f}/{:.3f}s'.format(
                            exit_lost_elapsed, self.exit_lost_confirm_s))

            # 门连续消失达到确认时长后，切换到盲走进圈状态。
            exit_lost_elapsed = 0.0 if self.exit_lost_start_time is None else self.now_sec() - self.exit_lost_start_time
            if (self.has_seen_exit and
                    exit_lost_elapsed >= self.exit_lost_confirm_s):
                self.get_logger().info("⛩️ 视野已彻底越过大门！开启盲走冲线，确保进圈！")
                self.p6_set_state('CROSS_FINISH_LINE')
                self.state_ticks = 0
                return

            # 【绝技 2：看球侧滑】如果此时还能看见球，就用侧滑去包抄球！
            if ball_visible:
                vy = - (self.ball_offset_x * self.push_vy_kp)
                vy = max(min(vy, 0.15), -0.15)

            msg.vel_des = [vx, vy, vyaw]
            self.p6_send_cmd(msg)

            # 【超时保底】出口始终不消失或从未识别时，按最终推球总时限结束。
            if state_elapsed > self.push_timeout_s:
                self.get_logger().info("⏱️ 推球超时！强行闭眼冲线！")
                self.p6_set_state('CROSS_FINISH_LINE')
                self.state_ticks = 0

        # ==========================================
        # 14. 进圈冲刺：确保后腿完全越过终点线
        # ==========================================
        elif self.state == 'CROSS_FINISH_LINE':
            msg.mode = 11
            msg.gait_id = 27
            msg.step_height = [0.02, 0.02]

            # 闭着眼睛直直地往前推，不回头！
            msg.vel_des = [self.push_vx, 0.0, 0.0]
            self.p6_send_cmd(msg)

            if state_elapsed >= self.cross_line_time_s:
                self.get_logger().info("🎉 四腿已进圈！完美趴下！")
                self.p6_set_state('MISSION_COMPLETE')
                self.state_ticks = 0

        # ==========================================
        # 15. 任务结束：受控趴下
        # ==========================================
        elif self.state == 'MISSION_COMPLETE':
            msg.mode = 7
            msg.gait_id = 1
            msg.vel_des = [0.0, 0.0, 0.0]
            self.p6_send_cmd(msg)



def main(args=None):
    rclpy.init(args=args)
    node = Stage6Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        try:
            node.destroy_vision_windows()
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
