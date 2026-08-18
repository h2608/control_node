#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各赛段真正共享的基础设施。

只包含 QoS、数学与 TF 工具、StageNodeBase、公共机器人控制命令，以及
图像/深度缓存。赛段专用感知逻辑必须放在对应赛段模块中。
"""

import math
import time
from typing import Optional, Tuple

import cv2

from cv_bridge import CvBridge

from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

try:
    # Gazebo-only support messages.  The physical robot does not need this
    # package merely to start the control node.
    from cyberdog_msg.msg import YamlParam, ApplyForce
except ImportError:
    YamlParam = None
    ApplyForce = None

from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from control_node.robot_interface import create_robot_controller
from control_node.stage_entry import (
    SOURCE_DEFAULT,
    is_default_request,
)


def mission_latched_qos(depth: int = 1) -> QoSProfile:
    """/mission/* 话题使用的 latched QoS（TRANSIENT_LOCAL，晚启动的节点也能收到）。"""
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def mission_signal_qos(depth: int = 10) -> QoSProfile:
    """Non-latched QoS for live lifecycle acknowledgements."""
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def find_contours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    """跨 OpenCV 版本的 findContours：只返回 contours 列表。

    物理机是 **OpenCV 3.2.0**，findContours 返回 (image, contours, hierarchy)
    三个值；仿真容器是 OpenCV 4.2.0，只返回 (contours, hierarchy) 两个值。
    ``contours, _ = cv2.findContours(...)`` 在容器里跑得好好的，上机后第一帧就
    ``ValueError: too many values to unpack (expected 2)``，节点直接死掉 ——
    2026-08-18 第一赛段实测就是这样：站立指令发出去 0.1 s 后节点崩溃。

    ``[-2]`` 在两个版本里都指向 contours。
    """
    return cv2.findContours(mask, mode, method)[-2]


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ControlParameterValueKind:
    kDOUBLE = 1
    kS64 = 2
    kVEC_X_DOUBLE = 3
    kMAT_X_DOUBLE = 4


class StageNodeBase(Node):
    """所有赛段节点的公共基类。

    生命周期：
    1. 待命：启动后声明参数、订阅图像（只缓存原始消息，不做视觉处理）、创建
       控制定时器，但不创建 Robot_Ctrl —— Robot_Ctrl.run() 一旦启动就会以约
       10Hz 持续重发 cmd_msg（心跳），六个赛段节点同时发心跳会互相干扰。
    2. 激活：收到 /mission/active_stage == 本赛段编号后创建并启动 Robot_Ctrl，
       调用 on_activated()（各赛段设置姿态/初始状态并立即发出第一条命令），
       此后每个控制周期调用 stage_control_loop()。
    3. 完成：赛段逻辑到达终点后调用 complete_stage()：先停掉 Robot_Ctrl
       （心跳线程退出），再在 /mission/stage_complete 上发布本赛段编号；
       节点保持存活但不再处理图像和控制。

    人工跳段也必须遵循 ``active_stage=0``、等待 inactive ACK、再发布目标
    赛段的顺序；直接发布另一个非零编号只执行安全中止，不会启动新赛段。
    """

    def __init__(self, node_name: str, stage_id: int):
        super().__init__(node_name)
        self.stage_id = int(stage_id)

        # Platform is the single switch between the Gazebo/LCM backend and
        # the physical ROS 2 motion API.  An explicit use_sim_time override
        # still wins; otherwise sim uses /clock and real uses wall time.
        self.declare_parameter('platform', 'sim')
        self.platform = str(self.get_parameter('platform').value).strip().lower()
        if self.platform not in ('sim', 'real'):
            raise ValueError("platform must be 'sim' or 'real'")
        if 'use_sim_time' not in self._parameter_overrides:
            self.set_parameters([
                Parameter(
                    'use_sim_time', Parameter.Type.BOOL, self.platform == 'sim')
            ])

        # =========================
        # 共享参数：话题 / TF / 控制频率 / 调试
        # =========================
        default_rgb_topic = (
            '/mi_desktop_48_b0_2d_7b_00_e2/image_rgb'
            if self.platform == 'real' else '/rgb_camera/rgb_camera/image_raw'
        )
        default_depth_topic = (
            '/mi_desktop_48_b0_2d_7b_00_e2/camera/depth/image_rect_raw'
            if self.platform == 'real' else '/d435/depth/d435_depth/depth/image_raw'
        )
        # The physical robot also exposes an AI camera.  No current stage
        # consumes it, so keep only the configurable topic here and do not
        # subscribe until a stage actually needs the stream.
        default_ai_camera_topic = (
            '/mi_desktop_48_b0_2d_7b_00_e2/image'
            if self.platform == 'real' else ''
        )
        self.declare_parameter('rgb_topic', default_rgb_topic)
        self.declare_parameter('depth_topic', default_depth_topic)
        self.declare_parameter('ai_camera_topic', default_ai_camera_topic)
        self.declare_parameter('global_frame', 'vodom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('control_hz', 30.0)
        # 拆分后默认关闭调试窗口：七个节点同时开 OpenCV 窗口会互相干扰。
        self.declare_parameter('show_debug_vis', False)
        self.declare_parameter('show_yellow_mask', False)
        self.declare_parameter('mission_active_topic', '/mission/active_stage')
        self.declare_parameter('mission_complete_topic', '/mission/stage_complete')
        self.declare_parameter('mission_inactive_topic', '/mission/stage_inactive')

        # 调试入口：跳过赛段前面的流程，直接从某一段开始（例如第五赛段的
        # ``ramp`` = 上坡段）。取 default/空 时走赛段正常起点。每个赛段用
        # StageEntryTable 声明自己的入口名，见 stage_entry.py。
        # 这个参数只决定状态机的起始状态；机器人本体必须已经被摆到对应位置。
        self.declare_parameter('entry_point', 'default')

        # Physical CyberDog motion API.  Keep these as parameters because the
        # robot namespace contains the individual machine identifier.
        self.declare_parameter('real_motion_servo_cmd_topic', '/motion_servo_cmd')
        self.declare_parameter('real_motion_servo_response_topic', '/motion_servo_response')
        self.declare_parameter('real_motion_result_service', '/motion_result_cmd')
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
        # Temporary legacy gait -> servo mappings used while Stage1-6 are
        # migrated away from mode/gait_id.  The current competition code uses
        # gait 3 and 27 most often; physical locomotion defaults to slow walking (303).
        self.declare_parameter('real_legacy_gait0_motion_id', 303)
        self.declare_parameter('real_legacy_gait1_motion_id', 303)
        self.declare_parameter('real_legacy_gait3_motion_id', 303)
        self.declare_parameter('real_legacy_gait27_motion_id', 303)
        # Stage 6 pushes the ball with legacy gait 28 -> real servo 308
        # (medium speed).  Declared here because the robot's real_robot.yaml
        # sets it, and an undeclared key in a params file fails node startup.
        self.declare_parameter('real_legacy_gait28_motion_id', 308)

        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.ai_camera_topic = str(self.get_parameter('ai_camera_topic').value)
        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.show_yellow_mask = bool(self.get_parameter('show_yellow_mask').value)
        self.mission_active_topic = self.get_parameter('mission_active_topic').value
        self.mission_complete_topic = self.get_parameter('mission_complete_topic').value
        self.mission_inactive_topic = self.get_parameter('mission_inactive_topic').value
        self.entry_point_request = str(self.get_parameter('entry_point').value)
        self.entry_table = None
        self.entry_resolution = None

        # =========================
        # 控制接口：adapter 延迟到激活时创建。self.Ctrl 暂时保留这个名字，
        # 让尚未清理的 Stage4/5/6 旧调用通过兼容层继续工作。
        # =========================
        self.Ctrl = None
        self.robot = None
        self.msg = robot_control_cmd_lcmt()
        if not hasattr(self.msg, 'life_count'):
            self.msg.life_count = 0

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_encoding = None
        self.latest_depth_msg = None
        self.latest_bgr = None
        self.latest_rgb_msg = None

        # 传感器流新鲜度：只有 cv_bridge 成功解码的帧才推进序号。
        # 接收时间使用独立于 ROS /clock 的 monotonic 时钟，因此实体机没有
        # /clock、Gazebo 暂停或 executor 曾被阻塞时，watchdog 仍能看到真实经过时间。
        self.latest_rgb_seq = 0
        self.last_rgb_rx_time_s: Optional[float] = None
        self.latest_depth_seq = 0
        self.last_depth_rx_time_s: Optional[float] = None

        # TF 只作为可选调试/兼容信息使用。主状态机不因为 TF 不可用而停止。
        self.last_known_pose: Optional[Tuple[float, float, float]] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_w = 640
        self.rgb_h = 480

        self.state = 'IDLE'
        self.active = False
        self.finished = False
        # Existing nodes must observe the neutral stage 0 between nonzero
        # stages.  This prevents a direct 1 -> 2 publication from starting the
        # new LCM owner before the old owner has acknowledged deactivation.
        # A newly launched node remains armed so it can join the current stage.
        self.activation_armed = True
        self.timed_turn_start_time_sec = None

        # Gazebo-only YAML/apply_force channels.  Do not require or create
        # these publishers on the physical robot.
        self.yaml_node = self
        self.para_pub = None
        self.force_pub = None
        if self.platform == 'sim':
            if YamlParam is None or ApplyForce is None:
                raise ImportError(
                    'cyberdog_msg is required by the simulator backend')
            self.para_pub = self.create_publisher(YamlParam, 'yaml_parameter', 10)
            self.force_pub = self.create_publisher(ApplyForce, 'apply_force', 10)

        # =========================
        # 任务控制话题
        # =========================
        self.mission_active_sub = self.create_subscription(
            Int32, self.mission_active_topic, self._mission_cb, mission_latched_qos(1))
        self.stage_complete_pub = self.create_publisher(
            Int32, self.mission_complete_topic, mission_latched_qos(6))
        self.stage_inactive_pub = self.create_publisher(
            Int32, self.mission_inactive_topic, mission_signal_qos(10))

        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)

        self.control_timer = self.create_timer(1.0 / self.control_hz, self._control_timer_cb)

        self.get_logger().info(
            f'{node_name} started, waiting for {self.mission_active_topic} == {self.stage_id}')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')
        if self.ai_camera_topic:
            self.get_logger().info(
                f'ai_camera_topic={self.ai_camera_topic} (reserved, not subscribed)')
        self.get_logger().info(f'tf: {self.global_frame} -> {self.base_frame}')
        self.get_logger().info(f'platform={self.platform}, use_sim_time={self.get_parameter("use_sim_time").value}')

    # ============================================================
    # 任务控制：激活 / 停用 / 完成
    # ============================================================
    def _mission_cb(self, msg: Int32):
        stage = int(msg.data)
        if stage == 0:
            self.activation_armed = True
            if self.active:
                self._deactivate('active_stage=0', publish_ack=True)
            elif not self.finished:
                self._publish_inactive_ack()
            return

        if stage == self.stage_id:
            if not self.active and not self.finished:
                if not self.activation_armed:
                    self.get_logger().error(
                        f'[MISSION] reject unsafe direct activation of stage {self.stage_id}; '
                        'publish active_stage=0 before switching stages'
                    )
                    return
                self.activation_armed = False
                self._activate()
        else:
            self.activation_armed = False
            if self.active:
                self._deactivate(f'active_stage={stage}', publish_ack=True)

    def _activate(self):
        self.get_logger().info(f'[MISSION] stage {self.stage_id} activated')
        self.start_ctrl()
        self.active = True
        try:
            self.on_activated()
        except Exception:
            self.active = False
            self.stop_ctrl()
            raise

    def _deactivate(self, reason: str = '', publish_ack: bool = False):
        # 任务控制切走本赛段（人工跳段/中止）：停心跳，让接管的赛段独占命令通道。
        self.get_logger().warn(f'[MISSION] stage {self.stage_id} deactivated ({reason})')
        self.active = False
        self.stop_ctrl()
        if publish_ack:
            self._publish_inactive_ack()

    def _publish_inactive_ack(self):
        """Confirm that this stage owns no live robot-control backend and is armed."""
        out = Int32()
        out.data = self.stage_id
        self.stage_inactive_pub.publish(out)
        self.get_logger().info(
            f'[MISSION] stage {self.stage_id} inactive acknowledgement published',
            throttle_duration_sec=1.0)

    def start_ctrl(self):
        if self.Ctrl is None:
            self.Ctrl = create_robot_controller(self)
            self.robot = self.Ctrl
            self.Ctrl.run()
            self.get_logger().info(
                f'[MISSION] robot backend started: {self.Ctrl.backend_name}')

    def stop_ctrl(self):
        if self.Ctrl is not None:
            try:
                self.Ctrl.quit()
            except Exception as e:
                self.get_logger().warn(f'[MISSION] robot backend quit failed: {e}')
            self.Ctrl = None
            self.robot = None
            self.get_logger().info('[MISSION] robot backend stopped')

    def complete_stage(self, reason: str = ''):
        if self.finished:
            return
        self.finished = True
        self.active = False
        suffix = f': {reason}' if reason else ''
        self.get_logger().info(f'[MISSION] stage {self.stage_id} complete{suffix}')
        # 先停心跳线程，再发布完成消息：避免本赛段残留命令和下一赛段抢通道。
        self.stop_ctrl()
        out = Int32()
        out.data = self.stage_id
        self.stage_complete_pub.publish(out)

    # ============================================================
    # 调试入口
    # ============================================================
    def resolve_stage_entry(self, table, legacy_request=None):
        """解析本赛段的调试入口，返回要进入的状态名。

        优先级：
        1. 统一的 ``entry_point`` 参数（launch 里的 ``stage<N>_entry``）；
        2. 赛段原有的 ``*_initial_state`` 参数（直接写状态名，保持兼容）；
        3. 赛段正常起点。

        入口名非法时不抛异常：回退到正常起点并把整张入口表打进日志，这样写错
        一个调试参数不会变成一个起不来的节点。
        """
        requested = self.entry_point_request
        if is_default_request(requested):
            requested = legacy_request

        resolution = table.resolve(requested)
        self.entry_table = table
        self.entry_resolution = resolution

        if not resolution.ok:
            self.get_logger().error('[ENTRY] ' + resolution.message)
            for line in table.describe():
                self.get_logger().error('[ENTRY] ' + line)
        elif resolution.source == SOURCE_DEFAULT:
            self.get_logger().info('[ENTRY] ' + resolution.message)
            self.get_logger().info('[ENTRY] ' + table.summary())
        else:
            # 非默认入口一定是调试运行：用 warn 让忘记复位的参数足够显眼。
            self.get_logger().warn('[ENTRY] ' + resolution.message)
            self.get_logger().warn(
                '[ENTRY] DEBUG START: the robot must already be placed at this '
                'point of the course; earlier states are skipped')
            for requirement in resolution.requires:
                self.get_logger().warn('[ENTRY] requires: ' + requirement)

        return resolution.state

    def on_activated(self):
        """赛段被激活时调用：设置姿态/初始状态。子类实现。"""
        pass

    def stage_control_loop(self):
        """每个控制周期调用（仅激活且未完成时）。子类实现。"""
        pass

    def _control_timer_cb(self):
        if self.active and not self.finished:
            self.stage_control_loop()

    # ============================================================
    # 图像回调：未激活时只缓存原始消息，不做 cv_bridge 转换
    # ============================================================
    def rgb_callback(self, msg: Image):
        self.latest_rgb_msg = msg
        if not self.active or self.finished:
            return
        self.handle_rgb_msg(msg)

    def handle_rgb_msg(self, msg: Image):
        """默认实现：转换并缓存 BGR 图像，再调用 on_rgb_frame()。

        第五/六赛段的视觉入口直接消费原始消息，重载本方法。
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return
        self.record_valid_rgb_frame(msg, frame)
        self.on_rgb_frame(frame)

    def record_valid_rgb_frame(self, msg: Image, frame):
        """Record one successfully decoded RGB frame."""
        self.latest_rgb_msg = msg
        self.latest_bgr = frame
        self.latest_rgb_seq += 1
        self.last_rgb_rx_time_s = time.monotonic()

    def on_rgb_frame(self, frame):
        pass

    def depth_callback(self, msg: Image):
        if not self.active or self.finished:
            self.latest_depth_msg = msg
            return
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'depth convert failed: {e}')
            return
        self.latest_depth_msg = msg
        self.latest_depth = depth_img
        self.latest_depth_encoding = msg.encoding
        self.latest_depth_seq += 1
        self.last_depth_rx_time_s = time.monotonic()
        self.on_depth_frame(depth_img, msg)

    def on_depth_frame(self, depth_img, msg: Image):
        """Offer decoded depth to stages that override this hook."""
        pass

    def rgb_age_s(self) -> Optional[float]:
        """Return monotonic age of the last decoded RGB frame."""
        if self.last_rgb_rx_time_s is None:
            return None
        return max(0.0, time.monotonic() - self.last_rgb_rx_time_s)

    def depth_age_s(self) -> Optional[float]:
        """Return monotonic age of the last decoded depth frame."""
        if self.last_depth_rx_time_s is None:
            return None
        return max(0.0, time.monotonic() - self.last_depth_rx_time_s)

    # ============================================================
    # YAML 参数 / 施力发布（原 FullCompetitionNode.publish_*）
    # ============================================================
    def publish_yaml_kDOUBLE(self, name: str, value: float, is_user: int = 0):
        if self.para_pub is None or YamlParam is None:
            self.get_logger().warn(
                '[SIM_ONLY] yaml_parameter ignored on physical robot',
                throttle_duration_sec=2.0)
            return
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kDOUBLE
        msg.s64_value = float(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_yaml_s64(self, name: str, value: int, is_user: int = 0):
        if self.para_pub is None or YamlParam is None:
            self.get_logger().warn(
                '[SIM_ONLY] yaml_parameter ignored on physical robot',
                throttle_duration_sec=2.0)
            return
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kS64
        msg.s64_value = int(value)
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_yaml_vecxd(self, name: str, values, is_user: int = 1):
        if self.para_pub is None or YamlParam is None:
            self.get_logger().warn(
                '[SIM_ONLY] yaml_parameter ignored on physical robot',
                throttle_duration_sec=2.0)
            return
        msg = YamlParam()
        msg.name = name
        msg.kind = ControlParameterValueKind.kVEC_X_DOUBLE
        vec = [0.0] * 12
        for i, v in enumerate(values):
            if i < 12:
                vec[i] = float(v)
        msg.vecxd_value = vec
        msg.is_user = int(is_user)
        self.para_pub.publish(msg)

    def publish_apply_force(self, link_name: str, rel_pos, force, duration: float):
        if self.platform == 'real':
            self.get_logger().warn(
                '[SIM_ONLY] apply_force ignored on physical robot',
                throttle_duration_sec=2.0)
            return
        if self.force_pub is None or ApplyForce is None:
            self.get_logger().warn(
                '[SIM_ONLY] apply_force publisher unavailable',
                throttle_duration_sec=2.0)
            return
        msg = ApplyForce()
        msg.link_name = link_name
        msg.rel_pos = [float(x) for x in rel_pos]
        msg.force = [float(x) for x in force]
        msg.time = float(duration)
        self.force_pub.publish(msg)

    # ============================================================
    # 基础工具
    # ============================================================
    def planar_distance(self, pose0: Tuple[float, float, float], pose1: Tuple[float, float, float]) -> float:
        x0, y0, _ = pose0
        x1, y1, _ = pose1
        return math.hypot(x1 - x0, y1 - y0)

    def local_lateral_displacement(self, start_pose: Tuple[float, float, float],
                                   current_pose: Tuple[float, float, float]) -> float:
        """
        计算 current_pose 相对 start_pose 的横向位移。
        返回值 > 0 表示相对 start_pose 的朝向向左移动；< 0 表示向右移动。
        这样可以避免把前后方向的漂移算进横移距离。
        """
        sx, sy, syaw = start_pose
        cx, cy, _ = current_pose
        dx = cx - sx
        dy = cy - sy
        return -math.sin(syaw) * dx + math.cos(syaw) * dy

    def apply_min_abs_velocity(self, v: float, v_min: float, deadband: float = 0.0) -> float:
        if abs(v) <= deadband:
            return 0.0
        if 0.0 < abs(v) < v_min:
            return math.copysign(v_min, v)
        return v

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    # ============================================================
    # 控制
    # ============================================================
    def send_stop_command(self):
        if self.Ctrl is None:
            return
        self.Ctrl.stop_motion()
        self.get_logger().info('[CMD] STOP', throttle_duration_sec=1.0)

    def send_velocity_command(self, vx: float, vy: float, wz: float, step_height: float = 0.02):
        if self.Ctrl is None:
            return
        self.Ctrl.move(
            float(vx), float(vy), float(wz),
            step_height=float(step_height),
            legacy_gait_id=3,
        )
        self.get_logger().info(
            f'[CMD] vel_des=[{vx:.3f}, {vy:.3f}, {wz:.3f}]',
            throttle_duration_sec=0.3
        )

    def send_left_jump_action_once(self):
        if self.Ctrl is None:
            return
        self.Ctrl.run_action('left_jump', wait_finish=True)
        self.Ctrl.recovery_stand(wait_finish=True)

    def execute_left_jump_turn(self, jump_count: int, next_state: str):
        for _ in range(jump_count):
            self.send_left_jump_action_once()
        self.set_state(next_state)

    def execute_timed_turn(self, wz: float, duration_sec: float, next_state: str) -> bool:
        """
        用固定角速度 + 固定仿真时间转向，代替原来的原地左跳。
        不发送 STOP，转完后直接切换到 next_state。
        """
        now = self.now_sec()

        if self.timed_turn_start_time_sec is None:
            self.timed_turn_start_time_sec = now
            self.get_logger().info(
                f'[TIMED_TURN] start: wz={wz:.3f}, duration={duration_sec:.2f}s, next={next_state}'
            )

        self.timed_turn_start_time_sec = self.align_motion_timer_start(
            self.timed_turn_start_time_sec, now)
        elapsed = max(0.0, now - self.timed_turn_start_time_sec)

        if elapsed >= duration_sec:
            self.get_logger().info(
                f'[TIMED_TURN] done: elapsed={elapsed:.2f}s, next={next_state}'
            )
            self.timed_turn_start_time_sec = None
            self.set_state(next_state)
            return True

        self.send_velocity_command(
            0.0,
            0.0,
            wz,
            step_height=self.timed_turn_step_height
        )
        return True

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def align_motion_timer_start(self, start_s: Optional[float], now_s: Optional[float] = None):
        """Align a timed-motion start to the latest real Servo handoff.

        On the physical backend a state may be entered before its first move()
        finishes START -> ACK.  The adapter publishes a node-clock anchor for
        that handoff (and for failed retries / ResultCmd boundaries).  Moving
        the timer start forward to this anchor prevents BUSY/ACK latency from
        consuming a fixed-duration forward/turn/shift command.
        """
        now = self.now_sec() if now_s is None else float(now_s)
        start = now if start_s is None else float(start_s)
        ctrl = getattr(self, 'Ctrl', None)
        if ctrl is not None and getattr(ctrl, 'is_real', False):
            try:
                anchor = ctrl.get_motion_timer_anchor_node_time_s()
            except Exception:
                anchor = None
            if anchor is not None and float(anchor) > start:
                start = float(anchor)
        return start

    # ============================================================
    # TF
    # ============================================================
    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        return (t.x, t.y, yaw)
