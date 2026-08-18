#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务控制节点：负责六个赛段节点的总调度。

职责：
1. 按 stage_sequence 顺序激活各赛段节点（发布 /mission/active_stage）；
2. 监听各赛段的完成消息（/mission/stage_complete）；
3. 收到当前赛段完成后切换到下一赛段；全部完成后发布 0 并输出总结。

本节点不做任何 LCM / 视觉处理；机器人控制完全由各赛段节点负责。
激活话题使用 TRANSIENT_LOCAL（latched）QoS 并周期重发，
因此赛段节点晚启动也能收到当前激活状态。
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Int32, String


def latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def signal_qos(depth: int = 10) -> QoSProfile:
    """Reliable, non-latched QoS for live lifecycle acknowledgements."""
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')

        self.declare_parameter('platform', 'sim')
        self.platform = str(self.get_parameter('platform').value).strip().lower()
        if self.platform not in ('sim', 'real'):
            raise ValueError("platform must be 'sim' or 'real'")
        # Sim follows Gazebo /clock; the physical robot uses wall time.
        # An explicit launch/CLI use_sim_time override still wins.
        if 'use_sim_time' not in self._parameter_overrides:
            self.set_parameters([
                Parameter(
                    'use_sim_time', Parameter.Type.BOOL, self.platform == 'sim')
            ])

        self.declare_parameter('stage_sequence', [1, 2, 3, 4, 5, 6])
        # Start the mission from any stage while preserving the remaining order.
        # Example: start_stage=4 -> sequence becomes [4, 5, 6].
        self.declare_parameter('start_stage', 1)
        self.declare_parameter('single_stage', False)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('start_delay_sec', 2.0)
        self.declare_parameter('republish_period_sec', 1.0)
        # 单赛段超时（秒）。0 表示不启用超时。
        self.declare_parameter('stage_timeout_sec', 0.0)
        self.declare_parameter('force_advance_on_timeout', False)
        self.declare_parameter('deactivate_timeout_sec', 15.0)
        self.declare_parameter('shutdown_on_complete', False)
        self.declare_parameter('mission_active_topic', '/mission/active_stage')
        self.declare_parameter('mission_complete_topic', '/mission/stage_complete')
        self.declare_parameter('mission_inactive_topic', '/mission/stage_inactive')
        # Mission control writes its own markers into the shared diagnostic
        # event stream so ``ros2 topic echo /mission/diag/event`` shows the
        # startup order of all seven control processes in one place.
        self.declare_parameter('diag_event_topic_enabled', True)
        self.declare_parameter('diag_event_topic', '/mission/diag/event')

        # Physical-robot startup barrier.  This is owned by MissionControl so
        # it runs exactly once before whichever stage is selected as the start.
        self.declare_parameter('startup_recovery_enabled', True)
        self.declare_parameter('startup_recovery_motion_id', 111)
        self.declare_parameter('startup_recovery_timeout_sec', 30.0)
        self.declare_parameter('startup_recovery_settle_sec', 0.0)
        self.declare_parameter(
            'real_motion_result_service',
            '/mi_desktop_48_b0_2d_7b_00_e2/motion_result_cmd')
        self.declare_parameter('real_cmd_source', 0)

        self.stage_sequence = [int(s) for s in self.get_parameter('stage_sequence').value]
        if (
            not self.stage_sequence
            or any(stage < 1 or stage > 6 for stage in self.stage_sequence)
            or len(set(self.stage_sequence)) != len(self.stage_sequence)
        ):
            raise ValueError(
                'stage_sequence must be a non-empty list of unique stage ids in [1, 6]')

        self.start_stage = int(self.get_parameter('start_stage').value)
        if self.start_stage not in self.stage_sequence:
            raise ValueError(
                'start_stage must be present in stage_sequence, got {}'.format(
                    self.start_stage))
        self.single_stage = bool(self.get_parameter('single_stage').value)
        if self.single_stage:
            self.stage_sequence = [self.start_stage]
        else:
            self.stage_sequence = self.stage_sequence[
                self.stage_sequence.index(self.start_stage):]

        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.start_delay_sec = float(self.get_parameter('start_delay_sec').value)
        self.republish_period_sec = float(self.get_parameter('republish_period_sec').value)
        self.stage_timeout_sec = float(self.get_parameter('stage_timeout_sec').value)
        self.force_advance_on_timeout = bool(self.get_parameter('force_advance_on_timeout').value)
        self.deactivate_timeout_sec = float(self.get_parameter('deactivate_timeout_sec').value)
        self.shutdown_on_complete = bool(self.get_parameter('shutdown_on_complete').value)
        self.mission_active_topic = self.get_parameter('mission_active_topic').value
        self.mission_complete_topic = self.get_parameter('mission_complete_topic').value
        self.mission_inactive_topic = self.get_parameter('mission_inactive_topic').value

        self.startup_recovery_enabled = bool(
            self.get_parameter('startup_recovery_enabled').value)
        self.startup_recovery_motion_id = int(
            self.get_parameter('startup_recovery_motion_id').value)
        self.startup_recovery_timeout_sec = max(
            1.0, float(self.get_parameter('startup_recovery_timeout_sec').value))
        self.startup_recovery_settle_sec = max(
            0.0, float(self.get_parameter('startup_recovery_settle_sec').value))
        self.real_motion_result_service = str(
            self.get_parameter('real_motion_result_service').value)
        self.real_cmd_source = int(self.get_parameter('real_cmd_source').value)

        self.diag_event_pub = None
        if bool(self.get_parameter('diag_event_topic_enabled').value):
            self.diag_event_pub = self.create_publisher(
                String,
                str(self.get_parameter('diag_event_topic').value),
                signal_qos(20))
        self._last_published_active = None

        self.active_pub = self.create_publisher(Int32, self.mission_active_topic, latched_qos(1))
        self.complete_sub = self.create_subscription(
            Int32, self.mission_complete_topic, self.stage_complete_callback, latched_qos(6))
        self.inactive_sub = self.create_subscription(
            Int32, self.mission_inactive_topic, self.stage_inactive_callback, signal_qos(10))

        self.seq_index = 0
        self.mission_started = False
        self.mission_done = False
        self.timeout_reported = False
        # /clock 就绪后第一次 tick 的仿真时间，用作 start_delay 的基准。
        self.start_ref_time = None
        self.stage_start_time = None
        self.stage_durations = []
        self.pending_force_advance = None
        self.pending_activation = None

        # Startup recovery action state.  Only MissionControl owns this client;
        # the stage nodes remain the sole owners of continuous Servo control.
        self.startup_recovery_client = None
        self._motion_result_srv_type = None
        self.startup_recovery_future = None
        self.startup_recovery_started_at = None
        self.startup_recovery_complete = False
        self.startup_recovery_failed = False
        self.startup_recovery_settle_until = None
        if self.platform == 'real' and self.startup_recovery_enabled:
            # Lazy import keeps the Gazebo/sim package usable on machines
            # that do not have CyberDog's protocol package installed.
            from protocol.srv import MotionResultCmd
            self._motion_result_srv_type = MotionResultCmd
            self.startup_recovery_client = self.create_client(
                MotionResultCmd, self.real_motion_result_service)

        self.timer = self.create_timer(max(0.1, self.republish_period_sec), self.tick)

        self.get_logger().info(
            f'mission control started: sequence={self.stage_sequence}, '
            f'auto_start={self.auto_start}, start_delay={self.start_delay_sec:.1f}s, '
            f'platform={self.platform}, start_stage={self.start_stage}, '
            f'single_stage={self.single_stage}, '
            f'startup_recovery={self.startup_recovery_enabled}'
        )
        self.note_event(
            'MISSION_NODE_READY',
            'auto_start={} startup_recovery={} sequence={}'.format(
                self.auto_start, self.startup_recovery_enabled,
                self.stage_sequence))
        if not self.auto_start:
            self.get_logger().warn(
                '[DIAG_MODE] auto_start=False: no stage will be activated; '
                'every stage node stays idle')

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def current_stage(self):
        if self.seq_index >= len(self.stage_sequence):
            return None
        return self.stage_sequence[self.seq_index]

    def note_event(self, event: str, detail: str = ''):
        """Emit one lifecycle marker on the shared diagnostic event stream."""
        line = '[RXEVENT] node=mission_control t={:.3f} ev={}'.format(
            time.monotonic(), event)
        if detail:
            line += ' detail={}'.format(detail)
        self.get_logger().warn(line)
        if self.diag_event_pub is not None:
            out = String()
            out.data = line
            try:
                self.diag_event_pub.publish(out)
            except Exception:
                pass

    def publish_active(self, stage: int):
        msg = Int32()
        msg.data = int(stage)
        self.active_pub.publish(msg)
        # tick() republishes the same value every period; only the transitions
        # are worth a marker.
        if self._last_published_active != int(stage):
            self._last_published_active = int(stage)
            self.note_event('PUBLISH_ACTIVE_STAGE', 'stage={}'.format(int(stage)))

    @staticmethod
    def _set_request_field_if_present(request, name, value):
        if hasattr(request, name):
            setattr(request, name, value)

    def _fail_startup_recovery(self, message: str):
        if self.startup_recovery_failed:
            return
        self.startup_recovery_failed = True
        self.publish_active(0)
        self.get_logger().error(
            '[MISSION] startup recovery failed; mission will NOT start: {}'.format(
                message))

    def _tick_startup_recovery(self, now: float) -> bool:
        """Return True once the physical startup barrier is fully complete."""
        if self.platform != 'real' or not self.startup_recovery_enabled:
            return True

        if self.startup_recovery_failed:
            self.publish_active(0)
            return False

        if self.startup_recovery_complete:
            if self.startup_recovery_settle_until is None:
                self.startup_recovery_settle_until = (
                    now + self.startup_recovery_settle_sec)
                self.get_logger().info(
                    '[MISSION] recovery stand complete; settling for {:.1f}s'.format(
                        self.startup_recovery_settle_sec))
            if now < self.startup_recovery_settle_until:
                return False
            return True

        if self.startup_recovery_client is None:
            self._fail_startup_recovery('MotionResultCmd client was not created')
            return False

        if self.startup_recovery_future is None:
            if not self.startup_recovery_client.service_is_ready():
                self.get_logger().info(
                    '[MISSION] waiting for startup recovery service: {}'.format(
                        self.real_motion_result_service),
                    throttle_duration_sec=2.0)
                return False

            request = self._motion_result_srv_type.Request()
            if not hasattr(request, 'motion_id'):
                self._fail_startup_recovery(
                    "MotionResultCmd.Request has no 'motion_id' field")
                return False

            request.motion_id = int(self.startup_recovery_motion_id)
            self._set_request_field_if_present(
                request, 'cmd_source', int(self.real_cmd_source))
            self._set_request_field_if_present(request, 'vel_des', [0.0, 0.0, 0.0])
            self._set_request_field_if_present(request, 'rpy_des', [0.0, 0.0, 0.0])
            self._set_request_field_if_present(request, 'pos_des', [0.0, 0.0, 0.0])
            self._set_request_field_if_present(request, 'acc_des', [0.0, 0.0, 0.0])
            self._set_request_field_if_present(request, 'ctrl_point', [0.0, 0.0, 0.0])
            self._set_request_field_if_present(request, 'foot_pose', [0.0, 0.0, 0.0])
            self._set_request_field_if_present(request, 'step_height', [0.05, 0.05])
            self._set_request_field_if_present(request, 'duration', 0)

            try:
                self.startup_recovery_future = (
                    self.startup_recovery_client.call_async(request))
            except Exception as exc:
                self._fail_startup_recovery(
                    'service call raised: {}'.format(exc))
                return False

            self.startup_recovery_started_at = now
            self.get_logger().info(
                '[MISSION] startup: sent recovery stand motion_id={} ONCE'.format(
                    self.startup_recovery_motion_id))
            self.note_event('STARTUP_RECOVERY_SENT', 'motion_id={}'.format(
                self.startup_recovery_motion_id))
            return False

        if (
            self.startup_recovery_started_at is not None
            and now - self.startup_recovery_started_at
            > self.startup_recovery_timeout_sec
        ):
            self._fail_startup_recovery(
                'timeout after {:.1f}s'.format(self.startup_recovery_timeout_sec))
            return False

        if not self.startup_recovery_future.done():
            return False

        try:
            response = self.startup_recovery_future.result()
            success = bool(response.result)
            code = int(response.code)
            returned_motion_id = int(response.motion_id)
        except Exception as exc:
            self._fail_startup_recovery(
                'result future raised: {}'.format(exc))
            return False

        if not success:
            self._fail_startup_recovery(
                'robot returned result=False, motion_id={}, code={}'.format(
                    returned_motion_id, code))
            return False

        self.startup_recovery_complete = True
        self.get_logger().info(
            '[MISSION] startup recovery success: response_motion_id={} code={}'.format(
                returned_motion_id, code))
        self.note_event('STARTUP_RECOVERY_DONE', 'code={}'.format(code))
        return False

    def tick(self):
        if self.mission_done:
            # 保持发布 0，方便晚启动的观察者/节点确认任务已结束。
            self.publish_active(0)
            return

        if not self.auto_start:
            return

        now = self.now_sec()
        if now <= 0.0:
            # /clock 尚未就绪（use_sim_time=True 时 Gazebo 未启动）。
            return

        if self.start_ref_time is None:
            self.start_ref_time = now
            # Establish the neutral barrier before the first nonzero stage.
            self.publish_active(0)
            return

        if not self.mission_started:
            if now - self.start_ref_time < self.start_delay_sec:
                return

            # The physical robot may start prone.  Execute RecoveryStand once
            # and wait for the robot-side service result before any stage owns
            # the continuous Servo command channel.
            if not self._tick_startup_recovery(now):
                return

            self.mission_started = True
            stage = self.current_stage()
            if stage is None:
                self.finish_mission()
                return
            self.request_activation(stage, is_start=True)
            return

        if self.pending_force_advance is not None:
            self.publish_active(0)
            pending_elapsed = now - self.pending_force_advance['requested_at']
            if (
                self.deactivate_timeout_sec > 0.0
                and pending_elapsed > self.deactivate_timeout_sec
                and not self.pending_force_advance['timeout_reported']
            ):
                self.pending_force_advance['timeout_reported'] = True
                self.get_logger().error(
                    f'[MISSION] stage {self.pending_force_advance["stage"]} did not '
                    f'acknowledge deactivation within {self.deactivate_timeout_sec:.1f}s; '
                    'fail safe: next stage will not be activated'
                )
            return

        if self.pending_activation is not None:
            self.publish_active(0)
            pending_elapsed = now - self.pending_activation['requested_at']
            if (
                self.deactivate_timeout_sec > 0.0
                and pending_elapsed > self.deactivate_timeout_sec
                and not self.pending_activation['timeout_reported']
            ):
                self.pending_activation['timeout_reported'] = True
                self.get_logger().error(
                    f'[MISSION] stage {self.pending_activation["stage"]} did not '
                    f'acknowledge the neutral barrier within '
                    f'{self.deactivate_timeout_sec:.1f}s; '
                    'fail safe: stage will not be activated'
                )
            return

        stage = self.current_stage()
        if stage is None:
            self.finish_mission()
            return

        self.publish_active(stage)

        # 超时监控：只报警；force_advance_on_timeout=True 时才强制切下一赛段。
        if self.stage_timeout_sec > 0.0 and self.stage_start_time is not None:
            elapsed = now - self.stage_start_time
            if elapsed > self.stage_timeout_sec:
                if not self.timeout_reported:
                    self.timeout_reported = True
                    self.get_logger().error(
                        f'[MISSION] stage {stage} timeout: '
                        f'{elapsed:.1f}s > {self.stage_timeout_sec:.1f}s'
                    )
                if self.force_advance_on_timeout:
                    self.begin_force_advance(stage, elapsed)
                    return

    def stage_complete_callback(self, msg: Int32):
        stage = int(msg.data)
        current = self.current_stage()

        if not self.mission_started:
            # latched 队列里残留的上一次运行的完成消息：任务未开始前全部忽略。
            self.get_logger().warn(f'[MISSION] ignore completion of stage {stage} before mission start')
            return

        if self.mission_done or current is None:
            self.get_logger().warn(f'[MISSION] ignore completion of stage {stage}: mission already done')
            return

        if stage != current:
            self.get_logger().warn(
                f'[MISSION] ignore completion of stage {stage}: current stage is {current}'
            )
            return

        now = self.now_sec()
        elapsed = 0.0 if self.stage_start_time is None else now - self.stage_start_time
        self.get_logger().info(f'[MISSION] stage {stage} complete (took {elapsed:.1f}s)')
        how = 'done_after_timeout' if self.pending_force_advance is not None else 'done'
        self.stage_durations.append((stage, elapsed, how))
        self.pending_force_advance = None
        self.advance()

    def begin_force_advance(self, stage: int, elapsed: float):
        if self.pending_force_advance is not None:
            return
        self.pending_force_advance = {
            'stage': int(stage),
            'elapsed': float(elapsed),
            'requested_at': self.now_sec(),
            'timeout_reported': False,
        }
        self.get_logger().error(
            f'[MISSION] request safe force-advance past stage {stage}: '
            'publishing neutral stage 0 and waiting for inactive acknowledgement'
        )
        self.publish_active(0)

    def stage_inactive_callback(self, msg: Int32):
        stage = int(msg.data)
        pending = self.pending_force_advance
        if pending is not None and stage == pending['stage']:
            actual_elapsed = (
                pending['elapsed']
                if self.stage_start_time is None
                else max(0.0, self.now_sec() - self.stage_start_time)
            )
            self.get_logger().warn(
                f'[MISSION] stage {stage} acknowledged inactive; '
                'safe to prepare the next stage')
            self.stage_durations.append((stage, actual_elapsed, 'timeout'))
            self.pending_force_advance = None
            self.advance()
            return

        pending = self.pending_activation
        if pending is None or stage != pending['stage']:
            return

        self.pending_activation = None
        self.stage_start_time = self.now_sec()
        prefix = 'start: ' if pending['is_start'] else ''
        self.get_logger().info(
            f'[MISSION] {prefix}activating stage {stage} '
            f'({self.seq_index + 1}/{len(self.stage_sequence)})'
        )
        self.publish_active(stage)

    def advance(self):
        self.seq_index += 1
        self.timeout_reported = False
        nxt = self.current_stage()
        if nxt is None:
            self.finish_mission()
            return
        self.request_activation(nxt)

    def request_activation(self, stage: int, is_start: bool = False):
        # active_stage uses KEEP_LAST(1), so a back-to-back 0 -> N publish can
        # hide the neutral sample from a busy reader.  Wait until the target
        # stage explicitly confirms that it processed 0 and has no LCM owner.
        self.pending_activation = {
            'stage': int(stage),
            'requested_at': self.now_sec(),
            'timeout_reported': False,
            'is_start': bool(is_start),
        }
        self.get_logger().info(
            f'[MISSION] preparing stage {stage}: publish neutral barrier and '
            'wait for target acknowledgement'
        )
        self.publish_active(0)

    def finish_mission(self):
        if self.mission_done:
            return
        self.mission_done = True
        self.publish_active(0)
        summary = ', '.join(f'stage {s}: {t:.1f}s ({how})' for s, t, how in self.stage_durations)
        self.get_logger().info(f'[MISSION] all stages complete. {summary}')
        if self.shutdown_on_complete:
            self.get_logger().info('[MISSION] shutdown_on_complete=True; shutting down')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MissionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
