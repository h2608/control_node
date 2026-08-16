#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Optional

import rclpy
from rclpy.node import Node

from control_node.my_gait import Robot_Ctrl
from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt


class LcmPoseTestNode(Node):
    """
    只通过 LCM 测试：
    1. pos_des[2] 是否能够控制机身高度；
    2. rpy_des[0] 是否能够控制 roll；
    3. 参数为 None 时，本地是否会继续发送之前保存的姿态值。
    """

    def __init__(self):
        super().__init__('lcm_pose_test_node')

        # 与第四、第五赛段当前参数保持接近
        self.declare_parameter('normal_height', 0.25)
        self.declare_parameter('low_height', 0.15)
        self.declare_parameter('roll_angle', 0.30)
        self.declare_parameter('phase_seconds', 4.0)
        self.declare_parameter('step_height', 0.05)

        self.normal_height = float(
            self.get_parameter('normal_height').value
        )
        self.low_height = float(
            self.get_parameter('low_height').value
        )
        self.roll_angle = float(
            self.get_parameter('roll_angle').value
        )
        self.phase_seconds = float(
            self.get_parameter('phase_seconds').value
        )
        self.step_height = float(
            self.get_parameter('step_height').value
        )

        # 软件侧保存的当前姿态
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_body_height = self.normal_height

        self.ctrl = Robot_Ctrl()
        self.msg = robot_control_cmd_lcmt()
        self.ctrl.run()

        self.phase_index = -1
        self.phase_start_time = time.monotonic()
        self.test_finished = False

        self.get_logger().info(
            'LCM 姿态测试启动：'
            f'normal_height={self.normal_height:.3f}, '
            f'low_height={self.low_height:.3f}, '
            f'roll_angle={self.roll_angle:.3f}'
        )

        # 先让机器狗执行一次恢复站立
        self.recovery_stand(wait_finish=True)

        # 定时推进测试阶段
        self.timer = self.create_timer(0.05, self.timer_callback)

    def inc_life_count(self):
        self.msg.life_count += 1

        if self.msg.life_count > 127:
            self.msg.life_count = 1

    def recovery_stand(self, wait_finish: bool = True):
        """
        mode=12, gait_id=0：
        执行底层恢复站立。
        """

        self.msg.mode = 12
        self.msg.gait_id = 0

        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.msg.pos_des = [0.0, 0.0, 0.0]
        self.msg.step_height = [0.0, 0.0]

        self.inc_life_count()
        self.ctrl.Send_cmd(self.msg)

        self.get_logger().info(
            f'[RECOVERY] mode=12, gait=0, '
            f'life_count={self.msg.life_count}'
        )

        if wait_finish:
            finished = self.ctrl.Wait_finish(12, 0)

            if finished:
                self.get_logger().info('[RECOVERY] 起立动作完成')
            else:
                self.get_logger().warn(
                    '[RECOVERY] 未收到完成响应，继续测试'
                )

        # 软件侧同步回正常姿态
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_body_height = self.normal_height

    def send_motion_cmd(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        *,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        body_height: Optional[float] = None,
    ):
        """
        mode=11, gait_id=3 下发送连续控制命令。

        注意：
        None 不会发送给 LCM。
        None 只表示“不修改软件侧保存的旧值”。
        实际发送出去的仍然是具体 float 数值。
        """

        if roll is not None:
            self.current_roll = float(roll)

        if pitch is not None:
            self.current_pitch = float(pitch)

        if body_height is not None:
            self.current_body_height = float(body_height)

        self.msg.mode = 11
        self.msg.gait_id = 3

        self.msg.vel_des = [
            float(vx),
            float(vy),
            float(wz),
        ]

        self.msg.rpy_des = [
            float(self.current_roll),
            float(self.current_pitch),
            0.0,
        ]

        self.msg.pos_des = [
            0.0,
            0.0,
            float(self.current_body_height),
        ]

        self.msg.step_height = [
            self.step_height,
            self.step_height,
        ]

        self.inc_life_count()
        self.ctrl.Send_cmd(self.msg)

        self.get_logger().info(
            '[LCM_SEND] '
            f'vel=[{vx:.3f}, {vy:.3f}, {wz:.3f}], '
            f'rpy=[{self.current_roll:.3f}, '
            f'{self.current_pitch:.3f}, 0.000], '
            f'pos=[0.000, 0.000, '
            f'{self.current_body_height:.3f}], '
            f'life_count={self.msg.life_count}'
        )

    def enter_phase(self, phase_index: int):
        self.phase_index = phase_index
        self.phase_start_time = time.monotonic()

        if phase_index == 0:
            self.get_logger().warn(
                '[PHASE 0] 正常姿态，作为基准'
            )

            self.send_motion_cmd(
                roll=0.0,
                pitch=0.0,
                body_height=self.normal_height,
            )

        elif phase_index == 1:
            self.get_logger().warn(
                '[PHASE 1] 仅通过 LCM 降低机身高度'
            )

            self.send_motion_cmd(
                body_height=self.low_height,
            )

        elif phase_index == 2:
            self.get_logger().warn(
                '[PHASE 2] 不传 body_height，'
                '检查是否继续保持上一阶段的低姿态'
            )

            # body_height 默认是 None。
            # 实际发送的仍然应该是 self.low_height。
            self.send_motion_cmd()

        elif phase_index == 3:
            self.get_logger().warn(
                '[PHASE 3] 恢复正常高度'
            )

            self.send_motion_cmd(
                body_height=self.normal_height,
            )

        elif phase_index == 4:
            self.get_logger().warn(
                '[PHASE 4] 设置正方向 roll'
            )

            self.send_motion_cmd(
                roll=self.roll_angle,
            )

        elif phase_index == 5:
            self.get_logger().warn(
                '[PHASE 5] 不传 roll，'
                '检查是否继续保持上一阶段的倾斜姿态'
            )

            # roll 默认是 None。
            # 实际发送的仍然应该是 self.roll_angle。
            self.send_motion_cmd()

        elif phase_index == 6:
            self.get_logger().warn(
                '[PHASE 6] 设置反方向 roll'
            )

            self.send_motion_cmd(
                roll=-self.roll_angle,
            )

        elif phase_index == 7:
            self.get_logger().warn(
                '[PHASE 7] 恢复正常高度和水平姿态'
            )

            self.send_motion_cmd(
                roll=0.0,
                pitch=0.0,
                body_height=self.normal_height,
            )

        else:
            self.finish_test()

    def timer_callback(self):
        if self.test_finished:
            return

        if self.phase_index < 0:
            self.enter_phase(0)
            return

        elapsed = time.monotonic() - self.phase_start_time

        if elapsed >= self.phase_seconds:
            self.enter_phase(self.phase_index + 1)

    def finish_test(self):
        if self.test_finished:
            return

        self.test_finished = True

        # 最终保持正常站姿
        self.send_motion_cmd(
            vx=0.0,
            vy=0.0,
            wz=0.0,
            roll=0.0,
            pitch=0.0,
            body_height=self.normal_height,
        )

        self.get_logger().warn(
            '[TEST_FINISHED] 测试完成，机器狗保持正常姿态。'
            '按 Ctrl+C 退出。'
        )

        self.timer.cancel()

    def shutdown(self):
        if self.ctrl is None:
            return

        self.get_logger().info(
            '退出测试，恢复正常水平姿态'
        )

        try:
            self.send_motion_cmd(
                vx=0.0,
                vy=0.0,
                wz=0.0,
                roll=0.0,
                pitch=0.0,
                body_height=self.normal_height,
            )

            time.sleep(0.3)

            # 最后再发送一次底层恢复站立命令
            self.recovery_stand(wait_finish=False)
            time.sleep(0.1)

        except Exception as exc:
            self.get_logger().error(
                f'退出时恢复姿态失败：{exc}'
            )

        finally:
            self.ctrl.quit()
            self.ctrl = None


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = LcmPoseTestNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()