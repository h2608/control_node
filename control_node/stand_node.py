#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from control_node.my_gait import Robot_Ctrl
from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt


class StandNode(Node):
    """让机器狗恢复站立，并持续保持站立状态。"""

    def __init__(self):
        super().__init__('stand_node')

        # 第一赛段原代码使用的正常站立高度是 0.28 m
        self.declare_parameter('body_height', 0.28)
        self.body_height = float(
            self.get_parameter('body_height').value
        )

        self.ctrl = Robot_Ctrl()
        self.msg = robot_control_cmd_lcmt()

        # 启动 LCM 接收和心跳线程
        self.ctrl.run()

        self.get_logger().info(
            f'启动站立节点，目标机身高度={self.body_height:.2f} m'
        )

        self.send_stand_command()

    def increase_life_count(self):
        self.msg.life_count += 1

        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_stand_command(self):
        """发送恢复站立命令。"""

        # mode=12、gait_id=0：恢复站立/停止运动
        self.msg.mode = 12
        self.msg.gait_id = 0

        self.increase_life_count()

        # 不进行任何平移和旋转
        self.msg.vel_des = [0.0, 0.0, 0.0]

        # 保持机身水平
        self.msg.rpy_des = [0.0, 0.0, 0.0]

        # 设置正常站立高度
        self.msg.pos_des = [0.0, 0.0, self.body_height]

        self.msg.step_height = [0.0, 0.0]

        self.ctrl.Send_cmd(self.msg)

        self.get_logger().info(
            '[STAND] 已发送恢复站立命令，'
            'Robot_Ctrl 将持续发送该命令'
        )

        # 等待底层控制器完成起立动作，最长约 10 秒
        finished = self.ctrl.Wait_finish(12, 0)

        if finished:
            self.get_logger().info(
                '[STAND] 机器狗已经完成起立并保持站立'
            )
        else:
            self.get_logger().warn(
                '[STAND] 没有收到起立完成响应，'
                '但站立命令仍在持续发送'
            )

    def shutdown(self):
        """关闭 LCM 心跳线程。"""

        if self.ctrl is not None:
            self.get_logger().info('停止站立节点的 LCM 心跳')
            self.ctrl.quit()
            self.ctrl = None


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = StandNode()
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