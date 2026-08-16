"""Gazebo/LCM implementation of the robot-control adapter."""

from control_node.my_gait import Robot_Ctrl
from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt


class SimRobotControlAdapter:
    is_real = False
    backend_name = 'sim_lcm'

    def __init__(self, parent_node):
        self.node = parent_node
        self._ctrl = Robot_Ctrl()
        # Share the stage node's command message rather than owning a second
        # one.  The locomotion controller accepts a packet only when its
        # life_count differs from the last accepted value
        # (command_interface.cpp: `if (lcm_cmd_.life_count == life_count_)`),
        # so two independent counters publishing on robot_control_cmd will
        # eventually collide and the colliding command is dropped silently.
        # Stages that still build actions from node.msg (Stage 5's
        # p5_send_action_once) must therefore share one counter with move().
        self._semantic_msg = getattr(parent_node, 'msg', None)
        if self._semantic_msg is None:
            self._semantic_msg = robot_control_cmd_lcmt()
        if not hasattr(self._semantic_msg, 'life_count'):
            self._semantic_msg.life_count = 0

    def __getattr__(self, name):
        # Temporary compatibility for response_snapshot(), get_status(), etc.
        return getattr(self._ctrl, name)

    def run(self):
        self._ctrl.run()

    def quit(self):
        self._ctrl.quit()

    def _inc_life(self):
        self._semantic_msg.life_count += 1
        if self._semantic_msg.life_count > 127:
            self._semantic_msg.life_count = 0

    def move(self, vx, vy, wz, *, step_height=0.02, roll=0.0, pitch=0.0,
             yaw=0.0, body_height=0.0, legacy_gait_id=3, motion_id=None):
        del motion_id
        msg = self._semantic_msg
        msg.mode = 11
        msg.gait_id = int(legacy_gait_id)
        msg.vel_des = [float(vx), float(vy), float(wz)]
        msg.step_height = [float(step_height), float(step_height)]
        msg.rpy_des = [float(roll), float(pitch), float(yaw)]
        msg.pos_des = [0.0, 0.0, float(body_height)]
        self._inc_life()
        self._ctrl.Send_cmd(msg)

    def stop_motion(self, wait_finish=True):
        # Preserve the old simulator STOP behaviour exactly.  The physical
        # backend implements this as SERVO_END instead.
        #
        # wait_finish mirrors the pre-migration split: StageNodeBase's
        # send_stop_command() blocked on Wait_finish(12, 0), while Stage 5's
        # p5_send_stop_command() was fire-and-forget because it runs inside
        # the 30-50 Hz control loop and polls completion itself.  Wait_finish
        # blocks for up to 10 s, which would stall that loop.
        msg = self._semantic_msg
        msg.mode = 12
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.step_height = [0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        self._inc_life()
        self._ctrl.Send_cmd(msg)
        if not wait_finish:
            return True
        return bool(self._ctrl.Wait_finish(12, 0))

    def recovery_stand(self, wait_finish=True):
        msg = self._semantic_msg
        msg.mode = 12
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.step_height = [0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        self._inc_life()
        self._ctrl.Send_cmd(msg)
        return bool(self._ctrl.Wait_finish(12, 0)) if wait_finish else True

    def run_action(self, action_name, wait_finish=True):
        mapping = {
            'left_jump': (16, 0),
            'right_jump': (16, 3),
            'forward_jump': (16, 1),
            'recovery_stand': (12, 0),
            'emergency_stop': (0, 0),
            'lie_down': (7, 1),
        }
        if action_name not in mapping:
            raise ValueError(f'unsupported simulator action: {action_name}')
        mode, gait = mapping[action_name]
        msg = self._semantic_msg
        msg.mode = mode
        msg.gait_id = gait
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.step_height = [0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        self._inc_life()
        self._ctrl.Send_cmd(msg)
        return bool(self._ctrl.Wait_finish(mode, gait)) if wait_finish else True

    # Legacy surface -----------------------------------------------------
    def Send_cmd(self, msg):
        return self._ctrl.Send_cmd(msg)

    def Send_cmd_with_response_barrier(self, msg):
        return self._ctrl.Send_cmd_with_response_barrier(msg)

    def Wait_finish(self, mode, gait_id):
        return self._ctrl.Wait_finish(mode, gait_id)
