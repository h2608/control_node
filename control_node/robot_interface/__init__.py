"""Robot control backends for Gazebo and the physical CyberDog.

Stage code should depend on semantic methods (move/stop_motion/recovery_stand/
run_action) where possible.  The adapters also expose a temporary legacy
Send_cmd/Wait_finish compatibility surface so the six already-tested stage
state machines can be migrated incrementally.
"""

from .factory import create_robot_controller

__all__ = ['create_robot_controller']
