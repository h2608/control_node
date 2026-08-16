"""Select the simulator or physical-robot control backend."""


def create_robot_controller(parent_node):
    platform = str(parent_node.get_parameter('platform').value).strip().lower()
    if platform == 'sim':
        from .sim_controller import SimRobotControlAdapter
        return SimRobotControlAdapter(parent_node)
    if platform == 'real':
        # Lazy import: a Gazebo machine does not need the `protocol` package.
        from .real_controller import RealRobotControlAdapter
        return RealRobotControlAdapter(parent_node)
    raise ValueError("platform must be 'sim' or 'real', got {!r}".format(platform))
