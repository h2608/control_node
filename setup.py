from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'control_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'REAL_ROBOT_MIGRATION_README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cyberdog-team',
    maintainer_email='cyberdog@example.com',
    description='CyberDog multi-stage competition control with Gazebo and physical-robot backends.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'stage1_node = control_node.stage1_node:main',
            'stage2_node = control_node.stage2_node:main',
            'stage2_vision_preview = control_node.stage2_vision_preview:main',
            'stage3_node = control_node.stage3_node:main',
            'stage4_node = control_node.stage4_node:main',
            'stage4_vision_preview = control_node.stage4_vision_preview:main',
            'stage5_node = control_node.stage5_node:main',
            'stage6_node = control_node.stage6_node:main',
            'mission_control_node = control_node.mission_control_node:main',
            'stand_node = control_node.stand_node:main',
            'bridge_perception_replay = control_node.bridge_perception_replay:main',
            # Stationary hardware debug tools brought over from the robot.
            # Run them only with every other motion node stopped.
            'body_hold = control_node.body_hold:main',
            'target_hit_enter_test = control_node.target_hit_enter_test:main',
            'turn_pose_test = control_node.turn_pose_test:main',
        ],
    },
)
