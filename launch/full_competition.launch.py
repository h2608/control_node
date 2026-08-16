#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""启动六个赛段节点 + 任务控制节点。

激活话题为 latched（TRANSIENT_LOCAL），因此节点启动顺序不敏感；
任务控制节点会等 start_delay_sec 后再激活第一赛段。
"""

import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def effective_camera_topic(argument_name, sim_default, real_default):
    """Choose sim/real camera topic unless the launch argument overrides it."""
    return PythonExpression([
        "'", LaunchConfiguration(argument_name), "' if '",
        LaunchConfiguration(argument_name),
        "' != 'auto' else ('", real_default, "' if '",
        LaunchConfiguration('platform'), "' == 'real' else '",
        sim_default, "')",
    ])


def effective_robot_namespace():
    """Use the verified robot namespace on real hardware unless overridden."""
    return PythonExpression([
        "'", LaunchConfiguration('robot_namespace'), "' if '",
        LaunchConfiguration('robot_namespace'),
        "' != 'auto' else ('mi_desktop_48_b0_2d_7b_00_e2' if '",
        LaunchConfiguration('platform'), "' == 'real' else '')",
    ])


def single_stage_condition(n):
    """Launch every stage normally, or only start_stage in single-stage mode."""
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration('single_stage'), "' != 'true' or '",
        LaunchConfiguration('start_stage'), "' == '", str(n), "'",
    ]))


def stage_entry_argument(n, entry_points):
    """Declare the per-stage debug entry argument.

    Each stage node owns a StageEntryTable (see control_node/stage_entry.py)
    that maps these names onto its internal states; the node logs the whole
    table and falls back to its normal start when the name is unknown.
    """
    return DeclareLaunchArgument(
        f'stage{n}_entry',
        default_value='default',
        description=(
            f'Stage {n} debug entry point; default starts the stage normally. '
            f'Named entries: {entry_points}. A raw state name also works. '
            'The robot must already be placed at that point of the course.'
        ),
    )


def stage_node(n, extra_params=None, extra_param_files=None):
    params = {
        'platform': LaunchConfiguration('platform'),
        'use_sim_time': effective_use_sim_time(),
        'entry_point': LaunchConfiguration(f'stage{n}_entry'),
        'rgb_topic': effective_camera_topic(
            'rgb_topic',
            '/rgb_camera/rgb_camera/image_raw',
            '/mi_desktop_48_b0_2d_7b_00_e2/image_rgb',
        ),
        'depth_topic': effective_camera_topic(
            'depth_topic',
            '/d435/depth/d435_depth/depth/image_raw',
            '/mi_desktop_48_b0_2d_7b_00_e2/camera/depth/image_rect_raw',
        ),
        'ai_camera_topic': effective_camera_topic(
            'ai_camera_topic',
            '',
            '/mi_desktop_48_b0_2d_7b_00_e2/image',
        ),
        'global_frame': LaunchConfiguration('global_frame'),
        'base_frame': LaunchConfiguration('base_frame'),
        'show_debug_vis': False,
        'real_motion_servo_cmd_topic': LaunchConfiguration('real_motion_servo_cmd_topic'),
        'real_motion_servo_response_topic': LaunchConfiguration('real_motion_servo_response_topic'),
        'real_motion_result_service': LaunchConfiguration('real_motion_result_service'),
    }
    if extra_params:
        params.update(extra_params)
    parameters = [params]
    if extra_param_files:
        parameters.extend(extra_param_files)
    return launch_ros.actions.Node(
        package='control_node',
        executable=f'stage{n}_node',
        name=f'stage{n}_node',
        namespace=effective_robot_namespace(),
        output='screen',
        parameters=parameters,
        condition=single_stage_condition(n),
    )


def effective_use_sim_time():
    """Physical robot always uses wall time; Gazebo may use /clock."""
    return PythonExpression([
        "'false' if ('",
        LaunchConfiguration('platform'),
        "' == 'real' or '",
        LaunchConfiguration('stage5_profile'),
        "' == 'physical') else '",
        LaunchConfiguration('use_sim_time'),
        "'",
    ])


def stage5_profile_files():
    """Load the physical Stage-5 profile automatically on a real robot.

    For simulation the existing stage5_profile argument keeps its old
    sim/sim_odometry/sim_depth behaviour.  A real backend must never inherit
    Gazebo-tuned Stage-5 velocities/durations by accident.
    """
    is_real_or_physical = [
        "('", LaunchConfiguration('platform'), "' == 'real' or '",
        LaunchConfiguration('stage5_profile'), "' == 'physical')"
    ]
    base = PythonExpression([
        "'stage5_physical.yaml' if ", *is_real_or_physical,
        " else 'stage5_sim.yaml'",
    ])
    middle = PythonExpression([
        "'stage5_physical.yaml' if '", LaunchConfiguration('platform'),
        "' == 'real' else ('stage5_sim_odometry.yaml' if '",
        LaunchConfiguration('stage5_profile'),
        "' == 'sim_depth' else ('stage5_physical.yaml' if '",
        LaunchConfiguration('stage5_profile'),
        "' == 'physical' else 'stage5_sim.yaml'))",
    ])
    overlay = PythonExpression([
        "'stage5_physical.yaml' if '", LaunchConfiguration('platform'),
        "' == 'real' else ('stage5_' + '",
        LaunchConfiguration('stage5_profile'), "' + '.yaml')",
    ])
    return [
        PathJoinSubstitution([FindPackageShare('control_node'), 'config', name])
        for name in (base, middle, overlay)
    ]


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'platform',
            default_value='sim',
            description='Robot backend: sim (LCM/Gazebo) or real (MotionServo/Result API)',
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='auto',
            description='ROS namespace without a leading slash; auto selects the verified real-robot namespace',
        ),
        DeclareLaunchArgument(
            'start_stage',
            default_value='1',
            description='First mission stage to run (1-6); later stages continue in order',
        ),
        DeclareLaunchArgument(
            'single_stage',
            default_value='false',
            description='If true, launch and run only start_stage, then stop',
        ),
        # Per-stage debug entry points.  Combine with single_stage:=true to run
        # one section of one stage, e.g. only the fifth stage's ramp:
        #   single_stage:=true start_stage:=5 stage5_entry:=ramp
        stage_entry_argument(1, 'start, cruise, brake, align, restore, forward, '
                                'turn, ball, shift'),
        stage_entry_argument(2, 'start, track1, track1_exit, track1_turn, '
                                'track1_shift, track2, track2_turn, '
                                'track2_forward, track3, scan, scan_hit, '
                                'turn_back, final, final_forward, final_turn, '
                                'final_align, ball_align, ball_hit, ball_shift'),
        stage_entry_argument(3, 'start, s_curve, align'),
        stage_entry_argument(4, 'start, search, bar_center, bar, bar_target, '
                                'bar_hit, bar_back, bar_yellow, '
                                'obstacle_center, obstacle, obstacle_route, '
                                'target, target_hit, upright, post_hit, '
                                'post_hit_obstacle, final, final_yellow, '
                                'final_align'),
        stage_entry_argument(5, 'start, recovery, align, step_up, ramp, '
                                'ramp_exit, corner_1, slope_body, straight_1, '
                                'corner_2, straight_2, corner_3, straight_3, '
                                'reset_body, corner_4, descent, final, '
                                'final_jump'),
        stage_entry_argument(6, 'start, north, north_align, turn, east, '
                                'clear_ball, crab, west, west_march, '
                                'west_align, exit, push, finish'),
        DeclareLaunchArgument(
            'stage2_ball_return',
            default_value='default',
            description='Cruise state the Stage-2 ball sub-chain returns to; '
                        'required by stage2_entry:=ball_* (default reuses the '
                        'entry state, which would loop)',
        ),
        DeclareLaunchArgument(
            'stage4_dashed_side',
            default_value='auto',
            description='Stage-4 dashed-line side (auto/left/right); required '
                        'by stage4_entry:=obstacle_route and post_hit_obstacle',
        ),
        DeclareLaunchArgument(
            'startup_recovery_enabled',
            default_value='true',
            description='On a real robot, execute RecoveryStand (111) before the first selected stage',
        ),
        DeclareLaunchArgument(
            'startup_recovery_settle_sec',
            default_value='0.0',
            description='Extra settle time after RecoveryStand succeeds; 0 starts the selected stage immediately',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo /clock instead of wall time',
        ),
        DeclareLaunchArgument(
            'rgb_topic',
            default_value='auto',
            description='Front RGB Image topic; auto selects the sim/real default',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='auto',
            description='Depth Image topic; auto selects the sim/real default',
        ),
        DeclareLaunchArgument(
            'ai_camera_topic',
            default_value='auto',
            description='Stage-4 basketball AI camera topic; auto selects the physical robot default',
        ),
        DeclareLaunchArgument(
            'basketball_ai_roi_x_min_ratio',
            default_value='0.20',
            description='Stage-4 basketball AI ROI left ratio',
        ),
        DeclareLaunchArgument(
            'basketball_ai_roi_x_max_ratio',
            default_value='0.80',
            description='Stage-4 basketball AI ROI right ratio',
        ),
        DeclareLaunchArgument(
            'basketball_ai_roi_y_min_ratio',
            default_value='0.05',
            description='Stage-4 basketball AI ROI top ratio',
        ),
        DeclareLaunchArgument(
            'basketball_ai_roi_y_max_ratio',
            default_value='0.90',
            description='Stage-4 basketball AI ROI bottom ratio',
        ),
        DeclareLaunchArgument(
            'basketball_ai_max_age_s',
            default_value='0.50',
            description='Maximum accepted Stage-4 AI-camera frame age',
        ),
        DeclareLaunchArgument(
            'basketball_top_slow_y_ratio',
            default_value='0.35',
            description='Basketball top-edge ratio that selects slow approach',
        ),
        DeclareLaunchArgument(
            'basketball_top_trigger_y_ratio',
            default_value='0.25',
            description='Basketball top-edge ratio that starts upright action',
        ),
        DeclareLaunchArgument(
            'basketball_top_trigger_confirm_frames',
            default_value='3',
            description='Consecutive top-edge trigger frames required',
        ),
        DeclareLaunchArgument(
            'fisheye_left_topic',
            default_value='auto',
            description='Stage-2 left fisheye Image topic; auto selects the sim/real default',
        ),
        DeclareLaunchArgument(
            'fisheye_right_topic',
            default_value='auto',
            description='Stage-2 right fisheye Image topic; auto selects the sim/real default',
        ),
        DeclareLaunchArgument(
            'global_frame',
            default_value='vodom',
            description='Global TF frame',
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='base_link',
            description='Robot base TF frame',
        ),
        DeclareLaunchArgument(
            'real_motion_servo_cmd_topic',
            default_value='/mi_desktop_48_b0_2d_7b_00_e2/motion_servo_cmd',
            description='Physical robot MotionServoCmd topic',
        ),
        DeclareLaunchArgument(
            'real_motion_servo_response_topic',
            default_value='/mi_desktop_48_b0_2d_7b_00_e2/motion_servo_response',
            description='Physical robot MotionServoResponse topic',
        ),
        DeclareLaunchArgument(
            'real_motion_result_service',
            default_value='/mi_desktop_48_b0_2d_7b_00_e2/motion_result_cmd',
            description='Physical robot MotionResultCmd service',
        ),
        DeclareLaunchArgument(
            'voice_dir',
            default_value='/home/cyberdog_sim/voice',
            description='Directory containing Stage-4 voice wav files',
        ),
        DeclareLaunchArgument(
            'voice_backend',
            default_value='auto',
            description='Stage-4 voice backend: auto, local, ros_offline or ros_online',
        ),
        DeclareLaunchArgument(
            'voice_topic',
            default_value='speech_play_extend',
            description='Relative AudioPlayExtend topic resolved under robot_namespace',
        ),
        DeclareLaunchArgument(
            'stage5_profile',
            default_value='sim',
            description='Stage 5 parameter profile: sim (Gazebo-tuned, '
                        'yellow-driven), sim_odometry (Gazebo-tuned, odometry '
                        'segment ends, no yellow in the control loop) or '
                        'physical (fail-safe placeholders, must be recalibrated)',
        ),
        stage_node(1),
        stage_node(2, {
            'p2_ball_return_state': LaunchConfiguration('stage2_ball_return'),
            'fisheye_left_topic': effective_camera_topic(
                'fisheye_left_topic',
                '/image_left',
                '/mi_desktop_48_b0_2d_7b_00_e2/image_left',
            ),
            'fisheye_right_topic': effective_camera_topic(
                'fisheye_right_topic',
                '/image_right',
                '/mi_desktop_48_b0_2d_7b_00_e2/image_right',
            ),
        }),
        stage_node(3),
        stage_node(4, {
            'debug_dashed_side': LaunchConfiguration('stage4_dashed_side'),
            'voice_dir': LaunchConfiguration('voice_dir'),
            'voice_backend': LaunchConfiguration('voice_backend'),
            'voice_topic': LaunchConfiguration('voice_topic'),
            'basketball_ai_roi_x_min_ratio': ParameterValue(
                LaunchConfiguration('basketball_ai_roi_x_min_ratio'), value_type=float),
            'basketball_ai_roi_x_max_ratio': ParameterValue(
                LaunchConfiguration('basketball_ai_roi_x_max_ratio'), value_type=float),
            'basketball_ai_roi_y_min_ratio': ParameterValue(
                LaunchConfiguration('basketball_ai_roi_y_min_ratio'), value_type=float),
            'basketball_ai_roi_y_max_ratio': ParameterValue(
                LaunchConfiguration('basketball_ai_roi_y_max_ratio'), value_type=float),
            'basketball_ai_max_age_s': ParameterValue(
                LaunchConfiguration('basketball_ai_max_age_s'), value_type=float),
            'basketball_top_slow_y_ratio': ParameterValue(
                LaunchConfiguration('basketball_top_slow_y_ratio'), value_type=float),
            'basketball_top_trigger_y_ratio': ParameterValue(
                LaunchConfiguration('basketball_top_trigger_y_ratio'), value_type=float),
            'basketball_top_trigger_confirm_frames': ParameterValue(
                LaunchConfiguration('basketball_top_trigger_confirm_frames'), value_type=int),
        }),
        stage_node(5, extra_param_files=stage5_profile_files()),
        stage_node(6),
        launch_ros.actions.Node(
            package='control_node',
            executable='mission_control_node',
            name='mission_control_node',
            namespace=effective_robot_namespace(),
            output='screen',
            parameters=[{
                'platform': LaunchConfiguration('platform'),
                'use_sim_time': effective_use_sim_time(),
                'start_stage': ParameterValue(
                    LaunchConfiguration('start_stage'), value_type=int),
                'single_stage': ParameterValue(
                    LaunchConfiguration('single_stage'), value_type=bool),
                'start_delay_sec': 2.0,
                'startup_recovery_enabled': ParameterValue(
                    LaunchConfiguration('startup_recovery_enabled'), value_type=bool),
                'startup_recovery_settle_sec': ParameterValue(
                    LaunchConfiguration('startup_recovery_settle_sec'), value_type=float),
                'real_motion_result_service': LaunchConfiguration(
                    'real_motion_result_service'),
            }],
        ),
    ])
