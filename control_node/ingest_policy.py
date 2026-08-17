#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure (no ROS) resolution of the raw-image ingestion policy.

``full_competition.launch.py`` starts six stage nodes, and every one of them
inherits :class:`control_node.stage_common.StageNodeBase`.  Before this policy
existed each of those nodes held a full-rate raw RGB *and* raw depth
subscription from launch until shutdown, whether or not it was the active stage
and whether or not it ever looked at the frame -- ``rgb_callback`` caches the
message and returns early while inactive, but rclpy has already taken and
deserialised the whole image by then.  On the physical robot that is five extra
readers of a multi-hundred-kilobyte topic competing with the C++ compression
bridge for the same six logical CPUs and the same DDS transport.

The resolution rules live here, away from rclpy, so they can be unit tested on a
machine with no ROS installation.

Python 3.6 compatible: the physical robot runs 3.6.
"""

RAW_MODE_AUTO = 'auto'
RAW_MODE_ALWAYS = 'always'
RAW_MODE_ON_ACTIVATION = 'on_activation'
RAW_MODE_OFF = 'off'
RAW_MODES = (RAW_MODE_ALWAYS, RAW_MODE_ON_ACTIVATION, RAW_MODE_OFF)

# rclpy's qos_profile_sensor_data depth.  Passed in explicitly by stage_common
# so this module never imports rclpy; the default keeps the tests honest.
SENSOR_QOS_DEFAULT_DEPTH = 5


def _is_real(platform) -> bool:
    return str(platform).strip().lower() == 'real'


def resolve_raw_subscription_mode(requested, platform: str) -> str:
    """Resolve a raw-image subscription mode parameter.

    ``auto`` keeps the historical always-on behaviour in simulation, where the
    Gazebo host has CPU to spare and the sim regression is tuned against it, and
    selects ``on_activation`` on the physical robot, where it leaves at most one
    raw reader per stream instead of six.  An unknown value falls back to
    ``auto`` rather than refusing to start: a mistyped debug parameter must not
    turn into a node that will not launch.
    """
    value = str(requested).strip().lower()
    if value in RAW_MODES:
        return value
    return RAW_MODE_ON_ACTIVATION if _is_real(platform) else RAW_MODE_ALWAYS


def resolve_image_qos_depth(requested, platform: str,
                            sim_default: int = SENSOR_QOS_DEFAULT_DEPTH) -> int:
    """Resolve the KEEP_LAST depth used for raw image subscriptions.

    ``<= 0`` means auto: 1 on the physical robot, the rclpy sensor-data default
    in simulation.  A large-image reader gains nothing from a deep history --
    vision only ever consumes the newest frame -- while every extra slot is
    another full frame the middleware must keep alive for that reader.
    """
    try:
        depth = int(requested)
    except (TypeError, ValueError):
        depth = 0
    if depth > 0:
        return depth
    return 1 if _is_real(platform) else int(sim_default)


def resolve_resubscribe_after_s(requested, platform: str) -> float:
    """Resolve the silent-stream subscription-rebuild threshold, in seconds.

    ``< 0`` means auto: 3 s on the physical robot, off in simulation, where no
    wedged reader has ever been observed and the Gazebo regression should not
    gain a new moving part.  ``0`` disables the watchdog explicitly.
    """
    try:
        value = float(requested)
    except (TypeError, ValueError):
        value = -1.0
    if value >= 0.0:
        return value
    return 3.0 if _is_real(platform) else 0.0


def raw_stream_wanted(mode: str, activated: bool) -> bool:
    """Decide whether a stream is subscribed in the given lifecycle phase."""
    if mode == RAW_MODE_ALWAYS:
        return True
    if mode == RAW_MODE_ON_ACTIVATION:
        return bool(activated)
    return False
