#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the raw-image ingestion policy resolution."""

from control_node.ingest_policy import (
    RAW_MODE_ALWAYS,
    RAW_MODE_OFF,
    RAW_MODE_ON_ACTIVATION,
    raw_stream_wanted,
    resolve_image_qos_depth,
    resolve_raw_subscription_mode,
    resolve_resubscribe_after_s,
)


# ----------------------------------------------------------------------
# Mode resolution
# ----------------------------------------------------------------------
def test_auto_keeps_the_old_always_on_behaviour_in_sim():
    # The Gazebo regression is tuned against always-on ingestion; auto must not
    # silently change what the simulator does.
    assert resolve_raw_subscription_mode('auto', 'sim') == RAW_MODE_ALWAYS


def test_auto_defers_ingestion_to_activation_on_the_robot():
    assert resolve_raw_subscription_mode('auto', 'real') == RAW_MODE_ON_ACTIVATION


def test_explicit_modes_win_over_the_platform_default():
    for platform in ('sim', 'real'):
        assert resolve_raw_subscription_mode('off', platform) == RAW_MODE_OFF
        assert resolve_raw_subscription_mode('always', platform) == RAW_MODE_ALWAYS
        assert resolve_raw_subscription_mode(
            'on_activation', platform) == RAW_MODE_ON_ACTIVATION


def test_mode_parsing_is_case_and_space_insensitive():
    assert resolve_raw_subscription_mode('  OFF ', 'real') == RAW_MODE_OFF


def test_unknown_mode_falls_back_to_auto_instead_of_raising():
    # A mistyped debug parameter must not turn into a node that will not launch.
    assert resolve_raw_subscription_mode('alwyas', 'real') == RAW_MODE_ON_ACTIVATION
    assert resolve_raw_subscription_mode(None, 'sim') == RAW_MODE_ALWAYS


# ----------------------------------------------------------------------
# QoS depth
# ----------------------------------------------------------------------
def test_qos_depth_auto_is_one_on_the_robot_and_the_sensor_default_in_sim():
    assert resolve_image_qos_depth(0, 'real') == 1
    assert resolve_image_qos_depth(0, 'sim', sim_default=5) == 5


def test_negative_qos_depth_is_treated_as_auto():
    assert resolve_image_qos_depth(-3, 'real') == 1


def test_explicit_qos_depth_is_honoured():
    assert resolve_image_qos_depth(4, 'real') == 4


def test_unparsable_qos_depth_falls_back_to_auto():
    assert resolve_image_qos_depth('deep', 'real') == 1


# ----------------------------------------------------------------------
# Resubscribe watchdog threshold
# ----------------------------------------------------------------------
def test_resubscribe_auto_is_on_for_the_robot_and_off_in_sim():
    assert resolve_resubscribe_after_s(-1.0, 'real') == 3.0
    assert resolve_resubscribe_after_s(-1.0, 'sim') == 0.0


def test_resubscribe_zero_disables_the_watchdog_explicitly():
    assert resolve_resubscribe_after_s(0.0, 'real') == 0.0


def test_resubscribe_explicit_value_is_honoured():
    assert resolve_resubscribe_after_s(7.5, 'sim') == 7.5


# ----------------------------------------------------------------------
# Lifecycle decision
# ----------------------------------------------------------------------
def test_always_subscribes_in_both_phases():
    assert raw_stream_wanted(RAW_MODE_ALWAYS, False) is True
    assert raw_stream_wanted(RAW_MODE_ALWAYS, True) is True


def test_on_activation_only_subscribes_while_active():
    assert raw_stream_wanted(RAW_MODE_ON_ACTIVATION, False) is False
    assert raw_stream_wanted(RAW_MODE_ON_ACTIVATION, True) is True


def test_off_never_subscribes():
    assert raw_stream_wanted(RAW_MODE_OFF, False) is False
    assert raw_stream_wanted(RAW_MODE_OFF, True) is False
