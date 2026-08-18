#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 cruise velocity: the vision-dropout fallback must slow down, not speed up.

Measured on the physical robot (2026-08-18): ``image_rgb`` stutters for 0.5-1.0 s
routinely and once delivered a single frame in six seconds.  The old fallback
answered that by commanding ``p1_base_forward_speed`` with no correction at all,
which is *faster* than corrected cruise and blind.
"""

from control_node.stage1_node import p1_cruise_velocity

import pytest


PARAMS = dict(
    timeout_s=1.0, hold_s=0.5, decay_s=1.0, blind_min_speed=0.0,
    base_speed=0.40, min_speed=0.20, kp_turn=0.25, kp_lat=0.15,
    kd_slowdown=0.05, max_turn=0.15, max_lat=0.15,
)


def cruise(lateral_force, vision_age, **overrides):
    params = dict(PARAMS)
    params.update(overrides)
    return p1_cruise_velocity(lateral_force, vision_age, **params)


def test_fresh_frame_keeps_the_original_correction():
    """Inside the timeout nothing changes: the pre-existing gains still apply."""
    vx, vy, wz, mode = cruise(0.5, 0.0)
    assert mode == 'fresh'
    assert wz == pytest.approx(0.5 * 0.25)
    assert vy == pytest.approx(0.5 * 0.15)
    assert vx == pytest.approx(0.40 - 0.5 * 0.05)


def test_correction_saturates_at_the_configured_limits():
    vx, vy, wz, _ = cruise(1.0, 0.0, kp_turn=10.0, kp_lat=10.0)
    assert wz == pytest.approx(0.15)
    assert vy == pytest.approx(0.15)


def test_forward_speed_never_drops_below_min_while_fresh():
    vx, _, _, _ = cruise(1.0, 0.0, kd_slowdown=10.0)
    assert vx == pytest.approx(0.20)


def test_hold_keeps_the_last_correction_briefly():
    """A 1.2 s gap is routine on this hardware; do not throw the steering away."""
    fresh = cruise(0.5, 0.0)
    held = cruise(0.5, 1.2)
    assert held[3] == 'hold'
    assert held[:3] == fresh[:3]


def test_blind_cruise_is_never_faster_than_corrected_cruise():
    """The whole point: the fallback must not exceed the fresh-frame speed."""
    for age in (0.0, 0.9, 1.0, 1.4, 1.6, 2.0, 5.0, 60.0):
        vx = cruise(0.8, age)[0]
        assert vx <= cruise(0.8, 0.0)[0] + 1e-9, 'age=%.2f sped up' % age


def test_steering_stops_once_the_hold_expires():
    _, vy, wz, mode = cruise(0.9, 1.6)
    assert mode == 'decay'
    assert (vy, wz) == (0.0, 0.0)


def test_speed_ramps_down_monotonically_to_the_blind_minimum():
    ages = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    speeds = [cruise(0.0, a)[0] for a in ages]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[0] == pytest.approx(0.40)
    assert speeds[-1] == pytest.approx(0.0)


def test_six_second_stall_ends_stopped():
    """The worst measured stall: 1 frame in 6 s must not mean 2.4 m of blind travel."""
    assert cruise(0.0, 6.0)[0] == pytest.approx(0.0)


def test_blind_minimum_is_configurable_for_a_crawl():
    assert cruise(0.0, 6.0, blind_min_speed=0.08)[0] == pytest.approx(0.08)


def test_zero_decay_drops_straight_to_the_blind_minimum():
    assert cruise(0.0, 1.51, decay_s=0.0)[0] == pytest.approx(0.0)


def test_zero_hold_starts_decaying_immediately_after_the_timeout():
    assert cruise(0.0, 1.01, hold_s=0.0)[3] == 'decay'


def test_node_never_started_receiving_frames_is_treated_as_blind():
    """p1_last_update_time starts at 0.0, so a namespace typo yields a huge age."""
    vx, vy, wz, mode = cruise(0.0, 1.7e9)
    assert mode == 'decay'
    assert (vx, vy, wz) == (0.0, 0.0, 0.0)
