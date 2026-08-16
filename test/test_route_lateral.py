#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the odometry route-frame lateral observation."""

import math

from control_node.deck_lateral import CONTROL_ACTIVE, DeckLateralController
from control_node.route_lateral import (
    SOURCE_ODOMETRY,
    anchor_from_depth,
    USE_FULL,
    USE_HEADING_ONLY,
    route_frame_observation,
    select_observation,
)

NORTH = math.pi / 2.0     # the Stage 5 climb heading (+y)
WEST = math.pi           # the heading after corner 1 (-x)


def test_on_the_line_reports_no_deviation():
    """A body exactly on its entry line has nothing to correct."""
    obs = route_frame_observation((3.12, 7.30), NORTH, 3.12, 9.00, NORTH)
    assert obs['valid']
    assert obs['lateral_offset'] == pytest_approx(0.0)
    assert obs['heading_error'] == pytest_approx(0.0)
    assert obs['source'] == SOURCE_ODOMETRY


def test_drift_right_of_the_line_asks_for_a_left_correction():
    """Heading +y and drifting to +x is a drift right, so correct left."""
    obs = route_frame_observation((3.12, 7.30), NORTH, 3.20, 9.00, NORTH)
    # +x while facing +y is the body's right, so the line is to its left.
    assert obs['lateral_offset'] > 0.0
    assert obs['lateral_offset'] == pytest_approx(0.08)


def test_drift_left_of_the_line_asks_for_a_right_correction():
    """The mirror case must produce the mirror sign."""
    obs = route_frame_observation((3.12, 7.30), NORTH, 3.04, 9.00, NORTH)
    assert obs['lateral_offset'] == pytest_approx(-0.08)


def test_sign_convention_holds_after_the_corner():
    """Heading -x, drifting to +y is the body's right; correct left."""
    obs = route_frame_observation((3.13, 12.43), WEST, 2.50, 12.54, WEST)
    assert obs['lateral_offset'] == pytest_approx(0.11)


def test_heading_error_points_at_the_reference_direction():
    """Yawed right of the reference means the reference is front-left."""
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.0, 1.0, NORTH - 0.10)
    assert obs['heading_error'] == pytest_approx(0.10)
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.0, 1.0, NORTH + 0.10)
    assert obs['heading_error'] == pytest_approx(-0.10)


def test_along_track_motion_produces_no_lateral_term():
    """Distance along the line is progress, never deviation."""
    obs = route_frame_observation((3.12, 7.30), NORTH, 3.12, 11.80, NORTH)
    assert obs['lateral_offset'] == pytest_approx(0.0)


def test_invalid_odometry_is_never_silently_zero():
    """A dead odometry stream must read as unusable, not as 'on the line'."""
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.0, 1.0, NORTH,
                                  odometry_valid=False)
    assert not obs['valid']
    assert obs['reason'] == 'odometry_invalid'
    assert obs['lateral_offset'] is None


def test_missing_reference_is_rejected():
    """Before the segment origin is known there is no line to hold."""
    obs = route_frame_observation(None, NORTH, 0.0, 1.0, NORTH)
    assert not obs['valid']
    assert obs['reason'] == 'no_reference'


def test_nan_pose_is_rejected():
    """A NaN must not reach the gain."""
    obs = route_frame_observation((0.0, 0.0), NORTH, float('nan'), 1.0, NORTH)
    assert not obs['valid']
    assert obs['reason'] == 'nan_input'


def test_implausible_lateral_is_rejected():
    """Metres off the line is a lost fix, not a correctable deviation."""
    obs = route_frame_observation((0.0, 0.0), NORTH, 2.0, 0.0, NORTH)
    assert not obs['valid']
    assert obs['reason'] == 'lateral_implausible'


def test_implausible_heading_is_rejected():
    """Past 60 deg off the segment this is a corner problem, not a lateral one."""
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.0, 1.0, NORTH + 1.5)
    assert not obs['valid']
    assert obs['reason'] == 'heading_implausible'


def test_heading_only_mode_suppresses_the_position_term():
    """Where estimator position is untrusted, attitude alone still helps."""
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.30, 1.0, NORTH - 0.10,
                                  mode=USE_HEADING_ONLY)
    assert obs['valid']
    assert obs['lateral_offset'] == 0.0
    assert obs['heading_error'] == pytest_approx(0.10)


def test_ladder_prefers_depth_when_it_is_valid():
    """Depth is course-referenced; odometry only ever holds the entry line."""
    depth = {'valid': True, 'lateral_offset': 0.05, 'heading_error': 0.01}
    odom = route_frame_observation((0.0, 0.0), NORTH, 0.30, 1.0, NORTH)
    chosen, source = select_observation(depth, odom)
    assert source == 'depth'
    assert chosen['lateral_offset'] == 0.05


def test_ladder_falls_back_to_odometry_when_depth_is_blind():
    """Losing the deck must degrade the loop, not disable it."""
    depth = {'valid': False}
    odom = route_frame_observation((0.0, 0.0), NORTH, 0.30, 1.0, NORTH)
    chosen, source = select_observation(depth, odom)
    assert source == SOURCE_ODOMETRY
    assert chosen['lateral_offset'] == pytest_approx(0.30)


def test_ladder_reports_nothing_when_both_sources_are_out():
    """With no evidence at all the caller must see None, not a zero correction."""
    chosen, source = select_observation({'valid': False}, {'valid': False})
    assert chosen is None
    assert source == 'none'


def test_observation_drives_the_existing_controller_unchanged():
    """The whole point of the shared contract: no adapter in between."""
    controller = DeckLateralController()
    obs = route_frame_observation((3.12, 7.30), NORTH, 3.20, 9.00, NORTH,
                                  mode=USE_FULL)
    controller.update(obs, now_s=0.0)
    cmd = controller.update(obs, now_s=0.2)
    assert cmd.state == CONTROL_ACTIVE
    assert cmd.vy > 0.0


def pytest_approx(value, tol=1e-6):
    """Local tolerance helper (keeps the assertions readable)."""
    import pytest
    return pytest.approx(value, abs=tol)


def test_entry_line_reports_zero_for_a_body_that_entered_off_centre():
    """The failure this anchor exists to fix, stated as a test.

    Measured 2026-08-04: a run entered straight_1 0.205 m off the rail centre,
    the unanchored fallback's first observation read ``offset=0.0``, and the
    centre-first hold never fired.
    """
    obs = route_frame_observation((3.13, 12.615), WEST, 3.13, 12.615, WEST)
    assert obs['valid']
    assert obs['lateral_offset'] == pytest_approx(0.0)


def test_line_offset_shifts_the_reference_line():
    """A known offset of the true centre from the entry line is honoured."""
    obs = route_frame_observation((3.13, 12.615), WEST, 3.13, 12.615, WEST,
                                  line_offset_m=0.205)
    assert obs['lateral_offset'] == pytest_approx(0.205)
    assert obs['line_offset_m'] == pytest_approx(0.205)


def test_anchor_makes_odometry_reproduce_the_depth_fix():
    """The whole point: after anchoring, the two sources agree."""
    raw = route_frame_observation((3.13, 12.615), WEST, 3.13, 12.615, WEST)
    anchor = anchor_from_depth(0.205, raw['lateral_offset'])
    assert anchor == pytest_approx(0.205)
    anchored = route_frame_observation((3.13, 12.615), WEST, 3.13, 12.615, WEST,
                                       line_offset_m=anchor)
    assert anchored['lateral_offset'] == pytest_approx(0.205)


def test_anchor_survives_subsequent_motion():
    """Once anchored, dead reckoning tracks the true centre, not the entry pose."""
    anchor = anchor_from_depth(0.20, 0.0)
    # Heading -x, so moving +y is moving to the body's right: the centreline
    # gets further to the left and the reported offset grows.
    obs = route_frame_observation((3.13, 12.40), WEST, 2.80, 12.50, WEST,
                                  line_offset_m=anchor)
    assert obs['lateral_offset'] == pytest_approx(0.30)


def test_anchor_rejects_an_implausible_shift():
    """A wild depth reading must not be able to move the line arbitrarily."""
    assert anchor_from_depth(5.0, 0.0) is None
    assert anchor_from_depth(float('nan'), 0.0) is None
    assert anchor_from_depth(0.1, None) is None


def test_anchor_of_zero_is_a_no_op():
    """Depth agreeing with the entry line must not perturb anything."""
    assert anchor_from_depth(0.0, 0.0) == pytest_approx(0.0)


def test_bad_line_offset_is_rejected_not_silently_zeroed():
    """A NaN anchor must invalidate the observation, not vanish."""
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.0, 1.0, NORTH,
                                  line_offset_m=float('nan'))
    assert not obs['valid']
    assert obs['reason'] == 'bad_line_offset'


# ---------------------------------------------------------------------------
# The route heading grid (plan item 29).
#
# Measured over 20 sim runs on 2026-08-04: the grid used to be anchored to the
# body's yaw at the first segment, which ranged +1.11..+1.83 rad against a true
# course heading of +pi/2.  These tests pin down why that is fatal and what the
# declared grid buys.
# ---------------------------------------------------------------------------

def test_grid_error_fabricates_cross_track_over_a_straight():
    """A grid error e reports 3.4*sin(e) of deviation for a dead-straight walk.

    The body here walks 3.4 m along the *true* course heading and never leaves
    the centreline, but the reference line is drawn 0.057 rad off (the measured
    start-yaw error of the run that walked off the deck at the crest).
    """
    bad_reference = NORTH - 0.0572
    obs = route_frame_observation((0.0, 0.0), bad_reference,
                                  0.0, 3.40, NORTH)
    assert obs['valid']
    # Pure fiction: the body is exactly on the line it started on.
    assert obs['lateral_offset'] == pytest_approx(-3.40 * math.sin(0.0572), tol=2e-3)
    # And it is large enough to matter: the deck half-width is 0.245 m.
    assert abs(obs['lateral_offset']) > 0.19


def test_declared_grid_reports_no_deviation_for_the_same_walk():
    """With the course heading declared, the same straight walk reads ~zero."""
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.0, 3.40, NORTH)
    assert obs['valid']
    assert obs['lateral_offset'] == pytest_approx(0.0, tol=1e-9)


def test_declared_grid_turns_a_crooked_start_into_a_visible_heading_error():
    """A body placed 15 deg off must read as 15 deg off, not as 'on course'.

    Anchoring the grid to the measured pose declared the crooked placement to
    be the route direction, so the loop saw zero heading error and walked the
    body off at an angle.  Against a declared grid the error is visible and the
    wz term turns the body onto the route.
    """
    crooked = NORTH - math.radians(15.0)
    obs = route_frame_observation((0.0, 0.0), NORTH, 0.0, 0.0, crooked)
    assert obs['valid']
    assert obs['heading_error'] == pytest_approx(math.radians(15.0), tol=1e-9)

    controller = DeckLateralController()
    controller.update(obs, 0.0)
    command = controller.update(obs, 0.1)
    assert command.state == CONTROL_ACTIVE
    assert command.wz > 0.0                      # turning back onto the route
