#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the depth-derived deck-centring controller."""

from control_node.deck_lateral import (
    CONTROL_ACTIVE,
    CONTROL_BLIND,
    CONTROL_ENGAGING,
    CONTROL_HOLDING,
    DeckLateralConfig,
    DeckLateralController,
)

import pytest


def obs(offset, heading=0.0, valid=True):
    """Build a minimal bridge observation."""
    return {'valid': valid, 'lateral_offset': offset, 'heading_error': heading}


def engaged_controller(**cfg):
    """Return a controller already past its confirmation count, plus the clock."""
    controller = DeckLateralController(DeckLateralConfig(**cfg))
    controller.update(obs(0.0), now_s=0.0)
    return controller


def test_config_rejects_unknown_fields():
    """A typo in a gain name must fail loudly, not be silently ignored."""
    with pytest.raises(AttributeError):
        DeckLateralConfig(k_vy_typo=1.0)


def test_first_valid_frame_does_not_steer():
    """One valid frame is not evidence; the loop must wait for confirmation."""
    controller = DeckLateralController()
    cmd = controller.update(obs(0.20), now_s=0.0)
    assert cmd.state == CONTROL_ENGAGING
    assert cmd.vy == 0.0 and cmd.wz == 0.0
    assert not cmd.engaged


def test_second_valid_frame_engages_and_steers_left_when_offset_positive():
    """A confirmed positive offset steers left, per the observer sign convention."""
    controller = engaged_controller()
    cmd = controller.update(obs(0.20), now_s=0.2)
    assert cmd.state == CONTROL_ACTIVE
    # +offset means the centreline is to the left, so correct left (+vy).
    assert cmd.vy == pytest.approx(0.40 * 0.20)
    assert cmd.engaged


def test_negative_offset_steers_right():
    """A negative offset mirrors the correction to the right."""
    controller = engaged_controller()
    cmd = controller.update(obs(-0.20), now_s=0.2)
    assert cmd.vy == pytest.approx(-0.40 * 0.20)


def test_deadband_keeps_the_loop_quiet_when_centred():
    """Inside the deadband the loop emits nothing, so a centred robot does not wobble."""
    controller = engaged_controller()
    cmd = controller.update(obs(0.015, heading=0.01), now_s=0.2)
    assert cmd.state == CONTROL_ACTIVE
    assert cmd.vy == 0.0
    assert cmd.wz == 0.0


def test_corrections_saturate():
    """Both terms clamp to their configured limits."""
    controller = engaged_controller(max_vy=0.05, max_wz=0.10)
    cmd = controller.update(obs(0.35, heading=0.9), now_s=0.2)
    assert cmd.vy == pytest.approx(0.05)
    assert cmd.wz == pytest.approx(0.10)


def test_heading_term_is_independent_of_lateral_term():
    """Heading and lateral corrections do not leak into each other."""
    controller = engaged_controller()
    cmd = controller.update(obs(0.0, heading=0.10), now_s=0.2)
    assert cmd.vy == 0.0
    assert cmd.wz == pytest.approx(0.80 * 0.10)


def test_implausible_offset_is_rejected_not_acted_on():
    """An offset wider than the deck is a bad fit and must not steer the robot."""
    controller = engaged_controller()
    cmd = controller.update(obs(0.95), now_s=0.2)
    assert cmd.state == CONTROL_HOLDING
    assert cmd.reason == 'observation_stale'


def test_invalid_observation_holds_briefly_then_expires():
    """A dropout rides the last correction briefly, then releases and slows down."""
    controller = engaged_controller()
    controller.update(obs(0.20), now_s=0.2)
    held = controller.update(obs(0.0, valid=False), now_s=0.5)
    assert held.state == CONTROL_HOLDING
    assert held.vy == pytest.approx(0.40 * 0.20)
    assert held.vx_scale == 1.0

    expired = controller.update(obs(0.0, valid=False), now_s=2.0)
    assert expired.state == CONTROL_BLIND
    assert expired.vy == 0.0
    assert expired.vx_scale == pytest.approx(0.5)


def test_blind_before_any_observation_asks_caller_to_slow_down():
    """With nothing ever seen the loop stays disengaged and requests a slower walk."""
    controller = DeckLateralController()
    cmd = controller.update(None, now_s=0.0)
    assert cmd.state == CONTROL_BLIND
    assert cmd.reason == 'no_observation_yet'
    assert cmd.vx_scale == pytest.approx(0.5)
    assert not cmd.engaged


def test_a_single_dropout_restarts_the_confirmation_count():
    """Re-acquisition must re-confirm rather than resume steering immediately."""
    controller = engaged_controller()
    controller.update(obs(0.20), now_s=0.2)
    controller.update(obs(0.0, valid=False), now_s=0.4)
    resumed = controller.update(obs(0.20), now_s=0.6)
    assert resumed.state == CONTROL_ENGAGING
    assert resumed.vy == 0.0


def test_missing_fields_are_treated_as_invalid():
    """A valid flag without the actual fields is not usable."""
    controller = engaged_controller()
    cmd = controller.update({'valid': True}, now_s=0.2)
    assert cmd.state == CONTROL_HOLDING


def test_nan_offset_is_rejected():
    """Reject a NaN offset before it can reach the gain."""
    controller = engaged_controller()
    cmd = controller.update(obs(float('nan')), now_s=0.2)
    assert cmd.state == CONTROL_HOLDING


def test_reset_disengages_completely():
    """Reset clears the accepted history, not just the output."""
    controller = engaged_controller()
    controller.update(obs(0.20), now_s=0.2)
    controller.reset()
    cmd = controller.update(None, now_s=0.3)
    assert cmd.state == CONTROL_BLIND
    assert cmd.reason == 'no_observation_yet'


def test_to_dict_round_trips_the_decision():
    """The evidence record carries the inputs that produced the correction."""
    controller = engaged_controller()
    cmd = controller.update(obs(0.20, heading=0.05), now_s=0.2)
    payload = cmd.to_dict()
    assert payload['state'] == CONTROL_ACTIVE
    assert payload['lateral_offset'] == pytest.approx(0.20)
    assert payload['heading_error'] == pytest.approx(0.05)
    assert payload['vy'] == pytest.approx(0.08)


def test_measured_sim_drift_would_have_been_corrected():
    """Replay the 2026-08-04 measured drift; the loop must push left throughout."""
    measured = [0.0395, 0.0471, 0.0461, 0.0487, 0.0739,
                0.0865, 0.1135, 0.1431, 0.1485, 0.1663, 0.1821, 0.1693]
    controller = DeckLateralController()
    commands = [controller.update(obs(v), now_s=0.2 * i)
                for i, v in enumerate(measured)]
    steering = [c for c in commands if c.state == CONTROL_ACTIVE]
    assert len(steering) == len(measured) - 1
    assert all(c.vy > 0.0 for c in steering)
    assert max(c.vy for c in steering) <= DeckLateralConfig().max_vy


def centring_controller(**cfg):
    """Return an engaged controller with the centre-first hold enabled."""
    cfg.setdefault('centre_first_offset_m', 0.06)
    cfg.setdefault('centre_first_release_m', 0.03)
    cfg.setdefault('centre_first_vx_scale', 0.0)
    return engaged_controller(**cfg)


def test_centre_first_is_off_by_default():
    """Existing profiles must be bit-for-bit unchanged until they opt in."""
    controller = engaged_controller()
    cmd = controller.update(obs(0.20), now_s=0.2)
    assert cmd.vx_scale == 1.0
    assert cmd.reason == 'ok'


def test_centre_first_stops_forward_progress_when_badly_off_line():
    """Past the threshold the robot must fix the offset before advancing."""
    controller = centring_controller()
    cmd = controller.update(obs(0.13), now_s=0.2)
    assert cmd.vx_scale == 0.0
    assert cmd.reason == 'centring'
    assert cmd.vy > 0.0


def test_centre_first_leaves_a_centred_robot_walking():
    """Inside the threshold nothing changes; this must not cost speed."""
    controller = centring_controller()
    cmd = controller.update(obs(0.04), now_s=0.2)
    assert cmd.vx_scale == 1.0
    assert cmd.reason == 'ok'


def test_centre_first_has_hysteresis():
    """Between the two thresholds the latch keeps its previous state."""
    controller = centring_controller()
    assert controller.update(obs(0.13), now_s=0.2).vx_scale == 0.0
    # 0.045 is below the engage threshold but above the release threshold.
    assert controller.update(obs(0.045), now_s=0.4).vx_scale == 0.0
    assert controller.update(obs(0.02), now_s=0.6).vx_scale == 1.0
    assert controller.update(obs(0.045), now_s=0.8).vx_scale == 1.0


def test_centre_first_survives_a_single_dropout():
    """A lost frame is not evidence the robot got back on the line."""
    controller = centring_controller()
    controller.update(obs(0.13), now_s=0.2)
    held = controller.update(obs(0.0, valid=False), now_s=0.4)
    assert held.state == CONTROL_HOLDING
    assert held.vx_scale == 0.0


def test_centre_first_releases_when_the_observer_expires():
    """With no measurement left, blind's own slow-down takes over."""
    controller = centring_controller()
    controller.update(obs(0.13), now_s=0.2)
    expired = controller.update(obs(0.0, valid=False), now_s=5.0)
    assert expired.state == CONTROL_BLIND
    assert expired.vx_scale == pytest.approx(0.5)


def test_centre_first_replays_the_measured_corner_exit():
    """At the measured corner-exit offset the robot must not step forward.

    Ground truth 2026-08-04: corner 1 ends 0.13 m off the ring rail centre with
    the inner void 0.14 m away, and the observer read +0.127.
    """
    controller = centring_controller()
    controller.update(obs(0.1227), now_s=0.0)
    cmd = controller.update(obs(0.1271), now_s=0.1)
    assert cmd.vx_scale == 0.0
    # Note the recovery rate this implies: the default 0.40 gain only asks for
    # 0.051 m/s, so closing 0.127 m takes ~2.5 s.  That is affordable only
    # because vx is held at zero meanwhile.
    assert cmd.vy == pytest.approx(0.40 * 0.1271)


def test_centre_first_gives_up_rather_than_stalling_the_segment():
    """Side-stepping that is not working must not hold vx at zero forever."""
    controller = centring_controller(centre_first_max_s=2.0)
    assert controller.update(obs(0.13), now_s=0.0).vx_scale == 0.0
    assert controller.update(obs(0.13), now_s=1.9).vx_scale == 0.0
    late = controller.update(obs(0.13), now_s=2.1)
    assert late.vx_scale == 1.0
    assert late.reason == 'centring_timed_out'
    # The lateral correction is still applied — only the hold is abandoned.
    assert late.vy > 0.0


def test_centre_first_budget_is_restored_once_back_on_the_line():
    """A later deviation gets its own budget, not the exhausted one."""
    controller = centring_controller(centre_first_max_s=2.0)
    controller.update(obs(0.13), now_s=0.0)
    assert controller.update(obs(0.13), now_s=3.0).reason == 'centring_timed_out'
    assert controller.update(obs(0.01), now_s=3.2).reason == 'ok'
    again = controller.update(obs(0.13), now_s=3.4)
    assert again.vx_scale == 0.0
    assert again.reason == 'centring'
