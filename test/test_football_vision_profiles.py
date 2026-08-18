#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-4 far acquisition must survive Stage-6's near-field ball tuning.

The Stage-6 rework on the robot retuned ``detect_football`` in place for the
final close push: minimum radius 7 -> 38 px, candidates confined to the lower
image half, and a hard ``depth_m > 1.00`` rejection.  ``detect_football`` is
also Stage 4's only ball detector, and Stage 4 acquires the ball from several
metres away — so importing that file as-is would have made Stage 4 blind past
about a metre.

The range-dependent gates are therefore keyword arguments whose defaults are
the far-field Stage-4 behaviour; Stage 6 opts in with ``**NEAR_BALL``.
"""

import cv2
import numpy as np

import pytest

from control_node.football_vision import NEAR_BALL, detect_football


def ball_scene(cx, cy, radius, width=640, height=480):
    """A plain floor with one black-and-white ball on it.

    Slight blur and noise are deliberate: a pixel-perfect synthetic circle
    gives Canny nothing to find, and every candidate dies on ``edge_fraction``.
    """
    image = np.full((height, width, 3), 150, np.uint8)
    cv2.circle(image, (cx, cy), radius, (240, 240, 240), -1)
    for angle in (20, 140, 260):
        px = int(cx + 0.42 * radius * np.cos(np.deg2rad(angle)))
        py = int(cy + 0.42 * radius * np.sin(np.deg2rad(angle)))
        cv2.circle(image, (px, py), max(2, int(radius * 0.30)), (30, 30, 30), -1)
    image = cv2.GaussianBlur(image, (3, 3), 0.8)
    noise = np.random.RandomState(0).randint(-6, 7, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def constant_depth(value):
    return lambda x, y: value


def test_far_small_ball_is_still_found_by_default():
    """Stage 4's case: a 12 px ball three metres out."""
    scene = ball_scene(320, 300, 12)
    found = detect_football(scene, depth_at=constant_depth(3.0))
    assert found is not None
    assert found['radius'] == pytest.approx(12, abs=3)


def test_near_profile_rejects_the_far_small_ball():
    """The same frame under Stage 6's profile: too small to be the push ball."""
    scene = ball_scene(320, 300, 12)
    assert detect_football(scene, depth_at=constant_depth(3.0), **NEAR_BALL) is None


def test_both_profiles_find_the_close_push_ball():
    scene = ball_scene(320, 400, 70)
    assert detect_football(scene, depth_at=constant_depth(0.6)) is not None
    assert detect_football(scene, depth_at=constant_depth(0.6), **NEAR_BALL) is not None


def test_depth_gate_alone_separates_the_profiles():
    """A ball big enough for both radius windows, but beyond Stage 6's 1.00 m."""
    scene = ball_scene(320, 380, 40)
    assert detect_football(scene, depth_at=constant_depth(1.2)) is not None
    assert detect_football(scene, depth_at=constant_depth(1.2), **NEAR_BALL) is None


def test_defaults_are_the_far_field_values():
    """Guard against the near-field numbers being hard-coded back in."""
    import inspect

    defaults = {}
    signature = inspect.signature(detect_football)
    for name, param in signature.parameters.items():
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default

    assert defaults['min_radius_px'] == 7
    assert defaults['min_radius_ratio'] == 0.015
    assert defaults['min_y_ratio'] == 0.34
    assert defaults['max_depth_m'] is None
    assert defaults['min_dark_panels'] == 0

    for key in defaults:
        if key in NEAR_BALL:
            assert NEAR_BALL[key] != defaults[key], (
                '%s is identical in both profiles; either it is not '
                'range-dependent or a default drifted' % key)


def test_stage6_asks_for_the_near_profile():
    """Stage 6 must actually pass NEAR_BALL, or its castor rejection is off."""
    import control_node.stage6_node as stage6

    source = inspect_source(stage6)
    assert 'NEAR_BALL' in source


def inspect_source(module):
    import inspect

    return inspect.getsource(module)


def decoy_scene(ball, decoy, width=640, height=480):
    """A ball plus one big pale disc -- the shape that beat it on hardware.

    ``ball`` and ``decoy`` are ``(cx, cy, radius)``.  The decoy gets a single
    dark blob rather than separate panels: enough dark pixels to clear the
    black-fraction gate and become a candidate, but no panel structure.  Only
    a score that rewards size or nearness can rank it above the ball.
    """
    image = ball_scene(ball[0], ball[1], ball[2], width, height)
    cv2.circle(image, (decoy[0], decoy[1]), decoy[2], (225, 225, 228), -1)
    cv2.circle(image, (decoy[0], decoy[1]), decoy[2], (120, 120, 120), 2)
    cv2.circle(image, (decoy[0], decoy[1] - decoy[2] // 5),
               max(4, decoy[2] // 3), (35, 35, 35), -1)
    image = cv2.GaussianBlur(image, (3, 3), 0.8)
    noise = np.random.RandomState(1).randint(-6, 7, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_evidence_score_prefers_the_ball_over_a_bigger_nearer_blob():
    """The measured hardware failure: a large near blob outscored the ball.

    On the spare robot a bright floor blob at 0.29 m scored 5.87 under the
    legacy ``2.8 * radius_score + 0.35 / depth`` while the real ball scored
    4.00.  The evidence score must not have that preference.
    """
    scene = decoy_scene(ball=(240, 300, 26), decoy=(430, 330, 62))

    def depth_at(x, y):
        return 0.5 if x > 340 else 1.9

    found = detect_football(scene, depth_at=depth_at)
    assert found is not None
    assert found['x'] == pytest.approx(240, abs=25), (
        'evidence scoring locked onto the decoy at %.0f' % found['x'])

    # The same frame under the old ranking picks the decoy, which is what
    # makes this scene a regression test rather than a tautology.
    legacy = detect_football(scene, depth_at=depth_at, score_mode='legacy')
    assert legacy is not None
    assert legacy['x'] == pytest.approx(430, abs=25)


def test_legacy_score_is_still_reachable_for_stage6():
    """Stage 6 keeps the old ranking until its castor bench is redone."""
    assert NEAR_BALL['score_mode'] == 'legacy'
    assert NEAR_BALL['blur_ksize'] == 9
    assert NEAR_BALL['blur_sigma'] == 1.8


def test_prefer_center_is_a_tie_breaker_not_an_override():
    """Continuity must not let a stale track drag the lock off the ball."""
    scene = decoy_scene(ball=(240, 300, 26), decoy=(430, 330, 30))

    def depth_at(x, y):
        return 1.9

    free = detect_football(scene, depth_at=depth_at)
    assert free is not None and free['x'] == pytest.approx(240, abs=25)

    # Pointing continuity straight at the decoy does not move the lock:
    # the ball still wins on its own evidence.
    dragged = detect_football(
        scene, depth_at=depth_at, prefer_center=(430, 330))
    assert dragged is not None
    assert dragged['x'] == pytest.approx(240, abs=25)

    # Continuity does raise the score of the candidate it points at, which is
    # what breaks a tie between two otherwise equal candidates.
    held = detect_football(scene, depth_at=depth_at, prefer_center=(240, 300))
    assert held is not None
    assert held['score'] > free['score']


def test_prefer_center_never_invents_a_candidate():
    """An empty floor stays empty however confident the previous track was."""
    empty = np.full((480, 640, 3), 150, np.uint8)
    assert detect_football(
        empty, depth_at=constant_depth(1.9), prefer_center=(320, 300)) is None


def test_default_pre_blur_keeps_the_two_metre_ball_a_candidate():
    """A ~22 px ball is what 2 m looks like; the old 9x9 kernel erased it."""
    scene = ball_scene(320, 300, 22)
    assert detect_football(scene, depth_at=constant_depth(1.9)) is not None
