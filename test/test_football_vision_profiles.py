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
