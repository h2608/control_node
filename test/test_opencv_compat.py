#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guard the OpenCV 3 / OpenCV 4 findContours split.

The physical robot runs **OpenCV 3.2.0**, where ``cv2.findContours`` returns
``(image, contours, hierarchy)``.  The simulation container runs OpenCV 4.2.0,
where it returns ``(contours, hierarchy)``.  So ``contours, _ =
cv2.findContours(...)`` passes every test in the container and then raises
``ValueError: too many values to unpack (expected 2)`` on the first camera
frame on the robot, killing the node.

Measured on 2026-08-18: Stage 1 was activated on the physical robot, sent its
stand command, and died 0.1 s later on exactly this line — the state machine
never reached cruise.

Use ``stage_common.find_contours()``, which slices ``[-2]`` and works on both.
"""

import ast
import os

import pytest


PKG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'control_node')


def python_sources():
    """Every package module, excluding robot-side .bak snapshots."""
    for name in sorted(os.listdir(PKG_DIR)):
        if name.endswith('.py'):
            yield name, os.path.join(PKG_DIR, name)


def find_contours_calls(tree):
    """Yield every ``cv2.findContours`` Call node."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == 'findContours'
                and isinstance(func.value, ast.Name) and func.value.id == 'cv2'):
            yield node


def unpacked_directly(tree, call):
    """True if this call is assigned straight into a tuple target."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.value is call:
            return any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets)
    return False


def sliced(tree, call):
    """True if the call result is immediately subscripted, e.g. ``[-2]``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and node.value is call:
            return True
    return False


@pytest.mark.parametrize('name,path', list(python_sources()))
def test_findcontours_result_is_version_agnostic(name, path):
    """No module may assume the OpenCV 4 two-value findContours return."""
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=path)

    offenders = [
        call.lineno for call in find_contours_calls(tree)
        if unpacked_directly(tree, call) or not sliced(tree, call)
    ]
    assert not offenders, (
        '%s unpacks cv2.findContours() without a version-agnostic slice at '
        'line(s) %s. The robot runs OpenCV 3.2.0 and returns three values; '
        'use stage_common.find_contours() instead.'
        % (name, ', '.join(str(n) for n in offenders)))


def test_find_contours_helper_returns_contours_on_this_opencv():
    """The helper itself must work on whichever OpenCV is installed here."""
    import numpy as np

    from control_node.stage_common import find_contours

    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    contours = find_contours(mask)
    assert len(contours) == 1
    assert len(contours[0]) >= 4
