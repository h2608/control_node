#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Route-frame lateral observation built from odometry alone.

``bridge_perception`` can only see the deck centreline while deck geometry is
actually in the depth frustum.  Measured in ``race.world``, that holds for the
climb and stops within ~2 s of the ramp top: past it the camera looks at open
ground, the observer reports ``observation_expired``, and every remaining
segment runs open-loop.  That is where Stage 5 has been failing.

This module supplies the same *shape* of observation from a completely
different source: the segment's entry pose defines a reference line, and the
state estimator says where the body is relative to it.  Emitting the same
``{'valid', 'lateral_offset', 'heading_error'}`` contract as the depth observer
means ``DeckLateralController`` consumes either one unchanged, and the two can
be laddered — depth while it can see, odometry when it cannot.

**What this cannot do.**  The reference is the segment entry pose, not the real
course centreline.  Enter a segment 10 cm off centre and this will hold the
robot 10 cm off centre for the whole segment: it converts an unbounded drift
into a bounded offset, which is the useful part, but it never re-centres.  It
also inherits estimator position drift, so ``heading_error`` (attitude, which
the estimator gets essentially right) is far more trustworthy than
``lateral_offset`` (integrated position).  ``USE_HEADING_ONLY`` exists for
exactly that reason.

Sign conventions match ``bridge_perception``/``deck_lateral``:
``lateral_offset`` > 0 means the reference line lies to the body's left, so the
correction is +vy; ``heading_error`` > 0 means the reference direction points
front-left, so the correction is +wz.
"""

import math

SOURCE_ODOMETRY = 'odometry'

#: Emit both terms.
USE_FULL = 'full'
#: Emit the heading term only and report ``lateral_offset`` as 0.0.
#: Use when estimator position drift is untrusted but attitude is not.
USE_HEADING_ONLY = 'heading_only'

#: Beyond this the reference line is behind a bad fix, not a real deviation.
MAX_PLAUSIBLE_LATERAL_M = 0.60
#: Beyond this the body is not following the segment at all; a lateral
#: correction is meaningless and the caller should be looking at the corner
#: logic instead.
MAX_PLAUSIBLE_HEADING_RAD = math.pi / 3.0


#: Stage 5's route is a rectangular ring: every corner is a declared +/-90 deg.
#: A segment's true heading is therefore always the route's start heading plus
#: a whole number of quarter turns.
HEADING_QUANTUM_RAD = math.pi / 2.0
#: How far the measured entry yaw may sit from the nearest quarter turn before
#: snapping becomes a guess rather than a correction.
MAX_SNAP_RAD = math.radians(30.0)


def wrap_rad(angle):
    """Wrap an angle into (-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def snap_reference_heading(entry_yaw, base_yaw, quantum_rad=HEADING_QUANTUM_RAD,
                           max_snap_rad=MAX_SNAP_RAD):
    """Round a measured segment-entry yaw onto the route's declared grid.

    Returns ``(reference_yaw, snap_error_rad)``, or ``(None, snap_error_rad)``
    when the entry yaw is too far off the grid to snap honestly.

    This matters more than it looks.  Measured in ``race.world``, corner 1 ends
    with the body at yaw -3.053 rad while the rail actually runs at -pi: 0.089
    rad out.  Holding *that* heading over the 3.23 m rail walks the robot 0.29 m
    sideways — wider than the rail's 0.25 m half-width, so a loop referenced to
    the measured entry yaw would drive it off the edge while reporting zero
    error.  Snapping to the declared quarter turn removes the corner's own
    sloppiness from the reference instead of integrating it.
    """
    if entry_yaw is None or base_yaw is None:
        return None, None
    try:
        entry_yaw = float(entry_yaw)
        base_yaw = float(base_yaw)
        quantum_rad = float(quantum_rad)
    except (TypeError, ValueError):
        return None, None
    if quantum_rad <= 0.0:
        return None, None
    turns = round(wrap_rad(entry_yaw - base_yaw) / quantum_rad)
    reference = wrap_rad(base_yaw + turns * quantum_rad)
    error = wrap_rad(entry_yaw - reference)
    if abs(error) > max_snap_rad:
        return None, error
    return reference, error


def _invalid(reason):
    """Build a rejected observation carrying why it was rejected."""
    return {
        'valid': False,
        'source': SOURCE_ODOMETRY,
        'reason': reason,
        'lateral_offset': None,
        'heading_error': None,
    }


def route_frame_observation(origin_xy, reference_yaw, x, y, yaw,
                            odometry_valid=True, mode=USE_FULL,
                            line_offset_m=0.0):
    """Return a deck-centring observation relative to a segment reference line.

    ``origin_xy``/``reference_yaw`` define the line; ``x``/``y``/``yaw`` are the
    current estimator pose.  The result is directly consumable by
    ``DeckLateralController.update``.

    ``line_offset_m`` shifts that line sideways, and it is what stops this
    fallback from doing active harm.  Without it the line runs through the
    segment's entry pose, so a body that entered 0.20 m off centre reports
    *zero* error and the centre-first hold never fires — measured in
    ``race.world``: a run entered straight_1 0.205 m off the rail centre, the
    fallback's first observation read ``offset=0.0``, and it held that error
    down the whole rail and off the end.  Anchoring the line to the last depth
    fix instead (see ``anchor_from_depth``) makes the fallback dead-reckon from
    a course-referenced measurement rather than from wherever the corner
    happened to leave the body.
    """
    if not odometry_valid:
        return _invalid('odometry_invalid')
    if origin_xy is None or reference_yaw is None:
        return _invalid('no_reference')
    try:
        x0, y0 = float(origin_xy[0]), float(origin_xy[1])
        ref = float(reference_yaw)
        x, y, yaw = float(x), float(y), float(yaw)
    except (TypeError, ValueError, IndexError):
        return _invalid('bad_reference')
    for value in (x0, y0, ref, x, y, yaw):
        if value != value:                                      # NaN
            return _invalid('nan_input')

    heading_error = wrap_rad(ref - yaw)
    if abs(heading_error) > MAX_PLAUSIBLE_HEADING_RAD:
        return _invalid('heading_implausible')

    try:
        line_offset_m = float(line_offset_m)
    except (TypeError, ValueError):
        return _invalid('bad_line_offset')
    if line_offset_m != line_offset_m:
        return _invalid('bad_line_offset')

    if mode == USE_HEADING_ONLY:
        return {
            'valid': True,
            'source': SOURCE_ODOMETRY,
            'reason': 'ok',
            'mode': USE_HEADING_ONLY,
            'lateral_offset': 0.0,
            'heading_error': heading_error,
            'cross_track_m': None,
        }

    # Cross-track offset of the body from the reference line, left-positive in
    # the *reference* frame...
    cross = -math.sin(ref) * (x - x0) + math.cos(ref) * (y - y0)
    # ...then projected onto the body's own left axis, because that is the axis
    # vy actually moves along.  At small heading errors the factor is ~1.
    lateral_offset = -cross * math.cos(heading_error) + line_offset_m
    if abs(lateral_offset) > MAX_PLAUSIBLE_LATERAL_M:
        return _invalid('lateral_implausible')

    return {
        'valid': True,
        'source': SOURCE_ODOMETRY,
        'reason': 'ok',
        'mode': USE_FULL,
        'lateral_offset': lateral_offset,
        'heading_error': heading_error,
        'cross_track_m': cross,
        'line_offset_m': line_offset_m,
    }


def anchor_from_depth(depth_offset_m, unanchored_odometry_offset_m,
                      max_anchor_m=MAX_PLAUSIBLE_LATERAL_M):
    """Return the line shift that makes odometry agree with a depth fix.

    Call this on every tick where both sources are valid.  The result is the
    constant that, added to the raw odometry observation, reproduces what depth
    measured — so when depth drops out, the fallback carries on from the last
    course-referenced fix instead of resetting its idea of centre to wherever
    the body currently is.

    Returns ``None`` when either input is unusable or the implied shift is
    larger than a deck half-width can explain, because silently applying a
    nonsense anchor is worse than holding the previous one.
    """
    try:
        depth_offset_m = float(depth_offset_m)
        unanchored_odometry_offset_m = float(unanchored_odometry_offset_m)
    except (TypeError, ValueError):
        return None
    if depth_offset_m != depth_offset_m:
        return None
    if unanchored_odometry_offset_m != unanchored_odometry_offset_m:
        return None
    anchor = depth_offset_m - unanchored_odometry_offset_m
    if abs(anchor) > max_anchor_m:
        return None
    return anchor


def select_observation(depth_observation, odometry_observation):
    """Pick the depth observation when usable, else the odometry one.

    The ladder is deliberately one-way per tick and stateless: engagement,
    confirmation, and staleness are ``DeckLateralController``'s job, and giving
    two components a say over the same hysteresis is how loops start fighting.
    A source swap is reported so the caller can log it, because a silent swap
    between a course-referenced and a dead-reckoned line is exactly the kind of
    thing that must never be invisible in a failure trace.
    """
    if isinstance(depth_observation, dict) and depth_observation.get('valid'):
        chosen = dict(depth_observation)
        chosen.setdefault('source', 'depth')
        return chosen, chosen['source']
    if isinstance(odometry_observation, dict) and odometry_observation.get('valid'):
        return dict(odometry_observation), SOURCE_ODOMETRY
    return None, 'none'
