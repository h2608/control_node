#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Declarative Stage 5 route model (STAGE5_PHYSICAL_REDESIGN_PLAN.md §4.1/§4.2).

Pure Python, no ROS and no node state, so the whole gating policy is unit
testable without a simulator.  ``Stage5Node`` owns the sensors, commands and
state names; this module owns only:

* the **segment table**: what the course is made of, how long each segment is
  expected to be, what evidence may end it, and which fallback tier it runs in;
* **progress tracking** from odometry samples (path length and unwrapped yaw),
  with explicit rejection of implausible jumps and of stale/frozen streams;
* the **two-source gate**: an exit trigger from perception is only allowed to
  fire while odometry progress is inside the segment's expected window
  (§1 "Every transition needs two independent sources").

Distances are in metres, angles in degrees at the parameter surface and in
radians only inside the yaw math.  The default table carries *drawing-derived*
physical numbers on purpose: no simulator-tuned value may become a default
(plan §1 "Keep what is robot-proven, retune everything").  The Gazebo profile
overrides the windows explicitly in ``config/stage5_sim.yaml``.
"""

import math

# Progress measure used to judge a segment.
PROGRESS_DISTANCE = 'distance'
PROGRESS_YAW = 'yaw'
PROGRESS_NONE = 'none'

# How a segment's declared exit evidence is treated.
#   ENFORCE  - perception trigger is gated by the odometry window (two sources)
#   MONITOR  - transition owned by the state itself; window only observed/logged
ENFORCEMENT_ENFORCE = 'enforce'
ENFORCEMENT_MONITOR = 'monitor'

# Which source ends a segment.
#   EXIT_VISION    - perception says "segment over", odometry window may veto it
#                    (two independent sources, plan §1)
#   EXIT_ODOMETRY  - odometry distance alone ends the segment.  This is the
#                    declared, bounded dead-reckoning tier: on a straight there
#                    is no second source, so such a segment must run at
#                    TIER_DEAD_RECKONING and stays bounded by max_m and the
#                    per-state timeout.  Corners keep their yaw verification,
#                    which *is* an independent check.
EXIT_VISION = 'vision'
EXIT_ODOMETRY = 'odometry'

# Gate decisions returned by :func:`evaluate_gate`.
GATE_UNAVAILABLE = 'unavailable'   # no usable odometry
GATE_BELOW_MIN = 'below_min'       # too early: suppress the perception trigger
GATE_IN_WINDOW = 'in_window'       # inside the expected window
GATE_OVERRUN = 'overrun'           # past the maximum: route violation

# Fallback ladder tiers (plan §3 B3).
TIER_NOMINAL = 1
TIER_SINGLE_EDGE = 2
TIER_DEAD_RECKONING = 3
TIER_FAULT_STOP = 4


class RouteSegment(object):
    """One declarative course segment.

    Attributes mirror the plan's segment-table columns.  ``states`` lists the
    ``Stage5Node`` state names that belong to the segment; a state may appear in
    exactly one segment.
    """

    __slots__ = (
        'name', 'states', 'progress', 'expected_m', 'min_m', 'max_m',
        'expected_yaw_deg', 'yaw_tol_deg', 'lateral_profile', 'entry_evidence',
        'exit_evidence', 'enforcement', 'exit_source', 'speed_cap_mps',
        'fallback_tier', 'degraded_next_state', 'reference',
    )

    def __init__(
        self,
        name,
        states,
        progress=PROGRESS_NONE,
        expected_m=0.0,
        min_m=0.0,
        max_m=0.0,
        expected_yaw_deg=0.0,
        yaw_tol_deg=0.0,
        lateral_profile='flat',
        entry_evidence='',
        exit_evidence='',
        enforcement=ENFORCEMENT_MONITOR,
        exit_source=EXIT_VISION,
        speed_cap_mps=0.0,
        fallback_tier=TIER_NOMINAL,
        degraded_next_state='',
        reference='',
    ):
        """Build one segment; every field is a declared course property."""
        self.name = str(name)
        self.states = tuple(str(s) for s in states)
        self.progress = str(progress)
        self.expected_m = float(expected_m)
        self.min_m = float(min_m)
        self.max_m = float(max_m)
        self.expected_yaw_deg = float(expected_yaw_deg)
        self.yaw_tol_deg = float(yaw_tol_deg)
        self.lateral_profile = str(lateral_profile)
        self.entry_evidence = str(entry_evidence)
        self.exit_evidence = str(exit_evidence)
        self.enforcement = str(enforcement)
        self.exit_source = str(exit_source)
        self.speed_cap_mps = float(speed_cap_mps)
        self.fallback_tier = int(fallback_tier)
        self.degraded_next_state = str(degraded_next_state)
        self.reference = str(reference)

    @property
    def enforced(self):
        """Return True when this segment's exit needs both evidence sources."""
        return self.enforcement == ENFORCEMENT_ENFORCE

    @property
    def window(self):
        """Expected-distance window as ``(min_m, max_m)``."""
        return (self.min_m, self.max_m)

    @property
    def odometry_exit(self):
        """Return True when odometry distance alone ends this segment."""
        return self.exit_source == EXIT_ODOMETRY

    def replace(self, **kwargs):
        """Return a copy with the given fields replaced (parameter overrides)."""
        values = {slot: getattr(self, slot) for slot in self.__slots__}
        values.update(kwargs)
        return RouteSegment(**values)

    def to_dict(self):
        """Return the whole row as a plain dict (evidence logging)."""
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self):  # pragma: no cover - debug aid only
        """Return a short identifying representation."""
        return 'RouteSegment(name={!r}, states={!r})'.format(self.name, self.states)


def default_segments():
    """Drawing-derived default route table.

    Lengths come from the v3.1 drawing / replay geometry recorded in
    ``AGENTS.md`` and ``STAGE5_PHYSICAL_REDESIGN_PLAN.md`` §0.2.  Every entry
    records where its number came from; ``unmeasured`` means Phase E must
    measure it on the mock-up before the window means anything.  Windows are
    deliberately wide: their job is to reject a grossly early or grossly late
    trigger, not to time the course.
    """
    return (
        RouteSegment(
            name='entry_step_up',
            states=('P5_STEP_UP',),
            progress=PROGRESS_DISTANCE,
            expected_m=0.50, min_m=0.10, max_m=1.20,
            lateral_profile='flat',
            entry_evidence='stage activation + body attitude set',
            exit_evidence='timed step-up motion',
            enforcement=ENFORCEMENT_MONITOR,
            speed_cap_mps=0.25,
            reference='entrance 50 cm wide / 5 cm high; along-travel length unmeasured',
        ),
        RouteSegment(
            name='up_slope',
            states=('P5_UP_SLOPE',),
            progress=PROGRESS_DISTANCE,
            expected_m=3.00, min_m=2.00, max_m=4.00,
            lateral_profile='trapezoid_left_high',
            entry_evidence='entry step-up completed',
            exit_evidence='right-side course boundary lost on fresh frames',
            enforcement=ENFORCEMENT_ENFORCE,
            speed_cap_mps=0.25,
            degraded_next_state='P5_AFTER_UP_SLOPE_FORWARD',
            fallback_tier=TIER_NOMINAL,
            reference='drawing 300 cm straight class',
        ),
        RouteSegment(
            name='up_slope_exit',
            states=('P5_AFTER_UP_SLOPE_FORWARD',),
            progress=PROGRESS_DISTANCE,
            expected_m=0.30, min_m=0.05, max_m=0.90,
            lateral_profile='trapezoid_left_high',
            entry_evidence='up-slope exit evidence',
            exit_evidence='timed forward motion into the corner',
            enforcement=ENFORCEMENT_MONITOR,
            speed_cap_mps=0.25,
            reference='unmeasured corner approach',
        ),
        RouteSegment(
            name='corner_1',
            states=('P5_AFTER_UP_SLOPE_VELOCITY_CONTROL',),
            progress=PROGRESS_YAW,
            expected_yaw_deg=90.0, yaw_tol_deg=30.0,
            lateral_profile='flat',
            entry_evidence='corner approach completed',
            exit_evidence='turn action completed + odometry yaw delta verified',
            enforcement=ENFORCEMENT_ENFORCE,
            reference='ring corner, 90 deg; sign follows the configured turn action',
        ),
        RouteSegment(
            name='straight_1',
            states=('P5_RIGHT_SLOPE_1',),
            progress=PROGRESS_DISTANCE,
            expected_m=4.00, min_m=2.50, max_m=5.20,
            lateral_profile='trapezoid_left_high',
            entry_evidence='corner 1 yaw verified',
            exit_evidence='center course boundary absent on fresh frames',
            enforcement=ENFORCEMENT_ENFORCE,
            speed_cap_mps=0.25,
            degraded_next_state='P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST',
            reference='drawing 400 cm straight class',
        ),
        RouteSegment(
            name='straight_1_exit',
            states=('P5_RIGHT_SLOPE_1_FORWARD_AFTER_CENTER_LOST',),
            progress=PROGRESS_DISTANCE,
            expected_m=0.30, min_m=0.05, max_m=0.90,
            lateral_profile='trapezoid_left_high',
            entry_evidence='straight 1 exit evidence',
            exit_evidence='timed forward motion into the corner',
            enforcement=ENFORCEMENT_MONITOR,
            speed_cap_mps=0.25,
            reference='unmeasured corner approach',
        ),
        RouteSegment(
            name='corner_2',
            states=('P5_TURN_1',),
            progress=PROGRESS_YAW,
            expected_yaw_deg=-90.0, yaw_tol_deg=30.0,
            lateral_profile='flat',
            entry_evidence='straight 1 corner approach completed',
            exit_evidence='turn action completed + odometry yaw delta verified',
            enforcement=ENFORCEMENT_ENFORCE,
            reference='ring corner, 90 deg; sign follows the configured turn action',
        ),
        RouteSegment(
            name='straight_2',
            states=('P5_RIGHT_SLOPE_2',),
            progress=PROGRESS_DISTANCE,
            expected_m=3.00, min_m=1.80, max_m=4.20,
            lateral_profile='trapezoid_left_high',
            entry_evidence='corner 2 yaw verified',
            exit_evidence='center course boundary absent on fresh frames',
            enforcement=ENFORCEMENT_ENFORCE,
            speed_cap_mps=0.25,
            degraded_next_state='P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST',
            reference='drawing 300 cm straight class',
        ),
        RouteSegment(
            name='straight_2_exit',
            states=('P5_RIGHT_SLOPE_2_FORWARD_AFTER_CENTER_LOST',),
            progress=PROGRESS_DISTANCE,
            expected_m=0.30, min_m=0.05, max_m=0.90,
            lateral_profile='trapezoid_left_high',
            entry_evidence='straight 2 exit evidence',
            exit_evidence='timed forward motion into the corner',
            enforcement=ENFORCEMENT_MONITOR,
            speed_cap_mps=0.25,
            reference='unmeasured corner approach',
        ),
        RouteSegment(
            name='corner_3',
            states=('P5_TURN_2',),
            progress=PROGRESS_YAW,
            expected_yaw_deg=-90.0, yaw_tol_deg=30.0,
            lateral_profile='flat',
            entry_evidence='straight 2 corner approach completed',
            exit_evidence='turn action completed + odometry yaw delta verified',
            enforcement=ENFORCEMENT_ENFORCE,
            reference='ring corner, 90 deg; sign follows the configured turn action',
        ),
        RouteSegment(
            name='straight_3',
            states=('P5_RIGHT_SLOPE_3',),
            progress=PROGRESS_DISTANCE,
            expected_m=4.00, min_m=2.50, max_m=5.20,
            lateral_profile='trapezoid_left_high',
            entry_evidence='corner 3 yaw verified',
            exit_evidence='center course boundary absent on fresh frames',
            enforcement=ENFORCEMENT_ENFORCE,
            speed_cap_mps=0.25,
            degraded_next_state='P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST',
            reference='drawing 400 cm straight class',
        ),
        RouteSegment(
            name='straight_3_exit',
            states=('P5_RIGHT_SLOPE_3_FORWARD_AFTER_CENTER_LOST',),
            progress=PROGRESS_DISTANCE,
            expected_m=0.30, min_m=0.05, max_m=0.90,
            lateral_profile='trapezoid_left_high',
            entry_evidence='straight 3 exit evidence',
            exit_evidence='timed forward motion to the corner jump',
            enforcement=ENFORCEMENT_MONITOR,
            speed_cap_mps=0.25,
            reference='unmeasured corner approach',
        ),
        RouteSegment(
            name='corner_4',
            states=('P5_RIGHT_JUMP_AFTER_RESET_BODY',),
            progress=PROGRESS_YAW,
            expected_yaw_deg=-90.0, yaw_tol_deg=30.0,
            lateral_profile='flat',
            entry_evidence='straight 3 corner approach completed, body reset',
            exit_evidence='turn jump completed + odometry yaw delta verified',
            enforcement=ENFORCEMENT_ENFORCE,
            reference='ring corner, 90 deg; sign follows the configured jump action',
        ),
        RouteSegment(
            name='right_descent_align',
            states=('P5_ALIGN_AFTER_RIGHT_JUMP',),
            progress=PROGRESS_DISTANCE,
            expected_m=0.10, min_m=0.0, max_m=0.60,
            lateral_profile='flat',
            entry_evidence='corner 4 yaw verified',
            exit_evidence='timed post-jump alignment motion',
            enforcement=ENFORCEMENT_MONITOR,
            speed_cap_mps=0.20,
            reference='unmeasured post-jump alignment',
        ),
        RouteSegment(
            name='right_descent',
            states=(
                'P5_FORWARD_AFTER_RESET_BODY',
                'P5_FORWARD_NO_ALIGN_AFTER_RESET_BODY',
            ),
            progress=PROGRESS_DISTANCE,
            expected_m=1.50, min_m=0.80, max_m=2.20,
            lateral_profile='flat',
            entry_evidence='post-jump alignment completed',
            exit_evidence='timed forward motion; final-zone protocol pending Phase E',
            enforcement=ENFORCEMENT_MONITOR,
            speed_cap_mps=0.25,
            reference='drawing 150 cm right inner straight',
        ),
        RouteSegment(
            name='final_zone',
            states=(
                'P5_JUMP_EXIT_SLOPE',
                'P5_RECOVERY_AFTER_JUMP_2',
                'P5_FINAL_LONG_JUMP',
            ),
            progress=PROGRESS_NONE,
            expected_m=0.50, min_m=0.0, max_m=1.50,
            lateral_profile='flat',
            entry_evidence='right descent completed',
            exit_evidence='action completions only; §4.4 protocol not implemented',
            enforcement=ENFORCEMENT_MONITOR,
            reference='drawing final 50 cm jump-positioning zone',
        ),
    )


def normalize_angle_rad(angle):
    """Wrap an angle to (-pi, pi]."""
    wrapped = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def angle_delta_rad(previous, current):
    """Shortest signed rotation from ``previous`` to ``current``."""
    return normalize_angle_rad(float(current) - float(previous))


class SegmentProgress(object):
    """Odometry accumulator for the segment the state machine is currently in.

    ``update`` only integrates a sample whose generation counter advanced, so a
    frozen stream can neither inflate nor stall progress (plan §1
    "Frame-gated everything").  A single step longer than ``max_step_m`` is
    rejected rather than integrated: at 50 Hz a real walking step is ~1 cm, so
    a large delta means an estimator reset, a jump-induced discontinuity, or a
    teleport, none of which is travelled route distance.
    """

    def __init__(self, max_step_m=0.20):
        """Create an empty accumulator with the given single-step limit."""
        self.max_step_m = float(max_step_m)
        self.segment_name = ''
        self.reset('')

    def reset(self, segment_name=''):
        """Start accumulating a new segment from the next sample."""
        self.segment_name = str(segment_name)
        self.distance_m = 0.0
        self.along_track_m = 0.0
        self.cross_track_m = 0.0
        self.reference_yaw = None
        self.yaw_delta_rad = 0.0
        self.samples = 0
        self.rejected_steps = 0
        self.rejected_distance_m = 0.0
        self.last_seq = None
        self.last_xy = None
        self.last_yaw = None
        self.origin_xy = None
        self.origin_yaw = None

    def update(self, seq, x, y, yaw):
        """Integrate one odometry sample; returns True if it was integrated."""
        seq = int(seq)
        if self.last_seq is not None and seq == self.last_seq:
            return False
        first = self.last_seq is None
        self.last_seq = seq
        if first:
            self.origin_xy = (float(x), float(y))
            self.origin_yaw = float(yaw)
            self.last_xy = (float(x), float(y))
            self.last_yaw = float(yaw)
            self.samples = 1
            return True

        step = math.hypot(float(x) - self.last_xy[0], float(y) - self.last_xy[1])
        if step > self.max_step_m:
            self.rejected_steps += 1
            self.rejected_distance_m += step
        else:
            self.distance_m += step
        # Yaw is tracked even across a rejected translation step: a turn jump
        # moves the body little but rotates it a lot, and the rotation is
        # exactly what corner verification needs.
        self.yaw_delta_rad += angle_delta_rad(self.last_yaw, yaw)
        self.last_xy = (float(x), float(y))
        self.last_yaw = float(yaw)
        self.samples += 1
        self._update_along_track()
        return True

    def set_reference_yaw(self, reference_yaw):
        """Set the direction along which progress is measured for this segment.

        Callers that know the route's declared heading should supply it; without
        it the segment's own entry yaw is used, which inherits whatever error
        the preceding corner left behind.
        """
        self.reference_yaw = None if reference_yaw is None else float(reference_yaw)
        self._update_along_track()

    def _update_along_track(self):
        """Project displacement since segment entry onto the reference heading.

        The perpendicular component (``cross_track_m``, left of the line
        positive) falls out of the same projection and is the only run-time
        measure of "have we left the rail" that survives when depth is blind —
        which, on the ring rails, is most of the time.

        Path length and along-track progress are not the same number, and the
        difference is exactly the lateral work the centring loop does.  Measured
        2026-08-04 in ``race.world``: on one straight the body covered 3.13 m of
        path but only 2.47 m along the rail — a 27% excess, against 8% on runs
        where the loop had little to correct.  A segment gated on path length
        therefore ends *earlier* the harder the lateral loop works, which fired
        a corner 0.6 m short of the next rail and turned the robot into the
        ring's void.  Progress along a route has to be measured along the route.
        """
        if self.origin_xy is None or self.last_xy is None:
            self.along_track_m = 0.0
            self.cross_track_m = 0.0
            return
        heading = self.reference_yaw
        if heading is None:
            heading = self.origin_yaw
        if heading is None:
            self.along_track_m = 0.0
            self.cross_track_m = 0.0
            return
        dx = self.last_xy[0] - self.origin_xy[0]
        dy = self.last_xy[1] - self.origin_xy[1]
        self.along_track_m = dx * math.cos(heading) + dy * math.sin(heading)
        self.cross_track_m = -dx * math.sin(heading) + dy * math.cos(heading)

    @property
    def yaw_delta_deg(self):
        """Unwrapped rotation since segment entry, in degrees."""
        return math.degrees(self.yaw_delta_rad)

    def snapshot(self):
        """Return a JSON-ready summary of this segment's progress."""
        return {
            'segment': self.segment_name,
            'distance_m': float(self.distance_m),
            'along_track_m': float(self.along_track_m),
            'cross_track_m': float(self.cross_track_m),
            'reference_yaw': self.reference_yaw,
            'yaw_delta_deg': float(self.yaw_delta_deg),
            'samples': int(self.samples),
            'rejected_steps': int(self.rejected_steps),
            'rejected_distance_m': float(self.rejected_distance_m),
        }


class GateDecision(object):
    """Result of one two-source gate evaluation."""

    __slots__ = ('status', 'allow_exit', 'reason', 'progress_m', 'window')

    def __init__(self, status, allow_exit, reason, progress_m, window):
        """Record one gate outcome and the evidence behind it."""
        self.status = status
        self.allow_exit = bool(allow_exit)
        self.reason = reason
        self.progress_m = float(progress_m)
        self.window = tuple(window)

    def to_dict(self):
        """Return the decision as a plain dict (evidence logging)."""
        return {
            'status': self.status,
            'allow_exit': self.allow_exit,
            'reason': self.reason,
            'progress_m': self.progress_m,
            'window_m': list(self.window),
        }

    def __repr__(self):  # pragma: no cover - debug aid only
        """Return a short identifying representation."""
        return 'GateDecision(status={!r}, allow_exit={!r})'.format(
            self.status, self.allow_exit)


def evaluate_gate(segment, progress_m, exit_confirmed, odometry_valid):
    """Combine an odometry window with a perception trigger.

    ``allow_exit`` is True only when both sources agree.  A monitored segment
    keeps its own transition authority: the returned status still describes the
    window so the caller can log it, but ``allow_exit`` simply follows
    ``exit_confirmed``.
    """
    window = (float(segment.min_m), float(segment.max_m))
    if not segment.enforced:
        status = GATE_IN_WINDOW
        if odometry_valid:
            if progress_m < segment.min_m:
                status = GATE_BELOW_MIN
            elif segment.max_m > 0.0 and progress_m > segment.max_m:
                status = GATE_OVERRUN
        else:
            status = GATE_UNAVAILABLE
        return GateDecision(
            status, bool(exit_confirmed), 'monitor_only', progress_m, window)

    if not odometry_valid:
        return GateDecision(
            GATE_UNAVAILABLE, False, 'odometry_unavailable', progress_m, window)

    if progress_m < segment.min_m:
        return GateDecision(
            GATE_BELOW_MIN, False, 'below_min_window', progress_m, window)

    if segment.max_m > 0.0 and progress_m > segment.max_m:
        return GateDecision(
            GATE_OVERRUN, False, 'above_max_window', progress_m, window)

    if exit_confirmed:
        return GateDecision(
            GATE_IN_WINDOW, True, 'two_sources_agree', progress_m, window)

    return GateDecision(
        GATE_IN_WINDOW, False, 'awaiting_exit_evidence', progress_m, window)


def odometry_exit_reached(segment, progress_m, odometry_valid):
    """Decide whether an odometry-ended segment has run its declared length.

    Returns a :class:`GateDecision` so the caller logs the same shape as the
    two-source gate.  Without valid odometry this fails closed: a segment that
    has *no* perception exit and no odometry has nothing left to end it, so it
    must not advance.
    """
    window = (float(segment.min_m), float(segment.max_m))
    if not odometry_valid:
        return GateDecision(
            GATE_UNAVAILABLE, False, 'odometry_unavailable', progress_m, window)
    target = float(segment.expected_m)
    if target <= 0.0:
        return GateDecision(
            GATE_UNAVAILABLE, False, 'no_expected_length', progress_m, window)
    if progress_m < target:
        return GateDecision(
            GATE_BELOW_MIN, False, 'below_expected_length', progress_m, window)
    if segment.max_m > 0.0 and progress_m > segment.max_m:
        return GateDecision(
            GATE_OVERRUN, False, 'above_max_window', progress_m, window)
    return GateDecision(
        GATE_IN_WINDOW, True, 'odometry_length_reached', progress_m, window)


def verify_yaw(segment, yaw_delta_deg):
    """Check a corner's measured rotation against its declared one.

    Returns ``(ok, error_deg)``.  ``error_deg`` is signed: positive means the
    robot still has to rotate positively to reach the declared heading, so it
    doubles as the re-alignment command sign.
    """
    error_deg = math.degrees(normalize_angle_rad(
        math.radians(float(segment.expected_yaw_deg) - float(yaw_delta_deg))))
    tol = abs(float(segment.yaw_tol_deg))
    if tol <= 0.0:
        return True, error_deg
    return abs(error_deg) <= tol, error_deg


def clamp_speed(cap_mps, vx, vy):
    """Scale a planar velocity command down to ``cap_mps``.

    Returns ``(vx, vy, capped)``.  Scaling (not per-axis clipping) keeps the
    commanded direction, and a zero command stays zero.
    """
    vx = float(vx)
    vy = float(vy)
    cap = float(cap_mps)
    if cap <= 0.0:
        return vx, vy, False
    speed = math.hypot(vx, vy)
    if speed <= cap or speed <= 0.0:
        return vx, vy, False
    scale = cap / speed
    return vx * scale, vy * scale, True


class EntryDepthGate(object):
    """Course-entry integrity: the climb segment must produce depth evidence.

    Both false completions observed in the 2026-08-04 reliability batches
    (plan item 25) fell off the bridge in their first metres, self-recovered
    on the floor, and pattern-walked the whole route to ``P5_DONE`` — every
    distance window and corner yaw check is satisfiable anywhere.  The
    discriminator in the data is the climb: every genuine run produced 8-10
    valid depth-observer frames on ``up_slope`` (the bridge has proper
    two-sided edges), while both floor runs produced exactly zero *during the
    segment*.  Frames seen before the segment do not count: the floor runs
    still got valid fixes on the entrance step from the start zone.

    A rolling "N consecutive blind segments" guard was considered and refuted
    by the same data: with the two-sided extractor the ring rails are blind on
    genuine runs too (one run crossed straights 1-3 with zero valid frames and
    stayed on the deck), so any streak short enough to catch the floor walk
    also faults real runs.

    Trips at most once; the caller decides what a trip means.
    """

    def __init__(self, segment_name='', min_valid_frames=0):
        self.segment_name = str(segment_name)
        self.min_valid_frames = int(min_valid_frames)
        self.reset()

    def reset(self):
        self.valid_frames = 0
        self.checked = False
        self.tripped = False

    @property
    def enabled(self):
        return bool(self.segment_name) and self.min_valid_frames > 0

    def record_frame(self, valid, active_segment):
        """Count a depth-observer result while its segment is active."""
        if not self.enabled or self.checked:
            return
        if valid and str(active_segment) == self.segment_name:
            self.valid_frames += 1

    def segment_closed(self, segment_name):
        """Close ``segment_name``; True when the gate trips.

        Only the configured entry segment is examined, exactly once — a
        re-entered segment (realign detours re-enter the same segment name)
        must not re-arm the check.
        """
        if not self.enabled or self.checked:
            return False
        if str(segment_name) != self.segment_name:
            return False
        self.checked = True
        self.tripped = self.valid_frames < self.min_valid_frames
        return self.tripped

    def snapshot(self):
        return {
            'segment': self.segment_name,
            'min_valid_frames': int(self.min_valid_frames),
            'valid_frames': int(self.valid_frames),
            'checked': bool(self.checked),
            'tripped': bool(self.tripped),
        }


class CrossTrackGate(object):
    """Run-time departure check: how far sideways may a segment wander.

    ``EntryDepthGate`` only guards the bridge climb, so a body that walks off a
    *ring rail* still satisfies every distance window and corner yaw check and
    reaches ``P5_DONE`` on the floor.  Measured 2026-08-05 (plan item 32):
    three of six "completions" finished 0.3-0.6 m past straight_3's outer edge
    at floor height.  The audit caught them afterwards; nothing caught them
    while they were still walking.

    The measure is the odometry cross-track from the segment's entry line, so
    it is available on every rail regardless of what depth can see.  Its blind
    spot is real and must be stated: the line is anchored at segment entry, so
    a body that enters a segment already off centre reads zero here.  This
    bounds *drift within a segment*, which is what straight_3 does; it is not a
    course-referenced position check.

    ``consecutive_samples`` exists because the estimator is what it is: a
    single bad fix must not stop the stage.  At 50 Hz, 25 samples is 0.5 s.
    """

    #: Segments declaring this profile run on the floor, not on a raised deck:
    #: the turn jumps, the post-jump alignment, the descent and the final jump
    #: zone.  All of them translate the body sideways on purpose, and none of
    #: them has an edge to fall off, so a cross-track limit there measures the
    #: manoeuvre rather than a departure.  Measured 2026-08-05: with the check
    #: applied to every segment, three consecutive runs faulted inside
    #: ``P5_FINAL_LONG_JUMP`` — the last state before ``P5_DONE`` — at 0.36 m
    #: of entirely intentional sideways travel.
    FLAT_PROFILE = 'flat'

    def __init__(self, limit_m=0.0, consecutive_samples=25):
        self.limit_m = float(limit_m)
        self.consecutive_samples = int(consecutive_samples)
        self.reset()

    def reset(self):
        """Start a fresh segment; a new entry line means a new budget."""
        self.streak = 0
        self.tripped = False
        self.worst_m = 0.0

    @property
    def enabled(self):
        return self.limit_m > 0.0

    @classmethod
    def applies_to(cls, segment):
        """Report whether a segment has an edge worth measuring distance to."""
        return (segment is not None
                and str(getattr(segment, 'lateral_profile', cls.FLAT_PROFILE))
                != cls.FLAT_PROFILE)

    def record(self, cross_track_m):
        """Fold in one odometry sample; True the tick the gate trips."""
        if not self.enabled or self.tripped:
            return False
        try:
            value = float(cross_track_m)
        except (TypeError, ValueError):
            return False
        if value != value:                                      # NaN
            return False
        if abs(value) > abs(self.worst_m):
            self.worst_m = value
        if abs(value) <= self.limit_m:
            self.streak = 0
            return False
        self.streak += 1
        if self.streak < max(1, self.consecutive_samples):
            return False
        self.tripped = True
        return True

    def snapshot(self):
        return {
            'limit_m': float(self.limit_m),
            'consecutive_samples': int(self.consecutive_samples),
            'streak': int(self.streak),
            'worst_cross_track_m': float(self.worst_m),
            'tripped': bool(self.tripped),
        }


class ToppleGate(object):
    """Run-time check: has the body been on its side since the stage started.

    ``CrossTrackGate`` measures drift *within* a segment from the segment's own
    entry line, so it is blind to a body that arrives already off the course —
    which is exactly what a failed corner jump produces.  Measured 2026-08-16
    (``race_physical``, 12 runs): one run was thrown off the rail by the
    corner-3 jump, rolled through 3.14 rad, stood itself back up on the floor
    1.16 m outside straight_3, and then satisfied every remaining distance
    window and corner check to reach ``P5_DONE``.  Nothing in the stage noticed;
    only the offline ground-truth audit did.  A false completion publishes
    ``stage_complete`` and is strictly worse than a fault.

    Attitude is the right measure for this and position is not.  Plan item 37
    established that a tumble destroys the leg odometry's position state while
    leaving attitude intact — two measured pick-ups stood the robot up 0.29 m
    off the rail while odometry reported 0.024 m of displacement.  So the fact
    worth latching is not *where* the body is but *that it went over*, which
    attitude reports honestly and position does not.

    The latch is deliberately permanent for the run.  A body that has been on
    its side no longer has a trustworthy idea of where it is, so "it stood back
    up" is not evidence that it stood back up *on the course*.
    """

    def __init__(self, limit_rad=0.0, consecutive_samples=25):
        self.limit_rad = float(limit_rad)
        self.consecutive_samples = int(consecutive_samples)
        self.reset()

    def reset(self):
        """Clear the latch.  Call on stage entry, never on segment entry."""
        self.streak = 0
        self.tripped = False
        self.worst_rad = 0.0

    @property
    def enabled(self):
        return self.limit_rad > 0.0

    def record(self, roll_rad, pitch_rad):
        """Fold in one attitude sample; True the tick the gate trips."""
        if not self.enabled or self.tripped:
            return False
        try:
            roll = abs(float(roll_rad))
            pitch = abs(float(pitch_rad))
        except (TypeError, ValueError):
            return False
        if roll != roll or pitch != pitch:                      # NaN
            return False
        worst = max(roll, pitch)
        if worst > self.worst_rad:
            self.worst_rad = worst
        if worst <= self.limit_rad:
            self.streak = 0
            return False
        self.streak += 1
        if self.streak < max(1, self.consecutive_samples):
            return False
        self.tripped = True
        return True

    def snapshot(self):
        return {
            'limit_rad': float(self.limit_rad),
            'consecutive_samples': int(self.consecutive_samples),
            'streak': int(self.streak),
            'worst_attitude_rad': float(self.worst_rad),
            'tripped': bool(self.tripped),
        }


class StallGate(object):
    """Run-time check: is the robot being commanded to walk and not walking.

    Plan item 11 has been open since the route model was written, and it costs
    whole runs.  Measured 2026-08-16 (``race_physical``): the robot reaches the
    bridge ramp, fails to mount the entrance step, settles into a splayed
    stance (ground-truth body z drops from 0.30 to 0.13, below standing height
    on flat floor), and then sits there with ``vx`` commanded at 0.45 m/s and
    odometry frozen at 0.20 m until the 45 s state timeout ends the run.  Three
    such runs in eleven.  The body rotates while it is stuck, so the lateral
    loop then commands its full yaw authority against a robot that cannot move
    — which reads in the log exactly like a steering fault and is not one.

    What makes this detectable without a second sensor is the *pair*: a
    commanded speed well above zero together with progress that does not
    advance.  Either alone is normal — a stopped robot is commanded zero, and a
    turning-in-place robot legitimately makes no forward progress with vx at
    zero, which is why the speed term is required rather than a bare timer.

    ``min_progress_m`` must exceed the estimator's own jitter; odometry drifts
    by a few millimetres while standing, so a strict "no change at all" test
    never fires.

    ``latching`` picks which of the two questions this instance answers.  The
    default, True, is the fault detector it was written as: it fires once, on
    the edge, and stays tripped so the caller can act exactly one time.  False
    makes it a *level* — True on every tick the stall persists, back to False
    the moment progress resumes — which is what a consumer that has to keep
    suppressing something for the duration of a stall needs.

    That distinction cost a batch.  Reusing the latching instance to gate the
    lateral loop's turn term suppressed it for a single tick and then, being
    tripped, reported False for the rest of the run: the measured symptom was
    the trim continuing to wind (+0.094 -> +0.111) with odometry progress
    pinned at 0.00 m, which is precisely the state the suppression exists to
    prevent.  An edge-triggered signal cannot hold anything down.
    """

    def __init__(self, min_speed=0.0, timeout_s=0.0, min_progress_m=0.05,
                 latching=True):
        self.min_speed = float(min_speed)
        self.timeout_s = float(timeout_s)
        self.min_progress_m = float(min_progress_m)
        self.latching = bool(latching)
        self.reset()

    def reset(self):
        """Start a fresh segment: a new segment is a new progress baseline."""
        self.since_s = None
        self.reference_m = None
        self.tripped = False
        self.worst_stall_s = 0.0

    @property
    def enabled(self):
        return self.min_speed > 0.0 and self.timeout_s > 0.0

    def record(self, commanded_speed, progress_m, now_s):
        """Fold in one tick; True the tick the gate trips."""
        if not self.enabled or (self.latching and self.tripped):
            return False
        try:
            speed = abs(float(commanded_speed))
            progress = float(progress_m)
            now = float(now_s)
        except (TypeError, ValueError):
            return False
        if speed != speed or progress != progress or now != now:    # NaN
            return False
        if speed < self.min_speed:
            # Not being asked to move: whatever this is, it is not a stall.
            self.since_s = None
            self.reference_m = None
            return False
        if (self.reference_m is None
                or abs(progress - self.reference_m) >= self.min_progress_m):
            self.since_s = now
            self.reference_m = progress
            return False
        stalled = now - self.since_s
        if stalled > self.worst_stall_s:
            self.worst_stall_s = stalled
        if stalled < self.timeout_s:
            return False
        self.tripped = True
        return True

    def snapshot(self):
        return {
            'min_speed': float(self.min_speed),
            'timeout_s': float(self.timeout_s),
            'min_progress_m': float(self.min_progress_m),
            'worst_stall_s': float(self.worst_stall_s),
            'tripped': bool(self.tripped),
        }



class DropoffTrigger(object):
    """Fire once when the deck's far edge comes within ``trigger_m`` ahead.

    Every corner on this course is a fixed jump action, so where the robot
    takes off decides what rotation the jump delivers, and the take-off is
    currently set by accumulated odometry: ~3 m of integrated leg odometry plus
    a timed open-loop run.  Measured spread at corner 4 is ~0.33 m against a
    usable window of ~0.13 m, with the distribution already centred — so no
    segment length fixes it, because re-centring cannot narrow anything.  A
    course-referenced trigger can: "the deck ends 1.2 m ahead" does not
    accumulate error the way "I have walked 2.93 m" does.

    Confirmation is required rather than optional.  The observer is noisy on
    the ring rails — a measured straight_1 sequence reads 1.699, 1.983, 1.511,
    1.509, 1.217, 0.990 m, which is a clean downward trend with one reading
    that goes the wrong way — so a single frame under the threshold is not
    evidence of arrival.

    Latching, and deliberately so: a segment ends once.  Unlike the lateral
    suppression gates (see ``StallGate.latching``) there is no "situation
    improved" state to return to, because the caller has already left.
    """

    def __init__(self, trigger_m=0.0, samples=2, max_plausible_m=6.0):
        self.trigger_m = float(trigger_m)
        self.samples = max(1, int(samples))
        self.max_plausible_m = float(max_plausible_m)
        self.reset()

    def reset(self):
        """Start a fresh segment."""
        self.streak = 0
        self.tripped = False
        self.last_distance_m = None

    @property
    def enabled(self):
        return self.trigger_m > 0.0

    def record(self, distance_m):
        """Fold in one observation; True on the tick the trigger fires.

        ``None`` — no drop-off seen this frame — is *not* evidence that the
        deck continues, so it leaves the streak alone rather than breaking it.
        The observer reports a drop-off on well under half of frames even where
        it works, and treating each silent frame as a contradiction would mean
        the streak essentially never completes.
        """
        if not self.enabled or self.tripped:
            return False
        if distance_m is None:
            return False
        try:
            distance = float(distance_m)
        except (TypeError, ValueError):
            return False
        if distance != distance:                                # NaN
            return False
        if distance < 0.0 or distance > self.max_plausible_m:
            # Past the body or implausibly far: a bad fit, not a deck end.
            return False
        self.last_distance_m = distance
        if distance > self.trigger_m:
            self.streak = 0
            return False
        self.streak += 1
        if self.streak < self.samples:
            return False
        self.tripped = True
        return True

    def snapshot(self):
        return {
            'trigger_m': float(self.trigger_m),
            'samples': int(self.samples),
            'streak': int(self.streak),
            'tripped': bool(self.tripped),
            'last_distance_m': self.last_distance_m,
        }


class RouteModel(object):
    """The segment table plus lookup, override and gating helpers."""

    def __init__(self, segments=None):
        """Build the model, rejecting a state claimed by two segments."""
        self.segments = tuple(segments if segments is not None else default_segments())
        self._by_state = {}
        for segment in self.segments:
            for state in segment.states:
                if state in self._by_state:
                    raise ValueError(
                        'state {!r} appears in more than one route segment'.format(state))
                self._by_state[state] = segment

    @property
    def names(self):
        """Segment names in route order."""
        return tuple(segment.name for segment in self.segments)

    def segment_for_state(self, state):
        """Return the segment owning ``state``, or None for off-route states."""
        return self._by_state.get(str(state))

    def segment_by_name(self, name):
        """Return the segment with this name, or None."""
        for segment in self.segments:
            if segment.name == name:
                return segment
        return None

    def with_overrides(self, overrides):
        """Return a new model with per-segment field overrides applied.

        ``overrides`` maps segment name -> {field: value}.  Unknown segment
        names raise, so a typo in a parameter file cannot silently leave the
        drawing-derived default in place.
        """
        new_segments = []
        remaining = dict(overrides or {})
        for segment in self.segments:
            fields = remaining.pop(segment.name, None)
            new_segments.append(segment.replace(**fields) if fields else segment)
        if remaining:
            raise ValueError(
                'unknown route segment name(s): {}'.format(sorted(remaining)))
        return RouteModel(tuple(new_segments))

    def validate(self):
        """Return a list of human-readable table problems (empty when sane)."""
        problems = []
        for segment in self.segments:
            if segment.progress == PROGRESS_DISTANCE:
                if segment.max_m > 0.0 and segment.min_m > segment.max_m:
                    problems.append(
                        '{}: min_m {:.3f} > max_m {:.3f}'.format(
                            segment.name, segment.min_m, segment.max_m))
                if segment.enforced and segment.max_m <= 0.0:
                    problems.append(
                        '{}: enforced distance segment needs a positive max_m'.format(
                            segment.name))
            if segment.odometry_exit:
                if segment.expected_m <= 0.0:
                    problems.append(
                        '{}: odometry exit needs a positive expected_m'.format(
                            segment.name))
                elif not (segment.min_m <= segment.expected_m <= segment.max_m):
                    problems.append(
                        '{}: odometry exit length {:.3f} outside window '
                        '[{:.3f}, {:.3f}]'.format(
                            segment.name, segment.expected_m,
                            segment.min_m, segment.max_m))
                if segment.fallback_tier < TIER_DEAD_RECKONING:
                    problems.append(
                        '{}: odometry exit is single-source and must declare '
                        'fallback_tier >= {}'.format(
                            segment.name, TIER_DEAD_RECKONING))
            if segment.progress == PROGRESS_YAW and segment.enforced:
                if segment.yaw_tol_deg <= 0.0:
                    problems.append(
                        '{}: enforced yaw segment needs a positive yaw_tol_deg'.format(
                            segment.name))
        return problems
