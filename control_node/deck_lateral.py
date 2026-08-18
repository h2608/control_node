#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Depth-derived lateral/heading hold for the Stage 5 bridge deck.

``bridge_perception`` estimates where the deck centreline is relative to the
body; this module turns that estimate into bounded ``vy``/``wz`` corrections.
It is deliberately pure (no ROS, no clocks of its own) so the whole policy —
engagement, staleness, saturation, and what happens when the observer goes
blind — is unit-testable without a simulator.

Why this exists (2026-08-04 sim measurements, ``race.world``): the deck is
0.504 m wide, and across one climb the robot drifted monotonically from
+0.070 m to +0.225 m off the centreline while the observer tracked that drift
on 12 of 13 frames.  Nothing consumed the measurement, so the robot walked off
the side.  Odometry cannot close this loop — it measures progress along the
route, never offset from the centreline.

Sign convention follows ``bridge_perception``: ``lateral_offset`` > 0 means the
deck centreline lies to the body's left, so the correction is +vy (left).
``heading_error`` > 0 means the deck axis points front-left, so +wz (left).

The loop is proportional by default and PI when ``k_i_vy`` is opted into.  The
integral exists for one measured reason: on a banked surface the disturbance is
a constant, and a P loop answers a constant disturbance with a constant error.
See ``DeckLateralConfig.k_i_vy``.

Engagement is deliberately conservative.  A single valid frame is never enough
to steer on; the observer must agree with itself for ``min_consecutive_valid``
frames.  When it goes blind the last correction is held only briefly, then
released, and the caller is asked to slow down rather than to keep steering on
a stale estimate.
"""

CONTROL_IDLE = 'idle'
CONTROL_ENGAGING = 'engaging'
CONTROL_ACTIVE = 'active'
CONTROL_HOLDING = 'holding'
CONTROL_BLIND = 'blind'


def _clamp(value, limit):
    """Clamp ``value`` into ``[-limit, +limit]`` (``limit`` must be >= 0)."""
    if limit <= 0.0:
        return 0.0
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


class DeckLateralConfig:
    """Gains and guards for the deck-centring loop."""

    __slots__ = (
        'k_vy', 'k_wz', 'deadband_m', 'heading_deadband_rad',
        'max_vy', 'max_wz', 'max_age_s', 'min_consecutive_valid',
        'max_plausible_offset_m', 'blind_vx_scale',
        'centre_first_offset_m', 'centre_first_release_m',
        'centre_first_vx_scale', 'centre_first_max_s',
        'k_i_vy', 'max_i_vy', 'max_integration_dt_s', 'stall_vy_scale',
        'wz_futile_samples',
    )

    def __init__(self, **overrides):
        """Build the sim-tuned defaults; every field may be overridden."""
        # 0.011 m observer sigma -> a ~2 sigma deadband keeps the loop quiet
        # when it is already centred.
        self.deadband_m = 0.02
        self.heading_deadband_rad = 0.02
        # 0.20 m offset -> 0.08 m/s of side-step against a 0.45 m/s walk.
        self.k_vy = 0.40
        self.k_wz = 0.80
        self.max_vy = 0.12
        self.max_wz = 0.25
        # The observer runs at 5 Hz; one missed frame is normal, a second is not.
        self.max_age_s = 1.0
        self.min_consecutive_valid = 2
        # Half the deck is 0.252 m; anything past this is a bad fit, not a pose.
        self.max_plausible_offset_m = 0.40
        # What the caller should do with vx while the observer is blind.
        self.blind_vx_scale = 0.5
        # "Centre first": above this offset, stop making forward progress and
        # spend the time getting back on the line instead.  A 0.40 gain against
        # a 0.30 m/s walk corrects ~0.13 m of offset per metre travelled, which
        # is not enough when the surface runs out in 0.14 m.  Measured in
        # ``race.world``: corner 1 leaves the body 0.13 m off the ring rail's
        # centre with the rail's inner void 0.14 m away — less than half a
        # stance width — and the first forward step drops a foot into it.
        # 0.0 disables the behaviour, which is the default: it must be opted
        # into per profile, never inherited silently by the physical robot.
        self.centre_first_offset_m = 0.0
        self.centre_first_release_m = 0.03
        self.centre_first_vx_scale = 0.0
        # Centre first, but never forever.  If side-stepping is not closing the
        # offset, something is wrong with the estimate or the surface, and
        # holding vx at zero until the state times out converts a recoverable
        # deviation into a guaranteed stage failure.  Give up and walk (the
        # lateral term stays applied) rather than stand still.
        self.centre_first_max_s = 6.0
        # Integral term on the lateral offset.  A proportional loop cannot
        # cancel a *constant* lateral disturbance, and on a banked rail that is
        # exactly what the loop faces: gravity pulls the body down the camber
        # all segment long, and the segment's open-loop crab (``*_vy``) is a
        # hand-calibrated guess at how much to lean against it.  Measured in
        # ``race_physical.world`` 2026-08-16, straight_1: the crab overshoots by
        # ~0.010 m/s, which the P term settles at rather than removes, so the
        # body walks ~0.04 m per metre toward the rail's outer edge and topples
        # around 0.11-0.12 m off centre — five of six runs, always that sign.
        # The integrator makes the segment's crab self-calibrating: whatever
        # constant lateral velocity error the profile's open-loop value leaves
        # behind, it accumulates the offset until the correction cancels it.
        # 0.0 disables it, which is the default — no existing profile changes
        # behaviour unless it opts in.
        self.k_i_vy = 0.0
        # Authority bound.  The integral may never do more than the crab itself
        # plausibly got wrong; past that the estimate is at fault, not the trim.
        self.max_i_vy = 0.06
        # Longest gap that still counts as continuous integration.  A dropout
        # longer than this is a hole in the observation, not elapsed error.
        self.max_integration_dt_s = 0.5
        # 1.0 keeps the crab at full authority during a stall (the
        # behaviour before this was measured).
        self.stall_vy_scale = 1.0
        # Consecutive ticks of "wz is saturated and the heading error is still
        # growing" before the turn term is judged futile and dropped.  This is
        # the *direct* statement of the failure the freeze was built for, and
        # it needs no caller, no speed threshold and no progress measure: if
        # the actuator is on its bound and the error it is chasing keeps
        # growing in the same direction, the command is not being executed.
        #
        # Measured against the progress-based freeze it replaces: that gate
        # missed a slow creep entirely — the body advanced 0.14 -> 0.18 m,
        # just under its 0.05 m per 1.0 s threshold, so it stayed silent while
        # wz sat on +0.250 and the heading ran 0.356 -> 0.572 rad.  Same
        # physics, wrong signal.  0 disables it.
        self.wz_futile_samples = 0

        for key, value in overrides.items():
            if key not in self.__slots__:
                raise AttributeError(f'unknown config field: {key}')
            setattr(self, key, value)


class DeckLateralCommand:
    """One loop output: the correction plus why it is what it is."""

    __slots__ = ('state', 'vy', 'wz', 'vx_scale', 'reason',
                 'lateral_offset', 'heading_error', 'age_s', 'vy_integral')

    def __init__(self, state, vy=0.0, wz=0.0, vx_scale=1.0, reason='',
                 lateral_offset=None, heading_error=None, age_s=None,
                 vy_integral=0.0):
        """Store the correction and its provenance."""
        self.state = state
        self.vy = float(vy)
        self.wz = float(wz)
        self.vx_scale = float(vx_scale)
        self.reason = reason
        self.lateral_offset = lateral_offset
        self.heading_error = heading_error
        self.age_s = age_s
        self.vy_integral = float(vy_integral)

    @property
    def engaged(self):
        """Report whether this command is actually steering the robot."""
        return self.state in (CONTROL_ACTIVE, CONTROL_HOLDING)

    def to_dict(self):
        """Return a JSON-friendly record for the evidence log."""
        return {
            'state': self.state,
            'vy': self.vy,
            'wz': self.wz,
            'vx_scale': self.vx_scale,
            'reason': self.reason,
            'lateral_offset': self.lateral_offset,
            'heading_error': self.heading_error,
            'age_s': self.age_s,
            'vy_integral': self.vy_integral,
        }


class DeckLateralController:
    """Turn bridge observations into bounded lateral/heading corrections."""

    def __init__(self, config=None):
        """Start disengaged with no accepted observation."""
        self.config = config or DeckLateralConfig()
        self.reset()

    def reset(self):
        """Forget every accepted observation and disengage."""
        self._consecutive_valid = 0
        self._last_good_s = None
        self._last_vy = 0.0
        self._last_wz = 0.0
        self._last_offset = None
        self._last_heading = None
        self._centring = False
        self._centring_since_s = None
        self._centring_gave_up = False
        self._vy_integral = 0.0
        self._last_integration_s = None
        self._wz_futile_streak = 0
        self._wz_suppressed = False
        self._wz_suppressed_heading = None
        self._last_heading_for_futility = None

    def _accept(self, observation):
        """Return (offset, heading) when the observation is usable, else None."""
        if not isinstance(observation, dict) or not observation.get('valid'):
            return None
        offset = observation.get('lateral_offset')
        heading = observation.get('heading_error')
        if offset is None or heading is None:
            return None
        try:
            offset = float(offset)
            heading = float(heading)
        except (TypeError, ValueError):
            return None
        if offset != offset or heading != heading:      # NaN
            return None
        if abs(offset) > self.config.max_plausible_offset_m:
            return None
        return offset, heading

    def update(self, observation, now_s, forward_stalled=False):
        """Fold one observation in and return the correction to apply now.

        ``forward_stalled`` says the body is being asked to walk forward and is
        not advancing — the caller owns that judgement, since only it knows the
        segment's progress measure.  While it holds, the turn term is dropped
        and the trim is frozen, because **a body that is not advancing cannot
        turn**: the feet are loaded against whatever is blocking them, so wz
        buys no rotation and the error it is chasing does not shrink.

        Measured at the bridge's entrance step, where the loop had authority
        throughout: progress pinned at 0.14 m, wz saturated at its -0.250 bound,
        and the heading error grew straight through it, -0.134 -> -0.495 ->
        -0.518 -> -0.563 rad, with the trim winding +0.009 -> +0.085 alongside.
        The turn command is not merely wasted there — it twists a body that is
        already half up a step, and 4 of 16 runs in one batch toppled out of
        exactly this signature (1 of 16 in the batch before it).  Dropping wz
        leaves the vy centring that actually walks the robot onto the deck.
        """
        cfg = self.config
        accepted = self._accept(observation)

        if accepted is None:
            self._consecutive_valid = 0
            if self._last_good_s is None:
                return DeckLateralCommand(
                    CONTROL_BLIND, vx_scale=cfg.blind_vx_scale,
                    reason='no_observation_yet')
            age = float(now_s) - self._last_good_s
            if age <= cfg.max_age_s:
                # Briefly ride the last correction rather than snapping to zero,
                # and keep holding vx down if we were mid-centring: a dropout is
                # not evidence that the robot got back on the line.
                return DeckLateralCommand(
                    CONTROL_HOLDING, vy=self._last_vy,
                    wz=0.0 if forward_stalled else self._last_wz,
                    vx_scale=cfg.centre_first_vx_scale if self._centring else 1.0,
                    reason='observation_stale', age_s=age,
                    lateral_offset=self._last_offset,
                    heading_error=self._last_heading,
                    vy_integral=self._vy_integral)
            self._last_vy = 0.0
            self._last_wz = 0.0
            self._centring = False
            # Losing the observation for good means losing the reference the
            # trim was accumulated against, so it goes with it.  Holding a
            # blind integral is dead reckoning on a number nothing measured.
            self._vy_integral = 0.0
            self._last_integration_s = None
            return DeckLateralCommand(
                CONTROL_BLIND, vx_scale=cfg.blind_vx_scale,
                reason='observation_expired', age_s=age)

        offset, heading = accepted
        self._consecutive_valid += 1
        self._last_good_s = float(now_s)
        self._last_offset = offset
        self._last_heading = heading

        if self._consecutive_valid < cfg.min_consecutive_valid:
            # Seen once is not seen: do not steer on an unconfirmed estimate.
            self._last_integration_s = float(now_s)
            return DeckLateralCommand(
                CONTROL_ENGAGING, reason='awaiting_confirmation',
                lateral_offset=offset, heading_error=heading, age_s=0.0,
                vy_integral=self._vy_integral)

        lateral_term = 0.0
        if abs(offset) > cfg.deadband_m:
            lateral_term = cfg.k_vy * offset
        heading_term = 0.0
        if abs(heading) > cfg.heading_deadband_rad and not forward_stalled:
            heading_term = cfg.k_wz * heading
        if self._turn_is_futile(heading, heading_term):
            heading_term = 0.0

        if forward_stalled:
            # Freeze rather than zero: the offset is still real and still the
            # residual the trim was accumulated against.  Zeroing it would make
            # every stall cost the whole climb's worth of trim and hand the
            # camber back the standing error the integrator exists to remove.
            self._last_integration_s = float(now_s)
            lateral_term += self._vy_integral
            # The crab gets the same treatment as the turn term, for the same
            # reason: a body that is not advancing is not side-stepping either,
            # it is loading its feet against whatever is blocking them.
            # Measured with the wz freeze already in place, two runs of one
            # batch: pinned at 0.16 m and 0.03 m of 3.72 m for tens of seconds,
            # heading error stable (so the wz freeze was working), and vy held
            # at its -0.180 / -0.108 bound throughout with the centre-first
            # hold already timed out.  Both ended in P5_UP_SLOPE:timeout.
            lateral_term *= cfg.stall_vy_scale
        else:
            lateral_term += self._integrate(offset, lateral_term, float(now_s))
        self._last_vy = _clamp(lateral_term, cfg.max_vy)
        self._last_wz = _clamp(heading_term, cfg.max_wz)
        self._update_centring(abs(offset), float(now_s))
        if self._centring:
            reason = 'centring'
        elif self._centring_gave_up:
            reason = 'centring_timed_out'
        else:
            reason = 'ok'
        return DeckLateralCommand(
            CONTROL_ACTIVE, vy=self._last_vy, wz=self._last_wz,
            vx_scale=cfg.centre_first_vx_scale if self._centring else 1.0,
            reason=reason,
            lateral_offset=offset, heading_error=heading, age_s=0.0,
            vy_integral=self._vy_integral)

    def _turn_is_futile(self, heading, heading_term):
        """True while the turn command is demonstrably not being executed.

        The test is the actuator's own evidence: ``wz`` pinned on its bound
        with the heading error still growing in the same direction.  A turn
        that is working shrinks the error it is chasing; one that is not is
        either being resisted (feet loaded against an obstruction) or fighting
        something stronger than it, and in both cases the command buys no
        rotation while still twisting the body.

        Suppression latches deliberately.  Dropping ``wz`` immediately makes it
        un-saturated, so an edge test would clear on the very next tick and
        chatter; instead it holds until the error actually starts improving on
        the heading it had when suppression began.  Release is on the error, not
        on a timer, because the thing that matters is whether the situation
        changed, not how long it lasted.
        """
        cfg = self.config
        if cfg.wz_futile_samples <= 0:
            self._wz_futile_streak = 0
            self._wz_suppressed = False
            return False

        previous = self._last_heading_for_futility
        self._last_heading_for_futility = heading

        if self._wz_suppressed:
            improving = (self._wz_suppressed_heading is not None
                         and abs(heading) < abs(self._wz_suppressed_heading))
            if improving or abs(heading) <= cfg.heading_deadband_rad:
                self._wz_suppressed = False
                self._wz_futile_streak = 0
                self._wz_suppressed_heading = None
                return False
            return True

        saturated = abs(self._last_wz) >= cfg.max_wz - 1e-9
        growing = (previous is not None
                   and abs(heading) > abs(previous)
                   and (heading >= 0.0) == (previous >= 0.0))
        if saturated and growing and heading_term != 0.0:
            self._wz_futile_streak += 1
        else:
            self._wz_futile_streak = 0
        if self._wz_futile_streak >= cfg.wz_futile_samples:
            self._wz_suppressed = True
            self._wz_suppressed_heading = heading
            return True
        return False

    def _integrate(self, offset, proportional_term, now_s):
        """Accumulate the lateral trim and return its current contribution.

        Unlike the proportional term this integrates *inside* the deadband too:
        the deadband exists to stop the P term chattering around zero, whereas
        the whole point of the trim is to grind out the residual the P term
        settles at.  Anti-windup is a plain conditional — stop accumulating
        once the summed correction is already pinned against ``max_vy`` and the
        error would only push it further in — because a leaky or clamped-after
        integrator both keep winding while the actuator cannot respond.
        """
        cfg = self.config
        if cfg.k_i_vy <= 0.0 or cfg.max_i_vy <= 0.0:
            self._vy_integral = 0.0
            self._last_integration_s = now_s
            return 0.0
        previous = self._last_integration_s
        self._last_integration_s = now_s
        if previous is None:
            return self._vy_integral
        dt = now_s - previous
        if dt <= 0.0 or dt > cfg.max_integration_dt_s:
            return self._vy_integral
        total = proportional_term + self._vy_integral
        saturated = abs(total) >= cfg.max_vy and (total > 0.0) == (offset > 0.0)
        if not saturated:
            self._vy_integral = _clamp(
                self._vy_integral + cfg.k_i_vy * offset * dt, cfg.max_i_vy)
        return self._vy_integral

    def _update_centring(self, magnitude, now_s):
        """Latch/release the centre-first hold with hysteresis and a time bound.

        Two thresholds, not one: a single threshold makes the robot stutter
        between walking and side-stepping the moment it settles near it.
        """
        cfg = self.config
        if cfg.centre_first_offset_m <= 0.0:
            self._centring = False
            self._centring_since_s = None
            return
        if magnitude <= cfg.centre_first_release_m:
            # Back on the line: clear the latch and the give-up memory, so a
            # later deviation gets a full budget of its own.
            self._centring = False
            self._centring_since_s = None
            self._centring_gave_up = False
            return
        if magnitude < cfg.centre_first_offset_m:
            return                                   # hysteresis band: hold
        if self._centring_gave_up:
            return
        if not self._centring:
            self._centring = True
            self._centring_since_s = now_s
            return
        if (cfg.centre_first_max_s > 0.0 and self._centring_since_s is not None
                and now_s - self._centring_since_s >= cfg.centre_first_max_s):
            self._centring = False
            self._centring_since_s = None
            self._centring_gave_up = True
