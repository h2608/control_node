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

        for key, value in overrides.items():
            if key not in self.__slots__:
                raise AttributeError(f'unknown config field: {key}')
            setattr(self, key, value)


class DeckLateralCommand:
    """One loop output: the correction plus why it is what it is."""

    __slots__ = ('state', 'vy', 'wz', 'vx_scale', 'reason',
                 'lateral_offset', 'heading_error', 'age_s')

    def __init__(self, state, vy=0.0, wz=0.0, vx_scale=1.0, reason='',
                 lateral_offset=None, heading_error=None, age_s=None):
        """Store the correction and its provenance."""
        self.state = state
        self.vy = float(vy)
        self.wz = float(wz)
        self.vx_scale = float(vx_scale)
        self.reason = reason
        self.lateral_offset = lateral_offset
        self.heading_error = heading_error
        self.age_s = age_s

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

    def update(self, observation, now_s):
        """Fold one observation in and return the correction to apply now."""
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
                    CONTROL_HOLDING, vy=self._last_vy, wz=self._last_wz,
                    vx_scale=cfg.centre_first_vx_scale if self._centring else 1.0,
                    reason='observation_stale', age_s=age,
                    lateral_offset=self._last_offset,
                    heading_error=self._last_heading)
            self._last_vy = 0.0
            self._last_wz = 0.0
            self._centring = False
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
            return DeckLateralCommand(
                CONTROL_ENGAGING, reason='awaiting_confirmation',
                lateral_offset=offset, heading_error=heading, age_s=0.0)

        lateral_term = 0.0
        if abs(offset) > cfg.deadband_m:
            lateral_term = cfg.k_vy * offset
        heading_term = 0.0
        if abs(heading) > cfg.heading_deadband_rad:
            heading_term = cfg.k_wz * heading

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
            lateral_offset=offset, heading_error=heading, age_s=0.0)

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
