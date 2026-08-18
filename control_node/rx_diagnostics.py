#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure (no ROS) receive-rate bookkeeping for the camera-dropout investigation.

The physical robot loses raw RGB delivery to the C++ compression bridge shortly
after the competition control stack starts, while the publisher itself keeps
producing messages.  To turn that into a bisectable observation every stage node
needs two things:

1. per-stream receive counters with a monotonic clock, so a frozen counter is
   distinguishable from a slow one and from a stopped node clock;
2. a timestamped lifecycle event log, so the moment a stream freezes can be
   attributed to a specific startup transition (node construction, activation,
   motion-backend creation, Servo START, first Servo DATA, ...).

Everything here is deliberately free of rclpy so it can be unit tested on a
machine without ROS.  ``now_s`` is always injectable; callers on the robot pass
``time.monotonic()``.

Python 3.6 compatible: the physical robot runs 3.6, so no dataclasses and no
``from __future__ import annotations``.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple


def _now(now_s: Optional[float]) -> float:
    return time.monotonic() if now_s is None else float(now_s)


class StreamRxStats(object):
    """Monotonic receive statistics for a single message stream."""

    def __init__(self, name: str, stall_warn_s: float = 2.0,
                 rate_window_s: float = 5.0):
        self.name = str(name)
        self.stall_warn_s = max(0.0, float(stall_warn_s))
        self.rate_window_s = max(0.5, float(rate_window_s))

        self.count = 0
        self.first_rx_s = None      # type: Optional[float]
        self.last_rx_s = None       # type: Optional[float]
        self.max_gap_s = 0.0
        self.stalled = False
        self.stall_started_s = None  # type: Optional[float]
        self.stall_events = 0
        # Sliding window of receive timestamps used for the short-term rate.
        self._window = []           # type: List[float]

    # ------------------------------------------------------------------
    def record(self, now_s: Optional[float] = None) -> None:
        """Register one received message."""
        now = _now(now_s)
        if self.last_rx_s is not None:
            gap = now - self.last_rx_s
            if gap > self.max_gap_s:
                self.max_gap_s = gap
        else:
            self.first_rx_s = now
        self.count += 1
        self.last_rx_s = now
        self._window.append(now)
        self._trim_window(now)

    def _trim_window(self, now: float) -> None:
        cutoff = now - self.rate_window_s
        window = self._window
        while window and window[0] < cutoff:
            window.pop(0)

    # ------------------------------------------------------------------
    def age_s(self, now_s: Optional[float] = None) -> Optional[float]:
        if self.last_rx_s is None:
            return None
        return max(0.0, _now(now_s) - self.last_rx_s)

    def rate_hz(self, now_s: Optional[float] = None) -> float:
        """Receive rate over the trailing ``rate_window_s`` window.

        The divisor is the window itself, not the span of the surviving
        samples.  Dividing by the sample span would report a healthy rate for a
        stream that delivered a short burst and then went silent, which is
        precisely the failure this whole module exists to make visible.
        """
        now = _now(now_s)
        self._trim_window(now)
        if len(self._window) < 2 or self.first_rx_s is None:
            return 0.0
        span = min(self.rate_window_s, max(1e-6, now - self.first_rx_s))
        return (len(self._window) - 1) / span

    def poll_stall(self, now_s: Optional[float] = None
                   ) -> Optional[Tuple[str, float]]:
        """Return a transition tuple the first tick a stall starts or ends.

        Returns ``('stall', silent_s)`` on the transition into a stall,
        ``('recover', silent_s)`` on the transition out of one, else ``None``.
        A stream that has never received anything cannot stall: there is no
        evidence it was ever alive, and reporting it would bury the real signal.
        """
        if self.stall_warn_s <= 0.0 or self.last_rx_s is None:
            return None
        now = _now(now_s)
        silent = max(0.0, now - self.last_rx_s)
        if not self.stalled and silent > self.stall_warn_s:
            self.stalled = True
            self.stall_started_s = now
            self.stall_events += 1
            return ('stall', silent)
        if self.stalled and silent <= self.stall_warn_s:
            self.stalled = False
            self.stall_started_s = None
            return ('recover', silent)
        return None

    # ------------------------------------------------------------------
    def snapshot(self, now_s: Optional[float] = None) -> Dict:
        now = _now(now_s)
        return {
            'name': self.name,
            'count': int(self.count),
            'age_s': self.age_s(now),
            'rate_hz': self.rate_hz(now),
            'max_gap_s': float(self.max_gap_s),
            'stalled': bool(self.stalled),
            'stall_events': int(self.stall_events),
        }

    def format_compact(self, now_s: Optional[float] = None) -> str:
        """One ``name=n/rate/age/maxgap`` field for the periodic report line."""
        snap = self.snapshot(now_s)
        age = snap['age_s']
        age_text = '--' if age is None else '{:.2f}'.format(age)
        return '{}=n{} {:.1f}Hz age{} gap{:.2f}{}'.format(
            snap['name'],
            snap['count'],
            snap['rate_hz'],
            age_text,
            snap['max_gap_s'],
            ' STALLED' if snap['stalled'] else '',
        )


class RxDiagnostics(object):
    """Per-node receive counters plus a lifecycle event log.

    Used by :class:`control_node.stage_common.StageNodeBase` so every stage node
    emits the same two log shapes:

    ``[RXDIAG] node=... <stream fields> ...``  (periodic, one line)
    ``[RXEVENT] node=... ev=... <stream counters>``  (on each transition)

    Both lines carry the same counters, so grepping ``RXEVENT`` in one log and
    correlating with the bridge's own ``RX raw RGB=`` counter pins the freeze to
    a single transition.
    """

    def __init__(self, node_name: str, streams: Sequence[str] = (),
                 stall_warn_s: float = 2.0, report_period_s: float = 2.0,
                 rate_window_s: float = 5.0):
        self.node_name = str(node_name)
        self.report_period_s = max(0.0, float(report_period_s))
        self.streams = {}           # type: Dict[str, StreamRxStats]
        self._order = []            # type: List[str]
        for name in streams:
            self.add_stream(name, stall_warn_s=stall_warn_s,
                            rate_window_s=rate_window_s)
        self._last_report_s = None  # type: Optional[float]
        self.event_count = 0

    # ------------------------------------------------------------------
    def add_stream(self, name: str, stall_warn_s: float = 2.0,
                   rate_window_s: float = 5.0) -> StreamRxStats:
        if name not in self.streams:
            self.streams[name] = StreamRxStats(
                name, stall_warn_s=stall_warn_s, rate_window_s=rate_window_s)
            self._order.append(name)
        return self.streams[name]

    def record(self, name: str, now_s: Optional[float] = None) -> None:
        stream = self.streams.get(name)
        if stream is None:
            stream = self.add_stream(name)
        stream.record(now_s)

    def counters(self, now_s: Optional[float] = None) -> str:
        """Compact ``rgb=n123/0.07`` counter list used inside event lines."""
        now = _now(now_s)
        parts = []
        for name in self._order:
            stream = self.streams[name]
            age = stream.age_s(now)
            age_text = '--' if age is None else '{:.2f}'.format(age)
            parts.append('{}=n{}/{}'.format(name, stream.count, age_text))
        return ' '.join(parts)

    # ------------------------------------------------------------------
    def note_event(self, event: str, detail: str = '',
                   now_s: Optional[float] = None) -> str:
        """Format (but do not emit) one lifecycle marker line."""
        now = _now(now_s)
        self.event_count += 1
        text = '[RXEVENT] node={} t={:.3f} ev={}'.format(
            self.node_name, now, event)
        if detail:
            text += ' detail={}'.format(detail)
        counters = self.counters(now)
        if counters:
            text += ' | ' + counters
        return text

    def poll_stalls(self, now_s: Optional[float] = None
                    ) -> List[Tuple[str, str, float]]:
        """Return ``(stream, 'stall'|'recover', silent_s)`` transitions."""
        now = _now(now_s)
        out = []
        for name in self._order:
            transition = self.streams[name].poll_stall(now)
            if transition is not None:
                out.append((name, transition[0], transition[1]))
        return out

    def due_report(self, now_s: Optional[float] = None) -> Optional[str]:
        """Return the periodic report line, or None if the period has not elapsed."""
        if self.report_period_s <= 0.0:
            return None
        now = _now(now_s)
        if (self._last_report_s is not None and
                now - self._last_report_s < self.report_period_s):
            return None
        self._last_report_s = now
        return self.format_report(now)

    def format_report(self, now_s: Optional[float] = None,
                      extra: str = '') -> str:
        now = _now(now_s)
        parts = [self.streams[name].format_compact(now) for name in self._order]
        text = '[RXDIAG] node={} t={:.3f}'.format(self.node_name, now)
        if parts:
            text += ' | ' + ' | '.join(parts)
        if extra:
            text += ' | ' + extra
        return text
