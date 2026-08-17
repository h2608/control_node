#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the receive-diagnostics bookkeeping and ingestion policy."""

from control_node.rx_diagnostics import RxDiagnostics, StreamRxStats

import pytest


# ----------------------------------------------------------------------
# StreamRxStats
# ----------------------------------------------------------------------
def test_first_message_starts_the_stream_without_a_gap():
    stream = StreamRxStats('rgb')
    stream.record(100.0)
    assert stream.count == 1
    assert stream.first_rx_s == 100.0
    assert stream.max_gap_s == 0.0
    assert stream.age_s(100.5) == pytest.approx(0.5)


def test_max_gap_tracks_the_worst_inter_arrival_time():
    stream = StreamRxStats('rgb')
    for t in (0.0, 0.1, 0.2, 0.63, 0.7):
        stream.record(t)
    assert stream.max_gap_s == pytest.approx(0.43)


def test_rate_uses_only_the_trailing_window():
    stream = StreamRxStats('rgb', rate_window_s=1.0)
    # 10 Hz for two seconds; only the last second may count.
    for i in range(21):
        stream.record(i * 0.1)
    assert stream.rate_hz(2.0) == pytest.approx(10.0, abs=0.5)


def test_rate_decays_while_a_stream_is_silent():
    # A burst followed by silence must not keep reading as a healthy rate: the
    # divisor is the window, not the span of the surviving samples.
    stream = StreamRxStats('rgb', rate_window_s=5.0)
    for i in range(11):
        stream.record(i * 0.1)
    assert stream.rate_hz(1.0) == pytest.approx(10.0, abs=0.5)
    assert stream.rate_hz(4.0) < 3.0
    assert stream.rate_hz(10.0) == 0.0


def test_never_received_stream_never_reports_a_stall():
    # A topic that was never published is a configuration problem, not a
    # dropout, and must not drown out the stream that really froze.
    stream = StreamRxStats('rgb', stall_warn_s=1.0)
    assert stream.poll_stall(1000.0) is None
    assert stream.stall_events == 0


def test_stall_and_recovery_each_fire_once():
    stream = StreamRxStats('rgb', stall_warn_s=1.0)
    stream.record(0.0)
    assert stream.poll_stall(0.5) is None
    kind, silent = stream.poll_stall(2.0)
    assert kind == 'stall'
    assert silent == pytest.approx(2.0)
    # Still stalled: no repeat event.
    assert stream.poll_stall(3.0) is None
    assert stream.stall_events == 1

    stream.record(3.5)
    kind, _silent = stream.poll_stall(3.6)
    assert kind == 'recover'
    assert stream.poll_stall(3.7) is None


def test_stall_detection_disabled_by_zero_threshold():
    stream = StreamRxStats('rgb', stall_warn_s=0.0)
    stream.record(0.0)
    assert stream.poll_stall(100.0) is None


# ----------------------------------------------------------------------
# RxDiagnostics
# ----------------------------------------------------------------------
def test_counters_render_every_declared_stream():
    diag = RxDiagnostics('stage2_node', streams=('rgb', 'depth'))
    diag.record('rgb', 10.0)
    text = diag.counters(10.25)
    assert 'rgb=n1/0.25' in text
    # Depth has never arrived: the age must be shown as unknown, not as 0.
    assert 'depth=n0/--' in text


def test_recording_an_undeclared_stream_creates_it():
    diag = RxDiagnostics('stage2_node', streams=('rgb',))
    diag.record('fisheye_left', 1.0)
    assert 'fisheye_left' in diag.streams


def test_event_line_carries_the_counters():
    diag = RxDiagnostics('stage2_node', streams=('rgb',))
    diag.record('rgb', 5.0)
    line = diag.note_event('ACTIVATE_BEGIN', 'reason=test', now_s=5.5)
    assert '[RXEVENT]' in line
    assert 'node=stage2_node' in line
    assert 'ev=ACTIVATE_BEGIN' in line
    assert 'detail=reason=test' in line
    assert 'rgb=n1/0.50' in line


def test_due_report_respects_the_period():
    diag = RxDiagnostics('n', streams=('rgb',), report_period_s=2.0)
    assert diag.due_report(0.0) is not None
    assert diag.due_report(1.0) is None
    assert diag.due_report(2.5) is not None


def test_zero_period_disables_the_periodic_report():
    diag = RxDiagnostics('n', streams=('rgb',), report_period_s=0.0)
    assert diag.due_report(0.0) is None
    assert diag.due_report(100.0) is None


def test_poll_stalls_reports_each_stream_separately():
    diag = RxDiagnostics('n', streams=('rgb', 'depth'), stall_warn_s=1.0)
    diag.record('rgb', 0.0)
    diag.record('depth', 0.0)
    diag.record('depth', 9.5)
    transitions = diag.poll_stalls(10.0)
    assert transitions == [('rgb', 'stall', pytest.approx(10.0))]


def test_report_line_marks_the_stalled_stream():
    diag = RxDiagnostics('n', streams=('rgb',), stall_warn_s=1.0)
    diag.record('rgb', 0.0)
    diag.poll_stalls(10.0)
    assert 'STALLED' in diag.format_report(10.0)
