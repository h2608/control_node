#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendored Robot_Ctrl LCM helper.

Copied from the previously installed second_stage/my_gait.py so the package no
longer depends on a stale build. Differences from the original:
- imports resolve inside the local ``control_node`` package;
- the two worker threads are daemonized and quit() joins with a timeout, so a
  stage node cannot hang on shutdown while lc.handle() blocks waiting for
  LCM traffic.

This module is the single copy of the helper: a workspace-root variant that had
grown a ``get_status()`` accessor and a stricter ``Wait_finish()`` was folded
back in here, so both response accessors now read the same locked state.
"""

import site
import sys
import time
from threading import Thread, Lock

try:
    import lcm
except ModuleNotFoundError as exc:
    # The Galactic image builds LCM under /usr/local with CMake's conventional
    # ``site-packages`` destination, while Debian's system Python searches
    # ``dist-packages`` instead.  Keep normal installations untouched and add
    # the CMake location only when the lcm module itself is missing.
    if exc.name != 'lcm':
        raise
    site.addsitedir(
        '/usr/local/lib/python{}.{}'
        '/site-packages'.format(sys.version_info.major, sys.version_info.minor)
    )
    import lcm

from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from control_node.robot_control_response_lcmt import robot_control_response_lcmt
from control_node.state_estimator_lcmt import state_estimator_lcmt


class Robot_Ctrl(object):
    def __init__(self):
        self.rec_thread = Thread(target=self.rec_responce)
        self.send_thread = Thread(target=self.send_publish)
        self.rec_thread.daemon = True
        self.send_thread.daemon = True
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = Lock()
        self.response_lock = Lock()
        self.response_seq = 0
        self.response_rx_monotonic_s = None
        self.last_incomplete_response_seq = 0
        self.last_incomplete_mode = 0
        self.last_incomplete_gait_id = 0
        self.last_incomplete_rx_monotonic_s = None
        self.delay_cnt = 0
        self.mode_ok = 0
        self.gait_ok = 0
        self.runing = 1
        # Do not heartbeat the zero-initialized LCM message.  mode=0 means
        # kOff in the locomotion controller, so a newly activated stage must
        # first provide an intentional command.
        self.command_ready = False
        self.first_command = True

    def run(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        self.send_thread.start()
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        # Timestamp callback ingress before taking the lock. A packet whose
        # callback started before a command barrier must remain distinguishable
        # from a response received after the command was fully published.
        rx_monotonic_s = time.monotonic()
        # Decode outside the lock: the result is thread-local until published
        # below, and keeping the critical section to plain field stores means a
        # malformed packet cannot stall response_snapshot() readers.
        rec_msg = robot_control_response_lcmt().decode(data)

        with self.response_lock:
            self.rec_msg = rec_msg
            self.response_seq += 1
            self.response_rx_monotonic_s = rx_monotonic_s
            if rec_msg.order_process_bar < 95:
                self.last_incomplete_response_seq = self.response_seq
                self.last_incomplete_mode = rec_msg.mode
                self.last_incomplete_gait_id = rec_msg.gait_id
                self.last_incomplete_rx_monotonic_s = rx_monotonic_s
            if rec_msg.order_process_bar >= 95:
                self.mode_ok = rec_msg.mode
                self.gait_ok = rec_msg.gait_id
            else:
                self.mode_ok = 0
                self.gait_ok = 0

    def response_snapshot(self):
        """Return one coherent action-response snapshot for non-blocking polls."""
        with self.response_lock:
            return {
                'seq': int(self.response_seq),
                'rx_monotonic_s': self.response_rx_monotonic_s,
                'mode': int(self.rec_msg.mode),
                'gait_id': int(self.rec_msg.gait_id),
                'order_process_bar': int(self.rec_msg.order_process_bar),
                'switch_status': int(self.rec_msg.switch_status),
                'ori_error': int(self.rec_msg.ori_error),
                'footpos_error': int(self.rec_msg.footpos_error),
                'motor_error': [int(value) for value in self.rec_msg.motor_error],
                'last_incomplete_seq': int(self.last_incomplete_response_seq),
                'last_incomplete_mode': int(self.last_incomplete_mode),
                'last_incomplete_gait_id': int(self.last_incomplete_gait_id),
                'last_incomplete_rx_monotonic_s': (
                    self.last_incomplete_rx_monotonic_s),
            }

    def get_status(self):
        """Return one internally consistent snapshot of the latest response.

        A shorter-keyed view of response_snapshot() that reports staleness as an
        age instead of a receive time.  Unlike mode_ok/gait_ok it is kept for
        every progress value, including the intentional upright hold at
        progress 50.  ``age_s`` is infinite until the first response arrives, so
        a caller can never read the zero-initialized message as live state.
        """
        snapshot = self.response_snapshot()
        rx = snapshot['rx_monotonic_s']
        return {
            'mode': snapshot['mode'],
            'gait': snapshot['gait_id'],
            'progress': snapshot['order_process_bar'],
            'switch_status': snapshot['switch_status'],
            'ori_error': snapshot['ori_error'],
            'footpos_error': snapshot['footpos_error'],
            'motor_error': tuple(snapshot['motor_error']),
            'age_s': (
                float('inf')
                if snapshot['seq'] <= 0 or rx is None
                else time.monotonic() - rx
            ),
        }

    def rec_responce(self):
        while self.runing:
            # handle() can block forever and used to make stage hand-off wait
            # for the full join timeout when response traffic disappeared.
            # LCM's bounded wait lets quit() stop the receiver promptly.
            self.lc_r.handle_timeout(50)

    def Wait_finish(self, mode, gait_id):
        count = 0
        while self.runing and count < 2000: #10s
            # Test the progress bar explicitly rather than trusting the
            # mode_ok/gait_ok latch: that latch is *reset to zero* on an
            # incomplete response, so a Wait_finish(0, 0) caller would see an
            # unfinished action as finished.
            status = self.response_snapshot()
            if (
                status['mode'] == mode
                and status['gait_id'] == gait_id
                and status['order_process_bar'] >= 95
            ):
                return True
            time.sleep(0.005)
            count += 1
        return False

    def send_publish(self):
        while self.runing:
            with self.send_lock:
                if self.command_ready:
                    if self.delay_cnt > 20: # Heartbeat signal 10HZ, It is used to maintain the heartbeat when life count is not updated
                        self.lc_s.publish("robot_control_cmd", self.cmd_msg.encode())
                        self.delay_cnt = 0
                    self.delay_cnt += 1
            time.sleep( 0.005 )

    def Send_cmd(self, msg):
        with self.send_lock:
            # The stage state machines reuse and mutate one message object.
            # Keep an immutable-by-convention snapshot for the heartbeat so it
            # can never encode a new life_count with an old/partial payload.
            snapshot = robot_control_cmd_lcmt.decode(msg.encode())
            if self.first_command:
                # The locomotion controller accepts a command only when its
                # life_count differs from the previously accepted value.
                # Separate stage processes start their counters from zero, so
                # the first value can collide with the previous stage.  Send
                # the same intentional command with two consecutive values:
                # at least one must differ, and the final value becomes the
                # heartbeat/caller baseline.
                self.lc_s.publish("robot_control_cmd", snapshot.encode())
                # The receiver stores only its latest packet.  Leave enough
                # time for the locomotion loop to observe the first value
                # before publishing the second one.
                time.sleep(0.02)
                snapshot.life_count = (
                    1 if int(snapshot.life_count) >= 127
                    else int(snapshot.life_count) + 1
                )
                # Callers keep their own counter in the reusable message.
                msg.life_count = snapshot.life_count
                self.lc_s.publish("robot_control_cmd", snapshot.encode())
                self.first_command = False

            self.delay_cnt = 50
            self.cmd_msg = snapshot
            self.command_ready = True

    def Send_cmd_with_response_barrier(self, msg):
        """Publish msg and return a response generation/time barrier.

        Holding response_lock across the publish prevents a response callback
        from being committed between the baseline snapshot and completed send.
        The callback ingress timestamp also preserves pre-send packet identity.
        """
        with self.response_lock:
            response_seq = int(self.response_seq)
            self.Send_cmd(msg)
            sent_monotonic_s = time.monotonic()
        return response_seq, sent_monotonic_s

    def quit(self):
        self.runing = 0
        # Stop the publisher first so complete_stage() cannot spend seconds
        # waiting while the previous command is still heartbeating.
        if self.send_thread.is_alive():
            self.send_thread.join(timeout=0.2)
        if self.rec_thread.is_alive():
            self.rec_thread.join(timeout=0.2)


class Robot_Odom(object):
    """Read-only LCM reader for the locomotion state estimator.

    ``robot_runner.cpp`` publishes ``state_estimator`` (``state_estimator_lcmt``)
    on port 7669 at 50 Hz on both the simulator and the robot, so this is the
    one odometry/attitude source that does not depend on the simulator-only
    ``cyberdog_visual`` TF bridge.  It never sends anything: keeping it separate
    from ``Robot_Ctrl`` means an odometry consumer can never touch the command
    link.  Snapshots carry a monotonic receive time and a generation counter so
    callers can fail closed on a frozen stream instead of integrating stale
    samples.
    """

    LCM_URL = 'udpm://239.255.76.67:7669?ttl=255'
    CHANNEL = 'state_estimator'

    def __init__(self, lcm_url=None):
        self.lc = lcm.LCM(lcm_url or self.LCM_URL)
        self.state_lock = Lock()
        self.runing = 1
        self.seq = 0
        self.rx_monotonic_s = None
        self.msg = state_estimator_lcmt()
        self.rec_thread = Thread(target=self.rec_state)
        self.rec_thread.daemon = True

    def run(self):
        self.lc.subscribe(self.CHANNEL, self.msg_handler)
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        rx_monotonic_s = time.monotonic()
        try:
            msg = state_estimator_lcmt.decode(data)
        except ValueError:
            # A foreign packet on the multicast group must not corrupt the
            # last good sample or advance the generation counter.
            return
        with self.state_lock:
            self.msg = msg
            self.seq += 1
            self.rx_monotonic_s = rx_monotonic_s

    def rec_state(self):
        while self.runing:
            self.lc.handle_timeout(50)

    def snapshot(self):
        """Return one coherent odometry/attitude sample.

        ``seq == 0`` means nothing has ever been received; callers must treat
        that and a stale ``rx_monotonic_s`` as unavailable, not as zero motion.
        """
        with self.state_lock:
            msg = self.msg
            return {
                'seq': int(self.seq),
                'rx_monotonic_s': self.rx_monotonic_s,
                'p': [float(v) for v in msg.p],
                'rpy': [float(v) for v in msg.rpy],
                'v_world': [float(v) for v in msg.vWorld],
                'v_body': [float(v) for v in msg.vBody],
                'contact': [float(v) for v in msg.contactEstimate],
                'timestamp': int(msg.timestamp),
            }

    def quit(self):
        self.runing = 0
        if self.rec_thread.is_alive():
            self.rec_thread.join(timeout=0.2)
