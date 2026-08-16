#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone test for the custom CyberDog2 upright-hold gait.

Sequence:
    RecoveryStand (12/0) -> TwoLegStand upright hold (64/4, progress 50)
    -> timed hold -> smooth QpStand return (3/0)
    -> zero-velocity Locomotion stand (11/3)

Run this only when this process is the sole publisher of robot_control_cmd.
"""

import argparse
import sys
import time

from control_node.my_gait import Robot_Ctrl
from control_node.robot_control_cmd_lcmt import robot_control_cmd_lcmt


MODE_RECOVERY_STAND = 12
GAIT_RECOVERY_STAND = 0
MODE_QP_STAND = 3
MODE_LOCOMOTION = 11
GAIT_LOCOMOTION_STAND = 3
MODE_TWO_LEG_STAND = 64
GAIT_UPRIGHT_HOLD = 4
UPRIGHT_PROGRESS = 50
FINISHED_PROGRESS = 95


class TestFailure(RuntimeError):
    pass


def format_status(status):
    age = status['age_s']
    age_text = 'inf' if age == float('inf') else f'{age:.3f}'
    return (
        f"mode={status['mode']} gait={status['gait']} "
        f"progress={status['progress']} age={age_text}s "
        f"switch={status['switch_status']} "
        f"ori_error={status['ori_error']} "
        f"footpos_error={status['footpos_error']} "
        f"motor_error={status['motor_error']}"
    )


def has_fault(status):
    return (
        status['ori_error'] != 0
        or status['footpos_error'] != 0
        or any(value != 0 for value in status['motor_error'])
    )


def increment_life_count(msg):
    msg.life_count = (
        0 if int(msg.life_count) >= 127
        else int(msg.life_count) + 1
    )


def clear_payload(msg):
    """Prevent fields left by an earlier command from leaking into a mode."""
    msg.contact = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.rpy_des = [0.0, 0.0, 0.0]
    msg.pos_des = [0.0, 0.0, 0.0]
    msg.acc_des = [0.0] * 6
    msg.ctrl_point = [0.0, 0.0, 0.0]
    msg.foot_pose = [0.0] * 6
    msg.step_height = [0.0, 0.0]
    msg.value = 0
    msg.duration = 0


def send_mode(ctrl, msg, mode, gait, label):
    clear_payload(msg)
    msg.mode = int(mode)
    msg.gait_id = int(gait)
    increment_life_count(msg)
    ctrl.Send_cmd(msg)
    print(
        f'[{label}] sent mode={mode} gait={gait} '
        f'life_count={msg.life_count}',
        flush=True,
    )


def wait_for_status(ctrl, predicate, timeout_s, feedback_max_age_s, label):
    deadline = time.monotonic() + timeout_s
    next_log = 0.0

    while time.monotonic() < deadline:
        status = ctrl.get_status()
        now = time.monotonic()

        if predicate(status):
            print(f'[{label}] reached: {format_status(status)}', flush=True)
            return status

        if now >= next_log:
            print(f'[{label}] waiting: {format_status(status)}', flush=True)
            next_log = now + 0.5

        # age=inf is normal before the first response. Once at least one
        # response has arrived, reject a link that becomes stale.
        if (
            status['age_s'] != float('inf')
            and status['age_s'] > feedback_max_age_s
        ):
            raise TestFailure(
                f'{label}: response became stale: {format_status(status)}'
            )

        time.sleep(0.02)

    status = ctrl.get_status()
    raise TestFailure(
        f'{label}: timeout after {timeout_s:.1f}s: {format_status(status)}'
    )


def wait_recovery(ctrl, timeout_s, feedback_max_age_s, label):
    return wait_for_status(
        ctrl,
        lambda status: (
            status['age_s'] <= feedback_max_age_s
            and status['mode'] == MODE_RECOVERY_STAND
            and status['gait'] == GAIT_RECOVERY_STAND
            and status['progress'] >= FINISHED_PROGRESS
        ),
        timeout_s,
        feedback_max_age_s,
        label,
    )


def recover_with_retry(
    ctrl,
    msg,
    timeout_s,
    feedback_max_age_s,
    retry_after_s,
    label,
):
    """Recover to four-leg stand, retransmitting once if progress is stuck.

    TwoLegStand can report mode=12/gait=0 while the recovery order remains at
    progress 0.  A new life_count retriggers that order after the FSM has
    finished leaving TwoLegStand.
    """
    send_mode(
        ctrl,
        msg,
        MODE_RECOVERY_STAND,
        GAIT_RECOVERY_STAND,
        label,
    )

    deadline = time.monotonic() + timeout_s
    retry_at = time.monotonic() + retry_after_s
    retried = False
    next_log = 0.0

    while time.monotonic() < deadline:
        status = ctrl.get_status()
        now = time.monotonic()

        reached = (
            status['age_s'] <= feedback_max_age_s
            and status['mode'] == MODE_RECOVERY_STAND
            and status['gait'] == GAIT_RECOVERY_STAND
            and status['progress'] >= FINISHED_PROGRESS
        )
        if reached:
            print(f'[{label}] reached: {format_status(status)}', flush=True)
            return status

        if (
            not retried
            and now >= retry_at
            and status['age_s'] <= feedback_max_age_s
            and status['mode'] == MODE_RECOVERY_STAND
            and status['gait'] == GAIT_RECOVERY_STAND
            and status['progress'] == 0
        ):
            print(
                f'[{label}] progress stayed at 0 for {retry_after_s:.1f}s; '
                'retriggering with a new life_count.',
                flush=True,
            )
            send_mode(
                ctrl,
                msg,
                MODE_RECOVERY_STAND,
                GAIT_RECOVERY_STAND,
                label + '_RETRY',
            )
            retried = True
            # Give the retriggered recovery a complete timeout window.
            deadline = time.monotonic() + timeout_s
            next_log = 0.0
            continue

        if now >= next_log:
            print(f'[{label}] waiting: {format_status(status)}', flush=True)
            next_log = now + 0.5

        if (
            status['age_s'] != float('inf')
            and status['age_s'] > feedback_max_age_s
        ):
            raise TestFailure(
                f'{label}: response became stale: {format_status(status)}'
            )

        time.sleep(0.02)

    status = ctrl.get_status()
    raise TestFailure(
        f'{label}: recovery timeout after retry: {format_status(status)}'
    )


def enter_ready_stand(ctrl, msg, timeout_s, feedback_max_age_s):
    """Enter the ordinary zero-velocity stance used before later actions."""
    send_mode(
        ctrl,
        msg,
        MODE_LOCOMOTION,
        GAIT_LOCOMOTION_STAND,
        'READY_STAND',
    )
    return wait_for_status(
        ctrl,
        lambda status: (
            status['age_s'] <= feedback_max_age_s
            and status['mode'] == MODE_LOCOMOTION
            and status['gait'] == GAIT_LOCOMOTION_STAND
            and not has_fault(status)
        ),
        timeout_s,
        feedback_max_age_s,
        'READY_STAND',
    )


def smooth_return_to_qp_stand(
    ctrl,
    msg,
    timeout_s,
    feedback_max_age_s,
):
    """Ask TwoLegStand to lower smoothly, without running RecoveryStand."""
    send_mode(ctrl, msg, MODE_QP_STAND, 0, 'SMOOTH_RETURN')
    return wait_for_status(
        ctrl,
        lambda status: (
            status['age_s'] <= feedback_max_age_s
            and status['mode'] == MODE_QP_STAND
            and status['progress'] >= FINISHED_PROGRESS
            and not has_fault(status)
        ),
        timeout_s,
        feedback_max_age_s,
        'SMOOTH_RETURN',
    )


def keep_ready_stand(ctrl, feedback_max_age_s):
    """Keep Robot_Ctrl alive so its heartbeat continues publishing stand."""
    print(
        '[READY_STAND] heartbeat is active; the robot is ready for the next '
        'action. Press Ctrl+C only when you want to stop this test.',
        flush=True,
    )
    next_log = 0.0
    while True:
        status = ctrl.get_status()
        now = time.monotonic()
        if (
            status['age_s'] != float('inf')
            and status['age_s'] > feedback_max_age_s
        ):
            raise TestFailure(
                'READY_STAND: response became stale: '
                + format_status(status)
            )
        if has_fault(status):
            raise TestFailure(
                'READY_STAND: controller fault: ' + format_status(status)
            )
        if now >= next_log:
            print('[READY_STAND] holding: ' + format_status(status), flush=True)
            next_log = now + 2.0
        time.sleep(0.05)


def wait_upright(ctrl, timeout_s, feedback_max_age_s):
    deadline = time.monotonic() + timeout_s
    next_log = 0.0

    while time.monotonic() < deadline:
        status = ctrl.get_status()
        now = time.monotonic()

        if (
            status['age_s'] <= feedback_max_age_s
            and status['mode'] == MODE_TWO_LEG_STAND
            and status['gait'] == GAIT_UPRIGHT_HOLD
        ):
            if status['progress'] == UPRIGHT_PROGRESS:
                if has_fault(status):
                    raise TestFailure(
                        'upright reached with a controller fault: '
                        + format_status(status)
                    )
                print(
                    '[UPRIGHT] progress==50, upright hold reached: '
                    + format_status(status),
                    flush=True,
                )
                return status

            # For gait 4, progress 75/100 is not a successful hold. It usually
            # means the new lower-controller binary is not active or an
            # internal failure/recovery path was taken.
            if status['progress'] >= 75:
                raise TestFailure(
                    'gait 4 did not stay at progress 50: '
                    + format_status(status)
                )

        if now >= next_log:
            print('[UPRIGHT] waiting: ' + format_status(status), flush=True)
            next_log = now + 0.5

        if (
            status['age_s'] != float('inf')
            and status['age_s'] > feedback_max_age_s
        ):
            raise TestFailure(
                'UPRIGHT: response became stale: ' + format_status(status)
            )

        time.sleep(0.02)

    status = ctrl.get_status()
    raise TestFailure(
        f'UPRIGHT: timeout after {timeout_s:.1f}s: {format_status(status)}'
    )


def hold_upright(ctrl, hold_s, feedback_max_age_s):
    deadline = time.monotonic() + hold_s
    next_log = 0.0

    while time.monotonic() < deadline:
        status = ctrl.get_status()
        now = time.monotonic()

        valid = (
            status['age_s'] <= feedback_max_age_s
            and status['mode'] == MODE_TWO_LEG_STAND
            and status['gait'] == GAIT_UPRIGHT_HOLD
            and status['progress'] == UPRIGHT_PROGRESS
        )
        if not valid:
            raise TestFailure(
                'upright state changed during hold: ' + format_status(status)
            )
        if has_fault(status):
            raise TestFailure(
                'controller fault during upright hold: '
                + format_status(status)
            )

        if now >= next_log:
            remaining = max(0.0, deadline - now)
            print(
                f'[HOLD] remaining={remaining:.2f}s '
                + format_status(status),
                flush=True,
            )
            next_log = now + 0.25

        time.sleep(0.02)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test CyberDog2 mode=64/gait=4 upright hold.'
    )
    parser.add_argument(
        '--hold-seconds',
        type=float,
        default=1.0,
        help='Upright hold time after progress reaches exactly 50.',
    )
    parser.add_argument(
        '--timeout-seconds',
        type=float,
        default=12.0,
        help='Timeout for prepare, rise, and recovery phases.',
    )
    parser.add_argument(
        '--feedback-max-age',
        type=float,
        default=0.6,
        help='Maximum accepted age of a received LCM response.',
    )
    parser.add_argument(
        '--recovery-retry-after',
        type=float,
        default=2.5,
        help='Retrigger RecoveryStand once when progress remains 0 this long.',
    )
    parser.add_argument(
        '--exit-after-recovery',
        action='store_true',
        help='Exit instead of keeping the zero-velocity standing heartbeat.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hold_seconds < 0.0:
        print('--hold-seconds must be >= 0', file=sys.stderr)
        return 2
    if (
        args.timeout_seconds <= 0.0
        or args.feedback_max_age <= 0.0
        or args.recovery_retry_after <= 0.0
    ):
        print('timeouts must be > 0', file=sys.stderr)
        return 2

    ctrl = Robot_Ctrl()
    msg = robot_control_cmd_lcmt()
    needs_recovery = False
    test_ok = False

    print('WARNING: ensure no other process publishes robot_control_cmd.')
    print(
        'Starting safe sequence: '
        '12/0 -> 64/4 -> hold -> smooth 3/0 -> zero-velocity 11/3'
    )

    ctrl.run()
    try:
        # The first command is deliberately safe. Robot_Ctrl may publish the
        # first command with two life_count values to avoid a counter collision.
        send_mode(
            ctrl,
            msg,
            MODE_RECOVERY_STAND,
            GAIT_RECOVERY_STAND,
            'PREPARE',
        )
        wait_recovery(
            ctrl,
            args.timeout_seconds,
            args.feedback_max_age,
            'PREPARE',
        )

        needs_recovery = True
        send_mode(
            ctrl,
            msg,
            MODE_TWO_LEG_STAND,
            GAIT_UPRIGHT_HOLD,
            'UPRIGHT',
        )
        wait_upright(
            ctrl,
            args.timeout_seconds,
            args.feedback_max_age,
        )

        hold_upright(ctrl, args.hold_seconds, args.feedback_max_age)

        # Do not enter RecoveryStand directly from the high biped pose. Its
        # OnEnter height check can classify that pose as abnormal and choose a
        # FoldLegs path. Mode 3 asks the modified TwoLegStand state to run
        # jump_flag 7 and place the front feet down before entering QpStand.
        smooth_return_to_qp_stand(
            ctrl,
            msg,
            args.timeout_seconds,
            args.feedback_max_age,
        )
        enter_ready_stand(
            ctrl,
            msg,
            args.timeout_seconds,
            args.feedback_max_age,
        )
        needs_recovery = False
        test_ok = True
        print('[PASS] upright gait completed; normal standing mode is active.')
        if not args.exit_after_recovery:
            keep_ready_stand(ctrl, args.feedback_max_age)

    except KeyboardInterrupt:
        if test_ok:
            print(
                '\n[STOP] test heartbeat stopped. In the full controller, '
                'continue to the next action here instead of calling quit().'
            )
        else:
            print('\n[INTERRUPTED] Ctrl+C received; requesting recovery.')
    except TestFailure as exc:
        print(f'[FAIL] {exc}', file=sys.stderr, flush=True)
    finally:
        if needs_recovery:
            try:
                recover_with_retry(
                    ctrl,
                    msg,
                    args.timeout_seconds,
                    args.feedback_max_age,
                    args.recovery_retry_after,
                    'FINALLY_RECOVER',
                )
                print('[FINALLY_RECOVER] recovery confirmed.', flush=True)
            except Exception as exc:  # Keep heartbeat alive until this attempt ends.
                print(
                    f'[FINALLY_RECOVER] recovery could not be confirmed: {exc}',
                    file=sys.stderr,
                    flush=True,
                )
        ctrl.quit()

    return 0 if test_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
