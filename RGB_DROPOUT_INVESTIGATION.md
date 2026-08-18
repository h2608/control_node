# Physical-robot RGB dropout: findings, instrumentation and bisect plan

Branch `stage2/rgb-dropout-diagnosis`, worktree `worktrees/stage2-rgb-dropout`
(a worktree of the nested `src/control_node` repository, branched from `main`
at `ab9e4e9`).

Symptom being investigated: with the C++ compression bridge running alone,
`/mi_desktop_48_b0_2d_7b_00_e2/image_rgb` is delivered stably.  Starting the
competition control stack makes the bridge's raw RGB receive counter freeze
permanently (`RX raw RGB=151` repeated forever) while the publisher keeps
producing messages and the bridge process keeps logging.

**Nothing here has been reproduced on hardware.**  Section 2 is what the code
says; section 3 is ranked speculation; section 5 is the test that would decide
between them.

---

## 0. Scope problem you need to know about first

The current robot-side code is **not** in this repository.

| file | repo `main` (`ab9e4e9`) | what runs on the robot |
| --- | --- | --- |
| `stage2_node.py` | 2905 lines, subscribes to raw RGB + depth + both fisheyes and runs OpenCV on the robot | 3222 lines, consumes `/stage2/perception/*` `Float32MultiArray` results from the PC and destroys the inherited raw subscriptions (present in `机器狗ROS包备份8-17-605/cyberdog_ws/`, **not** committed) |
| `stage1_node.py` | no `p1_rgb_stale_stop_sec` | has RGB stale protection per your description — not present in the repo or in either snapshot directory |
| C++ compression bridge | absent | only on the robot / PC; the repo has the superseded Python `stage2_camera_bridge.py` in the snapshot directories only |
| PC-side `stage2_perception_node.py` | absent | only on the PC |

So this branch changes only the **shared infrastructure that is in the
repository** — `stage_common.py`, `real_controller.py`, `mission_control_node.py`,
`stage4_node.py`, the launch file.  That happens to be where the strongest
finding lives.

**Deployment rule for this branch:** sync `stage_common.py`, `ingest_policy.py`,
`rx_diagnostics.py`, `robot_interface/real_controller.py`,
`mission_control_node.py`, `stage4_node.py` and
`launch/full_competition.launch.py`.  Do **not** sync `stage1_node.py` or
`stage2_node.py` — the robot's copies are newer than this branch's, and
overwriting them would revert the perception offload and the Stage 1 stale
guard.  See §6 for the one-line edit the robot's `stage2_node.py` needs.

---

## 1. What was inspected

`full_competition.launch.py`, `mission_control_node.py`, `stage_common.py`
(`StageNodeBase` lifecycle, image callbacks, QoS, TF), `robot_interface/`
(`factory.py`, `real_controller.py`, `sim_controller.py`), the startup
RecoveryStand path, `stage1/3/4/5/6_node.py` subscription and executor setup,
the robot-side `stage2_node.py` from the snapshot directory, and
`stage4_vision_preview.py` / `stage2_vision_preview.py` as additional readers.

---

## 2. Confirmed code-level findings

### 2.1 `full_competition` creates five to six independent raw RGB readers, all from launch — this is the headline finding

`full_competition.launch.py` starts `stage1_node` … `stage6_node` as six
separate processes plus `mission_control_node`.  Every stage node inherits
`StageNodeBase.__init__`, which (before this branch) unconditionally created:

```python
self.rgb_sub   = self.create_subscription(Image, self.rgb_topic,   self.rgb_callback,   qos_profile_sensor_data)
self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
```

at construction time — **before activation, and never destroyed**.  Only one
stage is ever active, but all six subscribe for the whole run.

`rgb_callback` returns early while inactive:

```python
def rgb_callback(self, msg):
    self.latest_rgb_msg = msg
    if not self.active or self.finished:
        return
```

That early return saves the cv_bridge conversion.  It does **not** save
anything below it: by the time the callback runs, rclpy has already taken the
sample out of the middleware and fully deserialised a multi-hundred-kilobyte
`sensor_msgs/Image` into a Python object.  The DDS delivery, the copy and the
deserialisation are paid in full by every inactive stage node on every frame.

Per-node accounting on `main`:

| node | raw RGB reader | raw depth reader | extra |
| --- | --- | --- | --- |
| `stage1_node` | base (always) | base (always) | |
| `stage2_node` (repo) | base (always) | base (always) | + 2 fisheye readers |
| `stage2_node` (robot) | destroyed in `__init__` | destroyed in `__init__` | 3 small `Float32MultiArray` readers |
| `stage3_node` | base (always) | base (always) | |
| `stage4_node` | dedicated `stage4_rgb_rx` node + own executor thread, **started in `__init__`** | base (always) | + AI camera reader |
| `stage5_node` | base (always) | base (always) | + `CameraInfo`, + `Imu` |
| `stage6_node` | base (always) | base (always) | |

So on the robot today, during Stage 2, the raw RGB topic has **five** control-stack
readers (stages 1, 3, 4, 5, 6) plus the bridge, and raw depth has **five**
(stages 1, 3, 4, 5, 6).  Plus six TF listeners.

This directly explains observation 7 in your report: turning Stage 2's own raw
subscriptions off changed nothing, because Stage 2 was never the problem — the
other five nodes were, and they subscribe whether or not they are the active
stage.  It also explains observation 5: the experiment that redirected the
bridge's fisheye topics only removed *bridge-side* readers; the five stage-node
readers were still there.

### 2.2 Stage 4 already contains a hardware-proven "the subscription wedged" recovery

`stage4_node.py` runs a watchdog (`_p4_accept_fresh_rgb_for_visual_state`,
around line 6058) that, when RGB has been silent for `2 * p4_rgb_stale_stop_s`,
does **not** restart the process — it destroys and recreates just the receiver
node/subscription (`_p4_restart_rgb_receiver`).  Somebody wrote that because a
raw RGB subscription stopped delivering on this robot while the topic stayed
alive, and rebuilding the endpoint brought it back.

That is independent, in-repo evidence that the failure is at the
subscription/DDS-endpoint level rather than at the publisher or the process
level, and it matches `ros2 topic hz` (a brand new participant) still reporting
9–16 Hz while the bridge's existing reader is frozen.

### 2.3 The physical motion path blocks the stage node's only executor thread

`RealRobotControlAdapter.move()` runs entirely inside the caller's thread — the
stage node's control-timer callback, i.e. the same single-threaded executor that
takes camera messages:

* `for _ in range(start_repeat): publish(SERVO_START); time.sleep(1/publish_hz)`
  → 5 × 50 ms = 250 ms of `time.sleep` on the first `move()` of every Servo
  session;
* then an ACK retry loop bounded by `real_servo_start_ack_timeout_s` (**2.0 s**);
* `stop_motion()` sleeps `(1 + end_repeat)/publish_hz` = **300 ms** every stop;
* `Wait_finish()` busy-polls with `time.sleep(0.01)` for up to
  `real_action_wait_timeout_s` = **45 s**.

While that thread is blocked, the node takes no camera messages, so its reader
queues fill.  With the pre-change `qos_profile_sensor_data` depth of 5, each
blocked node pins up to 5 full images per stream in the middleware.

This does not by itself stop a *different process's* subscription, but it is the
mechanism that turns "activation" into a synchronised, multi-hundred-millisecond
burst of reader back-pressure across the whole DDS host — which is exactly the
moment the failure is reported.

### 2.4 `mission_control_node` is not a plausible load source

1 Hz timer, one latched `Int32` publisher, two small subscriptions, and an
`call_async` RecoveryStand with no blocking wait.  It adds one DDS participant
and nothing else.  Do not spend hardware time on it before the others.

### 2.5 Nothing in the control stack turns the camera off

There is no service call, parameter set, or lifecycle transition anywhere in
`control_node` that could stop `stereo_camera`.  Your observation 1 is
consistent with the code: the control stack cannot switch the RGB source off.

---

## 3. Remaining hypotheses, ranked

**H1 — Reader-count and CPU pressure from the five always-on stage-node raw
readers pushes the bridge's reader over a DDS threshold it cannot recover from.**
Best supported. Explains: bridge-alone stability; the trigger being "start the
control stack" rather than any particular stage logic; Stage 2's own
raw-subscription removal not helping; the RGB-only bridge experiment still
failing; the publisher's own rate falling to 9–16 Hz (a publisher spending its
time serialising/sending to six readers); Stage 4 failing later rather than
immediately (its dedicated receiver thread keeps draining until the CPU gets
worse).  §4.1 removes four of the five readers, so the next hardware run tests
this directly.

**H2 — A specific Fast DDS endpoint pathology on the large-image topic (SHM port
marked unhealthy, or a best-effort reader history wedged on incomplete
fragmented samples), triggered by H1's pressure, not recovered automatically.**
This is the part that explains *permanence*, which H1 alone does not: CPU
starvation degrades a stream, it does not freeze it forever while a new
`ros2 topic hz` participant works fine.  A raw 640×480 `rgb8` Image is ~922 kB,
larger than Fast DDS's default 512 kB shared-memory segment, so it is very
likely travelling as fragmented UDP on loopback; chronic fragment loss under
load can leave a best-effort reader unable to complete any sample.  §4.4 adds
the same endpoint-rebuild recovery Stage 4 already proved works.  Verify with
`FASTRTPS_DEFAULT_PROFILES_FILE`/`RMW_IMPLEMENTATION` on the robot and by
raising Fast DDS log level during a failure.

**H3 — Executor blocking in the stage nodes (§2.3) is the specific trigger
event.**  Consistent with "Stage 2 fails almost immediately" (activation runs
`move()` → SERVO_START → up to 2 s blocked) and "Stage 4 fails after a while"
(preset actions and `Wait_finish` come later).  §4.2 (`image_qos_depth=1`) and
the `real_servo_publish_enabled` / `real_result_actions_enabled` switches
isolate this.

**H4 — Total DDS participant/endpoint count.**  Seven processes, plus one helper
node per active stage (`*_real_motion_io`) and one more for Stage 4's receiver.
Cheap to test with `auto_start:=false` (step 2 below) — if the freeze happens
with every node up but nothing activated and no raw subscriptions, this jumps to
the top.

**H5 — The 20 Hz `SERVO_DATA` stream itself.**  Small messages at a low rate;
implausible as a transport-level cause, but it is the only thing that starts
exactly at "the robot begins walking".  Directly switchable now
(`real_servo_publish_enabled:=false`).

**H6 — Something in the robot's vendor stack reacts to Servo/motion state
(thread priority, CPU affinity, camera power management).**  Not visible in this
repository.  Only reachable by the bisect: if the freeze survives step 5 but not
step 4, this and H5 are what is left.

---

## 4. Changes made on this branch

All defaults are chosen so that **simulation behaviour is byte-for-byte the same
as before** and the physical competition path changes only in the ways described.

### 4.1 Raw-image ingestion policy (`control_node/ingest_policy.py`, `stage_common.py`)

New parameters on every stage node:

* `raw_rgb_subscription`, `raw_depth_subscription` — `auto` | `always` |
  `on_activation` | `off`.
  `auto` resolves to `always` in simulation (unchanged) and to `on_activation`
  on `platform:=real`.
* The base class now creates and destroys these subscriptions on the
  activate/deactivate/complete transitions instead of holding them for the
  process lifetime.

Effect on the robot: **five always-on raw RGB readers become at most one**, and
the same for depth.  An unknown mode falls back to `auto` rather than failing to
launch.

`disable_raw_image_ingestion(reason)` is provided for a stage that consumes
perception results instead of images (§6).

### 4.2 Explicit image QoS depth (`image_qos_depth`)

`0` = auto: **1** on the robot, the rclpy sensor-data default (5) in sim.
Vision only ever consumes the newest frame, so a deeper history buys nothing and
costs four extra full frames the middleware must keep alive per reader — which
is exactly what a blocked executor (§2.3) pins.

### 4.3 Independently switchable startup components

| parameter | default | what `false` removes |
| --- | --- | --- |
| `auto_start` (mission control) | `true` | any stage activation at all |
| `startup_recovery_enabled` (already existed) | `true` | the startup RecoveryStand(111) |
| `stage_backend_enabled` | `true` | `create_robot_controller()`, the `*_real_motion_io` helper node, its executor thread, all Servo traffic |
| `real_servo_publish_enabled` | `true` | `MotionServoCmd` START/DATA/END publishing (helper node and executor still exist) |
| `real_result_actions_enabled` | `true` | `MotionResultCmd` service calls (reported as instantly complete) |
| `tf_listener_enabled` | `true` | `/tf` + `/tf_static` subscriptions (`get_current_pose()` then always returns `None`) |

Every one of these makes the robot **stand still**, never move more, and each
logs `[DIAG_MODE]` at WARN when engaged.  `real_servo_publish_enabled:=false`
also short-circuits the START/ACK wait so it does not burn 2 s per `move()`
waiting for an ACK that cannot arrive.

### 4.4 Wedged-reader self-heal (`raw_resubscribe_after_sec`)

A raw stream that has delivered at least once and then goes silent longer than
the threshold has its subscription destroyed and recreated, with a cooldown.
Default `-1` = auto: **3 s on the robot, off in simulation**.  This is the
generalisation of Stage 4's existing, hardware-proven recovery (§2.2).

It is recovery, not a safety relaxation: no stale frame is ever marked fresh, and
the Stage 1 / Stage 2 vision-stale stops keep holding the robot for exactly as
long as the stream is stale.  Every rebuild is logged at ERROR with a counter.

### 4.5 Receive diagnostics (`control_node/rx_diagnostics.py`)

Three log shapes, all on a **monotonic** clock so a stopped `/clock` or a blocked
executor cannot fake freshness:

```
[RXCFG]   node=... raw_rgb_subscription=... image_qos_depth=... stage_backend_enabled=...
[RXDIAG]  node=stage2_node t=1234.567 | rgb=n1834 14.9Hz age0.07 gap0.43 | depth=... | state=... active=... backend=real_motion_api servo_tx_data=921 ...
[RXEVENT] node=stage2_node t=1234.567 ev=SERVO_FIRST_DATA detail=motion_id=303 | rgb=n1834/0.07 depth=n0/--
[RXSTALL] node=stage2_node stream=rgb silent_for=2.10s state=... active=True     (ERROR)
[RXHEAL]  node=stage2_node stream=rgb silent for 3.02s; rebuilding the subscription (attempt 1)   (ERROR)
```

`[RXEVENT]` markers are also published as `std_msgs/String` on
**`/mission/diag/event`**, so one `ros2 topic echo` gives an ordered,
cross-process timeline of all seven control processes.  Markers emitted:

`NODE_INIT_DONE`, `RAW_SUBS`, `MISSION_MSG` (transitions only),
`ACTIVATE_BEGIN`, `BACKEND_CREATE_BEGIN`, `BACKEND_CREATE_DONE`,
`ACTIVATE_DONE`, `MOTION_IO_EXECUTOR_START`, `SERVO_START_REQ`, `SERVO_ACK`,
`SERVO_FIRST_DATA`, `SERVO_START_TIMEOUT`, `SERVO_END`, `ACTION_START`,
`ACTION_DONE`, `WAIT_FINISH_TIMEOUT`, `RAW_RESUBSCRIBE`, `DEACTIVATE`,
`COMPLETE`, `MOTION_IO_EXECUTOR_STOP`, plus mission control's
`MISSION_NODE_READY`, `PUBLISH_ACTIVE_STAGE`, `STARTUP_RECOVERY_SENT`,
`STARTUP_RECOVERY_DONE`.

The `[RXDIAG]` line also carries the physical backend's own counters
(`servo_tx_start/data/end`, `servo_rx_resp`, `result_calls`), so "the 20 Hz servo
stream started here" and "raw RGB stopped here" are timestamps in one log.

Switches: `diagnostics_enabled`, `diag_report_period_sec` (default 2 s),
`diag_stall_warn_sec` (default 2 s), `diag_event_topic_enabled`.

### 4.6 Stage 4 brought under the same policy

Stage 4's dedicated `stage4_rgb_rx` receiver used to start in `__init__` and run
forever.  It now starts and stops with the ingestion policy, uses
`image_qos_depth`, and reports into the same `rgb` counter.  Its existing
restart-on-stale logic is untouched.

---

## 5. The hardware test that would make this decisive

Run each step for **at least 90 s** past the point where the previous
configuration used to fail.  In every step, watch the bridge's own
`RX raw RGB=` counter and, from another shell,
`ros2 topic echo /mission/diag/event`.

Prefix for all steps: `ros2 launch control_node full_competition.launch.py platform:=real single_stage:=true start_stage:=2`

| # | added arguments | what it isolates |
| --- | --- | --- |
| 1 | *(bridge only, no launch)* | baseline — confirm stability, record the steady `RX raw RGB` rate |
| 2 | `auto_start:=false raw_rgb_subscription:=off raw_depth_subscription:=off tf_listener_enabled:=false` | seven ROS processes and their DDS participants, and nothing else. **If RGB freezes here, it is H4 and none of the rest matters.** |
| 3 | `auto_start:=false raw_rgb_subscription:=off raw_depth_subscription:=off` | + six TF listeners |
| 4 | `auto_start:=false` | + the raw image subscriptions in their new `on_activation` form (so still none, since nothing is active) — this should look like step 3 |
| 5 | `auto_start:=false raw_rgb_subscription:=always raw_depth_subscription:=always` | **the old behaviour: five extra always-on raw readers, nothing activated.** If RGB freezes here and not in step 4, H1 is confirmed and §4.1 is the fix. |
| 6 | `startup_recovery_enabled:=false stage_backend_enabled:=false` | Stage 2 activates and runs its state machine; no motion backend at all |
| 7 | `startup_recovery_enabled:=false real_servo_publish_enabled:=false` | + `*_real_motion_io` helper node and executor thread; still no servo traffic |
| 8 | `startup_recovery_enabled:=false real_result_actions_enabled:=false` | + SERVO_START/ACK and continuous 20 Hz SERVO_DATA; no preset actions |
| 9 | *(no extra arguments)* | + startup RecoveryStand(111) and preset actions — the full competition path |

Notes:

* Steps 6–9 move the robot from step 8 onward. Steps 2–7 do not: with
  `stage_backend_enabled:=false` or `real_servo_publish_enabled:=false` no
  motion command leaves the process. Still hold the emergency stop.
* Stop `stage2_vision_preview` / `stage4_vision_preview` before every step —
  each of them is another raw RGB *and* depth reader and would invalidate the
  reader-count comparison.
* Record for each step: whether the bridge counter froze, at what wall time, and
  which `[RXEVENT]` was the last one before the freeze.

### What to watch

1. **Bridge**: `RX raw RGB=` — the moment it stops incrementing.
2. **`/mission/diag/event`**: the last marker before that moment. This is the
   whole point of the instrumentation.
3. **`[RXDIAG]` on each stage node**: `rgb=nN <Hz> age<s>` — whether the stage
   nodes lose the stream at the same instant as the bridge (shared DDS/transport
   cause) or keep receiving while only the bridge freezes (bridge-side endpoint
   cause). **This one observation splits H1/H2 from H6.**
4. **`[RXHEAL]`**: whether rebuilding the endpoint restores delivery. If it
   does, the failure is an endpoint wedge (H2) and the same recovery must be
   added to the C++ bridge.
5. `ros2 topic hz` on the raw RGB topic, from a shell that is *not* restarted
   between steps, to see whether the publisher's own rate degrades with reader
   count.
6. `top -H` / `mpstat`: per-thread CPU of `stereo_camera`, the bridge, and each
   `stageN_node`, before and after each step.
7. If steps 2–5 point at DDS: set `export RMW_FASTRTPS_...` log verbosity or run
   with `FASTRTPS_DEFAULT_PROFILES_FILE` and look for shared-memory port or
   fragment-reassembly errors during the freeze.

---

## 6. Required follow-up edits that cannot be made from this repository

1. **The robot's `stage2_node.py`** destroys the inherited `rgb_sub` / `depth_sub`
   in `__init__`.  With this branch, activation would *recreate* them.  Replace
   that destroy loop with:

   ```python
   self.disable_raw_image_ingestion('PC perception bridge owns all vision')
   ```

   which switches the policy off permanently instead of just destroying the
   current objects.  (Passing `raw_rgb_subscription:=off raw_depth_subscription:=off`
   at launch has the same effect but applies to every stage node.)

2. **The robot's Stage 1 / Stage 2 files should be committed to this repository.**
   Right now the only copies of the perception offload and the Stage 1 stale
   guard are on the robot and in an uncommitted snapshot directory. Until they
   are in git, any deployment risks reverting them.

3. **The C++ bridge should get the same two things** the Python side now has:
   a monotonic per-stream receive counter with an explicit stall log, and a
   destroy-and-recreate recovery for a subscription that has been silent for a
   few seconds. If §5 step 5 confirms H1 and the bridge still occasionally
   wedges, that recovery is what makes it survivable.

4. **Also worth measuring on the robot, cheaply:** `ros2 topic info -v` on the
   raw RGB topic during a failure (how many readers are matched, and with what
   QoS), and the actual image resolution/encoding — the H2 argument depends on
   the sample exceeding the default shared-memory segment size.

---

## 7. Local verification performed

No robot and no reproduction of the hardware failure — none was possible.

* `pytest test/` in the `cyberdog_dev` container with ROS Galactic sourced:
  **387 passed**, including the Stage 5 tests that construct real nodes through
  the modified `StageNodeBase`.
* New unit tests — 40 in total:
  * `test/test_rx_diagnostics.py` (15): stall/recovery transitions, the rate
    window decaying during silence, never-received streams not reporting stalls;
  * `test/test_ingest_policy.py` (14): every branch of the mode / QoS-depth /
    watchdog-threshold resolution, including the unknown-value fallback;
  * `test/test_real_controller_gates.py` (11): the physical backend's two
    diagnostic gates, driven against a stubbed CyberDog `protocol` package —
    suppressed Servo publishes nothing and does not block for the ACK timeout,
    suppressed result actions complete without touching the service, and an
    undeclared gate parameter falls back to fully enabled.
* `flake8` on every changed file: **no new findings** (the pre-existing
  `E501`/`F401`/`E302` set is unchanged in count and kind).
* `python3 -m compileall` clean over `control_node/`, `launch/`, `test/`.
  All new code is Python 3.6-compatible (no dataclasses, no f-string `=`, no
  walrus) because the robot runs 3.6.
* Live node smoke tests in the container on an **isolated `ROS_DOMAIN_ID=77`**
  (a Gazebo session was running on the default domain and must not be disturbed):
  * `platform:=real` → `[RXCFG] raw_rgb_subscription=on_activation … image_qos_depth=1`
    and **no** subscriptions created before activation;
  * `platform:=sim` → `always` / depth 5, i.e. unchanged;
  * synthetic 10 Hz image publisher → counters, rate, max gap, `[RXSTALL]` at the
    configured threshold, and `[RXHEAL]` rebuilding the subscription, all
    verified end to end;
  * `mission_control_node` + `stage1_node` with `stage_backend_enabled:=false` →
    the full `MISSION_NODE_READY` → `PUBLISH_ACTIVE_STAGE` → `MISSION_MSG` →
    `ACTIVATE_BEGIN` → `BACKEND_CREATE_BEGIN/DONE` marker chain, with the motion
    backend correctly not created.
