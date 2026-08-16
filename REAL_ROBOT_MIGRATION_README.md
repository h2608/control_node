# CyberDog 仿真 → 真机迁移版说明

本版本在原 `refactor(2)(3)` 基础上增加了双后端控制层，目标是保持 Stage1~Stage6 的视觉与状态机主体不变，把 Gazebo/LCM 与真机 Motion API 的差异集中到底层。

## 1. 新控制结构

```text
Stage1 ~ Stage6
    ↓
robot adapter
    ├── platform=sim  → Robot_Ctrl + LCM
    └── platform=real → MotionServoCmd / MotionServoResponse / MotionResultCmd
```

新增目录：

```text
control_node/robot_interface/
    __init__.py
    factory.py
    motion_ids.py
    sim_controller.py
    real_controller.py
```

真机连续运动默认使用 `motion_id=303`（慢速行走）。离散动作默认映射：

- RecoveryStand → 111
- Emergency Stop → 0
- Lie down → 101
- Left jump → 134
- Right jump → 135
- Forward jump → 132

所有 ID 都做成 ROS 参数，可以覆盖。

## 2. 真机 Servo 生命周期

`RealRobotControlAdapter` 内部维护 Servo 生命周期：

```text
SERVO_START (cmd_type=0)
    ↓
持续 SERVO_DATA (cmd_type=1, 默认 20 Hz)
    ↓
SERVO_END (cmd_type=2)
```

Stage 状态机只更新目标 `vx/vy/wz/rpy/pos/step_height`，持续发送由 adapter 的独立 ROS helper node 完成。

`cmd_source` 默认保持你已经在真机测试过的 `0`。

## 3. Wait_finish 兼容

真机离散动作通过 `MotionResultCmd` service 执行。Adapter 内部有独立 executor，因此旧代码中 Stage2/Stage4 的 `Wait_finish()` 暂时仍可使用，不会依赖 LCM 返回。

Stage5 原有非阻塞 action poll 也保留。对于真机，adapter 会把 service 开始/完成转换成 Stage5 能理解的兼容状态；真正的 STOP 在真机上使用 `SERVO_END`，不会再把 `(12,0)` 当成 RecoveryStand。

## 4. Stage4 自定义直立动作

原 Stage4 还有自定义底层动作：

```text
mode=64 gait=4
mode=3  gait=0
```

它们没有对应的公开 Motion ID，而且当前目标不是继续调自定义双足/直立动作。因此：

- Gazebo：保留原流程；
- 真机：RecoveryStand(111) 成功后直接恢复四足运动，跳过上述自定义动作。

## 5. Stage6

Stage6 暂时保留旧 `robot_control_cmd_lcmt` 形状作为“状态机内部命令对象”，但 `p6_send_cmd()` 最终进入 `RealRobotControlAdapter.Send_cmd()`，不会在真机发送 LCM 控制命令。

当前兼容映射：

- mode 11 + gait 3/27/1/0 → Servo locomotion（默认 303，可参数覆盖）
- mode 7 + gait 1 → motion_id 101
- mode 16 + gait 0 → 134
- mode 16 + gait 3 → 135
- mode 16 + gait 1 → 132
- mode 12 + gait 0 → 111（非零 body pose 时改为 Servo pose hold）
- mode 0 → motion_id 0

后续真机逐段跑通后，可以再把 Stage6 的旧 `mode/gait` 构造彻底删掉。

## 6. 仿真专用接口

`cyberdog_msg/YamlParam` 和 `ApplyForce` 已隔离：

- `platform=sim` 时正常创建 publisher；
- `platform=real` 时不要求 `cyberdog_msg` 包存在，调用会直接忽略并报警。

## 7. 真机启动参数

`full_competition.launch.py` 新增：

```text
platform
rgb_topic
depth_topic
global_frame
base_frame
real_motion_servo_cmd_topic
real_motion_servo_response_topic
real_motion_result_service
voice_dir
```

你的 Servo 测试已经验证命令 topic：

```text
/mi_desktop_48_b0_2d_7b_00_e2/motion_servo_cmd
```

因此 launch 中已把同一 namespace 作为三个 Motion 接口的默认值。请先用：

```bash
ros2 topic list -t | grep motion
ros2 service list | grep motion
```

确认 response/service 也使用相同 namespace。

真机完整启动示例：

```bash
ros2 launch control_node full_competition.launch.py \
  platform:=real \
  rgb_topic:=<真机前向RGB话题> \
  depth_topic:=<真机Depth Image话题> \
  global_frame:=<真机全局frame> \
  base_frame:=<真机机体frame>
```

当 `platform:=real` 时：

- `use_sim_time` 自动为 false；
- Stage5 自动加载 `stage5_physical.yaml`，不会误加载 Gazebo 参数；
- OpenCV debug 默认关闭。

只启动一个赛段时，设置 `single_stage:=true`，并通过 `start_stage` 选择
赛段编号。例如，只启动真机第四赛段：

```bash
ros2 launch control_node full_competition.launch.py \
  platform:=real \
  single_stage:=true \
  start_stage:=4
```

该模式只创建选中的赛段节点；赛段完成后任务控制器发布空闲赛段 `0`，
不会继续启动后续赛段。省略 `single_stage:=true` 时保持原行为：从
`start_stage` 开始并继续运行后面的赛段。

### 赛段内部调试入口（stageN_entry）

`start_stage` 只能选到赛段级；要只跑赛段里的某一段，再加 `stageN_entry`。
例如只跑第五赛段的坡道段：

```bash
ros2 launch control_node full_competition.launch.py \
  single_stage:=true \
  start_stage:=5 \
  stage5_entry:=ramp
```

六个赛段各有一张入口表（`control_node/stage_entry.py` + 各赛段的
`pN_entry_table()`）。入口名一律小写，也可以直接写内部状态名：

| 赛段 | 入口名 |
| --- | --- |
| 1 | `start` `cruise` `brake` `align` `restore` `forward` `turn` `ball` `shift` |
| 2 | `start` `track1` `track1_exit` `track1_turn` `track1_shift` `track2` `track2_turn` `track2_forward` `track3` `scan` `scan_hit` `turn_back` `final` `final_forward` `final_turn` `final_align` `ball_align` `ball_hit` `ball_shift` |
| 3 | `start` `s_curve` `align` |
| 4 | `start` `search` `bar_center` `bar` `bar_target` `bar_hit` `bar_back` `bar_yellow` `obstacle_center` `obstacle` `obstacle_route` `target` `target_hit` `upright` `post_hit` `post_hit_obstacle` `final` `final_yellow` `final_align` |
| 5 | `start` `recovery` `align` `step_up` `ramp` `ramp_exit` `corner_1` `slope_body` `straight_1` `corner_2` `straight_2` `corner_3` `straight_3` `reset_body` `corner_4` `descent` `final` `final_jump` |
| 6 | `start` `north` `north_align` `turn` `east` `clear_ball` `crab` `west` `west_march` `west_align` `exit` `push` `finish` |

第五赛段的入口名和 `route_model.py` 的段名是同一套词（`ramp` = `up_slope`，
`straight_1/2/3`、`corner_1..4`、`descent` = `right_descent`、`final` =
`final_zone`），跑到一半的证据日志和调试入口对得上。

几点必须注意：

1. **入口只决定状态机从哪个状态开始，不会把狗搬过去。** 机器人本体必须已经
   被摆到该段的起点和朝向；否则整段会带着一个固定的初始误差走完。节点每次
   用非默认入口启动都会以 WARN 打印这句话和该入口的额外前提。
2. 入口名写错不会让节点起不来：回退到该赛段正常起点，并把整张入口表打进
   日志（ERROR 级）。
3. 有前提的入口要一起给参数：
   - `stage2_entry:=ball_*` 需要 `stage2_ball_return:=track1|track2|track3`，
     否则撞球子链会回到自己；不给时退回赛道 1 并告警。
   - `stage4_entry:=obstacle_route` / `post_hit_obstacle` 需要
     `stage4_dashed_side:=left|right`。
4. 单独跑一个赛段节点（不经 launch）时用节点参数 `entry_point`：

```bash
ros2 run control_node stage5_node --ros-args -p entry_point:=ramp
```

   各赛段原有的 `p1_initial_state` / `second_stage_initial_state` /
   `p3_initial_state` / `p4_initial_state` / `p5_initial_state` /
   `p6_initial_state` 仍然可用（写状态名或入口名都行），`entry_point`
   优先级更高。

也可以给单独赛段加载：

```bash
--ros-args --params-file config/real_robot.yaml
```

注意 `real_robot.yaml` 只确定 Motion 控制接口；前向 RGB、Depth、TF 仍必须按真机实际 topic/frame 填写。

## 8. 相机说明

当前上传的鱼眼资料确认左右鱼眼为 500×400、mono8，主要用于侧向环境/结构感知。现有比赛黄线、蓝/橙球等颜色检测仍应使用前向 RGB，因此本轮没有把 Stage1~6 的 HSV 算法切到左右鱼眼。

## 9. 本轮没有做的事情

1. 没有重新标定真机 HSV/ROI/像素阈值；
2. 没有重新标定所有 timed motion 的速度和持续时间；
3. 没有假定前向 RGB/Depth 真机 topic；
4. 没有假定 RGB-Depth 已对齐；
5. 没有继续实现 Stage4 自定义双足/直立底层动作；
6. 没有把 Stage6 的旧命令对象一次性重写，先通过兼容层迁移。

## 10. 已完成的离线检查

- 所有 Python 文件通过 `compileall`；
- 对 `RealRobotControlAdapter` 使用假 ROS/Protocol 对象做了逻辑测试：
  - Servo START；
  - gait → motion_id；
  - Right jump → 135；
  - service 完成 → Wait_finish；
  - Stage5 stop compatibility barrier。

当前环境没有真机 ROS 2 `protocol` 包，因此仍需要在 NX/真机上完成接口级运行验证。

## 第四赛段只读视觉调试

新增 `stage4_vision_preview`。它只订阅 RGB/Depth，不创建控制器，也不发送运动命令；
网页同时显示综合检测框、八种检测掩膜和深度图。

在源码目录中可以不编译，直接用 Python 启动：

```bash
cd <control_node 项目目录>
python3 control_node/stage4_vision_preview.py \
  --platform real \
  --dog-ns mi_desktop_48_b0_2d_7b_00_e2 \
  --port 8084 \
  --ros-args --params-file config/real_robot.yaml
```

如果已经用 colcon 安装，也可以使用 ROS 2 入口：

```bash
ros2 run control_node stage4_vision_preview \
  --platform real \
  --dog-ns mi_desktop_48_b0_2d_7b_00_e2 \
  --port 8084 \
  --ros-args --params-file config/real_robot.yaml
```

从调试电脑建立 SSH 转发：

```bash
ssh -L 8084:127.0.0.1:8084 <user>@<robot-ip>
```

然后打开 `http://127.0.0.1:8084/`。可通过 `--rgb-topic`、`--depth-topic`
覆盖相机话题；第四赛段的 `bar.*`、`obstacle.*`、`yellow.*`、
`final_yellow.*`、`blue_ball.*`、`white_ball.*` 和 `cola.*` 参数也可用 ROS 参数覆盖。


## Physical camera topics

The real backend now selects the verified camera topics automatically when
`platform:=real`:

```text
Front RGB : /mi_desktop_48_b0_2d_7b_00_e2/image_rgb
Depth     : /mi_desktop_48_b0_2d_7b_00_e2/camera/depth/image_rect_raw
AI camera : /mi_desktop_48_b0_2d_7b_00_e2/image
Left fish : /mi_desktop_48_b0_2d_7b_00_e2/image_left
Right fish: /mi_desktop_48_b0_2d_7b_00_e2/image_right
```

Front RGB and depth continue to use the same processing/alignment assumptions
as the Gazebo code; this migration does not add registration or remapping.

Stage 2 already owns the left/right fisheye subscriptions and detection logic.
Only their topic names switch between simulation and the physical robot:

```text
sim : /image_left, /image_right
real: /mi_desktop_48_b0_2d_7b_00_e2/image_left,
      /mi_desktop_48_b0_2d_7b_00_e2/image_right
```

The AI camera is deliberately only exposed as the `ai_camera_topic` parameter
for now. No stage subscribes to it yet, so it consumes no extra image bandwidth
until a later stage explicitly needs it.

With the full launch file, the default camera arguments are `auto` and follow
`platform` automatically:

```bash
ros2 launch control_node full_competition.launch.py platform:=real
```

A topic can still be overridden explicitly, for example:

```bash
ros2 launch control_node full_competition.launch.py \
  platform:=real \
  rgb_topic:=/some/other/rgb/topic
```


## 真机日常操作速查

原 `使用文档.md` 已删除（它的解压安装流程和“`start_stage` 不能只跑一个赛段”
的说法都已经不成立），其中仍然有效的部分并入本节。接口层的原理见上面第 1、
2、7 节，话题清单见“Physical camera topics”。

### 编译与环境

机器狗上的 ROS 2 装在 `/opt/ros2/`，与仿真容器的 `/opt/ros/` 不是一个路径：

```bash
source /opt/ros2/galactic/setup.bash
source /opt/ros2/cyberdog/setup.bash

cd ~/cyberdog_ws
colcon build --packages-select control_node --symlink-install
source install/setup.bash
```

每次新开终端都要重新 source 这三个 setup.bash。

### 运行

完整比赛（先 RecoveryStand(111) 起立，再 Stage1 → … → Stage6）：

```bash
ros2 launch control_node full_competition.launch.py platform:=real start_stage:=1
```

`start_stage:=N` 是“从第 N 赛段开始并继续跑完后面的赛段”；只跑一个赛段要配合
`single_stage:=true`（见第 7 节）。起立可以用 `startup_recovery_enabled:=false` 关掉。

### 启动前自检相机

```bash
ros2 topic hz /mi_desktop_48_b0_2d_7b_00_e2/image_rgb
ros2 topic hz /mi_desktop_48_b0_2d_7b_00_e2/camera/depth/image_rect_raw
```

### 关键日志

连续运动正常接管时，`[REAL_CTRL]` 会依次打印这三条：

```text
[REAL_CTRL] SERVO_START motion_id=303 ...
[REAL_CTRL] SERVO_READY motion_id=303 robot_ack=True ...
[REAL_CTRL] FIRST SERVO_DATA motion_id=303
```

走不完就说明 303 还没真正进入控制，常见两种失败：

```text
[REAL_CTRL] SERVO_START ACK timeout motion_id=... after ...s
[REAL_CTRL] SERVO_START rejected/busy motion_id=...
```

相机断流的两条信号：

```text
[P4_RGB_STALE_STOP] ... dedicated RGB age=... >= 0.800s ...; cmd=(0,0,0)
[<状态名>] rgb stream stale: age=... > 2.00s, enter P5_SENSOR_FAULT_HOLD
```

Stage4 会把命令压成零速并尝试重建 RGB 接收线程（`p4_rgb_stale_stop_s`）；
Stage5 直接进 `P5_SENSOR_FAULT_HOLD` 停住等人工恢复
（`p5_sensor_max_frame_age_s`）。两者都优先查机器狗 RGB 相机。

### 停止

`Ctrl+C`。赛段节点在退出路径上先发 STOP——真机上就是 `SERVO_END`——
然后 `Ctrl.quit()` 再补一次 `SERVO_END`。
