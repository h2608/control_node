# -*- coding: utf-8 -*-
"""
Synthetic-scene unit tests for bridge_perception.

渲染器与观测器共享同一套旋转/投影约定，因此这些测试验证的是几何
自洽性与符号约定（左右/横向/航向/跌落方向），不能替代真实传感器
数据上的离线验证（STAGE5_PHYSICAL_REDESIGN_PLAN.md G1 门槛）.
"""

import math

import numpy as np
import pytest

from control_node.bridge_perception import (
    BridgePerceptionConfig,
    CameraIntrinsics,
    bridge_observation,
    depth_image_to_meters,
    rotation_gravity_from_body,
)

WIDTH = 640
HEIGHT = 480
INTR = CameraIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, 1.22)

CAMERA_HEIGHT = 0.35
CAMERA_PITCH = 0.45

DECK_HALF_WIDTH = 0.25
DECK_DROP = 0.15


def render_scene(camera_height=CAMERA_HEIGHT, camera_pitch=CAMERA_PITCH,
                 camera_roll=0.0, deck_center_y=0.0, deck_yaw=0.0,
                 deck_half_width=DECK_HALF_WIDTH, deck_drop=DECK_DROP,
                 deck_s_end=None, deck_forward_slope=0.0,
                 deck_lateral_slope=0.0, noise_std=0.0, seed=0,
                 kerb_side=None, kerb_height=0.05):
    """
    Render analytic depth for a bounded bridge plane above the ground.

    桥面中心线过 (0, deck_center_y)，方向 (cos(deck_yaw), sin(deck_yaw))；
    deck_s_end 非 None 时桥面沿纵向在该弧长处截止（其后是地面）.
    返回 float32 深度图（光学 z 深度，米），无交点处为 0.
    """
    us, vs = np.meshgrid(np.arange(WIDTH, dtype=np.float64),
                         np.arange(HEIGHT, dtype=np.float64))
    # 光学系方向（z 分量恒为 1），参数 t 即光学 z 深度
    d_opt_x = (us - INTR.cx) / INTR.fx
    d_opt_y = (vs - INTR.cy) / INTR.fy
    # 机体系：x 前 = z_o, y 左 = -x_o, z 上 = -y_o
    d_body = np.stack([np.ones_like(d_opt_x), -d_opt_x, -d_opt_y], axis=-1)
    rot = rotation_gravity_from_body(camera_roll, camera_pitch)
    d_grav = d_body @ rot.T

    dz = d_grav[..., 2]
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    down = dz < -1e-9

    # 先与桥面平面 z=0 求交（非下行射线给一个有限的负哨兵值，避免 inf*0=nan）
    deck_denom = (
        dz - deck_forward_slope * d_grav[..., 0]
        - deck_lateral_slope * d_grav[..., 1]
    )
    hits_deck_plane = deck_denom < -1e-9
    t_deck = np.where(
        hits_deck_plane,
        -camera_height / np.where(hits_deck_plane, deck_denom, -1.0),
        -1.0,
    )
    px = t_deck * d_grav[..., 0]
    py = t_deck * d_grav[..., 1]
    cos_y, sin_y = math.cos(deck_yaw), math.sin(deck_yaw)
    lateral = -px * sin_y + (py - deck_center_y) * cos_y
    longitudinal = px * cos_y + (py - deck_center_y) * sin_y
    on_deck = hits_deck_plane & (np.abs(lateral) <= deck_half_width) & (t_deck > 0)
    if deck_s_end is not None:
        on_deck &= longitudinal <= deck_s_end
    depth[on_deck] = t_deck[on_deck]

    # 其余下行射线与地面 z=-deck_drop 求交
    t_ground = np.where(down, (-deck_drop - camera_height) / np.where(down, dz, -1.0), -1.0)
    ground = down & ~on_deck & (t_ground > 0)
    depth[ground] = t_ground[ground]

    # 可选：一侧不是跌落而是抬高的路缘（环形赛道外侧边界就是这样），
    # 该侧因此没有任何低于桥面的证据。
    if kerb_side is not None:
        t_kerb = np.where(
            down, (kerb_height - camera_height) / np.where(down, dz, -1.0), -1.0)
        # 用桥面平面上的横向坐标判定“越过桥沿”：任何会落到桥沿之外的
        # 射线都被路缘挡住，看不到更低的地面。按路缘平面自身的横向坐标
        # 判定会漏掉一条贴着桥沿的窄缝，那条缝里仍是地面像素，
        # 单侧兜底要测的正是“这一侧没有任何低于桥面的证据”。
        beyond = (lateral > deck_half_width if kerb_side == 'left'
                  else lateral < -deck_half_width)
        beyond &= hits_deck_plane
        # 路缘高于桥面，所以对打到它的射线来说它是最近的一层，
        # 必须覆盖桥面与地面，否则边缘外侧仍会漏出低于桥面的地面像素。
        kerb = down & (t_kerb > 0) & beyond
        depth[kerb] = t_kerb[kerb]

    if noise_std > 0.0:
        rng = np.random.RandomState(seed)
        depth[depth > 0] += rng.normal(0.0, noise_std, size=int((depth > 0).sum()))

    return depth.astype(np.float32)


def observe(depth, camera_roll=0.0, camera_pitch=CAMERA_PITCH, **cfg_overrides):
    """用测试相机参数运行观测器."""
    config = BridgePerceptionConfig(**cfg_overrides) if cfg_overrides else None
    return bridge_observation(depth, INTR, camera_roll=camera_roll,
                              camera_pitch=camera_pitch, config=config)


def test_centered_straight_bridge():
    """居中直行：左右边缘各约半桥宽，横向/航向偏差接近零."""
    obs = observe(render_scene())
    assert obs['valid'], obs
    assert obs['d_left_edge'] == pytest.approx(DECK_HALF_WIDTH, abs=0.04)
    assert obs['d_right_edge'] == pytest.approx(DECK_HALF_WIDTH, abs=0.04)
    assert abs(obs['lateral_offset']) < 0.03
    assert abs(obs['heading_error']) < 0.03
    assert abs(obs['surface_roll']) < 0.03
    assert abs(obs['surface_pitch']) < 0.03
    assert obs['camera_height'] == pytest.approx(CAMERA_HEIGHT, abs=0.03)


def test_lateral_offset_sign_and_magnitude():
    """桥中心线在机体右侧 0.10 m：机体偏左，lateral_offset 应为约 -0.10."""
    obs = observe(render_scene(deck_center_y=-0.10))
    assert obs['valid'], obs
    assert obs['d_left_edge'] == pytest.approx(DECK_HALF_WIDTH - 0.10, abs=0.04)
    assert obs['d_right_edge'] == pytest.approx(DECK_HALF_WIDTH + 0.10, abs=0.04)
    assert obs['lateral_offset'] == pytest.approx(-0.10, abs=0.04)


def test_heading_error_sign():
    """桥轴向左偏转 0.15 rad：heading_error 应为约 +0.15（左转修正）."""
    obs = observe(render_scene(deck_yaw=0.15))
    assert obs['valid'], obs
    assert obs['heading_error'] == pytest.approx(0.15, abs=0.05)


def test_forward_dropoff_detection():
    """桥面在前方 1.0 m 截止：应检出跌落且 deck_end_x 接近 1.0."""
    obs = observe(render_scene(deck_s_end=1.0))
    assert obs['dropoff_detected'] is True
    assert obs['deck_end_x'] == pytest.approx(1.0, abs=0.15)


def test_no_dropoff_on_continuous_deck():
    """桥面覆盖整个量程时不应报告跌落."""
    obs = observe(render_scene())
    assert obs['dropoff_detected'] is False


def test_roll_compensation():
    """相机带横滚时，把同一横滚角传给观测器应恢复居中观测."""
    depth = render_scene(camera_roll=0.20)
    obs = observe(depth, camera_roll=0.20)
    assert obs['valid'], obs
    assert abs(obs['lateral_offset']) < 0.04
    assert abs(obs['surface_roll']) < 0.04


def test_noise_robustness():
    """1 cm 高斯深度噪声下仍应得到可用的居中估计."""
    obs = observe(render_scene(noise_std=0.01))
    assert obs['valid'], obs
    assert abs(obs['lateral_offset']) < 0.05
    assert abs(obs['heading_error']) < 0.06


def test_invalid_inputs():
    """空图/零图/None 都必须显式 invalid，而不是抛异常或给出伪观测."""
    assert bridge_observation(None, INTR)['valid'] is False
    zeros = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    assert bridge_observation(zeros, INTR)['valid'] is False
    nans = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    assert bridge_observation(nans, INTR)['valid'] is False


def test_yaw_does_not_create_lateral_offset():
    """桥中心线过机器人原点时，纯偏航不能被误报成横向偏移."""
    obs = observe(render_scene(deck_yaw=0.15, deck_center_y=0.0))
    assert obs['valid'], obs
    assert abs(obs['lateral_offset']) < 0.03
    assert obs['heading_error'] == pytest.approx(0.15, abs=0.05)


def test_shifted_continuous_bridge_has_no_false_forward_dropoff():
    """侧向偏移的无限长桥不能把侧面地面误认成前方跌落."""
    obs = observe(render_scene(deck_center_y=0.20))
    assert obs['valid'], obs
    assert obs['dropoff_detected'] is False
    assert obs['d_forward_dropoff'] is None


def test_robot_axis_outside_deck_is_invalid():
    """机器人前向轴已在 50 cm 桥外时不得输出伪有效控制量."""
    obs = observe(render_scene(deck_center_y=0.30))
    assert obs['valid'] is False
    assert obs['reason'] in {
        'robot_axis_outside_deck',
        'implausible_deck_width',
        'inconsistent_edge_widths',
    }


def test_depth_encoding_and_observer_metadata_contract():
    """16UC1 毫米深度正确换算，并保留帧序号/时间戳/前向跌落字段."""
    depth_mm = np.array([[0, 1250]], dtype=np.uint16)
    converted = depth_image_to_meters(depth_mm, '16UC1')
    assert converted.dtype == np.float64
    assert converted[0, 1] == pytest.approx(1.25)

    obs = bridge_observation(
        render_scene(deck_s_end=1.0),
        INTR,
        camera_pitch=CAMERA_PITCH,
        frame_seq=42,
        stamp_s=12.5,
    )
    assert obs['frame_seq'] == 42
    assert obs['stamp_s'] == pytest.approx(12.5)
    assert obs['d_forward_dropoff'] == pytest.approx(1.0, abs=0.15)


def test_yaw_offset_cross_matrix_preserves_sign_and_magnitude():
    """联合偏航/横移不能因 ROI 裁切输出反号横移."""
    for pitch in (0.0, CAMERA_PITCH):
        for yaw, center in ((0.15, 0.15), (-0.15, -0.15),
                            (0.30, 0.10), (-0.30, -0.10),
                            (0.30, 0.15), (-0.30, -0.15)):
            obs = observe(
                render_scene(camera_pitch=pitch, deck_yaw=yaw,
                             deck_center_y=center),
                camera_pitch=pitch,
            )
            if not obs['valid']:
                # 极端组合在视野裁切时必须 fail closed，不能给反号控制量.
                assert obs['reason'].startswith('edge_rows<')
                continue
            assert obs['lateral_offset'] == pytest.approx(center, abs=0.035)
            assert obs['heading_error'] == pytest.approx(yaw, abs=0.035)


@pytest.mark.parametrize('lateral_slope', [-0.20, 0.20])
def test_trapezoid_cross_slope_does_not_create_pose_error(lateral_slope):
    """10/20 cm 横坡只改变 surface_roll，不得伪造横移或航向误差."""
    depth = render_scene(
        camera_pitch=0.0,
        deck_lateral_slope=lateral_slope,
    )
    obs = observe(depth, camera_pitch=0.0)
    assert obs['valid'], obs
    assert obs['surface_roll'] == pytest.approx(
        math.atan(lateral_slope), abs=0.03)
    assert abs(obs['lateral_offset']) < 0.025
    assert abs(obs['heading_error']) < 0.025


def test_body_reference_extrinsic_removes_camera_offset():
    """桥轴过 body 原点时，D435 平移不得变成机体横向误差."""
    mount_x = 0.271994
    mount_y = 0.025
    yaw = 0.30
    center_at_camera = math.tan(yaw) * mount_x - mount_y
    depth = render_scene(deck_yaw=yaw, deck_center_y=center_at_camera)
    obs = bridge_observation(
        depth,
        INTR,
        camera_pitch=CAMERA_PITCH,
        control_point_x=-mount_x,
        control_point_y=-mount_y,
    )
    assert obs['valid'], obs
    assert abs(obs['lateral_offset']) < 0.03
    assert abs(obs['camera_lateral_offset']) > 0.04


def test_near_dropoff_outside_fov_fails_closed():
    """零俯角看不见 50 cm 近端时必须 invalid，不能伪造距离."""
    near = observe(
        render_scene(camera_pitch=0.0, deck_s_end=0.50),
        camera_pitch=0.0,
    )
    assert near['valid'] is False
    assert near['d_forward_dropoff'] is None

    visible = observe(
        render_scene(camera_pitch=0.0, deck_s_end=0.90),
        camera_pitch=0.0,
    )
    assert visible['valid'], visible
    assert visible['dropoff_detected'] is True
    assert visible['d_forward_dropoff'] == pytest.approx(0.90, abs=0.12)


def test_empty_depth_encoding_is_rejected():
    """未知/空 encoding 不能被静默当作米制 float depth."""
    with pytest.raises(ValueError):
        depth_image_to_meters(np.ones((2, 2), dtype=np.float32), '')


def test_low_support_yaw_offset_fails_closed():
    """A heavily clipped yaw/offset fit must be invalid, not under-corrected."""
    obs = observe(render_scene(deck_yaw=0.30, deck_center_y=0.15))
    assert obs['valid'] is False
    assert obs['reason'].startswith('edge_rows<')


def test_forward_dropoff_uses_body_reference_extrinsic():
    """Forward dropoff exposes camera distance but controls from body origin."""
    mount_x = 0.271994
    obs = bridge_observation(
        render_scene(deck_s_end=1.0),
        INTR,
        camera_pitch=CAMERA_PITCH,
        control_point_x=-mount_x,
        control_point_y=-0.025,
    )
    assert obs['valid'], obs
    assert obs['d_forward_dropoff_camera'] == pytest.approx(1.0, abs=0.15)
    assert obs['d_forward_dropoff'] == pytest.approx(
        obs['d_forward_dropoff_camera'] + mount_x, abs=0.03)


def test_invalid_observation_has_fixed_null_schema():
    """Invalid module outputs retain every consumer-facing geometry key."""
    obs = bridge_observation(None, INTR, frame_seq=5, stamp_s=2.0)
    expected_null = {
        'd_left_edge', 'd_right_edge', 'lateral_offset',
        'camera_lateral_offset', 'heading_error', 'surface_roll',
        'surface_pitch', 'camera_height', 'deck_end_x',
        'deck_end_camera_x', 'd_forward_dropoff_camera',
        'd_forward_dropoff',
    }
    assert obs['valid'] is False
    assert expected_null.issubset(obs)
    assert all(obs[key] is None for key in expected_null)
    assert obs['quality']['n_points'] == 0


# ---------------------------------------------------------------------------
# 单侧边缘兜底：环形赛道每条边只有内侧是跌落，外侧是抬高的路缘。
# ---------------------------------------------------------------------------

SINGLE_SIDED = {
    'single_sided_edges_enabled': True,
    'declared_deck_width': 2 * DECK_HALF_WIDTH,
}


def test_kerb_on_one_side_defeats_the_two_sided_extractor():
    """The baseline behaviour this fallback exists for, stated as a test.

    A raised kerb produces above-plane points, so the two-sided extractor never
    gets its below-plane evidence and the whole segment goes unobserved.
    """
    depth = render_scene(deck_center_y=0.10, kerb_side='left')
    result = observe(depth)
    assert not result['valid']
    assert result['reason'].startswith('edge_rows<')


def test_single_sided_recovers_the_offset_past_a_left_kerb():
    """One visible drop-off plus the declared width is enough to centre on."""
    depth = render_scene(deck_center_y=0.10, kerb_side='left')
    result = observe(depth, **SINGLE_SIDED)
    assert result['valid']
    assert result['reason'] == 'ok_single_sided'
    assert result['lateral_offset'] == pytest.approx(0.10, abs=0.03)


def test_single_sided_recovers_the_offset_past_a_right_kerb():
    """The mirror case must give the mirrored sign, not a mirrored bug."""
    depth = render_scene(deck_center_y=-0.10, kerb_side='right')
    result = observe(depth, **SINGLE_SIDED)
    assert result['valid']
    assert result['lateral_offset'] == pytest.approx(-0.10, abs=0.03)


def test_single_sided_still_reports_heading():
    """Heading comes from the one edge's direction and must survive the mode."""
    depth = render_scene(deck_yaw=0.08, kerb_side='left')
    result = observe(depth, **SINGLE_SIDED)
    assert result['valid']
    assert result['heading_error'] == pytest.approx(0.08, abs=0.03)


def test_single_sided_is_off_by_default():
    """It is a degraded mode; no profile may inherit it without asking."""
    depth = render_scene(deck_center_y=0.10, kerb_side='left')
    assert not observe(depth)['valid']


def test_single_sided_needs_a_declared_width():
    """Enabling the mode without a width is a configuration error, not a guess."""
    depth = render_scene(deck_center_y=0.10, kerb_side='left')
    result = observe(depth, single_sided_edges_enabled=True,
                     declared_deck_width=0.0)
    assert not result['valid']


def test_two_sided_fit_still_wins_when_both_edges_are_visible():
    """The fallback must never displace the self-checking measurement."""
    depth = render_scene(deck_center_y=0.10)
    result = observe(depth, **SINGLE_SIDED)
    assert result['valid']
    assert result['reason'] == 'ok'


def test_a_wrong_declared_width_biases_the_centre_by_half_the_error():
    """Document the cost: there is no second edge to catch a bad width."""
    depth = render_scene(deck_center_y=0.0, kerb_side='left')
    result = observe(depth, single_sided_edges_enabled=True,
                     declared_deck_width=2 * DECK_HALF_WIDTH + 0.10)
    assert result['valid']
    # A left kerb leaves only the right edge visible, and the centre is placed
    # half a declared width inboard of it.  Over-declaring the width by 0.10 m
    # therefore pushes the inferred centre 0.05 m away from that edge — and a
    # perfectly centred robot is told it is 0.05 m off.
    assert result['lateral_offset'] == pytest.approx(0.05, abs=0.03)
