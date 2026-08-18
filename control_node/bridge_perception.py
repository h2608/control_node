#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-based bridge geometry observer for the Stage 5 redesign.

实体赛道的黄线贴在地面而不是桥上（第八场培训 21:17–21:40 官方纠正），
桥缘控制必须来自可观测的桥体几何.本模块从单帧深度图中估计：

- 桥面平面（重力对齐坐标系里的 RANSAC 平面拟合）；
- 左右跌落边缘的横向距离、机体相对桥中心线的横向偏差与航向偏差；
- 前方桥面尽头 / 跌落（段尾、转角、最后 50 cm 起跳区的观测基础）；
- 桥面相对水平面的横滚/俯仰.

坐标约定：
- 相机光学系：x 右、y 下、z 前（ROS optical convention）.
- 重力对齐系：x 前、y 左、z 上（REP-103），原点在相机光心.
- ``camera_pitch`` 为正表示相机俯视（绕 y 轴，x 前/y 左/z 上系）；
  ``camera_roll`` 为正表示绕前向轴的右手旋转.观测点变换为
  ``p_gravity = Ry(pitch) @ Rx(roll) @ p_body``.

所有输入输出均为米/弧度.深度图为 float 单位米（0/NaN/inf 视为无效）.
本模块只做几何，不做话题、参数或时序处理；帧新鲜度、IMU 对接、
路线模型判据在节点层完成.仿真 D435 可直接喂入；真实传感器接入前
必须完成 STAGE5_PHYSICAL_REDESIGN_PLAN.md 的 G0/G1 门槛.
"""

import math

import numpy as np


class CameraIntrinsics:
    """针孔相机内参（像素焦距与主点）."""

    __slots__ = ('fx', 'fy', 'cx', 'cy')

    def __init__(self, fx, fy, cx, cy):
        """按像素单位保存 fx/fy/cx/cy."""
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    @classmethod
    def from_horizontal_fov(cls, width, height, horizontal_fov_rad):
        """按水平视场角构造（如仿真相机 640x480 / 1.22 rad）."""
        fx = (width / 2.0) / math.tan(horizontal_fov_rad / 2.0)
        # 方形像素假设：fy = fx.真实相机必须用标定内参替换.
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0)


class BridgePerceptionConfig:
    """观测器配置；默认值面向仿真 640x480 深度，实体使用前必须重标."""

    def __init__(self, **overrides):
        """构造默认配置，可用关键字参数覆盖任意字段."""
        # 图像 ROI（比例）与抽样步长
        self.roi_x_min = 0.05
        self.roi_x_max = 0.95
        self.roi_y_min = 0.30
        self.roi_y_max = 1.00
        self.pixel_stride = 4

        # 有效深度范围（米）
        self.min_valid_depth = 0.12
        self.max_valid_depth = 4.0

        # 桥面平面拟合：机体正前方近处窄带作为种子点
        self.seed_lateral_half_width = 0.15
        self.seed_forward_min = 0.25
        self.seed_forward_max = 1.60
        self.ransac_iterations = 60
        self.ransac_inlier_tol = 0.02
        self.ransac_min_inliers = 60
        self.plane_normal_min_z = 0.85
        self.random_seed = 0

        # 边缘提取：沿前向分带，每带在横向找包含机体正前的连续桥面段
        self.band_forward_min = 0.30
        self.band_forward_max = 1.50
        self.band_width = 0.15
        self.min_band_points = 10
        self.min_bands_for_estimate = 6
        self.edge_lateral_gap = 0.08
        self.min_deck_width = 0.30
        self.max_deck_width = 0.75
        self.min_axis_edge_clearance = 0.01
        self.band_width_consistency_ratio = 0.80

        # 桥面/跌落分类阈值
        self.on_deck_height_tol = 0.03
        self.drop_below_plane = 0.06

        # 单侧边缘兜底：只有一侧是跌落、另一侧是矮墙时的降级估计。
        # 两侧跌落的提取器在这种赛道上永远拿不到证据（实测 race.world
        # 环形赛道底边 27 s 里 26 s 报 edge_rows<6），但只要赛道宽度是
        # 已声明的课程属性，单边 + 宽度就足以定出中心线。
        # 代价必须说清楚：中心线精度直接等于 declared_deck_width 的精度，
        # 声明宽度错 2 cm，横向就偏 1 cm，而且没有第二个边缘能发现这件事。
        self.single_sided_edges_enabled = False
        self.declared_deck_width = 0.0

        # 前向跌落检测：中央走廊
        self.dropoff_lateral_half_width = 0.10
        self.dropoff_forward_gap = 0.20
        # 透视射线在桥面后落到低地时，前向坐标间隔可显著大于真实落差边界.
        self.dropoff_evidence_max_gap = 0.80
        self.dropoff_min_below_points = 3
        # 边缘拟合失败时，是否退回“沿机体前向轴”取走廊。
        self.dropoff_body_axis_fallback = False

        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f'unknown config field: {key}')
            setattr(self, key, value)


def depth_image_to_meters(depth, encoding):
    """Normalize ROS depth encodings to float64 metres without mutating input."""
    array = np.asarray(depth)
    if array.ndim != 2:
        raise ValueError('depth image must be two-dimensional')
    encoding_upper = str(encoding or '').upper()
    if '16U' in encoding_upper or encoding_upper in ('MONO16', 'TYPE_16UC1'):
        return array.astype(np.float64) * 0.001
    if '32F' in encoding_upper or encoding_upper == 'TYPE_32FC1':
        return array.astype(np.float64)
    raise ValueError(f'unsupported depth encoding: {encoding}')


def rotation_gravity_from_body(roll, pitch):
    """机体系到重力对齐系的旋转矩阵：Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr, -sr],
                   [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]])
    return ry @ rx


def depth_to_gravity_grid(
    depth_m, intrinsics, camera_roll, camera_pitch, config
):
    """Back-project the sampled ROI into an organized gravity-frame point grid."""
    h, w = depth_m.shape[:2]
    x1 = int(w * config.roi_x_min)
    x2 = int(w * config.roi_x_max)
    y1 = int(h * config.roi_y_min)
    y2 = int(h * config.roi_y_max)
    stride = max(1, int(config.pixel_stride))

    us, vs = np.meshgrid(
        np.arange(x1, x2, stride, dtype=np.float64),
        np.arange(y1, y2, stride, dtype=np.float64),
    )
    z = depth_m[y1:y2:stride, x1:x2:stride].astype(np.float64)
    grid = np.full(z.shape + (3,), np.nan, dtype=np.float64)

    valid = np.isfinite(z)
    z_safe = np.where(valid, z, 0.0)
    valid &= (z_safe >= config.min_valid_depth) & (z_safe <= config.max_valid_depth)
    if not np.any(valid):
        return grid

    z_valid = z[valid]
    x_o = (us[valid] - intrinsics.cx) / intrinsics.fx * z_valid
    y_o = (vs[valid] - intrinsics.cy) / intrinsics.fy * z_valid
    pts_body = np.stack([z_valid, -x_o, -y_o], axis=1)
    rot = rotation_gravity_from_body(camera_roll, camera_pitch)
    grid[valid] = pts_body @ rot.T
    return grid


def depth_to_gravity_points(depth_m, intrinsics, camera_roll, camera_pitch, config):
    """Back-project the sampled ROI to gravity-frame points (N x 3)."""
    grid = depth_to_gravity_grid(
        depth_m, intrinsics, camera_roll, camera_pitch, config)
    return grid[np.isfinite(grid[..., 0])]


def fit_deck_plane(points, config):
    """在种子区内 RANSAC 拟合桥面平面 z = a*x + b*y + c；失败返回 None."""
    seed_mask = (
        (np.abs(points[:, 1]) <= config.seed_lateral_half_width)
        & (points[:, 0] >= config.seed_forward_min)
        & (points[:, 0] <= config.seed_forward_max)
    )
    seed = points[seed_mask]
    if seed.shape[0] < max(3, config.ransac_min_inliers):
        return None

    rng = np.random.RandomState(config.random_seed)
    best_inliers = None
    best_count = 0
    best_level = -float('inf')
    best_rms = float('inf')

    for _ in range(config.ransac_iterations):
        idx = rng.choice(seed.shape[0], 3, replace=False)
        p0, p1, p2 = seed[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if abs(normal[2]) < config.plane_normal_min_z:
            continue
        dist = np.abs((seed - p0) @ normal)
        inlier_mask = dist <= config.ransac_inlier_tol
        count = int(np.count_nonzero(inlier_mask))
        if count < config.ransac_min_inliers:
            continue
        inlier_points = seed[inlier_mask]
        candidate_level = float(np.median(inlier_points[:, 2]))
        candidate_rms = float(np.sqrt(np.mean(dist[inlier_mask] ** 2)))
        height_margin = config.ransac_inlier_tol
        if (
            candidate_level > best_level + height_margin
            or (
                abs(candidate_level - best_level) <= height_margin
                and candidate_rms < best_rms
            )
        ):
            best_count = count
            best_level = candidate_level
            best_rms = candidate_rms
            best_inliers = inlier_mask

    if best_inliers is None or best_count < config.ransac_min_inliers:
        return None

    # 最小二乘精修：z = a*x + b*y + c
    inl = seed[best_inliers]
    design = np.column_stack([inl[:, 0], inl[:, 1], np.ones(inl.shape[0])])
    coef, _, _, _ = np.linalg.lstsq(design, inl[:, 2], rcond=None)
    a, b, c = (float(coef[0]), float(coef[1]), float(coef[2]))
    residual = design @ coef - inl[:, 2]
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return {
        'a': a, 'b': b, 'c': c,
        'inlier_count': best_count,
        'rms': rms,
        # 桥面相对水平面的姿态：沿前向坡度 -> pitch，向左升高 -> roll
        'surface_pitch': math.atan(a),
        'surface_roll': math.atan(b),
        'camera_height': -c,
    }


def _deck_run_edges(lateral_sorted, gap):
    """在横向排序的桥面点里找包含 0（机体正前）的连续段，返回 (右缘, 左缘)."""
    if lateral_sorted.size == 0:
        return None
    start = int(np.searchsorted(lateral_sorted, 0.0))
    if start >= lateral_sorted.size:
        start = lateral_sorted.size - 1
    if start > 0 and abs(lateral_sorted[start - 1]) < abs(lateral_sorted[start]):
        start -= 1
    # 机体正前没有桥面点（已经在桥外）：不视为有效带
    if abs(lateral_sorted[start]) > gap:
        return None

    lo = start
    while lo > 0 and (lateral_sorted[lo] - lateral_sorted[lo - 1]) <= gap:
        lo -= 1
    hi = start
    while hi < lateral_sorted.size - 1 and (lateral_sorted[hi + 1] - lateral_sorted[hi]) <= gap:
        hi += 1
    return float(lateral_sorted[lo]), float(lateral_sorted[hi])


def extract_edges_and_heading(points, plane, config):
    """按前向分带拟合两侧桥缘，并在相机原点前向轴处评估控制误差."""
    heights = points[:, 2] - (
        plane['a'] * points[:, 0] + plane['b'] * points[:, 1] + plane['c']
    )
    deck_pts = points[np.abs(heights) <= config.on_deck_height_tol]

    band_centers = []
    left_edges = []
    right_edges = []
    f = config.band_forward_min
    while f < config.band_forward_max:
        band_mask = (deck_pts[:, 0] >= f) & (deck_pts[:, 0] < f + config.band_width)
        band = deck_pts[band_mask]
        if band.shape[0] >= config.min_band_points:
            run = _deck_run_edges(np.sort(band[:, 1]), config.edge_lateral_gap)
            if run is not None:
                right_y, left_y = run
                band_centers.append(f + config.band_width / 2.0)
                right_edges.append(right_y)
                left_edges.append(left_y)
        f += config.band_width

    n_bands = len(band_centers)
    if n_bands < config.min_bands_for_estimate:
        return {'valid': False, 'reason': f'bands<{config.min_bands_for_estimate}',
                'n_bands': n_bands}

    xs = np.asarray(band_centers)
    left_values = np.asarray(left_edges)
    right_values = np.asarray(right_edges)
    widths = left_values - right_values
    median_width = float(np.median(widths))
    ratio = config.band_width_consistency_ratio
    consistent = (
        (widths >= config.min_deck_width)
        & (widths <= config.max_deck_width)
        & (widths >= median_width * ratio)
        & (widths <= median_width / ratio)
    )
    if int(np.count_nonzero(consistent)) < config.min_bands_for_estimate:
        return {'valid': False, 'reason': 'inconsistent_edge_widths',
                'n_bands': n_bands}
    xs = xs[consistent]
    left_values = left_values[consistent]
    right_values = right_values[consistent]
    n_bands = int(xs.size)
    left_line = np.polyfit(xs, left_values, 1)
    right_line = np.polyfit(xs, right_values, 1)
    center_slope = float((left_line[0] + right_line[0]) / 2.0)
    center_intercept = float((left_line[1] + right_line[1]) / 2.0)
    left_at_axis = float(left_line[1])
    right_at_axis = float(right_line[1])
    deck_width = left_at_axis - right_at_axis

    common = {
        'n_bands': n_bands,
        'centerline_slope': center_slope,
        'centerline_intercept': center_intercept,
        'deck_width': float(deck_width),
    }
    if not config.min_deck_width <= deck_width <= config.max_deck_width:
        common.update({'valid': False, 'reason': 'implausible_deck_width'})
        return common
    if (
        left_at_axis <= config.min_axis_edge_clearance
        or right_at_axis >= -config.min_axis_edge_clearance
    ):
        common.update({'valid': False, 'reason': 'robot_axis_outside_deck'})
        return common

    common.update({
        'valid': True,
        'reason': 'ok',
        'd_left_edge': left_at_axis,
        'd_right_edge': -right_at_axis,
        # 正值 = 桥中心线在机体左侧，机体应向左（+vy）修正.
        'lateral_offset': center_intercept,
        # 正值 = 桥轴指向机体前向左侧，应左转（+wz）修正.
        'heading_error': float(math.atan(center_slope)),
    })
    return common


def extract_depth_discontinuity_edges(
    point_grid, plane, config, control_point_x=0.0, control_point_y=0.0
):
    """Fit parallel deck edges from deck-to-below-plane pixel transitions."""
    valid = np.isfinite(point_grid[..., 0])
    heights = np.full(point_grid.shape[:2], np.inf, dtype=np.float64)
    heights[valid] = point_grid[..., 2][valid] - (
        plane['a'] * point_grid[..., 0][valid]
        + plane['b'] * point_grid[..., 1][valid]
        + plane['c']
    )
    deck = valid & (np.abs(heights) <= config.on_deck_height_tol)
    below = valid & (heights < -config.drop_below_plane)
    left_points = []
    right_points = []

    for row in range(point_grid.shape[0]):
        indices = np.flatnonzero(deck[row])
        if indices.size < 3:
            continue
        breaks = np.flatnonzero(np.diff(indices) > 1)
        starts = np.r_[0, breaks + 1]
        stops = np.r_[breaks + 1, indices.size]
        candidates = []
        for start_i, stop_i in zip(starts, stops):
            run = indices[start_i:stop_i]
            if run.size < 3:
                continue
            lo, hi = int(run[0]), int(run[-1])
            if lo < 1 or hi >= point_grid.shape[1] - 1:
                continue
            left_evidence = np.any(below[row, max(0, lo - 2):lo])
            right_evidence = np.any(
                below[row, hi + 1:min(point_grid.shape[1], hi + 3)])
            if not (left_evidence and right_evidence):
                continue
            left = point_grid[row, lo]
            right = point_grid[row, hi]
            forward_mid = 0.5 * (left[0] + right[0])
            if not config.band_forward_min <= forward_mid <= config.band_forward_max:
                continue
            candidates.append((run.size, left, right))
        if candidates:
            _, left, right = max(candidates, key=lambda item: item[0])
            left_points.append(left[:2])
            right_points.append(right[:2])

    n_rows = min(len(left_points), len(right_points))
    if n_rows < config.min_bands_for_estimate:
        return {
            'valid': False,
            'reason': f'edge_rows<{config.min_bands_for_estimate}',
            'n_bands': n_rows,
        }

    left = np.asarray(left_points)
    right = np.asarray(right_points)
    keep_left = np.ones(left.shape[0], dtype=bool)
    keep_right = np.ones(right.shape[0], dtype=bool)
    coef = None
    for _ in range(3):
        left_fit = left[keep_left]
        right_fit = right[keep_right]
        design = np.vstack([
            np.column_stack([
                left_fit[:, 0], np.ones(left_fit.shape[0]),
                np.zeros(left_fit.shape[0]),
            ]),
            np.column_stack([
                right_fit[:, 0], np.zeros(right_fit.shape[0]),
                np.ones(right_fit.shape[0]),
            ]),
        ])
        values = np.r_[left_fit[:, 1], right_fit[:, 1]]
        coef, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        slope, left_intercept, right_intercept = coef
        left_residual = left[:, 1] - (slope * left[:, 0] + left_intercept)
        right_residual = right[:, 1] - (slope * right[:, 0] + right_intercept)
        residual = np.r_[left_residual[keep_left], right_residual[keep_right]]
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        tolerance = max(0.015, 3.0 * 1.4826 * mad)
        new_keep_left = np.abs(left_residual - median) <= tolerance
        new_keep_right = np.abs(right_residual - median) <= tolerance
        if (
            np.count_nonzero(new_keep_left) < config.min_bands_for_estimate
            or np.count_nonzero(new_keep_right) < config.min_bands_for_estimate
        ):
            break
        keep_left, keep_right = new_keep_left, new_keep_right

    slope, left_intercept, right_intercept = (float(v) for v in coef)
    normalizer = math.sqrt(1.0 + slope * slope)
    deck_width = (left_intercept - right_intercept) / normalizer
    center_intercept = 0.5 * (left_intercept + right_intercept)
    camera_lateral_offset = center_intercept / normalizer
    center_at_control = (
        slope * float(control_point_x) + center_intercept
        - float(control_point_y)
    ) / normalizer
    d_left = (
        slope * float(control_point_x) + left_intercept
        - float(control_point_y)
    ) / normalizer
    d_right = (
        float(control_point_y)
        - (slope * float(control_point_x) + right_intercept)
    ) / normalizer
    common = {
        'n_bands': int(min(np.count_nonzero(keep_left),
                           np.count_nonzero(keep_right))),
        'centerline_slope': slope,
        'centerline_intercept': center_intercept,
        'deck_width': float(deck_width),
        'control_point_x': float(control_point_x),
        'control_point_y': float(control_point_y),
        'camera_lateral_offset': float(camera_lateral_offset),
    }
    if not config.min_deck_width <= deck_width <= config.max_deck_width:
        common.update({'valid': False, 'reason': 'implausible_deck_width'})
        return common
    if d_left <= config.min_axis_edge_clearance or d_right <= config.min_axis_edge_clearance:
        common.update({'valid': False, 'reason': 'robot_axis_outside_deck'})
        return common
    common.update({
        'valid': True,
        'reason': 'ok',
        'd_left_edge': float(d_left),
        'd_right_edge': float(d_right),
        'lateral_offset': float(center_at_control),
        'heading_error': float(math.atan(slope)),
    })
    return common


def extract_single_sided_edge(
    point_grid, plane, config, control_point_x=0.0, control_point_y=0.0
):
    """Estimate the centreline from one drop-off edge plus a declared width.

    ``extract_depth_discontinuity_edges`` requires below-plane evidence on both
    sides of the deck.  That holds on the bridge, whose two sides are both open
    drops, but not on the ring rails: each rail has the ring's void on its inner
    side and a raised border kerb on its outer side, and a kerb produces
    *above*-plane points, never below-plane ones.  Measured in ``race.world``,
    that costs the observer the entire bottom rail.

    This degraded mode keeps the one edge it can actually see and places the
    centreline half a declared deck width inboard of it.  It is strictly worse
    than the two-sided fit — the second edge is what normally catches a bad
    width — so it is opt-in and must never be the primary evidence for a
    segment.
    """
    half_width = 0.5 * float(config.declared_deck_width)
    if not config.single_sided_edges_enabled or half_width <= 0.0:
        return {'valid': False, 'reason': 'single_sided_disabled', 'n_bands': 0}

    valid = np.isfinite(point_grid[..., 0])
    heights = np.full(point_grid.shape[:2], np.inf, dtype=np.float64)
    heights[valid] = point_grid[..., 2][valid] - (
        plane['a'] * point_grid[..., 0][valid]
        + plane['b'] * point_grid[..., 1][valid]
        + plane['c']
    )
    deck = valid & (np.abs(heights) <= config.on_deck_height_tol)
    below = valid & (heights < -config.drop_below_plane)

    left_points = []
    right_points = []
    for row in range(point_grid.shape[0]):
        indices = np.flatnonzero(deck[row])
        if indices.size < 3:
            continue
        breaks = np.flatnonzero(np.diff(indices) > 1)
        starts = np.r_[0, breaks + 1]
        stops = np.r_[breaks + 1, indices.size]
        best = None
        for start_i, stop_i in zip(starts, stops):
            run = indices[start_i:stop_i]
            if run.size < 3:
                continue
            lo, hi = int(run[0]), int(run[-1])
            if lo < 1 or hi >= point_grid.shape[1] - 1:
                continue
            forward_mid = 0.5 * (point_grid[row, lo][0] + point_grid[row, hi][0])
            if not config.band_forward_min <= forward_mid <= config.band_forward_max:
                continue
            if best is None or run.size > best[0]:
                best = (run.size, lo, hi)
        if best is None:
            continue
        _, lo, hi = best
        if np.any(below[row, max(0, lo - 2):lo]):
            left_points.append(point_grid[row, lo][:2])
        if np.any(below[row, hi + 1:min(point_grid.shape[1], hi + 3)]):
            right_points.append(point_grid[row, hi][:2])

    side, edge_points = 'left', left_points
    if len(right_points) > len(left_points):
        side, edge_points = 'right', right_points
    n_rows = len(edge_points)
    if n_rows < config.min_bands_for_estimate:
        return {
            'valid': False,
            'reason': f'single_sided_rows<{config.min_bands_for_estimate}',
            'n_bands': n_rows,
        }

    edge = np.asarray(edge_points, dtype=np.float64)
    slope, intercept = np.polyfit(edge[:, 0], edge[:, 1], 1)
    normalizer = math.sqrt(1.0 + slope * slope)
    # The left edge lies at greater y than the centre, the right edge at less;
    # this matches the two-sided fit's ``left_intercept > right_intercept``.
    if side == 'left':
        center_intercept = intercept - half_width * normalizer
    else:
        center_intercept = intercept + half_width * normalizer
    center_at_control = (
        slope * float(control_point_x) + center_intercept - float(control_point_y)
    ) / normalizer
    d_left = center_at_control + half_width
    d_right = half_width - center_at_control
    common = {
        'n_bands': int(n_rows),
        'centerline_slope': float(slope),
        'centerline_intercept': float(center_intercept),
        'deck_width': float(config.declared_deck_width),
        'control_point_x': float(control_point_x),
        'control_point_y': float(control_point_y),
        'camera_lateral_offset': float(center_intercept / normalizer),
        'edge_side': side,
        'single_sided': True,
    }
    if d_left <= config.min_axis_edge_clearance or d_right <= config.min_axis_edge_clearance:
        common.update({'valid': False, 'reason': 'robot_axis_outside_deck'})
        return common
    common.update({
        'valid': True,
        'reason': 'ok_single_sided',
        'd_left_edge': float(d_left),
        'd_right_edge': float(d_right),
        'lateral_offset': float(center_at_control),
        'heading_error': float(math.atan(slope)),
    })
    return common


def detect_forward_dropoff(points, plane, edges, config):
    """沿估计桥中心线走廊寻找桥面末端及紧随其后的真实低点.

    The corridor only has to point roughly where the robot is going, and the
    two-sided edge fit is a much stronger thing to demand than that: on the
    ring rails it succeeds on 12-40% of frames, so "how far ahead does the deck
    end" — the one course-referenced measurement that could trigger a corner on
    the course rather than on accumulated odometry — is unavailable exactly
    where it is most wanted.  With ``dropoff_body_axis_fallback`` the corridor
    falls back to the body's own forward axis (y = 0), which is where the robot
    is heading by definition.

    The fallback is reported as ``ok_body_axis`` rather than ``ok`` because it
    is a weaker measurement and a consumer may reasonably want to know: a body
    yawed relative to the deck samples a corridor that walks off the centreline
    the further ahead it looks, so it reads the deck ending **early**.

    That under-read is a bias to calibrate out, not a safety margin.  Measured:
    a corner trigger set at 1.10 m fired 0.26 m before the course-referenced
    point, putting the corner-2 take-off at x = -0.03 against a usable window
    of -0.22 to -0.37, and the jump failed on both runs that reached it.  Early
    leaves the window on the near side exactly as fatally as late leaves it on
    the far side.  What the reading *is* good for is repeatability: with the
    bias calibrated out, the same trigger cut the take-off spread from 0.116 m
    to 0.075 m, which accumulated odometry cannot do at any segment length.
    """
    edges_valid = bool(edges.get('valid', False))
    if not edges_valid and not config.dropoff_body_axis_fallback:
        return {'deck_end_x': None, 'dropoff_detected': False,
                'reason': 'invalid_edges'}
    if edges_valid:
        center_y = (
            edges['centerline_slope'] * points[:, 0]
            + edges['centerline_intercept']
        )
    else:
        center_y = np.zeros(points.shape[0])
    corridor = points[
        np.abs(points[:, 1] - center_y) <= config.dropoff_lateral_half_width
    ]
    ok_reason = 'ok' if edges_valid else 'ok_body_axis'
    if corridor.shape[0] == 0:
        return {'deck_end_x': None, 'dropoff_detected': False,
                'reason': 'no_corridor_points'}

    heights = corridor[:, 2] - (
        plane['a'] * corridor[:, 0] + plane['b'] * corridor[:, 1] + plane['c']
    )
    deck_x = np.sort(corridor[np.abs(heights) <= config.on_deck_height_tol][:, 0])
    below_x = corridor[heights < -config.drop_below_plane][:, 0]
    if deck_x.size == 0:
        return {'deck_end_x': None, 'dropoff_detected': False,
                'reason': 'no_deck_in_corridor'}

    end_x = float(deck_x[0])
    for x in deck_x[1:]:
        if float(x) - end_x > config.dropoff_forward_gap:
            break
        end_x = float(x)

    evidence = below_x[
        (below_x > end_x)
        & (below_x <= end_x + config.dropoff_evidence_max_gap)
    ]
    dropoff = evidence.size >= config.dropoff_min_below_points
    return {
        'deck_end_x': end_x,
        'dropoff_detected': bool(dropoff),
        'reason': ok_reason,
        'below_evidence_points': int(evidence.size),
    }


def invalid_bridge_observation(metadata, reason, n_points=0):
    """Return the fixed observer schema for a failed depth observation."""
    return {
        **metadata,
        'valid': False,
        'reason': reason,
        'd_left_edge': None,
        'd_right_edge': None,
        'lateral_offset': None,
        'camera_lateral_offset': None,
        'heading_error': None,
        'surface_roll': None,
        'surface_pitch': None,
        'camera_height': None,
        'deck_end_x': None,
        'deck_end_camera_x': None,
        'd_forward_dropoff_camera': None,
        'd_forward_dropoff': None,
        'dropoff_detected': False,
        'quality': {'n_points': int(n_points)},
    }


def bridge_observation(
    depth_m,
    intrinsics,
    camera_roll=0.0,
    camera_pitch=0.0,
    config=None,
    control_point_x=0.0,
    control_point_y=0.0,
    frame_seq=None,
    stamp_s=None,
):
    """
    Convert one depth frame into the geometry required for control.

    返回字段（invalid 时只保证 valid/reason）：
    d_left_edge / d_right_edge / lateral_offset / heading_error /
    surface_roll / surface_pitch / camera_height /
    deck_end_x / dropoff_detected / quality（点数、带数、平面内点数、rms）.
    """
    cfg = config if config is not None else BridgePerceptionConfig()
    metadata = {'frame_seq': frame_seq, 'stamp_s': stamp_s}

    if depth_m is None or getattr(depth_m, 'ndim', 0) != 2:
        return invalid_bridge_observation(metadata, 'no_depth_image')

    point_grid = depth_to_gravity_grid(
        depth_m, intrinsics, camera_roll, camera_pitch, cfg)
    points = point_grid[np.isfinite(point_grid[..., 0])]
    if points.shape[0] < cfg.ransac_min_inliers:
        return invalid_bridge_observation(
            metadata,
            f'points<{cfg.ransac_min_inliers}',
            n_points=points.shape[0],
        )

    plane = fit_deck_plane(points, cfg)
    if plane is None:
        return invalid_bridge_observation(
            metadata, 'plane_fit_failed', n_points=points.shape[0])

    edges = extract_depth_discontinuity_edges(
        point_grid,
        plane,
        cfg,
        control_point_x=control_point_x,
        control_point_y=control_point_y,
    )
    if not edges['valid'] and cfg.single_sided_edges_enabled:
        # Only ever a fallback: the two-sided fit is the one that can catch its
        # own bad width, so it always gets first refusal.
        single = extract_single_sided_edge(
            point_grid,
            plane,
            cfg,
            control_point_x=control_point_x,
            control_point_y=control_point_y,
        )
        if single['valid']:
            edges = single
    dropoff = detect_forward_dropoff(points, plane, edges, cfg)

    d_forward_camera = None
    d_forward_control = None
    if dropoff['dropoff_detected'] and edges['valid']:
        end_x = float(dropoff['deck_end_x'])
        slope = float(edges['centerline_slope'])
        end_y = slope * end_x + float(edges['centerline_intercept'])
        normalizer = math.sqrt(1.0 + slope * slope)
        unit_x, unit_y = 1.0 / normalizer, slope / normalizer
        d_forward_camera = end_x * unit_x + end_y * unit_y
        d_forward_control = (
            (end_x - float(control_point_x)) * unit_x
            + (end_y - float(control_point_y)) * unit_y
        )
    elif dropoff['dropoff_detected'] and dropoff['reason'] == 'ok_body_axis':
        # No centreline to project onto, because the edge fit is what failed.
        # The body's own forward axis is the reference instead, so the distance
        # is just the deck end's x — which is the whole point of the fallback:
        # "how far ahead does the deck end" is answerable without knowing where
        # its sides are, and on the ring rails the sides are exactly what the
        # observer cannot see (a two-sided fit lands on 12-40% of frames).
        end_x = float(dropoff['deck_end_x'])
        d_forward_camera = end_x
        d_forward_control = end_x - float(control_point_x)

    result = {
        **metadata,
        'valid': bool(edges['valid']),
        'reason': edges['reason'],
        'surface_roll': plane['surface_roll'],
        'surface_pitch': plane['surface_pitch'],
        'camera_height': plane['camera_height'],
        'deck_end_x': dropoff['deck_end_x'],
        'deck_end_camera_x': dropoff['deck_end_x'],
        'd_forward_dropoff_camera': d_forward_camera,
        'd_forward_dropoff': d_forward_control,
        'dropoff_detected': dropoff['dropoff_detected'],
        'd_left_edge': None,
        'd_right_edge': None,
        'lateral_offset': None,
        'camera_lateral_offset': None,
        'heading_error': None,
        'quality': {
            'n_points': int(points.shape[0]),
            'plane_inliers': int(plane['inlier_count']),
            'plane_rms': float(plane['rms']),
            'n_bands': int(edges.get('n_bands', 0)),
        },
    }
    if edges['valid']:
        result.update({
            'd_left_edge': edges['d_left_edge'],
            'd_right_edge': edges['d_right_edge'],
            'lateral_offset': edges['lateral_offset'],
            'camera_lateral_offset': edges['camera_lateral_offset'],
            'heading_error': edges['heading_error'],
        })
    return result
