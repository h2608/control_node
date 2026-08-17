#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成图像上的第二赛段视觉调试工具单元测试。

只测试 stage2_vision_preview 里的纯函数与绘制函数，不启动 ROS 节点。
"""

import collections

import cv2
import numpy as np

from control_node.stage2_vision_preview import (
    BLUE_DEFAULTS,
    FISHEYE_DEFAULTS,
    ORANGE_DEFAULTS,
    RGB_BALL_DEFAULTS,
    YELLOW_DEFAULTS,
    ball_scene,
    depth_sample,
    depth_view,
    fisheye_approach_vx,
    fisheye_ball_records,
    fisheye_best,
    fisheye_entry_ok,
    fisheye_view,
    rgb_ball_records,
    rgb_ball_view,
    rgb_yellow_view,
    yellow_align_wz,
    yellow_line_records,
)


ORANGE_BGR = (0, 140, 255)
BLUE_BGR = (255, 60, 0)
YELLOW_BGR = (0, 255, 255)


def make_config():
    config = collections.OrderedDict()
    for defaults in (ORANGE_DEFAULTS, BLUE_DEFAULTS, FISHEYE_DEFAULTS,
                     RGB_BALL_DEFAULTS, YELLOW_DEFAULTS):
        config.update(defaults)
    return config


def blank_frame(width=640, height=480, value=70):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (value, value, value)
    return frame


def uniform_depth(width=640, height=480, millimetres=800):
    return np.full((height, width), millimetres, dtype=np.uint16)


def failed_names(record):
    return [item['name'] for item in record['checks'] if not item['passed']]


def test_fisheye_accepts_a_round_orange_ball():
    config = make_config()
    frame = blank_frame()
    cv2.circle(frame, (320, 240), 30, ORANGE_BGR, -1)

    records, mask = fisheye_ball_records(frame, config, config, 'orange')
    best = fisheye_best(records)

    assert mask.shape == frame.shape[:2]
    assert best is not None
    assert abs(best['radius'] - 30.0) < 4.0
    assert abs(best['center'][0] - 320) <= 2
    assert best['circularity'] >= config['fisheye_min_circularity']


def test_fisheye_rejects_a_flat_blob_on_aspect_ratio():
    config = make_config()
    frame = blank_frame()
    cv2.ellipse(frame, (320, 240), (60, 18), 0, 0, 360, ORANGE_BGR, -1)

    records, _ = fisheye_ball_records(frame, config, config, 'orange')

    assert records, 'the blob must stay visible as a FAIL candidate'
    assert fisheye_best(records) is None
    assert 'aspect_ratio' in failed_names(records[0])


def test_fisheye_detects_the_blue_ball_with_the_same_shape_gates():
    config = make_config()
    frame = blank_frame()
    cv2.circle(frame, (200, 260), 28, BLUE_BGR, -1)

    orange_records, _ = fisheye_ball_records(frame, config, config, 'orange')
    blue_records, _ = fisheye_ball_records(frame, config, config, 'blue')

    assert fisheye_best(orange_records) is None
    blue_best = fisheye_best(blue_records)
    assert blue_best is not None
    assert abs(blue_best['radius'] - 28.0) < 4.0


def test_fisheye_entry_window_and_approach_vx():
    config = make_config()
    centered = {'center': (320, 240), 'image_shape': (480, 640)}
    off_center = {'center': (600, 240), 'image_shape': (480, 640)}

    ok, x_ratio, _ = fisheye_entry_ok(centered, config)
    assert ok
    assert abs(x_ratio - 0.5) < 1e-6
    assert not fisheye_entry_ok(off_center, config)[0]

    vx_centered, error = fisheye_approach_vx(centered, config, 'left')
    assert vx_centered == 0.0, 'inside the deadband there is no vx'
    assert abs(error) < 1e-6

    vx_left, _ = fisheye_approach_vx(off_center, config, 'left')
    vx_right, _ = fisheye_approach_vx(off_center, config, 'right')
    assert vx_left > 0.0 and vx_right < 0.0, 'the two sides must mirror'
    assert abs(vx_left) <= config['fisheye_approach_vx_max'] + 1e-9


def test_rgb_ball_needs_a_valid_depth_sample():
    config = make_config()
    frame = blank_frame()
    cv2.circle(frame, (240, 300), 30, ORANGE_BGR, -1)
    depth = uniform_depth()

    with_depth, _ = rgb_ball_records(
        frame, depth, '16UC1', config, config, 'orange')
    assert with_depth[0]['accepted']
    assert abs(with_depth[0]['depth_m'] - 0.8) < 0.01

    without_depth, _ = rgb_ball_records(
        frame, None, '', config, config, 'orange')
    assert not without_depth[0]['accepted']
    assert 'depth_valid' in failed_names(without_depth[0])


def test_depth_sample_matches_the_stage2_percentile_window():
    config = make_config()
    depth = uniform_depth()
    depth[290:310, 230:250] = 500

    value, center, box = depth_sample(
        depth, '16UC1', (240, 300), (480, 640), config)

    assert center == (240, 300)
    assert box == (228, 288, 253, 313)
    assert abs(value - 0.5) < 0.05, 'the 20th percentile follows the near patch'


def test_ball_scene_picks_the_nearest_orange_and_the_lane_centre():
    config = make_config()
    frame = blank_frame()
    cv2.circle(frame, (160, 300), 30, ORANGE_BGR, -1)
    cv2.circle(frame, (480, 300), 30, BLUE_BGR, -1)
    depth = uniform_depth()
    depth[:, :320] = 600

    orange_records, _ = rgb_ball_records(
        frame, depth, '16UC1', config, config, 'orange')
    blue_records, _ = rgb_ball_records(
        frame, depth, '16UC1', config, config, 'blue')
    scene = ball_scene(orange_records, blue_records, frame.shape, config)

    assert len(scene['orange']) == 1 and len(scene['blue']) == 1
    assert scene['left_ref']['color'] == 'orange'
    assert scene['right_ref']['color'] == 'blue'
    assert abs(scene['lane_mid_x'] - 320.0) < 3.0
    assert abs(scene['center_error_px']) <= config['center_ok_px']
    assert scene['target']['color'] == 'orange'
    # 0.60 m 与 0.80 m 相差 0.20 m，小于 center_depth_diff_disable_align_m。
    assert scene['center_mode'] == 'CENTER_OK'
    assert scene['center_vy'] == 0.0


def test_ball_scene_flags_a_near_left_ball_for_the_avoid_gate():
    config = make_config()
    frame = blank_frame()
    cv2.circle(frame, (260, 320), 30, BLUE_BGR, -1)
    depth = uniform_depth(millimetres=400)

    blue_records, _ = rgb_ball_records(
        frame, depth, '16UC1', config, config, 'blue')
    scene = ball_scene([], blue_records, frame.shape, config)

    danger = scene['danger_ball']
    assert danger is not None
    assert danger['distance_to_center_px'] <= (
        config['stage2_left_ball_avoid_center_px'])
    assert danger['depth_m'] <= config['stage2_left_ball_avoid_depth_m']


def test_yellow_horizontal_line_passes_the_three_front_line_conditions():
    config = make_config()
    frame = blank_frame()
    cv2.rectangle(frame, (270, 400), (370, 410), YELLOW_BGR, -1)

    yellow = yellow_line_records(frame, config)

    assert yellow['best_strict'] is not None
    assert yellow['best_loose'] is not None
    assert yellow['best_strict']['bottom_y'] >= 405
    assert abs(yellow['best_strict']['angle_deg']) < 5.0
    assert yellow['roi'] == (256, 312, 384, 480)


def test_yellow_narrow_mark_fails_strict_but_survives_loose():
    config = make_config()
    frame = blank_frame()
    cv2.rectangle(frame, (300, 400), (316, 440), YELLOW_BGR, -1)

    yellow = yellow_line_records(frame, config)

    assert yellow['best_strict'] is None
    assert yellow['best_loose'] is not None
    assert 'wh_ratio' in failed_names(yellow['records'][0])


def test_yellow_angle_align_is_a_fixed_signed_wz():
    config = make_config()

    assert yellow_align_wz(0.2, config) == 0.0, 'inside the deadband'
    assert yellow_align_wz(None, config) == 0.0
    assert yellow_align_wz(8.0, config) < 0.0
    assert yellow_align_wz(-8.0, config) > 0.0
    assert abs(yellow_align_wz(8.0, config)) == (
        config['yellow_angle_align_fixed_wz'])


def test_views_render_annotated_images():
    config = make_config()
    frame = blank_frame()
    cv2.circle(frame, (300, 240), 32, ORANGE_BGR, -1)
    cv2.circle(frame, (140, 300), 26, BLUE_BGR, -1)
    cv2.rectangle(frame, (270, 400), (370, 410), YELLOW_BGR, -1)
    depth = uniform_depth()

    orange_records, _ = fisheye_ball_records(frame, config, config, 'orange')
    blue_records, _ = fisheye_ball_records(frame, config, config, 'blue')
    fisheye = fisheye_view(frame, 'left', orange_records, blue_records, config,
                           {'entry': 2, 'hit': 1}, 'bgr8', '')

    rgb_orange, _ = rgb_ball_records(
        frame, depth, '16UC1', config, config, 'orange')
    rgb_blue, _ = rgb_ball_records(
        frame, depth, '16UC1', config, config, 'blue')
    scene = ball_scene(rgb_orange, rgb_blue, frame.shape, config)
    balls = rgb_ball_view(frame, scene, rgb_orange, rgb_blue, config,
                          depth.shape)
    yellow = rgb_yellow_view(frame, yellow_line_records(frame, config), config)
    depth_image = depth_view(depth, '16UC1', frame.shape)

    for image in (fisheye, balls, yellow):
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.shape[1] == frame.shape[1]
        assert image.shape[0] > frame.shape[0], '面板必须追加在图像下方'
    assert depth_image.shape == frame.shape
    assert not np.array_equal(fisheye[:frame.shape[0]], frame), '必须有标注'
