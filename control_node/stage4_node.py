#!/usr/bin/env python3
# Stage4 LCM version: low-body wait + clear-bar distance + forward-pitch target vision
# -*- coding: utf-8 -*-
"""第四赛段节点：识别可乐瓶/橙球/足球/限高杆/障碍物，完成语音播报、
目标交互与避障动作。

原 control_node_123456.py 的 FourthStageMixin（状态机与视觉逻辑原样搬移）。
结尾 GLOBAL_FINAL_P3_ALIGN 复用第三赛段赛道对齐视觉（P3TrackVisionMixin）。
DONE 后向任务控制节点上报完成（原来是 enter_fifth_stage()）。
"""

import json
import math
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from control_node.cola_detector import HybridColaDetector
from control_node.football_vision import detect_football
from control_node.my_gait import Robot_Ctrl
from control_node.stage_common import StageNodeBase, clamp
from control_node.stage_entry import EntryPoint, StageEntryTable
from control_node.stage3_node import P3TrackVisionMixin
from control_node.cyberdog_voice import CyberdogVoicePlayer


class Detection:
    # Plain class for Python 3.6 compatibility (dataclasses are stdlib >=3.7).
    def __init__(self, det_type, center_img, bbox_img, score, extra):
        self.det_type = det_type
        self.center_img = center_img
        self.bbox_img = bbox_img
        self.score = score
        self.extra = extra


class _Stage4RgbReceiverNode(Node):
    """Dedicated raw RGB receiver isolated from the Stage4 control executor.

    The callback intentionally does no cv_bridge/OpenCV work.  It only swaps a
    reference to the newest ROS Image message, so the receiver executor stays
    extremely light even while Stage4 control/perception is busy.
    """

    def __init__(self, context, namespace: str, rgb_topic: str):
        # Ignore process-wide launch remaps such as ``__node:=stage4_node``.
        # Otherwise this helper would be renamed to stage4_node too, producing
        # duplicate node/logger names inside the same process.
        super().__init__(
            'stage4_rgb_rx',
            context=context,
            namespace=namespace,
            use_global_arguments=False,
        )
        self._lock = threading.Lock()
        self._latest_msg = None
        self._seq = 0
        self._last_rx_monotonic_s = None
        self._sub = self.create_subscription(
            Image,
            rgb_topic,
            self._rgb_cb,
            qos_profile_sensor_data,
        )

    def _rgb_cb(self, msg: Image):
        now = time.monotonic()
        with self._lock:
            self._latest_msg = msg
            self._seq += 1
            self._last_rx_monotonic_s = now

    def snapshot(self):
        with self._lock:
            return self._seq, self._latest_msg, self._last_rx_monotonic_s


class BaseDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)
        return (x1, y1, x2, y2), frame_bgr[y1:y2, x1:x2].copy()


class BarColorDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.lower_bar = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper_bar = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.hue_wraps = cfg['h_min'] > cfg['h_max']
        if self.hue_wraps:
            self.upper_bar[0] = 179
            self.lower_bar_high = np.array(
                [0, cfg['s_min'], cfg['v_min']], dtype=np.uint8)
            self.upper_bar_high = np.array(
                [cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel_h'], cfg['close_kernel_w']), np.uint8)
        self.min_area = cfg['min_area']
        self.min_width = cfg['min_width']
        self.max_height = cfg['max_height']
        self.min_aspect_ratio = cfg['min_aspect_ratio']
        self.max_aspect_ratio = cfg['max_aspect_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']
        self.structure_check_enabled = cfg['structure_check_enabled']
        self.structure_inner_ratio = cfg['structure_inner_ratio']
        self.structure_max_inner_red_ratio = cfg['structure_max_inner_red_ratio']
        self.structure_min_depth_gap_m = cfg['structure_min_depth_gap_m']
        self.structure_near_bypass_distance_m = cfg[
            'structure_near_bypass_distance_m']
        self.structure_near_bypass_width_ratio = cfg[
            'structure_near_bypass_width_ratio']
        self.structure_min_depth_pixels = cfg['structure_min_depth_pixels']

    @staticmethod
    def _depth_to_meters(depth_img):
        if depth_img is None:
            return None
        if depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def _check_structure(self, mask_patch, depth_patch, frame_width):
        ph, pw = mask_patch.shape[:2]
        inner_ratio = max(0.05, min(1.0, self.structure_inner_ratio))
        inner_w = max(1, int(round(pw * inner_ratio)))
        inner_h = max(1, int(round(ph * inner_ratio)))
        ix1 = max(0, (pw - inner_w) // 2)
        iy1 = max(0, (ph - inner_h) // 2)
        ix2 = min(pw, ix1 + inner_w)
        iy2 = min(ph, iy1 + inner_h)

        inner_mask = mask_patch[iy1:iy2, ix1:ix2]
        inner_red_ratio = float(np.count_nonzero(inner_mask)) / float(
            max(inner_mask.size, 1))

        bar_depth = None
        inner_depth = None
        depth_gap = None
        if depth_patch is not None:
            red_depths = depth_patch[mask_patch > 0]
            red_depths = red_depths[
                np.isfinite(red_depths) & (red_depths > 0.05)
                & (red_depths < 10.0)]
            inner_depths = depth_patch[iy1:iy2, ix1:ix2].reshape(-1)
            inner_depths = inner_depths[
                np.isfinite(inner_depths) & (inner_depths > 0.05)
                & (inner_depths < 10.0)]
            if red_depths.size >= self.structure_min_depth_pixels:
                bar_depth = float(np.percentile(red_depths, 20))
            if inner_depths.size >= self.structure_min_depth_pixels:
                inner_depth = float(np.median(inner_depths))
            if bar_depth is not None and inner_depth is not None:
                depth_gap = abs(inner_depth - bar_depth)

        width_ratio = pw / float(max(frame_width, 1))
        near_bypass = (
            (bar_depth is not None
             and bar_depth <= self.structure_near_bypass_distance_m)
            or width_ratio >= self.structure_near_bypass_width_ratio
        )
        color_hole_ok = (
            inner_red_ratio <= self.structure_max_inner_red_ratio)
        depth_gap_ok = (
            depth_gap is not None
            and depth_gap >= self.structure_min_depth_gap_m)
        passed = near_bypass or color_hole_ok or depth_gap_ok
        return passed, {
            'inner_red_ratio': inner_red_ratio,
            'bar_surface_depth_m': bar_depth,
            'inner_depth_m': inner_depth,
            'structure_depth_gap_m': depth_gap,
            'structure_near_bypassed': near_bypass,
        }

    def detect(self, frame_bgr, depth_img=None) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        depth_m = self._depth_to_meters(depth_img)
        roi_depth = None
        if depth_m is not None:
            if depth_m.shape[:2] != frame_bgr.shape[:2]:
                depth_m = cv2.resize(
                    depth_m, (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
            roi_depth = depth_m[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bar, self.upper_bar)
        if self.hue_wraps:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(hsv, self.lower_bar_high, self.upper_bar_high),
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw <= 0 or rh <= 0:
                continue
            aspect_ratio = rw / float(rh)
            center_y_ratio = (ry + rh * 0.5) / float(max(roi_h, 1))
            center_x = rx + rw / 2.0
            x_dist_norm = abs(center_x - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area:
                continue
            if rw < self.min_width:
                continue
            if rh > self.max_height:
                continue
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue

            structure_info = {}
            if self.structure_check_enabled:
                mask_patch = mask[ry:ry + rh, rx:rx + rw]
                depth_patch = None
                if roi_depth is not None:
                    depth_patch = roi_depth[ry:ry + rh, rx:rx + rw]
                structure_ok, structure_info = self._check_structure(
                    mask_patch, depth_patch, frame_bgr.shape[1])
                if not structure_ok:
                    continue

            center_score = max(center_bonus, 0.0)
            score = center_score

            # 限高杆角度：用当前轮廓拟合直线，得到相对图像水平线的有符号角度。
            # angle_deg > 0 表示图像中从左到右向下倾斜；angle_deg < 0 表示从左到右向上倾斜。
            angle_deg = 0.0
            if cnt is not None and len(cnt) >= 2:
                vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                vx = float(vx)
                vy = float(vy)
                angle_deg = math.degrees(math.atan2(vy, vx))
                while angle_deg > 90.0:
                    angle_deg -= 180.0
                while angle_deg < -90.0:
                    angle_deg += 180.0

            candidates.append((
                score, rx, ry, rw, rh, aspect_ratio, angle_deg,
                structure_info))
        if not candidates:
            return None
        score, rx, ry, rw, rh, aspect_ratio, angle_deg, structure_info = max(
            candidates, key=lambda x: x[0])
        bx1, by1 = x1 + rx, y1 + ry
        bx2, by2 = bx1 + rw, by1 + rh
        cx = bx1 + rw // 2
        cy = by1 + rh // 2
        extra = {
            'aspect_ratio': float(aspect_ratio),
            'angle_deg': float(angle_deg),
            'abs_tilt_deg': float(abs(angle_deg)),
        }
        extra.update(structure_info)
        return Detection(
            'bar', (cx, cy), (bx1, by1, bx2, by2), float(score), extra)


class BallDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any], det_type: str):
        super().__init__(cfg)
        self.det_type = det_type
        self.lower = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], dtype=np.uint8)
        self.upper = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], dtype=np.uint8)
        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8)
        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_radius = cfg['min_radius']
        self.max_radius = cfg['max_radius']
        self.min_circularity = cfg['min_circularity']
        self.min_wh_ratio = cfg['min_wh_ratio']
        self.max_wh_ratio = cfg['max_wh_ratio']
        self.max_center_y_ratio_in_roi = cfg['max_center_y_ratio_in_roi']
        self.center_weight_base = cfg['center_weight_base']
        self.center_weight_gain = cfg['center_weight_gain']
        self.radius_score_gain = cfg['radius_score_gain']

    def detect(self, frame_bgr) -> Optional[Detection]:
        (x1, y1, x2, y2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]
        frame_area = float(max(frame_bgr.shape[0] * frame_bgr.shape[1], 1))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_center_x = roi_w / 2.0
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 1e-6 else 0.0
            wh_ratio = bw / float(bh)
            center_y_ratio = cy / float(max(roi_h, 1))
            x_dist_norm = abs(cx - roi_center_x) / max(roi_w / 2.0, 1.0)
            center_bonus = 1.0 - x_dist_norm
            if area < self.min_area or area > self.max_area:
                continue
            if radius < self.min_radius or radius > self.max_radius:
                continue
            if circularity < self.min_circularity:
                continue
            if wh_ratio < self.min_wh_ratio or wh_ratio > self.max_wh_ratio:
                continue
            if center_y_ratio > self.max_center_y_ratio_in_roi:
                continue
            score = (radius * self.radius_score_gain) * max(circularity, 0.0) * (
                        self.center_weight_base + self.center_weight_gain * center_bonus)
            candidates.append((
                score, x, y, bw, bh, cx, cy, radius, circularity, area))
        if not candidates:
            return None
        score, x, y, bw, bh, cx, cy, radius, circularity, area = max(
            candidates, key=lambda c: c[0])
        bx1, by1 = x1 + x, y1 + y
        bx2, by2 = bx1 + bw, by1 + bh
        cx_img = x1 + int(round(cx))
        cy_img = y1 + int(round(cy))
        return Detection(self.det_type, (cx_img, cy_img), (bx1, by1, bx2, by2), float(score), {
            'radius': float(radius),
            'circularity': float(circularity),
            'area': float(area),
            'area_ratio': float(area / frame_area),
        })


class FootballDetector:
    """Adapt the shared football detector to Stage 4's Detection interface."""

    def detect(self, frame_bgr, depth_at=None) -> Optional[Detection]:
        result = detect_football(frame_bgr, depth_at=depth_at)
        if result is None:
            return None

        height, width = frame_bgr.shape[:2]
        x = float(result['x'])
        y = float(result['y'])
        radius = float(result['radius'])
        cx = int(round(x))
        cy = int(round(y))
        x1 = max(0, int(math.floor(x - radius)))
        y1 = max(0, int(math.floor(y - radius)))
        x2 = min(width - 1, int(math.ceil(x + radius)))
        y2 = min(height - 1, int(math.ceil(y + radius)))
        area = math.pi * radius * radius
        frame_area = float(max(width * height, 1))
        extra = {
            key: value
            for key, value in result.items()
            if key not in ('x', 'y', 'score')
        }
        extra.update({
            'area': float(area),
            'area_ratio': float(area / frame_area),
            'method': 'football_hough_black_white',
        })
        return Detection(
            'white_ball',
            (cx, cy),
            (x1, y1, x2, y2),
            float(result['score']),
            extra,
        )


class ColaDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.impl = HybridColaDetector(cfg)

    def detect(self, frame_bgr, candidate_key=None) -> Optional[Detection]:
        result = self.impl.detect(frame_bgr, candidate_key=candidate_key)
        if result is None:
            return None
        extra = {
            key: value
            for key, value in result.items()
            if key not in ('center', 'bbox', 'score')
        }
        return Detection(
            'cola',
            tuple(result['center']),
            tuple(result['bbox']),
            float(result['score']),
            extra,
        )


class ObstacleBlueDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        self.lower_blue = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_blue = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        self.use_depth_filter = bool(cfg.get('use_depth_filter', True))

        self.depth_min_m = cfg['depth_min_m']
        self.depth_max_m = cfg['depth_max_m']

        self.kernel_open = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.kernel_close = np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8)

        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']
        self.min_aspect_ratio = cfg['min_aspect_ratio']
        self.max_aspect_ratio = cfg['max_aspect_ratio']
        self.min_bottom_y_ratio_in_roi = cfg['min_bottom_y_ratio_in_roi']

        self.min_valid_depth_ratio = cfg['min_valid_depth_ratio']
        self.min_near_depth_ratio = cfg['min_near_depth_ratio']
        self.bbox_depth_percentile = cfg['bbox_depth_percentile']
        self.bbox_depth_margin_m = cfg['bbox_depth_margin_m']
        self.bbox_depth_min_pixels = cfg['bbox_depth_min_pixels']

    def depth_to_meters(self, depth_img):
        if depth_img is None:
            return None

        if depth_img.dtype == np.float32:
            depth_m = depth_img.copy()
        elif depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def refine_bbox_with_depth(self, cnt, roi_depth_m, rx, ry, rw, rh):
        """用障碍物表面深度收紧蓝色轮廓，避免地面阴影拉长外接框。"""
        depth_patch = roi_depth_m[ry:ry + rh, rx:rx + rw]
        contour_mask = np.zeros((rh, rw), dtype=np.uint8)
        shifted_cnt = cnt.copy()
        shifted_cnt[:, :, 0] -= rx
        shifted_cnt[:, :, 1] -= ry
        cv2.drawContours(contour_mask, [shifted_cnt], -1, 255, -1)

        valid = (
            (contour_mask > 0)
            & np.isfinite(depth_patch)
            & (depth_patch >= self.depth_min_m)
            & (depth_patch <= self.depth_max_m)
        )
        if np.count_nonzero(valid) < self.bbox_depth_min_pixels:
            return rx, ry, rw, rh, False, None

        surface_depth = float(np.percentile(
            depth_patch[valid], self.bbox_depth_percentile))
        support = valid & (
            np.abs(depth_patch - surface_depth) <= self.bbox_depth_margin_m)
        support_u8 = support.astype(np.uint8)

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            support_u8, connectivity=8)
        if component_count <= 1:
            return rx, ry, rw, rh, False, None

        component_index = 1 + int(np.argmax(
            stats[1:, cv2.CC_STAT_AREA]))
        component_area = int(stats[component_index, cv2.CC_STAT_AREA])
        if component_area < self.bbox_depth_min_pixels:
            return rx, ry, rw, rh, False, None

        sx = int(stats[component_index, cv2.CC_STAT_LEFT])
        sy = int(stats[component_index, cv2.CC_STAT_TOP])
        sw = int(stats[component_index, cv2.CC_STAT_WIDTH])
        sh = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        return rx + sx, ry + sy, sw, sh, True, float(component_area)

    def detect(self, frame_bgr, depth_img) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        depth_m = (self.depth_to_meters(depth_img)
                   if self.use_depth_filter else None)

        if self.use_depth_filter and depth_m is None:
            return {
                'detected': False,
                'candidates': [],
                'debug_infos': [],
                'frame_vis': frame_bgr.copy(),
                'mask': None,
            }

        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi_bgr = frame_bgr[y1:y2, x1:x2].copy()
        roi_depth_m = (
            depth_m[y1:y2, x1:x2].copy()
            if depth_m is not None else None)
        roi_h, roi_w = roi_bgr.shape[:2]

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        debug_infos = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            reasons = []

            if rw <= 0 or rh <= 0:
                reasons.append('invalid_bbox')
                continue

            bbox_depth_refined = False
            refined_area = None
            if self.use_depth_filter:
                rx, ry, rw, rh, bbox_depth_refined, refined_area = (
                    self.refine_bbox_with_depth(
                        cnt, roi_depth_m, rx, ry, rw, rh)
                )
                if refined_area is not None:
                    area = refined_area

            aspect_ratio = rw / float(max(rh, 1))
            bottom_y_ratio = (ry + rh) / float(max(roi_h, 1))

            if area < self.min_area:
                reasons.append(f'area<{self.min_area}')
            if area > self.max_area:
                reasons.append(f'area>{self.max_area}')
            if rw < self.min_width:
                reasons.append(f'width<{self.min_width}')
            if rh < self.min_height:
                reasons.append(f'height<{self.min_height}')
            if aspect_ratio < self.min_aspect_ratio:
                reasons.append(f'aspect<{self.min_aspect_ratio}')
            if aspect_ratio > self.max_aspect_ratio:
                reasons.append(f'aspect>{self.max_aspect_ratio}')
            if bottom_y_ratio < self.min_bottom_y_ratio_in_roi:
                reasons.append(f'bottom_y_ratio<{self.min_bottom_y_ratio_in_roi:.2f}')

            valid_depth_ratio = None
            near_depth_ratio = None
            median_depth = None
            if self.use_depth_filter:
                depth_patch = roi_depth_m[ry:ry + rh, rx:rx + rw]
                color_patch = mask[ry:ry + rh, rx:rx + rw] > 0
                valid_mask = (
                    color_patch
                    & np.isfinite(depth_patch)
                    & (depth_patch > 0.0)
                )
                near_mask = (
                    valid_mask
                    & (depth_patch >= self.depth_min_m)
                    & (depth_patch <= self.depth_max_m)
                )

                total_pixels = max(int(np.count_nonzero(color_patch)), 1)
                valid_depth_ratio = (
                    float(np.count_nonzero(valid_mask)) / float(total_pixels))
                near_depth_ratio = (
                    float(np.count_nonzero(near_mask)) / float(total_pixels))

                if np.any(near_mask):
                    median_depth = float(np.median(depth_patch[near_mask]))

                if valid_depth_ratio < self.min_valid_depth_ratio:
                    reasons.append(
                        f'valid_depth_ratio<{self.min_valid_depth_ratio:.2f}')
                if near_depth_ratio < self.min_near_depth_ratio:
                    reasons.append(
                        f'near_depth_ratio<{self.min_near_depth_ratio:.2f}')

            passed = len(reasons) == 0

            bx1 = x1 + rx
            by1 = y1 + ry
            bx2 = bx1 + rw
            by2 = by1 + rh
            cx = bx1 + rw // 2
            cy = by1 + rh // 2
            top_y_ratio = by1 / float(max(h, 1))

            info = {
                'idx': idx,
                'bbox_roi': (rx, ry, rw, rh),
                'bbox_img': (bx1, by1, bx2, by2),
                'center_img': (cx, cy),
                'area': float(area),
                'aspect_ratio': float(aspect_ratio),
                'bottom_y_ratio': float(bottom_y_ratio),
                'valid_depth_ratio': valid_depth_ratio,
                'near_depth_ratio': near_depth_ratio,
                'median_depth': median_depth,
                'bbox_depth_refined': bbox_depth_refined,
                'top_y_ratio': float(top_y_ratio),
                'passed': passed,
                'reasons': reasons,
            }
            debug_infos.append(info)

            if passed:
                score = area + 100.0 * bottom_y_ratio
                candidates.append(
                    Detection(
                        det_type='blue_obstacle',
                        center_img=(cx, cy),
                        bbox_img=(bx1, by1, bx2, by2),
                        score=float(score),
                        extra={
                            'median_depth': median_depth,
                            'area': float(area),
                            'aspect_ratio': float(aspect_ratio),
                            'bottom_y_ratio': float(bottom_y_ratio),
                            'valid_depth_ratio': valid_depth_ratio,
                            'near_depth_ratio': near_depth_ratio,
                            'bbox_depth_refined': bbox_depth_refined,
                            'top_y_ratio': float(top_y_ratio),
                        }
                    )
                )

        if self.use_depth_filter:
            candidates.sort(
                key=lambda d: (
                    d.extra.get('median_depth')
                    if d.extra.get('median_depth') is not None else 999.0,
                    -d.score
                )
            )
        else:
            candidates.sort(key=lambda d: -d.score)

        frame_vis = frame_bgr.copy()
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for i, det in enumerate(candidates):
            bx1, by1, bx2, by2 = det.bbox_img
            cx, cy = det.center_img
            d = det.extra.get('median_depth')
            top_y_ratio = det.extra.get('top_y_ratio')

            cv2.rectangle(frame_vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            cv2.circle(frame_vis, (cx, cy), 4, (255, 0, 0), -1)

            metric_text = (
                f'd={d:.2f}' if d is not None
                else f'top={top_y_ratio:.3f}')
            cv2.putText(
                frame_vis,
                f'OBS{i} {metric_text}',
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        return {
            'detected': len(candidates) > 0,
            'candidates': candidates,
            'debug_infos': debug_infos,
            'frame_vis': frame_vis,
            'mask': mask,
        }


class YellowDashedLineDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        self.lower_yellow = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_yellow = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        self.open_kernel = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.close_kernel = np.ones(
            (cfg['dash_close_kernel_h'], cfg['dash_close_kernel_w']),
            np.uint8
        )

        self.min_area = cfg['min_area']
        self.max_area = cfg['max_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']

        self.dash_min_segments = cfg['dash_min_segments']
        self.dash_min_total_span_y = cfg['dash_min_total_span_y']
        self.dash_max_adjacent_x_diff = cfg['dash_max_adjacent_x_diff']
        self.dash_max_gap_y = cfg['dash_max_gap_y']
        self.dash_min_gap_y = cfg['dash_min_gap_y']
        self.dash_max_total_x_range = cfg['dash_max_total_x_range']

        self.dash_segment_max_aspect_ratio = cfg['dash_segment_max_aspect_ratio']
        self.dash_segment_max_long_side = cfg['dash_segment_max_long_side']

        self.dash_duplicate_iou_thresh = cfg['dash_duplicate_iou_thresh']
        self.dash_duplicate_center_x_thresh = cfg['dash_duplicate_center_x_thresh']

        self.max_dashed_lines = cfg['max_dashed_lines']

    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi = frame_bgr[y1:y2, x1:x2].copy()
        return (x1, y1, x2, y2), roi

    def _make_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        return mask

    def _is_valid_dash_segment_blob(self, b) -> bool:
        if b['long_side'] > self.dash_segment_max_long_side:
            return False
        if b['aspect_ratio'] > self.dash_segment_max_aspect_ratio:
            return False
        return True

    def _get_all_yellow_blobs(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < self.min_width or h < self.min_height:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect

            long_side = max(rw, rh)
            short_side = max(1.0, min(rw, rh))
            aspect_ratio = long_side / short_side

            blob = {
                'cnt': cnt,
                'area': float(area),
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'cx': float(cx),
                'cy': float(cy),
                'long_side': float(long_side),
                'short_side': float(short_side),
                'aspect_ratio': float(aspect_ratio),
                'angle': float(angle),
                'valid_dash_segment': False,
            }

            blob['valid_dash_segment'] = self._is_valid_dash_segment_blob(blob)
            blobs.append(blob)

        return blobs

    def _get_dash_blobs(self, mask):
        all_blobs = self._get_all_yellow_blobs(mask)
        return [b for b in all_blobs if b['valid_dash_segment']]

    def _build_group_from_start(self, start_idx: int, blobs_sorted: List[Dict[str, Any]]):
        base = blobs_sorted[start_idx]
        group = [base]
        last = base

        for j in range(start_idx + 1, len(blobs_sorted)):
            b = blobs_sorted[j]

            x_diff = abs(b['cx'] - last['cx'])
            gap_y = b['y'] - (last['y'] + last['h'])

            if gap_y < self.dash_min_gap_y:
                continue

            if x_diff <= self.dash_max_adjacent_x_diff and gap_y <= self.dash_max_gap_y:
                group.append(b)
                last = b

        return group

    def _group_to_detection(self, group, rx1: int, ry1: int, roi_h: int) -> Optional[Detection]:
        if len(group) < self.dash_min_segments:
            return None

        min_x = min(b['x'] for b in group)
        min_y = min(b['y'] for b in group)
        max_x = max(b['x'] + b['w'] for b in group)
        max_y = max(b['y'] + b['h'] for b in group)

        total_span_y = max_y - min_y
        total_x_range = max(b['cx'] for b in group) - min(b['cx'] for b in group)
        total_area = sum(b['area'] for b in group)

        if total_span_y < self.dash_min_total_span_y:
            return None

        if total_x_range > self.dash_max_total_x_range:
            return None

        x1 = rx1 + min_x
        y1 = ry1 + min_y
        x2 = rx1 + max_x
        y2 = ry1 + max_y

        cx = rx1 + int((min_x + max_x) / 2)
        cy = ry1 + int((min_y + max_y) / 2)

        bottom_ratio = max_y / float(max(roi_h, 1))

        score = (
                300.0 * len(group)
                + 2.0 * total_span_y
                + 100.0 * bottom_ratio
                + 0.01 * total_area
                - 0.5 * total_x_range
        )

        return Detection(
            det_type='yellow_vertical_dashed_line',
            center_img=(cx, cy),
            bbox_img=(x1, y1, x2, y2),
            score=float(score),
            extra={
                'segments': len(group),
                'total_span_y': float(total_span_y),
                'total_x_range': float(total_x_range),
                'total_area': float(total_area),
                'bottom_ratio': float(bottom_ratio),
                'group_centers': [
                    (float(rx1 + b['cx']), float(ry1 + b['cy']))
                    for b in group
                ],
            }
        )

    def _bbox_iou(self, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))

        return inter_area / float(area_a + area_b - inter_area + 1e-6)

    def _remove_duplicate_dashed(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda d: (
                d.extra.get('total_span_y', 0.0),
                d.extra.get('segments', 0),
                d.score
            ),
            reverse=True
        )

        kept = []

        for det in detections:
            keep = True
            cx = det.center_img[0]

            for old in kept:
                old_cx = old.center_img[0]
                iou = self._bbox_iou(det.bbox_img, old.bbox_img)
                center_x_close = abs(cx - old_cx) <= self.dash_duplicate_center_x_thresh

                if iou >= self.dash_duplicate_iou_thresh or center_x_close:
                    keep = False
                    break

            if keep:
                kept.append(det)

        return kept

    def detect_dashed_lines(self, frame_bgr) -> List[Detection]:
        (rx1, ry1, rx2, ry2), roi = self._roi(frame_bgr)
        roi_h = ry2 - ry1

        mask = self._make_mask(roi)
        blobs = self._get_dash_blobs(mask)

        if len(blobs) < self.dash_min_segments:
            return []

        blobs_sorted = sorted(blobs, key=lambda b: b['cy'])

        raw_detections = []

        for i in range(len(blobs_sorted)):
            group = self._build_group_from_start(i, blobs_sorted)
            det = self._group_to_detection(group, rx1, ry1, roi_h)

            if det is not None:
                raw_detections.append(det)

        if not raw_detections:
            return []

        detections = self._remove_duplicate_dashed(raw_detections)

        detections.sort(
            key=lambda d: (
                d.extra.get('total_span_y', 0.0),
                d.extra.get('segments', 0),
                d.score
            ),
            reverse=True
        )

        return detections

    def detect_top_dashed_lines(self, frame_bgr) -> List[Detection]:
        dashed = self.detect_dashed_lines(frame_bgr)
        return dashed[:self.max_dashed_lines]


class YellowHorizontalLineDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.roi_x_ratio_min = cfg['roi_x_ratio_min']
        self.roi_x_ratio_max = cfg['roi_x_ratio_max']
        self.roi_y_ratio_min = cfg['roi_y_ratio_min']
        self.roi_y_ratio_max = cfg['roi_y_ratio_max']

        self.lower_yellow = np.array(
            [cfg['h_min'], cfg['s_min'], cfg['v_min']],
            dtype=np.uint8
        )
        self.upper_yellow = np.array(
            [cfg['h_max'], cfg['s_max'], cfg['v_max']],
            dtype=np.uint8
        )

        self.open_kernel = np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.close_kernel = np.ones((cfg['close_kernel_h'], cfg['close_kernel_w']), np.uint8)

        self.min_area = cfg['min_area']
        self.min_width = cfg['min_width']
        self.min_height = cfg['min_height']
        self.min_width_ratio = cfg['min_width_ratio']
        self.min_wh_ratio = cfg['min_wh_ratio']
        self.max_tilt_deg = cfg['max_tilt_deg']
        self.center_tolerance_ratio = cfg['center_tolerance_ratio']

    def _roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x1 = int(w * self.roi_x_ratio_min)
        x2 = int(w * self.roi_x_ratio_max)
        y1 = int(h * self.roi_y_ratio_min)
        y2 = int(h * self.roi_y_ratio_max)
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        return (x1, y1, x2, y2), frame_bgr[y1:y2, x1:x2].copy()

    def _signed_line_angle_deg(self, cnt) -> float:
        """
        Use RGB contour fitLine to estimate signed tilt angle relative to image horizontal.
        0 deg means horizontal. The sign is only used by wz correction.
        """
        if cnt is None or len(cnt) < 2:
            return 0.0
        vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx)
        vy = float(vy)
        angle = math.degrees(math.atan2(vy, vx))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return float(angle)

    def detect(self, frame_bgr) -> Optional[Detection]:
        """
        RGB-only horizontal yellow-line detector.
        Distance-to-line is represented by bottom_ratio = line_bottom_y / image_height,
        matching the previous second-stage logic: line_bottom_y >= h * ratio.
        """
        h, w = frame_bgr.shape[:2]
        (rx1, ry1, rx2, ry2), roi = self._roi(frame_bgr)
        roi_h, roi_w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < self.min_width or bh < self.min_height:
                continue

            wh_ratio = bw / float(max(bh, 1))
            if wh_ratio < self.min_wh_ratio:
                continue

            width_ratio = bw / float(max(roi_w, 1))
            if width_ratio < self.min_width_ratio:
                continue

            cx_roi = x + bw / 2.0
            roi_cx = roi_w / 2.0
            center_offset_ratio = abs(cx_roi - roi_cx) / float(max(roi_w, 1))
            if center_offset_ratio > self.center_tolerance_ratio:
                continue

            angle_deg = self._signed_line_angle_deg(cnt)
            abs_tilt_deg = abs(angle_deg)
            if abs_tilt_deg > self.max_tilt_deg:
                continue

            bx1 = rx1 + x
            by1 = ry1 + y
            bx2 = bx1 + bw
            by2 = by1 + bh
            cx = bx1 + bw // 2
            cy = by1 + bh // 2
            bottom_y = by2
            bottom_ratio = bottom_y / float(max(h, 1))

            # Prefer the bottom-most line, same idea as previous yellow stop-line logic.
            score = 3.0 * bottom_y + 0.02 * area + 100.0 * width_ratio - 2.0 * abs_tilt_deg

            candidates.append(
                Detection(
                    det_type='yellow_horizontal_line',
                    center_img=(int(cx), int(cy)),
                    bbox_img=(int(bx1), int(by1), int(bx2), int(by2)),
                    score=float(score),
                    extra={
                        'area': float(area),
                        'angle_deg': float(angle_deg),
                        'abs_tilt_deg': float(abs_tilt_deg),
                        'width_ratio': float(width_ratio),
                        'wh_ratio': float(wh_ratio),
                        'center_offset_ratio': float(center_offset_ratio),
                        'bottom_y': int(bottom_y),
                        'bottom_ratio': float(bottom_ratio),
                    }
                )
            )

        if not candidates:
            return None

        return max(candidates, key=lambda d: d.score)


class Stage4Node(P3TrackVisionMixin, StageNodeBase):

    # ============================================================
    # 全局总调度状态：启动预左移 -> 横向搜索 -> 居中 -> 子流程 -> 完成后左移
    # ============================================================
    # 程序启动后先固定向左移动一段距离，这段时间不识别目标；完成后才进入全局搜索。
    GLOBAL_INITIAL_LATERAL_SHIFT = 'GLOBAL_INITIAL_LATERAL_SHIFT'

    # 全局搜索：向左移动，同时按剩余任务数选择性检测限高杆/障碍物。
    GLOBAL_LATERAL_SEARCH = 'GLOBAL_LATERAL_SEARCH'
    GLOBAL_CENTER_BAR = 'GLOBAL_CENTER_BAR'
    GLOBAL_CENTER_OBSTACLE = 'GLOBAL_CENTER_OBSTACLE'
    GLOBAL_SHIFT_AFTER_SUBTASK = 'GLOBAL_SHIFT_AFTER_SUBTASK'

    # 限高杆子流程状态
    BAR_FORWARD_UNDER = 'BAR_FORWARD_UNDER'
    # LCM 改变机身高度需要一定时间：切到 low 后先零速度原地踏步，
    # 等机身充分降低，再继续向前搜索目标。
    BAR_BODY_LOWER_WAIT = 'BAR_BODY_LOWER_WAIT'
    # 低姿态完成后，先保持水平低姿态向前走一段距离，确保相机/机身越过限高杆。
    BAR_CLEAR_AFTER_UNDER = 'BAR_CLEAR_AFTER_UNDER'
    # 越过限高杆后原地前倾，等待 pitch 渐变到位，再开始目标搜索。
    BAR_FORWARD_PITCH_WAIT = 'BAR_FORWARD_PITCH_WAIT'
    BAR_SEARCH_TARGET = 'BAR_SEARCH_TARGET'
    BAR_APPROACH_TARGET = 'BAR_APPROACH_TARGET'
    BAR_HIT_TARGET = 'BAR_HIT_TARGET'
    BAR_BACKOFF_TO_BAR = 'BAR_BACKOFF_TO_BAR'
    BAR_BACKOFF_TIMED = 'BAR_BACKOFF_TIMED'
    BAR_BACKOFF_VOICE_WAIT = 'BAR_BACKOFF_VOICE_WAIT'
    BAR_TURN_TO_YELLOW = 'BAR_TURN_TO_YELLOW'
    BAR_YELLOW_FORWARD = 'BAR_YELLOW_FORWARD'
    BAR_TURN_BACK = 'BAR_TURN_BACK'
    BAR_FLOW_DONE = 'BAR_FLOW_DONE'

    # 障碍物流程真正结束后的中转状态：用于给全局计数，不直接 DONE
    OBSTACLE_FLOW_DONE = 'OBSTACLE_FLOW_DONE'

    APPROACH_OBSTACLES = 'APPROACH_OBSTACLES'

    # 靠近障碍物并锁定虚线方向后，先按虚线侧做一次预偏移。
    DASH_PRE_SIDE_SHIFT = 'DASH_PRE_SIDE_SHIFT'

    # 新障碍物路线：虚线只负责锁定 left/right，后续不再做虚线闭环对齐。
    # 两次转向前都先进入原地踏步/停稳缓冲状态，避免移动结束后立即转向。
    OBSTACLE_ROUTE_PRE_TURN1_STEP = 'OBSTACLE_ROUTE_PRE_TURN1_STEP'
    OBSTACLE_ROUTE_TURN_1 = 'OBSTACLE_ROUTE_TURN_1'
    OBSTACLE_ROUTE_LATERAL_SCAN = 'OBSTACLE_ROUTE_LATERAL_SCAN'
    OBSTACLE_ROUTE_FORWARD = 'OBSTACLE_ROUTE_FORWARD'
    # 固定前进完成后，在第二次90度转向前按 dashed_side 再横移一小段。
    OBSTACLE_ROUTE_PRE_TURN2_LATERAL = 'OBSTACLE_ROUTE_PRE_TURN2_LATERAL'
    OBSTACLE_ROUTE_PRE_TURN2_STEP = 'OBSTACLE_ROUTE_PRE_TURN2_STEP'
    OBSTACLE_ROUTE_TURN_2 = 'OBSTACLE_ROUTE_TURN_2'

    # 旧虚线闭环状态保留定义仅用于兼容旧日志/代码，不再进入正常状态流。
    ALIGN_DASHED_LINE = 'ALIGN_DASHED_LINE'
    FOLLOW_DASHED_UNTIL_LOST = 'FOLLOW_DASHED_UNTIL_LOST'

    # 新增：虚线消失后的后续任务
    POST_DASH_FORWARD = 'POST_DASH_FORWARD'
    POST_DASH_TURN_1 = 'POST_DASH_TURN_1'
    POST_TURN_FORWARD = 'POST_TURN_FORWARD'
    POST_DASH_TURN_2 = 'POST_DASH_TURN_2'

    # 第二次转向后：复用限高杆代码里的目标检测、对齐、撞击逻辑
    SEARCH_TARGET_AFTER_TURNS = 'SEARCH_TARGET_AFTER_TURNS'
    APPROACH_AND_ALIGN_TARGET = 'APPROACH_AND_ALIGN_TARGET'
    HIT_TARGET = 'HIT_TARGET'

    # Basketball top-edge-triggered upright sequence. These states are intentionally
    # non-blocking so the ROS executor remains responsive while the lower
    # controller performs the action.
    HIT_UPRIGHT_PREPARE = 'HIT_UPRIGHT_PREPARE'
    HIT_UPRIGHT_RISE = 'HIT_UPRIGHT_RISE'
    HIT_UPRIGHT_HOLD = 'HIT_UPRIGHT_HOLD'
    HIT_UPRIGHT_SMOOTH_RETURN = 'HIT_UPRIGHT_SMOOTH_RETURN'
    HIT_UPRIGHT_READY_STAND = 'HIT_UPRIGHT_READY_STAND'
    HIT_UPRIGHT_RECOVERY = 'HIT_UPRIGHT_RECOVERY'

    # 撞击完成后的后续动作：后退 -> 两次左跳 -> 前进识别障碍物并按虚线侧选择单个障碍物对齐
    HIT_BACKOFF_AFTER_HIT = 'HIT_BACKOFF_AFTER_HIT'
    POST_HIT_LEFT_JUMP = 'POST_HIT_LEFT_JUMP'
    APPROACH_SELECTED_OBSTACLE_AFTER_HIT = 'APPROACH_SELECTED_OBSTACLE_AFTER_HIT'
    POST_HIT_OBSTACLE_VOICE_WAIT = 'POST_HIT_OBSTACLE_VOICE_WAIT'

    # 新增：撞击后对齐障碍物到达指定距离后的后续动作
    # 逻辑：按当前对齐的障碍物侧转向 -> 前进一段 -> 反向转回 -> 最后向前走一段
    POST_HIT_OBS_TURN_1 = 'POST_HIT_OBS_TURN_1'
    POST_HIT_OBS_FORWARD = 'POST_HIT_OBS_FORWARD'
    POST_HIT_OBS_TURN_2 = 'POST_HIT_OBS_TURN_2'

    # 新增：第二次转回后，先不识别黄线，按 TF 向前走一段固定距离
    POST_HIT_PRE_FINAL_FORWARD = 'POST_HIT_PRE_FINAL_FORWARD'

    # 前进固定距离后，再进入最终横向黄线识别和朝向修正
    POST_HIT_FINAL_FORWARD = 'POST_HIT_FINAL_FORWARD'

    # 新增：最终对准横向黄线后，再执行两次左跳/等效 180 度掉头动作。
    FINAL_LEFT_JUMP = 'FINAL_LEFT_JUMP'

    # 新增：障碍物流程最终黄线 + 180 度掉头完成后，再恢复 normal 姿态。
    # 注意：不再在 POST_HIT_OBS_TURN_2 后恢复 normal。
    OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN = 'OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN'

    # 全部 2 次限高杆 + 1 次障碍物完成后的最终收尾动作：
    # 右跳一次 -> 前进识别前方横向黄线并矫正朝向 -> 黄线到达图像下方阈值 -> 左跳一次 -> DONE
    GLOBAL_FINAL_RIGHT_JUMP = 'GLOBAL_FINAL_RIGHT_JUMP'
    GLOBAL_FINAL_YELLOW_FORWARD = 'GLOBAL_FINAL_YELLOW_FORWARD'
    GLOBAL_FINAL_LEFT_JUMP = 'GLOBAL_FINAL_LEFT_JUMP'
    GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP = 'GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP'

    # 第四赛段结束后的最终视觉矫正：复用第三赛段结束时的 P3_ALIGN_TRACK 逻辑。
    # 位置与第三赛段结束位置一致，所以同样用 p3_process_yellow_track 得到 p3_s4_lat / p3_s4_yaw。
    GLOBAL_FINAL_P3_ALIGN = 'GLOBAL_FINAL_P3_ALIGN'

    DONE = 'DONE'

    STAGE_ID = 4

    def __init__(self):
        super().__init__('stage4_node', self.STAGE_ID)
        self.p3_declare_params()
        self.p3_load_params()
        self.p3_init_vision_caches()
        self.fourth_stage_init()

        # ============================================================
        # Dedicated RGB receiver
        # ============================================================
        # StageNodeBase created an RGB subscription on the Stage4 node.  Remove
        # it and receive RGB on a tiny helper node with its own executor/thread.
        # This isolates camera ingestion from Stage4 timers, OpenCV, voice, and
        # state-machine callbacks.
        try:
            if getattr(self, 'rgb_sub', None) is not None:
                self.destroy_subscription(self.rgb_sub)
        except Exception as exc:
            self.get_logger().warning(
                f'[P4_RGB_RX] destroy inherited RGB subscription failed: {exc}'
            )
        self.rgb_sub = None

        self._p4_rgb_rx_node = None
        self._p4_rgb_rx_executor = None
        self._p4_rgb_rx_thread = None
        self._p4_rgb_rx_running = False
        self._p4_rgb_rx_local_consumed_seq = -1
        self._p4_rgb_global_seq = 0
        self._p4_rgb_last_rx_monotonic_s = None
        self._p4_rgb_last_restart_monotonic_s = 0.0
        self._p4_rgb_restart_count = 0
        self._p4_start_rgb_receiver()

    def _p4_rgb_executor_loop(self):
        executor = self._p4_rgb_rx_executor
        while self._p4_rgb_rx_running and rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.10)
            except Exception as exc:
                if self._p4_rgb_rx_running:
                    self.get_logger().error(
                        f'[P4_RGB_RX] receiver executor exception: {repr(exc)}'
                    )
                break

    def _p4_start_rgb_receiver(self):
        namespace = self.get_namespace()
        self._p4_rgb_rx_node = _Stage4RgbReceiverNode(
            context=self.context,
            namespace=namespace,
            rgb_topic=self.rgb_topic,
        )
        self._p4_rgb_rx_executor = SingleThreadedExecutor(context=self.context)
        self._p4_rgb_rx_executor.add_node(self._p4_rgb_rx_node)
        self._p4_rgb_rx_running = True
        self._p4_rgb_rx_local_consumed_seq = -1
        self._p4_rgb_rx_thread = threading.Thread(
            target=self._p4_rgb_executor_loop,
            name='stage4_rgb_rx_executor',
            daemon=True,
        )
        self._p4_rgb_rx_thread.start()
        self._p4_rgb_last_restart_monotonic_s = time.monotonic()
        self.get_logger().warning(
            f'[P4_RGB_RX] dedicated receiver started: '
            f'node={namespace}/stage4_rgb_rx topic={self.rgb_topic}'
        )

    def _p4_stop_rgb_receiver(self):
        self._p4_rgb_rx_running = False
        thread = self._p4_rgb_rx_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.25)

        executor = self._p4_rgb_rx_executor
        node = self._p4_rgb_rx_node
        if executor is not None and node is not None:
            try:
                executor.remove_node(node)
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.10)
            except Exception:
                pass

        self._p4_rgb_rx_node = None
        self._p4_rgb_rx_executor = None
        self._p4_rgb_rx_thread = None

    def _p4_restart_rgb_receiver(self, reason: str):
        self._p4_rgb_restart_count += 1
        self.get_logger().error(
            f'[P4_RGB_RX] restarting dedicated receiver: '
            f'count={self._p4_rgb_restart_count}, reason={reason}'
        )
        self._p4_stop_rgb_receiver()
        self._p4_rgb_last_rx_monotonic_s = None
        self._p4_start_rgb_receiver()

    def _p4_sync_rgb_from_receiver(self):
        """Copy the newest helper-node frame into Stage4 at most once per seq."""
        node = self._p4_rgb_rx_node
        if node is None:
            return False

        local_seq, msg, rx_mono = node.snapshot()
        if rx_mono is not None:
            self._p4_rgb_last_rx_monotonic_s = rx_mono

        if msg is None or local_seq == self._p4_rgb_rx_local_consumed_seq:
            return False

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(
                f'[P4_RGB_RX] cv_bridge convert failed: {exc}',
                throttle_duration_sec=1.0,
            )
            return False

        self._p4_rgb_rx_local_consumed_seq = local_seq
        self._p4_rgb_global_seq += 1
        self.latest_rgb_seq = self._p4_rgb_global_seq
        self.latest_rgb_msg = msg
        self.latest_bgr = frame
        return True

    def rgb_age_s(self):
        """Wall-clock age of the last message received by the helper node."""
        last = self._p4_rgb_last_rx_monotonic_s
        if last is None:
            node = self._p4_rgb_rx_node
            if node is not None:
                _seq, _msg, last = node.snapshot()
                if last is not None:
                    self._p4_rgb_last_rx_monotonic_s = last
        if last is None:
            return None
        return max(0.0, time.monotonic() - float(last))

    def on_activated(self):
        self._start_voice_player()
        # Full mission startup already performs RecoveryStand(111) once.
        # Stage4 activation only restores the intended normal body pose through
        # Servo; it must not inject another preset stand action.
        if self.p4_force_refresh_body_at_start:
            self.reset_body_pose_to_normal_at_start()
        else:
            self.set_body_normal(
                do_stop=False,
                reason='stage4 activated',
                force=True,
            )
        self.enter_initial_state()

    def stage_control_loop(self):
        self.fourth_control_loop()
        if self.state == self.DONE:
            self.complete_stage('fourth stage DONE')

    def _start_voice_player(self):
        if not self.voice_enabled or self.voice is not None:
            return
        backend = self.voice_backend
        if backend == 'auto':
            backend = 'ros_offline' if self.platform == 'real' else 'local'
        if self.voice_topic.startswith('/'):
            raise ValueError(
                'voice_topic must be relative so robot_namespace can be applied')
        self.voice = CyberdogVoicePlayer(
            node=self,
            backend=backend,
            voice_dir=self.voice_dir,
            topic=self.voice_topic,
            module_name=self.voice_module_name,
        )
        self.get_logger().info(
            f'[VOICE] Stage 4 player started: platform={self.platform}, '
            f'backend={backend}, topic={self.voice_topic}')

    def _close_voice_player(self):
        voice = self.voice
        self.voice = None
        if voice is None:
            return
        try:
            voice.close()
        except Exception as exc:
            self.get_logger().warning(f'[VOICE] player close failed: {exc}')

    def stop_ctrl(self):
        # Stop accepting queued announcements before relinquishing Stage 4.
        self._close_voice_player()
        self._recover_upright_before_close()
        self._close_upright_lcm_controller(resume_servo=False)
        super().stop_ctrl()

    def fourth_stage_init(self):
        # Fourth stage is merged into this same ROS2 node.
        # Reuse the existing bridge, TF listener, subscriptions, Robot_Ctrl and timer.
        # 机身高度与姿态统一通过 robot_control_cmd LCM 下发。
        real_platform = self.platform == 'real'
        self.declare_parameter('voice_enabled', True)
        self.declare_parameter('voice_dir', '/home/cyberdog_sim/voice')
        self.declare_parameter('voice_backend', 'auto')
        self.declare_parameter('voice_topic', 'speech_play_extend')
        self.declare_parameter('voice_module_name', 'stage4_voice')
        # 需要“停车播报”的事件先保持零速度一小段，再开始播放；
        # 播放器异常时最多等待 voice_wait_timeout_s，避免状态机永久卡住。
        self.declare_parameter('voice_pre_stop_duration_s', 0.20)
        self.declare_parameter('voice_wait_timeout_s', 10.0)

        # Stage4 RGB freshness guard.
        # Visual states process each received RGB message at most once.
        # If no new RGB callback reaches this node for longer than this value,
        # vision-driven motion is stopped instead of reusing a stale frame.
        self.declare_parameter('p4_rgb_stale_stop_s', 0.8)

        # ============================================================
        # 调试用：指定程序启动后的初始状态
        # 默认从 GLOBAL_INITIAL_LATERAL_SHIFT 开始完整第四赛段流程。
        #
        # 注意：
        # 1. 现在大多数“固定走一段 / 转一段”的动作已经改成按仿真时间 duration_s 判断，
        #    不再依赖 TF 距离或 TF yaw。
        # 2. TF 仍然保留在代码中作为兼容/调试，但主动作段应尽量以仿真时间为准。
        # 3. 如果单独从中间状态启动，某些状态依赖之前保存的变量，
        #    例如 dashed_side、locked_target、selected_obstacle_after_hit_side，
        #    这些变量可能为空，需要配合 debug 参数或手动初始化。
        #
        # 可选状态及含义：
        #
        # ------------------------------------------------------------
        # 一、全局任务调度状态
        # ------------------------------------------------------------
        #
        # GLOBAL_INITIAL_LATERAL_SHIFT:
        #   第四赛段启动后的预横移状态。
        #   先按 global_initial_lateral_shift_vy 横向移动
        #   global_initial_lateral_shift_duration_s 秒。
        #   这段时间不检测限高杆和障碍物，目的是避免一启动就在原地误触发。
        #   完成后进入 GLOBAL_LATERAL_SEARCH。
        #
        # GLOBAL_LATERAL_SEARCH:
        #   全局搜索状态。
        #   机器狗一边按 global_lateral_search_vy 横向移动，一边根据任务完成情况选择性检测目标：
        #     - completed_bar_count < required_bar_count 时，检测限高杆；
        #     - completed_obstacle_count < required_obstacle_count 时，检测蓝色障碍物；
        #     - 已完成的任务类型不再检测，避免重复触发。
        #   如果检测到限高杆，进入 GLOBAL_CENTER_BAR。
        #   如果检测到蓝色障碍物，进入 GLOBAL_CENTER_OBSTACLE。
        #   如果所有任务已经完成，则进入 GLOBAL_FINAL_RIGHT_JUMP，开始全局最终收尾。
        #
        # GLOBAL_CENTER_BAR:
        #   全局搜索中识别到限高杆后，先横向居中限高杆。
        #   根据限高杆中心和图像中心的误差控制 vy。
        #   连续稳定 global_center_stable_frames 帧后，进入 BAR_FORWARD_UNDER。
        #
        # GLOBAL_CENTER_OBSTACLE:
        #   全局搜索中识别到蓝色障碍物后，先横向居中障碍物。
        #   根据障碍物中心和图像中心的误差控制 vy。
        #   连续稳定 global_center_stable_frames 帧后，进入 APPROACH_OBSTACLES。
        #
        # GLOBAL_SHIFT_AFTER_SUBTASK:
        #   完成一次子任务后的全局横移状态。
        #   如果限高杆/障碍物任务还没有全部完成，就继续横移一段时间后回到 GLOBAL_LATERAL_SEARCH。
        #   横移时间通常使用：
        #     - global_after_task_shift_duration_s
        #   如果刚完成的是障碍物流程，会根据 dashed_side 选择不同横移时间：
        #     - global_after_obstacle_shift_duration_left_dash_s
        #     - global_after_obstacle_shift_duration_right_dash_s
        #   如果全部任务已经完成，则不再继续搜索，而是进入 GLOBAL_FINAL_RIGHT_JUMP。
        #
        #
        # ------------------------------------------------------------
        # 二、限高杆流程
        # ------------------------------------------------------------
        #
        # BAR_FORWARD_UNDER:
        #   限高杆居中后，机器狗向前低身通过限高杆。
        #   主要根据限高杆深度/触发距离判断是否进入后续目标搜索。
        #   到达 bar_trigger_distance_m 或满足对应条件后，进入 BAR_SEARCH_TARGET。
        #
        # BAR_CLEAR_AFTER_UNDER（固定时间低姿态前进，不使用 TF）：
        #   机身降低完成后，保持水平低姿态继续向前一小段距离，确保越过限高杆。
        #   优先按 TF 位移判断；TF 不可用时使用超时兜底。
        #
        # BAR_FORWARD_PITCH_WAIT:
        #   越过限高杆后停止前进，通过 LCM 设置前倾 pitch，并原地等待姿态到位。
        #   这样固定在机身上的相机视线会向下，减小机器狗前下方盲区。
        #
        # BAR_SEARCH_TARGET:
        #   前倾完成后开始搜索目标物体。
        #   目标包括：
        #     - blue_ball
        #     - white_ball
        #     - cola
        #   检测到目标后锁定 locked_target，并播放对应语音，然后进入 BAR_APPROACH_TARGET。
        #
        # BAR_APPROACH_TARGET:
        #   锁定目标后，边向前走边根据目标中心做横向对齐。
        #   使用目标中心 x 与图像中心误差控制 vy。
        #   真机按目标视觉面积/半径触发，仿真按深度触发 BAR_HIT_TARGET。
        #
        # BAR_HIT_TARGET:
        #   按目标类型执行撞击动作。
        #   不同目标可以有不同撞击速度和持续时间：
        #     - hit_blue_ball_speed / hit_blue_ball_duration_s
        #     - hit_white_ball_speed / hit_white_ball_duration_s
        #     - hit_cola_speed / hit_cola_duration_s
        #   撞击完成后进入 BAR_BACKOFF_TO_BAR。
        #
        # BAR_BACKOFF_TIMED -> BAR_BACKOFF_TO_BAR:
        #   撞击目标后先按 bar_backoff_duration_s 固定时间盲退，
        #   这段时间完全不检测限高杆，也不做视觉横向/朝向纠偏。
        #   固定盲退结束后进入 BAR_BACKOFF_TO_BAR，继续后退并开始检测限高杆。
        #   一旦重新识别到限高杆，立即停车并进入 BAR_BACKOFF_VOICE_WAIT 播放 bar.wav。
        #
        # BAR_TURN_TO_YELLOW -> BAR_YELLOW_FORWARD -> BAR_TURN_BACK:
        #   回退完成后复用障碍物流程的 180 度转向参数；随后前进识别横向黄线，
        #   复用 final_yellow_* 参数停车，再执行一次 180 度转向。
        #   BAR_FLOW_DONE 才增加 completed_bar_count 并进入后续全局流程。
        #
        #
        # ------------------------------------------------------------
        # 三、障碍物 + 黄色虚线流程
        # ------------------------------------------------------------
        #
        # APPROACH_OBSTACLES:
        #   向前靠近两个蓝色障碍物，并用两个障碍物中点做横向修正。
        #   这一状态同时运行虚线检测器，只用于一次性锁定 dashed_side(left/right)。
        #   到达靠近阈值且方向已锁定后进入 DASH_PRE_SIDE_SHIFT。
        #
        # 新路线后续不再做虚线精对齐/沿虚线跟随：
        #   DASH_PRE_SIDE_SHIFT
        #     -> OBSTACLE_ROUTE_PRE_TURN1_STEP（原地踏步停稳）
        #     -> OBSTACLE_ROUTE_TURN_1（按真机实测符号：left=-wz / right=+wz）
        #     -> OBSTACLE_ROUTE_LATERAL_SCAN（按 dashed_side 横移，只检测单个蓝色障碍物边缘）
        #     -> OBSTACLE_ROUTE_FORWARD（固定时间前进）
        #     -> OBSTACLE_ROUTE_PRE_TURN2_LATERAL（按 dashed_side 再横移一段，不做视觉）
        #     -> OBSTACLE_ROUTE_PRE_TURN2_STEP（原地踏步停稳）
        #     -> OBSTACLE_ROUTE_TURN_2（与第一次相反的90°）
        #     -> SEARCH_TARGET_AFTER_TURNS（边前进边开始目标检测）。
        #
        # DASH_PRE_SIDE_SHIFT:
        #   第一次识别到黄色竖直虚线后，不马上开始精对齐，
        #   而是先朝虚线所在方向横移 dashed_pre_shift_duration_s 秒。
        #   横移速度由 dashed_pre_shift_speed 控制。
        #   这样可以让机器狗更接近后续对齐位置。
        #   单独从该状态启动时，需要通过 debug_dashed_side 指定虚线方向。
        #
        # ALIGN_DASHED_LINE:
        #   对黄色竖直虚线做偏置对齐。
        #   不是把虚线对到图像正中心，而是根据 dashed_target_offset_px 做偏置：
        #     - 虚线在左边：目标点 = 图像中心 + offset，让虚线位于中间偏右；
        #     - 虚线在右边：目标点 = 图像中心 - offset，让虚线位于中间偏左。
        #   对齐稳定 dashed_center_stable_frames 帧后，进入 FOLLOW_DASHED_UNTIL_LOST。
        #
        # FOLLOW_DASHED_UNTIL_LOST:
        #   沿黄色竖直虚线继续向前走。
        #   行走过程中持续检测虚线，并用较小 vy 进行横向修正。
        #   当虚线连续丢失 dashed_lost_stop_frames 帧后，认为已经通过虚线区域，
        #   进入 POST_DASH_FORWARD。
        #
        # POST_DASH_FORWARD:
        #   虚线消失后继续向前走 post_dash_forward_duration_s 秒。
        #   速度由 post_dash_forward_speed 控制。
        #   完成后进入 POST_DASH_TURN_1。
        #
        # POST_DASH_TURN_1:
        #   虚线消失后的第一次转向，按仿真时间执行。
        #   转向持续 post_dash_turn_duration_s 秒，角速度为 post_dash_turn_wz。
        #   转向方向由 dashed_side 决定：
        #     - dashed_side == 'left'  时右转；
        #     - dashed_side == 'right' 时左转。
        #   完成后进入 POST_TURN_FORWARD。
        #
        # POST_TURN_FORWARD:
        #   第一次转向后继续向前走 post_turn_forward_duration_s 秒。
        #   这段前进分成两段：
        #     - 前 post_turn_forward_fast_duration_s 秒使用 post_turn_forward_fast_speed 快速前进；
        #     - 剩余时间使用 post_turn_forward_slow_speed 慢速前进，
        #       让第二次转向前的姿态和位置更稳定。
        #   完成后进入 POST_DASH_TURN_2。
        #
        # POST_DASH_TURN_2:
        #   第二次转向，方向与 POST_DASH_TURN_1 相反。
        #   转向持续 post_second_turn_duration_s 秒，角速度为 post_second_turn_wz。
        #   完成后进入 SEARCH_TARGET_AFTER_TURNS。
        #
        #
        # ------------------------------------------------------------
        # 四、虚线后目标识别、撞击、回退、再绕障碍物
        # ------------------------------------------------------------
        #
        # SEARCH_TARGET_AFTER_TURNS:
        #   第二次转向后开始搜索目标物体：
        #     - blue_ball
        #     - white_ball
        #     - cola
        #   检测到目标后锁定 locked_target，并播放语音。
        #   然后进入 APPROACH_AND_ALIGN_TARGET。
        #
        # APPROACH_AND_ALIGN_TARGET:
        #   锁定目标后，边向前走边根据目标中心做横向对齐。
        #   真机按目标视觉面积/半径触发，仿真按深度触发；目标检测仍要求稳定帧数。
        #   进入 HIT_TARGET。
        #
        # HIT_TARGET:
        #   根据 locked_target 类型执行对应撞击动作。
        #   撞击速度和持续时间由目标类型决定：
        #     - blue_ball 使用 hit_blue_ball_speed / hit_blue_ball_duration_s；
        #     - white_ball 使用 hit_white_ball_speed / hit_white_ball_duration_s；
        #     - cola 使用 hit_cola_speed / hit_cola_duration_s。
        #   撞击完成后进入 HIT_BACKOFF_AFTER_HIT。
        #
        # HIT_BACKOFF_AFTER_HIT:
        #   撞击完成后，按 after_hit_backoff_speed 后退
        #   after_hit_backoff_duration_s 秒。
        #   完成后进入 POST_HIT_LEFT_JUMP。
        #
        # POST_HIT_LEFT_JUMP:
        #   后退完成后执行 after_hit_left_jump_count 次原地左跳。
        #   通常是两次左跳，用于调整朝向。
        #   完成后进入 APPROACH_SELECTED_OBSTACLE_AFTER_HIT。
        #
        # APPROACH_SELECTED_OBSTACLE_AFTER_HIT:
        #   左跳完成后继续向前走，并重新识别蓝色障碍物。
        #   根据之前记录的 dashed_side 选择一个障碍物进行对齐：
        #     - dashed_side == 'left'  时，对齐右边障碍物；
        #     - dashed_side == 'right' 时，对齐左边障碍物。
        #   真机按选中障碍物框上沿比例触发，仿真按深度触发，
        #   先进入 POST_HIT_OBSTACLE_VOICE_WAIT 停车播放 obstacle.wav，
        #   播报结束后再进入 POST_HIT_OBS_TURN_1。
        #
        # POST_HIT_OBS_TURN_1:
        #   对齐选中障碍物后进行第一次转向，按仿真时间执行。
        #   如果当前对齐的是左边障碍物，则左转；
        #   如果当前对齐的是右边障碍物，则右转。
        #   持续 post_hit_obs_turn_duration_s 秒，角速度为 post_hit_obs_turn_wz。
        #   完成后进入 POST_HIT_OBS_FORWARD。
        #
        # POST_HIT_OBS_FORWARD:
        #   第一次转向完成后向前走 post_hit_obs_forward_duration_s 秒。
        #   速度由 post_hit_obs_forward_speed 控制。
        #   完成后进入 POST_HIT_OBS_TURN_2。
        #
        # POST_HIT_OBS_TURN_2:
        #   第二次转向，方向与 POST_HIT_OBS_TURN_1 相反。
        #   持续 post_hit_obs_turn_duration_s 秒。
        #   完成后进入 POST_HIT_PRE_FINAL_FORWARD。
        #
        # POST_HIT_PRE_FINAL_FORWARD:
        #   第二次转回后，先不识别横向黄线，只按仿真时间向前走一小段。
        #   持续 post_hit_final_forward_duration_s 秒，
        #   速度由 post_hit_final_forward_speed 控制。
        #   完成后进入 POST_HIT_FINAL_FORWARD。
        #
        # POST_HIT_FINAL_FORWARD:
        #   障碍物流程内部的横向黄线收尾状态。
        #   开始边前进边识别前方横向黄线，并根据黄线 angle_deg 修正 wz，
        #   让机器狗尽量正对黄线。
        #   当黄线底部 bottom_ratio 到达 final_yellow_stop_line_y_ratio，
        #   且角度误差小于 final_yellow_done_tilt_deg 后，
        #   进入 FINAL_LEFT_JUMP。
        #   注意：这个状态是障碍物流程内部收尾，不是全局最终收尾。
        #
        # FINAL_LEFT_JUMP:
        #   障碍物流程内部黄线收尾完成后，执行原地左跳。
        #   当前逻辑通常执行两次左跳。
        #   完成后进入 OBSTACLE_FLOW_DONE，而不是直接 DONE。
        #
        # OBSTACLE_FLOW_DONE:
        #   障碍物流程真正完成的计数状态。
        #   在这里 completed_obstacle_count 加 1。
        #   然后根据总任务是否完成决定：
        #     - 如果限高杆和障碍物任务都完成，进入 GLOBAL_FINAL_RIGHT_JUMP；
        #     - 否则进入 GLOBAL_SHIFT_AFTER_SUBTASK，继续搜索剩余任务。
        #
        #
        # ------------------------------------------------------------
        # 五、全局最终收尾流程
        # ------------------------------------------------------------
        #
        # GLOBAL_FINAL_RIGHT_JUMP:
        #   当 required_bar_count 次限高杆流程和 required_obstacle_count 次障碍物流程全部完成后，
        #   进入全局最终收尾。
        #   该状态先执行一次右跳，用于调整进入最终出口的方向。
        #   完成后进入 GLOBAL_FINAL_YELLOW_FORWARD。
        #
        # GLOBAL_FINAL_YELLOW_FORWARD:
        #   全局最后的前方横向黄线识别和对正状态。
        #   机器狗按 global_final_yellow_forward_speed 向前走，
        #   同时识别前方横向黄线，并根据黄线 angle_deg 修正 wz。
        #   当黄线底部 bottom_ratio 到达 global_final_yellow_stop_line_y_ratio 后，
        #   不立刻停止，而是继续向前走，等待黄线从图像中消失。
        #   当黄线连续消失 global_final_yellow_disappear_confirm_count 帧后，
        #   认为机器狗已经越过最终黄线，进入 GLOBAL_FINAL_LEFT_JUMP。
        #
        # GLOBAL_FINAL_LEFT_JUMP:
        #   全局最后的左跳状态。
        #   执行一次左跳后进入 DONE。
        #
        # DONE:
        #   第四赛段全部任务结束。
        #   持续发送 STOP。
        # ============================================================
        # 也可以写具名调试入口（bar / obstacle / target / final …），
        # 见 p4_entry_table()。统一的 entry_point 参数（launch 里的
        # stage4_entry）优先于这一个。
        self.declare_parameter('p4_initial_state', 'default')


        # 调试用：当 initial_state 需要依赖 dashed_side 时，可手动指定。
        # 可选值：'auto' / 'left' / 'right'
        # auto 表示正常流程里由视觉第一次看到虚线时自动记录。
        self.declare_parameter('debug_dashed_side', 'auto')
        # 自动识别虚线方向时，需要连续多少帧检测为同一侧才锁定。
        # 设为 1 表示一帧锁定；真机可改为 2 或 3 过滤偶发误检。
        self.declare_parameter('dashed_side_confirm_frames', 1)

        # ============================================================
        # 全局整合流程参数
        # required_bar_count=2：限高杆流程需要完成两次
        # required_obstacle_count=1：障碍物流程只需要完成一次；完成后后续不再触发障碍物流程
        # ============================================================
        self.declare_parameter('required_bar_count', 2)
        self.declare_parameter('required_obstacle_count', 1)

        # 程序开始先固定向左移动一段仿真时间，不检测限高杆/障碍物，避免一启动就在原地误触发。
        self.declare_parameter('global_initial_lateral_shift_duration_s', 1.5)
        self.declare_parameter('global_initial_lateral_shift_vy', 0.30)

        # 启动预左移完成后，进入全局搜索；搜索阶段会按完成计数决定是否检测限高杆/障碍物。
        self.declare_parameter('global_lateral_search_vy', 0.30)
        self.declare_parameter('global_center_stable_frames', 3)
        self.declare_parameter('global_after_task_shift_vy', 0.30)
        self.declare_parameter('global_after_task_shift_duration_s', 1.5)
        # 障碍物流程完成后，如果虚线在右边，左移持续时间更长
        self.declare_parameter('global_after_obstacle_shift_duration_left_dash_s', 0.1)
        self.declare_parameter('global_after_obstacle_shift_duration_right_dash_s', 3.0)

        # 完成一次限高杆流程后，下一轮全局搜索只在图像左半边寻找目标。
        # 这样可以避免刚完成的限高杆仍出现在右半边时被重复选中。
        self.declare_parameter('search_left_half_after_bar_done', True)
        self.declare_parameter('after_bar_search_x_ratio_max', 0.50)
        # 完成障碍物流程后，下一轮全局搜索也只在图像左半边寻找目标。
        # 这样可以避免刚完成的障碍物/绕行区域仍出现在右侧时，影响下一个目标选择。
        self.declare_parameter('search_left_half_after_obstacle_done', True)
        self.declare_parameter('after_obstacle_search_x_ratio_max', 0.70)

        # 限高杆流程参数
        self.declare_parameter(
            'global_bar_center_fixed_vy', 0.10 if real_platform else 0.20)
        self.declare_parameter(
            'global_bar_center_px_deadband', 12 if real_platform else 7)
        self.declare_parameter(
            'bar_search_forward_speed', 0.40 if real_platform else 0.60)
        self.declare_parameter(
            'bar_search_near_forward_speed', 0.20 if real_platform else 0.60)
        self.declare_parameter(
            'bar_search_slow_distance_m', 0.95 if real_platform else 0.50)
        self.declare_parameter(
            'bar_trigger_distance_m', 0.85 if real_platform else 0.50)
        # 真机使用 RGB 外接框上沿比例判断接近程度；仿真继续使用深度。
        self.declare_parameter('use_rgb_distance_triggers', real_platform)
        self.declare_parameter('bar_search_slow_top_y_ratio', 0.18)
        self.declare_parameter('bar_trigger_top_y_ratio', 0.08)
        self.declare_parameter(
            'bar_trigger_confirm_frames', 2 if real_platform else 1)
        self.declare_parameter(
            'bar_align_vy_k', 0.22 if real_platform else 0.35)
        self.declare_parameter(
            'bar_align_vy_max', 0.16 if real_platform else 0.30)
        self.declare_parameter(
            'bar_align_vy_min', 0.08 if real_platform else 0.10)
        self.declare_parameter(
            'bar_center_px_deadband', 12 if real_platform else 7)
        self.declare_parameter('bar_center_stable_frames', 3)
        # 真机 RGB 与原始深度没有逐像素对齐。限高杆取样带里同时出现
        # 横杆和背景时，先按深度间隔分簇，再选择最近且像素数足够的簇。
        # 仿真保持原来的 20 分位数算法，避免改变已验证的仿真行为。
        self.declare_parameter('bar_depth_cluster_enabled', real_platform)
        self.declare_parameter('bar_depth_cluster_gap_m', 0.30)
        self.declare_parameter('bar_depth_cluster_min_pixels', 6)
        self.declare_parameter('bar_depth_cluster_min_ratio', 0.05)
        # 限高杆朝向矫正：居中和穿杆时，根据限高杆左右两侧深度差给一个小 wz。
        # left_depth / right_depth 差值越明显，说明机器狗相对限高杆越斜。
        self.declare_parameter(
            'bar_depth_yaw_align_enabled', not real_platform)
        self.declare_parameter(
            'bar_depth_yaw_fixed_wz', 0.06 if real_platform else 0.12)
        self.declare_parameter('bar_depth_yaw_deadband_m', 0.03)
        self.declare_parameter('bar_depth_yaw_sample_x_ratio', 0.30)
        self.declare_parameter('bar_depth_yaw_sample_y_ratio', 0.10)
        self.declare_parameter('bar_depth_yaw_sample_half_size', 4)
        # 如果实测发现越修越歪，把这个参数改成 -1。
        self.declare_parameter('bar_depth_yaw_sign', -1.0)
        # =========================
        # 第四赛段机身高度策略
        # =========================
        # 默认第四赛段搜索/全局移动保持 normal；
        # 限高杆到达 bar_trigger_distance_m 触发播报时再切 low；
        # 障碍物流程在 GLOBAL_CENTER_OBSTACLE 对齐完成后切 low；
        # 不是在 POST_HIT_OBS_TURN_2 后恢复 normal，
        # 而是在后续前进识别横向黄线，并完成 FINAL_LEFT_JUMP/等效 180 度掉头后恢复 normal。
        self.declare_parameter('p4_normal_body_height', 0.25)
        self.declare_parameter('bar_low_body_height', 0.20)
        self.declare_parameter('obstacle_low_body_height', 0.20)
        self.declare_parameter('bar_body_low_enabled', True)
        self.declare_parameter('bar_body_low_do_stop', True)
        # 切换到 low 后，以 mode=11/gait_id=3、零速度原地保持一段时间。
        # 这段等待使用仿真时间；真机上 use_sim_time=False 时使用 ROS 时钟。
        self.declare_parameter('bar_body_lower_wait_s', 3.0)

        # 低姿态到位后，不立刻前倾：先保持水平低姿态按固定时间向前，确保越过限高杆。
        # 这里完全不使用 TF 位移，便于仿真与真机保持一致。
        self.declare_parameter(
            'bar_clear_forward_speed', 0.25 if real_platform else 0.35)
        self.declare_parameter('bar_clear_forward_time_s', 0.0)

        # 越过限高杆后，为减小真机相机前下方盲区而设置的前倾姿态。
        # 真机测试中正 pitch 对应机身前倾、相机向下看，因此这里设置为 0.20 rad。
        self.declare_parameter('bar_target_forward_pitch', 0.15)
        self.declare_parameter('bar_target_pitch_wait_s', 0.5)

        # 障碍物流程使用独立低姿态。虚线阶段与目标识别阶段的前倾角度分开控制。
        self.declare_parameter('dash_forward_pitch', 0.15)
        self.declare_parameter('obstacle_target_forward_pitch', 0.15)
        self.declare_parameter('bar_restore_level_before_backoff', False)  # 兼容旧配置；当前回退始终保持前倾

        self.declare_parameter('restore_normal_after_bar_flow', True)
        self.declare_parameter('obstacle_flow_low_enabled', True)
        self.declare_parameter('obstacle_body_low_do_stop', True)
        self.declare_parameter('obstacle_restore_normal_after_final_turn', True)
        self.declare_parameter('p4_force_refresh_body_at_start', True)

        # 运动参数
        self.declare_parameter('obstacle_forward_speed', 0.20)
        self.declare_parameter('obstacle_search_forward_speed', 0.20)
        self.declare_parameter('obstacle_trigger_distance_m', 0.30)
        # 靠近障碍物时，距离小于该阈值后让机身前倾，扩大前下方视野。
        self.declare_parameter('obstacle_approach_pitch_distance_m', 0.65)
        self.declare_parameter('obstacle_approach_pitch_bottom_y_ratio', 0.75)
        self.declare_parameter('obstacle_trigger_bottom_y_ratio', 0.90)
        self.declare_parameter('obstacle_approach_forward_pitch', 0.15)

        self.declare_parameter(
            'obstacle_align_vy_k', 0.22 if real_platform else 0.35)
        self.declare_parameter(
            'obstacle_align_vy_max', 0.18 if real_platform else 0.30)
        self.declare_parameter(
            'obstacle_align_vy_min', 0.08 if real_platform else 0.20)
        self.declare_parameter(
            'obstacle_center_px_deadband', 12 if real_platform else 7)

        self.declare_parameter(
            'dashed_align_vy_k', 0.14 if real_platform else 0.35)
        self.declare_parameter(
            'dashed_align_vy_max', 0.10 if real_platform else 0.30)
        self.declare_parameter(
            'dashed_align_vy_min', 0.04 if real_platform else 0.20)
        self.declare_parameter(
            'dashed_center_px_deadband', 15 if real_platform else 7)
        self.declare_parameter('dashed_center_stable_frames', 1)

        # 黄线开始对齐前，先朝虚线所在方向横移一小段仿真时间。
        # 预偏移后退速度保留为参数；当前设为 0，仅做横移。
        self.declare_parameter('dashed_pre_shift_speed', 0.20)
        self.declare_parameter('dashed_pre_shift_backward_speed', 0.0)
        self.declare_parameter('dashed_pre_shift_duration_s', 4.0)

        # =========================
        # 新障碍物路线参数
        # =========================
        # 真机方向映射：预偏移完成后，dashed_side=left 使用负向转90度，right 使用正向转90度。
        self.declare_parameter('obstacle_route_turn_duration_s', 3.7)
        self.declare_parameter('obstacle_route_turn_wz', 0.60)
        # 每次90度转向前先原地踏步/停稳一小段时间，避免横移或前进结束后立刻转向。
        self.declare_parameter('obstacle_route_pre_turn_step_duration_s', 0.5)

        # 第一次90度转向后，按 dashed_side 对应方向横移，同时给一个小的后退速度，
        # 避免横移时继续贴近前方障碍物；这一阶段只检测蓝色障碍物。
        self.declare_parameter('obstacle_route_lateral_speed', 0.10)
        self.declare_parameter('obstacle_route_lateral_backward_speed', 0)
        # left 路线看障碍物 bbox 左边缘：x1/W >= 该阈值时停止横移。
        # right 路线镜像看 bbox 右边缘：x2/W <= 该阈值时停止横移。
        self.declare_parameter('obstacle_route_left_edge_trigger_ratio', 0.50)
        self.declare_parameter('obstacle_route_right_edge_trigger_ratio', 0.50)
        self.declare_parameter('obstacle_route_edge_confirm_frames', 2)

        # 单障碍物边缘到阈值后，先固定向前一段时间。
        self.declare_parameter('obstacle_route_forward_speed', 0.20)
        self.declare_parameter('obstacle_route_forward_duration_s', 2.5)

        # 第二次90度转向之前再做一次纯定时横移：
        # dashed_side=left -> 左移；dashed_side=right -> 右移。
        # 这一段不运行视觉检测并保持前倾；横移结束后进入 PRE_TURN2_STEP，在原地踏步时恢复 pitch=0。
        self.declare_parameter('obstacle_route_pre_turn2_lateral_speed', 0.10)
        self.declare_parameter('obstacle_route_pre_turn2_lateral_duration_s', 1.0)

        # ALIGN_DASHED_LINE 中短暂丢失虚线时仍按已锁定方向继续横移，
        # 但不要再复用预偏移速度，避免漏检期间快速冲过目标位置。
        self.declare_parameter('dashed_align_lost_vy_speed', 0.03)

        # 偏置对齐目标，单位：像素
        # 左边有虚线：目标点 = 图像中心 + offset，也就是中间偏右
        # 右边有虚线：目标点 = 图像中心 - offset，也就是中间偏左
        self.declare_parameter('dashed_target_offset_px', 30)

        self.declare_parameter('follow_forward_speed', 0.50)
        self.declare_parameter('follow_align_vy_k', 0.18)
        # FOLLOW_DASHED_UNTIL_LOST 阶段单独使用的横向速度上下限，
        # 不再和 ALIGN_DASHED_LINE 共用 dashed_align_vy_min / dashed_align_vy_max。
        self.declare_parameter('follow_align_vy_max', 0.15)
        self.declare_parameter('follow_align_vy_min', 0.05)
        self.declare_parameter('dashed_lost_stop_frames', 2)

        # 沿虚线前进阶段的有效识别范围。
        # 只有虚线中心 x 落在 get_dashed_target_x() 附近这个范围内，才认为虚线仍然有效。
        # 超过范围，即使视觉检测到了虚线，也按虚线消失处理。
        self.declare_parameter('follow_dashed_valid_x_range_px',50)

        # =========================
        # 虚线消失后的后续任务参数
        # =========================
        self.declare_parameter('tf_parent_frame', 'vodom')
        self.declare_parameter('tf_child_frame', 'base_link')

        # 虚线识别不到后，继续向前走一小段仿真时间
        self.declare_parameter('post_dash_forward_duration_s', 2.0)
        self.declare_parameter('post_dash_forward_speed', 0.20)

        # 第一次转向持续时间。左边虚线 -> 右转；右边虚线 -> 左转
        self.declare_parameter('post_dash_turn_duration_s', 3.85)
        self.declare_parameter('post_dash_turn_wz', 0.60)
        self.declare_parameter('post_dash_turn_tolerance_deg', 1.5)

        # 第一次转向完成后，继续前进一段仿真时间。
        # 这段前进分成两段：先快走，再慢走。
        self.declare_parameter('post_turn_forward_duration_s', 2.0)
        self.declare_parameter('post_turn_forward_fast_duration_s', 1.6)
        self.declare_parameter('post_turn_forward_fast_speed', 0.60)
        self.declare_parameter('post_turn_forward_slow_speed', 0.20)

        # 第二次转向持续时间，方向与第一次相反
        self.declare_parameter('post_second_turn_duration_s', 3.85)
        self.declare_parameter('post_second_turn_wz', 0.60)
        self.declare_parameter('post_second_turn_tolerance_deg', 1.5)

        # =========================
        # 第二次转向后的目标检测 / 对齐 / 撞击参数
        # 复用限高杆任务代码里的目标检测逻辑：蓝球、白球、可乐
        # =========================
        self.declare_parameter('target_search_forward_speed', 0.30)
        # 目标搜索阶段：只有持续稳定识别目标足够长时间后，才允许丢失时后退重找。
        # 这个确认与 target_stable_frames 分离：target_stable_frames 仍只负责进入目标对齐状态。
        self.declare_parameter('target_search_backward_speed', 0.15)
        self.declare_parameter('target_seen_confirm_frames', 8)
        self.declare_parameter('align_forward_speed_far', 0.15)
        self.declare_parameter(
            'align_forward_speed_near', 0.1 if real_platform else 0.30)
        self.declare_parameter(
            'align_vy_k', 0.22 if real_platform else 0.35)
        self.declare_parameter(
            'align_vy_max', 0.16 if real_platform else 0.30)
        self.declare_parameter(
            'align_vy_min', 0.08 if real_platform else 0.15)
        self.declare_parameter('target_stable_frames', 3)
        self.declare_parameter('hit_trigger_distance_m', 0.35)
        # Basketball is detected only by the high-mounted AI camera. Keep the
        # central ROI and top-edge thresholds configurable for field tuning.
        self.declare_parameter('basketball_ai_roi_x_min_ratio', 0.20)
        self.declare_parameter('basketball_ai_roi_x_max_ratio', 0.80)
        self.declare_parameter('basketball_ai_roi_y_min_ratio', 0.05)
        self.declare_parameter('basketball_ai_roi_y_max_ratio', 0.90)
        self.declare_parameter('basketball_ai_max_age_s', 0.50)
        self.declare_parameter('basketball_top_slow_y_ratio', 0.35)
        self.declare_parameter('basketball_top_trigger_y_ratio', 0.25)
        self.declare_parameter('basketball_top_trigger_confirm_frames', 3)
        # 真机目标接近度：可乐只看瓶盖面积；球同时满足面积和半径。
        self.declare_parameter('cola_slow_cap_area_ratio', 0.0015)
        self.declare_parameter('cola_hit_cap_area_ratio', 0.0060)
        self.declare_parameter('blue_ball_slow_area_ratio', 0.015)
        self.declare_parameter('blue_ball_hit_area_ratio', 0.040)
        self.declare_parameter('blue_ball_slow_radius_px', 35.0)
        self.declare_parameter('blue_ball_hit_radius_px', 60.0)
        self.declare_parameter('white_ball_slow_area_ratio', 0.015)
        self.declare_parameter('white_ball_hit_area_ratio', 0.040)
        self.declare_parameter('white_ball_slow_radius_px', 35.0)
        self.declare_parameter('white_ball_hit_radius_px', 60.0)
        self.declare_parameter(
            'center_px_deadband', 12 if real_platform else 7)

        self.declare_parameter('hit_blue_ball_speed', 0.30)
        self.declare_parameter('hit_blue_ball_duration_s', 1.5)
        self.declare_parameter('hit_white_ball_speed', 0.30)
        self.declare_parameter('hit_white_ball_duration_s', 1.5)
        self.declare_parameter('hit_cola_speed', 0.30)
        self.declare_parameter('hit_cola_duration_s', 1.5)
        self.declare_parameter('hit_timeout_s', 10.0)

        # Basketball only: when its AI-camera top edge reaches the threshold,
        # stand on the rear legs, hold, lower through QpStand, then resume.
        self.declare_parameter('hit_upright_enabled', True)
        self.declare_parameter('hit_upright_hold_s', 1.0)
        self.declare_parameter('hit_upright_phase_timeout_s', 12.0)
        self.declare_parameter('hit_upright_feedback_max_age_s', 0.6)
        self.declare_parameter('hit_upright_recovery_retry_after_s', 2.5)

        # 限高杆流程和障碍物流程分别使用独立的回退参数。
        self.declare_parameter('bar_backoff_duration_s', 1.0)
        self.declare_parameter('bar_backoff_speed', 0.40)

        # 障碍物流程撞击完成后：先固定时间后退，再执行后续转向。
        self.declare_parameter('after_hit_backoff_duration_s', 1.0)
        self.declare_parameter('after_hit_backoff_speed', 0.40)
        self.declare_parameter('after_hit_left_jump_count', 2)

        # 限高杆与障碍物转向统一使用固定机身姿态、角速度和时长。
        self.declare_parameter('p4_timed_turn_body_height', 0.20)
        self.declare_parameter('p4_timed_turn_body_roll', 0.0)
        self.declare_parameter('p4_timed_turn_body_pitch', 0.0)
        self.declare_parameter('p4_timed_turn_body_yaw', 0.0)
        self.declare_parameter('p4_timed_turn_step_height', 0.02)
        self.declare_parameter('p4_timed_turn_wz_90', 0.60)
        self.declare_parameter('p4_timed_turn_duration_90_s', 3.85)
        self.declare_parameter('p4_timed_turn_wz_180', 0.60)
        self.declare_parameter('p4_timed_turn_duration_180_s', 7.7)

        # 左跳完成后：前进并识别两个蓝色障碍物，按虚线侧选择其中一个居中
        # 之前虚线在左边 -> 对齐右边障碍物；之前虚线在右边 -> 对齐左边障碍物
        self.declare_parameter('post_hit_obstacle_forward_speed', 0.50)
        self.declare_parameter('post_hit_obstacle_search_forward_speed', 0.50)
        self.declare_parameter('post_hit_obstacle_trigger_distance_m', 0.25)
        self.declare_parameter('post_hit_obstacle_trigger_bottom_y_ratio', 0.95)
        self.declare_parameter(
            'post_hit_obstacle_align_vy_k', 0.22 if real_platform else 0.35)
        self.declare_parameter(
            'post_hit_obstacle_align_vy_max', 0.16 if real_platform else 0.30)
        self.declare_parameter(
            'post_hit_obstacle_align_vy_min', 0.08 if real_platform else 0.15)
        self.declare_parameter(
            'post_hit_obstacle_center_px_deadband', 12 if real_platform else 7)

        # 对齐选中障碍物并到达距离后：转向 -> 前进 -> 反向转向 -> 最后前进
        # 如果对齐左边障碍物：第一次左转；如果对齐右边障碍物：第一次右转
        self.declare_parameter('post_hit_obs_turn_duration_s', 3.85)
        self.declare_parameter('post_hit_obs_turn_wz', 0.60)
        self.declare_parameter('post_hit_obs_turn_tolerance_deg', 1.5)
        self.declare_parameter('post_hit_obs_forward_duration_s', 1.6)
        self.declare_parameter('post_hit_obs_forward_speed', 0.40)
        # 第二次转回后，先按仿真时间向前走一段，再进入最终横向黄线识别对正
        self.declare_parameter('post_hit_final_forward_duration_s',0.50)
        self.declare_parameter('post_hit_final_forward_speed', 0.20)
        # 绕过障碍物第二次转回后的预前进阶段：如果提前看到前方横向黄线，
        # 只用它的角度修正 wz，不用它提前结束该状态。
        self.declare_parameter('post_hit_pre_final_angle_align_enabled', True)

        # 最后前进阶段：前方横向黄线检测 + 朝向修正
        # 三个子流程完成后，右转并前进寻找最终横向黄线时使用独立前倾角度。
        self.declare_parameter('global_final_yellow_forward_pitch', 0.20)
        self.declare_parameter('final_yellow_stop_line_y_ratio', 0.95)
        self.declare_parameter('final_yellow_align_wz_k', 1.20)
        self.declare_parameter('final_yellow_align_wz_max', 0.30)
        self.declare_parameter('final_yellow_align_wz_min', 0.20)
        self.declare_parameter('final_yellow_tilt_deadband_deg', 0.5)
        self.declare_parameter('final_yellow_done_tilt_deg', 1.0)
        self.declare_parameter('final_yellow_confirm_count', 1)
        # 黄线到达图像下方后，不立刻停；继续前进，等横向黄线从画面中消失后再结束。
        self.declare_parameter('final_yellow_disappear_confirm_count', 2)

        # 全局最终收尾阶段：所有流程完成后，右跳 -> 前进识别横向黄线并矫正 -> 左跳一次。
        # 没接近最终横向黄线前，用这个速度快速前进
        self.declare_parameter('global_final_yellow_forward_speed', 0.60)
        # 横向黄线接近后，切换成这个慢速
        self.declare_parameter('global_final_yellow_slow_forward_speed', 0.40)
        # 当横向黄线底部达到图像高度的这个比例后，开始减速
        self.declare_parameter('global_final_yellow_slow_start_ratio', 0.90)
        # 当横向黄线底部达到这个比例后，认为已经到达下方区域
        # 到达后不马上左跳，而是继续前进，等黄线从画面中消失
        self.declare_parameter('global_final_yellow_stop_line_y_ratio', 1.0)
        # 黄线到达下方区域需要连续确认几帧
        self.declare_parameter('global_final_yellow_confirm_count', 1)
        # 黄线到达下方区域后，连续消失几帧才进入最终左跳
        self.declare_parameter('global_final_yellow_disappear_confirm_count', 2)

        self.declare_parameter('global_final_after_left_jump_right_shift_vy', -0.20)
        self.declare_parameter('global_final_after_left_jump_right_shift_duration_s', 1.0)

        # 限高杆参数
        self._declare_bar_params()

        # 蓝色障碍物参数
        self._declare_obstacle_params()

        # 黄色虚线参数
        self._declare_yellow_params()

        # 最后阶段前方横向黄线参数
        self._declare_final_yellow_params()

        # 第二次转向后的目标检测参数
        self._declare_ball_params('blue_ball', defaults={
            'h_min': 90, 'h_max': 135, 's_min': 80, 's_max': 255, 'v_min': 40, 'v_max': 255,
            'roi_x_ratio_min': 0.00, 'roi_x_ratio_max': 1.00,
            'roi_y_ratio_min': 0.00, 'roi_y_ratio_max': 1.00,
            'open_kernel': 3, 'close_kernel': 5,
            'min_area': 80, 'max_area': 5000000,
            'min_radius': 5.0, 'max_radius': 200.0,
            'min_circularity': 0.82,
            'min_wh_ratio': 0.75, 'max_wh_ratio': 1.33,
            'max_center_y_ratio_in_roi': 1.0,
            'center_weight_base': 0.3, 'center_weight_gain': 0.7,
            'radius_score_gain': 10.0,
        })
        self._declare_ball_params('white_ball', defaults={
            'h_min': 0, 'h_max': 20, 's_min': 0, 's_max': 20, 'v_min': 95, 'v_max': 255,
            'roi_x_ratio_min': 0.00, 'roi_x_ratio_max': 1.00,
            'roi_y_ratio_min': 0.00, 'roi_y_ratio_max': 1.00,
            'open_kernel': 3, 'close_kernel': 5,
            'min_area': 80, 'max_area': 50000,
            'min_radius': 10.0, 'max_radius': 150.0,
            'min_circularity': 0.55,
            'min_wh_ratio': 0.60, 'max_wh_ratio': 1.40,
            'max_center_y_ratio_in_roi': 1.0,
            'center_weight_base': 0.3, 'center_weight_gain': 0.7,
            'radius_score_gain': 10.0,
        })
        self._declare_cola_params()

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.show_debug_vis = bool(self.get_parameter('show_debug_vis').value)
        self.voice_enabled = bool(self.get_parameter('voice_enabled').value)
        self.voice_dir = str(self.get_parameter('voice_dir').value)
        self.voice_backend = str(self.get_parameter('voice_backend').value).strip().lower()
        self.voice_topic = str(self.get_parameter('voice_topic').value).strip()
        self.voice_module_name = str(
            self.get_parameter('voice_module_name').value).strip()
        self.voice_pre_stop_duration_s = max(
            0.0,
            float(self.get_parameter('voice_pre_stop_duration_s').value)
        )
        self.voice_wait_timeout_s = max(
            self.voice_pre_stop_duration_s,
            float(self.get_parameter('voice_wait_timeout_s').value)
        )
        self.p4_rgb_stale_stop_s = max(
            0.1,
            float(self.get_parameter('p4_rgb_stale_stop_s').value)
        )
        # 入口在参数读取时就解析：写错的调试入口要在节点起来时报出来，
        # 而不是等到任务控制激活第四赛段的那一刻。
        self.p4_initial_state = self.resolve_stage_entry(
            self.p4_entry_table(),
            str(self.get_parameter('p4_initial_state').value))
        self.debug_dashed_side = str(self.get_parameter('debug_dashed_side').value).lower().strip()
        self.dashed_side_confirm_frames = max(
            1,
            int(self.get_parameter('dashed_side_confirm_frames').value)
        )

        self.required_bar_count = int(self.get_parameter('required_bar_count').value)
        self.required_obstacle_count = int(self.get_parameter('required_obstacle_count').value)
        self.global_initial_lateral_shift_duration_s = float(
            self.get_parameter('global_initial_lateral_shift_duration_s').value)
        self.global_initial_lateral_shift_vy = float(self.get_parameter('global_initial_lateral_shift_vy').value)
        self.global_lateral_search_vy = float(self.get_parameter('global_lateral_search_vy').value)
        self.global_center_stable_frames = int(self.get_parameter('global_center_stable_frames').value)
        self.global_bar_center_fixed_vy = abs(float(self.get_parameter('global_bar_center_fixed_vy').value))
        self.global_bar_center_px_deadband = int(self.get_parameter('global_bar_center_px_deadband').value)
        self.global_after_task_shift_vy = float(self.get_parameter('global_after_task_shift_vy').value)
        self.global_after_task_shift_duration_s = float(self.get_parameter('global_after_task_shift_duration_s').value)
        self.global_after_obstacle_shift_duration_left_dash_s = float(
            self.get_parameter('global_after_obstacle_shift_duration_left_dash_s').value)
        self.global_after_obstacle_shift_duration_right_dash_s = float(
            self.get_parameter('global_after_obstacle_shift_duration_right_dash_s').value)
        self.search_left_half_after_bar_done = bool(self.get_parameter('search_left_half_after_bar_done').value)
        self.after_bar_search_x_ratio_max = float(self.get_parameter('after_bar_search_x_ratio_max').value)
        self.search_left_half_after_obstacle_done = bool(
            self.get_parameter('search_left_half_after_obstacle_done').value)
        self.after_obstacle_search_x_ratio_max = float(self.get_parameter('after_obstacle_search_x_ratio_max').value)

        self.bar_search_forward_speed = float(self.get_parameter('bar_search_forward_speed').value)
        self.bar_search_near_forward_speed = float(
            self.get_parameter('bar_search_near_forward_speed').value)
        self.bar_search_slow_distance_m = float(
            self.get_parameter('bar_search_slow_distance_m').value)
        self.bar_trigger_distance_m = float(self.get_parameter('bar_trigger_distance_m').value)
        self.use_rgb_distance_triggers = bool(
            self.get_parameter('use_rgb_distance_triggers').value)
        self.bar_search_slow_top_y_ratio = float(
            self.get_parameter('bar_search_slow_top_y_ratio').value)
        self.bar_trigger_top_y_ratio = float(
            self.get_parameter('bar_trigger_top_y_ratio').value)
        self.bar_trigger_confirm_frames = max(
            1, int(self.get_parameter('bar_trigger_confirm_frames').value))
        self.bar_align_vy_k = float(self.get_parameter('bar_align_vy_k').value)
        self.bar_align_vy_max = float(self.get_parameter('bar_align_vy_max').value)
        self.bar_align_vy_min = float(self.get_parameter('bar_align_vy_min').value)
        self.bar_center_px_deadband = int(self.get_parameter('bar_center_px_deadband').value)
        self.bar_center_stable_frames = int(self.get_parameter('bar_center_stable_frames').value)
        self.bar_depth_cluster_enabled = bool(
            self.get_parameter('bar_depth_cluster_enabled').value)
        self.bar_depth_cluster_gap_m = max(
            0.05, float(self.get_parameter('bar_depth_cluster_gap_m').value))
        self.bar_depth_cluster_min_pixels = max(
            1, int(self.get_parameter('bar_depth_cluster_min_pixels').value))
        self.bar_depth_cluster_min_ratio = min(
            1.0,
            max(0.0, float(
                self.get_parameter('bar_depth_cluster_min_ratio').value)),
        )
        self.bar_depth_yaw_align_enabled = bool(self.get_parameter('bar_depth_yaw_align_enabled').value)
        self.bar_depth_yaw_fixed_wz = abs(float(self.get_parameter('bar_depth_yaw_fixed_wz').value))
        self.bar_depth_yaw_deadband_m = float(self.get_parameter('bar_depth_yaw_deadband_m').value)
        self.bar_depth_yaw_sample_x_ratio = float(self.get_parameter('bar_depth_yaw_sample_x_ratio').value)
        self.bar_depth_yaw_sample_y_ratio = float(self.get_parameter('bar_depth_yaw_sample_y_ratio').value)
        self.bar_depth_yaw_sample_half_size = int(self.get_parameter('bar_depth_yaw_sample_half_size').value)
        self.bar_depth_yaw_sign = 1.0 if float(self.get_parameter('bar_depth_yaw_sign').value) >= 0.0 else -1.0
        self.latest_bar_depth_yaw_info = {
            'left_depth': None,
            'right_depth': None,
            'depth_error': None,
            'wz': 0.0,
        }
        self.p4_normal_body_height = float(self.get_parameter('p4_normal_body_height').value)
        self.bar_low_body_height = float(self.get_parameter('bar_low_body_height').value)
        self.obstacle_low_body_height = float(self.get_parameter('obstacle_low_body_height').value)
        self.bar_body_low_enabled = bool(self.get_parameter('bar_body_low_enabled').value)
        self.bar_body_low_do_stop = bool(self.get_parameter('bar_body_low_do_stop').value)
        self.bar_body_lower_wait_s = max(0.0, float(
            self.get_parameter('bar_body_lower_wait_s').value
        ))
        self.bar_clear_forward_speed = float(
            self.get_parameter('bar_clear_forward_speed').value
        )
        self.bar_clear_forward_time_s = max(0.0, float(
            self.get_parameter('bar_clear_forward_time_s').value
        ))
        self.bar_target_forward_pitch = float(
            self.get_parameter('bar_target_forward_pitch').value
        )
        self.bar_target_pitch_wait_s = max(0.0, float(
            self.get_parameter('bar_target_pitch_wait_s').value
        ))
        self.dash_forward_pitch = float(
            self.get_parameter('dash_forward_pitch').value
        )
        self.obstacle_target_forward_pitch = float(
            self.get_parameter('obstacle_target_forward_pitch').value
        )
        self.global_final_yellow_forward_pitch = float(
            self.get_parameter('global_final_yellow_forward_pitch').value
        )
        self.bar_restore_level_before_backoff = bool(
            self.get_parameter('bar_restore_level_before_backoff').value
        )
        self.restore_normal_after_bar_flow = bool(self.get_parameter('restore_normal_after_bar_flow').value)
        self.obstacle_flow_low_enabled = bool(self.get_parameter('obstacle_flow_low_enabled').value)
        self.obstacle_body_low_do_stop = bool(self.get_parameter('obstacle_body_low_do_stop').value)
        self.obstacle_restore_normal_after_final_turn = bool(
            self.get_parameter('obstacle_restore_normal_after_final_turn').value
        )
        self.p4_force_refresh_body_at_start = bool(
            self.get_parameter('p4_force_refresh_body_at_start').value
        )

        self.obstacle_forward_speed = float(self.get_parameter('obstacle_forward_speed').value)
        self.obstacle_search_forward_speed = float(self.get_parameter('obstacle_search_forward_speed').value)
        self.obstacle_trigger_distance_m = float(self.get_parameter('obstacle_trigger_distance_m').value)
        self.obstacle_approach_pitch_distance_m = float(
            self.get_parameter('obstacle_approach_pitch_distance_m').value
        )
        self.obstacle_approach_pitch_bottom_y_ratio = float(
            self.get_parameter('obstacle_approach_pitch_bottom_y_ratio').value)
        self.obstacle_trigger_bottom_y_ratio = float(
            self.get_parameter('obstacle_trigger_bottom_y_ratio').value)
        self.obstacle_approach_forward_pitch = float(
            self.get_parameter('obstacle_approach_forward_pitch').value
        )

        self.obstacle_align_vy_k = float(self.get_parameter('obstacle_align_vy_k').value)
        self.obstacle_align_vy_max = float(self.get_parameter('obstacle_align_vy_max').value)
        self.obstacle_align_vy_min = float(self.get_parameter('obstacle_align_vy_min').value)
        self.obstacle_center_px_deadband = int(self.get_parameter('obstacle_center_px_deadband').value)

        self.dashed_align_vy_k = float(self.get_parameter('dashed_align_vy_k').value)
        self.dashed_align_vy_max = float(self.get_parameter('dashed_align_vy_max').value)
        self.dashed_align_vy_min = float(self.get_parameter('dashed_align_vy_min').value)
        self.dashed_center_px_deadband = int(self.get_parameter('dashed_center_px_deadband').value)
        self.dashed_center_stable_frames = int(self.get_parameter('dashed_center_stable_frames').value)

        self.dashed_pre_shift_speed = float(self.get_parameter('dashed_pre_shift_speed').value)
        self.dashed_pre_shift_backward_speed = abs(float(
            self.get_parameter('dashed_pre_shift_backward_speed').value))
        self.dashed_pre_shift_duration_s = float(self.get_parameter('dashed_pre_shift_duration_s').value)

        self.obstacle_route_turn_duration_s = float(
            self.get_parameter('obstacle_route_turn_duration_s').value)
        self.obstacle_route_turn_wz = abs(float(
            self.get_parameter('obstacle_route_turn_wz').value))
        self.obstacle_route_pre_turn_step_duration_s = max(0.0, float(
            self.get_parameter('obstacle_route_pre_turn_step_duration_s').value))
        self.obstacle_route_lateral_speed = abs(float(
            self.get_parameter('obstacle_route_lateral_speed').value))
        self.obstacle_route_lateral_backward_speed = abs(float(
            self.get_parameter('obstacle_route_lateral_backward_speed').value))
        self.obstacle_route_left_edge_trigger_ratio = float(
            self.get_parameter('obstacle_route_left_edge_trigger_ratio').value)
        self.obstacle_route_right_edge_trigger_ratio = float(
            self.get_parameter('obstacle_route_right_edge_trigger_ratio').value)
        self.obstacle_route_edge_confirm_frames = max(1, int(
            self.get_parameter('obstacle_route_edge_confirm_frames').value))
        self.obstacle_route_forward_speed = float(
            self.get_parameter('obstacle_route_forward_speed').value)
        self.obstacle_route_forward_duration_s = max(0.0, float(
            self.get_parameter('obstacle_route_forward_duration_s').value))
        self.obstacle_route_pre_turn2_lateral_speed = abs(float(
            self.get_parameter('obstacle_route_pre_turn2_lateral_speed').value))
        self.obstacle_route_pre_turn2_lateral_duration_s = max(0.0, float(
            self.get_parameter('obstacle_route_pre_turn2_lateral_duration_s').value))

        self.dashed_align_lost_vy_speed = abs(float(
            self.get_parameter('dashed_align_lost_vy_speed').value))
        self.dashed_target_offset_px = int(self.get_parameter('dashed_target_offset_px').value)

        self.follow_forward_speed = float(self.get_parameter('follow_forward_speed').value)
        self.follow_align_vy_k = float(self.get_parameter('follow_align_vy_k').value)
        self.follow_align_vy_max = float(self.get_parameter('follow_align_vy_max').value)
        self.follow_align_vy_min = float(self.get_parameter('follow_align_vy_min').value)
        self.dashed_lost_stop_frames = int(self.get_parameter('dashed_lost_stop_frames').value)
        self.follow_dashed_valid_x_range_px = int(self.get_parameter('follow_dashed_valid_x_range_px').value)

        self.tf_parent_frame = str(self.get_parameter('tf_parent_frame').value)
        self.tf_child_frame = str(self.get_parameter('tf_child_frame').value)

        self.post_dash_forward_duration_s = float(self.get_parameter('post_dash_forward_duration_s').value)
        self.post_dash_forward_speed = float(self.get_parameter('post_dash_forward_speed').value)

        self.post_dash_turn_duration_s = float(self.get_parameter('post_dash_turn_duration_s').value)
        self.post_dash_turn_wz = float(self.get_parameter('post_dash_turn_wz').value)
        self.post_dash_turn_tolerance_rad = math.radians(
            float(self.get_parameter('post_dash_turn_tolerance_deg').value))

        self.post_turn_forward_duration_s = float(self.get_parameter('post_turn_forward_duration_s').value)
        self.post_turn_forward_fast_duration_s = float(self.get_parameter('post_turn_forward_fast_duration_s').value)
        self.post_turn_forward_fast_speed = float(self.get_parameter('post_turn_forward_fast_speed').value)
        self.post_turn_forward_slow_speed = float(self.get_parameter('post_turn_forward_slow_speed').value)

        self.post_second_turn_duration_s = float(self.get_parameter('post_second_turn_duration_s').value)
        self.post_second_turn_wz = float(self.get_parameter('post_second_turn_wz').value)
        self.post_second_turn_tolerance_rad = math.radians(
            float(self.get_parameter('post_second_turn_tolerance_deg').value))

        self.target_search_forward_speed = float(self.get_parameter('target_search_forward_speed').value)
        self.target_search_backward_speed = abs(float(
            self.get_parameter('target_search_backward_speed').value))
        self.target_seen_confirm_frames = max(1, int(
            self.get_parameter('target_seen_confirm_frames').value))
        self.align_forward_speed_far = float(self.get_parameter('align_forward_speed_far').value)
        self.align_forward_speed_near = float(self.get_parameter('align_forward_speed_near').value)
        self.align_vy_k = float(self.get_parameter('align_vy_k').value)
        self.align_vy_max = float(self.get_parameter('align_vy_max').value)
        self.align_vy_min = float(self.get_parameter('align_vy_min').value)
        self.target_stable_frames = int(self.get_parameter('target_stable_frames').value)
        self.hit_trigger_distance_m = float(self.get_parameter('hit_trigger_distance_m').value)
        self.basketball_ai_roi_x_min_ratio = float(
            self.get_parameter('basketball_ai_roi_x_min_ratio').value)
        self.basketball_ai_roi_x_max_ratio = float(
            self.get_parameter('basketball_ai_roi_x_max_ratio').value)
        self.basketball_ai_roi_y_min_ratio = float(
            self.get_parameter('basketball_ai_roi_y_min_ratio').value)
        self.basketball_ai_roi_y_max_ratio = float(
            self.get_parameter('basketball_ai_roi_y_max_ratio').value)
        self.basketball_ai_max_age_s = max(
            0.05, float(self.get_parameter('basketball_ai_max_age_s').value))
        self.basketball_top_slow_y_ratio = float(
            self.get_parameter('basketball_top_slow_y_ratio').value)
        self.basketball_top_trigger_y_ratio = float(
            self.get_parameter('basketball_top_trigger_y_ratio').value)
        self.basketball_top_trigger_confirm_frames = max(
            1,
            int(self.get_parameter(
                'basketball_top_trigger_confirm_frames').value),
        )

        self.target_visual_thresholds = {
            'cola': {
                'slow_area_ratio': float(self.get_parameter(
                    'cola_slow_cap_area_ratio').value),
                'hit_area_ratio': float(self.get_parameter(
                    'cola_hit_cap_area_ratio').value),
            },
            'blue_ball': {
                'slow_area_ratio': float(self.get_parameter(
                    'blue_ball_slow_area_ratio').value),
                'hit_area_ratio': float(self.get_parameter(
                    'blue_ball_hit_area_ratio').value),
                'slow_radius_px': float(self.get_parameter(
                    'blue_ball_slow_radius_px').value),
                'hit_radius_px': float(self.get_parameter(
                    'blue_ball_hit_radius_px').value),
            },
            'white_ball': {
                'slow_area_ratio': float(self.get_parameter(
                    'white_ball_slow_area_ratio').value),
                'hit_area_ratio': float(self.get_parameter(
                    'white_ball_hit_area_ratio').value),
                'slow_radius_px': float(self.get_parameter(
                    'white_ball_slow_radius_px').value),
                'hit_radius_px': float(self.get_parameter(
                    'white_ball_hit_radius_px').value),
            },
        }
        self.center_px_deadband = int(self.get_parameter('center_px_deadband').value)

        self.hit_timeout_s = float(self.get_parameter('hit_timeout_s').value)

        self.hit_upright_enabled = bool(
            self.get_parameter('hit_upright_enabled').value)
        self.hit_upright_hold_s = max(
            0.0, float(self.get_parameter('hit_upright_hold_s').value))
        self.hit_upright_phase_timeout_s = max(
            1.0, float(self.get_parameter('hit_upright_phase_timeout_s').value))
        self.hit_upright_feedback_max_age_s = max(
            0.05, float(self.get_parameter('hit_upright_feedback_max_age_s').value))
        self.hit_upright_recovery_retry_after_s = max(
            0.1, float(self.get_parameter('hit_upright_recovery_retry_after_s').value))

        self.bar_backoff_duration_s = float(
            self.get_parameter('bar_backoff_duration_s').value)
        self.bar_backoff_speed = abs(float(
            self.get_parameter('bar_backoff_speed').value))
        self.after_hit_backoff_duration_s = float(self.get_parameter('after_hit_backoff_duration_s').value)
        self.after_hit_backoff_speed = abs(float(self.get_parameter('after_hit_backoff_speed').value))
        self.after_hit_left_jump_count = int(self.get_parameter('after_hit_left_jump_count').value)
        self.p4_timed_turn_body_height = float(
            self.get_parameter('p4_timed_turn_body_height').value)
        self.p4_timed_turn_body_roll = float(
            self.get_parameter('p4_timed_turn_body_roll').value)
        self.p4_timed_turn_body_pitch = float(
            self.get_parameter('p4_timed_turn_body_pitch').value)
        self.p4_timed_turn_body_yaw = float(
            self.get_parameter('p4_timed_turn_body_yaw').value)
        self.p4_timed_turn_step_height = float(
            self.get_parameter('p4_timed_turn_step_height').value)
        self.p4_timed_turn_wz_90 = abs(float(self.get_parameter('p4_timed_turn_wz_90').value))
        self.p4_timed_turn_duration_90_s = float(self.get_parameter('p4_timed_turn_duration_90_s').value)
        self.p4_timed_turn_wz_180 = abs(float(self.get_parameter('p4_timed_turn_wz_180').value))
        self.p4_timed_turn_duration_180_s = float(self.get_parameter('p4_timed_turn_duration_180_s').value)

        self.post_hit_obstacle_forward_speed = float(self.get_parameter('post_hit_obstacle_forward_speed').value)
        self.post_hit_obstacle_search_forward_speed = float(
            self.get_parameter('post_hit_obstacle_search_forward_speed').value)
        self.post_hit_obstacle_trigger_distance_m = float(
            self.get_parameter('post_hit_obstacle_trigger_distance_m').value)
        self.post_hit_obstacle_trigger_bottom_y_ratio = float(
            self.get_parameter('post_hit_obstacle_trigger_bottom_y_ratio').value)
        self.post_hit_obstacle_align_vy_k = float(self.get_parameter('post_hit_obstacle_align_vy_k').value)
        self.post_hit_obstacle_align_vy_max = float(self.get_parameter('post_hit_obstacle_align_vy_max').value)
        self.post_hit_obstacle_align_vy_min = float(self.get_parameter('post_hit_obstacle_align_vy_min').value)
        self.post_hit_obstacle_center_px_deadband = int(
            self.get_parameter('post_hit_obstacle_center_px_deadband').value)

        self.post_hit_obs_turn_duration_s = float(self.get_parameter('post_hit_obs_turn_duration_s').value)
        self.post_hit_obs_turn_wz = float(self.get_parameter('post_hit_obs_turn_wz').value)
        self.post_hit_obs_turn_tolerance_rad = math.radians(
            float(self.get_parameter('post_hit_obs_turn_tolerance_deg').value))
        self.post_hit_obs_forward_duration_s = float(self.get_parameter('post_hit_obs_forward_duration_s').value)
        self.post_hit_obs_forward_speed = float(self.get_parameter('post_hit_obs_forward_speed').value)
        self.post_hit_final_forward_duration_s = float(self.get_parameter('post_hit_final_forward_duration_s').value)
        self.post_hit_final_forward_speed = float(self.get_parameter('post_hit_final_forward_speed').value)
        self.post_hit_pre_final_angle_align_enabled = bool(
            self.get_parameter('post_hit_pre_final_angle_align_enabled').value
        )

        self.final_yellow_stop_line_y_ratio = float(self.get_parameter('final_yellow_stop_line_y_ratio').value)
        self.final_yellow_align_wz_k = float(self.get_parameter('final_yellow_align_wz_k').value)
        self.final_yellow_align_wz_max = float(self.get_parameter('final_yellow_align_wz_max').value)
        self.final_yellow_align_wz_min = float(self.get_parameter('final_yellow_align_wz_min').value)
        self.final_yellow_tilt_deadband_deg = float(self.get_parameter('final_yellow_tilt_deadband_deg').value)
        self.final_yellow_done_tilt_deg = float(self.get_parameter('final_yellow_done_tilt_deg').value)
        self.final_yellow_confirm_count = int(self.get_parameter('final_yellow_confirm_count').value)
        self.final_yellow_disappear_confirm_count = int(
            self.get_parameter('final_yellow_disappear_confirm_count').value)

        self.global_final_yellow_forward_speed = float(
            self.get_parameter('global_final_yellow_forward_speed').value
        )

        self.global_final_yellow_slow_forward_speed = float(
            self.get_parameter('global_final_yellow_slow_forward_speed').value
        )

        self.global_final_yellow_slow_start_ratio = float(
            self.get_parameter('global_final_yellow_slow_start_ratio').value
        )

        self.global_final_yellow_stop_line_y_ratio = float(
            self.get_parameter('global_final_yellow_stop_line_y_ratio').value
        )

        self.global_final_yellow_confirm_count = int(
            self.get_parameter('global_final_yellow_confirm_count').value
        )

        self.global_final_yellow_disappear_confirm_count = int(
            self.get_parameter('global_final_yellow_disappear_confirm_count').value
        )

        self.global_final_after_left_jump_right_shift_vy = float(
            self.get_parameter('global_final_after_left_jump_right_shift_vy').value
        )

        self.global_final_after_left_jump_right_shift_duration_s = float(
            self.get_parameter('global_final_after_left_jump_right_shift_duration_s').value
        )

        self.hit_params = {
            'blue_ball': {
                'speed': float(self.get_parameter('hit_blue_ball_speed').value),
                'duration_s': float(self.get_parameter('hit_blue_ball_duration_s').value),
            },
            'white_ball': {
                'speed': float(self.get_parameter('hit_white_ball_speed').value),
                'duration_s': float(self.get_parameter('hit_white_ball_duration_s').value),
            },
            'cola': {
                'speed': float(self.get_parameter('hit_cola_speed').value),
                'duration_s': float(self.get_parameter('hit_cola_duration_s').value),
            },
        }

        self.bar_detector = BarColorDetector(self._read_bar_cfg())
        self.obstacle_detector = ObstacleBlueDetector(self._read_obstacle_cfg())
        self.dashed_detector = YellowDashedLineDetector(self._read_yellow_cfg())
        self.final_yellow_detector = YellowHorizontalLineDetector(self._read_final_yellow_cfg())

        basketball_cfg = self._read_ball_cfg('blue_ball')
        basketball_cfg.update({
            'roi_x_ratio_min': self.basketball_ai_roi_x_min_ratio,
            'roi_x_ratio_max': self.basketball_ai_roi_x_max_ratio,
            'roi_y_ratio_min': self.basketball_ai_roi_y_min_ratio,
            'roi_y_ratio_max': self.basketball_ai_roi_y_max_ratio,
        })
        self.blue_ball_detector = BallDetector(basketball_cfg, 'blue_ball')
        self.white_ball_detector = FootballDetector()
        self.cola_detector = ColaDetector(self._read_cola_cfg())

        self.latest_ai_bgr = None
        self.latest_ai_rx_monotonic_s = None
        self.ai_camera_sub = None
        if self.ai_camera_topic:
            self.ai_camera_sub = self.create_subscription(
                Image,
                self.ai_camera_topic,
                self.ai_camera_callback,
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f'[BASKETBALL_AI] subscribed topic={self.ai_camera_topic}, '
                f'roi=({self.basketball_ai_roi_x_min_ratio:.2f},'
                f'{self.basketball_ai_roi_y_min_ratio:.2f})-('
                f'{self.basketball_ai_roi_x_max_ratio:.2f},'
                f'{self.basketball_ai_roi_y_max_ratio:.2f})')
        self.cola_interface_stable_frames = max(
            1,
            int(self.get_parameter('cola.interface_stable_frames').value),
        )
        self.cola_interface_count = 0
        self.cola_detected_topic = str(
            self.get_parameter('cola.detected_topic').value
        )
        self.cola_detection_topic = str(
            self.get_parameter('cola.detection_topic').value
        )
        self.cola_detected_pub = self.create_publisher(
            Bool,
            self.cola_detected_topic,
            10,
        )
        self.cola_detection_pub = self.create_publisher(
            String,
            self.cola_detection_topic,
            10,
        )

        # 语音播报：用事件 ID 防止同一个触发点重复播报。
        # bar_1 / bar_2 可以分别播报；同一个目标类型只在准备撞击时播一次。
        # The player is created only while Stage 4 owns the robot.  Simulation
        # keeps local WAV playback; the physical robot publishes the verified
        # offline AudioPlayExtend IDs on the relative speech_play_extend topic.
        self.voice = None
        self.voice_events_spoken = set()
        self.pending_voice_event_id: Optional[str] = None
        self.pending_voice_key: Optional[str] = None
        self.pending_voice_resume_state: Optional[str] = None
        self.pending_voice_started = False
        self.pending_voice_started_time: Optional[float] = None

        self.motion_cmd = (0.0, 0.0, 0.0)

        # 当前连续运动命令所携带的机身姿态。
        # send_motion_cmd() 的姿态参数为 None 时，不是把 None 发给 LCM，
        # 而是继续把这里保存的旧数值写入下一条 LCM 消息。
        self.body_height_cmd = float(self.p4_normal_body_height)
        self.body_roll_cmd = 0.0
        self.body_pitch_cmd = 0.0
        self.body_yaw_cmd = 0.0
        self.step_height_cmd = 0.02
        self.body_is_low = False
        # 第四赛段单独调试/程序重启时，控制器可能残留上一次 low 姿态。
        # 用 STOP -> LOW -> STOP -> NORMAL 强制刷新一次，确保初始是 normal。
        # 拆分版：STOP -> LOW -> STOP -> NORMAL 的姿态刷新需要 LCM 连接，
        # Robot_Ctrl 在激活时才创建，所以姿态恢复移动到 on_activated() 执行。

        # Fourth-stage state is entered explicitly after P3 finishes.
        # 状态计时起点延迟到 control_loop 第一次真正运行时再记录，
        # 避免节点初始化阶段 /clock 还没就绪导致 state_enter_time=0，
        # 从而让 GLOBAL_INITIAL_LATERAL_SHIFT 等固定时间状态被瞬间跳过。
        self.state_enter_time = None

        self.dashed_center_count = 0
        self.dashed_lost_count = 0
        self.obstacle_pair_seen_in_approach = False
        # APPROACH_OBSTACLES 中一旦达到前倾阈值就锁存，避免检测框抖动导致 pitch 在 0/前倾之间反复切换。
        self.obstacle_approach_pitch_latched = False

        # 第一次看到虚线时，记录它在图像左边还是右边。
        # 后续预横移和偏置对齐都会使用这个方向，不在 ALIGN 状态里清空。
        self.dashed_side = None  # None / 'left' / 'right'
        self.dashed_side_candidate = None
        self.dashed_side_candidate_count = 0
        # DASH_PRE_SIDE_SHIFT 不再按时间结束，而是记录 TF 起点，按横向位移结束。
        self.dashed_pre_shift_start_pose = None
        self.dashed_pre_shift_dir_sign = 0.0

        # 新障碍物路线：单障碍物边缘阈值连续确认计数。
        self.obstacle_route_edge_confirm_count = 0
        self.obstacle_route_selected_obstacle: Optional[Detection] = None

        # 后续任务使用的 TF 起点
        self.post_forward_start_pose = None
        self.post_turn_forward_start_pose = None
        self.turn_start_yaw = None
        self.current_turn_dir = 0  # +1 左转，-1 右转
        self.current_turn_angle_rad = 0.0
        self.current_turn_tolerance_rad = 0.0
        self.current_turn_wz = 0.0

        # 第二次转向后目标检测/撞击使用
        self.latest_target: Optional[Detection] = None
        self.locked_target: Optional[Detection] = None
        self.target_stable_count = 0
        self.stable_target_type = None
        # 从第二次转向结束开始记录目标是否已经“长时间稳定出现”。
        # 只有连续识别同一种目标达到 target_seen_confirm_frames 后，
        # target_seen_after_turn 才置 True；之后丢失目标才允许后退重找。
        self.target_seen_after_turn = False
        self.target_seen_confirm_count = 0
        self.target_seen_confirm_type = None
        self.hit_start_pose = None
        self.basketball_top_trigger_count = 0

        # Basketball upright action bookkeeping.
        self.hit_upright_retry_sent = False
        self.hit_upright_failure_reason = ''
        self.upright_lcm_ctrl = None

        # 撞击完成后的后退 / 左跳 / 障碍物选择对齐使用
        self.after_hit_backoff_start_pose = None
        self.selected_obstacle_after_hit: Optional[Detection] = None
        self.selected_obstacle_after_hit_side = None  # 'left' / 'right'
        self.post_hit_obs_forward_start_pose = None
        self.post_hit_pre_final_forward_start_pose = None
        self.post_hit_final_forward_start_pose = None  # 保留旧变量名，避免外部引用出错

        # 最后阶段前方横向黄线对正 / 到达判定
        self.final_yellow_done_counter = 0
        self.final_yellow_reached_lower_area = False
        self.final_yellow_disappear_counter = 0
        self.latest_final_yellow_line: Optional[Detection] = None

        # 全局最终收尾黄线确认计数器
        self.global_final_yellow_done_counter = 0
        self.global_final_yellow_reached_lower_area = False
        self.global_final_yellow_disappear_counter = 0

        # 全局整合流程变量
        self.completed_bar_count = 0
        self.completed_obstacle_count = 0

        # 记录障碍物流程是不是作为全局第 3 个子流程启动。
        # 如果障碍物是最后一个要完成的物体：
        #   POST_HIT_FINAL_FORWARD 识别到横向黄线并靠近后，
        #   FINAL_LEFT_JUMP 只执行 1 次左跳，
        #   然后直接进入 GLOBAL_FINAL_YELLOW_FORWARD，
        #   跳过 GLOBAL_FINAL_RIGHT_JUMP。
        self.obstacle_flow_is_third_object = False

        # 完成限高杆流程后的下一轮搜索过滤标志。
        # True 时，GLOBAL_LATERAL_SEARCH 只允许 center_x 位于图像左半边的目标参与选择；
        # 一旦真正选中下一个目标，就自动关闭。
        self.only_search_left_half_after_bar = False
        self.left_half_filter_reason: Optional[str] = None
        self.current_left_half_search_x_ratio_max = self.after_bar_search_x_ratio_max

        self.global_center_stable_count = 0
        self.current_global_target: Optional[Detection] = None

        # RGB freshness bookkeeping for Stage4 perception.
        # Keep a strong reference to the last processed ROS Image message so
        # object identity cannot be accidentally reused by Python.
        self._p4_last_processed_rgb_seq = -1
        self._p4_last_new_rgb_monotonic_s = time.monotonic()
        self._p4_rgb_stale_stop_active = False

        self.global_initial_lateral_shift_start_pose = None
        self.global_after_task_shift_start_pose = None
        self.current_after_task_shift_duration_s = self.global_after_task_shift_duration_s
        self.global_after_task_shift_reason = 'init'

        # 限高杆子流程变量
        self.current_bar_det: Optional[Detection] = None
        self.bar_center_stable_count = 0
        self.bar_trigger_confirm_count = 0
        self.bar_hit_start_pose = None

        self.task_done_stop_sent = False
        self.last_log_time = self.now_s()

        self.get_logger().info('Fourth-stage mixin initialized; waiting for P3 to hand off.')


    # ---------- 全局整合 / 限高杆辅助 ----------
    def _declare_bar_params(self):
        p = self.declare_parameter
        if self.platform == 'real':
            # OpenCV 的红色跨 HSV 色相边界；h_min > h_max 表示合并
            # 170..179 和 0..10 两个区间。仿真仍沿用原限高杆颜色。
            bar_h_min, bar_h_max = 170, 10
            bar_s_min, bar_s_max = 80, 255
            bar_v_min, bar_v_max = 50, 255
            bar_max_aspect_ratio = 6.0
        else:
            bar_h_min, bar_h_max = 85, 100
            bar_s_min, bar_s_max = 15, 45
            bar_v_min, bar_v_max = 35, 80
            bar_max_aspect_ratio = 50.0
        p('bar.h_min', bar_h_min)
        p('bar.h_max', bar_h_max)
        p('bar.s_min', bar_s_min)
        p('bar.s_max', bar_s_max)
        p('bar.v_min', bar_v_min)
        p('bar.v_max', bar_v_max)
        p('bar.roi_x_ratio_min', 0.00);
        p('bar.roi_x_ratio_max', 1.00)
        p('bar.roi_y_ratio_min', 0.00);
        p('bar.roi_y_ratio_max', 1.00)
        p('bar.open_kernel', 3)
        p('bar.close_kernel_h', 7);
        p('bar.close_kernel_w', 11)
        p('bar.min_area', 1000)
        p('bar.min_width', 15)
        p('bar.max_height', 1000)
        p('bar.min_aspect_ratio', 1.5)
        p('bar.max_aspect_ratio', bar_max_aspect_ratio)
        p('bar.max_center_y_ratio_in_roi', 1.0)
        p('bar.center_weight_base', 0.3)
        p('bar.center_weight_gain', 0.7)
        p('bar.structure_check_enabled', self.platform == 'real')
        p('bar.structure_inner_ratio', 0.40)
        p('bar.structure_max_inner_red_ratio', 0.35)
        p('bar.structure_min_depth_gap_m', 0.20)
        p('bar.structure_near_bypass_distance_m', 1.00)
        p('bar.structure_near_bypass_width_ratio', 0.40)
        p('bar.structure_min_depth_pixels', 10)

    def _read_bar_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('bar.h_min').value), 'h_max': int(gp('bar.h_max').value),
            's_min': int(gp('bar.s_min').value), 's_max': int(gp('bar.s_max').value),
            'v_min': int(gp('bar.v_min').value), 'v_max': int(gp('bar.v_max').value),
            'roi_x_ratio_min': float(gp('bar.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('bar.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('bar.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('bar.roi_y_ratio_max').value),
            'open_kernel': int(gp('bar.open_kernel').value),
            'close_kernel_h': int(gp('bar.close_kernel_h').value),
            'close_kernel_w': int(gp('bar.close_kernel_w').value),
            'min_area': int(gp('bar.min_area').value),
            'min_width': int(gp('bar.min_width').value),
            'max_height': int(gp('bar.max_height').value),
            'min_aspect_ratio': float(gp('bar.min_aspect_ratio').value),
            'max_aspect_ratio': float(gp('bar.max_aspect_ratio').value),
            'max_center_y_ratio_in_roi': float(gp('bar.max_center_y_ratio_in_roi').value),
            'center_weight_base': float(gp('bar.center_weight_base').value),
            'center_weight_gain': float(gp('bar.center_weight_gain').value),
            'structure_check_enabled': bool(gp('bar.structure_check_enabled').value),
            'structure_inner_ratio': float(gp('bar.structure_inner_ratio').value),
            'structure_max_inner_red_ratio': float(
                gp('bar.structure_max_inner_red_ratio').value),
            'structure_min_depth_gap_m': float(
                gp('bar.structure_min_depth_gap_m').value),
            'structure_near_bypass_distance_m': float(
                gp('bar.structure_near_bypass_distance_m').value),
            'structure_near_bypass_width_ratio': float(
                gp('bar.structure_near_bypass_width_ratio').value),
            'structure_min_depth_pixels': int(
                gp('bar.structure_min_depth_pixels').value),
        }

    def all_global_tasks_done(self) -> bool:
        return (
                self.completed_bar_count >= self.required_bar_count
                and self.completed_obstacle_count >= self.required_obstacle_count
        )

    def enter_global_final_sequence(self):
        """全部子流程完成后，不直接 DONE，而是执行最终收尾动作。"""
        self.get_logger().info(
            '[GLOBAL_FINAL] all required flows completed, start final sequence: '
            'right jump -> yellow forward align -> one left jump -> DONE'
        )
        self.enter_state(self.GLOBAL_FINAL_RIGHT_JUMP)

    def speak_event_once(self, event_id: str, voice_key: str) -> bool:
        """
        在指定事件第一次发生时播报一次。
        event_id 用来防重复，例如 bar_1、bar_2、obstacle_1、target_cola。
        voice_key 对应 CyberdogVoicePlayer 的离线 ID/本地 WAV 映射。
        """
        if not getattr(self, 'voice_enabled', True):
            return False
        if event_id in self.voice_events_spoken:
            return False
        if self.voice is None:
            self.get_logger().warning('[VOICE] player is not active')
            return False
        played = self.voice.speak_once(event_id, voice_key)
        # 队列拒绝事件时返回 False；此时不登记为已播，
        # 让停车等待状态能够在播放器可用后重试。
        if played:
            self.voice_events_spoken.add(event_id)
        self.get_logger().info(f'[VOICE] event={event_id}, key={voice_key}, played={played}')
        return played

    def begin_voice_wait(
            self,
            wait_state: str,
            event_id: str,
            voice_key: str,
            resume_state: str
    ):
        """进入零速度语音等待状态；若播放器正忙，会等空闲后再播指定语音。"""
        self.pending_voice_event_id = event_id
        self.pending_voice_key = voice_key
        self.pending_voice_resume_state = resume_state
        self.pending_voice_started = False
        self.pending_voice_started_time = None
        self.enter_state(wait_state)

    def finish_voice_wait(self):
        """清理语音等待上下文并恢复原流程。"""
        resume_state = self.pending_voice_resume_state
        self.pending_voice_event_id = None
        self.pending_voice_key = None
        self.pending_voice_resume_state = None
        self.pending_voice_started = False
        self.pending_voice_started_time = None

        if resume_state is None:
            self.get_logger().error('[VOICE_WAIT] resume_state is None, stop in current state')
            return
        self.get_logger().info(f'[VOICE_WAIT] finished, resume state={resume_state}')
        self.enter_state(resume_state)

    def handle_voice_wait(self, pitch: Optional[float] = None, body_height: Optional[float] = None):
        """持续发送零速度，等停稳、播放完成后再恢复动作。"""
        self.send_motion_cmd(0.0, 0.0, 0.0, pitch=pitch, body_height=body_height)
        if not self.voice_enabled or self.voice is None:
            self.get_logger().info('[VOICE_WAIT] voice disabled; resume motion')
            self.finish_voice_wait()
            return
        elapsed = self.state_elapsed_s()

        if elapsed < self.voice_pre_stop_duration_s:
            self.get_logger().info(
                f'[VOICE_WAIT] stopping before speech: '
                f'{elapsed:.3f}/{self.voice_pre_stop_duration_s:.3f}s',
                throttle_duration_sec=0.2
            )
            return

        if not self.pending_voice_started and elapsed >= self.voice_wait_timeout_s:
            self.get_logger().warning(
                f'[VOICE_WAIT] timeout before playback: event={self.pending_voice_event_id}, '
                f'elapsed={elapsed:.3f}/{self.voice_wait_timeout_s:.3f}s; resume motion'
            )
            self.finish_voice_wait()
            return

        if not self.pending_voice_started:
            if self.pending_voice_event_id in self.voice_events_spoken:
                self.finish_voice_wait()
                return

            # 目标物体语音可能还在播；先等它结束，再播放当前事件，避免被 busy 跳过。
            if self.voice.is_playing():
                self.get_logger().info(
                    f'[VOICE_WAIT] player busy; wait to play event={self.pending_voice_event_id}',
                    throttle_duration_sec=0.2
                )
                return

            if self.pending_voice_event_id is None or self.pending_voice_key is None:
                self.get_logger().error('[VOICE_WAIT] missing pending voice context; resume motion')
                self.finish_voice_wait()
                return

            played = self.speak_event_once(
                self.pending_voice_event_id,
                self.pending_voice_key
            )
            if not played:
                self.get_logger().warning(
                    f'[VOICE_WAIT] voice not started: event={self.pending_voice_event_id}; resume motion'
                )
                self.finish_voice_wait()
                return

            self.pending_voice_started = True
            self.pending_voice_started_time = self.now_s()
            return

        voice_elapsed = (
            self.now_s() - self.pending_voice_started_time
            if self.pending_voice_started_time is not None
            else 0.0
        )
        if voice_elapsed >= self.voice_wait_timeout_s:
            self.get_logger().warning(
                f'[VOICE_WAIT] playback timeout: event={self.pending_voice_event_id}, '
                f'elapsed={voice_elapsed:.3f}/{self.voice_wait_timeout_s:.3f}s; resume motion'
            )
            self.finish_voice_wait()
            return

        if self.voice.is_playing():
            self.get_logger().info(
                f'[VOICE_WAIT] speaking event={self.pending_voice_event_id}, '
                f'elapsed={voice_elapsed:.3f}s',
                throttle_duration_sec=0.2
            )
            return

        self.finish_voice_wait()

    def speak_bar_at_trigger(self):
        # 两次限高杆分别播报：bar_1、bar_2
        event_id = f'bar_{self.completed_bar_count + 1}'
        self.speak_event_once(event_id, 'bar')

    def speak_obstacle_at_trigger(self):
        event_id = f'obstacle_{self.completed_obstacle_count + 1}'
        self.speak_event_once(event_id, 'obstacle')

    def target_voice_key(self, det_type: str) -> Optional[str]:
        """
        当前代码里的目标类型与赛题播报类型映射。
        cola       -> 识别到可乐瓶
        blue_ball  -> 识别到橙色小球（如果你后续改成 orange_ball，也兼容）
        white_ball -> 识别到足球（如果你后续改成 football，也兼容）
        """
        if det_type == 'cola':
            return 'cola'
        if det_type in ('orange_ball', 'blue_ball'):
            return 'orange_ball'
        if det_type in ('football', 'white_ball'):
            return 'football'
        return None

    def speak_target_at_hit_trigger(self, det_type: str):
        key = self.target_voice_key(det_type)
        if key is None:
            self.get_logger().warn(f'[VOICE] no voice mapping for target det_type={det_type}')
            return
        # 三个目标物体每类只在真正准备撞击时播一次
        event_id = f'target_{key}'
        self.speak_event_once(event_id, key)

    def is_obstacle_flow_state(self) -> bool:
        return self.state in {
            self.APPROACH_OBSTACLES,
            self.DASH_PRE_SIDE_SHIFT,
            self.OBSTACLE_ROUTE_PRE_TURN1_STEP,
            self.OBSTACLE_ROUTE_TURN_1,
            self.OBSTACLE_ROUTE_LATERAL_SCAN,
            self.OBSTACLE_ROUTE_FORWARD,
            self.OBSTACLE_ROUTE_PRE_TURN2_LATERAL,
            self.OBSTACLE_ROUTE_PRE_TURN2_STEP,
            self.OBSTACLE_ROUTE_TURN_2,
            self.ALIGN_DASHED_LINE,
            self.FOLLOW_DASHED_UNTIL_LOST,
            self.POST_DASH_FORWARD,
            self.POST_DASH_TURN_1,
            self.POST_TURN_FORWARD,
            self.POST_DASH_TURN_2,
            self.SEARCH_TARGET_AFTER_TURNS,
            self.APPROACH_AND_ALIGN_TARGET,
            self.HIT_TARGET,
            self.HIT_UPRIGHT_PREPARE,
            self.HIT_UPRIGHT_RISE,
            self.HIT_UPRIGHT_HOLD,
            self.HIT_UPRIGHT_SMOOTH_RETURN,
            self.HIT_UPRIGHT_READY_STAND,
            self.HIT_UPRIGHT_RECOVERY,
            self.HIT_BACKOFF_AFTER_HIT,
            self.POST_HIT_LEFT_JUMP,
            self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
            self.POST_HIT_OBSTACLE_VOICE_WAIT,
            self.POST_HIT_OBS_TURN_1,
            self.POST_HIT_OBS_FORWARD,
            self.POST_HIT_OBS_TURN_2,
            self.POST_HIT_PRE_FINAL_FORWARD,
            self.POST_HIT_FINAL_FORWARD,
            self.FINAL_LEFT_JUMP,
            self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN,
            self.OBSTACLE_FLOW_DONE,
        }

    def is_in_after_bar_search_region(self, det: Optional[Detection]) -> bool:
        """
        完成限高杆流程后的下一轮全局搜索，只允许图像左半边目标参与选择。

        默认 after_bar_search_x_ratio_max=0.50：
            det.center_img[0] < image_width * 0.50 才有效。

        如果 only_search_left_half_after_bar=False，则不过滤。
        """
        if det is None:
            return False

        if self.latest_bgr is None:
            return True

        if not self.only_search_left_half_after_bar:
            return True

        img_w = self.latest_bgr.shape[1]
        x_limit = img_w * self.current_left_half_search_x_ratio_max
        return det.center_img[0] < x_limit

    def choose_global_object(self, bar: Optional[Detection], obs_candidates: List[Detection]):
        if self.latest_bgr is None:
            return None, None

        img_w = self.latest_bgr.shape[1]
        img_center_x = img_w / 2.0
        choices = []

        if self.only_search_left_half_after_bar:
            x_limit = img_w * self.current_left_half_search_x_ratio_max
            self.get_logger().info(
                f'[GLOBAL_FILTER] after subtask done: only search left region, '
                f'center_x < {x_limit:.1f} ({self.current_left_half_search_x_ratio_max:.2f} * image_width)',
                throttle_duration_sec=0.8
            )

        if bar is not None and self.completed_bar_count < self.required_bar_count:
            if self.is_in_after_bar_search_region(bar):
                choices.append(('bar', bar, abs(bar.center_img[0] - img_center_x)))
            else:
                self.get_logger().info(
                    f'[GLOBAL_FILTER] ignore BAR outside left-search region: center={bar.center_img}',
                    throttle_duration_sec=0.5
                )

        if obs_candidates and self.completed_obstacle_count < self.required_obstacle_count:
            # 完成限高杆后的“左半边搜索”限制只用于 BAR，
            # 蓝色障碍物始终允许在整张图中参与全局目标选择。
            valid_obs = list(obs_candidates)

            if valid_obs:
                # 多个障碍物时先保留外接框底边最靠下的两个，
                # 用两者中心连线的中点参与全局目标选择，而不是使用单个障碍物中心。
                if len(valid_obs) >= 2:
                    pair = self.choose_obstacle_pair(valid_obs)
                    if pair is not None:
                        left, right = pair
                        pair_center_x = (
                            left.center_img[0] + right.center_img[0]) / 2.0
                        pair_center_y = int(round(
                            (left.center_img[1] + right.center_img[1]) / 2.0))
                        pair_bbox = (
                            min(left.bbox_img[0], right.bbox_img[0]),
                            min(left.bbox_img[1], right.bbox_img[1]),
                            max(left.bbox_img[2], right.bbox_img[2]),
                            max(left.bbox_img[3], right.bbox_img[3]),
                        )
                        obs = Detection(
                            'blue_obstacle',
                            (int(round(pair_center_x)), pair_center_y),
                            pair_bbox,
                            float(left.score + right.score),
                            {'global_pair_center': True},
                        )
                        obs_center_error = abs(pair_center_x - img_center_x)
                    else:
                        obs = valid_obs[0]
                        obs_center_error = abs(
                            obs.center_img[0] - img_center_x)
                else:
                    obs = valid_obs[0]
                    obs_center_error = abs(obs.center_img[0] - img_center_x)
                choices.append(('obstacle', obs, obs_center_error))

        if not choices:
            return None, None

        obj_type, det, _ = min(choices, key=lambda x: x[2])

        # 左半边限制只用于下一轮 BAR 的全局选择。
        # 一旦真的选中了任意下一目标（BAR 或障碍物），就关闭该临时限制。
        if self.only_search_left_half_after_bar:
            self.get_logger().info(
                f'[GLOBAL_FILTER] selected {obj_type}, center={det.center_img}; '
                f'disable temporary BAR left-half filter'
            )
            self.only_search_left_half_after_bar = False
            self.left_half_filter_reason = None

        return obj_type, det

    def finish_bar_flow(self):
        self.completed_bar_count += 1
        self.get_logger().info(
            f'[GLOBAL] bar flow finished: bar={self.completed_bar_count}/{self.required_bar_count}, '
            f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}'
        )
        # 限高杆流程结束后恢复 normal，避免后续全局搜索/最终收尾继续低身。
        self.restore_body_normal_after_bar_flow()

        if self.all_global_tasks_done():
            self.get_logger().info('[GLOBAL] all flows done after bar flow, start final sequence')
            self.enter_global_final_sequence()
            return

        # 新增：完成一次限高杆后，下一轮 GLOBAL_LATERAL_SEARCH 只在左半边找目标。
        # 目的：避免刚刚通过的限高杆还在右半边画面里，被重复当成下一个目标。
        if self.search_left_half_after_bar_done:
            self.only_search_left_half_after_bar = True
            self.left_half_filter_reason = 'bar_done'
            self.current_left_half_search_x_ratio_max = self.after_bar_search_x_ratio_max
            self.get_logger().info(
                f'[GLOBAL_FILTER] bar done: next global search only uses left region, '
                f'x_ratio_max={self.current_left_half_search_x_ratio_max:.2f}'
            )

        self.current_after_task_shift_duration_s = self.global_after_task_shift_duration_s
        self.global_after_task_shift_reason = 'bar_done'
        self.enter_state(self.GLOBAL_SHIFT_AFTER_SUBTASK)

    def finish_obstacle_flow(self):
        self.completed_obstacle_count += 1
        self.get_logger().info(
            f'[GLOBAL] obstacle flow finished: bar={self.completed_bar_count}/{self.required_bar_count}, '
            f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}, dashed_side={self.dashed_side}'
        )
        if self.all_global_tasks_done():
            self.get_logger().info('[GLOBAL] all flows done after obstacle flow, start final sequence')
            self.enter_global_final_sequence()
            return

        # 新增：完成障碍物流程后，下一轮 GLOBAL_LATERAL_SEARCH 也只在左半边找目标。
        # 目的：和限高杆完成后的逻辑一致，避免刚完成的障碍物流程相关目标仍在右侧画面中干扰下一目标选择。
        if self.search_left_half_after_obstacle_done:
            self.only_search_left_half_after_bar = True
            self.left_half_filter_reason = 'obstacle_done'
            self.current_left_half_search_x_ratio_max = self.after_obstacle_search_x_ratio_max
            self.get_logger().info(
                f'[GLOBAL_FILTER] obstacle done: next global search only uses left region, '
                f'x_ratio_max={self.current_left_half_search_x_ratio_max:.2f}'
            )

        if self.dashed_side == 'right':
            self.current_after_task_shift_duration_s = self.global_after_obstacle_shift_duration_right_dash_s
        elif self.dashed_side == 'left':
            self.current_after_task_shift_duration_s = self.global_after_obstacle_shift_duration_left_dash_s
        else:
            self.current_after_task_shift_duration_s = self.global_after_task_shift_duration_s
            self.get_logger().warn('[GLOBAL] obstacle finished but dashed_side is None, use default shift distance')
        self.global_after_task_shift_reason = f'obstacle_done_dash_{self.dashed_side}'
        self.get_logger().info(
            f'[GLOBAL] after obstacle shift duration={self.current_after_task_shift_duration_s:.3f}s, '
            f'reason={self.global_after_task_shift_reason}'
        )
        self.enter_state(self.GLOBAL_SHIFT_AFTER_SUBTASK)

    def estimate_bar_depth(self, bar_det: Detection) -> Optional[float]:
        if self.latest_depth is None or self.latest_bgr is None or bar_det is None:
            return None
        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None
        ih, iw = self.latest_bgr.shape[:2]
        dh, dw = depth_m.shape[:2]
        x1, y1, x2, y2 = bar_det.bbox_img
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        sx1 = x1 + int(0.15 * bw)
        sx2 = x2 - int(0.15 * bw)
        sy1 = y1 + int(0.05 * bh)
        sy2 = y1 + int(0.15 * bh)
        sx1 = max(0, min(iw - 1, sx1));
        sx2 = max(sx1 + 1, min(iw, sx2))
        sy1 = max(0, min(ih - 1, sy1));
        sy2 = max(sy1 + 1, min(ih, sy2))
        dx1 = int(sx1 * dw / max(iw, 1));
        dx2 = int(sx2 * dw / max(iw, 1))
        dy1 = int(sy1 * dh / max(ih, 1));
        dy2 = int(sy2 * dh / max(ih, 1))
        dx1 = max(0, min(dw - 1, dx1));
        dx2 = max(dx1 + 1, min(dw, dx2))
        dy1 = max(0, min(dh - 1, dy1));
        dy2 = max(dy1 + 1, min(dh, dy2))
        patch = depth_m[dy1:dy2, dx1:dx2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size == 0:
            return None

        if self.bar_depth_cluster_enabled:
            sorted_depths = np.sort(valid)
            split_indexes = np.where(
                np.diff(sorted_depths) > self.bar_depth_cluster_gap_m
            )[0] + 1
            clusters = np.split(sorted_depths, split_indexes)
            min_support = max(
                self.bar_depth_cluster_min_pixels,
                int(math.ceil(
                    valid.size * self.bar_depth_cluster_min_ratio)),
            )
            supported_clusters = [
                cluster for cluster in clusters
                if cluster.size >= min_support
            ]
            if supported_clusters:
                # 取样带里横杆比背景更近；选择最近的有效深度簇，
                # 再取簇中位数，避免像素占比刚好跨过 20% 时跳到背景。
                selected = min(
                    supported_clusters,
                    key=lambda cluster: float(np.median(cluster)),
                )
                selected_depth = float(np.median(selected))
                cluster_text = ','.join(
                    '{:.2f}m/{}px'.format(
                        float(np.median(cluster)), int(cluster.size))
                    for cluster in supported_clusters
                )
                self.get_logger().info(
                    '[BAR_DEPTH] clusters=[{}], selected={:.3f}m, '
                    'min_support={}px'.format(
                        cluster_text, selected_depth, min_support),
                    throttle_duration_sec=0.5,
                )
                return selected_depth

            self.get_logger().warn(
                '[BAR_DEPTH] no supported cluster: valid={}px, '
                'min_support={}px; fall back to p20'.format(
                    int(valid.size), min_support),
                throttle_duration_sec=1.0,
            )
        return float(np.percentile(valid, 20))

    def is_bar_centered(self, bar: Detection, deadband_px: Optional[int] = None) -> bool:
        if self.latest_bgr is None or bar is None:
            return False

        if deadband_px is None:
            deadband_px = self.bar_center_px_deadband

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x

        return abs(err_px) <= int(deadband_px)

    def compute_bar_align_vy(self, bar: Detection) -> float:
        if self.latest_bgr is None or bar is None:
            return 0.0
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x
        if abs(err_px) <= self.bar_center_px_deadband:
            return 0.0
        err_norm = err_px / max(self.latest_bgr.shape[1] / 2.0, 1.0)
        vy = -self.bar_align_vy_k * err_norm
        vy = float(np.clip(vy, -self.bar_align_vy_max, self.bar_align_vy_max))
        if 0.0 < abs(vy) < self.bar_align_vy_min:
            vy = math.copysign(self.bar_align_vy_min, vy)
        return vy

    def compute_global_bar_center_fixed_vy(self, bar: Detection) -> float:
        if self.latest_bgr is None or bar is None:
            return 0.0

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = bar.center_img[0] - img_center_x

        if abs(err_px) <= self.global_bar_center_px_deadband:
            return 0.0

        vy = -math.copysign(self.global_bar_center_fixed_vy, err_px)
        return float(vy)

    def sample_depth_patch_by_rgb(self, rgb_x: int, rgb_y: int, half_size: int) -> Optional[float]:
        """
        在 RGB 像素点附近采样深度，返回较近的 20 分位数。
        用于限高杆左右两侧深度差判断朝向。
        """
        if self.latest_depth is None or self.latest_bgr is None:
            return None

        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None

        ih, iw = self.latest_bgr.shape[:2]
        dh, dw = depth_m.shape[:2]

        dx = int(rgb_x * dw / max(iw, 1))
        dy = int(rgb_y * dh / max(ih, 1))

        half = max(1, int(half_size))
        x1 = max(0, dx - half)
        x2 = min(dw, dx + half + 1)
        y1 = max(0, dy - half)
        y2 = min(dh, dy + half + 1)

        patch = depth_m[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 20))

    def compute_bar_depth_yaw_align_wz(self, bar: Detection) -> float:
        """
        限高杆朝向矫正：根据限高杆左右两侧深度差，给一个固定小角速度 wz。

        left_depth / right_depth 基本相等：说明机器狗大致正对限高杆，不转。
        两边深度差超过 deadband：说明一侧更近，机器狗相对限高杆有偏航，给固定 wz 修正。

        注意：wz 正负号和相机/机器人坐标有关。
        如果实测发现越修越歪，把参数 bar_depth_yaw_sign 从 1.0 改成 -1.0。
        """
        self.latest_bar_depth_yaw_info = {
            'left_depth': None,
            'right_depth': None,
            'depth_error': None,
            'wz': 0.0,
        }

        if not self.bar_depth_yaw_align_enabled:
            return 0.0
        if bar is None or self.latest_bgr is None or self.latest_depth is None:
            return 0.0

        x1, y1, x2, y2 = bar.bbox_img
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # 在限高杆 bbox 内，左右各取一个采样点。
        # y 取靠近横杆上沿的位置，避免采到地面或背景。
        x_ratio = min(max(float(self.bar_depth_yaw_sample_x_ratio), 0.05), 0.45)
        y_ratio = min(max(float(self.bar_depth_yaw_sample_y_ratio), 0.02), 0.80)

        left_x = int(x1 + bw * x_ratio)
        right_x = int(x2 - bw * x_ratio)
        sample_y = int(y1 + bh * y_ratio)

        left_depth = self.sample_depth_patch_by_rgb(
            left_x,
            sample_y,
            self.bar_depth_yaw_sample_half_size
        )
        right_depth = self.sample_depth_patch_by_rgb(
            right_x,
            sample_y,
            self.bar_depth_yaw_sample_half_size
        )

        if left_depth is None or right_depth is None:
            self.get_logger().info(
                f'[BAR_DEPTH_YAW] invalid sample: left={left_depth}, right={right_depth}',
                throttle_duration_sec=0.5
            )
            return 0.0

        # 正值：左侧比右侧更远；负值：左侧比右侧更近。
        depth_error = float(left_depth - right_depth)
        if abs(depth_error) <= self.bar_depth_yaw_deadband_m:
            wz = 0.0
        else:
            # 默认符号：depth_error > 0 时给正 wz；如果实测反了，改 bar_depth_yaw_sign=-1。
            wz = math.copysign(self.bar_depth_yaw_fixed_wz, depth_error) * self.bar_depth_yaw_sign

        self.latest_bar_depth_yaw_info = {
            'left_depth': float(left_depth),
            'right_depth': float(right_depth),
            'depth_error': float(depth_error),
            'wz': float(wz),
            'left_point': (left_x, sample_y),
            'right_point': (right_x, sample_y),
        }

        self.get_logger().info(
            f'[BAR_DEPTH_YAW] left={left_depth:.3f}, right={right_depth:.3f}, '
            f'err=L-R={depth_error:.3f}m, deadband={self.bar_depth_yaw_deadband_m:.3f}m, '
            f'wz={wz:.3f}',
            throttle_duration_sec=0.3
        )
        return float(wz)

    # ---------- 参数 ----------
    def _declare_obstacle_params(self):
        p = self.declare_parameter
        p('obstacle.use_depth_filter', self.platform != 'real')
        p('obstacle.roi_x_ratio_min', 0.00)
        p('obstacle.roi_x_ratio_max', 1.00)
        p('obstacle.roi_y_ratio_min', 0.00)
        p('obstacle.roi_y_ratio_max', 1.00)

        p('obstacle.h_min', 90)
        p('obstacle.h_max', 140)
        p('obstacle.s_min', 60)
        p('obstacle.s_max', 255)
        p('obstacle.v_min', 40)
        p('obstacle.v_max', 255)

        p('obstacle.depth_min_m', 0.05)
        p('obstacle.depth_max_m', 1.50)

        p('obstacle.open_kernel', 3)
        p('obstacle.close_kernel', 5)

        p('obstacle.min_area', 7000 if self.platform == 'real' else 150)
        p('obstacle.max_area', 40000 if self.platform == 'real' else 1000000)
        p('obstacle.min_width', 10)
        p('obstacle.min_height', 10)
        p('obstacle.min_aspect_ratio', 0.7 if self.platform == 'real' else 0.0)
        p('obstacle.max_aspect_ratio', 1.3 if self.platform == 'real' else 4.5)
        p('obstacle.min_bottom_y_ratio_in_roi',
          0.5 if self.platform == 'real' else 0.2)

        p('obstacle.min_valid_depth_ratio', 0.20)
        p('obstacle.min_near_depth_ratio', 0.35)
        p('obstacle.bbox_depth_percentile', 50.0)
        p('obstacle.bbox_depth_margin_m', 0.10)
        p('obstacle.bbox_depth_min_pixels', 40)

    def _read_obstacle_cfg(self):
        gp = self.get_parameter
        return {
            'use_depth_filter': bool(gp('obstacle.use_depth_filter').value),
            'roi_x_ratio_min': float(gp('obstacle.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('obstacle.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('obstacle.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('obstacle.roi_y_ratio_max').value),

            'h_min': int(gp('obstacle.h_min').value),
            'h_max': int(gp('obstacle.h_max').value),
            's_min': int(gp('obstacle.s_min').value),
            's_max': int(gp('obstacle.s_max').value),
            'v_min': int(gp('obstacle.v_min').value),
            'v_max': int(gp('obstacle.v_max').value),

            'depth_min_m': float(gp('obstacle.depth_min_m').value),
            'depth_max_m': float(gp('obstacle.depth_max_m').value),

            'open_kernel': int(gp('obstacle.open_kernel').value),
            'close_kernel': int(gp('obstacle.close_kernel').value),

            'min_area': int(gp('obstacle.min_area').value),
            'max_area': int(gp('obstacle.max_area').value),
            'min_width': int(gp('obstacle.min_width').value),
            'min_height': int(gp('obstacle.min_height').value),
            'min_aspect_ratio': float(gp('obstacle.min_aspect_ratio').value),
            'max_aspect_ratio': float(gp('obstacle.max_aspect_ratio').value),
            'min_bottom_y_ratio_in_roi': float(gp('obstacle.min_bottom_y_ratio_in_roi').value),

            'min_valid_depth_ratio': float(gp('obstacle.min_valid_depth_ratio').value),
            'min_near_depth_ratio': float(gp('obstacle.min_near_depth_ratio').value),
            'bbox_depth_percentile': float(gp('obstacle.bbox_depth_percentile').value),
            'bbox_depth_margin_m': float(gp('obstacle.bbox_depth_margin_m').value),
            'bbox_depth_min_pixels': int(gp('obstacle.bbox_depth_min_pixels').value),
        }

    def _declare_yellow_params(self):
        p = self.declare_parameter

        # =========================
        # 黄色 HSV 参数
        # =========================
        p('yellow.h_min', 18)
        p('yellow.h_max', 45)
        p('yellow.s_min', 70)
        p('yellow.s_max', 255)
        p('yellow.v_min', 70)
        p('yellow.v_max', 255)

        # =========================
        # ROI 参数
        # 只看图像下方 60%~100% 区域
        # =========================
        p('yellow.roi_x_ratio_min', 0.00)
        p('yellow.roi_x_ratio_max', 1.00)
        p('yellow.roi_y_ratio_min', 0.60)
        p('yellow.roi_y_ratio_max', 1.00)

        # =========================
        # 基础黄色块过滤
        # =========================
        p('yellow.open_kernel', 3)
        p('yellow.min_area', 50)
        p('yellow.max_area', 4000)
        p('yellow.min_width', 3)
        p('yellow.min_height', 5)

        # =========================
        # 虚线形态学参数
        # =========================
        p('yellow.dash_close_kernel_h', 3)
        p('yellow.dash_close_kernel_w', 5)

        # =========================
        # 虚线组合参数
        # =========================
        p('yellow.dash_min_segments', 2)
        p('yellow.dash_min_total_span_y', 20)
        p('yellow.dash_max_adjacent_x_diff', 110)
        p('yellow.dash_max_gap_y', 3000)
        p('yellow.dash_min_gap_y', -10)
        p('yellow.dash_max_total_x_range', 5000)

        # =========================
        # 避免长条参与虚线组合
        # =========================
        p('yellow.dash_segment_max_aspect_ratio', 10.0)
        p('yellow.dash_segment_max_long_side', 200)

        # =========================
        # 多条虚线去重参数
        # =========================
        p('yellow.dash_duplicate_iou_thresh', 0.35)
        p('yellow.dash_duplicate_center_x_thresh', 30)

        # =========================
        # 最多显示 / 使用最长几条虚线
        # =========================
        p('yellow.max_dashed_lines', 2)

    def _read_yellow_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('yellow.h_min').value),
            'h_max': int(gp('yellow.h_max').value),
            's_min': int(gp('yellow.s_min').value),
            's_max': int(gp('yellow.s_max').value),
            'v_min': int(gp('yellow.v_min').value),
            'v_max': int(gp('yellow.v_max').value),

            'roi_x_ratio_min': float(gp('yellow.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('yellow.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('yellow.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('yellow.roi_y_ratio_max').value),

            'open_kernel': int(gp('yellow.open_kernel').value),
            'min_area': int(gp('yellow.min_area').value),
            'max_area': int(gp('yellow.max_area').value),
            'min_width': int(gp('yellow.min_width').value),
            'min_height': int(gp('yellow.min_height').value),

            'dash_close_kernel_h': int(gp('yellow.dash_close_kernel_h').value),
            'dash_close_kernel_w': int(gp('yellow.dash_close_kernel_w').value),

            'dash_min_segments': int(gp('yellow.dash_min_segments').value),
            'dash_min_total_span_y': int(gp('yellow.dash_min_total_span_y').value),
            'dash_max_adjacent_x_diff': int(gp('yellow.dash_max_adjacent_x_diff').value),
            'dash_max_gap_y': int(gp('yellow.dash_max_gap_y').value),
            'dash_min_gap_y': int(gp('yellow.dash_min_gap_y').value),
            'dash_max_total_x_range': int(gp('yellow.dash_max_total_x_range').value),

            'dash_segment_max_aspect_ratio': float(gp('yellow.dash_segment_max_aspect_ratio').value),
            'dash_segment_max_long_side': int(gp('yellow.dash_segment_max_long_side').value),

            'dash_duplicate_iou_thresh': float(gp('yellow.dash_duplicate_iou_thresh').value),
            'dash_duplicate_center_x_thresh': int(gp('yellow.dash_duplicate_center_x_thresh').value),

            'max_dashed_lines': int(gp('yellow.max_dashed_lines').value),
        }

    def _declare_final_yellow_params(self):
        p = self.declare_parameter

        # 最后阶段前方横向黄线：HSV 默认沿用黄色阈值，但单独开放参数便于调试
        p('final_yellow.h_min', 18)
        p('final_yellow.h_max', 45)
        p('final_yellow.s_min', 70)
        p('final_yellow.s_max', 255)
        p('final_yellow.v_min', 70)
        p('final_yellow.v_max', 255)

        # 看前方中下区域；比虚线 ROI 更靠上，方便提前看到横线
        p('final_yellow.roi_x_ratio_min', 0.30)
        p('final_yellow.roi_x_ratio_max', 0.70)
        p('final_yellow.roi_y_ratio_min', 0.50)
        p('final_yellow.roi_y_ratio_max', 1.00)

        p('final_yellow.open_kernel', 3)
        p('final_yellow.close_kernel_h', 5)
        p('final_yellow.close_kernel_w', 11)

        p('final_yellow.min_area', 3000)
        p('final_yellow.min_width', 20)
        p('final_yellow.min_height', 3)
        p('final_yellow.min_width_ratio', 0.70)
        p('final_yellow.min_wh_ratio', 1.5)
        p('final_yellow.max_tilt_deg', 35.0)

        # RGB-only: use bottom_y/image_height to judge distance to line
        p('final_yellow.center_tolerance_ratio', 0.60)

    def _read_final_yellow_cfg(self):
        gp = self.get_parameter
        return {
            'h_min': int(gp('final_yellow.h_min').value),
            'h_max': int(gp('final_yellow.h_max').value),
            's_min': int(gp('final_yellow.s_min').value),
            's_max': int(gp('final_yellow.s_max').value),
            'v_min': int(gp('final_yellow.v_min').value),
            'v_max': int(gp('final_yellow.v_max').value),

            'roi_x_ratio_min': float(gp('final_yellow.roi_x_ratio_min').value),
            'roi_x_ratio_max': float(gp('final_yellow.roi_x_ratio_max').value),
            'roi_y_ratio_min': float(gp('final_yellow.roi_y_ratio_min').value),
            'roi_y_ratio_max': float(gp('final_yellow.roi_y_ratio_max').value),

            'open_kernel': int(gp('final_yellow.open_kernel').value),
            'close_kernel_h': int(gp('final_yellow.close_kernel_h').value),
            'close_kernel_w': int(gp('final_yellow.close_kernel_w').value),

            'min_area': int(gp('final_yellow.min_area').value),
            'min_width': int(gp('final_yellow.min_width').value),
            'min_height': int(gp('final_yellow.min_height').value),
            'min_width_ratio': float(gp('final_yellow.min_width_ratio').value),
            'min_wh_ratio': float(gp('final_yellow.min_wh_ratio').value),
            'max_tilt_deg': float(gp('final_yellow.max_tilt_deg').value),
            'center_tolerance_ratio': float(gp('final_yellow.center_tolerance_ratio').value),
        }

    # ---------- 第二次转向后的目标检测参数 ----------
    def _declare_ball_params(self, prefix: str, defaults: Dict[str, Any]):
        for k, v in defaults.items():
            self.declare_parameter(f'{prefix}.{k}', v)

    def _read_ball_cfg(self, prefix: str):
        keys = [
            'h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max',
            'roi_x_ratio_min', 'roi_x_ratio_max', 'roi_y_ratio_min', 'roi_y_ratio_max',
            'open_kernel', 'close_kernel',
            'min_area', 'max_area',
            'min_radius', 'max_radius',
            'min_circularity',
            'min_wh_ratio', 'max_wh_ratio',
            'max_center_y_ratio_in_roi',
            'center_weight_base', 'center_weight_gain',
            'radius_score_gain',
        ]
        cfg = {}
        for k in keys:
            val = self.get_parameter(f'{prefix}.{k}').value
            cfg[k] = float(val) if isinstance(val, float) else int(val) if isinstance(val, int) else val
        return cfg

    def _declare_cola_params(self):
        p = self.declare_parameter
        # 兼容旧 launch/YAML；混合检测器不再用单一 HSV 盒子直接判定可乐。
        p('cola.h_min', 0)
        p('cola.h_max', 20)
        p('cola.s_min', 0)
        p('cola.s_max', 20)
        p('cola.v_min', 0)
        p('cola.v_max', 20)
        # 真机调试时可乐可能出现在画面边缘，因此使用完整图像。
        p('cola.roi_x_ratio_min', 0.0)
        p('cola.roi_x_ratio_max', 1.0)
        p('cola.roi_y_ratio_min', 0.0)
        p('cola.roi_y_ratio_max', 1.0)
        p('cola.cap_only_mode', self.platform == 'real')
        # The physical bottle has a visible red cap.  Disable the ambiguous
        # dark-only route on hardware, while preserving it for simulation.
        p('cola.enable_dark_shape', self.platform != 'real')

        p('cola.dark_v_max', 145 if self.platform == 'real' else 110)
        p('cola.dark_s_min', 20 if self.platform == 'real' else 35)
        p('cola.very_dark_v_max', 70 if self.platform == 'real' else 55)
        p('cola.open_kernel', 3)
        p('cola.close_kernel', 9 if self.platform == 'real' else 7)
        p('cola.min_area', 120)
        p('cola.max_area', 100000)
        p('cola.min_width', 6)
        p('cola.max_width', 300)
        p('cola.min_height', 24)
        p('cola.max_height', 470)
        p('cola.min_aspect', 1.90)
        p('cola.max_aspect', 5.5)
        p('cola.target_aspect', 3.0)
        p('cola.min_fill_ratio', 0.28)
        p('cola.max_fill_ratio', 0.95)
        p('cola.min_symmetry', 0.50)
        p('cola.min_shoulder_ratio', 0.18)
        p('cola.max_shoulder_ratio', 0.95)
        p('cola.min_solidity', 0.68)
        p('cola.min_bottom_ratio', 0.28)
        p('cola.full_size_area', 5000.0)
        p('cola.min_score', 55.0)

        # 真机主路径：红色瓶盖锚点 + 下方局部深色瓶身。
        p('cola.cap_h_low_max', 15)
        p('cola.cap_h_high_min', 165)
        p('cola.cap_s_min', 75)
        p('cola.cap_v_min', 55)
        p('cola.cap_min_area_ratio', 0.0003)
        p('cola.cap_max_area_ratio', 0.01)
        p('cola.cap_min_aspect', 0.45)
        p('cola.cap_max_aspect', 2.20)
        p('cola.cap_body_min_area', 60.0)
        p('cola.cap_body_area_gain', 3.5)
        p('cola.cap_body_min_width', 5)
        p('cola.cap_body_min_height', 18)
        p('cola.cap_bottle_min_aspect', 1.60)
        p('cola.cap_bottle_max_aspect', 5.5)

        p('cola.center_smoothing_alpha', 0.35)
        p('cola.max_center_jump_ratio', 0.25)
        p('cola.interface_stable_frames', 3)
        p('cola.detected_topic', '/stage4/vision/cola_detected')
        p('cola.detection_topic', '/stage4/vision/cola_detection')

    def _read_cola_cfg(self):
        gp = self.get_parameter
        keys = [
            'roi_x_ratio_min', 'roi_x_ratio_max',
            'roi_y_ratio_min', 'roi_y_ratio_max',
            'cap_only_mode', 'enable_dark_shape',
            'dark_v_max', 'dark_s_min', 'very_dark_v_max',
            'open_kernel', 'close_kernel',
            'min_area', 'max_area', 'min_width', 'max_width',
            'min_height', 'max_height',
            'min_aspect', 'max_aspect', 'target_aspect',
            'min_fill_ratio', 'max_fill_ratio', 'min_symmetry',
            'min_shoulder_ratio', 'max_shoulder_ratio',
            'min_solidity', 'min_bottom_ratio', 'full_size_area', 'min_score',
            'cap_h_low_max', 'cap_h_high_min', 'cap_s_min', 'cap_v_min',
            'cap_min_area_ratio', 'cap_max_area_ratio',
            'cap_min_aspect', 'cap_max_aspect',
            'cap_body_min_area', 'cap_body_area_gain',
            'cap_body_min_width', 'cap_body_min_height',
            'cap_bottle_min_aspect', 'cap_bottle_max_aspect',
            'center_smoothing_alpha', 'max_center_jump_ratio',
        ]
        return {key: gp(f'cola.{key}').value for key in keys}

    # ---------- ROS ----------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def state_elapsed_s(self) -> float:
        """
        返回当前状态已经运行的仿真时间。

        注意：使用 use_sim_time=True 时，节点刚初始化时 /clock 可能还没就绪，
        如果在 enter_state() 里直接记录 now_s()，可能会记录到 0.0。
        后续第一帧 control_loop 看到的 /clock 可能已经是几百秒，导致 elapsed 很大，
        固定时间状态会被瞬间跳过。

        所以这里采用延迟启动计时：
        第一次进入控制循环且 now_s()>0 时，才把当前仿真时间作为状态起点。
        """
        now = self.now_s()
        if now <= 0.0:
            return 0.0

        if self.state_enter_time is None or self.state_enter_time <= 0.0:
            self.state_enter_time = now
            self.get_logger().info(
                f'[STATE_TIMER] start {self.state} at sim_time={now:.3f}',
                throttle_duration_sec=1.0
            )
            return 0.0

        self.state_enter_time = self.align_motion_timer_start(
            self.state_enter_time, now)
        return max(0.0, now - self.state_enter_time)

    def ai_camera_callback(self, msg: Image):
        try:
            self.latest_ai_bgr = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
            self.latest_ai_rx_monotonic_s = time.monotonic()
        except Exception as exc:
            self.get_logger().error(f'AI camera convert failed: {exc}')

    def get_fresh_ai_frame(self):
        if (
            self.latest_ai_bgr is None
            or self.latest_ai_rx_monotonic_s is None
        ):
            return None
        age_s = time.monotonic() - self.latest_ai_rx_monotonic_s
        if age_s > self.basketball_ai_max_age_s:
            self.get_logger().warn(
                f'[BASKETBALL_AI] stale frame: age={age_s:.3f}s, '
                f'max={self.basketball_ai_max_age_s:.3f}s',
                throttle_duration_sec=1.0,
            )
            return None
        return self.latest_ai_bgr

    def fourth_rgb_callback(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB convert failed: {e}')

    def fourth_depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'DEPTH convert failed: {e}')

    def depth_to_meters(self, depth_img):
        if depth_img is None:
            return None

        if depth_img.dtype == np.float32:
            depth_m = depth_img.copy()
        elif depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def estimate_depth_at_center(self, center_img: Tuple[int, int]) -> Optional[float]:
        """
        复用限高杆代码的目标深度估计：
        在目标中心附近取一个 7x7 深度窗口，取较近的 20 分位数。
        """
        if self.latest_depth is None or self.latest_bgr is None:
            return None

        depth_m = self.depth_to_meters(self.latest_depth)
        if depth_m is None:
            return None

        dh, dw = depth_m.shape[:2]
        ih, iw = self.latest_bgr.shape[:2]
        cx, cy = center_img

        dx = int(cx * dw / max(iw, 1))
        dy = int(cy * dh / max(ih, 1))

        half = 3
        x1 = max(0, dx - half)
        x2 = min(dw, dx + half + 1)
        y1 = max(0, dy - half)
        y2 = min(dh, dy + half + 1)

        patch = depth_m[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]

        if valid.size == 0:
            return None

        return float(np.percentile(valid, 20))

    # ---------- 控制 ----------
    def _reset_pose_cache_to_normal(self):
        """同步软件侧姿态缓存为第四赛段 normal 姿态。"""
        self.body_height_cmd = float(self.p4_normal_body_height)
        self.body_roll_cmd = 0.0
        self.body_pitch_cmd = 0.0
        self.body_yaw_cmd = 0.0
        self.body_is_low = False

    def send_motion_cmd(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        *,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        body_height: Optional[float] = None,
        step_height: Optional[float] = None,
    ):
        """通过统一 adapter 同时发送速度、机身姿态和机身高度。

        ``None`` 只表示“不更新本地缓存”；实际下发的永远是 float。
        因此先设置 low/roll，后续只发送 vx/vy/wz 时，仍会持续携带之前的姿态。
        """
        if self.Ctrl is None:
            self.get_logger().warn(
                '[CMD] Robot_Ctrl is not active; motion command ignored',
                throttle_duration_sec=1.0,
            )
            return

        if roll is not None:
            self.body_roll_cmd = float(roll)
        if pitch is not None:
            self.body_pitch_cmd = float(pitch)
        if yaw is not None:
            self.body_yaw_cmd = float(yaw)
        if body_height is not None:
            self.body_height_cmd = float(body_height)
        if step_height is not None:
            self.step_height_cmd = float(step_height)

        vx = float(vx)
        vy = float(vy)
        wz = float(wz)
        self.motion_cmd = (vx, vy, wz)

        self.Ctrl.move(
            vx, vy, wz,
            step_height=float(self.step_height_cmd),
            roll=float(self.body_roll_cmd),
            pitch=float(self.body_pitch_cmd),
            yaw=float(self.body_yaw_cmd),
            body_height=float(self.body_height_cmd),
            legacy_gait_id=3,
        )

    def stop_motion(self):
        """只把 vx/vy/wz 清零，保持当前高度、roll 和 pitch。"""
        self.send_motion_cmd(0.0, 0.0, 0.0)
        self.get_logger().info(
            '[CMD] stop motion and keep body pose',
            throttle_duration_sec=1.0,
        )

    def stop(self):
        """兼容旧状态机调用：普通 STOP 只停止移动，不重置机身姿态。"""
        self.stop_motion()

    def recovery_stand(self, wait_finish: bool = True):
        """执行恢复站立；真机映射 MotionResultCmd motion_id=111。"""
        if self.Ctrl is None:
            self.get_logger().warn(
                '[RECOVERY] robot backend is not active; command ignored',
                throttle_duration_sec=1.0,
            )
            return False

        self.motion_cmd = (0.0, 0.0, 0.0)
        finished = bool(self.Ctrl.recovery_stand(wait_finish=wait_finish))
        if not finished:
            self.get_logger().warn('[RECOVERY] stand response timeout/failure')

        self._reset_pose_cache_to_normal()
        return finished

    def _clear_upright_action_payload(self):
        """Clear fields that must not leak from the preceding locomotion hit."""
        payload = {
            'contact': 0,
            'vel_des': [0.0, 0.0, 0.0],
            'rpy_des': [0.0, 0.0, 0.0],
            'pos_des': [0.0, 0.0, 0.0],
            'acc_des': [0.0] * 6,
            'ctrl_point': [0.0, 0.0, 0.0],
            'foot_pose': [0.0] * 6,
            'step_height': [0.0, 0.0],
            'value': 0,
            'duration': 0,
        }
        for name, value in payload.items():
            if hasattr(self.msg, name):
                setattr(self.msg, name, value)

    def _upright_action_ctrl(self):
        if self.upright_lcm_ctrl is not None:
            return self.upright_lcm_ctrl
        return self.Ctrl

    def _start_upright_lcm_controller(self) -> bool:
        if not getattr(self.Ctrl, 'is_real', False):
            return True
        if self.upright_lcm_ctrl is not None:
            return True

        try:
            self.Ctrl.stop_motion()
        except Exception as exc:
            self.get_logger().error(
                f'[HIT_UPRIGHT] failed to stop Motion Servo: {exc}')
            return False

        ctrl = Robot_Ctrl()
        try:
            ctrl.run()
        except Exception as exc:
            try:
                ctrl.quit()
            except Exception:
                pass
            self.get_logger().error(
                f'[HIT_UPRIGHT] failed to start dedicated LCM controller: {exc}')
            self._reset_pose_cache_to_normal()
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                body_height=self.p4_normal_body_height,
                step_height=0.02,
            )
            return False

        self.upright_lcm_ctrl = ctrl
        self.get_logger().warn(
            '[HIT_UPRIGHT] Motion Servo stopped; dedicated LCM controller owns '
            'robot_control_cmd until the upright sequence finishes')
        return True

    def _recover_upright_before_close(self) -> bool:
        ctrl = getattr(self, 'upright_lcm_ctrl', None)
        if ctrl is None:
            return True
        if not self._send_upright_action_mode(
            12, 0, 'HIT_UPRIGHT_STOP_RECOVERY'):
            return False

        deadline = time.monotonic() + self.hit_upright_phase_timeout_s
        retry_at = time.monotonic() + self.hit_upright_recovery_retry_after_s
        retried = False
        while time.monotonic() < deadline:
            status = self._get_lower_status()
            if self._lower_status_matches(
                status, 12, 0, min_progress=95
            ):
                self.get_logger().warn(
                    '[HIT_UPRIGHT] stop requested during action; '
                    'RecoveryStand confirmed before releasing LCM control')
                return True

            if (
                not retried
                and time.monotonic() >= retry_at
                and status is not None
                and int(status.get('mode', -1)) == 12
                and int(status.get('gait', -1)) == 0
                and int(status.get('progress', -1)) == 0
            ):
                retried = True
                self._send_upright_action_mode(
                    12, 0, 'HIT_UPRIGHT_STOP_RECOVERY_RETRY')
                deadline = time.monotonic() + self.hit_upright_phase_timeout_s
            time.sleep(0.02)

        self.get_logger().error(
            '[HIT_UPRIGHT] stop recovery could not be confirmed before timeout')
        return False
    def _close_upright_lcm_controller(self, resume_servo: bool) -> bool:
        ctrl = getattr(self, 'upright_lcm_ctrl', None)
        if ctrl is None:
            return True

        try:
            ctrl.quit()
        except Exception as exc:
            self.get_logger().error(
                f'[HIT_UPRIGHT] dedicated LCM controller quit failed: {exc}; '
                'Motion Servo will not be resumed')
            return False

        self.upright_lcm_ctrl = None
        if resume_servo and self.Ctrl is not None:
            self._reset_pose_cache_to_normal()
            self.step_height_cmd = 0.02
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                body_height=self.p4_normal_body_height,
                step_height=0.02,
            )
            self.get_logger().info(
                '[HIT_UPRIGHT] dedicated LCM controller stopped; Motion Servo resumed')
        return True
    def _send_upright_action_mode(
        self,
        mode: int,
        gait: int,
        label: str,
    ) -> bool:
        """Send one upright order through the transitional compatibility API."""
        ctrl = self._upright_action_ctrl()
        if ctrl is None:
            self.get_logger().error(
                f'[{label}] robot backend is not active; command not sent')
            return False


        self.motion_cmd = (0.0, 0.0, 0.0)
        self._clear_upright_action_payload()
        self.msg.mode = int(mode)
        self.msg.gait_id = int(gait)
        self._inc_life_count()
        ctrl.Send_cmd(self.msg)
        self.get_logger().info(
            f'[{label}] sent mode={mode} gait={gait} '
            f'life_count={self.msg.life_count}')
        return True

    def _get_lower_status(self) -> Optional[Dict[str, Any]]:
        ctrl = self._upright_action_ctrl()
        if ctrl is None or not hasattr(ctrl, 'get_status'):
            return None
        try:
            return ctrl.get_status()
        except Exception as exc:
            self.get_logger().error(
                f'[HIT_UPRIGHT] get_status failed: {exc}',
                throttle_duration_sec=1.0,
            )
            return None

    @staticmethod
    def _lower_status_has_fault(status: Dict[str, Any]) -> bool:
        return (
            int(status.get('ori_error', 0)) != 0
            or int(status.get('footpos_error', 0)) != 0
            or any(int(value) != 0 for value in status.get('motor_error', ()))
        )

    def _lower_status_matches(
        self,
        status: Optional[Dict[str, Any]],
        mode: int,
        gait: int,
        *,
        min_progress: Optional[int] = None,
        exact_progress: Optional[int] = None,
    ) -> bool:
        if status is None:
            return False
        if float(status.get('age_s', float('inf'))) > self.hit_upright_feedback_max_age_s:
            return False
        if int(status.get('mode', -1)) != int(mode):
            return False
        if int(status.get('gait', -1)) != int(gait):
            return False
        progress = int(status.get('progress', -1))
        if min_progress is not None and progress < int(min_progress):
            return False
        if exact_progress is not None and progress != int(exact_progress):
            return False
        return not self._lower_status_has_fault(status)

    def _start_hit_upright_recovery(self, reason: str):
        self.hit_upright_failure_reason = str(reason)
        self.get_logger().error(
            f'[HIT_UPRIGHT] {reason}; enter RecoveryStand fallback')
        if self.state != self.HIT_UPRIGHT_RECOVERY:
            self.enter_state(self.HIT_UPRIGHT_RECOVERY)

    def _process_hit_upright_state(self) -> bool:
        """Advance the basketball upright action without blocking ROS callbacks."""
        action_states = {
            self.HIT_UPRIGHT_PREPARE,
            self.HIT_UPRIGHT_RISE,
            self.HIT_UPRIGHT_HOLD,
            self.HIT_UPRIGHT_SMOOTH_RETURN,
            self.HIT_UPRIGHT_READY_STAND,
            self.HIT_UPRIGHT_RECOVERY,
        }
        if self.state not in action_states:
            return False


        status = self._get_lower_status()
        elapsed = self.state_elapsed_s()
        timeout_s = self.hit_upright_phase_timeout_s

        if self.state != self.HIT_UPRIGHT_RECOVERY and status is not None:
            age_s = float(status.get('age_s', float('inf')))
            if age_s != float('inf') and age_s > self.hit_upright_feedback_max_age_s:
                self._start_hit_upright_recovery(
                    f'lower feedback stale: age={age_s:.3f}s')
                return True
            if self._lower_status_has_fault(status):
                self._start_hit_upright_recovery(
                    'lower controller reported ori/foot/motor fault')
                return True

        if self.state == self.HIT_UPRIGHT_PREPARE:
            if self._lower_status_matches(
                status, 12, 0, min_progress=95):
                self.enter_state(self.HIT_UPRIGHT_RISE)
            elif (
                not self.hit_upright_retry_sent
                and elapsed >= self.hit_upright_recovery_retry_after_s
                and status is not None
                and int(status.get('mode', -1)) == 12
                and int(status.get('gait', -1)) == 0
                and int(status.get('progress', -1)) == 0
            ):
                self.hit_upright_retry_sent = True
                self._send_upright_action_mode(12, 0, 'HIT_UPRIGHT_PREPARE_RETRY')
            elif elapsed >= timeout_s:
                self._start_hit_upright_recovery('prepare stand timeout')

        elif self.state == self.HIT_UPRIGHT_RISE:
            if self._lower_status_matches(
                status, 64, 4, exact_progress=50):
                self.enter_state(self.HIT_UPRIGHT_HOLD)
            elif (
                status is not None
                and int(status.get('mode', -1)) == 64
                and int(status.get('gait', -1)) == 4
                and int(status.get('progress', -1)) >= 75
            ):
                self._start_hit_upright_recovery(
                    'gait 4 passed progress 50 instead of holding')
            elif elapsed >= timeout_s:
                self._start_hit_upright_recovery('upright rise timeout')

        elif self.state == self.HIT_UPRIGHT_HOLD:
            if not self._lower_status_matches(
                status, 64, 4, exact_progress=50):
                self._start_hit_upright_recovery(
                    'upright state changed during hold')
            elif elapsed >= self.hit_upright_hold_s:
                self.enter_state(self.HIT_UPRIGHT_SMOOTH_RETURN)

        elif self.state == self.HIT_UPRIGHT_SMOOTH_RETURN:
            if self._lower_status_matches(
                status, 3, 0, min_progress=95):
                self.enter_state(self.HIT_UPRIGHT_READY_STAND)
            elif elapsed >= timeout_s:
                self._start_hit_upright_recovery('smooth QpStand return timeout')

        elif self.state == self.HIT_UPRIGHT_READY_STAND:
            if self._lower_status_matches(status, 11, 3):
                self._close_upright_lcm_controller(resume_servo=True)
                self.get_logger().info(
                    '[HIT_UPRIGHT] sequence complete; continue original backoff flow')
                self.enter_state(self.HIT_BACKOFF_AFTER_HIT)
            elif elapsed >= timeout_s:
                self._start_hit_upright_recovery('ready stand timeout')

        elif self.state == self.HIT_UPRIGHT_RECOVERY:
            if self._lower_status_matches(
                status, 12, 0, min_progress=95):
                self.get_logger().warn(
                    '[HIT_UPRIGHT_RECOVERY] recovery confirmed; '
                    'continue through normal ready stand')
                self.enter_state(self.HIT_UPRIGHT_READY_STAND)
            elif (
                not self.hit_upright_retry_sent
                and elapsed >= self.hit_upright_recovery_retry_after_s
                and status is not None
                and int(status.get('mode', -1)) == 12
                and int(status.get('gait', -1)) == 0
                and int(status.get('progress', -1)) == 0
            ):
                self.hit_upright_retry_sent = True
                self._send_upright_action_mode(12, 0, 'HIT_UPRIGHT_RECOVERY_RETRY')
            elif elapsed >= timeout_s:
                # Do not start moving after an unconfirmed recovery. Keep the
                # RecoveryStand heartbeat active and wait for operator action.
                self.get_logger().error(
                    '[HIT_UPRIGHT_RECOVERY] recovery not confirmed; '
                    'holding RecoveryStand and refusing to continue movement',
                    throttle_duration_sec=1.0,
                )

        return True

    def set_body_pose(
        self,
        *,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        body_height: Optional[float] = None,
        do_stop: bool = False,
        reason: str = '',
    ):
        """使用 LCM 修改机身姿态；可选择先停止移动但保留当前姿态。"""
        if do_stop:
            self.stop_motion()

        self.send_motion_cmd(
            0.0,
            0.0,
            0.0,
            roll=roll,
            pitch=pitch,
            body_height=body_height,
        )

        suffix = f', reason={reason}' if reason else ''
        self.get_logger().warn(
            f'[BODY][{self.Ctrl.backend_name if self.Ctrl is not None else "none"}] roll={self.body_roll_cmd:.3f}, '
            f'pitch={self.body_pitch_cmd:.3f}, '
            f'height={self.body_height_cmd:.3f}{suffix}'
        )

    def set_body_low(
        self,
        do_stop: bool = True,
        reason: str = '',
        force: bool = False,
        body_height: Optional[float] = None,
    ):
        """通过 LCM 切换到第四赛段 low 姿态；限高杆与障碍物可使用不同高度。"""
        low_height = float(
            body_height if body_height is not None
            else getattr(self, 'bar_low_body_height', 0.15)
        )
        already_low = (
            getattr(self, 'body_is_low', False)
            and abs(float(getattr(self, 'body_height_cmd', low_height)) - low_height) < 1e-6
            and abs(float(getattr(self, 'body_roll_cmd', 0.0))) < 1e-6
            and abs(float(getattr(self, 'body_pitch_cmd', 0.0))) < 1e-6
        )
        if already_low and not force:
            return

        self.body_is_low = True
        self.set_body_pose(
            roll=0.0,
            pitch=0.0,
            body_height=low_height,
            do_stop=do_stop,
            reason=reason or 'set_body_low',
        )

    def set_body_normal(self, do_stop: bool = True, reason: str = '', force: bool = False):
        """通过 LCM 恢复第四赛段 normal 姿态。"""
        normal_height = float(getattr(self, 'p4_normal_body_height', 0.25))
        already_normal = (
            not getattr(self, 'body_is_low', False)
            and abs(float(getattr(self, 'body_height_cmd', normal_height)) - normal_height) < 1e-6
            and abs(float(getattr(self, 'body_roll_cmd', 0.0))) < 1e-6
            and abs(float(getattr(self, 'body_pitch_cmd', 0.0))) < 1e-6
        )
        if already_normal and not force:
            return

        self.body_is_low = False
        self.set_body_pose(
            roll=0.0,
            pitch=0.0,
            body_height=normal_height,
            do_stop=do_stop,
            reason=reason or 'set_body_normal',
        )

    def reset_body_pose_to_normal_at_start(self):
        """Stage4 activation: refresh normal pose without another RecoveryStand."""
        self.get_logger().info(
            '[BODY_INIT] refresh NORMAL pose only; startup RecoveryStand already handled by mission'
        )
        self.set_body_normal(
            do_stop=False,
            reason='stage4_start_normal',
            force=True,
        )

    def set_body_low_for_bar_trigger(self):
        """限高杆触发后同时设置机身高度与前倾角度。"""
        if not getattr(self, 'bar_body_low_enabled', True):
            return
        if getattr(self, 'bar_body_low_do_stop', True):
            self.stop_motion()
        self.send_motion_cmd(
            0.0, 0.0, 0.0,
            pitch=getattr(self, 'bar_target_forward_pitch', 0.15),
            body_height=getattr(self, 'bar_low_body_height', 0.20),
        )
        self.get_logger().warn(
            f'[BAR_POSE] set height and pitch together: '
            f'height={self.bar_low_body_height:.3f}, '
            f'pitch={self.bar_target_forward_pitch:.3f}'
        )

    def set_body_low_for_obstacle_flow(self):
        """障碍物全局居中完成、进入障碍物流程前切 low。"""
        if not getattr(self, 'obstacle_flow_low_enabled', True):
            return
        self.set_body_low(
            do_stop=getattr(self, 'obstacle_body_low_do_stop', True),
            reason='obstacle_centered_enter_flow',
            body_height=getattr(self, 'obstacle_low_body_height', 0.17),
        )

    def restore_body_normal_after_bar_flow(self):
        """限高杆流程结束后恢复 normal。"""
        if not getattr(self, 'restore_normal_after_bar_flow', True):
            return
        self.set_body_normal(
            do_stop=True,
            reason='bar_flow_finished',
            force=True,
        )

    def restore_body_normal_after_obstacle_final_turn(self):
        """障碍物流程最终横向黄线和 180 度掉头完成后恢复 normal。"""
        if not getattr(self, 'obstacle_restore_normal_after_final_turn', True):
            return
        self.set_body_normal(
            do_stop=True,
            reason='obstacle_final_180_turn_finished',
            force=True,
        )

    def _inc_life_count(self):
        self.msg.life_count += 1
        if self.msg.life_count > 127:
            self.msg.life_count = 0

    def send_left_jump_action_once(self):
        """执行一次左跳，然后恢复站立并重新下发 normal 姿态。"""
        self.motion_cmd = (0.0, 0.0, 0.0)
        self.Ctrl.run_action('left_jump', wait_finish=True)
        self.recovery_stand(wait_finish=True)
        self.set_body_normal(
            do_stop=False,
            reason='left_jump_recovery',
            force=True,
        )

    def send_right_jump_action_once(self):
        """执行一次右跳，然后恢复站立并重新下发 normal 姿态。"""
        self.motion_cmd = (0.0, 0.0, 0.0)
        self.Ctrl.run_action('right_jump', wait_finish=True)
        self.recovery_stand(wait_finish=True)
        self.set_body_normal(
            do_stop=False,
            reason='right_jump_recovery',
            force=True,
        )

    def execute_timed_turn_by_jump_count(self, jump_count: int, next_state: str, direction: int, label: str):
        """
        用固定角速度 + 固定仿真时间代替原来的旋转跳。

        direction: +1 表示左转，-1 表示右转。
        jump_count=1 使用 90 度参数；jump_count=2 使用 180 度参数。
        其他次数作为兜底：按 jump_count * 90 度时间执行。
        """
        count = max(0, int(jump_count))
        if count <= 0:
            self.get_logger().warn(f'[{label}] jump_count<=0, directly enter {next_state}')
            self.enter_state(next_state)
            return

        if count == 1:
            turn_name = '90'
            duration = self.p4_timed_turn_duration_90_s
            base_wz = self.p4_timed_turn_wz_90
        elif count == 2:
            turn_name = '180'
            duration = self.p4_timed_turn_duration_180_s
            base_wz = self.p4_timed_turn_wz_180
        else:
            turn_name = f'{count}x90'
            duration = count * self.p4_timed_turn_duration_90_s
            base_wz = self.p4_timed_turn_wz_90

        wz = float(direction) * abs(base_wz)
        elapsed = self.state_elapsed_s()

        if elapsed >= duration:
            self.get_logger().info(
                f'[{label}] timed turn done: count={count}, turn={turn_name}, '
                f'elapsed={elapsed:.3f}/{duration:.3f}s, next={next_state}'
            )
            self.enter_state(next_state)
            return

        self.send_fixed_turn_cmd(wz)
        self.get_logger().info(
            f'[{label}] timed turn running: count={count}, turn={turn_name}, '
            f'wz={wz:.3f}, elapsed={elapsed:.3f}/{duration:.3f}s',
            throttle_duration_sec=0.2
        )

    def send_fixed_turn_cmd(self, wz: float):
        """Send one turn command with the shared Stage-4 turn posture."""
        self.send_motion_cmd(
            0.0,
            0.0,
            float(wz),
            roll=self.p4_timed_turn_body_roll,
            pitch=self.p4_timed_turn_body_pitch,
            yaw=self.p4_timed_turn_body_yaw,
            body_height=self.p4_timed_turn_body_height,
            step_height=self.p4_timed_turn_step_height,
        )

    def execute_left_jump_turn(self, jump_count: int, next_state: str):
        """原来的左跳转向入口：现在改为固定角速度左转固定仿真时间。"""
        self.execute_timed_turn_by_jump_count(
            jump_count=jump_count,
            next_state=next_state,
            direction=+1,
            label='TIMED_LEFT_TURN'
        )

    def execute_right_jump_turn(self, jump_count: int, next_state: str):
        """原来的右跳转向入口：现在改为固定角速度右转固定仿真时间。"""
        self.execute_timed_turn_by_jump_count(
            jump_count=jump_count,
            next_state=next_state,
            direction=-1,
            label='TIMED_RIGHT_TURN'
        )

    @classmethod
    def get_all_state_names(cls) -> List[str]:
        """返回所有允许作为 initial_state 的状态名。"""
        return [
            cls.GLOBAL_INITIAL_LATERAL_SHIFT,
            cls.GLOBAL_LATERAL_SEARCH,
            cls.GLOBAL_CENTER_BAR,
            cls.GLOBAL_CENTER_OBSTACLE,
            cls.GLOBAL_SHIFT_AFTER_SUBTASK,
            cls.BAR_FORWARD_UNDER,
            cls.BAR_BODY_LOWER_WAIT,
            cls.BAR_CLEAR_AFTER_UNDER,
            cls.BAR_FORWARD_PITCH_WAIT,
            cls.BAR_SEARCH_TARGET,
            cls.BAR_APPROACH_TARGET,
            cls.BAR_HIT_TARGET,
            cls.BAR_BACKOFF_TO_BAR,
            cls.BAR_BACKOFF_TIMED,
            cls.BAR_BACKOFF_VOICE_WAIT,
            cls.BAR_TURN_TO_YELLOW,
            cls.BAR_YELLOW_FORWARD,
            cls.BAR_TURN_BACK,
            cls.BAR_FLOW_DONE,
            cls.OBSTACLE_FLOW_DONE,
            cls.APPROACH_OBSTACLES,
            cls.DASH_PRE_SIDE_SHIFT,
            cls.OBSTACLE_ROUTE_PRE_TURN1_STEP,
            cls.OBSTACLE_ROUTE_TURN_1,
            cls.OBSTACLE_ROUTE_LATERAL_SCAN,
            cls.OBSTACLE_ROUTE_FORWARD,
            cls.OBSTACLE_ROUTE_PRE_TURN2_LATERAL,
            cls.OBSTACLE_ROUTE_PRE_TURN2_STEP,
            cls.OBSTACLE_ROUTE_TURN_2,
            cls.SEARCH_TARGET_AFTER_TURNS,
            cls.APPROACH_AND_ALIGN_TARGET,
            cls.HIT_TARGET,
            cls.HIT_UPRIGHT_PREPARE,
            cls.HIT_UPRIGHT_RISE,
            cls.HIT_UPRIGHT_HOLD,
            cls.HIT_UPRIGHT_SMOOTH_RETURN,
            cls.HIT_UPRIGHT_READY_STAND,
            cls.HIT_UPRIGHT_RECOVERY,
            cls.HIT_BACKOFF_AFTER_HIT,
            cls.POST_HIT_LEFT_JUMP,
            cls.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
            cls.POST_HIT_OBSTACLE_VOICE_WAIT,
            cls.POST_HIT_OBS_TURN_1,
            cls.POST_HIT_OBS_FORWARD,
            cls.POST_HIT_OBS_TURN_2,
            cls.POST_HIT_PRE_FINAL_FORWARD,
            cls.POST_HIT_FINAL_FORWARD,
            cls.FINAL_LEFT_JUMP,
            cls.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN,
            cls.GLOBAL_FINAL_RIGHT_JUMP,
            cls.GLOBAL_FINAL_YELLOW_FORWARD,
            cls.GLOBAL_FINAL_LEFT_JUMP,
            cls.GLOBAL_FINAL_P3_ALIGN,
            cls.GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP,
            cls.DONE,
        ]

    @classmethod
    def p4_entry_table(cls) -> StageEntryTable:
        """第四赛段调试入口表。

        状态集合沿用 get_all_state_names()（即原有的合法 initial_state 列表），
        在它上面加一层具名入口，方便只调限高杆、只调障碍物或只调收尾。
        """
        dashed_note = 'dashed_side：用 debug_dashed_side:=left/right 指定'
        return StageEntryTable(
            cls.STAGE_ID,
            cls.GLOBAL_INITIAL_LATERAL_SHIFT,
            cls.get_all_state_names(),
            (
                EntryPoint('start', cls.GLOBAL_INITIAL_LATERAL_SHIFT,
                           '启动预横移，完整第四赛段'),
                EntryPoint('search', cls.GLOBAL_LATERAL_SEARCH,
                           '全局横移搜索限高杆/障碍物'),
                # 限高杆流程
                EntryPoint('bar_center', cls.GLOBAL_CENTER_BAR, '限高杆横向居中',
                           requires=('限高杆必须在前向 RGB 视野内',)),
                EntryPoint('bar', cls.BAR_FORWARD_UNDER, '低身穿过限高杆'),
                EntryPoint('bar_target', cls.BAR_SEARCH_TARGET, '杆后搜索目标'),
                EntryPoint('bar_hit', cls.BAR_HIT_TARGET, '杆后击打目标'),
                EntryPoint('bar_back', cls.BAR_BACKOFF_TO_BAR, '击打后退回限高杆'),
                EntryPoint('bar_yellow', cls.BAR_TURN_TO_YELLOW, '限高杆流程收尾：转向黄线'),
                # 障碍物流程
                EntryPoint('obstacle_center', cls.GLOBAL_CENTER_OBSTACLE,
                           '蓝色障碍物横向居中',
                           requires=('蓝色障碍物必须在前向 RGB 视野内',)),
                EntryPoint('obstacle', cls.APPROACH_OBSTACLES, '靠近障碍物'),
                EntryPoint('obstacle_route', cls.OBSTACLE_ROUTE_PRE_TURN1_STEP,
                           '绕障固定路线（第一次转向前）', requires=(dashed_note,)),
                EntryPoint('target', cls.SEARCH_TARGET_AFTER_TURNS,
                           '绕障后搜索目标'),
                EntryPoint('target_hit', cls.HIT_TARGET, '击打目标'),
                EntryPoint('upright', cls.HIT_UPRIGHT_PREPARE, '直立击打动作'),
                EntryPoint('post_hit', cls.HIT_BACKOFF_AFTER_HIT, '击打后后退'),
                EntryPoint('post_hit_obstacle',
                           cls.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
                           '击打后走向选定障碍物', requires=(dashed_note,)),
                # 收尾
                EntryPoint('final', cls.GLOBAL_FINAL_RIGHT_JUMP, '全局收尾右跳'),
                EntryPoint('final_yellow', cls.GLOBAL_FINAL_YELLOW_FORWARD,
                           '收尾沿黄线前进'),
                EntryPoint('final_align', cls.GLOBAL_FINAL_P3_ALIGN,
                           '收尾赛道对齐，之后上报完成'),
            ),
        )

    def enter_initial_state(self):
        """
        根据调试入口进入指定初始状态，方便单独调试某一段流程。

        说明：
        1. 正常跑完整流程时不传参数，从 GLOBAL_INITIAL_LATERAL_SHIFT 开始。
        2. 如果直接从依赖 dashed_side 的状态开始，建议同时传入：
           -p debug_dashed_side:=left
           或
           -p debug_dashed_side:=right
        """
        if self.debug_dashed_side in ('left', 'right'):
            self.dashed_side = self.debug_dashed_side
            self.get_logger().info(f"[INIT_STATE] debug_dashed_side={self.dashed_side}")
        elif self.debug_dashed_side != 'auto':
            self.get_logger().warn(
                f"[INIT_STATE] invalid debug_dashed_side='{self.debug_dashed_side}', use auto. "
                "valid values: auto / left / right"
            )
            self.debug_dashed_side = 'auto'

        states_need_dashed_side = [
            self.DASH_PRE_SIDE_SHIFT,
            self.OBSTACLE_ROUTE_PRE_TURN1_STEP,
            self.OBSTACLE_ROUTE_TURN_1,
            self.OBSTACLE_ROUTE_LATERAL_SCAN,
            self.OBSTACLE_ROUTE_FORWARD,
            self.OBSTACLE_ROUTE_PRE_TURN2_LATERAL,
            self.OBSTACLE_ROUTE_PRE_TURN2_STEP,
            self.OBSTACLE_ROUTE_TURN_2,
            self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
        ]
        if self.p4_initial_state in states_need_dashed_side and self.dashed_side not in ('left', 'right'):
            self.get_logger().warn(
                f"[INIT_STATE] {self.p4_initial_state} needs dashed_side, "
                "but dashed_side is None. You can run with: "
                "-p debug_dashed_side:=left or -p debug_dashed_side:=right"
            )

        self.get_logger().info(f"[INIT_STATE] start from: {self.p4_initial_state}")
        self.enter_state(self.p4_initial_state)

    def enter_state(self, new_state: str):
        self.state = new_state
        # 不在这里直接记录 now_s()。
        # 使用 Gazebo /clock 时，enter_state 可能发生在 /clock 首帧到达前，
        # 这会导致 state_enter_time=0，后续 elapsed 异常变大。
        # 让 state_elapsed_s() 在 control_loop 第一次真正执行时再开始计时。
        self.state_enter_time = None

        self.get_logger().info(f'ENTER STATE -> {new_state}')

        if new_state == self.GLOBAL_INITIAL_LATERAL_SHIFT:
            pass

        if new_state == self.GLOBAL_LATERAL_SEARCH:
            self.current_global_target = None
            self.global_center_stable_count = 0
            self.bar_center_stable_count = 0

        if new_state == self.GLOBAL_CENTER_BAR:
            self.global_center_stable_count = 0
            self.bar_center_stable_count = 0

        if new_state == self.GLOBAL_CENTER_OBSTACLE:
            self.global_center_stable_count = 0

        if new_state == self.GLOBAL_SHIFT_AFTER_SUBTASK:
            pass

        if new_state == self.BAR_FORWARD_UNDER:
            self.bar_center_stable_count = 0
            self.bar_trigger_confirm_count = 0
            self.target_stable_count = 0
            self.stable_target_type = None
            self.locked_target = None
            self.latest_target = None

        if new_state == self.BAR_BODY_LOWER_WAIT:
            # 同时下发限高杆流程的机身高度和前倾角度，不再分两次调整。
            self.set_body_low_for_bar_trigger()
            self.get_logger().warn(
                f'[BAR_BODY_LOWER_WAIT] wait for combined body pose: '
                f'{self.bar_body_lower_wait_s:.3f}s, '
                f'height_cmd={self.body_height_cmd:.3f}, '
                f'pitch_cmd={self.body_pitch_cmd:.3f}'
            )

        if new_state == self.BAR_CLEAR_AFTER_UNDER:
            # 高度和前倾已经同时到位，穿杆/清杆阶段继续保持该姿态。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            self.get_logger().warn(
                f'[BAR_CLEAR] start timed forward: '
                f'speed={self.bar_clear_forward_speed:.3f}, '
                f'time={self.bar_clear_forward_time_s:.3f}s'
            )

        if new_state == self.BAR_FORWARD_PITCH_WAIT:
            # 先停止，再下发前倾；姿态变化为渐变，因此单独等待。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            self.get_logger().warn(
                f'[BAR_PITCH] start forward pitch: '
                f'pitch={self.bar_target_forward_pitch:.3f}rad, '
                f'wait={self.bar_target_pitch_wait_s:.3f}s'
            )

        if new_state == self.BAR_SEARCH_TARGET:
            self.target_stable_count = 0
            self.stable_target_type = None
            self.locked_target = None
            self.latest_target = None

        if new_state == self.BAR_APPROACH_TARGET:
            self.bar_hit_start_pose = None

        if new_state == self.BAR_HIT_TARGET:
            pass

        if new_state == self.BAR_BACKOFF_TO_BAR:
            # 撞击后不恢复水平，整个回退阶段继续保持限高杆前倾姿态。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            self.get_logger().warn(
                '[BAR_BACKOFF] keep forward pitch during backing; restore after flow finishes'
            )

        if new_state == self.BAR_BACKOFF_VOICE_WAIT:
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            self.get_logger().warn('[BAR_BACKOFF_VOICE_WAIT] stop and announce detected bar')

        if new_state in (
                self.BAR_TURN_TO_YELLOW,
                self.BAR_TURN_BACK,
                self.BAR_FLOW_DONE):
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )

        if new_state == self.OBSTACLE_FLOW_DONE:
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.APPROACH_OBSTACLES:
            self.obstacle_pair_seen_in_approach = False
            self.obstacle_approach_pitch_latched = False
            # 障碍物居中完成后，从这里开始确认虚线方向。
            # 保留原检测器和原多虚线排序，直接使用 dashed_lines[0]。
            self.dashed_side_candidate = None
            self.dashed_side_candidate_count = 0
            forced_side = self.get_forced_dashed_side()
            self.dashed_side = forced_side

        if new_state == self.DASH_PRE_SIDE_SHIFT:
            # 从虚线预偏移开始进入前倾；后续对齐、沿线前进会持续保持该姿态。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.dashed_pre_shift_start_pose = None
            self.dashed_pre_shift_dir_sign = self.get_pre_shift_dir_sign()
            self.dashed_center_count = 0
            self.dashed_lost_count = 0
            self.get_logger().info(
                f'[DASH_PRE_SHIFT] enter: side={self.dashed_side}, '
                f'dir_sign={self.dashed_pre_shift_dir_sign:.1f}, '
                f'duration={self.dashed_pre_shift_duration_s:.3f}s'
            )

        if new_state == self.OBSTACLE_ROUTE_PRE_TURN1_STEP:
            # 预偏移结束后先用零速度 gait 原地踏步/停稳，并恢复水平 pitch；保持低机身。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=0.0,
                body_height=self.obstacle_low_body_height,
                step_height=self.p4_timed_turn_step_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_PRE_TURN1_STEP] enter: duration='
                f'{self.obstacle_route_pre_turn_step_duration_s:.3f}s; '
                f'cmd=(0,0,0), pitch=0.000'
            )

        if new_state == self.OBSTACLE_ROUTE_TURN_1:
            self.current_turn_dir = self.get_obstacle_route_first_turn_dir()
            self.current_turn_duration_s = self.obstacle_route_turn_duration_s
            self.current_turn_wz = self.obstacle_route_turn_wz
            # 转向阶段先恢复水平 pitch；保持低机身。转向完成进入横移后会重新前倾。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                roll=self.p4_timed_turn_body_roll,
                pitch=0.0,
                yaw=self.p4_timed_turn_body_yaw,
                body_height=self.obstacle_low_body_height,
                step_height=self.p4_timed_turn_step_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_TURN1] enter: side={self.dashed_side}, '
                f'dir={self.current_turn_dir:+d}, duration={self.current_turn_duration_s:.3f}s, '
                f'wz={self.current_turn_wz:.3f}, pitch=0.000'
            )

        if new_state == self.OBSTACLE_ROUTE_LATERAL_SCAN:
            self.obstacle_route_edge_confirm_count = 0
            self.obstacle_route_selected_obstacle = None
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )

        if new_state == self.OBSTACLE_ROUTE_FORWARD:
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )

        if new_state == self.OBSTACLE_ROUTE_PRE_TURN2_LATERAL:
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_PRE_TURN2_LATERAL] enter: side={self.dashed_side}, '
                f'duration={self.obstacle_route_pre_turn2_lateral_duration_s:.3f}s, '
                f'speed={self.obstacle_route_pre_turn2_lateral_speed:.3f}'
            )

        if new_state == self.OBSTACLE_ROUTE_PRE_TURN2_STEP:
            # 第二次转向前横移结束后先恢复水平 pitch，再原地踏步/停稳；TURN_2 继续保持水平并开始转。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=0.0,
                body_height=self.obstacle_low_body_height,
                step_height=self.p4_timed_turn_step_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_PRE_TURN2_STEP] enter: duration='
                f'{self.obstacle_route_pre_turn_step_duration_s:.3f}s; '
                f'cmd=(0,0,0), pitch=0.000'
            )

        if new_state == self.OBSTACLE_ROUTE_TURN_2:
            self.current_turn_dir = -self.get_obstacle_route_first_turn_dir()
            self.current_turn_duration_s = self.obstacle_route_turn_duration_s
            self.current_turn_wz = self.obstacle_route_turn_wz
            # 第二次转向同样保持水平 pitch；转完进入目标搜索时重新前倾。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                roll=self.p4_timed_turn_body_roll,
                pitch=0.0,
                yaw=self.p4_timed_turn_body_yaw,
                body_height=self.obstacle_low_body_height,
                step_height=self.p4_timed_turn_step_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_TURN2] enter: side={self.dashed_side}, '
                f'dir={self.current_turn_dir:+d}, duration={self.current_turn_duration_s:.3f}s, '
                f'wz={self.current_turn_wz:.3f}, pitch=0.000'
            )

        if new_state == self.ALIGN_DASHED_LINE:
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.dashed_center_count = 0
            self.dashed_lost_count = 0

        if new_state == self.FOLLOW_DASHED_UNTIL_LOST:
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.dashed_lost_count = 0

        if new_state == self.POST_DASH_FORWARD:
            pass

        if new_state == self.POST_DASH_TURN_1:
            self.turn_start_yaw = None
            self.current_turn_dir = self.get_first_turn_dir()
            self.current_turn_duration_s = self.post_dash_turn_duration_s
            self.current_turn_wz = self.post_dash_turn_wz

        if new_state == self.POST_TURN_FORWARD:
            pass

        if new_state == self.POST_DASH_TURN_2:
            self.turn_start_yaw = None
            self.current_turn_dir = -self.get_first_turn_dir()
            self.current_turn_duration_s = self.post_second_turn_duration_s
            self.current_turn_wz = self.post_second_turn_wz

        if new_state == self.SEARCH_TARGET_AFTER_TURNS:
            # 第二次转向结束、正式识别目标前，切换到目标识别专用前倾角度。
            # target_seen_after_turn 不在这里清零：若 APPROACH 阶段发生漏检并退回搜索，
            # 需要保留“此前已经看到过目标”的信息，从而改为后退重新搜索。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.obstacle_target_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.latest_target = None
            self.locked_target = None
            self.target_stable_count = 0
            self.stable_target_type = None
            self.hit_start_pose = None
            self.basketball_top_trigger_count = 0

        if new_state == self.APPROACH_AND_ALIGN_TARGET:
            self.hit_start_pose = None
            self.basketball_top_trigger_count = 0

        if new_state == self.HIT_TARGET:
            self.hit_start_pose = None
            self.get_logger().info(
                f'[HIT] start by sim time, target={self.locked_target.det_type if self.locked_target else None}')

        if new_state == self.HIT_UPRIGHT_PREPARE:
            self.hit_upright_failure_reason = ''
            self.hit_upright_retry_sent = False
            if not self._start_upright_lcm_controller():
                self.get_logger().error(
                    '[HIT_UPRIGHT] cannot acquire dedicated LCM control; '
                    'skip upright action and continue backoff')
                self.enter_state(self.HIT_BACKOFF_AFTER_HIT)
                return
            self._send_upright_action_mode(12, 0, 'HIT_UPRIGHT_PREPARE')

        if new_state == self.HIT_UPRIGHT_RISE:
            self._send_upright_action_mode(64, 4, 'HIT_UPRIGHT_RISE')

        if new_state == self.HIT_UPRIGHT_HOLD:
            self.get_logger().info(
                f'[HIT_UPRIGHT_HOLD] progress=50 confirmed; '
                f'hold={self.hit_upright_hold_s:.3f}s')

        if new_state == self.HIT_UPRIGHT_SMOOTH_RETURN:
            # The matching lower-controller patch converts this QpStand
            # request into TwoLegStand jump_flag 7 before switching states.
            self._send_upright_action_mode(3, 0, 'HIT_UPRIGHT_SMOOTH_RETURN')

        if new_state == self.HIT_UPRIGHT_READY_STAND:
            self._reset_pose_cache_to_normal()
            self._send_upright_action_mode(11, 3, 'HIT_UPRIGHT_READY_STAND')
            self.get_logger().info(
                '[HIT_UPRIGHT_READY_STAND] sent zero-velocity mode=11 gait=3')

        if new_state == self.HIT_UPRIGHT_RECOVERY:
            self.hit_upright_retry_sent = False
            self._send_upright_action_mode(12, 0, 'HIT_UPRIGHT_RECOVERY')

        if new_state == self.HIT_BACKOFF_AFTER_HIT:
            self.after_hit_backoff_start_pose = None
            self.selected_obstacle_after_hit = None

        if new_state == self.POST_HIT_LEFT_JUMP:
            self.selected_obstacle_after_hit = None

        if new_state == self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT:
            self.selected_obstacle_after_hit = None
            self.selected_obstacle_after_hit_side = None

        if new_state == self.POST_HIT_OBSTACLE_VOICE_WAIT:
            self.send_motion_cmd(0.0, 0.0, 0.0)
            self.get_logger().warn(
                '[POST_HIT_OBSTACLE_VOICE_WAIT] obstacle reached; stop and announce before turn'
            )

        if new_state == self.POST_HIT_OBS_TURN_1:
            self.turn_start_yaw = None
            self.current_turn_dir = self.get_post_hit_obs_first_turn_dir()
            self.current_turn_duration_s = self.post_hit_obs_turn_duration_s
            self.current_turn_wz = self.post_hit_obs_turn_wz
            self.get_logger().warn(
                f'[POST_HIT_OBS_TURN_1] use dashed_side to turn by sim time: '
                f'dashed_side={self.dashed_side}, turn_dir={self.current_turn_dir}, '
                f'duration={self.current_turn_duration_s:.3f}s, '
                f'wz_cmd={self.current_turn_dir * abs(self.current_turn_wz):.3f}'
            )

        if new_state == self.POST_HIT_OBS_FORWARD:
            pass

        if new_state == self.POST_HIT_OBS_TURN_2:
            self.turn_start_yaw = None
            self.current_turn_dir = -self.get_post_hit_obs_first_turn_dir()
            self.current_turn_duration_s = self.post_hit_obs_turn_duration_s
            self.current_turn_wz = self.post_hit_obs_turn_wz
            self.get_logger().warn(
                f'[POST_HIT_OBS_TURN_2] reverse first turn by sim time: '
                f'dashed_side={self.dashed_side}, turn_dir={self.current_turn_dir}, '
                f'duration={self.current_turn_duration_s:.3f}s, '
                f'wz_cmd={self.current_turn_dir * abs(self.current_turn_wz):.3f}'
            )

        if new_state == self.POST_HIT_PRE_FINAL_FORWARD:
            # 第二次转回后，先按仿真时间向前走一段。
            self.post_hit_pre_final_forward_start_pose = None
            self.post_hit_final_forward_start_pose = None

        if new_state in (
                self.POST_HIT_FINAL_FORWARD,
                self.BAR_YELLOW_FORWARD):
            # 固定距离前进完成后，再进入 RGB 横向黄线识别和朝向修正。
            # 新逻辑：先等黄线底部到达下方阈值，再继续前进等黄线从画面中消失。
            self.final_yellow_done_counter = 0
            self.final_yellow_reached_lower_area = False
            self.final_yellow_disappear_counter = 0
            self.latest_final_yellow_line = None

        if new_state == self.FINAL_LEFT_JUMP:
            # 障碍物流程内部的最终掉头动作，这里只负责进入状态；动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN:
            # 最终掉头完成后才恢复 normal。恢复姿态函数内部会 STOP。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_RIGHT_JUMP:
            # 全部流程完成后的最终右跳，动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_YELLOW_FORWARD:
            # 右跳后，开始识别前方横向黄线并修正朝向。
            # 新逻辑：黄线到达下方阈值后继续前进，直到黄线消失再停。
            self.global_final_yellow_done_counter = 0
            self.global_final_yellow_reached_lower_area = False
            self.global_final_yellow_disappear_counter = 0
            self.latest_final_yellow_line = None

        if new_state == self.GLOBAL_FINAL_LEFT_JUMP:
            # 最终左跳一次，动作在 control_loop 中执行。
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.GLOBAL_FINAL_P3_ALIGN:
            # 第四赛段最后左跳后，复用第三赛段结束的黄线近/远中心矫正逻辑。
            # 清空 P3 视觉缓存，避免刚切入时使用旧帧误差。
            self.p3_s4_lat = 0.0
            self.p3_s4_yaw = 0.0
            self.p3_s4_valid = 0.0
            self.p3_align_near_center = -1.0
            self.p3_align_far_center = -1.0
            self.send_motion_cmd(0.0, 0.0, 0.0)

        if new_state == self.DONE:
            self.task_done_stop_sent = False

    # ---------- TF 工具 ----------
    def normalize_angle(self, angle: float) -> float:
        """把角度归一化到 [-pi, pi]。"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q) -> float:
        """四元数转 yaw，不依赖 tf_transformations。"""
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_current_pose_2d(self):
        """
        从 TF 获取当前平面位姿。
        返回 (x, y, yaw)，失败返回 None。
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.tf_parent_frame,
                self.tf_child_frame,
                rclpy.time.Time()
            )
            x = float(tf.transform.translation.x)
            y = float(tf.transform.translation.y)
            yaw = self.quaternion_to_yaw(tf.transform.rotation)
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().warn(
                f'[TF] lookup {self.tf_parent_frame}->{self.tf_child_frame} failed: {e}',
                throttle_duration_sec=1.0
            )
            return None

    def get_current_yaw(self):
        pose = self.get_current_pose_2d()
        if pose is None:
            return None
        return pose[2]

    def distance_from_pose(self, start_pose) -> Optional[float]:
        """计算当前 TF 位置相对 start_pose 的平面位移距离。"""
        if start_pose is None:
            return None

        cur = self.get_current_pose_2d()
        if cur is None:
            return None

        dx = cur[0] - start_pose[0]
        dy = cur[1] - start_pose[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_first_turn_dir(self) -> int:
        """
        虚线在左边：第一次右转，返回 -1。
        虚线在右边：第一次左转，返回 +1。

        当前假设 wz > 0 是左转，wz < 0 是右转。
        如果实测方向反了，把这里的返回值正负号对调。
        """
        if self.dashed_side == 'left':
            return -1
        if self.dashed_side == 'right':
            return 1
        return 1

    def turn_finished_by_tf(self) -> bool:
        """用 TF yaw 判断当前转向是否达到目标角度。"""
        if self.turn_start_yaw is None:
            return False

        cur_yaw = self.get_current_yaw()
        if cur_yaw is None:
            return False

        signed_delta = self.normalize_angle(cur_yaw - self.turn_start_yaw)
        target = self.current_turn_angle_rad
        tol = self.current_turn_tolerance_rad

        if self.current_turn_dir > 0:
            done = signed_delta >= (target - tol)
        else:
            done = signed_delta <= -(target - tol)

        self.get_logger().info(
            f'[TF_TURN] dir={self.current_turn_dir}, '
            f'delta_deg={math.degrees(signed_delta):.1f}, '
            f'target_deg={math.degrees(target):.1f}, done={done}',
            throttle_duration_sec=0.2
        )
        return done

    # ---------- 对齐计算 ----------
    def choose_obstacle_pair(self, candidates: List[Detection]) -> Optional[Tuple[Detection, Detection]]:
        if len(candidates) < 2:
            return None

        # 统一选择外接矩形底边最靠近图像下方的两个障碍物。
        # bottom_y 相同时，再用面积作为次级排序，避免候选次序跳动。
        top = sorted(
            candidates,
            key=lambda d: (
                int(d.bbox_img[3]),
                float(d.extra.get('area', 0.0)),
            ),
            reverse=True
        )[:2]

        left, right = sorted(top, key=lambda d: d.center_img[0])
        return left, right

    def compute_obstacle_mid_align_vy(self, left: Detection, right: Detection) -> float:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        pair_center_x = (left.center_img[0] + right.center_img[0]) / 2.0

        err_px = pair_center_x - img_center_x

        if abs(err_px) <= self.obstacle_center_px_deadband:
            return 0.0

        err_norm = err_px / max(img_center_x, 1.0)

        # 如果方向反了，把这里的负号改成正号
        vy = -self.obstacle_align_vy_k * err_norm

        vy = float(np.clip(vy, -self.obstacle_align_vy_max, self.obstacle_align_vy_max))

        if 0.0 < abs(vy) < self.obstacle_align_vy_min:
            vy = math.copysign(self.obstacle_align_vy_min, vy)

        return vy

    def choose_selected_obstacle_after_hit(self, candidates: List[Detection]) -> Optional[Detection]:
        """
        左跳后再次前进识别障碍物时，只选择一个障碍物做居中对齐。

        规则：
        - 之前记录虚线在左边：选择右边障碍物居中
        - 之前记录虚线在右边：选择左边障碍物居中
        """
        pair = self.choose_obstacle_pair(candidates)
        if pair is None:
            return None

        left, right = pair
        if self.dashed_side == 'left':
            return right
        if self.dashed_side == 'right':
            return left

        # 兜底：如果没有记录虚线侧，就选更靠近图像中心的那个
        img_center_x = self.latest_bgr.shape[1] / 2.0
        return min(pair, key=lambda d: abs(d.center_img[0] - img_center_x))

    def get_obstacle_side_in_pair(self, obstacle: Detection, candidates: List[Detection]) -> str:
        """判断当前选中的障碍物是两障碍物中的左边还是右边。"""
        pair = self.choose_obstacle_pair(candidates)
        if pair is not None:
            left, right = pair
            if obstacle.center_img[0] <= left.center_img[0]:
                return 'left'
            if obstacle.center_img[0] >= right.center_img[0]:
                return 'right'

        img_center_x = self.latest_bgr.shape[1] / 2.0
        return 'left' if obstacle.center_img[0] < img_center_x else 'right'

    def get_post_hit_obs_first_turn_dir(self) -> int:
        """
        撞击后到达蓝色障碍物距离阈值后的第一次转向方向。

        现在不再根据 selected_obstacle_after_hit_side 判断，
        而是直接根据前面识别到的黄色竖直虚线方向 dashed_side 判断：

        - dashed_side == 'left'
            说明之前虚线在左边，流程里会偏向右侧障碍物路线，
            第一次转向按右转处理，返回 -1。

        - dashed_side == 'right'
            说明之前虚线在右边，流程里会偏向左侧障碍物路线，
            第一次转向按左转处理，返回 +1。

        当前假设 wz > 0 是左转，wz < 0 是右转。
        如果实测方向反了，只需要把这里的 +1 / -1 对调。
        """
        if self.dashed_side == 'right':
            return 1
        if self.dashed_side == 'left':
            return -1

        self.get_logger().warn(
            '[POST_HIT_OBS] dashed_side is None, fallback first turn LEFT',
            throttle_duration_sec=1.0
        )
        return 1

    def compute_selected_obstacle_align_vy_after_hit(self, obstacle: Detection) -> float:
        """左跳后，对单个选中的蓝色障碍物做居中对齐。"""
        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_px = obstacle.center_img[0] - img_center_x

        if abs(err_px) <= self.post_hit_obstacle_center_px_deadband:
            return 0.0

        err_norm = err_px / max(img_center_x, 1.0)

        # 如果方向反了，把这里的负号改成正号
        vy = -self.post_hit_obstacle_align_vy_k * err_norm
        vy = float(np.clip(vy, -self.post_hit_obstacle_align_vy_max, self.post_hit_obstacle_align_vy_max))

        if 0.0 < abs(vy) < self.post_hit_obstacle_align_vy_min:
            vy = math.copysign(self.post_hit_obstacle_align_vy_min, vy)

        return vy

    def get_obstacle_route_first_turn_dir(self) -> int:
        """新障碍物路线第一次90度转向的真机符号映射。

        实测原映射方向相反，因此这里反转：
        dashed_side='left'  -> dir=-1
        dashed_side='right' -> dir=+1
        第二次90度转向仍在状态逻辑中自动取相反方向。
        """
        if self.dashed_side == 'left':
            return -1
        if self.dashed_side == 'right':
            return 1
        return -1

    def get_obstacle_route_lateral_vy(self) -> float:
        """第一次90度转向后的横移方向按真机实测修正。

        转向符号保持修正版：left->dir=-1, right->dir=+1；
        但转向完成后的横移应恢复为与预偏移相同的 vy 符号：
        left->+vy, right->-vy。
        """
        return self.get_pre_shift_dir_sign() * abs(self.obstacle_route_lateral_speed)

    def choose_obstacle_route_single(self, candidates: List[Detection]) -> Optional[Detection]:
        """横移定位阶段只跟踪一个障碍物；多个候选时优先面积最大的一个。"""
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda d: (
                float(d.extra.get('area', 0.0)),
                float(d.score),
            )
        )

    def obstacle_route_edge_metrics(self, obstacle: Detection) -> Tuple[float, float, bool, str]:
        """返回 (edge_x, threshold_x, reached, edge_name)。

        left 路线：观察 bbox 左边缘 x1，横移时它向右进入阈值，x1 >= threshold。
        right 路线：镜像观察 bbox 右边缘 x2，x2 <= threshold。
        """
        image_w = float(self.latest_bgr.shape[1] if self.latest_bgr is not None else 640)
        x1, _, x2, _ = obstacle.bbox_img
        if self.dashed_side == 'left':
            edge_x = float(x1)
            threshold_x = image_w * self.obstacle_route_left_edge_trigger_ratio
            return edge_x, threshold_x, edge_x >= threshold_x, 'left_edge'
        edge_x = float(x2)
        threshold_x = image_w * self.obstacle_route_right_edge_trigger_ratio
        return edge_x, threshold_x, edge_x <= threshold_x, 'right_edge'

    def send_obstacle_route_turn_cmd(self, wz: float):
        """障碍物新路线90度转向时保持低机身，但 pitch=0，不前倾。"""
        self.send_motion_cmd(
            0.0, 0.0, float(wz),
            roll=self.p4_timed_turn_body_roll,
            pitch=0.0,
            yaw=self.p4_timed_turn_body_yaw,
            body_height=self.obstacle_low_body_height,
            step_height=self.p4_timed_turn_step_height,
        )

    def get_dashed_side(self, dashed: Detection) -> str:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        return 'left' if dashed.center_img[0] < img_center_x else 'right'

    def get_forced_dashed_side(self) -> Optional[str]:
        """
        debug_dashed_side 控制规则：
        - left/right：强制使用这个方向
        - auto：不强制，交给视觉判断
        """
        side = str(getattr(self, 'debug_dashed_side', 'auto')).strip().lower()
        if side in ('left', 'right'):
            return side
        return None

    def get_dashed_target_x(self) -> float:
        img_center_x = self.latest_bgr.shape[1] / 2.0
        if self.dashed_side == 'left':
            return img_center_x + self.dashed_target_offset_px
        if self.dashed_side == 'right':
            return img_center_x - self.dashed_target_offset_px
        return img_center_x

    def is_dashed_valid_for_follow(self, dashed: Optional[Detection]) -> bool:
        """
        FOLLOW_DASHED_UNTIL_LOST 阶段专用判断。

        对齐虚线完成后，机器狗沿虚线向前走。
        此时不能只要检测器检测到黄色虚线就认为还在跟随。

        只有当前检测到的虚线中心 x 落在“对齐目标线 target_x”附近一定范围内，
        才认为这条虚线还是当前正在跟随的那条线。

        如果虚线中心偏离 target_x 太远：
            - 不参与跟随修正；
            - 按 dashed lost 处理；
            - 连续 lost 若干帧后进入 POST_DASH_FORWARD。
        """
        if dashed is None:
            return False

        if self.latest_bgr is None:
            return False

        target_x = self.get_dashed_target_x()
        cx = float(dashed.center_img[0])
        valid_range = float(self.follow_dashed_valid_x_range_px)

        err_px = cx - target_x
        return abs(err_px) <= valid_range

    def get_pre_shift_dir_sign(self) -> float:
        """
        DASH_PRE_SIDE_SHIFT 的横移方向符号。

        设计意图：
          dashed_side == 'left'  -> 朝左侧虚线方向横移
          dashed_side == 'right' -> 朝右侧虚线方向横移

        注意：如果实测方向反了，只需要把这里的 1.0 和 -1.0 对调。
        """
        if self.dashed_side == 'left':
            return 1.0
        if self.dashed_side == 'right':
            return -1.0
        return 0.0

    def get_pre_shift_vy(self) -> float:
        return self.get_pre_shift_dir_sign() * abs(self.dashed_pre_shift_speed)

    def get_local_lateral_displacement_from_start(self, start_pose, current_pose) -> float:
        """
        计算 current_pose 相对 start_pose 的横向位移。

        start_pose/current_pose 格式为 (x, y, yaw)。
        返回的是以 start_pose 的 yaw 为基准的侧向位移，避免把前后漂移算进预横移距离。
        """
        if start_pose is None or current_pose is None:
            return 0.0

        sx, sy, syaw = start_pose
        cx, cy, _ = current_pose

        dx = cx - sx
        dy = cy - sy

        # 起始朝向左侧法向量：(-sin(yaw), cos(yaw))
        lateral = -math.sin(syaw) * dx + math.cos(syaw) * dy
        return float(lateral)

    def compute_dashed_align_vy(
            self,
            dashed: Detection,
            k: Optional[float] = None,
            vy_max: Optional[float] = None,
            vy_min: Optional[float] = None,
    ) -> float:
        if k is None:
            k = self.dashed_align_vy_k
        if vy_max is None:
            vy_max = self.dashed_align_vy_max
        if vy_min is None:
            vy_min = self.dashed_align_vy_min

        # 不再对齐图像中心，而是对齐偏置目标点
        target_x = self.get_dashed_target_x()
        err_px = dashed.center_img[0] - target_x

        if abs(err_px) <= self.dashed_center_px_deadband:
            return 0.0

        img_center_x = self.latest_bgr.shape[1] / 2.0
        err_norm = err_px / max(img_center_x, 1.0)

        # 如果方向反了，把这里的负号改成正号
        vy = -float(k) * err_norm

        vy = float(np.clip(vy, -float(vy_max), float(vy_max)))

        if 0.0 < abs(vy) < float(vy_min):
            vy = math.copysign(float(vy_min), vy)

        return vy

    def is_dashed_centered(self, dashed: Detection) -> bool:
        target_x = self.get_dashed_target_x()
        err_px = dashed.center_img[0] - target_x
        return abs(err_px) <= self.dashed_center_px_deadband

    def compute_final_yellow_wz(self, yellow_line: Optional[Detection]) -> float:
        """
        根据前方横向黄线倾斜角修正 yaw。
        angle_deg 来自图像坐标系中的黄线斜率，0 度表示水平。

        当前符号：wz = -k * angle。
        如果实测发现越修越歪，把下面的负号改成正号即可。
        """
        if yellow_line is None:
            return 0.0

        angle_deg = float(yellow_line.extra.get('angle_deg', 0.0))
        if abs(angle_deg) <= self.final_yellow_tilt_deadband_deg:
            return 0.0

        angle_rad = math.radians(angle_deg)
        wz = -self.final_yellow_align_wz_k * angle_rad
        wz = float(np.clip(wz, -self.final_yellow_align_wz_max, self.final_yellow_align_wz_max))

        if 0.0 < abs(wz) < self.final_yellow_align_wz_min:
            wz = math.copysign(self.final_yellow_align_wz_min, wz)

        return wz

    def get_global_final_yellow_forward_speed(self, final_yellow_line: Optional[Detection]) -> float:
        """
        全局最终横向黄线阶段的前进速度选择。

        逻辑：
        1. 没看到黄线时：使用快速速度 global_final_yellow_forward_speed
        2. 看到黄线但还没到减速阈值：继续快速
        3. 黄线 bottom_ratio >= global_final_yellow_slow_start_ratio：切换慢速

        bottom_ratio 含义：
            bottom_ratio = 黄线检测框底部 y 坐标 / 图像高度

        例如图像高度 480，slow_start_ratio=0.85：
            480 * 0.85 = 408
        也就是横向黄线底部到达 y=408 附近后开始减速。
        """
        if final_yellow_line is None:
            return self.global_final_yellow_forward_speed

        bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))

        if bottom_ratio >= self.global_final_yellow_slow_start_ratio:
            return self.global_final_yellow_slow_forward_speed

        return self.global_final_yellow_forward_speed

    # ---------- 第二次转向后的目标检测 / 对齐 / 撞击 ----------
    def publish_cola_interface(
        self,
        cola: Optional[Detection],
        frame_bgr,
    ):
        """发布可供播报/调试节点使用的稳定检测结果和中心偏移。"""
        if cola is None:
            self.cola_interface_count = 0
        else:
            self.cola_interface_count += 1

        stable = (
            cola is not None
            and self.cola_interface_count >= self.cola_interface_stable_frames
        )

        detected_msg = Bool()
        detected_msg.data = bool(stable)
        self.cola_detected_pub.publish(detected_msg)

        height, width = frame_bgr.shape[:2]
        payload = {
            'detected': bool(stable),
            'raw_detected': cola is not None,
            'stable_frames': int(self.cola_interface_count),
            'required_stable_frames': int(self.cola_interface_stable_frames),
            'image_width': int(width),
            'image_height': int(height),
        }

        if cola is not None:
            image_center_x = width / 2.0
            offset_x_px = float(
                cola.extra.get(
                    'offset_x_px',
                    cola.center_img[0] - image_center_x,
                )
            )
            payload.update({
                'center_x': int(cola.center_img[0]),
                'center_y': int(cola.center_img[1]),
                'bbox': [int(value) for value in cola.bbox_img],
                'score': float(cola.score),
                'method': str(cola.extra.get('method', 'unknown')),
                'offset_x_px': offset_x_px,
                'offset_x_norm': float(
                    cola.extra.get(
                        'offset_x_norm',
                        offset_x_px / max(image_center_x, 1.0),
                    )
                ),
                'roi_bbox': [
                    int(value)
                    for value in cola.extra.get(
                        'roi_bbox',
                        (0, 0, width, height),
                    )
                ],
            })

        detail_msg = String()
        detail_msg.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(',', ':'),
        )
        self.cola_detection_pub.publish(detail_msg)

    def detect_all_targets(self, frame_bgr) -> List[Detection]:
        """
        复用限高杆任务代码的目标检测逻辑：
        同时检测蓝球、白球、可乐，返回所有检测到的目标。
        """
        detections = []

        ai_frame = self.get_fresh_ai_frame()
        if ai_frame is not None:
            blue_ball = self.blue_ball_detector.detect(ai_frame)
            if blue_ball is not None:
                ai_h, ai_w = ai_frame.shape[:2]
                blue_ball.extra.update({
                    'source': 'ai_camera',
                    'image_width': int(ai_w),
                    'image_height': int(ai_h),
                    'top_y_ratio': (
                        float(blue_ball.bbox_img[1]) / float(max(ai_h, 1))),
                })
                detections.append(blue_ball)

        def football_depth_at(x, y):
            depth_m = self.estimate_depth_at_center((int(round(x)), int(round(y))))
            return -1.0 if depth_m is None else depth_m

        white_ball = self.white_ball_detector.detect(
            frame_bgr, depth_at=football_depth_at)
        if white_ball is not None:
            detections.append(white_ball)

        image_center_x = frame_bgr.shape[1] * 0.5
        max_cap_area_ratio = max(
            float(self.cola_detector.impl.cfg['cap_max_area_ratio']), 1e-9)
        max_area_bonus_px = frame_bgr.shape[1] * 0.05

        def nearest_cola_key(candidate):
            centerline_distance_px = abs(
                float(candidate['center'][0]) - image_center_x)
            cap_area_ratio = float(candidate.get('cap_area_ratio', 0.0))
            area_bonus_px = max_area_bonus_px * min(
                cap_area_ratio / max_cap_area_ratio, 1.0)
            selection_cost = centerline_distance_px - area_bonus_px
            candidate['selection_centerline_distance_px'] = (
                centerline_distance_px)
            candidate['selection_area_bonus_px'] = area_bonus_px
            return (selection_cost, centerline_distance_px,
                    -float(candidate['score']))

        cola = self.cola_detector.detect(
            frame_bgr, candidate_key=nearest_cola_key)
        self.publish_cola_interface(cola, frame_bgr)
        if cola is not None:
            detections.append(cola)

        return detections

    def choose_best_target(self, candidates: List[Detection]) -> Optional[Detection]:
        """
        和限高杆代码一样，优先选择最靠近图像中心的目标。
        """
        if not candidates:
            return None

        basketball = [c for c in candidates if c.det_type == 'blue_ball']
        pool = basketball if basketball else candidates

        def center_distance(candidate: Detection) -> float:
            fallback_width = (
                self.latest_bgr.shape[1] if self.latest_bgr is not None else 1)
            image_width = float(candidate.extra.get(
                'image_width', fallback_width))
            return abs(float(candidate.center_img[0]) - image_width / 2.0)

        return min(pool, key=center_distance)

    def bbox_top_y_ratio(self, det: Optional[Detection]) -> Optional[float]:
        if det is None:
            return None
        fallback_height = (
            self.latest_bgr.shape[0] if self.latest_bgr is not None else 1)
        image_height = float(det.extra.get('image_height', fallback_height))
        return float(det.bbox_img[1]) / float(max(image_height, 1.0))

    def bbox_bottom_y_ratio(self, det: Optional[Detection]) -> Optional[float]:
        if det is None:
            return None
        fallback_height = (
            self.latest_bgr.shape[0] if self.latest_bgr is not None else 1)
        image_height = float(det.extra.get('image_height', fallback_height))
        return float(det.bbox_img[3]) / float(max(image_height, 1.0))

    def target_visual_metrics(self, target: Detection) -> Dict[str, Optional[float]]:
        if target.det_type == 'cola':
            area_ratio = target.extra.get('cap_area_ratio')
            radius = None
        else:
            area_ratio = target.extra.get('area_ratio')
            radius = target.extra.get('radius')
        return {
            'area_ratio': (float(area_ratio) if area_ratio is not None else None),
            'radius_px': (float(radius) if radius is not None else None),
        }

    def target_visual_threshold_reached(
            self, target: Detection, phase: str) -> bool:
        if target.det_type == 'blue_ball':
            top_y_ratio = self.bbox_top_y_ratio(target)
            threshold = (
                self.basketball_top_slow_y_ratio
                if phase == 'slow'
                else self.basketball_top_trigger_y_ratio
            )
            return top_y_ratio is not None and top_y_ratio <= threshold
        thresholds = self.target_visual_thresholds.get(target.det_type)
        if thresholds is None:
            return False
        metrics = self.target_visual_metrics(target)
        area_ratio = metrics['area_ratio']
        if area_ratio is None or area_ratio < thresholds[f'{phase}_area_ratio']:
            return False
        if target.det_type == 'cola':
            return True
        radius = metrics['radius_px']
        return (
            radius is not None
            and radius >= thresholds[f'{phase}_radius_px'])

    def compute_target_align_cmd(self, target: Detection):
        """
        复用限高杆代码中的目标接近对齐逻辑：
        - 根据目标中心 x 偏差算横移 vy
        - 真机根据目标面积/半径选择远近前进速度，仿真保留深度判断
        """
        if self.latest_bgr is None:
            return 0.0, 0.0

        if target.det_type == 'blue_ball' or self.use_rgb_distance_triggers:
            near_target = self.target_visual_threshold_reached(target, 'slow')
        else:
            depth_m = self.estimate_depth_at_center(target.center_img)
            near_target = depth_m is not None and depth_m < 0.35

        image_width = float(target.extra.get(
            'image_width', self.latest_bgr.shape[1]))
        img_center_x = image_width / 2.0
        err_px = target.center_img[0] - img_center_x

        if near_target:
            vx = self.align_forward_speed_near
        else:
            vx = self.align_forward_speed_far

        if abs(err_px) <= self.center_px_deadband:
            vy = 0.0
        else:
            err_norm = err_px / max(img_center_x, 1.0)

            # 和你前面所有视觉对齐一致：如果方向反了，把负号改成正号
            vy = -self.align_vy_k * err_norm
            vy = float(np.clip(vy, -self.align_vy_max, self.align_vy_max))

            if 0.0 < abs(vy) < self.align_vy_min:
                vy = math.copysign(self.align_vy_min, vy)

        return vx, vy

    # ---------- Stage4 RGB 新帧门控 ----------
    def _p4_visual_state_requires_fresh_rgb(self) -> bool:
        """Return True only for states whose decision depends on current RGB.

        Pure timed-motion / voice / turn states are intentionally excluded so
        their timers remain independent of camera FPS.
        """
        visual_states = {
            self.GLOBAL_LATERAL_SEARCH,
            self.GLOBAL_CENTER_BAR,
            self.GLOBAL_CENTER_OBSTACLE,
            self.BAR_FORWARD_UNDER,
            self.BAR_SEARCH_TARGET,
            self.BAR_APPROACH_TARGET,
            self.BAR_BACKOFF_TO_BAR,
            self.APPROACH_OBSTACLES,
            self.OBSTACLE_ROUTE_LATERAL_SCAN,
            self.SEARCH_TARGET_AFTER_TURNS,
            self.APPROACH_AND_ALIGN_TARGET,
            self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
            self.POST_HIT_FINAL_FORWARD,
            self.BAR_YELLOW_FORWARD,
            self.GLOBAL_FINAL_YELLOW_FORWARD,
            self.GLOBAL_FINAL_P3_ALIGN,
        }

        # This state is mainly timed motion.  It only needs fresh RGB when its
        # optional yellow-line angle correction is enabled.
        if self.state == self.POST_HIT_PRE_FINAL_FORWARD:
            return bool(self.post_hit_pre_final_angle_align_enabled)

        return self.state in visual_states

    def _p4_accept_fresh_rgb_for_visual_state(self) -> bool:
        """Run perception once per helper-received RGB frame, with watchdog."""
        if not self._p4_visual_state_requires_fresh_rgb():
            return True

        now_mono = time.monotonic()
        rgb_seq = int(getattr(self, 'latest_rgb_seq', -1))
        age_s = self.rgb_age_s()

        if age_s is None:
            self.send_motion_cmd(0.0, 0.0, 0.0)
            self.get_logger().warning(
                f'[P4_RGB_GATE] state={self.state}: dedicated receiver has no RGB yet; stop and wait',
                throttle_duration_sec=0.5
            )
            return False

        # Use actual helper-node receive time.  This catches the important case
        # where the state spent several seconds in a non-visual timed action and
        # the RGB receiver died during that interval.
        if age_s >= self.p4_rgb_stale_stop_s:
            self.send_motion_cmd(0.0, 0.0, 0.0)
            self._p4_rgb_stale_stop_active = True
            thread_alive = bool(
                self._p4_rgb_rx_thread is not None
                and self._p4_rgb_rx_thread.is_alive()
            )
            self.get_logger().warning(
                f'[P4_RGB_STALE_STOP] state={self.state}, '
                f'dedicated RGB age={age_s:.3f}s >= {self.p4_rgb_stale_stop_s:.3f}s, '
                f'rgb_seq={rgb_seq}, rx_thread_alive={thread_alive}; cmd=(0,0,0)',
                throttle_duration_sec=0.5
            )

            # If the topic is alive but this subscription has wedged, rebuild
            # only the tiny receiver node/executor.  Cooldown avoids restart
            # storms if the camera really is unavailable.
            restart_after = max(2.0, 2.0 * self.p4_rgb_stale_stop_s)
            restart_cooldown = 3.0
            since_restart = now_mono - self._p4_rgb_last_restart_monotonic_s
            if age_s >= restart_after and since_restart >= restart_cooldown:
                self._p4_restart_rgb_receiver(
                    f'age={age_s:.3f}s state={self.state}'
                )
            return False

        # Receiver is healthy.  Only run the expensive perception once for each
        # Stage4-consumed frame.
        if rgb_seq == self._p4_last_processed_rgb_seq:
            return False

        self._p4_last_processed_rgb_seq = rgb_seq
        if self._p4_rgb_stale_stop_active:
            self.get_logger().warning(
                f'[P4_RGB_GATE] RGB recovered: state={self.state}, '
                f'rgb_seq={rgb_seq}, age={age_s:.3f}s'
            )
        self._p4_rgb_stale_stop_active = False
        return True

    # ---------- 主循环 ----------
    def fourth_control_loop(self):
        # Pull the newest raw message from the dedicated RGB receiver on every
        # control tick, including pure timed-motion states.  Therefore RGB age
        # remains truthful across long turns/backoffs that do not use vision.
        self._p4_sync_rgb_from_receiver()

        # The basketball upright action is driven entirely by LCM feedback and
        # must keep advancing even if RGB/depth frames temporarily stop.
        if self._process_hit_upright_state():
            return

        # GLOBAL_INITIAL_LATERAL_SHIFT is a pure timed-motion state.  It must
        # not depend on RGB/depth availability.  In particular, when Stage4
        # is launched directly on the physical robot the camera streams may
        # need a moment to arrive; waiting for them here used to leave the
        # robot in an accepted 303 gait with zero velocity (stepping in place)
        # and the lateral-shift timer never even started.
        if self.state == self.GLOBAL_INITIAL_LATERAL_SHIFT:
            elapsed = self.state_elapsed_s()
            if self.global_initial_lateral_shift_duration_s <= 0.0:
                self.get_logger().info('[GLOBAL_INIT_SHIFT] duration <= 0, skip initial shift')
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            if elapsed >= self.global_initial_lateral_shift_duration_s:
                self.get_logger().info(
                    f'[GLOBAL_INIT_SHIFT] finished: elapsed={elapsed:.3f}/{self.global_initial_lateral_shift_duration_s:.3f}s, '
                    'enter GLOBAL_LATERAL_SEARCH'
                )
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            self.send_motion_cmd(0.0, self.global_initial_lateral_shift_vy, 0.0)
            self.get_logger().info(
                f'[GLOBAL_INIT_SHIFT] shifting left by motion time: elapsed={elapsed:.3f}/{self.global_initial_lateral_shift_duration_s:.3f}s, '
                f'vy={self.global_initial_lateral_shift_vy:.3f}',
                throttle_duration_sec=0.3
            )
            return

        # 真机的主要距离触发已改为 RGB 几何量；深度只保留给限高杆结构、
        # 回退和可选的左右深度差纠偏，缺少深度时不再卡住整个赛段。
        if self.latest_bgr is None:
            self.get_logger().warn(
                '[P4 VISION] waiting for RGB frame before vision-driven state',
                throttle_duration_sec=1.0
            )
            return

        if (not self.use_rgb_distance_triggers and self.latest_depth is None):
            self.get_logger().warn(
                '[P4 VISION] waiting for depth frame in depth-trigger mode',
                throttle_duration_sec=1.0
            )
            return

        if (not self.use_rgb_distance_triggers
                and self.latest_bgr.shape[:2] != self.latest_depth.shape[:2]):
            self.get_logger().warn(
                f'RGB size {self.latest_bgr.shape[:2]} != DEPTH size {self.latest_depth.shape[:2]}',
                throttle_duration_sec=1.0
            )
            return

        # Vision-driven states are allowed to process each received RGB message
        # at most once.  Repeated 30 Hz timer callbacks on the same cached frame
        # return immediately, which prevents perception from starving rgb_callback
        # on a SingleThreadedExecutor.
        if not self._p4_accept_fresh_rgb_for_visual_state():
            return

        frame = self.latest_bgr
        now = self.now_s()

        # Do not run the obstacle detector globally on every Stage4 timer tick.
        # Only the states that actually consume obstacle_candidates pay this cost.
        need_obstacle_detection = (
            self.state in {
                self.GLOBAL_LATERAL_SEARCH,
                self.GLOBAL_CENTER_OBSTACLE,
                self.APPROACH_OBSTACLES,
                self.OBSTACLE_ROUTE_LATERAL_SCAN,
                self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT,
            }
            and (
                self.completed_obstacle_count < self.required_obstacle_count
                or self.is_obstacle_flow_state()
            )
        )

        if need_obstacle_detection:
            obstacle_result = self.obstacle_detector.detect(
                frame, self.latest_depth
            )
            obstacle_candidates = obstacle_result['candidates']
        else:
            obstacle_candidates = []

        # 新障碍物路线里，黄色虚线只在靠近两个障碍物时用于锁定 left/right。
        # 一旦进入 DASH_PRE_SIDE_SHIFT，后续转向/横移/固定前进都不再运行虚线检测器。
        need_dashed_detection = self.state == self.APPROACH_OBSTACLES

        if need_dashed_detection:
            dashed_lines = self.dashed_detector.detect_top_dashed_lines(frame)
            dashed = dashed_lines[0] if dashed_lines else None
        else:
            dashed = None

        chosen_pair = None
        target_candidates_for_vis: List[Detection] = []
        chosen_target_for_vis: Optional[Detection] = None
        final_yellow_line: Optional[Detection] = None
        bar_for_vis: Optional[Detection] = None

        if self.state == self.GLOBAL_LATERAL_SEARCH:
            if self.all_global_tasks_done():
                self.enter_global_final_sequence()
                return

            # 只检测还没有完成的任务类型：
            # - 限高杆已经完成 required_bar_count 次后，后续不再识别/触发限高杆；
            # - 障碍物已经完成 required_obstacle_count 次后，后续不再识别/触发障碍物。
            need_bar = self.completed_bar_count < self.required_bar_count
            need_obstacle = self.completed_obstacle_count < self.required_obstacle_count

            bar = self.bar_detector.detect(
                frame, self.latest_depth) if need_bar else None
            bar_for_vis = bar

            obs_for_choice = obstacle_candidates if need_obstacle else []
            obj_type, det = self.choose_global_object(bar, obs_for_choice)

            if obj_type == 'bar':
                self.current_global_target = det
                self.get_logger().info(
                    f'[GLOBAL_SEARCH] detect BAR {self.completed_bar_count + 1}/{self.required_bar_count}, center={det.center_img}'
                )
                self.enter_state(self.GLOBAL_CENTER_BAR)
                return

            if obj_type == 'obstacle':
                self.current_global_target = det
                self.get_logger().info(
                    f'[GLOBAL_SEARCH] detect OBSTACLE {self.completed_obstacle_count + 1}/{self.required_obstacle_count}, center={det.center_img}'
                )
                self.enter_state(self.GLOBAL_CENTER_OBSTACLE)
                return

            self.send_motion_cmd(0.0, self.global_lateral_search_vy, 0.0)

        elif self.state == self.GLOBAL_CENTER_BAR:
            # ------------------------------------------------------------
            # RGB/限高杆诊断日志：
            # 1) rgb_seq / rgb_age 用来判断 Stage4 是否持续收到新 RGB 帧；
            # 2) frame_sig 是对当前图像做稀疏采样后的简单签名，作为额外兜底；
            # 3) center / bbox 用来判断新帧正常时，限高杆检测结果是否卡死。
            # 这里只增加诊断信息，不改变任何控制行为。
            # ------------------------------------------------------------
            rgb_seq = int(getattr(self, 'latest_rgb_seq',
                                  getattr(self, 'rgb_frame_seq', -1)))

            rgb_age = None
            rgb_age_fn = getattr(self, 'rgb_age_s', None)
            if callable(rgb_age_fn):
                try:
                    rgb_age = rgb_age_fn()
                except Exception:
                    rgb_age = None

            # 稀疏采样，开销很小；移动时图像变化，frame_sig 通常也会变化。
            try:
                sample = frame[::24, ::32]
                frame_sig = int(np.sum(sample.astype(np.uint64)) % 1000000007)
            except Exception:
                frame_sig = -1

            bar = self.bar_detector.detect(frame, self.latest_depth)
            bar_for_vis = bar

            if bar is None:
                self.global_center_stable_count = 0
                self.send_motion_cmd(0.0, self.global_lateral_search_vy, 0.0)

                rgb_age_text = (
                    f'{rgb_age:.3f}s' if rgb_age is not None else 'N/A'
                )
                self.get_logger().info(
                    f'[GLOBAL_CENTER_BAR][RGB_DIAG] bar=None, '
                    f'rgb_seq={rgb_seq}, rgb_age={rgb_age_text}, '
                    f'frame_sig={frame_sig}, continue lateral search motion',
                    throttle_duration_sec=0.5
                )

            else:
                vy = self.compute_global_bar_center_fixed_vy(bar)
                wz = self.compute_bar_depth_yaw_align_wz(bar)
                self.send_motion_cmd(0.0, vy, wz)

                centered = self.is_bar_centered(
                    bar,
                    deadband_px=self.global_bar_center_px_deadband
                )

                if centered:
                    self.global_center_stable_count += 1
                else:
                    self.global_center_stable_count = 0

                img_center_x = self.latest_bgr.shape[1] / 2.0
                err_px = bar.center_img[0] - img_center_x
                rgb_age_text = (
                    f'{rgb_age:.3f}s' if rgb_age is not None else 'N/A'
                )

                self.get_logger().info(
                    f'[GLOBAL_CENTER_BAR][RGB_DIAG] '
                    f'rgb_seq={rgb_seq}, rgb_age={rgb_age_text}, '
                    f'frame_sig={frame_sig}, '
                    f'center={bar.center_img}, bbox={bar.bbox_img}, '
                    f'err_px={err_px:.1f}, '
                    f'deadband={self.global_bar_center_px_deadband}, '
                    f'vy={vy:.3f}, wz={wz:.3f}, '
                    f'depth_yaw={self.latest_bar_depth_yaw_info}, '
                    f'stable={self.global_center_stable_count}/{self.global_center_stable_frames}',
                    throttle_duration_sec=0.2
                )

                # 这里保留诊断数据；真正的 stale 停车由
                # _p4_accept_fresh_rgb_for_visual_state() 统一处理。

                if self.global_center_stable_count >= self.global_center_stable_frames:
                    self.current_bar_det = bar

                    self.get_logger().info(
                        f'[GLOBAL_CENTER_BAR] centered by fixed-vy, '
                        f'enter BAR_FORWARD_UNDER'
                    )

                    self.enter_state(self.BAR_FORWARD_UNDER)
                    return

        elif self.state == self.GLOBAL_CENTER_OBSTACLE:
            # ============================================================
            # 全局障碍物居中：
            # 不再对齐“离画面中心最近的单个障碍物”，而是先从当前
            # 有效候选中选出一对左右障碍物，再将两者中心的中点
            # 对齐到图像中心。该阶段只做横移，不前进。
            # ============================================================
            pair = self.choose_obstacle_pair(obstacle_candidates)
            chosen_pair = pair

            # 必须同时看到两个有效障碍物才能定义通道中点。
            if pair is None:
                self.global_center_stable_count = 0
                self.send_motion_cmd(0.0, self.global_lateral_search_vy, 0.0)
                self.get_logger().info(
                    f'[GLOBAL_CENTER_OBS] need 2 obstacles, '
                    f'current_candidates={len(obstacle_candidates)}, '
                    f'continue lateral search, vy={self.global_lateral_search_vy:.3f}',
                    throttle_duration_sec=0.5
                )
                return

            left, right = pair
            img_center_x = frame.shape[1] / 2.0
            pair_center_x = (left.center_img[0] + right.center_img[0]) / 2.0
            err_px = pair_center_x - img_center_x

            # 复用后续 APPROACH_OBSTACLES 已经使用的两障碍物中点对齐函数。
            vy = self.compute_obstacle_mid_align_vy(left, right)

            if abs(err_px) <= self.obstacle_center_px_deadband:
                self.global_center_stable_count += 1
            else:
                self.global_center_stable_count = 0

            # 全局居中阶段严格只横移，不允许向前。
            self.send_motion_cmd(0.0, vy, 0.0)

            self.get_logger().info(
                f'[GLOBAL_CENTER_OBS] '
                f'left_x={left.center_img[0]}, right_x={right.center_img[0]}, '
                f'pair_center_x={pair_center_x:.1f}, image_center_x={img_center_x:.1f}, '
                f'err_px={err_px:.1f}, deadband={self.obstacle_center_px_deadband}, '
                f'vy={vy:.3f}, '
                f'stable={self.global_center_stable_count}/{self.global_center_stable_frames}',
                throttle_duration_sec=0.2
            )

            if self.global_center_stable_count >= self.global_center_stable_frames:
                # 记录当前障碍物流程是不是全局第 3 个子流程。
                # 例如 required_bar_count=2、required_obstacle_count=1 时：
                # 两个限高杆都完成后才处理障碍物，则这个障碍物就是第 3 个物体。
                flow_index = self.completed_bar_count + self.completed_obstacle_count + 1
                total_required_flows = self.required_bar_count + self.required_obstacle_count
                self.obstacle_flow_is_third_object = (flow_index >= total_required_flows)

                self.get_logger().info(
                    f'[GLOBAL_CENTER_OBS] pair centered, enter obstacle flow, '
                    f'left_x={left.center_img[0]}, right_x={right.center_img[0]}, '
                    f'pair_center_x={pair_center_x:.1f}, '
                    f'flow_index={flow_index}/{total_required_flows}, '
                    f'obstacle_flow_is_third_object={self.obstacle_flow_is_third_object}'
                )

                # 障碍物流程参数是按 low 姿态调好的：
                # 先完成两个障碍物通道中点的全局横向居中，再进入 low，
                # 随后进入正式障碍物流程。
                self.set_body_low_for_obstacle_flow()
                self.enter_state(self.APPROACH_OBSTACLES)
                return


        elif self.state == self.GLOBAL_SHIFT_AFTER_SUBTASK:
            if self.all_global_tasks_done():
                self.enter_global_final_sequence()
                return

            elapsed = self.state_elapsed_s()
            target_duration = self.current_after_task_shift_duration_s if self.current_after_task_shift_duration_s > 0.0 else self.global_after_task_shift_duration_s

            if elapsed >= target_duration:
                self.get_logger().info(
                    f'[GLOBAL_SHIFT] finished: elapsed={elapsed:.3f}/{target_duration:.3f}s, '
                    f'reason={self.global_after_task_shift_reason}'
                )
                self.enter_state(self.GLOBAL_LATERAL_SEARCH)
                return

            self.send_motion_cmd(0.0, self.global_after_task_shift_vy, 0.0)
            self.get_logger().info(
                f'[GLOBAL_SHIFT] shifting left by sim time: elapsed={elapsed:.3f}/{target_duration:.3f}s, '
                f'reason={self.global_after_task_shift_reason}',
                throttle_duration_sec=0.3
            )


        elif self.state == self.BAR_FORWARD_UNDER:
            bar = self.bar_detector.detect(frame, self.latest_depth)
            bar_for_vis = bar
            vy = self.compute_bar_align_vy(bar) if bar is not None else 0.0
            wz = self.compute_bar_depth_yaw_align_wz(bar) if bar is not None else 0.0
            top_y_ratio = self.bbox_top_y_ratio(bar)
            d = (self.estimate_bar_depth(bar)
                 if bar is not None and not self.use_rgb_distance_triggers
                 else None)

            if self.use_rgb_distance_triggers:
                trigger_reached = (
                    top_y_ratio is not None
                    and top_y_ratio <= self.bar_trigger_top_y_ratio)
            else:
                trigger_reached = d is not None and d <= self.bar_trigger_distance_m

            if trigger_reached:
                self.bar_trigger_confirm_count += 1
            else:
                self.bar_trigger_confirm_count = 0

            if self.bar_trigger_confirm_count >= self.bar_trigger_confirm_frames:
                # 本帧已经满足停车条件，先发零速度再切姿态，不能再多发一帧前进命令。
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().warn(
                    f'[BAR_FORWARD] stop before lowering: depth={d}, '
                    f'top_y_ratio={top_y_ratio}, '
                    f'top_trigger={self.bar_trigger_top_y_ratio:.3f}, '
                    f'confirm={self.bar_trigger_confirm_count}/'
                    f'{self.bar_trigger_confirm_frames}'
                )
                self.speak_bar_at_trigger()
                self.set_body_low_for_bar_trigger()
                self.enter_state(self.BAR_BODY_LOWER_WAIT)
                return

            if self.use_rgb_distance_triggers:
                slow_reached = (
                    top_y_ratio is not None
                    and top_y_ratio <= self.bar_search_slow_top_y_ratio)
            else:
                slow_reached = (
                    d is not None and d <= self.bar_search_slow_distance_m)
            if slow_reached:
                forward_speed = self.bar_search_near_forward_speed
            else:
                forward_speed = self.bar_search_forward_speed
            self.send_motion_cmd(forward_speed, vy, wz)

            if bar is not None:
                self.get_logger().info(
                    f'[BAR_FORWARD] depth={d}, top_y_ratio={top_y_ratio}, '
                    f'slow_top={self.bar_search_slow_top_y_ratio:.3f}, '
                    f'trigger_top={self.bar_trigger_top_y_ratio:.3f}, '
                    f'vx={forward_speed:.3f}, vy={vy:.3f}, wz={wz:.3f}, '
                    f'trigger_confirm={self.bar_trigger_confirm_count}/'
                    f'{self.bar_trigger_confirm_frames}, '
                    f'depth_yaw={self.latest_bar_depth_yaw_info}',
                    throttle_duration_sec=0.2
                )

        elif self.state == self.BAR_BODY_LOWER_WAIT:
            # 零速度持续携带机身高度和前倾角度，让两者同时渐变到位。
            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            elapsed = self.state_elapsed_s()

            self.get_logger().info(
                f'[BAR_BODY_LOWER_WAIT] in-place gait: '
                f'elapsed={elapsed:.3f}/{self.bar_body_lower_wait_s:.3f}s, '
                f'height_cmd={self.body_height_cmd:.3f}',
                throttle_duration_sec=0.25,
            )

            if elapsed >= self.bar_body_lower_wait_s:
                self.get_logger().warn(
                    '[BAR_BODY_LOWER_WAIT] combined height/pitch wait finished, '
                    'start clearing the bar with the same pose'
                )
                self.enter_state(self.BAR_CLEAR_AFTER_UNDER)
                return

        elif self.state == self.BAR_CLEAR_AFTER_UNDER:
            # 保持已经同时设置好的高度和前倾姿态向前，不使用 TF 位移。
            self.send_motion_cmd(
                self.bar_clear_forward_speed,
                0.0,
                0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )

            elapsed = self.state_elapsed_s()

            self.get_logger().info(
                f'[BAR_CLEAR] timed forward: '
                f'elapsed={elapsed:.3f}/{self.bar_clear_forward_time_s:.3f}s, '
                f'speed={self.bar_clear_forward_speed:.3f}',
                throttle_duration_sec=0.25,
            )

            if elapsed >= self.bar_clear_forward_time_s:
                self.get_logger().warn(
                    '[BAR_CLEAR] timed forward finished; start target search directly'
                )
                self.enter_state(self.BAR_SEARCH_TARGET)
                return

        elif self.state == self.BAR_FORWARD_PITCH_WAIT:
            # 零速度持续携带 low + forward pitch，等待姿态渐变完成。
            self.send_motion_cmd(
                0.0,
                0.0,
                0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            elapsed = self.state_elapsed_s()

            self.get_logger().info(
                f'[BAR_PITCH] elapsed={elapsed:.3f}/{self.bar_target_pitch_wait_s:.3f}s, '
                f'pitch_cmd={self.body_pitch_cmd:.3f}, '
                f'height_cmd={self.body_height_cmd:.3f}',
                throttle_duration_sec=0.25,
            )

            if elapsed >= self.bar_target_pitch_wait_s:
                self.get_logger().warn(
                    '[BAR_PITCH] forward pitch ready; start target search'
                )
                self.enter_state(self.BAR_SEARCH_TARGET)
                return

        elif self.state == self.BAR_SEARCH_TARGET:
            self.send_motion_cmd(self.target_search_forward_speed, 0.0, 0.0)
            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target
            if target is None:
                self.stable_target_type = None
                self.target_stable_count = 0
            else:
                if self.stable_target_type == target.det_type:
                    self.target_stable_count += 1
                else:
                    self.stable_target_type = target.det_type
                    self.target_stable_count = 1
                self.latest_target = target
                if self.target_stable_count >= self.target_stable_frames:
                    self.locked_target = target
                    self.enter_state(self.BAR_APPROACH_TARGET)
                    return

        elif self.state == self.BAR_APPROACH_TARGET:
            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target
            if target is None:
                self.locked_target = None
                self.enter_state(self.BAR_SEARCH_TARGET)
                return
            self.locked_target = target
            vx, vy = self.compute_target_align_cmd(target)
            self.send_motion_cmd(vx, vy, 0.0)
            metrics = self.target_visual_metrics(target)
            if self.use_rgb_distance_triggers:
                d = None
                hit_reached = self.target_visual_threshold_reached(
                    target, 'hit')
            else:
                d = self.estimate_depth_at_center(target.center_img)
                hit_reached = d is not None and d < self.hit_trigger_distance_m
            self.get_logger().info(
                f'[BAR_TARGET_ALIGN] target={target.det_type}, depth={d}, '
                f'area_ratio={metrics["area_ratio"]}, radius={metrics["radius_px"]}, '
                f'hit_reached={hit_reached}, cmd=({vx:.3f},{vy:.3f},0)',
                throttle_duration_sec=0.2
            )
            if hit_reached:
                # 目标物体：靠近到撞击距离、刚进入撞击状态前播报
                self.speak_target_at_hit_trigger(target.det_type)
                self.enter_state(self.BAR_HIT_TARGET)
                return

        elif self.state == self.BAR_HIT_TARGET:
            if self.locked_target is None:
                self.enter_state(self.BAR_SEARCH_TARGET)
                return

            chosen_target_for_vis = self.locked_target
            target_candidates_for_vis = [self.locked_target]
            params = self.hit_params.get(self.locked_target.det_type, {'speed': 0.20, 'duration_s': 0.85})
            elapsed = self.state_elapsed_s()
            duration = float(params.get('duration_s', 0.85))

            if elapsed >= duration:
                self.get_logger().info(
                    f'[BAR_HIT] finished by sim time: target={self.locked_target.det_type}, '
                    f'elapsed={elapsed:.3f}/{duration:.3f}s'
                )
                self.enter_state(self.BAR_BACKOFF_TIMED)
                return

            self.send_motion_cmd(params['speed'], 0.0, 0.0)
            self.get_logger().info(
                f'[BAR_HIT] target={self.locked_target.det_type}, elapsed={elapsed:.3f}/{duration:.3f}s',
                throttle_duration_sec=0.2
            )


        elif self.state == self.BAR_BACKOFF_TIMED:
            # 撞击结束后的固定时间盲退阶段：
            # 这里故意不调用 bar_detector.detect()，也不做任何视觉纠偏。
            elapsed = self.state_elapsed_s()

            if elapsed >= self.bar_backoff_duration_s:
                self.get_logger().info(
                    f'[BAR_BACKOFF_TIMED] blind backoff finished: '
                    f'elapsed={elapsed:.3f}/{self.bar_backoff_duration_s:.3f}s; '
                    f'start detecting bar while continuing to back off'
                )
                self.enter_state(self.BAR_BACKOFF_TO_BAR)
                return

            self.send_motion_cmd(
                -self.bar_backoff_speed, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            self.get_logger().info(
                f'[BAR_BACKOFF_TIMED] blind backoff, no bar detection: '
                f'elapsed={elapsed:.3f}/{self.bar_backoff_duration_s:.3f}s, '
                f'vx={-self.bar_backoff_speed:.3f}',
                throttle_duration_sec=0.2,
            )
            return

        elif self.state == self.BAR_BACKOFF_TO_BAR:
            # 固定盲退结束后才开始检测限高杆。
            # 未检测到时继续直线后退；一旦检测到就立即停车播报。
            bar = self.bar_detector.detect(frame, self.latest_depth)
            bar_for_vis = bar

            if bar is None:
                self.send_motion_cmd(
                    -self.bar_backoff_speed, 0.0, 0.0,
                    pitch=self.bar_target_forward_pitch,
                    body_height=self.bar_low_body_height,
                )
                self.get_logger().info(
                    '[BAR_BACKOFF_SEARCH] after blind backoff: bar=None, keep backing until redetected',
                    throttle_duration_sec=0.5,
                )
                return

            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            self.get_logger().info(
                f'[BAR_BACKOFF_SEARCH] bar redetected at center={bar.center_img}; '
                f'stop and announce before 180-degree turn'
            )
            event_id = f'bar_backoff_finished_{self.completed_bar_count + 1}'
            self.begin_voice_wait(
                wait_state=self.BAR_BACKOFF_VOICE_WAIT,
                event_id=event_id,
                voice_key='bar',
                resume_state=self.BAR_TURN_TO_YELLOW,
            )
            return

        elif self.state == self.BAR_TURN_TO_YELLOW:
            self.execute_left_jump_turn(2, self.BAR_YELLOW_FORWARD)
            return

        elif self.state == self.APPROACH_OBSTACLES:
            # 障碍物已经完成全局居中；从进入本状态开始检测虚线方向。
            # 不改变原来的多虚线选择规则：dashed 就是 dashed_lines[0]。
            forced_side = self.get_forced_dashed_side()
            if forced_side is not None:
                self.dashed_side = forced_side
            elif self.dashed_side is None:
                if dashed is None:
                    self.dashed_side_candidate = None
                    self.dashed_side_candidate_count = 0
                else:
                    candidate_side = self.get_dashed_side(dashed)

                    if candidate_side == self.dashed_side_candidate:
                        self.dashed_side_candidate_count += 1
                    else:
                        self.dashed_side_candidate = candidate_side
                        self.dashed_side_candidate_count = 1

                    self.get_logger().info(
                        f'[OBS_DASH_SIDE] candidate={candidate_side}, '
                        f'confirm={self.dashed_side_candidate_count}/'
                        f'{self.dashed_side_confirm_frames}',
                        throttle_duration_sec=0.2
                    )

                    if self.dashed_side_candidate_count >= self.dashed_side_confirm_frames:
                        self.dashed_side = candidate_side
                        self.get_logger().info(
                            f'[OBS_DASH_SIDE] locked side={self.dashed_side}, '
                            f'center={dashed.center_img}'
                        )

            pair = self.choose_obstacle_pair(obstacle_candidates)
            chosen_pair = pair

            if pair is None:
                if self.obstacle_pair_seen_in_approach:
                    forced_side = self.get_forced_dashed_side()
                    if forced_side is not None:
                        self.dashed_side = forced_side
                    if self.dashed_side not in ('left', 'right'):
                        self.send_motion_cmd(
                            0.0, 0.0, 0.0,
                            pitch=(self.obstacle_approach_forward_pitch
                                   if self.obstacle_approach_pitch_latched else 0.0),
                            body_height=self.obstacle_low_body_height,
                        )
                        self.get_logger().warn(
                            '[OBS_ROUTE] obstacle pair disappeared but dashed_side is unknown; '
                            'stop and keep waiting for dashed direction'
                        )
                        return
                    self.speak_obstacle_at_trigger()
                    self.get_logger().warn(
                        f'[OBS_ROUTE] obstacle pair disappeared after being seen; '
                        f'treat as close enough, side={self.dashed_side}, go pre-shift'
                    )
                    self.dashed_pre_shift_dir_sign = 0.0
                    self.enter_state(self.DASH_PRE_SIDE_SHIFT)
                    return

                # 从未看到过有效障碍物对时，继续慢速前进搜索，不能用丢失触发。
                self.send_motion_cmd(self.obstacle_search_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    '[OBS_ALIGN] no valid obstacle pair, searching forward',
                    throttle_duration_sec=0.5
                )
                return

            self.obstacle_pair_seen_in_approach = True
            left, right = pair

            vy = self.compute_obstacle_mid_align_vy(left, right)

            d_left = left.extra.get('median_depth')
            d_right = right.extra.get('median_depth')

            depths = [d for d in [d_left, d_right] if d is not None]
            obstacle_dist = min(depths) if depths else None
            obstacle_bottom_y_ratio = max(
                self.bbox_bottom_y_ratio(left),
                self.bbox_bottom_y_ratio(right),
            )

            if self.use_rgb_distance_triggers:
                pitch_reached = (
                    obstacle_bottom_y_ratio >=
                    self.obstacle_approach_pitch_bottom_y_ratio)
                trigger_reached = (
                    obstacle_bottom_y_ratio >=
                    self.obstacle_trigger_bottom_y_ratio)
            else:
                pitch_reached = (
                    obstacle_dist is not None
                    and obstacle_dist <= self.obstacle_approach_pitch_distance_m)
                trigger_reached = (
                    obstacle_dist is not None
                    and obstacle_dist <= self.obstacle_trigger_distance_m)

            # 一旦靠近到前倾阈值，就锁住前倾直到离开 APPROACH_OBSTACLES。
            # 这样即使后续检测框 bottom_y_ratio 在阈值附近抖动，也不会再恢复 pitch=0。
            if pitch_reached and not self.obstacle_approach_pitch_latched:
                self.obstacle_approach_pitch_latched = True
                self.get_logger().info(
                    f'[OBS_PITCH] latch forward pitch={self.obstacle_approach_forward_pitch:.3f}; '
                    f'bottom_y_ratio={obstacle_bottom_y_ratio:.3f}, '
                    f'trigger={self.obstacle_approach_pitch_bottom_y_ratio:.3f}'
                )

            approach_pitch = (
                self.obstacle_approach_forward_pitch
                if self.obstacle_approach_pitch_latched else 0.0
            )
            self.send_motion_cmd(
                self.obstacle_forward_speed,
                vy,
                0.0,
                pitch=approach_pitch,
                body_height=self.obstacle_low_body_height,
            )

            if not self.use_rgb_distance_triggers and obstacle_dist is None:
                self.get_logger().info(
                    f'[OBS_ALIGN] pair_center=({left.center_img[0]},{right.center_img[0]}), '
                    f'dist=None, trigger={self.obstacle_trigger_distance_m:.3f}, vy={vy:.3f}, keep approaching',
                    throttle_duration_sec=0.2
                )
                return

            self.get_logger().info(
                f'[OBS_ALIGN] pair_center=({left.center_img[0]},{right.center_img[0]}), '
                f'dist={obstacle_dist}, bottom_y_ratio={obstacle_bottom_y_ratio:.3f}, '
                f'bottom_trigger={self.obstacle_trigger_bottom_y_ratio:.3f}, '
                f'pitch_bottom_trigger={self.obstacle_approach_pitch_bottom_y_ratio:.3f}, '
                f'pitch={approach_pitch:.3f}, pitch_latched={self.obstacle_approach_pitch_latched}, '
                f'vy={vy:.3f}, keep approaching',
                throttle_duration_sec=0.2
            )

            if not trigger_reached:
                # 还没靠近到触发距离，继续靠近障碍物
                return

            # 靠近阈值已到。新路线必须先锁定 dashed_side，后面不再检测虚线。
            forced_side = self.get_forced_dashed_side()
            if forced_side is not None:
                self.dashed_side = forced_side

            if self.dashed_side not in ('left', 'right'):
                self.send_motion_cmd(
                    0.0, 0.0, 0.0,
                    pitch=approach_pitch,
                    body_height=self.obstacle_low_body_height,
                )
                self.get_logger().warn(
                    f'[OBS_ROUTE] close enough but dashed_side is still unknown; '
                    f'dist={obstacle_dist}, bottom_y_ratio={obstacle_bottom_y_ratio:.3f}; '
                    f'stop here and keep detecting dashed direction'
                )
                return

            self.speak_obstacle_at_trigger()
            self.get_logger().info(
                f'[OBS_ROUTE] approach finished: side={self.dashed_side}, '
                f'dist={obstacle_dist}, bottom_y_ratio={obstacle_bottom_y_ratio:.3f}; '
                f'go pre-shift'
            )
            self.dashed_pre_shift_dir_sign = 0.0
            self.enter_state(self.DASH_PRE_SIDE_SHIFT)
            return

        elif self.state == self.OBSTACLE_ROUTE_PRE_TURN1_STEP:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.obstacle_route_pre_turn_step_duration_s:
                self.get_logger().info(
                    f'[OBS_ROUTE_PRE_TURN1_STEP] finished: elapsed={elapsed:.3f}/'
                    f'{self.obstacle_route_pre_turn_step_duration_s:.3f}s; start first 90deg turn'
                )
                self.enter_state(self.OBSTACLE_ROUTE_TURN_1)
                return

            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=0.0,
                body_height=self.obstacle_low_body_height,
                step_height=self.p4_timed_turn_step_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_PRE_TURN1_STEP] stepping in place: elapsed={elapsed:.3f}/'
                f'{self.obstacle_route_pre_turn_step_duration_s:.3f}s, cmd=(0,0,0), pitch=0.000',
                throttle_duration_sec=0.2
            )
            return

        elif self.state == self.OBSTACLE_ROUTE_TURN_1:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.obstacle_route_turn_duration_s:
                self.get_logger().info(
                    f'[OBS_ROUTE_TURN1] finished: side={self.dashed_side}, '
                    f'elapsed={elapsed:.3f}/{self.obstacle_route_turn_duration_s:.3f}s; '
                    f'go lateral single-obstacle scan'
                )
                self.enter_state(self.OBSTACLE_ROUTE_LATERAL_SCAN)
                return

            wz = self.get_obstacle_route_first_turn_dir() * abs(self.obstacle_route_turn_wz)
            self.send_obstacle_route_turn_cmd(wz)
            return

        elif self.state == self.OBSTACLE_ROUTE_LATERAL_SCAN:
            vy = self.get_obstacle_route_lateral_vy()
            vx = -abs(self.obstacle_route_lateral_backward_speed)
            obstacle = self.choose_obstacle_route_single(obstacle_candidates)
            self.obstacle_route_selected_obstacle = obstacle

            if obstacle is None:
                self.obstacle_route_edge_confirm_count = 0
                self.send_motion_cmd(
                    vx, vy, 0.0,
                    pitch=self.dash_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                )
                self.get_logger().info(
                    f'[OBS_ROUTE_LATERAL] side={self.dashed_side}, obstacle=None, '
                    f'continue lateral+backward vx={vx:.3f}, vy={vy:.3f}; dashed detector OFF',
                    throttle_duration_sec=0.5
                )
                return

            edge_x, threshold_x, reached, edge_name = self.obstacle_route_edge_metrics(obstacle)
            if reached:
                self.obstacle_route_edge_confirm_count += 1
            else:
                self.obstacle_route_edge_confirm_count = 0

            if self.obstacle_route_edge_confirm_count >= self.obstacle_route_edge_confirm_frames:
                self.send_motion_cmd(
                    0.0, 0.0, 0.0,
                    pitch=self.dash_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                )
                self.get_logger().info(
                    f'[OBS_ROUTE_LATERAL] edge reached: side={self.dashed_side}, '
                    f'{edge_name}={edge_x:.1f}px threshold={threshold_x:.1f}px, '
                    f'confirm={self.obstacle_route_edge_confirm_count}/'
                    f'{self.obstacle_route_edge_confirm_frames}; go fixed forward'
                )
                self.enter_state(self.OBSTACLE_ROUTE_FORWARD)
                return

            self.send_motion_cmd(
                vx, vy, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_LATERAL] side={self.dashed_side}, '
                f'vx={vx:.3f}, vy={vy:.3f}, '
                f'candidates={len(obstacle_candidates)}, selected_bbox={obstacle.bbox_img}, '
                f'{edge_name}={edge_x:.1f}px threshold={threshold_x:.1f}px, '
                f'reached={reached}, confirm={self.obstacle_route_edge_confirm_count}/'
                f'{self.obstacle_route_edge_confirm_frames}, vy={vy:.3f}; dashed detector OFF',
                throttle_duration_sec=0.2
            )
            return

        elif self.state == self.OBSTACLE_ROUTE_FORWARD:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.obstacle_route_forward_duration_s:
                self.get_logger().info(
                    f'[OBS_ROUTE_FORWARD] finished: elapsed={elapsed:.3f}/'
                    f'{self.obstacle_route_forward_duration_s:.3f}s; '
                    f'go pre-turn2 lateral shift'
                )
                self.enter_state(self.OBSTACLE_ROUTE_PRE_TURN2_LATERAL)
                return

            self.send_motion_cmd(
                self.obstacle_route_forward_speed, 0.0, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_FORWARD] vx={self.obstacle_route_forward_speed:.3f}, '
                f'elapsed={elapsed:.3f}/{self.obstacle_route_forward_duration_s:.3f}s',
                throttle_duration_sec=0.2
            )
            return

        elif self.state == self.OBSTACLE_ROUTE_PRE_TURN2_LATERAL:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.obstacle_route_pre_turn2_lateral_duration_s:
                self.get_logger().info(
                    f'[OBS_ROUTE_PRE_TURN2_LATERAL] finished: side={self.dashed_side}, '
                    f'elapsed={elapsed:.3f}/'
                    f'{self.obstacle_route_pre_turn2_lateral_duration_s:.3f}s; '
                    f'go pre-turn2 in-place step'
                )
                self.enter_state(self.OBSTACLE_ROUTE_PRE_TURN2_STEP)
                return

            vy = self.get_pre_shift_dir_sign() * abs(
                self.obstacle_route_pre_turn2_lateral_speed)
            self.send_motion_cmd(
                0.0, vy, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_PRE_TURN2_LATERAL] side={self.dashed_side}, '
                f'vy={vy:.3f}, elapsed={elapsed:.3f}/'
                f'{self.obstacle_route_pre_turn2_lateral_duration_s:.3f}s; vision OFF',
                throttle_duration_sec=0.2
            )
            return

        elif self.state == self.OBSTACLE_ROUTE_PRE_TURN2_STEP:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.obstacle_route_pre_turn_step_duration_s:
                self.get_logger().info(
                    f'[OBS_ROUTE_PRE_TURN2_STEP] finished: elapsed={elapsed:.3f}/'
                    f'{self.obstacle_route_pre_turn_step_duration_s:.3f}s; start second 90deg turn'
                )
                self.enter_state(self.OBSTACLE_ROUTE_TURN_2)
                return

            self.send_motion_cmd(
                0.0, 0.0, 0.0,
                pitch=0.0,
                body_height=self.obstacle_low_body_height,
                step_height=self.p4_timed_turn_step_height,
            )
            self.get_logger().info(
                f'[OBS_ROUTE_PRE_TURN2_STEP] stepping in place: elapsed={elapsed:.3f}/'
                f'{self.obstacle_route_pre_turn_step_duration_s:.3f}s, cmd=(0,0,0), pitch=0.000',
                throttle_duration_sec=0.2
            )
            return

        elif self.state == self.OBSTACLE_ROUTE_TURN_2:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.obstacle_route_turn_duration_s:
                self.get_logger().info(
                    f'[OBS_ROUTE_TURN2] finished: side={self.dashed_side}, '
                    f'elapsed={elapsed:.3f}/{self.obstacle_route_turn_duration_s:.3f}s; '
                    f'enter target search'
                )
                self.target_seen_after_turn = False
                self.target_seen_confirm_count = 0
                self.target_seen_confirm_type = None
                self.enter_state(self.SEARCH_TARGET_AFTER_TURNS)
                return

            wz = -self.get_obstacle_route_first_turn_dir() * abs(self.obstacle_route_turn_wz)
            self.send_obstacle_route_turn_cmd(wz)
            return

        elif self.state == self.ALIGN_DASHED_LINE:
            if dashed is None:
                self.dashed_center_count = 0

                # 已经通过视觉确定过虚线方向时，
                # 暂时丢失虚线仍按照预偏移方向继续横移。
                if self.dashed_side in ('left', 'right'):
                    # 按用户要求：虚线暂时丢失时仍继续原方向横移。
                    # 但使用独立的小速度，不再复用 DASH_PRE_SIDE_SHIFT 的较大速度，
                    # 防止视觉漏检几秒时直接横移越过 target_x。
                    vy = (
                        self.get_pre_shift_dir_sign()
                        * abs(self.dashed_align_lost_vy_speed)
                    )

                    self.send_motion_cmd(
                        0.0,
                        vy,
                        0.0,
                        pitch=self.dash_forward_pitch,
                        body_height=self.obstacle_low_body_height,
                    )

                    self.get_logger().info(
                        f'[DASH_ALIGN] dashed=None, continue slowly: '
                        f'side={self.dashed_side}, vy={vy:.3f}, '
                        f'lost_vy_speed={self.dashed_align_lost_vy_speed:.3f}',
                        throttle_duration_sec=0.5
                    )
                else:
                    # 尚未确定虚线在左边还是右边，无法安全决定横移方向。
                    self.send_motion_cmd(0.0, 0.0, 0.0)

                    self.get_logger().warn(
                        '[DASH_ALIGN] dashed=None and dashed_side=None, waiting',
                        throttle_duration_sec=0.5
                    )
            else:
                # 第一次看到虚线时，先记录它在左边还是右边，随后进入预横移状态
                if self.dashed_side is None:
                    forced_side = self.get_forced_dashed_side()

                    if forced_side is not None:
                        self.dashed_side = forced_side
                        side_source = 'debug'
                    else:
                        self.dashed_side = self.get_dashed_side(dashed)
                        side_source = 'vision'

                    self.get_logger().info(
                        f'[DASH_ALIGN] first dashed side={self.dashed_side}, source={side_source}, '
                        f'center={dashed.center_img}, target_x={self.get_dashed_target_x():.1f}, '
                        f'enter pre side shift'
                    )

                    self.enter_state(self.DASH_PRE_SIDE_SHIFT)
                    return

                vy = self.compute_dashed_align_vy(dashed)
                self.send_motion_cmd(
                    0.0, vy, 0.0,
                    pitch=self.dash_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                )

                if self.is_dashed_centered(dashed):
                    self.dashed_center_count += 1
                else:
                    self.dashed_center_count = 0

                self.get_logger().info(
                    f'[DASH_ALIGN] side={self.dashed_side}, center={dashed.center_img}, '
                    f'target_x={self.get_dashed_target_x():.1f}, '
                    f'vy={vy:.3f}, stable={self.dashed_center_count}/{self.dashed_center_stable_frames}',
                    throttle_duration_sec=0.2
                )

                if self.dashed_center_count >= self.dashed_center_stable_frames:
                    self.enter_state(self.FOLLOW_DASHED_UNTIL_LOST)

        elif self.state == self.DASH_PRE_SIDE_SHIFT:
            if self.dashed_side not in ('left', 'right'):
                self.get_logger().warn('[DASH_PRE_SHIFT] dashed_side is None, return to APPROACH_OBSTACLES')
                self.enter_state(self.APPROACH_OBSTACLES)
                return

            if self.dashed_pre_shift_dir_sign == 0.0:
                self.dashed_pre_shift_dir_sign = self.get_pre_shift_dir_sign()

            elapsed = self.state_elapsed_s()
            if elapsed >= self.dashed_pre_shift_duration_s:
                self.get_logger().info(
                    f'[DASH_PRE_SHIFT] done by sim time: side={self.dashed_side}, '
                    f'elapsed={elapsed:.3f}/{self.dashed_pre_shift_duration_s:.3f}s, '
                    f'go pre-turn1 in-place step'
                )
                self.enter_state(self.OBSTACLE_ROUTE_PRE_TURN1_STEP)
                return

            vy = self.get_pre_shift_vy()
            vx = -self.dashed_pre_shift_backward_speed
            self.send_motion_cmd(
                vx, vy, 0.0,
                pitch=self.dash_forward_pitch,
                body_height=self.obstacle_low_body_height,
            )
            self.get_logger().info(
                f'[DASH_PRE_SHIFT] moving by sim time: side={self.dashed_side}, '
                f'elapsed={elapsed:.3f}/{self.dashed_pre_shift_duration_s:.3f}s, '
                f'vx={vx:.3f}, vy={vy:.3f}',
                throttle_duration_sec=0.2
            )


        elif self.state == self.FOLLOW_DASHED_UNTIL_LOST:
            # 新逻辑：
            # 对齐虚线之后，沿虚线向前走时，不是所有检测到的虚线都算有效。
            # 只有虚线中心 x 落在 get_dashed_target_x() 附近 follow_dashed_valid_x_range_px 范围内，
            # 才认为当前虚线仍然存在。
            # 如果检测到的虚线偏得太远，也按“虚线消失”处理。
            dashed_valid = self.is_dashed_valid_for_follow(dashed)

            if not dashed_valid:
                self.dashed_lost_count += 1

                if dashed is None:
                    self.get_logger().info(
                        f'[FOLLOW_DASH] dashed=None, '
                        f'lost_count={self.dashed_lost_count}/{self.dashed_lost_stop_frames}',
                        throttle_duration_sec=0.2
                    )
                else:
                    target_x = self.get_dashed_target_x()
                    cx = float(dashed.center_img[0])
                    err_px = cx - target_x

                    self.get_logger().info(
                        f'[FOLLOW_DASH] dashed detected but outside valid range, treat as lost: '
                        f'center_x={cx:.1f}, target_x={target_x:.1f}, '
                        f'err={err_px:.1f}px, valid_range=±{self.follow_dashed_valid_x_range_px}px, '
                        f'lost_count={self.dashed_lost_count}/{self.dashed_lost_stop_frames}',
                        throttle_duration_sec=0.2
                    )

                if self.dashed_lost_count >= self.dashed_lost_stop_frames:
                    self.get_logger().info(
                        f'[FOLLOW_DASH] dashed lost/out-of-range '
                        f'{self.dashed_lost_count} frames, go post dash forward'
                    )
                    self.enter_state(self.POST_DASH_FORWARD)
                else:
                    # 防止单帧漏检或单帧跳变，短暂继续向前。
                    # 注意这里不给 vy 修正，避免被错误虚线带偏。
                    self.send_motion_cmd(
                        self.follow_forward_speed, 0.0, 0.0,
                        pitch=self.dash_forward_pitch,
                        body_height=self.obstacle_low_body_height,
                    )

            else:
                self.dashed_lost_count = 0

                vy = self.compute_dashed_align_vy(
                    dashed,
                    k=self.follow_align_vy_k,
                    vy_max=self.follow_align_vy_max,
                    vy_min=self.follow_align_vy_min,
                )

                self.send_motion_cmd(
                    self.follow_forward_speed, vy, 0.0,
                    pitch=self.dash_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                )

                target_x = self.get_dashed_target_x()
                cx = float(dashed.center_img[0])
                err_px = cx - target_x

                self.get_logger().info(
                    f'[FOLLOW_DASH] valid dashed: center={dashed.center_img}, '
                    f'target_x={target_x:.1f}, err={err_px:.1f}px, '
                    f'valid_range=±{self.follow_dashed_valid_x_range_px}px, '
                    f'cmd=({self.follow_forward_speed:.3f},{vy:.3f},0.000), '
                    f'vy_limit=[{self.follow_align_vy_min:.3f},{self.follow_align_vy_max:.3f}]',
                    throttle_duration_sec=0.2
                )

        elif self.state == self.POST_DASH_FORWARD:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_dash_forward_duration_s:
                self.get_logger().info(
                    f'[POST_DASH_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_dash_forward_duration_s:.3f}s, go first turn'
                )
                self.enter_state(self.POST_DASH_TURN_1)
            else:
                self.send_motion_cmd(self.post_dash_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_DASH_FORWARD] elapsed={elapsed:.3f}/{self.post_dash_forward_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_DASH_TURN_1:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_DASH_TURN_1] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, go forward'
                )
                self.enter_state(self.POST_TURN_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_fixed_turn_cmd(wz)


        elif self.state == self.POST_TURN_FORWARD:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_turn_forward_duration_s:
                self.get_logger().info(
                    f'[POST_TURN_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_turn_forward_duration_s:.3f}s, go second turn'
                )
                self.enter_state(self.POST_DASH_TURN_2)
            else:
                if elapsed < self.post_turn_forward_fast_duration_s:
                    vx = self.post_turn_forward_fast_speed
                    phase = 'FAST'
                else:
                    vx = self.post_turn_forward_slow_speed
                    phase = 'SLOW'

                self.send_motion_cmd(vx, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_TURN_FORWARD] phase={phase}, vx={vx:.3f}, '
                    f'elapsed={elapsed:.3f}/{self.post_turn_forward_duration_s:.3f}s, '
                    f'fast_until={self.post_turn_forward_fast_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_DASH_TURN_2:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_DASH_TURN_2] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, start target search'
                )
                self.target_seen_after_turn = False
                self.target_seen_confirm_count = 0
                self.target_seen_confirm_type = None
                self.enter_state(self.SEARCH_TARGET_AFTER_TURNS)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_fixed_turn_cmd(wz)


        elif self.state == self.SEARCH_TARGET_AFTER_TURNS:
            # 先做目标检测，再根据“是否曾经看到过目标”决定搜索方向。
            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target

            if target is None:
                self.stable_target_type = None
                self.target_stable_count = 0

                # 在“长时间稳定识别”尚未确认前，任何一次漏检都会打断连续计数。
                # 这样远处偶尔闪现一两帧不会立刻把搜索模式切成后退。
                if not self.target_seen_after_turn:
                    self.target_seen_confirm_count = 0
                    self.target_seen_confirm_type = None

                if self.target_seen_after_turn:
                    search_vx = -abs(self.target_search_backward_speed)
                    search_mode = 'backward_reacquire'
                else:
                    search_vx = abs(self.target_search_forward_speed)
                    search_mode = 'forward_first_search'

                self.send_motion_cmd(
                    search_vx, 0.0, 0.0,
                    pitch=self.obstacle_target_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                )
                self.get_logger().info(
                    f'[TARGET_SEARCH] target=None, mode={search_mode}, '
                    f'seen_before={self.target_seen_after_turn}, '
                    f'seen_confirm={self.target_seen_confirm_count}/'
                    f'{self.target_seen_confirm_frames}, vx={search_vx:.3f}',
                    throttle_duration_sec=0.5
                )
            else:
                # “进入对齐”与“允许丢失后后退”使用两套确认门槛。
                # 这里累计同一种目标连续出现的帧数；达到较长确认门槛后才真正 seen_before=True。
                if not self.target_seen_after_turn:
                    if self.target_seen_confirm_type == target.det_type:
                        self.target_seen_confirm_count += 1
                    else:
                        self.target_seen_confirm_type = target.det_type
                        self.target_seen_confirm_count = 1

                    if self.target_seen_confirm_count >= self.target_seen_confirm_frames:
                        self.target_seen_after_turn = True
                        self.get_logger().info(
                            f'[TARGET_REACQUIRE_ARMED] target={target.det_type}, '
                            f'continuous={self.target_seen_confirm_count}/'
                            f'{self.target_seen_confirm_frames}; '
                            f'future loss will use backward reacquire'
                        )

                # 当前帧看到了目标，仍按原逻辑小步向前搜索/确认稳定帧。
                self.send_motion_cmd(
                    self.target_search_forward_speed, 0.0, 0.0,
                    pitch=self.obstacle_target_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                )

                if self.stable_target_type == target.det_type:
                    self.target_stable_count += 1
                else:
                    self.stable_target_type = target.det_type
                    self.target_stable_count = 1

                self.latest_target = target

                self.get_logger().info(
                    f'[TARGET_SEARCH] target={target.det_type}, center={target.center_img}, '
                    f'stable={self.target_stable_count}/{self.target_stable_frames}, '
                    f'seen_confirm={self.target_seen_confirm_count}/'
                    f'{self.target_seen_confirm_frames}, '
                    f'seen_before={self.target_seen_after_turn}, '
                    f'vx={self.target_search_forward_speed:.3f}',
                    throttle_duration_sec=0.2
                )

                if self.target_stable_count >= self.target_stable_frames:
                    self.locked_target = target
                    self.enter_state(self.APPROACH_AND_ALIGN_TARGET)

        elif self.state == self.APPROACH_AND_ALIGN_TARGET:
            target_candidates_for_vis = self.detect_all_targets(frame)
            target = self.choose_best_target(target_candidates_for_vis)
            chosen_target_for_vis = target

            if target is None:
                self.locked_target = None
                # 如果还没有达到长时间稳定确认，这次丢失只算一次普通漏检：
                # 清空长确认计数，退回 SEARCH 后仍继续向前找。
                # 只有已经 armed 后，退回 SEARCH 才会改成后退重找。
                if not self.target_seen_after_turn:
                    self.target_seen_confirm_count = 0
                    self.target_seen_confirm_type = None
                self.enter_state(self.SEARCH_TARGET_AFTER_TURNS)
            else:
                # SEARCH 可能在第3帧就进入 APPROACH，因此长确认计数需要跨状态继续累计。
                if not self.target_seen_after_turn:
                    if self.target_seen_confirm_type == target.det_type:
                        self.target_seen_confirm_count += 1
                    else:
                        self.target_seen_confirm_type = target.det_type
                        self.target_seen_confirm_count = 1

                    if self.target_seen_confirm_count >= self.target_seen_confirm_frames:
                        self.target_seen_after_turn = True
                        self.get_logger().info(
                            f'[TARGET_REACQUIRE_ARMED] target={target.det_type}, '
                            f'continuous={self.target_seen_confirm_count}/'
                            f'{self.target_seen_confirm_frames}; '
                            f'future loss will use backward reacquire'
                        )

                self.locked_target = target
                vx, vy = self.compute_target_align_cmd(target)
                self.send_motion_cmd(
                    vx, vy, 0.0,
                    pitch=self.obstacle_target_forward_pitch,
                    body_height=self.obstacle_low_body_height,
                )

                metrics = self.target_visual_metrics(target)
                if target.det_type == 'blue_ball' or self.use_rgb_distance_triggers:
                    d = None
                    hit_reached = self.target_visual_threshold_reached(
                        target, 'hit')
                else:
                    d = self.estimate_depth_at_center(target.center_img)
                    hit_reached = (
                        d is not None and d < self.hit_trigger_distance_m)

                top_y_ratio = self.bbox_top_y_ratio(target)
                if target.det_type == 'blue_ball':
                    if hit_reached:
                        self.basketball_top_trigger_count += 1
                    else:
                        self.basketball_top_trigger_count = 0
                    hit_confirmed = (
                        self.basketball_top_trigger_count
                        >= self.basketball_top_trigger_confirm_frames)
                else:
                    hit_confirmed = hit_reached
                self.get_logger().info(
                    f'[TARGET_ALIGN] target={target.det_type}, center={target.center_img}, '
                    f'depth={d}, area_ratio={metrics["area_ratio"]}, '
                    f'radius={metrics["radius_px"]}, top_y_ratio={top_y_ratio}, '
                    f'hit_reached={hit_reached}, hit_confirmed={hit_confirmed}, '
                    f'confirm={self.basketball_top_trigger_count}/'
                    f'{self.basketball_top_trigger_confirm_frames}, '
                    f'cmd=({vx:.3f},{vy:.3f},0.000)',
                    throttle_duration_sec=0.2
                )

                if hit_confirmed:
                    # 目标物体：靠近到撞击距离、刚进入撞击状态前播报
                    self.speak_target_at_hit_trigger(target.det_type)
                    if target.det_type == 'blue_ball' and self.hit_upright_enabled:
                        self.get_logger().warn(
                            '[BASKETBALL_AI] top edge reached trigger; '
                            'skip forward hit and start upright action')
                        self.enter_state(self.HIT_UPRIGHT_PREPARE)
                    else:
                        self.enter_state(self.HIT_TARGET)

        elif self.state == self.HIT_TARGET:
            if self.locked_target is None:
                self.enter_state(self.SEARCH_TARGET_AFTER_TURNS)
            else:
                chosen_target_for_vis = self.locked_target
                target_candidates_for_vis = [self.locked_target]

                params = self.hit_params.get(
                    self.locked_target.det_type,
                    {'speed': 0.20, 'duration_s': 0.85}
                )
                elapsed = self.state_elapsed_s()
                duration = float(params.get('duration_s', 0.85))

                if elapsed >= duration:
                    target_type = self.locked_target.det_type
                    do_upright = (
                        self.hit_upright_enabled
                        and target_type in ('blue_ball', 'orange_ball')
                    )
                    next_state = (
                        self.HIT_UPRIGHT_PREPARE
                        if do_upright
                        else self.HIT_BACKOFF_AFTER_HIT
                    )
                    self.get_logger().info(
                        f'[HIT] finished by sim time: target={target_type}, '
                        f'elapsed={elapsed:.3f}/{duration:.3f}s, '
                        f'next={next_state}'
                    )
                    self.enter_state(next_state)
                    return

                self.send_motion_cmd(params['speed'], 0.0, 0.0)
                self.get_logger().info(
                    f'[HIT] target={self.locked_target.det_type}, elapsed={elapsed:.3f}/{duration:.3f}s',
                    throttle_duration_sec=0.2
                )

                if elapsed >= self.hit_timeout_s:
                    self.get_logger().warn('[HIT] timeout reached, go backoff after hit')
                    self.enter_state(self.HIT_BACKOFF_AFTER_HIT)


        elif self.state == self.HIT_BACKOFF_AFTER_HIT:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.after_hit_backoff_duration_s:
                self.send_motion_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'[AFTER_HIT_BACKOFF] finished by sim time: elapsed={elapsed:.3f}/{self.after_hit_backoff_duration_s:.3f}s, go two left jumps'
                )
                self.enter_state(self.POST_HIT_LEFT_JUMP)
            else:
                self.send_motion_cmd(-self.after_hit_backoff_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[AFTER_HIT_BACKOFF] elapsed={elapsed:.3f}/{self.after_hit_backoff_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_HIT_LEFT_JUMP:
            self.execute_left_jump_turn(
                jump_count=self.after_hit_left_jump_count,
                next_state=self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT
            )
            return

        elif self.state == self.APPROACH_SELECTED_OBSTACLE_AFTER_HIT:
            # 这个状态有三种情况：
            # 1) 0 个障碍物：继续向前搜索。
            # 2) 1 个障碍物：不居中对齐，只检查距离；距离到阈值后进入后续转向。
            # 3) 2 个及以上障碍物：保持原逻辑，根据 dashed_side 选择左/右障碍物并居中对齐。
            obs_count = len(obstacle_candidates)

            if obs_count == 0:
                self.selected_obstacle_after_hit = None
                self.selected_obstacle_after_hit_side = None
                self.send_motion_cmd(self.post_hit_obstacle_search_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    '[POST_HIT_OBS] no obstacle detected, keep searching forward',
                    throttle_duration_sec=0.5
                )
                return

            if obs_count == 1:
                selected = obstacle_candidates[0]
                self.selected_obstacle_after_hit = selected

                img_w = self.latest_bgr.shape[1] if self.latest_bgr is not None else 640
                img_center_x = img_w // 2
                cx = selected.center_img[0]
                self.selected_obstacle_after_hit_side = 'left' if cx < img_center_x else 'right'

                d = selected.extra.get('median_depth')
                bottom_y_ratio = self.bbox_bottom_y_ratio(selected)
                trigger_reached = (
                    bottom_y_ratio is not None
                    and bottom_y_ratio >= self.post_hit_obstacle_trigger_bottom_y_ratio
                    if self.use_rgb_distance_triggers
                    else d is not None
                    and d <= self.post_hit_obstacle_trigger_distance_m)

                self.get_logger().warn(
                    f'[POST_HIT_OBS] only one obstacle detected, distance-only mode: '
                    f'side={self.selected_obstacle_after_hit_side}, center={selected.center_img}, '
                    f'depth={d}, bottom_y_ratio={bottom_y_ratio}, '
                    f'bottom_threshold={self.post_hit_obstacle_trigger_bottom_y_ratio:.3f}',
                    throttle_duration_sec=0.3
                )

                if trigger_reached:
                    self.get_logger().info(
                        f'[POST_HIT_OBS] one obstacle close enough: '
                        f'depth={d}, bottom_y_ratio={bottom_y_ratio}, '
                        f'dashed_side={self.dashed_side}, go post-hit turn task'
                    )
                    self.begin_voice_wait(
                        wait_state=self.POST_HIT_OBSTACLE_VOICE_WAIT,
                        event_id=f'post_hit_obstacle_before_turn_{self.completed_obstacle_count + 1}',
                        voice_key='obstacle',
                        resume_state=self.POST_HIT_OBS_TURN_1,
                    )
                    return

                # 只有一个障碍物但还没到距离：不居中，不给 vy，只继续直走。
                self.send_motion_cmd(self.post_hit_obstacle_forward_speed, 0.0, 0.0)
                return

            # 2 个及以上障碍物：保持原逻辑，按 dashed_side 选左/右障碍物并居中对齐。
            selected = self.choose_selected_obstacle_after_hit(obstacle_candidates)
            self.selected_obstacle_after_hit = selected

            if selected is None:
                self.selected_obstacle_after_hit_side = None
                self.send_motion_cmd(self.post_hit_obstacle_search_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_HIT_OBS] selected=None, dashed_side={self.dashed_side}, keep searching forward',
                    throttle_duration_sec=0.5
                )
                return

            vy = self.compute_selected_obstacle_align_vy_after_hit(selected)
            d = selected.extra.get('median_depth')
            bottom_y_ratio = self.bbox_bottom_y_ratio(selected)
            trigger_reached = (
                bottom_y_ratio is not None
                and bottom_y_ratio >= self.post_hit_obstacle_trigger_bottom_y_ratio
                if self.use_rgb_distance_triggers
                else d is not None
                and d <= self.post_hit_obstacle_trigger_distance_m)
            self.send_motion_cmd(self.post_hit_obstacle_forward_speed, vy, 0.0)

            self.get_logger().info(
                f'[POST_HIT_OBS] dashed_side={self.dashed_side}, selected={selected.center_img}, '
                f'depth={d}, bottom_y_ratio={bottom_y_ratio}, '
                f'cmd=({self.post_hit_obstacle_forward_speed:.3f},{vy:.3f},0.000)',
                throttle_duration_sec=0.2
            )

            if trigger_reached:
                self.selected_obstacle_after_hit_side = self.get_obstacle_side_in_pair(selected, obstacle_candidates)
                self.get_logger().info(
                    f'[POST_HIT_OBS] selected obstacle trigger: depth={d}, '
                    f'bottom_y_ratio={bottom_y_ratio}, '
                    f'dashed_side={self.dashed_side}, go post-hit turn task'
                )
                self.begin_voice_wait(
                    wait_state=self.POST_HIT_OBSTACLE_VOICE_WAIT,
                    event_id=f'post_hit_obstacle_before_turn_{self.completed_obstacle_count + 1}',
                    voice_key='obstacle',
                    resume_state=self.POST_HIT_OBS_TURN_1,
                )
                return

        elif self.state == self.BAR_BACKOFF_VOICE_WAIT:
            self.handle_voice_wait(
                pitch=self.bar_target_forward_pitch,
                body_height=self.bar_low_body_height,
            )
            return

        elif self.state == self.POST_HIT_OBSTACLE_VOICE_WAIT:
            self.handle_voice_wait()
            return

        elif self.state == self.POST_HIT_OBS_TURN_1:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_OBS_TURN_1] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, go forward'
                )
                self.enter_state(self.POST_HIT_OBS_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_fixed_turn_cmd(wz)


        elif self.state == self.POST_HIT_OBS_FORWARD:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_hit_obs_forward_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_OBS_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_hit_obs_forward_duration_s:.3f}s, go opposite turn'
                )
                self.enter_state(self.POST_HIT_OBS_TURN_2)
            else:
                self.send_motion_cmd(self.post_hit_obs_forward_speed, 0.0, 0.0)
                self.get_logger().info(
                    f'[POST_HIT_OBS_FORWARD] elapsed={elapsed:.3f}/{self.post_hit_obs_forward_duration_s:.3f}s',
                    throttle_duration_sec=0.2
                )


        elif self.state == self.POST_HIT_OBS_TURN_2:
            elapsed = self.state_elapsed_s()
            if elapsed >= self.current_turn_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_OBS_TURN_2] finished by sim time: elapsed={elapsed:.3f}/{self.current_turn_duration_s:.3f}s, go pre-final fixed forward'
                )
                # 注意：这里不恢复 normal，也不额外发送 0 速度。
                # 后面还要继续前进识别横向黄线，并完成最终 180 度掉头，
                # 这些障碍物收尾动作仍保持 low 姿态。
                self.enter_state(self.POST_HIT_PRE_FINAL_FORWARD)
            else:
                wz = self.current_turn_dir * abs(self.current_turn_wz)
                self.send_fixed_turn_cmd(wz)


        elif self.state == self.POST_HIT_PRE_FINAL_FORWARD:
            # 第二次转回后，先按仿真时间向前走一小段。
            # 新增：这一段如果已经能看到前方横向黄线，就只用黄线角度修正 wz，
            # 不用黄线提前结束该状态，仍然按 post_hit_final_forward_duration_s 到时后进入正式黄线阶段。
            elapsed = self.state_elapsed_s()
            if elapsed >= self.post_hit_final_forward_duration_s:
                self.get_logger().info(
                    f'[POST_HIT_PRE_FINAL_FORWARD] finished by sim time: elapsed={elapsed:.3f}/{self.post_hit_final_forward_duration_s:.3f}s, start final yellow detection'
                )
                self.enter_state(self.POST_HIT_FINAL_FORWARD)
            else:
                wz = 0.0
                final_yellow_line = None

                if self.post_hit_pre_final_angle_align_enabled:
                    final_yellow_line = self.final_yellow_detector.detect(frame)
                    self.latest_final_yellow_line = final_yellow_line
                    wz = self.compute_final_yellow_wz(final_yellow_line)

                self.send_motion_cmd(self.post_hit_final_forward_speed, 0.0, wz)

                if final_yellow_line is None:
                    self.get_logger().info(
                        f'[POST_HIT_PRE_FINAL_FORWARD] elapsed={elapsed:.3f}/{self.post_hit_final_forward_duration_s:.3f}s, '
                        f'no yellow angle reference, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )
                else:
                    angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
                    abs_tilt = float(final_yellow_line.extra.get('abs_tilt_deg', abs(angle_deg)))
                    bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
                    self.get_logger().info(
                        f'[POST_HIT_PRE_FINAL_FORWARD] elapsed={elapsed:.3f}/{self.post_hit_final_forward_duration_s:.3f}s, '
                        f'angle={angle_deg:.1f}deg, abs_tilt={abs_tilt:.1f}, bottom_ratio={bottom_ratio:.3f}, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )


        elif self.state in (
                self.POST_HIT_FINAL_FORWARD,
                self.BAR_YELLOW_FORWARD):
            # 障碍物与限高杆流程复用同一套横向黄线收尾参数。
            # 1. 没看到黄线：继续向前找黄线；
            # 2. 看到黄线但还没到下方阈值：vx 正常前进，同时用 wz 修正角度；
            # 3. 黄线已经到达下方阈值但角度没对正：vx=0, vy=0，只原地 wz 调角度；
            # 4. 黄线到达下方阈值且角度满足：进入各自的第二次 180 度转向。
            if self.state == self.BAR_YELLOW_FORWARD:
                log_label = 'BAR_YELLOW_FORWARD'
                next_state = self.BAR_TURN_BACK
                # The shared timed-turn posture uses pitch=0 by default.
                # Re-apply the under-bar forward-lean posture immediately after
                # the first 180-degree turn and keep carrying it throughout the
                # yellow-line approach/alignment stage.
                yellow_motion_pose = {
                    'pitch': self.bar_target_forward_pitch,
                    'body_height': self.bar_low_body_height,
                }
            else:
                log_label = 'POST_HIT_FINAL_FORWARD'
                next_state = self.FINAL_LEFT_JUMP
                yellow_motion_pose = {}

            final_yellow_line = self.final_yellow_detector.detect(frame)
            self.latest_final_yellow_line = final_yellow_line

            if final_yellow_line is None:
                self.final_yellow_done_counter = 0
                self.send_motion_cmd(
                    self.post_hit_final_forward_speed, 0.0, 0.0,
                    **yellow_motion_pose
                )
                self.get_logger().info(
                    f'[{log_label}] no horizontal yellow line, keep moving forward',
                    throttle_duration_sec=0.3
                )
                return

            bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
            bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
            angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
            abs_tilt = float(final_yellow_line.extra.get('abs_tilt_deg', abs(angle_deg)))
            wz = self.compute_final_yellow_wz(final_yellow_line)

            reached_line = bottom_ratio >= self.final_yellow_stop_line_y_ratio
            angle_ok = abs_tilt <= self.final_yellow_done_tilt_deg

            if reached_line and angle_ok:
                self.final_yellow_done_counter += 1
            else:
                self.final_yellow_done_counter = 0

            if self.final_yellow_done_counter >= self.final_yellow_confirm_count:
                self.get_logger().info(
                    f'[{log_label}] yellow reached lower area and aligned, '
                    f'go {next_state}'
                )
                self.send_motion_cmd(
                    0.0, 0.0, 0.0,
                    **yellow_motion_pose
                )
                self.enter_state(next_state)
                return

            if reached_line:
                # 黄线已经到图像下方，但角度还没满足：
                # 停止前进，原地只用 wz 调整朝向，避免黄线继续往下跑出画面。
                vx_cmd = 0.0
                vy_cmd = 0.0
                phase = 'reached_lower_align_in_place'
            else:
                # 黄线还没到下方阈值：
                # 正常向前靠近，同时用 wz 修正角度。
                vx_cmd = self.post_hit_final_forward_speed
                vy_cmd = 0.0
                phase = 'approach_with_angle_align'

            self.send_motion_cmd(
                vx_cmd, vy_cmd, wz,
                **yellow_motion_pose
            )

            self.get_logger().info(
                f'[{log_label}] {phase}: '
                f'bottom={bottom_y}, ratio={bottom_ratio:.3f}/{self.final_yellow_stop_line_y_ratio:.3f}, '
                f'angle={angle_deg:.1f}deg, abs_tilt={abs_tilt:.1f}/{self.final_yellow_done_tilt_deg:.1f}, '
                f'vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, wz={wz:.3f}, '
                f'pitch={self.body_pitch_cmd:.3f}, height={self.body_height_cmd:.3f}, '
                f'counter={self.final_yellow_done_counter}/{self.final_yellow_confirm_count}',
                throttle_duration_sec=0.2
            )

        elif self.state == self.BAR_TURN_BACK:
            self.execute_left_jump_turn(2, self.BAR_FLOW_DONE)
            return

        elif self.state == self.BAR_FLOW_DONE:
            self.get_logger().info(
                '[BAR_FLOW_DONE] second 180-degree turn finished; complete bar flow'
            )
            self.finish_bar_flow()
            return

        elif self.state == self.FINAL_LEFT_JUMP:
            # 障碍物流程内部的最终掉头。
            # 新姿态策略：
            #   POST_HIT_FINAL_FORWARD 前进识别横向黄线并对正后，
            #   先在 low 姿态下完成这里的最终掉头；
            #   掉头完成后进入 OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN，
            #   再 STOP -> NORMAL。
            if self.obstacle_flow_is_third_object:
                if self.completed_obstacle_count < self.required_obstacle_count:
                    self.completed_obstacle_count += 1

                self.after_obstacle_restore_next_state = self.GLOBAL_FINAL_YELLOW_FORWARD
                self.get_logger().info(
                    f'[FINAL_LEFT_JUMP] obstacle is the 3rd flow, '
                    f'execute final turn first, then restore NORMAL and jump directly to GLOBAL_FINAL_YELLOW_FORWARD. '
                    f'bar={self.completed_bar_count}/{self.required_bar_count}, '
                    f'obstacle={self.completed_obstacle_count}/{self.required_obstacle_count}'
                )
                self.execute_left_jump_turn(1, self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN)
                return

            self.after_obstacle_restore_next_state = self.OBSTACLE_FLOW_DONE
            self.get_logger().info(
                '[FINAL_LEFT_JUMP] timed left turn equivalent to 180 deg, then restore NORMAL and obstacle flow done')
            self.execute_left_jump_turn(2, self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN)

        elif self.state == self.OBSTACLE_RESTORE_NORMAL_AFTER_FINAL_TURN:
            next_state = getattr(self, 'after_obstacle_restore_next_state', self.OBSTACLE_FLOW_DONE)
            self.get_logger().warn(
                f'[BODY] obstacle final yellow + final turn finished, restore NORMAL, next={next_state}'
            )
            self.restore_body_normal_after_obstacle_final_turn()
            self.enter_state(next_state)
            return

        elif self.state == self.OBSTACLE_FLOW_DONE:
            self.finish_obstacle_flow()
            return

        elif self.state == self.GLOBAL_FINAL_RIGHT_JUMP:
            # 全部任务完成后的第一步：右跳一次。
            self.get_logger().info(
                '[GLOBAL_FINAL_RIGHT_JUMP] timed right turn equivalent to one right jump, then start final yellow alignment')
            self.execute_right_jump_turn(1, self.GLOBAL_FINAL_YELLOW_FORWARD)
            return

        elif self.state == self.GLOBAL_FINAL_YELLOW_FORWARD:
            # 右跳后继续前进，同时识别前方横向黄线并用倾斜角修正朝向。
            #
            # 新逻辑：
            # 1. 没看到黄线：快速前进
            # 2. 看到黄线但 bottom_ratio 还没到 slow_start_ratio：快速前进
            # 3. bottom_ratio >= slow_start_ratio：切换慢速前进
            # 4. bottom_ratio >= stop_line_y_ratio：认为黄线已经到图像下方区域
            # 5. 黄线到下方区域后，继续慢速前进，直到横向黄线从画面中消失
            # 6. 连续消失 global_final_yellow_disappear_confirm_count 帧后，进入最终左跳

            final_yellow_line = self.final_yellow_detector.detect(frame)
            self.latest_final_yellow_line = final_yellow_line

            if final_yellow_line is None:
                self.global_final_yellow_done_counter = 0

                if self.global_final_yellow_reached_lower_area:
                    # 已经确认黄线到过图像下方区域，现在看不到黄线，
                    # 说明机器狗可能已经越过黄线。
                    self.global_final_yellow_disappear_counter += 1

                    # 这里建议停住等待确认，避免继续冲太远。
                    self.send_motion_cmd(
                        0.0, 0.0, 0.0,
                        pitch=self.global_final_yellow_forward_pitch
                    )

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] yellow disappeared after reaching lower area: '
                        f'disappear_counter={self.global_final_yellow_disappear_counter}/'
                        f'{self.global_final_yellow_disappear_confirm_count}',
                        throttle_duration_sec=0.2
                    )

                    if self.global_final_yellow_disappear_counter >= self.global_final_yellow_disappear_confirm_count:
                        self.get_logger().info(
                            '[GLOBAL_FINAL_YELLOW_FORWARD] yellow disappeared after reached lower area, '
                            'go forward before final left jump'
                        )
                        self.send_motion_cmd(
                            0.0, 0.0, 0.0,
                            pitch=self.global_final_yellow_forward_pitch
                        )
                        self.enter_state(self.GLOBAL_FINAL_LEFT_JUMP)
                        return

                else:
                    # 还没有确认黄线到达过下方区域。
                    # 没看到黄线时继续快速前进寻找。
                    self.global_final_yellow_disappear_counter = 0

                    vx = self.global_final_yellow_forward_speed
                    self.send_motion_cmd(
                        vx, 0.0, 0.0,
                        pitch=self.global_final_yellow_forward_pitch
                    )

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] no horizontal yellow line before lower-area reached, '
                        f'keep moving forward, vx={vx:.3f}',
                        throttle_duration_sec=0.3
                    )

            else:
                # 看到横向黄线
                self.global_final_yellow_disappear_counter = 0

                bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
                bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))
                angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))

                wz = self.compute_final_yellow_wz(final_yellow_line)
                vx = self.get_global_final_yellow_forward_speed(final_yellow_line)

                reached_slow_area = bottom_ratio >= self.global_final_yellow_slow_start_ratio
                reached_line = bottom_ratio >= self.global_final_yellow_stop_line_y_ratio

                if reached_line:
                    # 黄线已经到达图像下方区域
                    self.global_final_yellow_done_counter += 1

                    if self.global_final_yellow_done_counter >= self.global_final_yellow_confirm_count:
                        self.global_final_yellow_reached_lower_area = True

                    # 到达下方区域后继续前进，等待黄线消失。
                    # 这里 vx 会因为 bottom_ratio 已经很大而自动变成慢速。
                    self.send_motion_cmd(
                        vx, 0.0, wz,
                        pitch=self.global_final_yellow_forward_pitch
                    )

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] yellow reached lower area, keep moving until it disappears: '
                        f'bottom={bottom_y}, '
                        f'ratio={bottom_ratio:.3f}/{self.global_final_yellow_stop_line_y_ratio:.3f}, '
                        f'slow_start={self.global_final_yellow_slow_start_ratio:.3f}, '
                        f'slow={reached_slow_area}, '
                        f'angle={angle_deg:.1f}deg, '
                        f'vx={vx:.3f}, wz={wz:.3f}, '
                        f'reach_counter={self.global_final_yellow_done_counter}/'
                        f'{self.global_final_yellow_confirm_count}, '
                        f'armed={self.global_final_yellow_reached_lower_area}',
                        throttle_duration_sec=0.2
                    )

                else:
                    # 黄线还没到最终下方阈值
                    self.global_final_yellow_done_counter = 0

                    # 如果 bottom_ratio 已经超过 slow_start_ratio，这里会自动用慢速；
                    # 否则继续快速靠近。
                    self.send_motion_cmd(
                        vx, 0.0, wz,
                        pitch=self.global_final_yellow_forward_pitch
                    )

                    self.get_logger().info(
                        f'[GLOBAL_FINAL_YELLOW] approach and align: '
                        f'bottom={bottom_y}, '
                        f'ratio={bottom_ratio:.3f}/{self.global_final_yellow_stop_line_y_ratio:.3f}, '
                        f'slow_start={self.global_final_yellow_slow_start_ratio:.3f}, '
                        f'slow={reached_slow_area}, '
                        f'angle={angle_deg:.1f}deg, '
                        f'vx={vx:.3f}, wz={wz:.3f}',
                        throttle_duration_sec=0.2
                    )

            return

        elif self.state == self.GLOBAL_FINAL_LEFT_JUMP:
            self.get_logger().info(
                '[GLOBAL_FINAL_LEFT_JUMP] timed left turn equivalent to one left jump, then right shift'
            )
            self.execute_left_jump_turn(1, self.GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP)
            return

        elif self.state == self.GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP:
            elapsed = self.state_elapsed_s()

            if elapsed < self.global_final_after_left_jump_right_shift_duration_s:
                self.send_motion_cmd(
                    0.0,
                    self.global_final_after_left_jump_right_shift_vy,
                    0.0
                )
                self.get_logger().info(
                    f'[GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP] right shift after left jump: '
                    f'elapsed={elapsed:.3f}/{self.global_final_after_left_jump_right_shift_duration_s:.3f}s, '
                    f'vy={self.global_final_after_left_jump_right_shift_vy:.3f}',
                    throttle_duration_sec=0.2
                )
                return

            self.get_logger().info(
                '[GLOBAL_FINAL_RIGHT_SHIFT_AFTER_LEFT_JUMP] done, go GLOBAL_FINAL_P3_ALIGN'
            )
            self.send_motion_cmd(0.0, 0.0, 0.0)
            self.enter_state(self.GLOBAL_FINAL_P3_ALIGN)
            return

        elif self.state == self.GLOBAL_FINAL_P3_ALIGN:
            # 第四赛段结束位置和第三赛段结束位置相同，直接复用第三赛段 P3_ALIGN_TRACK 的矫正逻辑。
            # 注意：第四赛段 RGB 回调不会跑 P3 视觉，所以这里主动调用 p3_process_yellow_track(frame)。
            self.p3_process_yellow_track(frame)
            elapsed = self.state_elapsed_s()

            if elapsed >= self.p3_align_max_duration_sec:
                self.get_logger().info(
                    f'[GLOBAL_FINAL_P3_ALIGN] timeout, finish all stages: '
                    f'elapsed={elapsed:.2f}/{self.p3_align_max_duration_sec:.2f}s'
                )
                if self.show_debug_vis:
                    self.p3_show_debug_window(frame)
                self.enter_state(self.DONE)
                return

            if self.p3_s4_valid > 0.5:
                err_lat = self.p3_s4_lat
                err_yaw = self.p3_s4_yaw

                if abs(err_lat) < self.p3_align_lat_tol and abs(err_yaw) < self.p3_align_yaw_tol:
                    self.get_logger().info(
                        f'[GLOBAL_FINAL_P3_ALIGN] complete: '
                        f'lat={err_lat:.4f}/{self.p3_align_lat_tol:.4f}, '
                        f'yaw={err_yaw:.4f}/{self.p3_align_yaw_tol:.4f}'
                    )
                    self.send_motion_cmd(0.0, 0.0, 0.0)
                    if self.show_debug_vis:
                        self.p3_show_debug_window(frame)
                    self.enter_state(self.DONE)
                    return

                lateral_speed = clamp(
                    err_lat * self.p3_align_lat_gain,
                    -self.p3_align_lat_max,
                    self.p3_align_lat_max
                )
                turn_speed = clamp(
                    err_yaw * self.p3_align_yaw_gain,
                    -self.p3_align_yaw_max,
                    self.p3_align_yaw_max
                )
                self.send_motion_cmd(0.0, lateral_speed, turn_speed)
                self.get_logger().info(
                    f'[GLOBAL_FINAL_P3_ALIGN] align: '
                    f'lat={err_lat:.4f}, yaw={err_yaw:.4f}, '
                    f'cmd=(0.000,{lateral_speed:.3f},{turn_speed:.3f})',
                    throttle_duration_sec=0.3
                )
                if self.show_debug_vis:
                    self.p3_show_debug_window(frame)
            else:
                self.send_motion_cmd(0.05, -0.04, 0.0)
                self.get_logger().info(
                    f'[GLOBAL_FINAL_P3_ALIGN] no valid track, searching: '
                    f'cmd=({self.p3_align_search_vx:.3f},0.000,{self.p3_align_search_wz:.3f})',
                    throttle_duration_sec=0.5
                )
                if self.show_debug_vis:
                    self.p3_show_debug_window(frame)
            return

        elif self.state == self.DONE:
            if not self.task_done_stop_sent:
                self.stop()
                self.task_done_stop_sent = True

        if self.show_debug_vis:
            self.update_debug_visualization(
                frame,
                obstacle_candidates,
                chosen_pair,
                dashed,
                target_candidates_for_vis,
                chosen_target_for_vis,
                final_yellow_line,
                bar_for_vis,
            )

        if now - self.last_log_time > 0.5:
            self.last_log_time = now
            vx, vy, wz = self.motion_cmd
            dashed_text = 'None' if dashed is None else f'{dashed.center_img}'
            self.get_logger().info(
                f'state={self.state} cmd=({vx:.3f},{vy:.3f},{wz:.3f}) bar={self.completed_bar_count}/{self.required_bar_count} obs={self.completed_obstacle_count}/{self.required_obstacle_count} '
                f'obs_candidates={len(obstacle_candidates)} dashed={dashed_text}'
            )

    # ---------- 可视化 ----------
    def update_debug_visualization(
            self,
            frame,
            obstacle_candidates: List[Detection],
            obstacle_pair: Optional[Tuple[Detection, Detection]],
            dashed: Optional[Detection],
            target_candidates: Optional[List[Detection]] = None,
            chosen_target: Optional[Detection] = None,
            final_yellow_line: Optional[Detection] = None,
            bar_det: Optional[Detection] = None,
    ):
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)

        # 黄线偏置对齐目标线：绿色是图像中心线，紫色是当前虚线对齐目标线
        if self.dashed_side is not None:
            target_x = int(round(self.get_dashed_target_x()))
            target_x = max(0, min(w - 1, target_x))
            cv2.line(vis, (target_x, 0), (target_x, h - 1), (255, 0, 255), 2)
            cv2.putText(
                vis,
                f'dash target side={self.dashed_side}',
                (target_x + 5, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2
            )

        cv2.putText(
            vis,
            f'state: {self.state}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        vx, vy, wz = self.motion_cmd
        cv2.putText(
            vis,
            f'cmd: ({vx:.2f},{vy:.2f},{wz:.2f})',
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # 画限高杆检测结果：红色框表示当前识别到的限高杆，中心点用于全局居中对齐。
        # 注意：限高杆完成 required_bar_count 次后，全局搜索阶段不再检测限高杆，窗口中也不会再更新 BAR 框。
        if bar_det is not None:
            x1, y1, x2, y2 = bar_det.bbox_img
            cx, cy = bar_det.center_img
            aspect = float(bar_det.extra.get('aspect_ratio', 0.0))
            top_y_ratio = y1 / float(max(h, 1))
            depth_yaw_info = getattr(self, 'latest_bar_depth_yaw_info', {})
            depth_err = depth_yaw_info.get('depth_error', None)
            left_depth = depth_yaw_info.get('left_depth', None)
            right_depth = depth_yaw_info.get('right_depth', None)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(vis, (cx, 0), (cx, h - 1), (0, 0, 255), 1)
            cv2.putText(
                vis,
                f'BAR {self.completed_bar_count}/{self.required_bar_count} '
                f'top={top_y_ratio:.3f} aspect={aspect:.1f} '
                f'L={left_depth} R={right_depth} err={depth_err}',
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )
            left_pt = depth_yaw_info.get('left_point', None)
            right_pt = depth_yaw_info.get('right_point', None)
            if left_pt is not None:
                cv2.circle(vis, tuple(left_pt), 5, (255, 0, 255), -1)
            if right_pt is not None:
                cv2.circle(vis, tuple(right_pt), 5, (255, 0, 255), -1)

        # 左上角显示总流程完成进度，方便判断当前还会不会继续检测限高杆/障碍物。
        cv2.putText(
            vis,
            f'progress: BAR {self.completed_bar_count}/{self.required_bar_count}  OBS {self.completed_obstacle_count}/{self.required_obstacle_count}',
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        if self.state in [self.POST_DASH_TURN_1, self.POST_DASH_TURN_2, self.POST_HIT_OBS_TURN_1,
                          self.POST_HIT_OBS_TURN_2]:
            turn_name = 'LEFT' if self.current_turn_dir > 0 else 'RIGHT'
            cv2.putText(
                vis,
                f'tf turn: {turn_name} target={math.degrees(self.current_turn_angle_rad):.0f}deg',
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 0, 255),
                2
            )

        # 画所有蓝色障碍物候选
        for i, det in enumerate(obstacle_candidates):
            x1, y1, x2, y2 = det.bbox_img
            cx, cy = det.center_img
            d = det.extra.get('median_depth')
            top_y_ratio = det.extra.get('top_y_ratio')

            color = (255, 0, 0)
            thickness = 2

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (cx, cy), 4, color, -1)

            metric_text = (
                f'd={d:.2f}' if d is not None
                else f'top={float(top_y_ratio):.3f}')
            cv2.putText(
                vis,
                f'OBS{i} {metric_text}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # 画用于居中的两个障碍物
        if obstacle_pair is not None:
            left, right = obstacle_pair
            lx, ly = left.center_img
            rx, ry = right.center_img
            mid_x = int((lx + rx) / 2)
            mid_y = int((ly + ry) / 2)

            cv2.circle(vis, (lx, ly), 6, (0, 255, 255), -1)
            cv2.circle(vis, (rx, ry), 6, (0, 255, 255), -1)
            cv2.circle(vis, (mid_x, mid_y), 7, (0, 0, 255), -1)
            cv2.line(vis, (lx, ly), (rx, ry), (0, 255, 255), 2)
            cv2.line(vis, (mid_x, 0), (mid_x, h - 1), (0, 0, 255), 1)

            cv2.putText(
                vis,
                'OBSTACLE MID',
                (mid_x + 5, max(20, mid_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

        # 画虚线
        if dashed is not None:
            x1, y1, x2, y2 = dashed.bbox_img
            cx, cy = dashed.center_img

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.circle(vis, (cx, cy), 6, (0, 165, 255), -1)

            centers = dashed.extra.get('group_centers', [])
            for k, p in enumerate(centers):
                px = int(round(p[0]))
                py = int(round(p[1]))
                cv2.circle(vis, (px, py), 4, (0, 165, 255), -1)

                if k > 0:
                    qx = int(round(centers[k - 1][0]))
                    qy = int(round(centers[k - 1][1]))
                    cv2.line(vis, (qx, qy), (px, py), (0, 165, 255), 2)

            cv2.putText(
                vis,
                f'DASH seg={dashed.extra.get("segments", 0)} span={dashed.extra.get("total_span_y", 0):.0f}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2
            )

        # 画第二次转向后的目标检测结果
        if target_candidates is None:
            target_candidates = []

        for det in target_candidates:
            x1, y1, x2, y2 = det.bbox_img
            cx, cy = det.center_img

            if det.det_type == 'blue_ball':
                color = (255, 0, 0)
            elif det.det_type == 'white_ball':
                color = (255, 255, 255)
            else:
                color = (0, 0, 255)

            thickness = 2
            if chosen_target is not None and det.det_type == chosen_target.det_type and det.center_img == chosen_target.center_img:
                thickness = 4

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (cx, cy), 5, color, -1)
            metrics = self.target_visual_metrics(det)
            cv2.putText(
                vis,
                f'TARGET {det.det_type} area={metrics["area_ratio"]} '
                f'r={metrics["radius_px"]}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

        # 画最后阶段前方横向黄线
        if final_yellow_line is not None:
            x1, y1, x2, y2 = final_yellow_line.bbox_img
            cx, cy = final_yellow_line.center_img
            angle_deg = float(final_yellow_line.extra.get('angle_deg', 0.0))
            bottom_y = int(final_yellow_line.extra.get('bottom_y', 0))
            bottom_ratio = float(final_yellow_line.extra.get('bottom_ratio', 0.0))

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
            cv2.putText(
                vis,
                f'FINAL YELLOW bottom={bottom_y} ratio={bottom_ratio:.2f} angle={angle_deg:.1f}',
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        cv2.imshow('obstacle_dashed_task_debug', vis)
        cv2.waitKey(1)




def main(args=None):
    rclpy.init(args=args)
    node = Stage4Node()

    # Camera ingestion is handled by stage4_rgb_rx on its own executor thread.
    # Keep the Stage4 state-machine executor single-threaded and deterministic;
    # this also avoids depending on MultiThreadedExecutor scheduling for camera
    # delivery on older rclpy/Galactic.
    executor = SingleThreadedExecutor(context=node.context)
    executor.add_node(node)
    node.get_logger().warning(
        '[P4_EXECUTOR] Stage4 main executor=SingleThreaded; '
        'RGB receiver uses independent SingleThreadedExecutor/thread'
    )

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node._close_voice_player()
        try:
            node._p4_stop_rgb_receiver()
        except Exception:
            pass
        try:
            if node.Ctrl is not None:
                node.send_stop_command()
        except Exception:
            pass
        try:
            if node.Ctrl is not None:
                node.Ctrl.quit()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            executor.shutdown(timeout_sec=0.5)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
