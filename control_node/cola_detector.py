#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四赛段可乐瓶视觉检测核心。

不依赖 ROS 消息类型，便于比赛节点和独立调试节点共用。检测采用两条路径：
1. 红色瓶盖锚点 + 下方局部深色瓶身（真机主路径）；
2. 深色直立瓶形（仿真和瓶盖不可见时的后备路径）。
"""

import math

import cv2
import numpy as np


def _clamp(value, low, high):
    return max(low, min(high, value))


class HybridColaDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.open_kernel = np.ones(
            (cfg['open_kernel'], cfg['open_kernel']), np.uint8)
        self.close_kernel = np.ones(
            (cfg['close_kernel'], cfg['close_kernel']), np.uint8)
        self.smoothed_center_x = None

    @staticmethod
    def _find_contours(mask):
        # OpenCV 3: (image, contours, hierarchy)
        # OpenCV 4: (contours, hierarchy)
        return cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    @staticmethod
    def _row_widths(binary):
        widths = []
        for row in binary:
            xs = np.flatnonzero(row)
            widths.append(0.0 if xs.size == 0 else float(xs[-1] - xs[0] + 1))
        return np.asarray(widths, dtype=np.float32)

    @staticmethod
    def _median_positive(values):
        values = values[values > 0]
        return 0.0 if values.size == 0 else float(np.median(values))

    def _dark_shape_candidates(self, mask, roi, offset_x, offset_y):
        roi_h, roi_w = roi.shape[:2]
        candidates = []

        for contour in self._find_contours(mask):
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue
            if not (self.cfg['min_area'] <= area <= self.cfg['max_area']):
                continue
            if not (self.cfg['min_width'] <= width <= self.cfg['max_width']):
                continue
            if not (self.cfg['min_height'] <= height <= self.cfg['max_height']):
                continue

            aspect = height / float(width)
            if not (self.cfg['min_aspect'] <= aspect <= self.cfg['max_aspect']):
                continue
            bottom_ratio = (y + height) / float(max(roi_h, 1))
            if bottom_ratio < self.cfg['min_bottom_ratio']:
                continue

            contour_mask = np.zeros((height, width), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, 0, 0] -= x
            shifted[:, 0, 1] -= y
            cv2.drawContours(contour_mask, [shifted], -1, 255, cv2.FILLED)

            fill_ratio = area / float(width * height)
            if not (self.cfg['min_fill_ratio'] <= fill_ratio <=
                    self.cfg['max_fill_ratio']):
                continue

            mirrored = cv2.flip(contour_mask, 1)
            intersection = np.count_nonzero(
                (contour_mask > 0) & (mirrored > 0))
            union = np.count_nonzero((contour_mask > 0) | (mirrored > 0))
            symmetry = intersection / float(max(union, 1))
            if symmetry < self.cfg['min_symmetry']:
                continue

            widths = self._row_widths(contour_mask)
            top_end = max(1, int(height * 0.28))
            body_start = min(height - 1, int(height * 0.38))
            body_end = max(body_start + 1, int(height * 0.82))
            top_width = self._median_positive(widths[:top_end])
            body_width = self._median_positive(widths[body_start:body_end])
            shoulder_ratio = top_width / max(body_width, 1.0)
            if not (self.cfg['min_shoulder_ratio'] <= shoulder_ratio <=
                    self.cfg['max_shoulder_ratio']):
                continue

            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(hull_area, 1.0)
            if solidity < self.cfg['min_solidity']:
                continue

            center_x = x + width * 0.5
            center_bonus = 1.0 - abs(center_x - roi_w * 0.5) / max(
                roi_w * 0.5, 1.0)
            aspect_score = math.exp(
                -abs(aspect - self.cfg['target_aspect']) /
                max(self.cfg['target_aspect'], 0.1))
            shoulder_score = _clamp(
                (self.cfg['max_shoulder_ratio'] - shoulder_ratio) /
                max(self.cfg['max_shoulder_ratio'] -
                    self.cfg['min_shoulder_ratio'], 0.01),
                0.0,
                1.0,
            )
            size_score = _clamp(
                area / self.cfg['full_size_area'], 0.0, 1.0)
            score = 100.0 * (
                0.25 * aspect_score +
                0.22 * symmetry +
                0.20 * shoulder_score +
                0.13 * solidity +
                0.10 * fill_ratio +
                0.10 * center_bonus
            ) * (0.75 + 0.25 * size_score)
            if score < self.cfg['min_score']:
                continue

            candidates.append({
                'bbox': (offset_x + x, offset_y + y,
                         offset_x + x + width, offset_y + y + height),
                'center': (int(offset_x + x + width * 0.5),
                           int(offset_y + y + height * 0.5)),
                'score': float(score),
                'method': 'dark_shape',
                'area': area,
                'aspect': float(aspect),
                'fill_ratio': float(fill_ratio),
                'symmetry': float(symmetry),
                'shoulder_ratio': float(shoulder_ratio),
                'solidity': float(solidity),
            })

        return candidates

    def _red_cap_mask(self, hsv):
        red_low = cv2.inRange(
            hsv,
            np.array([0, self.cfg['cap_s_min'], self.cfg['cap_v_min']],
                     dtype=np.uint8),
            np.array([self.cfg['cap_h_low_max'], 255, 255], dtype=np.uint8),
        )
        red_high = cv2.inRange(
            hsv,
            np.array([self.cfg['cap_h_high_min'], self.cfg['cap_s_min'],
                      self.cfg['cap_v_min']], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        red_mask = cv2.bitwise_or(red_low, red_high)
        return cv2.morphologyEx(
            red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _cap_only_candidates(self, hsv, roi, offset_x, offset_y):
        """Detect compact red caps without requiring a dark bottle body."""
        red_mask = self._red_cap_mask(hsv)
        roi_h, roi_w = roi.shape[:2]
        roi_area = float(max(roi_h * roi_w, 1))
        candidates = []

        for contour in self._find_contours(red_mask):
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue
            area_ratio = area / roi_area
            aspect = width / float(height)
            if not (self.cfg['cap_min_area_ratio'] <= area_ratio <=
                    self.cfg['cap_max_area_ratio']):
                continue
            if not (self.cfg['cap_min_aspect'] <= aspect <=
                    self.cfg['cap_max_aspect']):
                continue

            fill_ratio = area / float(max(width * height, 1))
            aspect_score = math.exp(-abs(aspect - 1.0))
            score = 100.0 * (0.65 * aspect_score + 0.35 * fill_ratio)
            bbox = (offset_x + x, offset_y + y,
                    offset_x + x + width, offset_y + y + height)
            candidates.append({
                'bbox': bbox,
                'center': (int(offset_x + x + width * 0.5),
                           int(offset_y + y + height * 0.5)),
                'score': float(score),
                'method': 'cap_only',
                'cap_bbox': bbox,
                'area': area,
                'cap_area_ratio': float(area_ratio),
                'cap_aspect': float(aspect),
                'fill_ratio': float(fill_ratio),
            })

        return candidates

    def _cap_anchor_candidates(self, hsv, dark_mask, roi, offset_x, offset_y):
        red_mask = self._red_cap_mask(hsv)

        roi_h, roi_w = roi.shape[:2]
        roi_area = float(max(roi_h * roi_w, 1))
        candidates = []

        for cap_contour in self._find_contours(red_mask):
            cap_area = float(cv2.contourArea(cap_contour))
            cap_x, cap_y, cap_w, cap_h = cv2.boundingRect(cap_contour)
            if cap_w <= 0 or cap_h <= 0:
                continue
            cap_area_ratio = cap_area / roi_area
            cap_aspect = cap_w / float(cap_h)
            if not (self.cfg['cap_min_area_ratio'] <= cap_area_ratio <=
                    self.cfg['cap_max_area_ratio']):
                continue
            if not (self.cfg['cap_min_aspect'] <= cap_aspect <=
                    self.cfg['cap_max_aspect']):
                continue

            cap_center_x = cap_x + cap_w * 0.5
            # A real bottle cap must anchor a nearby body.  The old search
            # window was wide enough to pair a red desk/bar edge with an
            # unrelated dark object below it.
            search_half_w = int(max(cap_w * 2.4, roi_w * 0.025))
            search_x1 = max(0, int(cap_center_x) - search_half_w)
            search_x2 = min(roi_w, int(cap_center_x) + search_half_w + 1)
            search_y1 = min(roi_h, cap_y + cap_h)
            search_height = int(max(cap_h * 18.0, roi_h * 0.18))
            search_y2 = min(roi_h, search_y1 + search_height)
            if search_x2 <= search_x1 or search_y2 <= search_y1:
                continue

            local_mask = dark_mask[search_y1:search_y2, search_x1:search_x2]
            for body_contour in self._find_contours(local_mask):
                body_area = float(cv2.contourArea(body_contour))
                local_x, local_y, body_w, body_h = cv2.boundingRect(body_contour)
                if body_w <= 0 or body_h <= 0:
                    continue
                if body_area < max(self.cfg['cap_body_min_area'],
                                   cap_area * self.cfg['cap_body_area_gain']):
                    continue
                if body_h < max(self.cfg['cap_body_min_height'], cap_h * 2.0):
                    continue
                if body_w < self.cfg['cap_body_min_width']:
                    continue

                body_x = search_x1 + local_x
                body_y = search_y1 + local_y
                body_center_x = body_x + body_w * 0.5
                body_to_cap_width = body_w / float(max(cap_w, 1))
                if not (1.25 <= body_to_cap_width <= 6.0):
                    continue
                x_error = abs(body_center_x - cap_center_x)
                if x_error > max(body_w * 0.30, cap_w * 1.0):
                    continue
                gap = max(0, body_y - (cap_y + cap_h))
                max_gap = max(body_h * 0.22, cap_h * 3.0)
                if gap > max_gap:
                    continue

                box_x1 = min(cap_x, body_x)
                box_y1 = cap_y
                box_x2 = max(cap_x + cap_w, body_x + body_w)
                box_y2 = body_y + body_h
                box_w = box_x2 - box_x1
                box_h = box_y2 - box_y1
                aspect = box_h / float(max(box_w, 1))
                if not (self.cfg['cap_bottle_min_aspect'] <= aspect <=
                        self.cfg['cap_bottle_max_aspect']):
                    continue

                body_fill = body_area / float(max(body_w * body_h, 1))
                hull_area = float(cv2.contourArea(cv2.convexHull(body_contour)))
                solidity = body_area / max(hull_area, 1.0)
                alignment = 1.0 - _clamp(
                    x_error / max(box_w * 0.5, 1.0), 0.0, 1.0)
                aspect_score = math.exp(-abs(aspect - 2.6) / 2.6)
                gap_score = 1.0 - _clamp(
                    gap / max(max_gap, 1.0), 0.0, 1.0)
                # All identity checks above are explicit hard conditions.
                # This score only ranks multiple valid bottles; it is not a
                # second accept/reject gate.
                score = 100.0 * (
                    0.50 * alignment +
                    0.30 * gap_score +
                    0.20 * aspect_score
                )

                candidates.append({
                    'bbox': (offset_x + box_x1, offset_y + box_y1,
                             offset_x + box_x2, offset_y + box_y2),
                    'center': (
                        int(offset_x + (cap_center_x + body_center_x) * 0.5),
                        int(offset_y + (box_y1 + box_y2) * 0.5),
                    ),
                    'score': float(score),
                    'method': 'cap_anchor',
                    'cap_bbox': (
                        offset_x + cap_x, offset_y + cap_y,
                        offset_x + cap_x + cap_w, offset_y + cap_y + cap_h,
                    ),
                    'area': body_area,
                    'aspect': float(aspect),
                    'fill_ratio': float(body_fill),
                    'symmetry': float(alignment),
                    'shoulder_ratio': -1.0,
                    'solidity': float(solidity),
                    'body_to_cap_width': float(body_to_cap_width),
                    'cap_body_gap': float(gap),
                })

        return candidates

    def detect(self, frame_bgr, candidate_key=None):
        frame_h, frame_w = frame_bgr.shape[:2]
        x1 = int(frame_w * self.cfg['roi_x_ratio_min'])
        x2 = int(frame_w * self.cfg['roi_x_ratio_max'])
        y1 = int(frame_h * self.cfg['roi_y_ratio_min'])
        y2 = int(frame_h * self.cfg['roi_y_ratio_max'])
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(y1 + 1, min(y2, frame_h))
        roi = frame_bgr[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if self.cfg.get('cap_only_mode', False):
            candidates = self._cap_only_candidates(hsv, roi, x1, y1)
        else:
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            dark_coloured = (value <= self.cfg['dark_v_max']) & (
                saturation >= self.cfg['dark_s_min'])
            very_dark = value <= self.cfg['very_dark_v_max']
            dark_mask = np.where(
                dark_coloured | very_dark, 255, 0).astype(np.uint8)
            dark_mask = cv2.morphologyEx(
                dark_mask, cv2.MORPH_OPEN, self.open_kernel)
            dark_mask = cv2.morphologyEx(
                dark_mask, cv2.MORPH_CLOSE, self.close_kernel)

            candidates = []
            if self.cfg.get('enable_dark_shape', True):
                candidates.extend(
                    self._dark_shape_candidates(dark_mask, roi, x1, y1))
            candidates.extend(
                self._cap_anchor_candidates(hsv, dark_mask, roi, x1, y1))
        if not candidates:
            self.smoothed_center_x = None
            return None

        # By default choose the strongest visual candidate.  Callers may
        # provide a key without coupling this detector to stage-specific
        # candidate-priority rules.
        if candidate_key is None:
            result = max(candidates, key=lambda item: item['score'])
        else:
            result = min(candidates, key=candidate_key)
        raw_center_x = float(result['center'][0])
        if (self.smoothed_center_x is None or
                abs(raw_center_x - self.smoothed_center_x) >
                frame_w * self.cfg['max_center_jump_ratio']):
            self.smoothed_center_x = raw_center_x
        else:
            alpha = self.cfg['center_smoothing_alpha']
            self.smoothed_center_x = (
                (1.0 - alpha) * self.smoothed_center_x + alpha * raw_center_x)

        result = dict(result)
        result['raw_center_x'] = raw_center_x
        result['smoothed_center_x'] = float(self.smoothed_center_x)
        result['center'] = (
            int(round(self.smoothed_center_x)), int(result['center'][1]))
        result['offset_x_px'] = float(self.smoothed_center_x - frame_w * 0.5)
        result['offset_x_norm'] = float(
            result['offset_x_px'] / max(frame_w * 0.5, 1.0))
        result['roi_bbox'] = (x1, y1, x2, y2)
        return result
