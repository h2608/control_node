"""Shared Stage 6 perception used by both control and the always-on preview."""

from collections import deque
import math

import cv2
import numpy as np


EXIT_YELLOW_LOWER = np.array([10, 50, 40])
EXIT_YELLOW_UPPER = np.array([40, 255, 255])
# Stage 6 north-line alignment must only accept vivid yellow tape.  Keep this
# separate from the permissive exit/wall colour range used later in the stage.
ALIGN_YELLOW_LOWER = np.array([26, 55, 170])
ALIGN_YELLOW_UPPER = np.array([38, 255, 255])


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    midpoint = 0.5 * float(np.sum(weights))
    index = int(np.searchsorted(np.cumsum(weights), midpoint, side='left'))
    return float(values[min(index, len(values) - 1)])


def detect_alignment_reference(image, hsv, depth_at, previous_y=None,
                               expected_angle_rad=None):
    """Find the single long, vivid-yellow transverse tape for Stage 6."""
    height, width = image.shape[:2]
    roi_x0, roi_x1 = int(width * 0.08), int(width * 0.92)
    # Preserve the old red box height and only widen it to the former blue-box
    # width, exactly matching the intended on-field search area.
    roi_y0, roi_y1 = int(height * 0.52), height - 1
    mask = cv2.inRange(hsv, ALIGN_YELLOW_LOWER, ALIGN_YELLOW_UPPER)
    mask[:roi_y0, :] = 0
    mask[roi_y1:, :] = 0
    mask[:, :roi_x0] = 0
    mask[:, roi_x1:] = 0
    # Stage4's useful part: remove isolated yellow speckles before joining the
    # long tape horizontally.  This reduces false Hough support from glare and
    # tiny yellowish objects without merging the cardboard into the tape.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8))
    edges = cv2.Canny(mask, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(36, int(width * 0.08)),
        minLineLength=max(90, int(width * 0.34)),
        maxLineGap=max(18, int(width * 0.08)),
    )
    empty = {
        'visible': False,
        'angle_rad': 0.0,
        'center_y': -1.0,
        'confidence': 0.0,
        'depth_m': -1.0,
        'segments': [],
        'roi': (roi_x0, roi_y0, roi_x1, roi_y1),
    }
    if lines is None:
        return empty

    center_x = width * 0.5
    candidates = []
    for packed in lines:
        x_start, y_start, x_end, y_end = [float(v) for v in packed[0]]
        if x_end < x_start:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start
        dy = y_end - y_start
        length = math.hypot(dx, dy)
        if dx < max(80.0, width * 0.30):
            continue
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if abs(angle_deg) > 15.0:
            continue
        center_y = y_start + (dy / max(dx, 1e-6)) * (center_x - x_start)
        if not (roi_y0 - 12 <= center_y <= roi_y1 + 12):
            continue
        # A geometrically long edge is not enough: most samples along it must
        # actually touch the strict vivid-yellow mask.  This rejects the brown
        # cardboard/wall base and chair/table edges.
        yellow_hits = 0
        yellow_checks = 31
        for sample_x in np.linspace(x_start, x_end, yellow_checks):
            sample_y = y_start + (dy / max(dx, 1e-6)) * (
                sample_x - x_start)
            sx = int(round(sample_x))
            sy = int(round(sample_y))
            patch = mask[
                max(0, sy - 6):min(height, sy + 7),
                max(0, sx - 2):min(width, sx + 3)]
            if patch.size and float(np.count_nonzero(patch)) / patch.size >= 0.10:
                yellow_hits += 1
        yellow_coverage = yellow_hits / float(yellow_checks)
        if yellow_coverage < 0.70:
            continue

        sample_x0 = max(float(roi_x0), x_start)
        sample_x1 = min(float(roi_x1), x_end)
        depths = []
        if sample_x1 - sample_x0 >= width * 0.18:
            for sample_x in np.linspace(sample_x0, sample_x1, 5):
                sample_y = y_start + (dy / max(dx, 1e-6)) * (
                    sample_x - x_start)
                value = float(depth_at(
                    sample_x, sample_y - 6.0,
                    patch_radius=3, max_depth=5.0))
                if value > 0.0 and np.isfinite(value):
                    depths.append(value)

        depth_m = float(np.median(depths)) if depths else -1.0
        depth_quality = 0.55
        if len(depths) >= 3:
            spread = float(np.percentile(depths, 90) - np.percentile(depths, 10))
            allowed = max(0.10, depth_m * 0.28)
            depth_quality = 1.0 / (1.0 + (spread / allowed) ** 2)

        horizontal_quality = math.exp(-0.5 * (abs(angle_deg) / 9.0) ** 2)
        score = length * (0.25 + 0.75 * horizontal_quality)
        score *= (0.55 + 0.45 * depth_quality)
        score *= yellow_coverage
        # If two genuine yellow transverse lines exist, prefer the lower one,
        # which is the nearest field boundary in the camera view.
        bottom_fraction = (
            (center_y - roi_y0) / max(1.0, float(roi_y1 - roi_y0)))
        score *= 0.85 + 0.15 * max(0.0, min(1.0, bottom_fraction))
        candidates.append({
            'angle_rad': angle_rad,
            'angle_deg': angle_deg,
            'center_y': center_y,
            'length': length,
            'score': score,
            'depth_m': depth_m,
            'depth_quality': depth_quality,
            'yellow_coverage': yellow_coverage,
            'x0': x_start,
            'y0': y_start,
            'x1': x_end,
            'y1': y_end,
        })

    if not candidates:
        return empty

    best_cluster = None
    best_cluster_score = -1.0
    for seed in candidates:
        cluster = [
            item for item in candidates
            if abs(item['center_y'] - seed['center_y']) <= 16.0
            and abs(item['angle_deg'] - seed['angle_deg']) <= 4.0
        ]
        span = max(item['x1'] for item in cluster) - min(
            item['x0'] for item in cluster)
        cluster_score = sum(item['score'] for item in cluster)
        cluster_score *= 0.45 + 0.55 * min(1.0, span / (width * 0.65))
        if cluster_score > best_cluster_score:
            best_cluster = cluster
            best_cluster_score = cluster_score

    weights = [item['score'] for item in best_cluster]
    angles = [item['angle_rad'] for item in best_cluster]
    center_ys = [item['center_y'] for item in best_cluster]
    angle_rad = _weighted_median(angles, weights)
    center_y = _weighted_median(center_ys, weights)
    angle_mad_deg = math.degrees(_weighted_median(
        [abs(value - angle_rad) for value in angles], weights))
    span = max(item['x1'] for item in best_cluster) - min(
        item['x0'] for item in best_cluster)
    total_support = sum(item['length'] for item in best_cluster)
    # A single clean Hough segment across 40% of the full image is already a
    # long yellow field line.  The former 52% accumulated-length gate rejected
    # valid motion-blurred frames even when the strict-yellow contour itself
    # spanned more than 80% of the image.
    if span < width * 0.40 or total_support < width * 0.40:
        return empty
    coverage_quality = min(1.0, span / (width * 0.55))
    support_quality = min(
        1.0, total_support / (width * 1.20))
    coherence_quality = math.exp(-angle_mad_deg / 2.0)
    yellow_quality = float(np.average(
        [item['yellow_coverage'] for item in best_cluster], weights=weights))
    depth_quality = float(np.average(
        [item['depth_quality'] for item in best_cluster], weights=weights))
    confidence = (
        0.25 * coverage_quality +
        0.20 * support_quality +
        0.25 * coherence_quality +
        0.20 * yellow_quality +
        0.10 * depth_quality
    )
    valid_depths = [
        item['depth_m'] for item in best_cluster if item['depth_m'] > 0.0]
    depth_m = float(np.median(valid_depths)) if valid_depths else -1.0
    return {
        'visible': confidence >= 0.55,
        'angle_rad': float(angle_rad),
        'center_y': float(center_y),
        'confidence': float(confidence),
        'depth_m': depth_m,
        'segments': [
            (int(item['x0']), int(item['y0']),
             int(item['x1']), int(item['y1']))
            for item in best_cluster
        ],
        'roi': (roi_x0, roi_y0, roi_x1, roi_y1),
    }


class AlignmentReferenceTracker:
    """Time-windowed robust filter for the precision alignment angle."""

    def __init__(self, history_window_s=0.90, minimum_history_s=0.60,
                 minimum_samples=6):
        self.history_window_s = float(history_window_s)
        self.minimum_history_s = float(minimum_history_s)
        self.minimum_samples = int(minimum_samples)
        self.history = deque(maxlen=90)
        self.filtered_angle_rad = None
        self.filtered_center_y = None

    def reset(self):
        self.history.clear()
        self.filtered_angle_rad = None
        self.filtered_center_y = None

    def update(self, observation, timestamp_s, frame_seq):
        timestamp_s = float(timestamp_s)
        accepted = bool(
            observation.get('visible', False)
            and observation.get('confidence', 0.0) >= 0.55)
        if accepted and self.history:
            recent_angles = [item['angle_rad'] for item in self.history]
            recent_ys = [item['center_y'] for item in self.history]
            median_angle = float(np.median(recent_angles))
            median_y = float(np.median(recent_ys))
            angle_jump = abs(math.degrees(
                observation['angle_rad'] - median_angle))
            y_jump = abs(observation['center_y'] - median_y)
            # Reject identity switches, but allow a real robot turn to move the
            # same reference when its vertical location remains continuous.
            if angle_jump > 5.0 and y_jump > 28.0:
                accepted = False

        if accepted:
            self.history.append({
                'timestamp_s': timestamp_s,
                'frame_seq': int(frame_seq),
                'angle_rad': float(observation['angle_rad']),
                'center_y': float(observation['center_y']),
                'confidence': float(observation['confidence']),
            })

        cutoff = timestamp_s - self.history_window_s
        while self.history and self.history[0]['timestamp_s'] < cutoff:
            self.history.popleft()

        result = {
            'valid': False,
            'stable': False,
            'angle_rad': 0.0,
            'center_y': -1.0,
            'confidence': 0.0,
            'sample_count': len(self.history),
            'history_span_s': 0.0,
            'mad_deg': 999.0,
            'range_deg': 999.0,
            'frame_seq': int(frame_seq),
            'raw_angle_rad': float(observation.get('angle_rad', 0.0)),
            'raw_visible': bool(observation.get('visible', False)),
            'depth_m': float(observation.get('depth_m', -1.0)),
            'segments': observation.get('segments', []),
            'roi': observation.get('roi'),
        }
        # Show a valid green TRACK line from the first accepted frame.  The
        # separate ``stable`` flag below still requires the configured sample
        # count and time span before robot control may use it.
        if not accepted or len(self.history) < 1:
            return result

        angles = [item['angle_rad'] for item in self.history]
        weights = [item['confidence'] for item in self.history]
        center_ys = [item['center_y'] for item in self.history]
        median_angle = _weighted_median(angles, weights)
        median_y = _weighted_median(center_ys, weights)
        if self.filtered_angle_rad is None:
            self.filtered_angle_rad = median_angle
            self.filtered_center_y = median_y
        else:
            alpha = 0.35
            self.filtered_angle_rad = (
                (1.0 - alpha) * self.filtered_angle_rad + alpha * median_angle)
            self.filtered_center_y = (
                (1.0 - alpha) * self.filtered_center_y + alpha * median_y)

        deviations = [abs(value - median_angle) for value in angles]
        mad_deg = math.degrees(_weighted_median(deviations, weights))
        robust_range_deg = math.degrees(
            float(np.percentile(angles, 90) - np.percentile(angles, 10)))
        history_span_s = self.history[-1]['timestamp_s'] - self.history[0]['timestamp_s']
        mean_confidence = float(np.mean(weights))
        stable = bool(
            len(self.history) >= self.minimum_samples
            and history_span_s >= self.minimum_history_s
            and mad_deg <= 0.35
            and robust_range_deg <= 1.00
            and mean_confidence >= 0.60
        )
        result.update({
            'valid': True,
            'stable': stable,
            'angle_rad': float(self.filtered_angle_rad),
            'center_y': float(self.filtered_center_y),
            'confidence': mean_confidence,
            'sample_count': len(self.history),
            'history_span_s': float(history_span_s),
            'mad_deg': float(mad_deg),
            'range_deg': float(robust_range_deg),
        })
        return result


def draw_alignment_reference(image, alignment):
    """Draw the one and only Stage 6 yellow-line result in green."""
    roi = alignment.get('roi')
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)

    if alignment.get('valid', False):
        height, width = image.shape[:2]
        center_x = width * 0.5
        center_y = float(alignment['center_y'])
        slope = math.tan(float(alignment['angle_rad']))
        x0, x1 = int(width * 0.08), int(width * 0.92)
        y0 = int(center_y + slope * (x0 - center_x))
        y1 = int(center_y + slope * (x1 - center_x))
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.circle(image, (int(center_x), int(center_y)),
                   7, (255, 0, 255), -1)
        status = 'PASSABLE' if alignment.get('stable', False) else 'TRACK'
        cv2.putText(
            image,
            'YELLOW {} angle={:.1f}deg depth={:.2f}m conf={:.2f} n={} mad={:.1f}'.format(
                status,
                math.degrees(float(alignment['angle_rad'])),
                float(alignment.get('depth_m', -1.0)),
                float(alignment['confidence']),
                int(alignment['sample_count']),
                float(alignment['mad_deg'])),
            (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
    else:
        cv2.putText(image, 'YELLOW LINE: NOT DETECTED', (12, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2)


def annotate_exit(image, hsv, depth_at):
    """Draw Stage 6 wall/exit contours and return the detected gap."""
    height, width = image.shape[:2]
    mask = cv2.inRange(hsv, EXIT_YELLOW_LOWER, EXIT_YELLOW_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    walls = []
    for contour in contours:
        if cv2.contourArea(contour) <= 1500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        walls.append({'x': x, 'y': y, 'w': w, 'h': h,
                      'bottom_y': y + h, 'contour': contour})
    if not walls:
        cv2.putText(image, 'EXIT: SEARCHING', (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 165, 255), 2)
        return {'visible': False, 'offset_norm': -999.0, 'distance_m': -1.0}
    walls.sort(key=lambda item: item['bottom_y'], reverse=True)
    nearest_y = walls[0]['bottom_y']
    front = [wall for wall in walls if abs(wall['bottom_y'] - nearest_y) < 350]
    front.sort(key=lambda item: item['x'])
    for index, wall in enumerate(front):
        cv2.drawContours(image, [wall['contour']], -1, (0, 255, 255), 2)
        cv2.putText(image, 'EW{}'.format(index + 1),
                    (wall['x'], max(20, wall['y'] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    best = None
    for left, right in zip(front, front[1:]):
        left_edge = left['x'] + left['w']
        right_edge = right['x']
        gap = right_edge - left_edge
        if gap > 20 and (best is None or gap > best[0]):
            best = (gap, left_edge, right_edge)
    if best is None:
        cv2.putText(image, 'EXIT: GAP INVALID', (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 165, 255), 2)
        return {'visible': False, 'offset_norm': -999.0, 'distance_m': -1.0}
    _, left_edge, right_edge = best
    gap_x = (left_edge + right_edge) / 2.0
    cv2.line(image, (int(gap_x), 0), (int(gap_x), height), (255, 0, 255), 3)
    py = max(5, min(height - 6, nearest_y - 10))
    distances = [float(depth_at(max(5, left_edge - 5), py,
                                patch_radius=5, max_depth=5.0)),
                 float(depth_at(min(width - 6, right_edge + 5), py,
                                patch_radius=5, max_depth=5.0))]
    distances = [value for value in distances if value > 0]
    distance = min(distances) if distances else -1.0
    offset = float((gap_x - width / 2.0) / (width / 2.0))
    cv2.putText(image, 'EXIT off={:.2f} depth={:.2f}m'.format(offset, distance),
                (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 0, 255), 2)
    return {'visible': True, 'offset_norm': offset, 'distance_m': distance}
