"""Black-and-white size-4 football detector for the physical CyberDog RGB camera.

Stage 4 acquires the ball from several metres away; Stage 6 only ever sees it
during the final close push, where chair castors are the dominant false
positive.  The shape/panel checks are shared, but the range-dependent gates
differ, so they are keyword arguments: the defaults are the far-field
Stage-4 behaviour, and Stage 6 passes ``**NEAR_BALL``.

Hard-coding the Stage-6 values is what made this file undeployable on its
own -- min radius 38 px, lower image half only and a 1.00 m depth cap make
Stage 4 blind past about a metre.
"""

import math

import cv2
import numpy as np


# Stage 6 final push: the ball is large, close and low in the frame.
NEAR_BALL = {
    'min_radius_px': 38,
    'min_radius_ratio': 0.080,
    'max_radius_ratio': 0.34,
    'roi_top_ratio': 0.22,
    'min_y_ratio': 0.46,
    'min_pixels': 180,
    'min_visible_fraction': 0.58,
    'min_dark_panels': 2,
    'max_depth_m': 1.00,
    'max_diameter_m': 0.28,
    'hough_param2': 28,
}


def _dark_panel_count(hsv, inner_mask, x, y, radius):
    """Count separate, ball-sized dark panels inside a circle candidate."""
    height, width = hsv.shape[:2]
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(width, int(math.ceil(x + radius + 1)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(height, int(math.ceil(y + radius + 1)))
    if x1 <= x0 or y1 <= y0:
        return 0, 0.0, 999.0

    dark = (hsv[y0:y1, x0:x1, 2] < 105).astype(np.uint8) * 255
    dark[~inner_mask[y0:y1, x0:x1]] = 0
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours = cv2.findContours(
        dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    circle_area = math.pi * radius * radius
    min_area = max(18.0, circle_area * 0.003)
    max_area = circle_area * 0.30
    count = 0
    largest_fraction = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    total_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (min_area <= area <= max_area):
            continue
        moments = cv2.moments(contour)
        if abs(moments['m00']) < 1e-6:
            continue
        cx = x0 + moments['m10'] / moments['m00']
        cy = y0 + moments['m01'] / moments['m00']
        if math.hypot(cx - x, cy - y) > radius * 0.88:
            continue
        count += 1
        largest_fraction = max(largest_fraction, area / circle_area)
        weighted_x += cx * area
        weighted_y += cy * area
        total_area += area
    if total_area <= 0.0:
        return count, largest_fraction, 999.0
    panel_cx = weighted_x / total_area
    panel_cy = weighted_y / total_area
    centroid_offset = math.hypot(panel_cx - x, panel_cy - y) / max(radius, 1.0)
    return count, largest_fraction, centroid_offset


def detect_football(image, depth_at=None, focal_px=400.0, *,
                    min_radius_px=7, min_radius_ratio=0.015,
                    max_radius_ratio=0.20, roi_top_ratio=0.28,
                    min_y_ratio=0.34, min_pixels=80,
                    min_visible_fraction=0.0, min_dark_panels=0,
                    max_depth_m=None, max_diameter_m=0.34,
                    hough_param2=40):
    """Return the best close black-and-white football candidate, or ``None``.

    Final Stage 6 pushing sees a large ball near the lower image boundary.
    Chair castors are also circular and black, so circularity alone is not
    sufficient: a candidate must contain a bright ball surface plus at least
    two separate dark football panels.  A close ball may be partly clipped by
    the bottom edge and remains valid as long as most of the circle is visible.
    """
    if image is None or image.ndim < 3:
        return None
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray_raw, (9, 9), 1.8)

    roi_y = int(height * roi_top_ratio)
    roi_gray = gray[roi_y:, :]
    min_radius = max(int(min_radius_px), int(min(width, height) * min_radius_ratio))
    max_radius = max(min_radius + 4, int(min(width, height) * max_radius_ratio))
    circles = cv2.HoughCircles(
        roi_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(38, int(width * 0.065)),
        param1=100,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    yy, xx = np.ogrid[:height, :width]
    edges = cv2.Canny(gray, 55, 135)
    candidates = []
    for raw_x, raw_y, raw_radius in circles[0]:
        x = float(raw_x)
        y = float(raw_y + roi_y)
        radius = float(raw_radius)

        # Stage 6 restricts this to the lower half, which is what removes the
        # table/chair castors behind the reported false lock.
        if y < height * min_y_ratio:
            continue
        if x + radius < 1 or x - radius >= width - 1:
            continue

        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        inner = dist2 <= (radius * 0.84) ** 2
        pixels = int(np.count_nonzero(inner))
        expected_pixels = math.pi * (radius * 0.84) ** 2
        visible_fraction = pixels / max(1.0, expected_pixels)
        if pixels < min_pixels or visible_fraction < min_visible_fraction:
            continue

        black_fraction = float(np.count_nonzero(
            inner & (hsv[:, :, 2] < 105))) / pixels
        white_fraction = float(np.count_nonzero(
            inner & (hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 115))) / pixels
        if not (0.065 <= black_fraction <= 0.58 and white_fraction >= 0.24):
            continue
        if black_fraction + white_fraction < 0.50:
            continue

        dark_panels, largest_dark_fraction, panel_centroid_offset = _dark_panel_count(
            hsv, inner, x, y, radius)
        if min_dark_panels > 0:
            # A far-field ball may not resolve two separate panels at all, so
            # this whole check is opt-in rather than unconditional.
            if dark_panels < min_dark_panels:
                continue
            if largest_dark_fraction > 0.30:
                continue
            if panel_centroid_offset > 0.38:
                continue

        ring = ((dist2 >= (radius * 0.76) ** 2) &
                (dist2 <= (radius * 1.14) ** 2))
        ring_pixels = max(1, int(np.count_nonzero(ring)))
        edge_fraction = float(np.count_nonzero(ring & (edges > 0))) / ring_pixels
        if edge_fraction < 0.035:
            continue

        depth_m = -1.0
        diameter_m = -1.0
        if depth_at is not None:
            # A black panel may have a missing depth pixel.  Sample several
            # points on the candidate ball surface, but never accept an RGB-
            # only candidate when RealSense was explicitly supplied.
            depth_samples = []
            for sx, sy in (
                    (x, y),
                    (x - radius * 0.24, y),
                    (x + radius * 0.24, y),
                    (x, y - radius * 0.24)):
                value = float(depth_at(sx, sy))
                if value > 0.0 and np.isfinite(value):
                    depth_samples.append(value)
            if not depth_samples:
                continue
            depth_m = float(np.median(depth_samples))
            if max_depth_m is not None and depth_m > max_depth_m:
                continue
            diameter_m = 2.0 * radius * depth_m / float(focal_px)
            if not (0.10 <= diameter_m <= max_diameter_m):
                continue

        radius_score = min(1.6, radius / max(1.0, height * 0.16))
        panel_score = min(1.0, dark_panels / 4.0)
        score = (
            2.8 * radius_score +
            1.2 * panel_score +
            1.0 * black_fraction +
            0.7 * white_fraction +
            1.2 * edge_fraction +
            0.6 * (y / height)
        )
        if depth_m > 0.0:
            score += 0.35 / max(depth_m, 0.20)
        candidates.append({
            'x': x,
            'y': y,
            'radius': radius,
            'depth_m': depth_m,
            'diameter_m': diameter_m,
            'black_fraction': black_fraction,
            'white_fraction': white_fraction,
            'edge_fraction': edge_fraction,
            'dark_panels': int(dark_panels),
            'panel_centroid_offset': float(panel_centroid_offset),
            'visible_fraction': float(visible_fraction),
            'score': float(score),
        })

    if not candidates:
        return None
    return max(candidates, key=lambda item: item['score'])
