"""Black-and-white size-4 football detector for the physical CyberDog RGB camera.

Stage 4 acquires the ball from several metres away; Stage 6 only ever sees it
during the final close push, where chair castors are the dominant false
positive.  The shape/panel checks are shared, but the range-dependent gates
differ, so they are keyword arguments: the defaults are the far-field
Stage-4 behaviour, and Stage 6 passes ``**NEAR_BALL``.

Hard-coding the Stage-6 values is what made this file undeployable on its
own -- min radius 38 px, lower image half only and a 1.00 m depth cap make
Stage 4 blind past about a metre.

The Stage-4 defaults were measured on the spare robot (namespace
``..._5f_b4_ff``) with the ball static on the competition floor.  The old
values found the ball on 55-95% of frames inside 2 m and, on the frames they
missed, reported a table leg instead: with a 9x9/sigma-1.8 pre-blur and
``hough_param2=40`` a 2.4 m ball (radius ~24 px) is not merely rejected by a
gate -- ``cv2.HoughCircles`` never emits it as a candidate at all, and the one
circle it does emit is background clutter.  Paired against the old values on
471 identical frames at 1.9 m, the current defaults detect 471 vs 450, and
every one of those 21 recovered frames is a correct lock.

Stage 6 is deliberately left on the old behaviour: ``NEAR_BALL`` pins the
pre-blur and ``score_mode`` it was tuned against, because its castor
rejection has not been re-measured on hardware.
"""

import math

import cv2
import numpy as np


# Measured black/white mix of the ball surface, spare robot, ball at
# 1.0-2.3 m, n=257 static frames: black 0.24-0.39, white 0.43-0.81.
BALL_BLACK_MEAN, BALL_BLACK_SPREAD = 0.31, 0.13
BALL_WHITE_MEAN, BALL_WHITE_SPREAD = 0.67, 0.18
# Size-4 football.
BALL_DIAMETER_M = 0.205


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
    # Stage 6's castor rejection was tuned against the pre-blur and the score
    # below; both are held at their pre-2026-08-18 values until Stage 6 gets
    # its own hardware bench.
    'blur_ksize': 9,
    'blur_sigma': 1.8,
    'score_mode': 'legacy',
}


def _bell(value, mean, spread):
    """Unit-height Gaussian preference, 1.0 at ``mean``."""
    return math.exp(-((value - mean) / spread) ** 2)


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


def detect_football(image, depth_at=None, focal_px=450.0, *,
                    min_radius_px=7, min_radius_ratio=0.015,
                    max_radius_ratio=0.20, roi_top_ratio=0.28,
                    min_y_ratio=0.34, min_pixels=80,
                    min_visible_fraction=0.0, min_dark_panels=0,
                    max_depth_m=None, max_diameter_m=0.34,
                    hough_param2=30, blur_ksize=5, blur_sigma=1.2,
                    score_mode='evidence', prefer_center=None,
                    continuity_px=70.0):
    """Return the best close black-and-white football candidate, or ``None``.

    Final Stage 6 pushing sees a large ball near the lower image boundary.
    Chair castors are also circular and black, so circularity alone is not
    sufficient: a candidate must contain a bright ball surface plus at least
    two separate dark football panels.  A close ball may be partly clipped by
    the bottom edge and remains valid as long as most of the circle is visible.

    ``score_mode`` selects how the surviving candidates are ranked:

    ``evidence`` (default)
        Rank on scale-invariant ball evidence only -- panel count, the
        measured black/white mix, ring edge support, agreement with the
        physical ball diameter, and proximity to ``prefer_center``.
    ``legacy``
        The original ``2.8 * radius_score + ... + 0.35 / depth``.  This
        rewards "big and near", which is what a false positive has: a bright
        floor blob measured at 0.29 m scored 5.87 against the real ball's
        4.00 and won.  Kept for Stage 6, which was tuned against it.

    ``prefer_center`` is the previous frame's accepted centre.  Passing it
    biases selection toward a continuous track; it never creates a candidate
    that the gates rejected, so a genuinely moved ball is still acquired.
    """
    if image is None or image.ndim < 3:
        return None
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A 2 m ball is only ~22 px across and the old 9x9/sigma-1.8 kernel is a
    # sizeable fraction of that, erasing the gradient ring HoughCircles needs.
    gray = cv2.GaussianBlur(gray_raw, (blur_ksize, blur_ksize), blur_sigma)

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

        # Everything below is evaluated in the candidate's own window.  A
        # full-frame distance grid per circle costs width*height each time,
        # and the lower hough_param2 that finds the 2 m ball also multiplies
        # the candidate count -- together that was a 5x slowdown.
        pad = int(math.ceil(radius * 1.15)) + 2
        wx0, wx1 = max(0, int(x) - pad), min(width, int(x) + pad + 1)
        wy0, wy1 = max(0, int(y) - pad), min(height, int(y) + pad + 1)
        if wx1 <= wx0 or wy1 <= wy0:
            continue
        yy, xx = np.ogrid[wy0:wy1, wx0:wx1]
        hsv_window = hsv[wy0:wy1, wx0:wx1]
        edges_window = edges[wy0:wy1, wx0:wx1]

        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        inner = dist2 <= (radius * 0.84) ** 2
        pixels = int(np.count_nonzero(inner))
        expected_pixels = math.pi * (radius * 0.84) ** 2
        visible_fraction = pixels / max(1.0, expected_pixels)
        if pixels < min_pixels or visible_fraction < min_visible_fraction:
            continue

        black_fraction = float(np.count_nonzero(
            inner & (hsv_window[:, :, 2] < 105))) / pixels
        white_fraction = float(np.count_nonzero(
            inner & (hsv_window[:, :, 1] < 105) &
            (hsv_window[:, :, 2] > 115))) / pixels
        if not (0.065 <= black_fraction <= 0.58 and white_fraction >= 0.24):
            continue
        if black_fraction + white_fraction < 0.50:
            continue

        # _dark_panel_count slices the frame it is given, so hand it the
        # window and window-local coordinates; the geometry is unchanged.
        dark_panels, largest_dark_fraction, panel_centroid_offset = _dark_panel_count(
            hsv_window, inner, x - wx0, y - wy0, radius)
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
        edge_fraction = float(
            np.count_nonzero(ring & (edges_window > 0))) / ring_pixels
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

        if score_mode == 'legacy':
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
        else:
            # Nothing here rewards a big or a near candidate: those are the
            # two properties a false positive reliably has, and the ball's
            # own evidence -- panels, colour mix, edge ring, physical size --
            # does not change with range.
            panel_term = 1.0 if 2 <= dark_panels <= 6 else (
                0.45 if dark_panels else 0.15)
            colour_term = (
                _bell(black_fraction, BALL_BLACK_MEAN, BALL_BLACK_SPREAD) *
                _bell(white_fraction, BALL_WHITE_MEAN, BALL_WHITE_SPREAD))
            edge_term = min(1.0, edge_fraction / 0.09)
            size_term = (_bell(diameter_m, BALL_DIAMETER_M, 0.06)
                         if diameter_m > 0.0 else 0.5)
            continuity_term = 0.0
            if prefer_center is not None:
                continuity_term = _bell(
                    math.hypot(x - prefer_center[0], y - prefer_center[1]),
                    0.0, max(continuity_px, 1.0))
            score = (
                1.5 * panel_term +
                1.4 * colour_term +
                1.0 * edge_term +
                1.3 * size_term +
                1.0 * continuity_term
            )
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
