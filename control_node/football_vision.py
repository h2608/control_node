"""Black-and-white size-4 football detector for the physical CyberDog RGB camera."""

import cv2
import numpy as np


def detect_football(image, depth_at=None, focal_px=400.0):
    """Return the best football candidate, or ``None``.

    The competition ball is a 20 cm black-and-white size-4 football.  A valid
    candidate must therefore be circular, contain both dark and low-saturation
    bright regions, lie on the floor side of the image, and (when depth is
    available) have a physically plausible apparent diameter.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.8)

    roi_y = int(height * 0.28)
    roi_gray = gray[roi_y:, :]
    min_radius = max(7, int(min(width, height) * 0.015))
    max_radius = max(min_radius + 2, int(min(width, height) * 0.20))
    circles = cv2.HoughCircles(
        roi_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(35, int(width * 0.07)),
        param1=110,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    yy, xx = np.ogrid[:height, :width]
    edges = cv2.Canny(gray, 60, 140)
    candidates = []
    for raw_x, raw_y, raw_radius in circles[0]:
        x = float(raw_x)
        y = float(raw_y + roi_y)
        radius = float(raw_radius)
        if y < height * 0.34 or y + radius > height * 0.96:
            continue
        if x - radius < 2 or x + radius >= width - 2:
            continue

        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        inner = dist2 <= (radius * 0.82) ** 2
        pixels = int(np.count_nonzero(inner))
        if pixels < 80:
            continue

        black_fraction = float(np.count_nonzero(
            inner & (hsv[:, :, 2] < 105))) / pixels
        white_fraction = float(np.count_nonzero(
            inner & (hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 110))) / pixels
        if not (0.08 <= black_fraction <= 0.58 and white_fraction >= 0.22):
            continue
        if black_fraction + white_fraction < 0.48:
            continue

        ring = ((dist2 >= (radius * 0.76) ** 2) &
                (dist2 <= (radius * 1.16) ** 2))
        ring_pixels = max(1, int(np.count_nonzero(ring)))
        edge_fraction = float(np.count_nonzero(ring & (edges > 0))) / ring_pixels
        if edge_fraction < 0.045:
            continue

        depth_m = -1.0
        diameter_m = -1.0
        if depth_at is not None:
            depth_m = float(depth_at(x, y))
            if depth_m > 0.0:
                diameter_m = 2.0 * radius * depth_m / float(focal_px)
                if not (0.10 <= diameter_m <= 0.34):
                    continue

        score = (2.0 * black_fraction + white_fraction +
                 2.0 * edge_fraction + 0.25 * (y / height))
        candidates.append({
            'x': x,
            'y': y,
            'radius': radius,
            'depth_m': depth_m,
            'diameter_m': diameter_m,
            'black_fraction': black_fraction,
            'white_fraction': white_fraction,
            'edge_fraction': edge_fraction,
            'score': score,
        })

    if not candidates:
        return None
    return max(candidates, key=lambda item: item['score'])
