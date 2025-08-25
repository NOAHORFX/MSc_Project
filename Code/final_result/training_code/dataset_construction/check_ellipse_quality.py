import cv2
import numpy as np
from math import pi, sqrt
import json

def check_ellipse_quality(mask_path, gt_ac_mm, tolerance=0.01, spacing=0.28):
    # 读取并二值化 mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.vstack(max(contours, key=cv2.contourArea)).reshape(-1, 2)

    ellipse = cv2.fitEllipse(points.astype(np.int32))
    (xc, yc), (maj_d, min_d), angle_deg = ellipse
    a, b = maj_d / 2.0, min_d / 2.0

    # Ramanujan公式计算周长（单位：像素）
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter_px = pi * (a + b) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)))
    ac_cv_mm = perimeter_px * spacing

    # 比较误差
    error_ratio = abs(ac_cv_mm - gt_ac_mm) / gt_ac_mm

    is_valid = error_ratio <= tolerance
    return is_valid, ac_cv_mm, error_ratio
