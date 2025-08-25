# models/ac_estimator.py

import cv2
import numpy as np
from math import pi, sqrt

def fit_ellipse_ac_mm_pred(mask_path, image_path=None, show_plot=False, draw_func=None,
                           spacing=0.28, scale_factor=1, epsilon=0.01):
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_gray = np.where(mask_gray == 255, 1, mask_gray).astype(np.uint8)
    _, binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(binary, 50, 150)
    h, w = binary.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                             minLineLength=int(min(h, w)*0.2), maxLineGap=10)
    arc_mask = edges.copy()
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            if min(x1, x2, y1, y2) < 10 or x2 > w-10 or y2 > h-10:
                continue
            cv2.line(arc_mask, (x1, y1), (x2, y2), 0, 1)
    arc_mask = cv2.morphologyEx(arc_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)
    arc_mask = cv2.dilate(arc_mask, np.ones((3,3),np.uint8), iterations=1)

    contours, _ = cv2.findContours(arc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    eps = epsilon * cv2.arcLength(cnt, True)
    pts = cv2.approxPolyDP(cnt, eps, True)
    pts = np.vstack(pts).reshape(-1,2)
    if len(pts) < 5:
        return 0.0

    ellipse = cv2.fitEllipse(pts.astype(np.int32))
    (xc, yc), (maj, min_d), ang = ellipse
    a, b = maj/2, min_d/2
    h_val = ((a - b)**2) / ((a + b)**2)
    per = pi*(a+b)*(1 + (3*h_val)/(10+sqrt(4-3*h_val)))
    ac_mm = per * spacing * scale_factor

    if show_plot and image_path and draw_func:
        raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        mask_col = np.zeros_like(rgb)
        mask_col[binary > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(rgb, 1.0, mask_col, 0.4, 0)
        cv2.ellipse(overlay, (int(xc), int(yc)), (int(a), int(b)), ang, 0, 360, (0,0,255), 2)
        draw_func(overlay)

    return ac_mm
