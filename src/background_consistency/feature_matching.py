from __future__ import annotations

import cv2
import numpy as np


def compute_homography_stability(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 1.0

    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    stabilities = []

    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            stabilities.append(0.5)
            continue

        matches = bf.match(des1, des2)
        if len(matches) < 4:
            stabilities.append(0.5)
            continue

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            stabilities.append(0.5)
            continue

        deviation = float(np.mean(np.abs(H - np.eye(3))))
        stabilities.append(float(np.clip(1.0 - deviation, 0, 1)))

    return float(np.mean(stabilities)) if stabilities else 1.0
