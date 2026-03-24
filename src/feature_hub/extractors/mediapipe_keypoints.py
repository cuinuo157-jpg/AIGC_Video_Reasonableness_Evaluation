from __future__ import annotations

import cv2
import numpy as np


_holistic = None


def _load_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """加载视频帧并返回帧列表和 fps。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _get_holistic():
    """懒加载 MediaPipe Holistic 模型单例（自包含，不依赖历史模块（已迁移））。"""
    global _holistic
    if _holistic is None:
        import mediapipe as mp

        _holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
        )
    return _holistic


def _extract_keypoints_from_frame(frame_rgb: np.ndarray) -> dict:
    """使用 Holistic 模型从单帧提取关键点。"""
    holistic = _get_holistic()
    results = holistic.process(frame_rgb)

    kp: dict = {"body": None, "left_hand": None, "right_hand": None, "face": None}

    if results.pose_landmarks:
        kp["body"] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        )
    if results.left_hand_landmarks:
        kp["left_hand"] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        )
    if results.right_hand_landmarks:
        kp["right_hand"] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        )
    if results.face_landmarks:
        kp["face"] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
        )
    return kp


def _compute_ear(face_landmarks: np.ndarray, eye_indices: dict) -> float | None:
    """计算单眼 EAR (Eye Aspect Ratio)。"""
    try:
        v1_top, v1_bottom = eye_indices["vertical1"]
        v2_top, v2_bottom = eye_indices["vertical2"]
        h_left, h_right = eye_indices["horizontal"]
        vertical1 = np.linalg.norm(face_landmarks[v1_top] - face_landmarks[v1_bottom])
        vertical2 = np.linalg.norm(face_landmarks[v2_top] - face_landmarks[v2_bottom])
        horizontal = np.linalg.norm(face_landmarks[h_left] - face_landmarks[h_right])
        if horizontal > 0:
            return float((vertical1 + vertical2) / (2.0 * horizontal))
        return 0.0
    except Exception:
        return None


def _compute_mar(face_landmarks: np.ndarray, mouth_indices: dict) -> float | None:
    """计算 MAR (Mouth Aspect Ratio)。"""
    try:
        top = mouth_indices["top"]
        bottom = mouth_indices["bottom"]
        left = mouth_indices["left"]
        right = mouth_indices["right"]
        vertical = np.linalg.norm(face_landmarks[top] - face_landmarks[bottom])
        horizontal = np.linalg.norm(face_landmarks[left] - face_landmarks[right])
        if horizontal > 0:
            return float(vertical / horizontal)
        return 0.0
    except Exception:
        return None


# MediaPipe 468-point face landmark 索引
_LEFT_EYE = {"vertical1": (159, 145), "vertical2": (158, 144), "horizontal": (33, 133)}
_RIGHT_EYE = {"vertical1": (386, 374), "vertical2": (385, 373), "horizontal": (362, 263)}
_MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}


def extract_mediapipe_keypoints(video_path: str, device: str) -> list[dict]:
    """提取 MediaPipe Holistic 关键点并预计算生理指标。

    返回每帧一个 dict，包含 body/left_hand/right_hand/face 数组
    以及预计算的 ear_left/ear_right/ear_avg/mar 标量。
    """
    frames, fps = _load_frames(video_path)

    results: list[dict] = []
    for idx, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp = _extract_keypoints_from_frame(rgb)

        ear_left: float | None = None
        ear_right: float | None = None
        ear_avg: float | None = None
        mar: float | None = None

        if kp.get("face") is not None:
            ear_left = _compute_ear(kp["face"], _LEFT_EYE)
            ear_right = _compute_ear(kp["face"], _RIGHT_EYE)
            if ear_left is not None and ear_right is not None:
                ear_avg = (ear_left + ear_right) / 2.0
            else:
                ear_avg = ear_left or ear_right
            mar = _compute_mar(kp["face"], _MOUTH)

        results.append(
            {
                "body": kp.get("body"),
                "left_hand": kp.get("left_hand"),
                "right_hand": kp.get("right_hand"),
                "face": kp.get("face"),
                "ear_left": ear_left,
                "ear_right": ear_right,
                "ear_avg": ear_avg,
                "mar": mar,
                "frame_idx": idx,
            }
        )

    return results
