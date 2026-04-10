"""
生物特征异常 (Biological Anomaly) 三级检测调试脚本

用法:
    python scripts/debug_bio_anomaly.py --input <视频路径>
    python scripts/debug_bio_anomaly.py --input <视频路径> --device cpu
    python scripts/debug_bio_anomaly.py --input <视频路径> --sample-rate 3 --save-vis

参数:
    --input        视频文件路径
    --device       推理设备 (cuda / cpu)，默认 cpu
    --sample-rate  每 N 帧采样一帧，默认 1（全帧）
    --save-vis     保存可视化结果到 outputs/bio_anomaly/
    --no-mllm      禁用 Level 3 MLLM 判定

Level 3 MLLM 默认 vllm（与 debug_dynamics / test_qwen_35_video 一致）。
环境变量: VLLM_OPENAI_BASE_URL、VLLM_API_KEY；dashscope 时需 DASHSCOPE_API_KEY。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.biological_anomaly.config import BiologicalAnomalyConfig
from src.biological_anomaly.eye_anomaly import (
    detect_eye_anomalies,
    detect_eye_symmetry_anomalies,
)
from src.biological_anomaly.hand_anomaly import (
    detect_finger_count_anomalies,
    detect_hand_anomalies_l1,
    detect_hand_structure_anomalies_l2,
)
from src.biological_anomaly.mouth_anomaly import (
    detect_mouth_anomalies_l1,
    detect_mouth_color_anomalies,
    detect_mouth_temporal_anomalies_l2,
)
from src.biological_anomaly.body_anomaly import (
    compute_body_part_scores,
    detect_body_angle_anomalies,
    detect_body_bone_anomalies,
)
from src.biological_anomaly.mllm_bio_judge import judge_biological_anomaly_mllm
from src.biological_anomaly.analyzer import _collect_suspicious
from src.mllm.client import MLLMClient
from src.mllm.config import MLLMConfig


# ── 1. 视频读取 ──────────────────────────────────────────


def _load_repo_dotenv(repo_root: Path = ROOT) -> None:
    """从仓库根 .env 注入环境变量（不覆盖已存在项）。"""
    path = repo_root / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def load_frames(video_path: str, sample_rate: int = 1) -> tuple[list[np.ndarray], float]:
    """读取视频帧，返回 (帧列表, fps)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {w}x{h}, {fps:.1f}fps, {total} 帧")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1
    cap.release()

    print(f"采样: 每 {sample_rate} 帧取 1 帧, 共 {len(frames)} 帧")
    return frames, fps


# ── 2. MediaPipe 关键点提取（直接调用，不经过历史模块（已迁移）） ──


# MediaPipe 468-point face landmark 索引
_LEFT_EYE = {"vertical1": (159, 145), "vertical2": (158, 144), "horizontal": (33, 133)}
_RIGHT_EYE = {"vertical1": (386, 374), "vertical2": (385, 373), "horizontal": (362, 263)}
_MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}

_mp_holistic = None


def _get_holistic():
    """懒加载 MediaPipe Holistic 模型。"""
    global _mp_holistic
    if _mp_holistic is None:
        import mediapipe as mp
        _mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _mp_holistic


def _landmarks_to_array(landmarks, count: int) -> np.ndarray:
    """将 MediaPipe landmark 列表转为 (N, 3) ndarray。"""
    arr = np.zeros((count, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        if i >= count:
            break
        arr[i] = [lm.x, lm.y, lm.z]
    return arr


def _compute_ear(face: np.ndarray, eye_indices: dict) -> float | None:
    """计算单眼 EAR (Eye Aspect Ratio)。"""
    try:
        v1_top, v1_bottom = eye_indices["vertical1"]
        v2_top, v2_bottom = eye_indices["vertical2"]
        h_left, h_right = eye_indices["horizontal"]
        vertical1 = np.linalg.norm(face[v1_top] - face[v1_bottom])
        vertical2 = np.linalg.norm(face[v2_top] - face[v2_bottom])
        horizontal = np.linalg.norm(face[h_left] - face[h_right])
        if horizontal > 0:
            return float((vertical1 + vertical2) / (2.0 * horizontal))
        return 0.0
    except Exception:
        return None


def _compute_mar(face: np.ndarray, mouth_indices: dict) -> float | None:
    """计算 MAR (Mouth Aspect Ratio)。"""
    try:
        vertical = np.linalg.norm(face[mouth_indices["top"]] - face[mouth_indices["bottom"]])
        horizontal = np.linalg.norm(face[mouth_indices["left"]] - face[mouth_indices["right"]])
        if horizontal > 0:
            return float(vertical / horizontal)
        return 0.0
    except Exception:
        return None


def extract_keypoints(frames: list[np.ndarray], fps: float) -> list[dict]:
    """使用 MediaPipe Holistic 提取关键点并预计算生理指标。"""
    holistic = _get_holistic()

    print(f"\n逐帧提取 MediaPipe 关键点...")
    results: list[dict] = []
    t0 = time.time()

    for idx, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = holistic.process(rgb)

        body = face = left_hand = right_hand = None
        if mp_result.pose_landmarks:
            body = _landmarks_to_array(mp_result.pose_landmarks, 33)
        if mp_result.face_landmarks:
            face = _landmarks_to_array(mp_result.face_landmarks, 468)
        if mp_result.left_hand_landmarks:
            left_hand = _landmarks_to_array(mp_result.left_hand_landmarks, 21)
        if mp_result.right_hand_landmarks:
            right_hand = _landmarks_to_array(mp_result.right_hand_landmarks, 21)

        ear_left = ear_right = ear_avg = mar = None
        if face is not None:
            ear_left = _compute_ear(face, _LEFT_EYE)
            ear_right = _compute_ear(face, _RIGHT_EYE)
            if ear_left is not None and ear_right is not None:
                ear_avg = (ear_left + ear_right) / 2.0
            else:
                ear_avg = ear_left or ear_right
            mar = _compute_mar(face, _MOUTH)

        results.append({
            "body": body,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "face": face,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "ear_avg": ear_avg,
            "mar": mar,
            "frame_idx": idx,
        })

        if (idx + 1) % 20 == 0 or idx == len(frames) - 1:
            parts = []
            if face is not None:
                parts.append("face")
            if left_hand is not None:
                parts.append("L-hand")
            if right_hand is not None:
                parts.append("R-hand")
            if body is not None:
                parts.append("body")
            tag = ", ".join(parts) if parts else "无检测"
            print(f"  [{idx+1}/{len(frames)}] {tag}")

    elapsed = time.time() - t0
    print(f"关键点提取完成 ({elapsed:.1f}s, {elapsed/len(frames):.2f}s/帧)")

    # 统计
    face_count = sum(1 for kp in results if kp["face"] is not None)
    lh_count = sum(1 for kp in results if kp["left_hand"] is not None)
    rh_count = sum(1 for kp in results if kp["right_hand"] is not None)
    print(f"  面部检出: {face_count}/{len(frames)} ({face_count/len(frames)*100:.1f}%)")
    print(f"  左手检出: {lh_count}/{len(frames)} ({lh_count/len(frames)*100:.1f}%)")
    print(f"  右手检出: {rh_count}/{len(frames)} ({rh_count/len(frames)*100:.1f}%)")

    return results


# ── 3. 三级检测 ───────────────────────────────────────────


def run_level1(keypoints_seq: list[dict], fps: float, config: BiologicalAnomalyConfig) -> dict:
    """Level 1: 快速筛选。"""
    sw = config.smoothing_window
    print(f"\n{'='*60}")
    print(f"Level 1: 快速筛选 (EAR/MAR/手部速度/全身骨骼)")
    print(f"{'='*60}")

    ear_avg = [kp.get("ear_avg") for kp in keypoints_seq]
    ear_left = [kp.get("ear_left") for kp in keypoints_seq]
    ear_right = [kp.get("ear_right") for kp in keypoints_seq]
    mar_seq = [kp.get("mar") for kp in keypoints_seq]

    # 眼部
    eye_anomalies = detect_eye_anomalies(ear_avg, fps=fps, smoothing_window=sw)
    eye_sym = detect_eye_symmetry_anomalies(
        ear_left, ear_right, fps=fps,
        tolerance=config.eye_symmetry_tolerance,
        smoothing_window=sw,
    )
    print(f"\n  [眼部] 长时间未眨眼: {len(eye_anomalies)} 处")
    for a in eye_anomalies[:5]:
        print(f"    帧 {a['frame_idx']}: {a['description']}")
    print(f"  [眼部] 左右不对称: {len(eye_sym)} 处")

    # 嘴部
    mouth_anomalies = detect_mouth_anomalies_l1(mar_seq, fps=fps, smoothing_window=sw)
    discontinuities = [a for a in mouth_anomalies if a["type"] == "mouth_discontinuity"]
    prolonged = [a for a in mouth_anomalies if a["type"] == "prolonged_mouth_opening"]
    print(f"\n  [嘴部] MAR 跳跃: {len(discontinuities)} 处")
    for a in discontinuities[:5]:
        print(f"    帧 {a['frame_idx']}: {a['description']}")
    print(f"  [嘴部] 持续张嘴: {len(prolonged)} 处")

    # 手部
    hand_anomalies = detect_hand_anomalies_l1(keypoints_seq, fps=fps, smoothing_window=sw)
    by_type: dict[str, int] = {}
    for a in hand_anomalies:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    print(f"\n  [手部] 运动异常: {len(hand_anomalies)} 处")
    for t, c in sorted(by_type.items()):
        print(f"    {t}: {c} 处")

    # 全身骨骼（借鉴 VMBench OIS）
    body_seq = [kp.get("body") for kp in keypoints_seq]
    body_bone = detect_body_bone_anomalies(body_seq, smoothing_window=sw)
    body_angle = detect_body_angle_anomalies(body_seq, smoothing_window=sw)
    print(f"\n  [全身骨骼] 骨段长度突变: {len(body_bone)} 处")
    for a in body_bone[:5]:
        print(f"    帧 {a['frame_idx']}: {a['description']}")
    print(f"  [全身骨骼] 关节角度突变: {len(body_angle)} 处")
    for a in body_angle[:5]:
        print(f"    帧 {a['frame_idx']}: {a['description']}")

    all_l1 = eye_anomalies + eye_sym + mouth_anomalies + hand_anomalies + body_bone + body_angle
    print(f"\n  Level 1 总异常: {len(all_l1)} 处")

    return {
        "eye": eye_anomalies,
        "eye_sym": eye_sym,
        "mouth": mouth_anomalies,
        "hand": hand_anomalies,
        "body_bone": body_bone,
        "body_angle": body_angle,
        "all": all_l1,
    }


def run_level2(keypoints_seq: list[dict], fps: float, frames: list[np.ndarray] | None = None) -> dict:
    """Level 2: 结构检测。"""
    print(f"\n{'='*60}")
    print(f"Level 2: 结构检测 (手指融合/计数/骨骼突变/嘴部时序/颜色)")
    print(f"{'='*60}")

    # 嘴部 landmark 时序
    face_seq = [kp.get("face") for kp in keypoints_seq]
    mouth_anomalies = detect_mouth_temporal_anomalies_l2(face_seq, fps=fps)
    by_type: dict[str, int] = {}
    for a in mouth_anomalies:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    print(f"\n  [嘴部结构] 异常: {len(mouth_anomalies)} 处")
    for t, c in sorted(by_type.items()):
        print(f"    {t}: {c} 处")
    for a in mouth_anomalies[:5]:
        print(f"    帧 {a['frame_idx']}: {a['description']}")

    # 嘴内颜色直方图突变
    mouth_color_anomalies: list[dict] = []
    if frames is not None:
        mouth_color_anomalies = detect_mouth_color_anomalies(frames, face_seq)
        print(f"\n  [嘴内颜色] 直方图突变: {len(mouth_color_anomalies)} 处")
        for a in mouth_color_anomalies[:5]:
            print(f"    帧 {a['frame_idx']}: corr={a['correlation']:.3f} - {a['description']}")

    # 手部结构
    hand_anomalies: list[dict] = []
    finger_count_anomalies: list[dict] = []
    for hand_key in ("left_hand", "right_hand"):
        hand_seq = [kp.get(hand_key) for kp in keypoints_seq]
        if any(h is not None for h in hand_seq):
            anomalies = detect_hand_structure_anomalies_l2(hand_seq, hand_key=hand_key)
            hand_anomalies.extend(anomalies)
            # 手指计数
            fc_anomalies = detect_finger_count_anomalies(hand_seq, hand_key=hand_key)
            finger_count_anomalies.extend(fc_anomalies)

    by_type = {}
    for a in hand_anomalies:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    print(f"\n  [手部结构] 异常: {len(hand_anomalies)} 处")
    for t, c in sorted(by_type.items()):
        print(f"    {t}: {c} 处")

    # 手指计数
    print(f"  [手指计数] 异常: {len(finger_count_anomalies)} 处")
    for a in finger_count_anomalies[:5]:
        print(f"    帧 {a['frame_idx']}: {a['description']}")

    # 重点展示手指融合
    fusions = [a for a in hand_anomalies if a["type"] == "finger_fusion"]
    if fusions:
        print(f"\n  !! 检测到手指融合 {len(fusions)} 处:")
        for a in fusions[:10]:
            print(f"    帧 {a['frame_idx']}: {a['description']}")

    all_l2 = mouth_anomalies + mouth_color_anomalies + hand_anomalies + finger_count_anomalies
    print(f"\n  Level 2 总异常: {len(all_l2)} 处")

    return {
        "mouth": mouth_anomalies,
        "mouth_color": mouth_color_anomalies,
        "hand": hand_anomalies,
        "finger_count": finger_count_anomalies,
        "all": all_l2,
    }


# ── 4. 综合评分 ───────────────────────────────────────────


def compute_score(
    l1: dict,
    l2: dict,
    config: BiologicalAnomalyConfig,
    n_frames: int,
    l3_score: float = 1.0,
) -> dict:
    """计算各级和综合得分。"""
    from src.biological_anomaly.analyzer import _score_from_anomalies

    l1_score = _score_from_anomalies(l1["all"], n_frames)
    l2_score = _score_from_anomalies(l2["all"], n_frames)
    bio_score = (
        config.level1_weight * l1_score
        + config.level2_weight * l2_score
        + config.level3_weight * l3_score
    )
    bio_score = float(np.clip(bio_score, 0, 1))

    return {
        "level1_score": round(l1_score, 4),
        "level2_score": round(l2_score, 4),
        "level3_score": round(l3_score, 4),
        "bio_quality_score": round(bio_score, 4),
    }


def build_mllm_client(args: argparse.Namespace, enable_mllm: bool) -> MLLMClient | None:
    if not enable_mllm:
        return None
    api_key = (args.mllm_api_key or "").strip()
    if args.mllm_provider != "vllm" and not api_key:
        raise ValueError(
            "启用 MLLM 且非 vllm 时必须提供 API Key（DASHSCOPE_API_KEY 或 --mllm-api-key）"
        )
    cfg = MLLMConfig(
        backend="api",
        api_provider=args.mllm_provider,
        api_model=args.mllm_model,
        api_key=api_key or None,
        api_base_url=(args.mllm_base_url or "").strip() or None,
        dashscope_video_fps=args.mllm_fps,
    )
    return MLLMClient(cfg)


def run_level3(
    frames: list[np.ndarray],
    keypoints_seq: list[dict],
    l1: dict,
    l2: dict,
    config: BiologicalAnomalyConfig,
    mllm_client: MLLMClient | None,
) -> dict:
    if not config.enable_mllm or mllm_client is None:
        return {"skipped": True, "anomalies": [], "level3_score": 1.0}

    suspicious = _collect_suspicious(l1["all"] + l2["all"])
    if not suspicious:
        return {"skipped": True, "anomalies": [], "level3_score": 1.0}

    result = judge_biological_anomaly_mllm(
        frames,
        keypoints_seq,
        suspicious,
        mllm_client,
        max_crops=config.mllm_max_crops,
    )
    if result.get("skipped"):
        return {"skipped": True, "anomalies": [], "level3_score": 1.0}

    anomalies = result.get("anomalies", [])
    level3_score = 0.3 if result.get("has_anomalies") else 1.0
    return {
        "skipped": False,
        "anomalies": anomalies,
        "level3_score": level3_score,
        "raw_result": result,
    }


# ── 5. 可视化 ─────────────────────────────────────────────

# ---------- 可视化语义配色方案 ----------
# 每种可视化元素有独立颜色和语义含义

# 面部网格 (Face Mesh) — 青色半透明线框，表征面部 468 点拓扑结构
_VIS_FACE_MESH_COLOR = (200, 180, 0)       # 青色 (BGR)
_VIS_FACE_MESH_ALPHA = 0.3

# 眼部 ROI 轮廓 — 绿色，表征眨眼检测区域 (EAR 计算范围)
_VIS_EYE_COLOR = (0, 220, 0)               # 绿色

# 嘴部 ROI 轮廓 — 品红色，表征嘴部开合检测区域 (MAR 计算范围)
_VIS_MOUTH_COLOR = (200, 0, 200)            # 品红

# 嘴内轮廓 — 橙色填充，表征舌头/牙齿消失检测的内唇多边形
_VIS_INNER_LIP_COLOR = (0, 140, 255)        # 橙色

# 手部骨骼 — 左手绿/右手蓝，表征手指融合 & 骨段长度检测
_VIS_HAND_LEFT_COLOR = (0, 255, 100)        # 青绿
_VIS_HAND_RIGHT_COLOR = (255, 100, 0)       # 蓝橙

# 指尖高亮 — 黄色，表征手指融合距离检测点
_VIS_FINGERTIP_COLOR = (0, 255, 255)        # 黄色

# 异常标记 — 红色，表征检测到的异常区域
_VIS_ANOMALY_COLOR = (0, 0, 255)            # 红色

# 正常状态 — 顶栏绿色
_VIS_NORMAL_COLOR = (0, 200, 0)             # 绿色

# MediaPipe 面部网格关键连接 (精简版，避免全连接过密)
# 面部轮廓
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]
# 左眼轮廓
_LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
# 右眼轮廓
_RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
# 外唇轮廓
_OUTER_LIP = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
]
# 内唇轮廓 (与 anomaly_rules 中一致)
_INNER_LIP = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
]

# 手部骨骼连接 (MediaPipe 21 点)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (9, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),# 小指
    (0, 17),                                # 掌根连接
]

# 异常类型 → 语义中文标签
_ANOMALY_LABEL = {
    "no_blink": "EYE: Long no-blink",
    "eye_asymmetry": "EYE: L/R asymmetry",
    "mouth_discontinuity": "MOUTH: MAR jump",
    "prolonged_mouth_opening": "MOUTH: Prolonged open",
    "mouth_area_sudden_change": "MOUTH: Area sudden change",
    "mouth_landmark_jump": "MOUTH: Landmark jump",
    "mouth_color_sudden_change": "MOUTH: Color histogram change",
    "hand_appear": "HAND: Sudden appear",
    "hand_disappear": "HAND: Sudden disappear",
    "hand_velocity_jump": "HAND: Velocity jump",
    "hand_jitter": "HAND: Jitter",
    "finger_fusion": "HAND: Finger fusion",
    "finger_count_abnormal": "HAND: Finger count abnormal",
    "bone_length_change": "HAND: Bone length change",
    "impossible_joint_angle": "HAND: Impossible angle",
    "body_bone_length_change": "BODY: Bone length change",
    "body_angle_change": "BODY: Joint angle change",
}


def _draw_face_mesh(vis: np.ndarray, face: np.ndarray) -> None:
    """绘制面部网格：轮廓 + 眼部 ROI + 嘴部 ROI + 内唇多边形。"""
    h, w = vis.shape[:2]
    overlay = vis.copy()

    def _pt(idx: int) -> tuple[int, int]:
        return int(face[idx][0] * w), int(face[idx][1] * h)

    # 面部轮廓 (青色细线)
    for i in range(len(_FACE_OVAL) - 1):
        cv2.line(overlay, _pt(_FACE_OVAL[i]), _pt(_FACE_OVAL[i + 1]),
                 _VIS_FACE_MESH_COLOR, 1, cv2.LINE_AA)

    # 左眼轮廓 (绿色) — EAR 检测区域
    pts = np.array([_pt(i) for i in _LEFT_EYE_CONTOUR], dtype=np.int32)
    cv2.polylines(overlay, [pts], True, _VIS_EYE_COLOR, 2, cv2.LINE_AA)

    # 右眼轮廓 (绿色) — EAR 检测区域
    pts = np.array([_pt(i) for i in _RIGHT_EYE_CONTOUR], dtype=np.int32)
    cv2.polylines(overlay, [pts], True, _VIS_EYE_COLOR, 2, cv2.LINE_AA)

    # 外唇轮廓 (品红) — MAR 检测区域
    pts = np.array([_pt(i) for i in _OUTER_LIP], dtype=np.int32)
    cv2.polylines(overlay, [pts], True, _VIS_MOUTH_COLOR, 2, cv2.LINE_AA)

    # 内唇多边形 (橙色半透明填充) — 舌头/牙齿消失检测区域
    inner_pts = np.array([_pt(i) for i in _INNER_LIP], dtype=np.int32)
    cv2.fillPoly(overlay, [inner_pts], _VIS_INNER_LIP_COLOR)

    # 混合叠加
    cv2.addWeighted(overlay, _VIS_FACE_MESH_ALPHA, vis, 1.0 - _VIS_FACE_MESH_ALPHA, 0, vis)

    # 在轮廓外再画一层描边 (不透明)
    for i in range(len(_FACE_OVAL) - 1):
        cv2.line(vis, _pt(_FACE_OVAL[i]), _pt(_FACE_OVAL[i + 1]),
                 _VIS_FACE_MESH_COLOR, 1, cv2.LINE_AA)
    pts_le = np.array([_pt(i) for i in _LEFT_EYE_CONTOUR], dtype=np.int32)
    pts_re = np.array([_pt(i) for i in _RIGHT_EYE_CONTOUR], dtype=np.int32)
    cv2.polylines(vis, [pts_le], True, _VIS_EYE_COLOR, 1, cv2.LINE_AA)
    cv2.polylines(vis, [pts_re], True, _VIS_EYE_COLOR, 1, cv2.LINE_AA)
    pts_ol = np.array([_pt(i) for i in _OUTER_LIP], dtype=np.int32)
    cv2.polylines(vis, [pts_ol], True, _VIS_MOUTH_COLOR, 1, cv2.LINE_AA)
    inner_pts2 = np.array([_pt(i) for i in _INNER_LIP], dtype=np.int32)
    cv2.polylines(vis, [inner_pts2], True, _VIS_INNER_LIP_COLOR, 1, cv2.LINE_AA)


def _draw_hand_skeleton(
    vis: np.ndarray, hand: np.ndarray, color: tuple[int, int, int],
) -> None:
    """绘制手部骨骼连接线 + 关节点 + 指尖高亮。"""
    h, w = vis.shape[:2]

    def _pt(idx: int) -> tuple[int, int]:
        return int(hand[idx][0] * w), int(hand[idx][1] * h)

    # 骨骼连接线
    for start, end in _HAND_CONNECTIONS:
        cv2.line(vis, _pt(start), _pt(end), color, 2, cv2.LINE_AA)

    # 关节点
    for i in range(21):
        cv2.circle(vis, _pt(i), 3, color, -1, cv2.LINE_AA)

    # 指尖高亮 (黄色大圆)
    for tip_idx in [4, 8, 12, 16, 20]:
        cv2.circle(vis, _pt(tip_idx), 6, _VIS_FINGERTIP_COLOR, 2, cv2.LINE_AA)


def _draw_legend(vis: np.ndarray, kp: dict, is_anomaly: bool) -> None:
    """在右上角绘制图例面板，说明各颜色语义。"""
    h, w = vis.shape[:2]
    legend_items: list[tuple[tuple[int, int, int], str]] = []

    face = kp.get("face")
    if face is not None:
        legend_items.append((_VIS_FACE_MESH_COLOR, "Face contour"))
        legend_items.append((_VIS_EYE_COLOR, "Eye ROI (EAR)"))
        legend_items.append((_VIS_MOUTH_COLOR, "Outer lip (MAR)"))
        legend_items.append((_VIS_INNER_LIP_COLOR, "Inner lip (tongue/teeth)"))

    if kp.get("left_hand") is not None:
        legend_items.append((_VIS_HAND_LEFT_COLOR, "Left hand skeleton"))
    if kp.get("right_hand") is not None:
        legend_items.append((_VIS_HAND_RIGHT_COLOR, "Right hand skeleton"))
    if kp.get("left_hand") is not None or kp.get("right_hand") is not None:
        legend_items.append((_VIS_FINGERTIP_COLOR, "Fingertip (fusion check)"))

    if is_anomaly:
        legend_items.append((_VIS_ANOMALY_COLOR, "ANOMALY detected"))

    if not legend_items:
        return

    line_h = 18
    pad = 8
    max_text_w = max(
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        for _, text in legend_items
    )
    panel_w = max_text_w + 30 + pad * 2
    panel_h = len(legend_items) * line_h + pad * 2

    x0 = w - panel_w - 10
    y0 = 10

    # 半透明黑底
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    for i, (color, text) in enumerate(legend_items):
        cy = y0 + pad + i * line_h + line_h // 2
        # 色块
        cv2.rectangle(vis, (x0 + pad, cy - 5), (x0 + pad + 14, cy + 5), color, -1)
        # 文字
        cv2.putText(vis, text, (x0 + pad + 20, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_info_bar(
    vis: np.ndarray, idx: int, kp: dict, anomaly_types: list[str] | None,
) -> None:
    """底部信息栏：帧号 + EAR/MAR 数值 + 异常类型。"""
    h, w = vis.shape[:2]
    bar_h = 50 if anomaly_types else 30

    # 半透明黑底
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # 第一行: 帧号 + EAR/MAR
    ear = kp.get("ear_avg")
    mar = kp.get("mar")
    line1 = f"Frame {idx}"
    line1 += f"  |  EAR={ear:.3f}" if ear is not None else "  |  EAR=N/A"
    line1 += f"  MAR={mar:.3f}" if mar is not None else "  MAR=N/A"

    ear_l = kp.get("ear_left")
    ear_r = kp.get("ear_right")
    if ear_l is not None and ear_r is not None:
        line1 += f"  (L={ear_l:.3f} R={ear_r:.3f})"

    text_color = (255, 255, 255)
    cv2.putText(vis, line1, (10, h - bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # 第二行: 异常标签
    if anomaly_types:
        unique = list(dict.fromkeys(anomaly_types))  # 去重保序
        labels = [_ANOMALY_LABEL.get(t, t) for t in unique[:4]]
        line2 = " | ".join(labels)
        cv2.putText(vis, line2, (10, h - bar_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _VIS_ANOMALY_COLOR, 1, cv2.LINE_AA)


def save_visualization(
    frames: list[np.ndarray],
    keypoints_seq: list[dict],
    l1: dict,
    l2: dict,
    output_dir: str,
) -> None:
    """保存带语义标注的关键帧可视化。

    可视化元素及语义:
    - 青色线框: 面部轮廓 (468 点网格拓扑)
    - 绿色轮廓: 眼部 ROI (EAR 眨眼检测区域)
    - 品红轮廓: 外唇 ROI (MAR 嘴部开合检测区域)
    - 橙色填充: 内唇多边形 (舌头/牙齿消失检测区域)
    - 青绿骨骼: 左手骨架 (手指融合/骨段突变检测)
    - 蓝橙骨骼: 右手骨架
    - 黄色圆环: 指尖标记 (融合距离检测点)
    - 红色标注: 异常检测结果
    - 右上图例: 当前帧各可视化元素语义说明
    - 底部信息栏: 帧号 + EAR/MAR 数值 + 异常类型
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 收集所有异常帧及其类型
    anomaly_frames: dict[int, list[str]] = {}
    for a in l1["all"] + l2["all"]:
        fidx = a.get("frame_idx", -1)
        if 0 <= fidx < len(frames):
            if fidx not in anomaly_frames:
                anomaly_frames[fidx] = []
            anomaly_frames[fidx].append(a.get("type", "unknown"))

    # 选择要保存的帧: 首帧 + 尾帧 + 异常帧 + 均匀采样
    key_indices: set[int] = {0, len(frames) - 1}
    key_indices.update(anomaly_frames.keys())
    step = max(1, len(frames) // 20)
    for i in range(0, len(frames), step):
        key_indices.add(i)

    for idx in sorted(key_indices):
        if idx >= len(frames):
            continue
        vis = frames[idx].copy()
        kp = keypoints_seq[idx] if idx < len(keypoints_seq) else {}
        is_anomaly = idx in anomaly_frames

        # ── 顶部状态条 (红=异常 / 绿=正常) ──
        bar_color = _VIS_ANOMALY_COLOR if is_anomaly else _VIS_NORMAL_COLOR
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 4), bar_color, -1)

        # ── 面部网格 + 眼/嘴 ROI 掩膜 ──
        face = kp.get("face")
        if face is not None:
            _draw_face_mesh(vis, face)

        # ── 手部骨骼连接 ──
        left_hand = kp.get("left_hand")
        if left_hand is not None:
            _draw_hand_skeleton(vis, left_hand, _VIS_HAND_LEFT_COLOR)
        right_hand = kp.get("right_hand")
        if right_hand is not None:
            _draw_hand_skeleton(vis, right_hand, _VIS_HAND_RIGHT_COLOR)

        # ── 图例面板 ──
        _draw_legend(vis, kp, is_anomaly)

        # ── 底部信息栏 ──
        _draw_info_bar(vis, idx, kp, anomaly_frames.get(idx))

        tag = "ANOMALY_" if is_anomaly else ""
        cv2.imwrite(str(out / f"{tag}frame_{idx:04d}.jpg"), vis)

    print(f"\n可视化已保存到 {out}/ ({len(key_indices)} 张)")
    anomaly_count = sum(1 for idx in key_indices if idx in anomaly_frames)
    print(f"  其中异常帧: {anomaly_count} 张")
    print(f"\n  可视化语义说明:")
    print(f"    青色线框   → 面部轮廓 (468 点网格)")
    print(f"    绿色轮廓   → 眼部 ROI (EAR 眨眼检测)")
    print(f"    品红轮廓   → 外唇 ROI (MAR 开合检测)")
    print(f"    橙色填充   → 内唇区域 (舌头/牙齿消失检测)")
    print(f"    青绿骨骼   → 左手骨架 (手指融合/骨段检测)")
    print(f"    蓝橙骨骼   → 右手骨架")
    print(f"    黄色圆环   → 指尖标记 (融合距离检测点)")
    print(f"    红色标注   → 异常检测结果")


# ── 6. 主入口 ─────────────────────────────────────────────


def main():
    _load_repo_dotenv()
    parser = argparse.ArgumentParser(description="生物特征异常三级检测调试脚本")
    parser.add_argument("--input", required=True, help="视频文件路径")
    parser.add_argument("--device", default="cpu", help="推理设备 (cuda/cpu)")
    parser.add_argument("--sample-rate", type=int, default=1, help="每 N 帧采样 1 帧")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化到 outputs/bio_anomaly/")
    parser.add_argument("--no-mllm", action="store_true", help="禁用 Level 3 MLLM")
    parser.add_argument(
        "--mllm-provider",
        default="vllm",
        choices=["vllm", "openai", "anthropic", "dashscope"],
        help="MLLM API 提供方（默认 vllm：OpenAI 兼容本地服务）",
    )
    parser.add_argument(
        "--mllm-model",
        default="qwen3.5:9b",
        help="MLLM 模型名（dashscope 可传 qwen3-vl-8b-thinking 等）",
    )
    parser.add_argument(
        "--mllm-api-key",
        default=os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("VLLM_API_KEY", ""),
        help="API Key（dashscope 必填；vllm 可空）",
    )
    parser.add_argument(
        "--mllm-base-url",
        default=os.environ.get("DASHSCOPE_BASE_URL", "")
        or os.environ.get("VLLM_OPENAI_BASE_URL", ""),
        help="Base URL（vllm 默认代码内 localhost:8201/v1；dashscope 可设国际区）",
    )
    parser.add_argument(
        "--mllm-fps",
        type=int,
        default=2,
        help="视频路径模式抽帧 fps（dashscope / vllm）",
    )
    parser.add_argument("--mllm-max-crops", type=int, default=8, help="Level 3 最大 ROI 裁剪数")
    args = parser.parse_args()

    video_path = args.input
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在 {video_path}")
        sys.exit(1)

    config = BiologicalAnomalyConfig(
        enable_mllm=not args.no_mllm,
        mllm_max_crops=args.mllm_max_crops,
    )
    mllm_client = build_mllm_client(args, config.enable_mllm)
    t_total = time.time()

    print(f"{'='*60}")
    print(f"生物特征异常检测 (Biological Anomaly Detection)")
    print(f"三级混合方案: L1(快筛) → L2(结构) → L3(MLLM)")
    print(f"{'='*60}")
    print(f"视频: {video_path}")
    print(f"设备: {args.device}")
    print(f"MLLM: {'启用' if config.enable_mllm else '禁用'}")

    # Step 1: 读帧
    frames, fps = load_frames(video_path, args.sample_rate)
    if len(frames) < 2:
        print("错误: 视频帧数不足")
        sys.exit(1)

    # Step 2: MediaPipe 关键点提取
    keypoints_seq = extract_keypoints(frames, fps)

    # Step 3: Level 1 快速筛选
    l1 = run_level1(keypoints_seq, fps, config)

    # Step 4: Level 2 结构检测
    l2 = run_level2(keypoints_seq, fps, frames=frames)

    # Step 4.7: Level 3 MLLM 语义兜底
    l3 = run_level3(frames, keypoints_seq, l1, l2, config, mllm_client)
    print(f"\n  [Level3-MLLM] 异常: {len(l3['anomalies'])} 处")

    # Step 4.5: 按部位正常帧比例 (VMBench OIS 风格)
    body_seq = [kp.get("body") for kp in keypoints_seq]
    part_scores = compute_body_part_scores(body_seq, smoothing_window=config.smoothing_window)

    # Step 5: 综合评分
    scores = compute_score(l1, l2, config, len(frames), l3_score=l3["level3_score"])

    print(f"\n{'='*60}")
    print(f"综合评分")
    print(f"{'='*60}")
    print(f"  Level 1 (快筛) 得分:    {scores['level1_score']:.4f}  (权重 {config.level1_weight})")
    print(f"  Level 2 (结构) 得分:    {scores['level2_score']:.4f}  (权重 {config.level2_weight})")
    print(f"  Level 3 (MLLM) 得分:    {scores['level3_score']:.4f}  (权重 {config.level3_weight})")
    print(f"  ─────────────────────────────")
    print(f"  生物特征质量分:          {scores['bio_quality_score']:.4f}")

    # 评级
    s = scores["bio_quality_score"]
    if s >= 0.90:
        grade = "优秀 - 人体特征自然一致"
    elif s >= 0.75:
        grade = "良好 - 存在轻微异常"
    elif s >= 0.50:
        grade = "一般 - 存在明显结构异常"
    else:
        grade = "差 - 严重生物特征异常"
    print(f"  评级: {grade}")

    # 按部位正常帧比例
    if part_scores:
        print(f"\n{'='*60}")
        print(f"按部位正常帧比例 (VMBench OIS 风格)")
        print(f"{'='*60}")
        for part, score in sorted(part_scores.items()):
            bar_len = int(score * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            print(f"  {part:25s} [{bar}] {score:.2%}")

    # 异常摘要
    all_anomalies = l1["all"] + l2["all"] + l3["anomalies"]
    if all_anomalies:
        print(f"\n{'='*60}")
        print(f"异常摘要 (共 {len(all_anomalies)} 处)")
        print(f"{'='*60}")
        by_type: dict[str, int] = {}
        for a in all_anomalies:
            t = a.get("type", "mllm_anomaly")
            by_type[t] = by_type.get(t, 0) + 1
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c} 处")

    # Step 6: 可视化
    if args.save_vis:
        out_dir = str(ROOT / "outputs" / "bio_anomaly")
        save_visualization(frames, keypoints_seq, l1, l2, out_dir)

    # 保存 JSON 结果
    result_path = ROOT / "outputs" / "bio_anomaly_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed_total = time.time() - t_total
    result_json = {
        "video": video_path,
        "n_frames": len(frames),
        "fps": fps,
        "scores": scores,
        "anomaly_summary": {t: c for t, c in sorted(by_type.items(), key=lambda x: -x[1])}
        if all_anomalies
        else {},
        "l1_count": len(l1["all"]),
        "l2_count": len(l2["all"]),
        "l3_count": len(l3["anomalies"]),
        "l3_skipped": bool(l3.get("skipped", False)),
        "l3_raw_result": l3.get("raw_result", {}),
        "elapsed_sec": round(elapsed_total, 3),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"\n总耗时: {elapsed_total:.1f}s")
    print(f"结果已保存到 {result_path}")


if __name__ == "__main__":
    main()
