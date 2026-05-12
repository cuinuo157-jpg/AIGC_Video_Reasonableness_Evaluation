from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import BiologicalAnomalyConfig
from .eye_anomaly import detect_eye_anomalies, detect_eye_symmetry_anomalies
from .hand_anomaly import (
    detect_finger_count_anomalies,
    detect_hand_anomalies_l1,
    detect_hand_structure_anomalies_l2,
)
from .mouth_anomaly import (
    detect_mouth_anomalies_l1,
    detect_mouth_color_anomalies,
    detect_mouth_temporal_anomalies_l2,
)
from .body_anomaly import (
    compute_body_part_scores,
    detect_body_angle_anomalies,
    detect_body_bone_anomalies,
)


@dataclass
class BiologicalAnomalyResult:
    """三级生物特征异常检测结果。"""

    applicable: bool
    skip_reason: str | None = None
    # 按部位汇总
    eye_anomalies: list[dict] = field(default_factory=list)
    hand_anomalies: list[dict] = field(default_factory=list)
    mouth_anomalies: list[dict] = field(default_factory=list)
    body_anomalies: list[dict] = field(default_factory=list)
    mllm_anomalies: list[dict] = field(default_factory=list)
    mllm_raw_result: dict[str, Any] | None = None
    # 各级得分
    level1_score: float = 1.0
    level2_score: float = 1.0
    level3_score: float = 1.0
    # 按部位正常帧比例（VMBench OIS 风格）
    body_part_scores: dict[str, float] = field(default_factory=dict)
    # 汇总
    anomaly_count: int = 0
    bio_quality_score: float = 1.0


class BiologicalAnomalyAnalyzer:
    """三级混合生物特征异常分析器。

    Level 1: 基于 EAR/MAR/手部速度的快速筛选 + 全身骨骼突变
    Level 2: MediaPipe 关键点结构分析（手指融合/计数、嘴部时序/颜色）
    Level 3: MLLM 语义判定（可选兜底）
    """

    def __init__(
        self,
        config: BiologicalAnomalyConfig | None = None,
        mllm_client: Any = None,
    ) -> None:
        self.config = config or BiologicalAnomalyConfig()
        self._mllm_client = mllm_client

    def analyze(self, hub: Any) -> BiologicalAnomalyResult:
        """对视频执行三级生物特征异常检测。"""
        cfg = self.config
        sw = cfg.smoothing_window

        # ---------- 适用性检查 ----------
        face_data = hub.get("face_embedding")
        has_faces = any(fd["num_faces"] > 0 for fd in face_data)
        if not has_faces:
            return BiologicalAnomalyResult(
                applicable=False, skip_reason="no face detected"
            )

        # ---------- 获取关键点 ----------
        try:
            keypoints_seq = hub.get("keypoints")
        except (KeyError, RuntimeError):
            return BiologicalAnomalyResult(
                applicable=False, skip_reason="keypoint extraction unavailable"
            )

        n_frames = len(keypoints_seq)
        fps = 30.0

        # ========== Level 1: 快速筛选 ==========
        ear_avg_seq = [kp.get("ear_avg") for kp in keypoints_seq]
        ear_left_seq = [kp.get("ear_left") for kp in keypoints_seq]
        ear_right_seq = [kp.get("ear_right") for kp in keypoints_seq]
        mar_seq = [kp.get("mar") for kp in keypoints_seq]

        l1_eye = detect_eye_anomalies(
            ear_avg_seq, fps=fps, smoothing_window=sw,
        )
        l1_eye_sym = detect_eye_symmetry_anomalies(
            ear_left_seq, ear_right_seq, fps=fps,
            tolerance=cfg.eye_symmetry_tolerance,
            smoothing_window=sw,
        )
        l1_mouth = detect_mouth_anomalies_l1(
            mar_seq, fps=fps, smoothing_window=sw,
        )
        l1_hand = detect_hand_anomalies_l1(
            keypoints_seq, fps=fps, smoothing_window=sw,
        )

        # 全身骨骼突变（L1 级别，借鉴 VMBench OIS）
        body_seq = [kp.get("body") for kp in keypoints_seq]
        l1_body_bone = detect_body_bone_anomalies(
            body_seq, smoothing_window=sw,
        )
        l1_body_angle = detect_body_angle_anomalies(
            body_seq, smoothing_window=sw,
        )

        all_l1 = l1_eye + l1_eye_sym + l1_mouth + l1_hand + l1_body_bone + l1_body_angle
        l1_score = _score_from_anomalies(all_l1, n_frames)

        # ========== Level 2: 结构检测（全帧运行） ==========
        # 嘴部 landmark 时序
        face_seq = [kp.get("face") for kp in keypoints_seq]
        l2_mouth = detect_mouth_temporal_anomalies_l2(face_seq, fps=fps)

        # 嘴内颜色直方图突变（需要原始帧）
        l2_mouth_color: list[dict] = []
        try:
            frames = hub.get("video_frames")
            l2_mouth_color = detect_mouth_color_anomalies(frames, face_seq)
        except (KeyError, RuntimeError):
            frames = None

        # 手部结构（左手+右手分别检测）
        l2_hand: list[dict] = []
        for hand_key in ("left_hand", "right_hand"):
            hand_seq = [kp.get(hand_key) for kp in keypoints_seq]
            if any(h is not None for h in hand_seq):
                l2_hand.extend(
                    detect_hand_structure_anomalies_l2(
                        hand_seq, hand_key=hand_key,
                    )
                )
                # 手指计数检测
                l2_hand.extend(
                    detect_finger_count_anomalies(
                        hand_seq, hand_key=hand_key,
                        expected_count=cfg.finger_count_expected,
                    )
                )

        all_l2 = l2_mouth + l2_mouth_color + l2_hand
        l2_score = _score_from_anomalies(all_l2, n_frames)

        # ========== 按部位正常帧比例（VMBench OIS 风格）==========
        body_part_scores = compute_body_part_scores(
            body_seq, smoothing_window=sw,
        )

        # ========== Level 3: MLLM 兜底 ==========
        l3_anomalies: list[dict] = []
        l3_score = 1.0
        mllm_raw_result: dict[str, Any] | None = None

        if cfg.enable_mllm and self._mllm_client:
            suspicious = _collect_suspicious(all_l1 + all_l2)
            if suspicious:
                from .mllm_bio_judge import judge_biological_anomaly_mllm

                try:
                    if frames is None:
                        frames = hub.get("video_frames")
                    result = judge_biological_anomaly_mllm(
                        frames,
                        keypoints_seq,
                        suspicious,
                        self._mllm_client,
                        max_crops=cfg.mllm_max_crops,
                    )
                    mllm_raw_result = result
                    if not result.get("skipped"):
                        l3_anomalies = result.get("anomalies", [])
                        l3_score = 0.3 if result.get("has_anomalies") else 1.0
                except (KeyError, Exception):
                    pass

        # ========== 汇总 ==========
        eye_all = l1_eye + l1_eye_sym
        hand_all = l1_hand + l2_hand
        mouth_all = l1_mouth + l2_mouth + l2_mouth_color
        body_all = l1_body_bone + l1_body_angle
        total_count = (
            len(eye_all) + len(hand_all) + len(mouth_all)
            + len(body_all) + len(l3_anomalies)
        )

        bio_score = (
            cfg.level1_weight * l1_score
            + cfg.level2_weight * l2_score
            + cfg.level3_weight * l3_score
        )
        bio_score = float(np.clip(bio_score, 0, 1))

        return BiologicalAnomalyResult(
            applicable=True,
            eye_anomalies=eye_all,
            hand_anomalies=hand_all,
            mouth_anomalies=mouth_all,
            body_anomalies=body_all,
            mllm_anomalies=l3_anomalies,
            mllm_raw_result=mllm_raw_result,
            level1_score=l1_score,
            level2_score=l2_score,
            level3_score=l3_score,
            body_part_scores=body_part_scores,
            anomaly_count=total_count,
            bio_quality_score=bio_score,
        )


def _score_from_anomalies(anomalies: list[dict], n_frames: int) -> float:
    """根据异常数量和严重性计算得分。"""
    if not anomalies:
        return 1.0
    severity_penalty = {
        "low": 0.02,
        "medium": 0.05,
        "high": 0.10,
    }
    total_penalty = sum(
        severity_penalty.get(a.get("severity", "low"), 0.02) for a in anomalies
    )
    return float(np.clip(1.0 - total_penalty, 0, 1))


def _collect_suspicious(anomalies: list[dict]) -> list[dict]:
    """从 L1/L2 异常中收集疑似帧，去重。"""
    seen: set[int] = set()
    result: list[dict] = []
    for a in anomalies:
        idx = a.get("frame_idx", -1)
        if idx >= 0 and idx not in seen:
            seen.add(idx)
            result.append(a)
    return result
