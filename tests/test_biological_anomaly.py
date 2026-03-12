"""测试 biological_anomaly 基础检测函数。"""

from src.biological_anomaly.anomaly_rules import (
    HAND_CONSTRAINTS,
    HAND_STRUCTURE_CONSTRAINTS,
    EYE_CONSTRAINTS,
    MOUTH_CONSTRAINTS,
)
from src.biological_anomaly.eye_anomaly import (
    detect_eye_anomalies,
    detect_eye_symmetry_anomalies,
)
from src.biological_anomaly.mouth_anomaly import detect_mouth_anomalies_l1


# ---------- 约束常量 ----------


def test_eye_constraints_exist():
    assert "ear_blink_threshold" in EYE_CONSTRAINTS
    assert "max_no_blink_frames" in EYE_CONSTRAINTS


def test_hand_constraints_exist():
    assert HAND_CONSTRAINTS["finger_count"] == 5


def test_mouth_constraints_exist():
    assert "mar_jump_threshold" in MOUTH_CONSTRAINTS
    assert "inner_lip_indices" in MOUTH_CONSTRAINTS


def test_hand_structure_constraints_exist():
    assert "fingertip_indices" in HAND_STRUCTURE_CONSTRAINTS
    assert len(HAND_STRUCTURE_CONSTRAINTS["fingertip_indices"]) == 5


# ---------- Level 1: 眼部 ----------


def test_eye_no_blink_detected():
    ear_sequence = [0.35] * 100
    anomalies = detect_eye_anomalies(ear_sequence, fps=30.0)
    assert any(a["type"] == "no_blink" for a in anomalies)


def test_eye_normal_blink():
    ear_seq = [0.35] * 20 + [0.15, 0.10, 0.15] + [0.35] * 20
    anomalies = detect_eye_anomalies(ear_seq, fps=30.0)
    no_blink = [a for a in anomalies if a["type"] == "no_blink"]
    assert len(no_blink) == 0


def test_eye_handles_none():
    ear_seq = [0.35, None, 0.35, 0.35]
    anomalies = detect_eye_anomalies(ear_seq, fps=30.0)
    assert isinstance(anomalies, list)


def test_eye_symmetry_anomaly():
    left = [0.30] * 10
    right = [0.10] * 10  # 差值 0.20 > 0.15
    anomalies = detect_eye_symmetry_anomalies(left, right, tolerance=0.15)
    assert any(a["type"] == "eye_asymmetry" for a in anomalies)


def test_eye_symmetry_normal():
    left = [0.30] * 10
    right = [0.29] * 10
    anomalies = detect_eye_symmetry_anomalies(left, right, tolerance=0.15)
    assert len(anomalies) == 0


# ---------- Level 1: 嘴部 ----------


def test_mouth_l1_mar_jump():
    mar_seq = [0.1, 0.1, 0.5, 0.1]  # 0.4 > 0.3 threshold
    anomalies = detect_mouth_anomalies_l1(mar_seq, fps=30.0)
    assert any(a["type"] == "mouth_discontinuity" for a in anomalies)


def test_mouth_l1_sustained_open():
    # 超过 3 秒（90 帧 @30fps）的持续张嘴
    mar_seq = [0.6] * 100 + [0.1]
    anomalies = detect_mouth_anomalies_l1(mar_seq, fps=30.0)
    assert any(a["type"] == "prolonged_mouth_opening" for a in anomalies)


def test_mouth_l1_normal():
    mar_seq = [0.2, 0.21, 0.19, 0.2]
    anomalies = detect_mouth_anomalies_l1(mar_seq, fps=30.0)
    assert len(anomalies) == 0
