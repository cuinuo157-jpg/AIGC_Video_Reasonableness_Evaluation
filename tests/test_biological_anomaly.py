from src.biological_anomaly.anomaly_rules import HAND_CONSTRAINTS, EYE_CONSTRAINTS
from src.biological_anomaly.eye_anomaly import detect_eye_anomalies
from src.biological_anomaly.hand_anomaly import detect_hand_anomalies


def test_eye_constraints_exist():
    assert "ear_blink_threshold" in EYE_CONSTRAINTS
    assert "max_no_blink_frames" in EYE_CONSTRAINTS


def test_hand_constraints_exist():
    assert HAND_CONSTRAINTS["finger_count"] == 5


def test_eye_no_blink_detected():
    ear_sequence = [0.35] * 100
    anomalies = detect_eye_anomalies(ear_sequence, fps=30.0)
    assert any(a["type"] == "no_blink" for a in anomalies)


def test_eye_normal_blink():
    ear_seq = [0.35] * 20 + [0.15, 0.10, 0.15] + [0.35] * 20
    anomalies = detect_eye_anomalies(ear_seq, fps=30.0)
    no_blink = [a for a in anomalies if a["type"] == "no_blink"]
    assert len(no_blink) == 0


def test_hand_wrong_finger_count():
    finger_counts = [5, 5, 6, 5, 5]
    anomalies = detect_hand_anomalies(finger_counts=finger_counts)
    assert any(a["type"] == "wrong_finger_count" for a in anomalies)
