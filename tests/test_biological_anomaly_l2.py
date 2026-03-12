"""测试 Level 2 结构异常检测。"""

from typing import List, Optional, Tuple

import numpy as np

from src.biological_anomaly.hand_anomaly import detect_hand_structure_anomalies_l2
from src.biological_anomaly.mouth_anomaly import detect_mouth_temporal_anomalies_l2


# ---------- 手指融合检测 ----------


def _make_hand_landmarks(fingertip_offsets: Optional[List[Tuple[float, float]]] = None):
    """构造 (21, 3) 手部 landmark 数组。

    默认手指均匀分布，可通过 fingertip_offsets 覆盖指尖位置。
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)
    # 手腕
    landmarks[0] = [0.5, 0.8, 0.0]
    # 五个指尖默认分散
    default_tips = [
        (0.3, 0.2),  # 拇指 idx=4
        (0.4, 0.1),  # 食指 idx=8
        (0.5, 0.08), # 中指 idx=12
        (0.6, 0.1),  # 无名指 idx=16
        (0.7, 0.2),  # 小指 idx=20
    ]
    tip_indices = [4, 8, 12, 16, 20]
    offsets = fingertip_offsets or default_tips
    for i, idx in enumerate(tip_indices):
        landmarks[idx] = [offsets[i][0], offsets[i][1], 0.0]

    # MCP 关节放在指尖和手腕之间
    mcp_indices = [2, 5, 9, 13, 17]
    for i, idx in enumerate(mcp_indices):
        landmarks[idx] = [
            (landmarks[tip_indices[i]][0] + 0.5) / 2,
            (landmarks[tip_indices[i]][1] + 0.8) / 2,
            0.0,
        ]

    # 填充中间关节
    for finger_i, segments in enumerate([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                                          (13, 14, 15, 16), (17, 18, 19, 20)]):
        start = landmarks[0]  # 手腕
        end = landmarks[segments[-1]]  # 指尖
        for seg_i, seg_idx in enumerate(segments):
            t = (seg_i + 1) / len(segments)
            landmarks[seg_idx] = start * (1 - t) + end * t

    return landmarks


def test_finger_fusion_detected():
    """两个指尖重合 -> 应检测到 finger_fusion。"""
    # 食指和中指指尖重合
    fused_tips = [
        (0.3, 0.2),
        (0.45, 0.1),   # 食指
        (0.45, 0.1),   # 中指 - 与食指重合
        (0.6, 0.1),
        (0.7, 0.2),
    ]
    hand_seq = [_make_hand_landmarks(fused_tips) for _ in range(5)]
    anomalies = detect_hand_structure_anomalies_l2(hand_seq, hand_key="left_hand")
    assert any(a["type"] == "finger_fusion" for a in anomalies)


def test_no_fusion_normal_hand():
    """正常手部 -> 不应检测到融合。"""
    hand_seq = [_make_hand_landmarks() for _ in range(10)]
    anomalies = detect_hand_structure_anomalies_l2(hand_seq)
    fusion = [a for a in anomalies if a["type"] == "finger_fusion"]
    assert len(fusion) == 0


def test_bone_length_change():
    """骨段长度帧间突变 -> 应检测到 bone_length_change。"""
    normal = _make_hand_landmarks()
    # 第二帧中食指骨段突然拉长
    stretched = normal.copy()
    stretched[8] = [0.4, -0.2, 0.0]  # 食指尖大幅移动

    hand_seq = [normal, stretched]
    anomalies = detect_hand_structure_anomalies_l2(hand_seq)
    assert any(a["type"] == "bone_length_change" for a in anomalies)


def test_handles_none_landmarks():
    """None landmark 不应崩溃。"""
    hand_seq = [None, _make_hand_landmarks(), None]
    anomalies = detect_hand_structure_anomalies_l2(hand_seq)
    assert isinstance(anomalies, list)


# ---------- 嘴部时序检测 ----------


def _make_face_with_mouth(inner_scale: float = 1.0):
    """构造 (468, 3) 面部 landmark，可调节嘴部区域大小。"""
    landmarks = np.random.rand(468, 3).astype(np.float32) * 0.01
    # 设置嘴部内唇 landmark（使用 MOUTH_CONSTRAINTS 中的索引）
    from src.biological_anomaly.anomaly_rules import MOUTH_CONSTRAINTS

    inner_lip_indices = MOUTH_CONSTRAINTS["inner_lip_indices"]
    # 构造一个环形嘴部轮廓
    n = len(inner_lip_indices)
    for i, idx in enumerate(inner_lip_indices):
        angle = 2.0 * np.pi * i / n
        landmarks[idx] = [
            0.5 + 0.02 * inner_scale * np.cos(angle),
            0.6 + 0.01 * inner_scale * np.sin(angle),
            0.0,
        ]
    return landmarks


def test_mouth_area_sudden_change():
    """嘴内面积突变 -> 应检测到 mouth_area_sudden_change。"""
    normal = _make_face_with_mouth(inner_scale=1.0)
    shrunken = _make_face_with_mouth(inner_scale=0.1)  # 面积缩至原来 1%
    face_seq = [normal, normal, shrunken]
    anomalies = detect_mouth_temporal_anomalies_l2(face_seq)
    assert any(a["type"] == "mouth_area_sudden_change" for a in anomalies)


def test_mouth_stable():
    """稳定嘴部 -> 不应有异常。"""
    face_seq = [_make_face_with_mouth(1.0) for _ in range(10)]
    anomalies = detect_mouth_temporal_anomalies_l2(face_seq)
    area_changes = [a for a in anomalies if a["type"] == "mouth_area_sudden_change"]
    assert len(area_changes) == 0


def test_mouth_handles_none():
    """None face landmarks 不应崩溃。"""
    face_seq = [None, _make_face_with_mouth(), None]
    anomalies = detect_mouth_temporal_anomalies_l2(face_seq)
    assert isinstance(anomalies, list)
