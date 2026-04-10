from unittest.mock import MagicMock, patch

import numpy as np

from src.physics_consistency.pixel_drift import detect_pixel_drift
from src.physics_consistency.analyzer import PhysicsConsistencyAnalyzer, PhysicsConsistencyResult
from src.physics_consistency.config import PhysicsConfig
from src.mllm.prompts.physics_commonsense import build_physics_prompt


# ── pixel_drift 辅助信号测试 ──


def test_no_drift_in_static():
    flows = [(np.zeros((50, 50)), np.zeros((50, 50))) for _ in range(10)]
    mask = np.ones((50, 50), dtype=bool)
    assert len(detect_pixel_drift(flows, static_mask=mask)) == 0


def test_detect_drift():
    flows = [
        (np.ones((50, 50)) * 2.0, np.zeros((50, 50))) for _ in range(10)
    ]
    mask = np.ones((50, 50), dtype=bool)
    assert len(detect_pixel_drift(flows, static_mask=mask)) > 0


# ── CoT prompt 构建测试 ──


def test_build_prompt_no_drift():
    prompt = build_physics_prompt()
    assert "运动方向分析" in prompt
    assert "辅助信息" not in prompt


def test_build_prompt_with_drift():
    drift_events = [
        {"avg_magnitude": 3.5, "direction_std": 12.0, "duration_frames": 20}
    ]
    prompt = build_physics_prompt(drift_events=drift_events)
    assert "辅助信息" in prompt
    assert "3.50" in prompt
    assert "12.0" in prompt


# ── Analyzer VLM 主路径测试 ──


def _make_flows(n: int = 10):
    return [(np.ones((50, 50)) * 0.1, np.ones((50, 50)) * 0.1) for _ in range(n)]


def test_analyzer_vlm_returns_score():
    """VLM 可用时，physics_score 取 VLM 返回的 physics_score。"""
    mock_client = MagicMock()
    mock_client.config.api_provider = "openai"
    mock_client.judge_video_clip.return_value = {
        "reasoning": "场景正常",
        "scene_type": "street",
        "has_violations": False,
        "physics_score": 0.85,
        "violations": [],
    }

    hub = MagicMock()
    hub.get.side_effect = lambda key: _make_flows() if key == "optical_flow" else [np.zeros((100, 100, 3))] * 5

    analyzer = PhysicsConsistencyAnalyzer(
        config=PhysicsConfig(enable_mllm=True),
        mllm_client=mock_client,
    )
    result = analyzer.analyze(hub)

    assert result.applicable
    assert result.vlm_score == 0.85
    assert result.physics_score == 0.85
    assert result.vlm_reasoning == "场景正常"


def test_analyzer_vlm_with_violations():
    """VLM 检测到物理违规时，返回低分和违规列表。"""
    mock_client = MagicMock()
    mock_client.config.api_provider = "openai"
    mock_client.judge_video_clip.return_value = {
        "reasoning": "车辆逆行",
        "scene_type": "street",
        "has_violations": True,
        "physics_score": 0.2,
        "violations": [
            {"type": "direction_anomaly", "description": "车辆逆行", "severity": "severe", "confidence": 0.9}
        ],
    }

    hub = MagicMock()
    hub.get.side_effect = lambda key: _make_flows() if key == "optical_flow" else [np.zeros((100, 100, 3))] * 5

    analyzer = PhysicsConsistencyAnalyzer(
        config=PhysicsConfig(enable_mllm=True),
        mllm_client=mock_client,
    )
    result = analyzer.analyze(hub)

    assert result.physics_score == 0.2
    assert len(result.vlm_violations) == 1
    assert result.vlm_violations[0]["type"] == "direction_anomaly"


# ── 降级测试 ──


def test_analyzer_fallback_to_drift_when_no_mllm():
    """无 MLLM 时降级为漂移检测评分。"""
    hub = MagicMock()
    hub.get.return_value = _make_flows()

    analyzer = PhysicsConsistencyAnalyzer(
        config=PhysicsConfig(enable_mllm=False),
        mllm_client=None,
    )
    result = analyzer.analyze(hub)

    assert result.applicable
    assert result.vlm_score is None
    assert result.physics_score == result.drift_score


def test_analyzer_not_applicable_when_no_motion():
    """无运动时标记为不适用。"""
    hub = MagicMock()
    hub.get.return_value = []

    analyzer = PhysicsConsistencyAnalyzer()
    result = analyzer.analyze(hub)

    assert not result.applicable
    assert result.skip_reason == "no motion"


# ── DashScope provider 路由测试 ──


def test_analyzer_routes_to_dashscope():
    """DashScope provider 优先走 judge_video_path。"""
    mock_client = MagicMock()
    mock_client.config.api_provider = "dashscope"
    mock_client.judge_video_path.return_value = {
        "reasoning": "正常",
        "physics_score": 0.9,
        "has_violations": False,
        "violations": [],
    }

    hub = MagicMock()
    hub.get.return_value = _make_flows()
    hub.video_path = "test.mp4"

    analyzer = PhysicsConsistencyAnalyzer(
        config=PhysicsConfig(enable_mllm=True),
        mllm_client=mock_client,
    )
    result = analyzer.analyze(hub)

    assert result.vlm_score == 0.9
    mock_client.judge_video_path.assert_called_once()


def test_analyzer_routes_to_vllm_video_path():
    mock_client = MagicMock()
    mock_client.config.api_provider = "vllm"
    mock_client.judge_video_path.return_value = {
        "reasoning": "正常",
        "physics_score": 0.85,
        "has_violations": False,
        "violations": [],
    }

    hub = MagicMock()
    hub.get.return_value = _make_flows()
    hub.video_path = "test.mp4"

    analyzer = PhysicsConsistencyAnalyzer(
        config=PhysicsConfig(enable_mllm=True),
        mllm_client=mock_client,
    )
    result = analyzer.analyze(hub)

    assert result.vlm_score == 0.85
    mock_client.judge_video_path.assert_called_once()
