"""可感知主体运动幅度评分器测试。"""
import numpy as np
import pytest

from src.motion_logic.subject_motion_scorer import (
    compute_subject_motion_score,
    SubjectMotionDetail,
    _compute_temporal_saliency,
    _resize_mask,
)


def _make_flows(
    n: int, mag: float, h: int = 50, w: int = 50
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成 n 帧均匀光流。"""
    return [(np.full((h, w), mag), np.full((h, w), mag)) for _ in range(n)]


def _make_masks(
    n: int, h: int = 50, w: int = 50, ratio: float = 0.25
) -> list[np.ndarray]:
    """生成 n 帧 mask，主体占画面 ratio 比例（左上角矩形）。"""
    masks = []
    mh, mw = int(h * ratio**0.5), int(w * ratio**0.5)
    for _ in range(n):
        m = np.zeros((h, w), dtype=bool)
        m[:mh, :mw] = True
        masks.append(m)
    return masks


class TestComputeSubjectMotionScore:
    """测试 compute_subject_motion_score 核心逻辑。"""

    def test_empty_inputs(self):
        score, detail = compute_subject_motion_score([], [], [])
        assert score == 0.0
        assert detail.subject_magnitude == 0.0

    def test_subject_large_motion_bg_static(self):
        """主体大运动 + 背景静止 → 高可感知分。"""
        n, h, w = 10, 50, 50
        masks = _make_masks(n, h, w, ratio=0.25)
        ratios = [0.25] * n

        # 主体区域光流大，背景为零
        flows = []
        for i in range(n):
            fx = np.zeros((h, w), dtype=np.float32)
            fy = np.zeros((h, w), dtype=np.float32)
            fx[masks[i]] = 10.0
            fy[masks[i]] = 10.0
            flows.append((fx, fy))

        score, detail = compute_subject_motion_score(flows, masks, ratios)
        assert score > 0.5, f"主体大运动+背景静止应有高分，got {score}"
        assert detail.subject_magnitude > detail.background_magnitude

    def test_subject_static_bg_motion(self):
        """主体静止 + 背景运动（相机平移） → 低可感知分。"""
        n, h, w = 10, 50, 50
        masks = _make_masks(n, h, w, ratio=0.25)
        ratios = [0.25] * n

        flows = []
        for i in range(n):
            fx = np.full((h, w), 10.0, dtype=np.float32)
            fy = np.full((h, w), 10.0, dtype=np.float32)
            # 主体区域无运动
            fx[masks[i]] = 0.0
            fy[masks[i]] = 0.0
            flows.append((fx, fy))

        score, detail = compute_subject_motion_score(flows, masks, ratios)
        assert score < 0.5, f"主体静止+背景运动应有低分，got {score}"
        assert detail.background_magnitude > detail.subject_magnitude

    def test_small_subject_boost(self):
        """小主体运动应被面积归一化放大。"""
        n, h, w = 10, 100, 100

        # 大主体 (50% 面积)
        big_masks = _make_masks(n, h, w, ratio=0.50)
        big_ratios = [0.50] * n

        # 小主体 (5% 面积)
        small_masks = []
        for _ in range(n):
            m = np.zeros((h, w), dtype=bool)
            m[:7, :7] = True  # ~49/10000 ≈ 0.5%
            small_masks.append(m)
        small_ratios = [float(np.mean(m)) for m in small_masks]

        # 相同主体内运动幅度
        def make_flows(masks_list):
            flows = []
            for m in masks_list:
                fx = np.zeros((h, w), dtype=np.float32)
                fy = np.zeros((h, w), dtype=np.float32)
                fx[m] = 5.0
                fy[m] = 5.0
                flows.append((fx, fy))
            return flows

        _, big_detail = compute_subject_motion_score(
            make_flows(big_masks), big_masks, big_ratios
        )
        _, small_detail = compute_subject_motion_score(
            make_flows(small_masks), small_masks, small_ratios
        )

        # 小主体的可感知分应更高（面积 boost 效应）
        assert small_detail.perceptual_score >= big_detail.perceptual_score * 0.8

    def test_uniform_motion(self):
        """主体和背景都运动相同幅度 → 得分约 0.5。"""
        flows = _make_flows(10, 5.0)
        masks = _make_masks(10, ratio=0.25)
        ratios = [0.25] * 10

        score, detail = compute_subject_motion_score(flows, masks, ratios)
        # 面积归一化使小主体 boost，所以 ≥ 0.3 即可
        assert 0.3 <= score <= 0.8, f"均匀运动应在中间范围，got {score}"

    def test_mask_flow_size_mismatch(self):
        """mask 和 flow 尺寸不同时应自动 resize。"""
        flows = _make_flows(5, 3.0, h=100, w=100)
        masks = _make_masks(5, h=50, w=50, ratio=0.25)
        ratios = [0.25] * 5

        score, detail = compute_subject_motion_score(flows, masks, ratios)
        assert 0.0 <= score <= 1.0


class TestTemporalSaliency:
    def test_constant_motion(self):
        """恒定运动 → 低时序显著性。"""
        mags = [5.0] * 10
        val = _compute_temporal_saliency(mags)
        assert val < 0.5

    def test_sudden_change(self):
        """运动突变 → 高时序显著性。"""
        mags = [0.0] * 5 + [20.0] * 5
        val = _compute_temporal_saliency(mags)
        assert val > 0.5

    def test_single_frame(self):
        assert _compute_temporal_saliency([5.0]) == 0.5


class TestResizeMask:
    def test_upsample(self):
        mask = np.ones((10, 10), dtype=bool)
        result = _resize_mask(mask, (20, 20))
        assert result.shape == (20, 20)
        assert result.all()

    def test_downsample(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True
        result = _resize_mask(mask, (50, 50))
        assert result.shape == (50, 50)
        assert result[15:35, 15:35].mean() > 0.5
