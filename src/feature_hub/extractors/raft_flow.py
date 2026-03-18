"""RAFT / TV-L1 / Farneback 多方法光流提取器。

将原 aux_motion_intensity.flow_predictor.SimpleRAFT 迁移至此，
直接引用 third_party/RAFT/core。RAFT 不可用时自动降级为 Farneback。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── 项目根目录 & RAFT 路径 ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RAFT_CORE_DIR = _PROJECT_ROOT / "third_party" / "RAFT" / "core"
_DEFAULT_MODEL_CANDIDATES = [
    _PROJECT_ROOT / ".cache" / "raft-things.pth",
    _PROJECT_ROOT / "third_party" / "pretrained_models" / "raft-things.pth",
]


def _find_raft_model() -> str | None:
    for p in _DEFAULT_MODEL_CANDIDATES:
        if p.exists():
            return str(p)
    return None


# ── SimpleRAFT (自包含，不再依赖已删除的 aux_motion_intensity) ──
class SimpleRAFT:
    """Unified optical-flow predictor: RAFT / TV-L1 / Farneback."""

    def __init__(
        self,
        device: str = "cpu",
        method: str = "farneback",
        model_path: str | None = None,
    ) -> None:
        try:
            import torch
            self._torch = torch
            self.device = (
                device
                if torch.cuda.is_available() and device.startswith("cuda")
                else "cpu"
            )
        except ImportError:
            self._torch = None
            self.device = "cpu"

        self.method = method
        self.model_path = model_path or (
            _find_raft_model() if method == "raft" else None
        )
        self.raft_model = None
        self._tvl1 = None

        if method == "tvl1":
            self._init_tvl1()
        elif method == "raft":
            self._init_raft()
        elif method != "farneback":
            logger.warning("未知光流方法 '%s'，回退到 Farneback", method)
            self.method = "farneback"

    # ── 初始化子方法 ──────────────────────────────────────────

    def _init_tvl1(self) -> None:
        try:
            self._tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            self._tvl1.setTau(0.25)
            self._tvl1.setLambda(0.15)
            self._tvl1.setTheta(0.3)
            self._tvl1.setScalesNumber(5)
            self._tvl1.setWarpingsNumber(5)
            self._tvl1.setEpsilon(0.01)
        except AttributeError:
            logger.warning("TV-L1 不可用，回退到 Farneback")
            self.method = "farneback"

    def _init_raft(self) -> None:
        torch = self._torch
        if torch is None:
            logger.warning("PyTorch 未安装，RAFT 不可用，回退到 Farneback")
            self.method = "farneback"
            return
        if not self.model_path or not Path(self.model_path).exists():
            logger.warning("RAFT 模型权重未找到，回退到 Farneback")
            self.method = "farneback"
            return
        if not _RAFT_CORE_DIR.exists():
            logger.warning("third_party/RAFT/core 不存在，回退到 Farneback")
            self.method = "farneback"
            return

        try:
            # 临时添加 RAFT core 到 sys.path
            core_str = str(_RAFT_CORE_DIR)
            if core_str not in sys.path:
                sys.path.insert(0, core_str)

            from raft import RAFT  # type: ignore[import-untyped]

            args = argparse.Namespace(
                small=False,
                mixed_precision=False,
                alternate_corr=False,
                dropout=0,
                corr_levels=4,
                corr_radius=4,
            )
            self.raft_model = RAFT(args)

            state_dict = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
            self.raft_model.load_state_dict(state_dict, strict=False)
            self.raft_model.to(self.device)
            self.raft_model.eval()

            # CUDA 兼容性测试
            if self.device.startswith("cuda"):
                try:
                    dummy = torch.zeros(1, 3, 64, 64, device=self.device)
                    with torch.no_grad():
                        self.raft_model(dummy, dummy, iters=1, test_mode=True)
                    logger.info("RAFT 初始化成功 (CUDA)")
                except RuntimeError:
                    logger.warning("CUDA 推理测试失败，RAFT 回退到 CPU")
                    self.device = "cpu"
                    self.raft_model.to(self.device)
            else:
                logger.info("RAFT 初始化成功 (CPU)")
        except Exception as e:
            logger.warning("RAFT 初始化失败: %s，回退到 Farneback", e)
            self.raft_model = None
            self.method = "farneback"

    # ── 推理 ─────────────────────────────────────────────────

    def predict_flow(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """计算双帧光流，返回 (2, H, W) ndarray。"""
        if self.method == "raft" and self.raft_model is not None:
            return self._predict_raft(image1, image2)
        return self._predict_opencv(image1, image2)

    def _predict_raft(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        torch = self._torch
        import torch.nn.functional as F  # noqa: N812

        with torch.no_grad():
            img1 = self._preprocess(image1)
            img2 = self._preprocess(image2)
            try:
                _, flow_up = self.raft_model(img1, img2, iters=20, test_mode=True)
                return flow_up[0].cpu().numpy()  # (2, H, W)
            except RuntimeError as e:
                if "CUDA" in str(e) or "kernel" in str(e).lower():
                    logger.warning("CUDA 推理失败，尝试 CPU 回退: %s", e)
                    try:
                        _, flow_up = self.raft_model(
                            img1.cpu(), img2.cpu(), iters=20, test_mode=True
                        )
                        return flow_up[0].cpu().numpy()
                    except Exception:
                        logger.warning("CPU 回退也失败，降级到 Farneback")
                        self.method = "farneback"
                        return self._predict_opencv(image1, image2)
                raise

    def _preprocess(self, img: np.ndarray):
        """RGB uint8 → (1, 3, H', W') float tensor, 8-aligned padding."""
        torch = self._torch
        import torch.nn.functional as F  # noqa: N812

        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        _, _, h, w = t.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            t = F.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
        return t.to(self.device)

    def _predict_opencv(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if image1.ndim == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if image2.ndim == 3 else image2
        if self.method == "tvl1" and self._tvl1 is not None:
            flow = self._tvl1.calc(gray1, gray2, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 5, 15, 3, 7, 1.5, 0
            )
        return flow.transpose(2, 0, 1).astype(np.float32)  # (2, H, W)


# ── 全局单例 (避免每次调用都重新加载模型) ─────────────────────
_raft_instance: SimpleRAFT | None = None


def _get_raft(device: str) -> SimpleRAFT:
    global _raft_instance
    if _raft_instance is None:
        _raft_instance = SimpleRAFT(device=device, method="raft")
    return _raft_instance


# ── Hub Extractor 接口 ───────────────────────────────────────

def _load_frames_rgb(video_path: str) -> list[np.ndarray]:
    """加载视频帧为 RGB 格式。"""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_raft_flow(
    video_path: str,
    device: str,
    hub: Any = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """使用 RAFT 提取光流序列（RAFT 不可用时降级为 Farneback）。

    返回格式与 optical_flow extractor 一致:
      list[tuple[flow_x (H,W), flow_y (H,W)]]
    """
    frames = _load_frames_rgb(video_path)
    if len(frames) < 2:
        return []

    predictor = _get_raft(device)

    flows: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(frames) - 1):
        flow_2hw = predictor.predict_flow(frames[i], frames[i + 1])
        flows.append((flow_2hw[0], flow_2hw[1]))
    return flows
