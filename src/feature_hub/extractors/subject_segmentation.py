"""主体分割特征提取器。

两级策略：
  1. Grounding DINO zero-shot 检测 → SAM2 精确分割
  2. Grounding DINO 无结果时 fallback 到 SAM2 auto-mask generator
  3. SAM2 不可用时降级为 MediaPipe 关键点椭圆近似 mask

采样策略：每 5 帧分割一次，中间帧复用最近的 mask。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SEGMENT_INTERVAL = 5
_GROUNDING_TEXT = "person . animal . vehicle . object"
_GROUNDING_CONFIDENCE = 0.3
_TOP_K_MASKS = 3


@dataclass
class SubjectSegmentationResult:
    """主体分割结果。"""

    masks: list[np.ndarray] = field(default_factory=list)
    """每帧二值 mask (H, W) bool，多个主体合并为一个 mask。"""
    subject_ratios: list[float] = field(default_factory=list)
    """每帧主体面积占比。"""
    method: str = "none"
    """使用的分割方法: 'sam2_grounding' | 'sam2_auto' | 'keypoint_fallback' | 'none'"""


def _sam2_recommended_cache_paths() -> list[Path]:
    """返回建议放置 SAM2 权重的 .cache 路径。"""
    project_root = Path(__file__).resolve().parents[3]
    return [
        project_root / ".cache" / "sam2.1_hiera_large.pt",
        project_root / ".cache" / "sam2.1_hiera_b+.pt",
        project_root / ".cache" / "sam2.1_hiera_s.pt",
        project_root / ".cache" / "sam2.1_hiera_tiny.pt",
    ]


def _grounding_dino_recommended_cache_paths() -> list[Path]:
    """返回建议放置 GroundingDINO 权重的 .cache 路径。"""
    project_root = Path(__file__).resolve().parents[3]
    return [
        project_root / ".cache" / "groundingdino_swinb_cogcoor.pth",
        project_root / ".cache" / "groundingdino_swint_ogc.pth",
    ]


def _resolve_grounding_dino_paths(sam2_dir: Path) -> tuple[str, str] | None:
    """解析 GroundingDINO 的本地 config 与 checkpoint。"""
    checkpoint_to_config = [
        (
            Path(__file__).resolve().parents[3] / ".cache" / "groundingdino_swinb_cogcoor.pth",
            sam2_dir / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py",
        ),
        (
            Path(__file__).resolve().parents[3] / ".cache" / "groundingdino_swint_ogc.pth",
            sam2_dir / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
        ),
    ]
    for checkpoint, config in checkpoint_to_config:
        if checkpoint.exists() and config.exists():
            return str(config), str(checkpoint)
    return None


def _resolve_local_bert_path() -> str | None:
    """解析本地 BERT 目录，避免 GroundingDINO 在线下载。"""
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / ".cache" / "bert-base-uncased",
        project_root / ".cache" / "google-bert" / "bert-base-uncased",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return str(p)
    return None


def _resolve_sam2_checkpoint_and_config(sam2_dir: Path) -> tuple[str, str] | None:
    """解析可用 SAM2 checkpoint，并返回匹配的配置路径。"""
    project_root = Path(__file__).resolve().parents[3]
    checkpoint_candidates = [
        # 项目约定：third_party 相关模型权重统一放在项目根目录 .cache
        project_root / ".cache" / "sam2.1_hiera_large.pt",
        project_root / ".cache" / "sam2.1_hiera_l.pt",
        project_root / ".cache" / "sam2.1_hiera_base_plus.pt",
        project_root / ".cache" / "sam2.1_hiera_b+.pt",
        project_root / ".cache" / "sam2.1_hiera_small.pt",
        project_root / ".cache" / "sam2.1_hiera_s.pt",
        project_root / ".cache" / "sam2.1_hiera_tiny.pt",
        project_root / ".cache" / "sam2_hiera_large.pt",
        project_root / ".cache" / "sam2_hiera_l.pt",
        project_root / ".cache" / "sam2_hiera_base_plus.pt",
        project_root / ".cache" / "sam2_hiera_b+.pt",
        project_root / ".cache" / "sam2_hiera_small.pt",
        project_root / ".cache" / "sam2_hiera_s.pt",
        project_root / ".cache" / "sam2_hiera_tiny.pt",
    ]
    checkpoint = next((p for p in checkpoint_candidates if p.exists()), None)
    if checkpoint is None:
        return None

    name = checkpoint.name.lower()
    is_v21 = "sam2.1" in name

    if "hiera_large" in name or "hiera_l" in name:
        size = "l"
    elif "hiera_base_plus" in name or "hiera_b+" in name:
        size = "b+"
    elif "hiera_small" in name or "hiera_s" in name:
        size = "s"
    else:
        size = "t"

    if is_v21:
        config_path = f"configs/sam2.1/sam2.1_hiera_{size}.yaml"
    else:
        config_path = f"configs/sam2/sam2_hiera_{size}.yaml"

    return str(checkpoint), config_path


# ── SAM2 + Grounding DINO ─────────────────────────────────────


def _try_load_grounding_dino(device: str, offline: bool = False) -> tuple[Any, Any] | None:
    """尝试加载本地 GroundingDINO 模型（不依赖 HuggingFace 在线下载）。"""
    try:
        import sys
        import torch

        sam2_dir = Path(__file__).resolve().parents[3] / "third_party" / "Grounded-SAM-2"
        if str(sam2_dir) not in sys.path:
            sys.path.insert(0, str(sam2_dir))
        from grounding_dino.groundingdino.models import build_model
        from grounding_dino.groundingdino.util.misc import clean_state_dict
        from grounding_dino.groundingdino.util.slconfig import SLConfig
        import grounding_dino.groundingdino.datasets.transforms as T
        from grounding_dino.groundingdino.models.GroundingDINO import ms_deform_attn as msda

        resolved = _resolve_grounding_dino_paths(sam2_dir)
        if resolved is None:
            suggested = " | ".join(str(p) for p in _grounding_dino_recommended_cache_paths())
            logger.warning(
                "Grounding DINO 本地权重未找到。请将权重放到项目 .cache，例如: %s",
                suggested,
            )
            return None

        config_path, checkpoint_path = resolved
        # GroundingDINO 在 CUDA 路径依赖自定义 _C 扩展；若未编译则自动回退 CPU。
        gdino_device = device
        if device.startswith("cuda") and getattr(msda, "_C", None) is None:
            logger.warning(
                "GroundingDINO 自定义算子 _C 不可用，自动回退到 CPU 推理（速度会较慢）。"
            )
            gdino_device = "cpu"

        args = SLConfig.fromfile(config_path)
        args.device = gdino_device
        # In inference-only flow, disable checkpoint to avoid
        # reentrant warnings and unnecessary compute overhead.
        if hasattr(args, "use_checkpoint"):
            args.use_checkpoint = False
        if hasattr(args, "use_transformer_ckpt"):
            args.use_transformer_ckpt = False
        local_bert = _resolve_local_bert_path()
        if local_bert is not None:
            args.text_encoder_type = local_bert
            logger.info("GroundingDINO 使用本地 BERT: %s", local_bert)
        else:
            if offline:
                logger.error(
                    "离线模式下未找到本地 BERT（bert-base-uncased）。"
                    "请放置到 .cache/bert-base-uncased 或 .cache/google-bert/bert-base-uncased"
                )
                return None
            logger.warning(
                "未找到本地 BERT（bert-base-uncased），GroundingDINO 可能触发联网下载。"
            )

        if offline:
            # 强制 Hugging Face 走离线模式，防止任何网络请求。
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model = build_model(args)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(clean_state_dict(state_dict), strict=False)
        model = model.to(gdino_device)
        model.eval()

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        logger.info(
            "Grounding DINO 本地加载成功: checkpoint=%s, config=%s",
            checkpoint_path, config_path,
        )
        # 保持二元组返回以兼容现有调用方
        return model, transform
    except Exception as e:
        logger.warning("Grounding DINO 加载失败: %s", e)
        return None


def _try_load_sam2(device: str) -> Any | None:
    """尝试加载 SAM2ImagePredictor。"""
    try:
        import sys
        import torch

        sam2_dir = Path(__file__).resolve().parents[3] / "third_party" / "Grounded-SAM-2"
        if str(sam2_dir) not in sys.path:
            sys.path.insert(0, str(sam2_dir))

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        resolved = _resolve_sam2_checkpoint_and_config(sam2_dir)
        if resolved is None:
            suggested = " | ".join(str(p) for p in _sam2_recommended_cache_paths())
            logger.warning(
                "SAM2 checkpoint 未找到。请将权重放到项目 .cache，例如: %s",
                suggested,
            )
            return None
        checkpoint, config_path = resolved

        sam2_model = build_sam2(
            config_path,
            checkpoint,
            device=device,
        )
        predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2ImagePredictor 加载成功: checkpoint=%s, config=%s", checkpoint, config_path)
        return predictor
    except Exception as e:
        logger.warning("SAM2 加载失败: %s", e)
        return None


def _detect_boxes_grounding_dino(
    frame_rgb: np.ndarray,
    gdino_model: Any,
    gdino_transform: Any,
    device: str,
    return_semantics: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str], np.ndarray] | None:
    """用 Grounding DINO 检测主体 bounding boxes。

    Returns:
        boxes (N, 4) in xyxy format, or None if no detection.
    """
    import torch
    from PIL import Image

    pil_image = Image.fromarray(frame_rgb)
    image_tensor, _ = gdino_transform(pil_image, None)
    caption = _GROUNDING_TEXT.lower().strip()
    if not caption.endswith("."):
        caption += "."

    model_device = str(next(gdino_model.parameters()).device)
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=model_device.startswith("cuda"),
        ):
            outputs = gdino_model(image_tensor[None].to(model_device), captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (Nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (Nq, 4), cxcywh normalized
    keep = prediction_logits.max(dim=1)[0] > _GROUNDING_CONFIDENCE
    if not torch.any(keep):
        return None

    logits_kept = prediction_logits[keep]
    boxes = prediction_boxes[keep].numpy()
    scores = logits_kept.max(dim=1)[0].numpy()

    h, w = frame_rgb.shape[:2]
    # cxcywh(normalized) -> xyxy(pixel)
    cx = boxes[:, 0] * w
    cy = boxes[:, 1] * h
    bw = boxes[:, 2] * w
    bh = boxes[:, 3] * h
    xyxy = np.stack([
        cx - bw / 2.0,
        cy - bh / 2.0,
        cx + bw / 2.0,
        cy + bh / 2.0,
    ], axis=1)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)

    order = np.argsort(-scores)
    topk = order[:_TOP_K_MASKS]
    top_boxes = xyxy[topk]
    top_scores = scores[topk]

    if not return_semantics:
        return top_boxes

    labels: list[str] = []
    try:
        from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap

        tokenizer = gdino_model.tokenizer
        tokenized = tokenizer(caption)
        for idx in topk:
            phrase = get_phrases_from_posmap(
                logits_kept[idx] > _GROUNDING_CONFIDENCE,
                tokenized,
                tokenizer,
            ).replace(".", "").strip()
            labels.append(phrase or "object")
    except Exception:
        labels = ["object"] * len(top_boxes)

    return top_boxes, labels, top_scores


def _segment_with_sam2_boxes(
    frame_rgb: np.ndarray,
    sam2_predictor: Any,
    boxes: np.ndarray,
) -> np.ndarray:
    """用 SAM2 + box prompt 分割，返回合并 mask (H, W) bool。"""
    import torch

    sam2_predictor.set_image(frame_rgb)
    masks_list = []
    predictor_device = str(next(sam2_predictor.model.parameters()).device)
    for box in boxes:
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=predictor_device.startswith("cuda"),
            ):
                masks, scores, _ = sam2_predictor.predict(
                    box=box,
                    multimask_output=True,
                )
        # 取分数最高的 mask
        best_idx = int(np.argmax(scores))
        masks_list.append(masks[best_idx])

    h, w = frame_rgb.shape[:2]
    merged = np.zeros((h, w), dtype=bool)
    for m in masks_list:
        merged |= m.astype(bool)
    return merged


def _segment_with_sam2_auto(
    frame_rgb: np.ndarray,
    sam2_predictor: Any,
) -> np.ndarray:
    """SAM2 auto mask fallback: 全图分割取面积最大的 Top-K mask。"""
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        mask_generator = SAM2AutomaticMaskGenerator(sam2_predictor.model)
        predictor_device = str(next(sam2_predictor.model.parameters()).device)
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=predictor_device.startswith("cuda"),
            ):
                masks = mask_generator.generate(frame_rgb)

        if not masks:
            return np.zeros(frame_rgb.shape[:2], dtype=bool)

        # 按面积排序取 Top-K
        masks.sort(key=lambda x: x["area"], reverse=True)
        h, w = frame_rgb.shape[:2]
        merged = np.zeros((h, w), dtype=bool)
        for m in masks[:_TOP_K_MASKS]:
            merged |= m["segmentation"].astype(bool)
        return merged
    except Exception as e:
        logger.warning("SAM2 auto mask 失败: %s", e)
        return np.zeros(frame_rgb.shape[:2], dtype=bool)


# ── MediaPipe 关键点降级 ──────────────────────────────────────


def _keypoint_fallback_mask(
    frame_rgb: np.ndarray,
    keypoints_per_frame: list[Any] | None,
    frame_idx: int,
) -> np.ndarray:
    """使用 MediaPipe 关键点生成椭圆近似 mask。"""
    h, w = frame_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    if keypoints_per_frame is None or frame_idx >= len(keypoints_per_frame):
        return mask

    kps = keypoints_per_frame[frame_idx]
    if kps is None:
        return mask

    # MediaPipe 33-point body pose: 提取有效关键点坐标
    try:
        if hasattr(kps, "landmark"):
            points = [
                (int(lm.x * w), int(lm.y * h))
                for lm in kps.landmark
                if lm.visibility > 0.5
            ]
        elif isinstance(kps, np.ndarray) and kps.ndim == 2:
            points = [(int(x * w), int(y * h)) for x, y in kps[:, :2] if x > 0 and y > 0]
        elif isinstance(kps, list):
            points = []
            for kp in kps:
                if isinstance(kp, dict) and kp.get("visibility", 0) > 0.5:
                    points.append((int(kp["x"] * w), int(kp["y"] * h)))
                elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    px, py = int(kp[0] * w), int(kp[1] * h)
                    if 0 < px < w and 0 < py < h:
                        points.append((px, py))
        else:
            return mask
    except (TypeError, ValueError, IndexError):
        return mask

    if len(points) < 3:
        return mask

    pts = np.array(points, dtype=np.int32)
    # 用最小外接椭圆拟合人体
    if len(pts) >= 5:
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(mask.astype(np.uint8), ellipse, 1, thickness=-1)
        mask = mask.astype(bool)
    else:
        # 点太少时用凸包
        hull = cv2.convexHull(pts)
        temp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(temp, hull, 1)
        mask = temp.astype(bool)

    return mask


# ── 主入口 ────────────────────────────────────────────────────


def extract_subject_masks(
    video_path: str,
    device: str,
    hub: Any = None,
) -> SubjectSegmentationResult:
    """提取每帧主体二值 mask。

    Args:
        video_path: 视频文件路径。
        device: 推理设备。
        hub: FeatureHub 实例。

    Returns:
        SubjectSegmentationResult。
    """
    if hub is None:
        logger.warning("subject_segmentation 需要 hub 参数")
        return SubjectSegmentationResult()

    frames = hub.get("video_frames")
    if not frames or len(frames) < 1:
        return SubjectSegmentationResult()

    n_frames = len(frames)
    masks: list[np.ndarray] = [np.zeros(frames[0].shape[:2], dtype=bool)] * n_frames
    method = "none"

    # 尝试加载 SAM2
    sam2_predictor = _try_load_sam2(device)
    offline_mode = os.environ.get("AIGC_OFFLINE_MODE", "0") == "1"
    gdino = (
        _try_load_grounding_dino(device, offline=offline_mode)
        if sam2_predictor is not None else None
    )

    if sam2_predictor is not None:
        # 确定采样帧索引
        sample_indices = list(range(0, n_frames, _SEGMENT_INTERVAL))
        if (n_frames - 1) not in sample_indices:
            sample_indices.append(n_frames - 1)

        for idx in sample_indices:
            frame_rgb = frames[idx]
            # BGR → RGB (video_frames 通常为 BGR)
            if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3:
                frame_rgb_conv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb_conv = frame_rgb

            mask = None

            # 策略 1: Grounding DINO → SAM2 box prompt
            if gdino is not None:
                gdino_model, gdino_transform = gdino
                boxes = _detect_boxes_grounding_dino(
                    frame_rgb_conv, gdino_model, gdino_transform, device
                )
                if boxes is not None and len(boxes) > 0:
                    mask = _segment_with_sam2_boxes(
                        frame_rgb_conv, sam2_predictor, boxes
                    )
                    if method == "none":
                        method = "sam2_grounding"

            # 策略 2: SAM2 auto mask fallback
            if mask is None or not mask.any():
                mask = _segment_with_sam2_auto(frame_rgb_conv, sam2_predictor)
                if mask.any() and method == "none":
                    method = "sam2_auto"

            if mask is not None:
                masks[idx] = mask

        # 中间帧复用最近的采样帧 mask
        _interpolate_masks(masks, sample_indices)

    else:
        # 降级: MediaPipe 关键点椭圆 mask
        keypoints = None
        try:
            keypoints = hub.get("keypoints")
        except (KeyError, Exception):
            pass

        if keypoints is not None:
            for i in range(n_frames):
                frame_rgb = frames[i]
                masks[i] = _keypoint_fallback_mask(frame_rgb, keypoints, i)
            method = "keypoint_fallback"

    # 计算面积占比
    subject_ratios = []
    for m in masks:
        total_pixels = m.shape[0] * m.shape[1]
        ratio = float(np.sum(m)) / total_pixels if total_pixels > 0 else 0.0
        subject_ratios.append(ratio)

    return SubjectSegmentationResult(
        masks=masks,
        subject_ratios=subject_ratios,
        method=method,
    )


def _interpolate_masks(
    masks: list[np.ndarray],
    sample_indices: list[int],
) -> None:
    """将未采样帧的 mask 填充为最近采样帧的 mask（就近原则）。"""
    if not sample_indices:
        return

    for i in range(len(masks)):
        if i in sample_indices:
            continue
        # 找最近的采样帧
        nearest_idx = min(sample_indices, key=lambda s: abs(s - i))
        masks[i] = masks[nearest_idx]
