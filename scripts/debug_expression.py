"""
表情自然度 (Expression Naturalness) 单模块调试脚本

用法:
    python scripts/debug_expression.py --input <视频路径>
    python scripts/debug_expression.py --input <视频路径> --device cpu
    python scripts/debug_expression.py --input <视频路径> --sample-rate 5 --save-vis

参数:
    --input        视频文件路径
    --device       推理设备 (cuda / cpu)，默认 cuda
    --sample-rate  每 N 帧采样一帧，默认 1（全帧）
    --save-vis     保存可视化图片到 outputs/expression/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.expression_naturalness.au_extractor import AUExtractor
from src.expression_naturalness.au_rules import (
    NATURAL_EXPRESSIONS,
    Violation,
    check_au_combination,
    AU_ACTIVATION_THRESHOLD,
)
from src.expression_naturalness.config import ExpressionConfig
from src.expression_naturalness.temporal_analysis import (
    compute_all_au_smoothness,
    compute_au_smoothness,
)


# ── 1. 视频读取 ──────────────────────────────────────────


def load_frames(video_path: str, sample_rate: int = 1) -> tuple[list[np.ndarray], float]:
    """读取视频帧，返回 (帧列表, fps)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
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


# ── 2. AU 提取 ───────────────────────────────────────────


def extract_aus(
    frames: list[np.ndarray],
) -> list[dict[str, float]]:
    """用 Py-Feat 逐帧提取 Action Units。"""
    extractor = AUExtractor()

    print(f"\n加载 Py-Feat AU 检测模型...")
    t0 = time.time()
    # 触发惰性加载
    extractor._ensure_detector()
    print(f"模型加载完成 ({time.time() - t0:.1f}s)")

    print(f"\n逐帧提取 Action Units...")
    au_per_frame: list[dict[str, float]] = []
    t0 = time.time()
    for i, frame in enumerate(frames):
        aus = extractor.extract(frame)
        au_per_frame.append(aus)

        n = len(aus)
        tag = f"({n} 个AU)" if n > 0 else "(无AU/无人脸)"
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"  [{i+1}/{len(frames)}] {tag}")

    elapsed = time.time() - t0
    print(f"AU 提取完成 ({elapsed:.1f}s, {elapsed/len(frames):.2f}s/帧)")

    # 统计
    detected = sum(1 for aus in au_per_frame if aus)
    print(f"  检测到 AU 的帧: {detected}/{len(frames)} ({detected/len(frames)*100:.1f}%)")

    return au_per_frame


# ── 3. AU 分析 ───────────────────────────────────────────


def analyze_aus(
    au_per_frame: list[dict[str, float]],
    config: ExpressionConfig | None = None,
) -> dict:
    """分析 AU 序列: 组合冲突检测 + 时序平滑性 + 综合评分。"""
    cfg = config or ExpressionConfig()

    if not any(au_per_frame):
        print("\n未检测到任何 AU，跳过表情分析。")
        return {}

    # ── AU 序列整理 ──
    all_aus: set[str] = set()
    for aus in au_per_frame:
        all_aus.update(aus.keys())
    au_sequences = {
        au: [f.get(au, 0.0) for f in au_per_frame] for au in sorted(all_aus)
    }

    print(f"\n{'='*50}")
    print(f"AU 序列统计:")
    print(f"  检测到的 AU 类型: {len(all_aus)}")
    for au in sorted(all_aus):
        seq = au_sequences[au]
        active_count = sum(1 for v in seq if v >= AU_ACTIVATION_THRESHOLD)
        print(f"  {au}: 均值={np.mean(seq):.3f}, "
              f"最大={np.max(seq):.3f}, "
              f"激活帧={active_count}/{len(seq)}")

    # ── AU 组合冲突检测 ──
    print(f"\n{'='*50}")
    print(f"AU 组合冲突检测:")
    all_violations: list[Violation] = []
    violation_frames: list[int] = []
    for i, aus in enumerate(au_per_frame):
        vs = check_au_combination(aus)
        if vs:
            all_violations.extend(vs)
            violation_frames.append(i)

    if all_violations:
        print(f"  发现 {len(all_violations)} 处冲突 (在 {len(violation_frames)} 帧中)")
        # 按冲突类型分组
        seen = set()
        for v in all_violations:
            key = v.description
            if key not in seen:
                seen.add(key)
                count = sum(1 for vv in all_violations if vv.description == key)
                print(f"    - {v.description} (出现 {count} 次)")
    else:
        print(f"  无 AU 组合冲突")

    # ── 表情模式识别 ──
    print(f"\n{'='*50}")
    print(f"表情模式识别:")

    # 中文映射
    expr_cn = {
        "genuine_smile": "真笑", "surprise": "惊讶",
        "frown": "皱眉", "fear": "恐惧",
    }

    # 逐帧识别表情标签
    frame_expressions: list[list[str]] = []
    expr_frame_count: dict[str, int] = {}
    for aus in au_per_frame:
        active = {k for k, v in aus.items() if v >= AU_ACTIVATION_THRESHOLD}
        labels = []
        for expr_name, pattern in NATURAL_EXPRESSIONS.items():
            required_met = all(au in active for au in pattern["required"])
            forbidden_clear = all(au not in active for au in pattern["forbidden"])
            if required_met and forbidden_clear:
                labels.append(expr_name)
                expr_frame_count[expr_name] = expr_frame_count.get(expr_name, 0) + 1
        if not labels:
            labels.append("neutral")
            expr_frame_count["neutral"] = expr_frame_count.get("neutral", 0) + 1
        frame_expressions.append(labels)

    for expr_name, count in sorted(expr_frame_count.items(), key=lambda x: -x[1]):
        cn = expr_cn.get(expr_name, "中性")
        print(f"  {expr_name}({cn}): {count} 帧 "
              f"({count/len(au_per_frame)*100:.1f}%)")

    # 打印逐帧表情序列（摘要）
    print(f"\n  逐帧表情序列 (共 {len(frame_expressions)} 帧):")
    for i, labels in enumerate(frame_expressions):
        cn_labels = [expr_cn.get(l, "中性") for l in labels]
        if (i + 1) % 10 == 0 or i == 0 or i == len(frame_expressions) - 1:
            print(f"    帧{i:3d}: {', '.join(cn_labels)}")

    print(f"  (检测 AU: {', '.join(sorted(all_aus))})")

    # ── 时序平滑性 ──
    print(f"\n{'='*50}")
    print(f"AU 时序平滑性:")
    smoothness_scores = compute_all_au_smoothness(au_sequences)
    for au, score in sorted(smoothness_scores.items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {au}: {bar} {score:.4f}")

    mean_smoothness = (
        float(np.mean(list(smoothness_scores.values())))
        if smoothness_scores
        else 1.0
    )
    print(f"  ─────────────────────────")
    print(f"  平均平滑性: {mean_smoothness:.4f}")

    # ── 综合评分 ──
    violation_penalty = min(
        len(all_violations) / max(len(au_per_frame), 1), 1.0
    )
    expression_score = (
        cfg.au_combination_weight * (1.0 - violation_penalty)
        + cfg.au_smoothness_weight * mean_smoothness
        + cfg.flow_consistency_weight * 1.0
    )
    expression_score = float(np.clip(expression_score, 0, 1))

    print(f"\n{'='*50}")
    print(f"表情自然度评分:")
    print(f"{'='*50}")
    print(f"  AU 组合合理性:   {1.0 - violation_penalty:.4f} (权重 {cfg.au_combination_weight})")
    print(f"  时序平滑性:      {mean_smoothness:.4f} (权重 {cfg.au_smoothness_weight})")
    print(f"  光流一致性:      1.0000 (权重 {cfg.flow_consistency_weight}, 此脚本未计算)")
    print(f"  ─────────────────────────")
    print(f"  表情自然度综合分: {expression_score:.4f}")

    # 评级
    if expression_score >= 0.85:
        grade = "优秀 - 表情自然流畅"
    elif expression_score >= 0.70:
        grade = "良好 - 轻微不自然"
    elif expression_score >= 0.50:
        grade = "一般 - 存在明显不自然表情"
    else:
        grade = "差 - 表情生硬/冲突严重"
    print(f"  评级: {grade}")

    return {
        "au_sequences": au_sequences,
        "violations": all_violations,
        "violation_frames": violation_frames,
        "frame_expressions": frame_expressions,
        "smoothness_scores": smoothness_scores,
        "mean_smoothness": mean_smoothness,
        "expression_score": expression_score,
    }


# ── 4. 可视化 ───────────────────────────────────────────


def save_visualization(
    frames: list[np.ndarray],
    au_per_frame: list[dict[str, float]],
    analysis: dict,
    output_dir: str,
) -> None:
    """保存带 AU 强度和表情标签标注的关键帧。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    violation_set = set(analysis.get("violation_frames", []))
    frame_expressions = analysis.get("frame_expressions", [])
    expr_cn = {
        "genuine_smile": "真笑", "surprise": "惊讶",
        "frown": "皱眉", "fear": "恐惧", "neutral": "中性",
    }

    # 选择要保存的帧: 首帧 + 尾帧 + 冲突帧 + 均匀采样
    key_indices = {0, len(frames) - 1}
    key_indices.update(violation_set)

    step = max(1, len(frames) // 20)
    for i in range(0, len(frames), step):
        key_indices.add(i)

    for idx in sorted(key_indices):
        if idx >= len(frames):
            continue
        vis = frames[idx].copy()
        aus = au_per_frame[idx]
        is_violation = idx in violation_set

        # 标注顶部状态
        border_color = (0, 0, 255) if is_violation else (0, 200, 0)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 3), border_color, -1)

        # 标注激活的 AU
        y_offset = 25
        active_aus = {k: v for k, v in sorted(aus.items()) if v >= AU_ACTIVATION_THRESHOLD}
        for au_name, au_val in active_aus.items():
            # 强度映射颜色 (绿 → 黄 → 红)
            intensity = min(au_val / 5.0, 1.0)
            color = (
                0,
                int(255 * (1 - intensity)),
                int(255 * intensity),
            )
            text = f"{au_name}={au_val:.1f}"
            cv2.putText(vis, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 18

        if is_violation:
            cv2.putText(vis, "CONFLICT!", (10, vis.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 标注表情类型（右上角）
        if idx < len(frame_expressions):
            labels = frame_expressions[idx]
            # OpenCV putText 不支持中文，使用英文标签
            expr_text = " | ".join(labels)
            # 背景框提高可读性
            (tw, th), _ = cv2.getTextSize(expr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            rx = vis.shape[1] - tw - 15
            cv2.rectangle(vis, (rx - 5, 5), (vis.shape[1] - 5, th + 15), (0, 0, 0), -1)
            cv2.putText(vis, expr_text, (rx, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        tag = "CONFLICT_" if is_violation else ""
        cv2.imwrite(str(out / f"{tag}frame_{idx:04d}.jpg"), vis)

    print(f"\n可视化已保存到 {out}/ ({len(key_indices)} 张)")


# ── 5. 主入口 ───────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="表情自然度调试脚本")
    parser.add_argument("--input", required=True, help="视频文件路径")
    parser.add_argument("--device", default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument("--sample-rate", type=int, default=1, help="每 N 帧采样 1 帧")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化到 outputs/expression/")
    args = parser.parse_args()

    video_path = args.input
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在 {video_path}")
        sys.exit(1)

    print(f"{'='*50}")
    print(f"表情自然度 (Expression Naturalness) 分析")
    print(f"{'='*50}")
    print(f"视频: {video_path}")
    print(f"设备: {args.device}")

    # Step 1: 读帧
    frames, fps = load_frames(video_path, args.sample_rate)
    if len(frames) < 2:
        print("错误: 视频帧数不足")
        sys.exit(1)

    # Step 2: Py-Feat 提取 AU
    au_per_frame = extract_aus(frames)

    # Step 3: AU 分析 (组合冲突 + 时序平滑 + 评分)
    analysis = analyze_aus(au_per_frame)

    if not analysis:
        sys.exit(0)

    # Step 4: 可视化
    if args.save_vis:
        out_dir = str(ROOT / "outputs" / "expression")
        save_visualization(frames, au_per_frame, analysis, out_dir)

    print(f"\n完成。")


if __name__ == "__main__":
    main()
