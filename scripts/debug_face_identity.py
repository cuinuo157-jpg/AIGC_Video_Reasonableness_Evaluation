"""
人脸身份一致性 (CSIM) 单模块调试脚本

用法:
    python scripts/debug_face_identity.py --input <视频路径>
    python scripts/debug_face_identity.py --input <视频路径> --device cpu
    python scripts/debug_face_identity.py --input <视频路径> --sample-rate 5 --save-vis

参数:
    --input        视频文件路径
    --device       推理设备 (cuda / cpu)，默认 cuda
    --sample-rate  每 N 帧采样一帧，默认 1（全帧）
    --save-vis     保存可视化图片到 outputs/face_identity/
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

from src.face_identity.analyzer import FaceIdentityAnalyzer, FaceIdentityResult
from src.face_identity.config import FaceIdentityConfig
from src.face_identity.csim_scorer import CSIMScorer
from src.face_identity.face_tracker import FaceTracker


# ── 1. 视频读取 + 人脸嵌入提取 ──────────────────────────


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


def extract_embeddings(
    frames: list[np.ndarray], device: str
) -> list[dict]:
    """用 InsightFace 提取每帧人脸嵌入。"""
    from insightface.app import FaceAnalysis

    print(f"\n加载 InsightFace 模型 (device={device})...")
    t0 = time.time()

    ctx_id = 0 if "cuda" in device else -1
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ctx_id >= 0
        else ["CPUExecutionProvider"]
    )
    insightface_root = str(ROOT / "third_party" / "insightface")
    app = FaceAnalysis(name="buffalo_l", root=insightface_root, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print(f"模型加载完成 ({time.time() - t0:.1f}s)")

    print(f"\n逐帧提取人脸嵌入...")
    results = []
    t0 = time.time()
    for i, frame in enumerate(frames):
        faces_raw = app.get(frame)
        faces = []
        for f in faces_raw:
            emb = f.embedding / np.linalg.norm(f.embedding)
            faces.append({
                "embedding": emb,
                "bbox": f.bbox.tolist(),
                "det_score": float(f.det_score),
            })
        results.append({"faces": faces, "num_faces": len(faces)})

        n = len(faces)
        tag = f"({n} 张脸)" if n > 0 else "(无人脸)"
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"  [{i+1}/{len(frames)}] {tag}")

    elapsed = time.time() - t0
    print(f"嵌入提取完成 ({elapsed:.1f}s, {elapsed/len(frames):.2f}s/帧)")
    return results


# ── 2. CSIM 分析 ────────────────────────────────────────


def run_csim_analysis(
    frame_data: list[dict],
    config: FaceIdentityConfig | None = None,
) -> FaceIdentityResult | None:
    """执行跨帧追踪 + CSIM 评分。"""
    total_faces = sum(fd["num_faces"] for fd in frame_data)
    face_frames = sum(1 for fd in frame_data if fd["num_faces"] > 0)

    print(f"\n{'='*50}")
    print(f"人脸检测统计:")
    print(f"  总帧数: {len(frame_data)}")
    print(f"  含人脸帧: {face_frames} ({face_frames/len(frame_data)*100:.1f}%)")
    print(f"  检测总人脸数: {total_faces}")

    if total_faces == 0:
        print("\n未检测到任何人脸，跳过 CSIM 分析。")
        return None

    # 跨帧追踪
    tracker = FaceTracker(match_threshold=0.4)
    tracks = tracker.track(frame_data)
    print(f"\n追踪结果: {len(tracks)} 条人脸轨迹")
    for t in tracks:
        print(f"  Track #{t.track_id}: {len(t.embeddings)} 帧 "
              f"(帧范围 {t.frame_indices[0]}~{t.frame_indices[-1]})")

    # 取最长轨迹做 CSIM
    main_track = max(tracks, key=lambda t: len(t.embeddings))
    print(f"\n主轨迹: Track #{main_track.track_id} ({len(main_track.embeddings)} 帧)")

    cfg = config or FaceIdentityConfig()
    scorer = CSIMScorer(cfg)
    csim = scorer.compute(main_track.embeddings)

    # 输出
    print(f"\n{'='*50}")
    print(f"CSIM 评分结果:")
    print(f"{'='*50}")
    print(f"  CSIM-Ref (全局身份保持):  {csim.csim_ref:.4f}")
    print(f"  CSIM-Adj (相邻帧平滑):    {csim.csim_adj:.4f}")
    print(f"  CSIM-Min (最差帧对):       {csim.csim_min:.4f}")
    print(f"  Drop 事件数:               {len(csim.drop_events)}")
    print(f"  ─────────────────────────")
    print(f"  身份一致性综合分:          {csim.identity_score:.4f}")
    print()

    # 评级
    s = csim.identity_score
    if s >= 0.85:
        grade = "优秀 - 身份高度一致"
    elif s >= 0.70:
        grade = "良好 - 轻微波动"
    elif s >= 0.50:
        grade = "一般 - 存在明显身份变化"
    else:
        grade = "差 - 身份不一致严重"
    print(f"  评级: {grade}")

    if csim.drop_events:
        print(f"\n身份突变事件:")
        for d in csim.drop_events:
            print(f"  帧 {d.frame_idx}: "
                  f"相似度 {d.similarity_before:.3f} → {d.similarity_after:.3f} "
                  f"(跌落 {d.drop_magnitude:.3f})")

    return FaceIdentityResult(
        applicable=True,
        face_tracks=tracks,
        csim_ref=csim.csim_ref,
        csim_adj=csim.csim_adj,
        csim_min=csim.csim_min,
        drop_events=csim.drop_events,
        identity_score=csim.identity_score,
    )


# ── 3. 可视化 ───────────────────────────────────────────


def save_visualization(
    frames: list[np.ndarray],
    frame_data: list[dict],
    result: FaceIdentityResult,
    output_dir: str,
) -> None:
    """保存带人脸框和相似度标注的关键帧。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 计算相邻帧相似度用于标注
    main_track = max(result.face_tracks, key=lambda t: len(t.embeddings))
    adj_sims = []
    for i in range(len(main_track.embeddings) - 1):
        sim = float(np.dot(main_track.embeddings[i], main_track.embeddings[i + 1]))
        adj_sims.append(sim)

    # 选择要保存的帧: 首帧 + 尾帧 + drop 帧 + 均匀采样
    key_indices = {0, len(frames) - 1}
    for d in result.drop_events:
        if d.frame_idx < len(frames):
            key_indices.add(d.frame_idx)
            key_indices.add(max(0, d.frame_idx - 1))

    # 均匀补充到最多 20 帧
    step = max(1, len(frames) // 20)
    for i in range(0, len(frames), step):
        key_indices.add(i)

    drop_frame_set = {d.frame_idx for d in result.drop_events}

    for idx in sorted(key_indices):
        if idx >= len(frames):
            continue
        vis = frames[idx].copy()
        fd = frame_data[idx]

        for face in fd["faces"]:
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            is_drop = idx in drop_frame_set
            color = (0, 0, 255) if is_drop else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"det:{face['det_score']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 标注相邻帧相似度
        track_idx_map = {fi: i for i, fi in enumerate(main_track.frame_indices)}
        if idx in track_idx_map:
            ti = track_idx_map[idx]
            if ti > 0 and ti - 1 < len(adj_sims):
                sim = adj_sims[ti - 1]
                sim_color = (0, 255, 0) if sim > 0.7 else (0, 165, 255) if sim > 0.5 else (0, 0, 255)
                cv2.putText(vis, f"sim:{sim:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, sim_color, 2)

        tag = "DROP_" if idx in drop_frame_set else ""
        cv2.imwrite(str(out / f"{tag}frame_{idx:04d}.jpg"), vis)

    print(f"\n可视化已保存到 {out}/ ({len(key_indices)} 张)")


# ── 4. 主入口 ───────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="人脸身份一致性 CSIM 调试脚本")
    parser.add_argument("--input", required=True, help="视频文件路径")
    parser.add_argument("--device", default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument("--sample-rate", type=int, default=1, help="每 N 帧采样 1 帧")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化到 outputs/face_identity/")
    parser.add_argument("--drop-threshold", type=float, default=0.3, help="Drop 检测阈值")
    args = parser.parse_args()

    video_path = args.input
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在 {video_path}")
        sys.exit(1)

    print(f"{'='*50}")
    print(f"人脸身份一致性 (CSIM) 分析")
    print(f"{'='*50}")
    print(f"视频: {video_path}")
    print(f"设备: {args.device}")

    # Step 1: 读帧
    frames, fps = load_frames(video_path, args.sample_rate)
    if len(frames) < 2:
        print("错误: 视频帧数不足")
        sys.exit(1)

    # Step 2: InsightFace 提取嵌入
    frame_data = extract_embeddings(frames, args.device)

    # Step 3: CSIM 分析
    config = FaceIdentityConfig(drop_threshold=args.drop_threshold)
    result = run_csim_analysis(frame_data, config)

    if result is None:
        sys.exit(0)

    # Step 4: 可视化
    if args.save_vis:
        out_dir = str(ROOT / "outputs" / "face_identity")
        save_visualization(frames, frame_data, result, out_dir)

    print(f"\n完成。")


if __name__ == "__main__":
    main()
