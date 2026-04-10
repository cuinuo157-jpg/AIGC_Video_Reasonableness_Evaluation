#!/usr/bin/env python3
import os
import cv2
import base64
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_frames(video_path: str, fps: int = 2) -> list[bytes]:
    """按指定 fps 从视频抽帧，返回 JPEG 字节列表"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25  # 兜底值
    frame_interval = int(original_fps / fps)  # 每隔 N 帧取 1 帧
    
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # 编码为 JPEG 字节（质量 85 平衡清晰度与体积）
            success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                frames.append(buf.tobytes())
        frame_count += 1
    cap.release()
    logger.info(f"抽帧完成: {len(frames)} 帧 (原始 {frame_count} 帧, fps={fps})")
    return frames

def frames_to_base64(frames: list[bytes]) -> list[str]:
    """JPEG 字节列表 → base64 字符串列表"""
    return [base64.b64encode(f).decode('utf-8') for f in frames]

def save_frames_debug(frames: list[bytes], output_base_dir: str, video_path: str, timestamp: str) -> str:
    """
    保存抽帧结果到指定目录
    返回实际保存路径，便于日志记录
    """
    video_name = Path(video_path).stem  # 去掉扩展名
    # 创建时间戳 + 视频名的子目录，避免冲突
    save_dir = Path(output_base_dir) / f"{timestamp}_{video_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame_bytes in enumerate(frames):
        save_path = save_dir / f"frame_{i:03d}.jpg"
        with open(save_path, 'wb') as f:
            f.write(frame_bytes)
    
    logger.info(f"💾 关键帧已保存: {save_dir} ({len(frames)} 帧)")
    return str(save_dir)

def analyze_video_by_frames(
    video_path: str,
    question: str,
    fps: int = 2,
    model: str = "qwen3.5:9b",
    max_frames: int = 5,
    save_frames: bool = True,  # 控制是否保存
    frames_output_dir: str = "/data/AIGC_Video_Reasonableness_Evaluation/tests/video_frames"
) -> str | None:
    """
    通过抽帧图像列表分析视频内容（openai 兼容接口）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 抽帧 + 编码
    try:
        frames = extract_frames(video_path, fps)
        if not frames:
            logger.error("未抽到任何帧")
            return None
        
        # 2. 限制最大帧数（均匀采样）
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]
            logger.info(f"帧数裁剪: {len(frames)} 帧 (上限 {max_frames})")
        
        # 3. 【新增】保存关键帧到本地（在 base64 编码前保存，避免重复解码）
        if save_frames:
            save_frames_debug(frames, frames_output_dir, video_path, timestamp)
        
        b64_frames = frames_to_base64(frames)
    except Exception as e:
        logger.error(f"预处理失败: {e}")
        return None

    # 4. 构造 openai 兼容消息
    content = []
    for b64 in b64_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": question})
    
    messages = [{"role": "user", "content": content}]

    # 5. 调用本地服务
    client = OpenAI(
        base_url="http://localhost:8201/v1",
        api_key="not-needed"  # 本地服务绕过校验
    )
    
    try:
        logger.info(f"发起推理请求 (模型: {model}, 帧数: {len(b64_frames)})")
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=300  # 多图推理耗时较长
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"推理失败: {e}")
        return None

if __name__ == "__main__":
    # 👇 替换为你的实际视频路径
    video_path = "/data/AIGC_Video_Reasonableness_Evaluation/data/videos/The camera orbits around. Airpods Max, the camera circles around.-0.mp4"
    
    # 👇 针对「相机环绕 + 物体一致性」任务的优化 prompt
    question = """请分析AI生成的视频内容，若发现异常请明确指出。"""
    
    result = analyze_video_by_frames(
        video_path=video_path,
        question=question,
        fps=3,          # 环绕运动建议稍高抽帧率
        max_frames=5,   # 适配后端 5 帧限制
        save_frames=True,  # 👈 启用保存功能
        frames_output_dir="/data/AIGC_Video_Reasonableness_Evaluation/tests/video_frames"
    )
    
    if result:
        print("\n=== 模型返回 ===")
        print(result)
    else:
        print("❌ 分析失败，请检查日志")
