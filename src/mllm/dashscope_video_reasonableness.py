"""阿里云 DashScope 视频 VLM 评测辅助函数（无 dashscope 依赖，便于单测）。"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_SYSTEM_PROMPT = """你是 AIGC 生成视频质量评估专家。请根据用户提供的视频帧序列，从「物理与运动合理性」「时序连贯性」「主体与背景是否自然」等角度判断视频是否像真实或合理的合成内容。
你必须严格按下面 JSON 模式输出，不要输出 JSON 以外的多余文字（不要 markdown 代码块）:
{"overall_score": <1-10的整数>, "is_reasonable": <true或false>, "confidence": <0-1的小数>, "issues": [<字符串列表，具体问题>], "summary": <一句话总结>}"""


def extract_frame_paths(video_path: Path, max_frames: int = 16) -> tuple[list[str], str]:
    """均匀采样帧并写入临时目录，返回 (绝对路径字符串列表, 临时目录路径供调用方清理)。"""
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    n = len(vr)
    if n == 0:
        raise ValueError(f"视频无有效帧: {video_path}")
    if n <= max_frames:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n - 1, max_frames, dtype=int).tolist()
    tmp = tempfile.mkdtemp(prefix="dashscope_vl_frames_")
    paths: list[str] = []
    try:
        for i, idx in enumerate(indices):
            frame = vr[idx].asnumpy()
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Unexpected frame shape: {frame.shape}")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fp = Path(tmp) / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(fp), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            paths.append(str(fp.resolve()))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return paths, tmp


def build_user_text(task_prompt: str, context: str | None) -> str:
    parts = [task_prompt.strip()]
    if context and context.strip():
        parts.append("【生成意图 / 参考描述（可能为空）】\n" + context.strip())
    parts.append("请只输出一行合法 JSON 对象。")
    return "\n\n".join(parts)


def parse_json_from_model_text(text: str) -> dict[str, Any]:
    """从模型输出中尽量解析 JSON（允许前后有少量杂质）。"""
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析为 JSON，模型原文前 500 字: {text[:500]!r}")


def configure_dashscope(dashscope: Any, base_url: str | None) -> None:
    """设置 DashScope HTTP base（可选，如国际区 endpoint）。"""
    if base_url:
        normalized = base_url.rstrip("/")
        # MultiModalConversation 走 DashScope SDK 原生接口，不是 OpenAI compatible-mode。
        # 若误配 compatible-mode，会在本地文件上传取证书阶段失败（UploadFileException）。
        if "/compatible-mode/" in normalized:
            normalized = normalized.replace("/compatible-mode", "")
        dashscope.base_http_api_url = normalized


def call_vlm(
    *,
    model: str,
    api_key: str,
    frame_paths: list[str],
    system_prompt: str,
    user_text: str,
    stream: bool,
    dashscope: Any,
    MultiModalConversation: Any,
) -> str:
    """调用 MultiModalConversation，返回模型文本（非流式为完整字符串）。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"video": frame_paths},
                {"text": user_text},
            ],
        },
    ]
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "messages": messages,
        "incremental_output": stream,
        "stream": stream,
    }
    effective_stream = stream

    def _requires_incremental_output(err_text: str) -> bool:
        needle = "only supports incremental_output set to true"
        return needle in err_text.lower()

    response = MultiModalConversation.call(**kwargs)
    if (
        not effective_stream
        and getattr(response, "status_code", None) != 200
        and _requires_incremental_output(
            str(getattr(response, "message", "") or getattr(response, "code", "") or response)
        )
    ):
        # 部分模型（如 qwen3-vl-8b-thinking）强制 incremental_output=True。
        kwargs["incremental_output"] = True
        kwargs["stream"] = True
        response = MultiModalConversation.call(**kwargs)
        effective_stream = True

    if effective_stream:
        chunks: list[str] = []
        for chunk in response:
            if chunk.status_code != 200:
                raise RuntimeError(
                    getattr(chunk, "message", None) or getattr(chunk, "code", None) or str(chunk)
                )
            out = getattr(chunk, "output", None) or {}
            choices = out.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                for part in msg.get("content") or []:
                    if isinstance(part, dict) and "text" in part:
                        chunks.append(part["text"])
        return "".join(chunks)

    if response.status_code != 200:
        raise RuntimeError(
            getattr(response, "message", None) or getattr(response, "code", None) or str(response)
        )
    out = response.output
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
                return "".join(texts)
    return str(out)
