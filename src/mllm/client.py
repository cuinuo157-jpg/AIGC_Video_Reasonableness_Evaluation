from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from .config import MLLMConfig
from .dashscope_video_reasonableness import parse_json_from_model_text


class MLLMClient:
    """统一 MLLM 调用接口，支持本地模型和 API 切换。"""

    def __init__(self, config: MLLMConfig) -> None:
        self.config = config
        self._local_model = None

    def judge_video_clip(self, frames: list[np.ndarray], prompt: str) -> dict:
        if self.config.backend == "local":
            return self._call_local(frames, prompt)
        elif self.config.backend == "api":
            return self._call_api(frames, prompt)
        else:  # hybrid
            return self.judge_with_fallback(frames, prompt)

    def judge_with_fallback(self, frames: list[np.ndarray], prompt: str) -> dict:
        try:
            return self._call_local(frames, prompt)
        except Exception:
            return self._call_api(frames, prompt)

    def _call_api(self, frames: list[np.ndarray], prompt: str) -> dict:
        images_b64 = self._encode_frames(frames)
        if self.config.api_provider == "openai":
            return self._call_openai(images_b64, prompt)
        elif self.config.api_provider == "anthropic":
            return self._call_anthropic(images_b64, prompt)
        elif self.config.api_provider == "dashscope":
            raise ValueError(
                "DashScope 视频模型需要视频路径输入，请使用 judge_video_path(video_path, prompt, fps=2)。"
            )
        elif self.config.api_provider == "vllm":
            return self._call_vllm_openai(images_b64, prompt)
        raise ValueError(f"Unknown provider: {self.config.api_provider}")

    def judge_video_path(self, video_path: str, prompt: str, *, fps: int | None = None) -> dict:
        """DashScope：原生 video + fps；vllm：本地抽帧 + OpenAI 兼容 chat。"""
        if self.config.api_provider == "dashscope":
            f = self.config.dashscope_video_fps if fps is None else fps
            return self._call_dashscope_video(video_path=video_path, prompt=prompt, fps=f)
        if self.config.api_provider == "vllm":
            f = self.config.dashscope_video_fps if fps is None else fps
            return self._call_vllm_from_video_path(video_path, prompt, f)
        raise ValueError(
            "judge_video_path 仅适用于 api_provider='dashscope' 或 'vllm'"
        )

    def _call_local(self, frames: list[np.ndarray], prompt: str) -> dict:
        if self.config.local_model == "Qwen-VL":
            return self._call_qwen_vl_local(frames, prompt)
        raise NotImplementedError("Local MLLM not yet integrated")

    def _call_qwen_vl_local(self, frames: list[np.ndarray], prompt: str) -> dict:
        """Qwen-VL 本地模型推理接口（预留，尚未部署）。"""
        raise NotImplementedError(
            "Qwen-VL 本地模型尚未部署。请使用 backend='api' 或 backend='hybrid' 回退到 API。"
        )

    def _max_frames_for_encode(self) -> int:
        if self.config.api_provider == "vllm":
            return self.config.vllm_max_frames
        return self.config.max_frames

    def _encode_frames(self, frames: list[np.ndarray]) -> list[str]:
        cap = self._max_frames_for_encode()
        step = max(1, len(frames) // cap)
        sampled = frames[::step][:cap]
        encoded = []
        for f in sampled:
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            encoded.append(base64.b64encode(buf).decode("utf-8"))
        return encoded

    def _call_openai(self, images_b64: list[str], prompt: str) -> dict:
        import openai

        client = openai.OpenAI(
            api_key=self.config.api_key, base_url=self.config.api_base_url
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
            )
        resp = client.chat.completions.create(
            model=self.config.api_model,
            messages=[{"role": "user", "content": content}],
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    def _call_anthropic(self, images_b64: list[str], prompt: str) -> dict:
        import anthropic

        client = anthropic.Anthropic(api_key=self.config.api_key)
        content: list[dict[str, Any]] = []
        for img in images_b64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img,
                    },
                }
            )
        content.append({"type": "text", "text": prompt + "\n\n请以 JSON 格式回答。"})
        resp = client.messages.create(
            model=self.config.api_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return json.loads(resp.content[0].text)

    def _resolve_vllm_base_url(self) -> str:
        if self.config.api_base_url and self.config.api_base_url.strip():
            return self.config.api_base_url.strip().rstrip("/")
        return os.environ.get("VLLM_OPENAI_BASE_URL", "http://localhost:8201/v1").rstrip(
            "/"
        )

    def _resolve_vllm_api_key(self) -> str:
        if self.config.api_key and str(self.config.api_key).strip():
            return str(self.config.api_key).strip()
        return os.environ.get("VLLM_API_KEY", "not-needed")

    def _call_vllm_openai(self, images_b64: list[str], prompt: str) -> dict:
        from .vllm_openai_video import (
            build_user_content_image_then_text,
            chat_completions_text,
        )

        content = build_user_content_image_then_text(images_b64, prompt)
        text = chat_completions_text(
            base_url=self._resolve_vllm_base_url(),
            api_key=self._resolve_vllm_api_key(),
            model=self.config.api_model,
            user_content=content,
            temperature=self.config.temperature,
            timeout=float(self.config.vllm_timeout),
        )
        return parse_json_from_model_text(text)

    def _call_vllm_from_video_path(
        self, video_path: str, prompt: str, sample_fps: int
    ) -> dict:
        from .vllm_openai_video import (
            extract_frames_jpeg_bytes,
            subsample_uniform,
            frames_bytes_to_base64,
            build_user_content_image_then_text,
            chat_completions_text,
        )

        raw = extract_frames_jpeg_bytes(video_path, sample_fps)
        raw = subsample_uniform(raw, self.config.vllm_max_frames)
        if not raw:
            raise ValueError(f"未从视频抽到任何帧: {video_path}")
        b64 = frames_bytes_to_base64(raw)
        content = build_user_content_image_then_text(b64, prompt)
        text = chat_completions_text(
            base_url=self._resolve_vllm_base_url(),
            api_key=self._resolve_vllm_api_key(),
            model=self.config.api_model,
            user_content=content,
            temperature=self.config.temperature,
            timeout=float(self.config.vllm_timeout),
        )
        return parse_json_from_model_text(text)

    def _ensure_dashscope(self) -> tuple[Any, Any]:
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as e:
            raise ImportError(
                "使用 DashScope 需要安装依赖: pip install dashscope>=1.19.0"
            ) from e
        return dashscope, MultiModalConversation

    def _to_file_uri(self, video_path: str) -> str:
        if video_path.startswith("file://"):
            return video_path
        if "://" in video_path:
            return video_path
        p = Path(video_path)
        if p.is_absolute():
            normalized_abs = str(p).replace("\\", "/")
            # DashScope 对 Windows 本地文件更兼容 file://D:/...（而非 file:///D:/...）
            if re.match(r"^[A-Za-z]:/", normalized_abs):
                return f"file://{normalized_abs}"
            return f"file://{normalized_abs}"
        normalized = video_path.replace("\\", "/")
        return f"file://{normalized}"

    def _call_dashscope_video(self, video_path: str, prompt: str, fps: int = 2) -> dict:
        dashscope, MultiModalConversation = self._ensure_dashscope()
        if self.config.api_base_url:
            dashscope.base_http_api_url = self.config.api_base_url.rstrip("/")

        video_uri = self._to_file_uri(video_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": video_uri, "fps": fps},
                    {"text": prompt},
                ],
            }
        ]
        response = MultiModalConversation.call(
            api_key=self.config.api_key,
            model=self.config.api_model,
            messages=messages,
        )
        if getattr(response, "status_code", None) != 200:
            raise RuntimeError(
                getattr(response, "message", None)
                or getattr(response, "code", None)
                or str(response)
            )

        out = getattr(response, "output", None) or {}
        choices = out.get("choices") if isinstance(out, dict) else None
        if not choices:
            raise ValueError("DashScope 返回为空，未找到 choices")

        msg = choices[0].get("message") or {}
        content = msg.get("content")
        text = ""
        if isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and "text" in part
            )
        elif isinstance(content, str):
            text = content

        if not text:
            raise ValueError("DashScope 响应中未提取到文本内容")
        return parse_json_from_model_text(text)
