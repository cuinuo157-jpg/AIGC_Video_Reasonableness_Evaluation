from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np

from .config import MLLMConfig


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
        raise ValueError(f"Unknown provider: {self.config.api_provider}")

    def _call_local(self, frames: list[np.ndarray], prompt: str) -> dict:
        raise NotImplementedError("Local MLLM not yet integrated")

    def _encode_frames(self, frames: list[np.ndarray]) -> list[str]:
        step = max(1, len(frames) // self.config.max_frames)
        sampled = frames[::step][: self.config.max_frames]
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
