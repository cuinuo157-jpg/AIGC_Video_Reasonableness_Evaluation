#!/usr/bin/env python3
"""
VLM (Vision Language Model) 图片理解 API 调用脚本 —— OpenAI 兼容接口

适配研发环境大模型平台：
  - 域名访问: https://console-mlops.hwcloudtest.cn
  - IP 访问:   http://10.34.239.193:8989

密钥格式自动检测:
  - sk-* 开头 → 使用 /v1 端点，model 填令牌调用模型名称
  - 纯大写字母+数字 → 使用 /v3 端点，model 填镜像环境模型名称@版本号
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL_DOMAIN = "https://console-mlops.hwcloudtest.cn"
DEFAULT_BASE_URL_IP = "http://10.34.239.193:8989"
DEFAULT_MODEL = "qwen3vl_8b_wf_jbw"


# ---------------------------------------------------------------------------
# 密钥 / 端点检测
# ---------------------------------------------------------------------------

def detect_api_version(api_key: str) -> str:
    """根据密钥格式自动检测应使用的 API 版本。"""
    if not api_key:
        raise ValueError("API key 不能为空")
    if api_key.startswith("sk-"):
        return "v1"
    # 纯大写字母+数字 → v3
    if api_key.replace("-", "").isalnum() and not api_key.startswith("sk-"):
        return "v3"
    # 兜底：未知格式假定 v1
    print(f"[warn] 无法识别密钥格式，默认使用 /v1 端点", file=sys.stderr)
    return "v1"


def build_endpoint(base_url: str, api_version: str) -> str:
    """构建 OpenAI 兼容的 chat completions 端点。"""
    base = base_url.rstrip("/")
    return f"{base}/{api_version}/chat/completions"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def encode_image_b64(image_path: str) -> str:
    """将本地图片编码为 base64 data URL。"""
    path = Path(image_path)
    ext = path.suffix.lower().lstrip(".")
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# VLM Client
# ---------------------------------------------------------------------------

@dataclass
class VLMClient:
    """OpenAI 兼容的 VLM API 客户端。"""

    api_key: str = field(default_factory=lambda: os.environ.get("VLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("VLM_BASE_URL", DEFAULT_BASE_URL_DOMAIN))
    model: str = field(default_factory=lambda: os.environ.get("VLM_MODEL", DEFAULT_MODEL))
    temperature: float = 0.7
    max_tokens: int = 2048

    # ---- 内部自动检测 ----
    _api_version: str = field(init=False, default="")
    _endpoint: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self._api_version = detect_api_version(self.api_key)
        self._endpoint = build_endpoint(self.base_url, self._api_version)

    # ---- 公共属性 ----
    @property
    def api_version(self) -> str:
        return self._api_version

    @property
    def endpoint(self) -> str:
        return self._endpoint

    # ---- 消息构建 ----

    @staticmethod
    def text_message(role: str, text: str) -> Dict[str, Any]:
        """构建纯文本消息。"""
        return {"role": role, "content": text}

    @staticmethod
    def image_message(role: str, text: str, image_path: str) -> Dict[str, Any]:
        """构建图片+文本消息。"""
        return {
            "role": role,
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": encode_image_b64(image_path)}},
            ],
        }

    @staticmethod
    def multi_image_message(role: str, text: str, image_paths: List[str]) -> Dict[str, Any]:
        """构建多图片+文本消息。"""
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": encode_image_b64(p)}})
        return {"role": role, "content": content}

    # ---- API 调用 ----

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """发送 chat completion 请求。

        Args:
            messages: 消息列表
            temperature: 温度参数（默认 0.7）
            max_tokens: 最大输出 token 数（默认 2048）
            stream: 是否流式输出
            **kwargs: 其他 OpenAI 兼容参数

        Returns:
            API 响应 JSON
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }
        payload.update(kwargs)

        resp = requests.post(self._endpoint, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()

        if stream:
            return self._handle_stream(resp)
        return resp.json()

    def chat_simple(self, prompt: str) -> str:
        """发送纯文本对话，返回回复文本。"""
        resp = self.chat([self.text_message("user", prompt)])
        return resp["choices"][0]["message"]["content"]

    def chat_with_image(self, prompt: str, image_path: str) -> str:
        """发送图片+文本对话，返回回复文本。"""
        resp = self.chat([self.image_message("user", prompt, image_path)])
        return resp["choices"][0]["message"]["content"]

    # ---- 内部 ----

    @staticmethod
    def _handle_stream(response: requests.Response) -> Dict[str, Any]:
        """处理流式响应，收集完整结果。"""
        collected_content: List[str] = []
        finish_reason = None
        usage = None

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # 去掉 "data: " 前缀
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content:
                    collected_content.append(content)
                    print(content, end="", flush=True)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
            except json.JSONDecodeError:
                continue

        print()  # 换行
        return {
            "choices": [{"message": {"role": "assistant", "content": "".join(collected_content)}}],
            "usage": usage or {},
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM API 调用脚本（OpenAI 兼容接口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 纯文本对话
  python vlm_client.py --prompt "描述一下这张图片的内容" --image ./test.jpg

  # 多图片
  python vlm_client.py --prompt "比较这两张图" --images ./a.jpg ./b.jpg

  # 使用 IP 地址访问
  python vlm_client.py --base-url http://10.34.239.193:8989 --prompt "hello"

  # 纯文本
  python vlm_client.py --prompt "你好，你是谁？"
        """,
    )

    # ---- 连接参数 ----
    parser.add_argument("--api-key", default=None, help="API 密钥（或通过 VLM_API_KEY 环境变量设置）")
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"API 基础地址（默认: {DEFAULT_BASE_URL_DOMAIN}）",
    )
    parser.add_argument("--model", default=None, help=f"模型名称（默认: {DEFAULT_MODEL}）")

    # ---- 输入 ----
    parser.add_argument("--prompt", "-p", required=True, help="提示词")
    parser.add_argument("--image", "-i", default=None, help="单张图片路径")
    parser.add_argument("--images", nargs="+", default=None, help="多张图片路径")

    # ---- 生成参数 ----
    parser.add_argument("--temperature", type=float, default=0.7, help="温度（默认 0.7）")
    parser.add_argument("--max-tokens", type=int, default=2048, help="最大输出 token 数")
    parser.add_argument("--stream", action="store_true", help="启用流式输出")
    parser.add_argument("--raw", action="store_true", help="输出原始 JSON")

    args = parser.parse_args()

    # ---- 实例化客户端 ----
    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("[error] 未提供 API 密钥，请用 --api-key 或环境变量 VLM_API_KEY 设置", file=sys.stderr)
        sys.exit(1)

    base_url = args.base_url or os.environ.get("VLM_BASE_URL", DEFAULT_BASE_URL_DOMAIN)
    model = args.model or os.environ.get("VLM_MODEL", DEFAULT_MODEL)

    client = VLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"[info] 端点: {client.endpoint}", file=sys.stderr)
    print(f"[info] 模型: {model}  |  API 版本: {client.api_version}", file=sys.stderr)

    try:
        if args.images:
            content = client.chat(
                [client.multi_image_message("user", args.prompt, args.images)],
                stream=args.stream,
            )
            if not args.stream:
                content = content["choices"][0]["message"]["content"]
        elif args.image:
            content = client.chat_with_image(args.prompt, args.image)
        else:
            if args.stream:
                content = client.chat([client.text_message("user", args.prompt)], stream=True)
            else:
                content = client.chat_simple(args.prompt)

        if args.stream:
            # 流式输出已在 _handle_stream 中实时打印
            pass
        elif args.raw:
            print(json.dumps(content, ensure_ascii=False, indent=2))
        else:
            print(content)

    except requests.HTTPError as e:
        print(f"[error] HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"[error] 请求失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
