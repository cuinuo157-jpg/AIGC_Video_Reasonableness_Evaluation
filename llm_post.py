"""
轻量级 LLM 请求脚本 - 直接封装 requests.post()
用于简化与大模型的交互
"""

import json
import base64
import requests

# LLM 服务地址
MODEL_URL = "http://aitest-beta.rnd.huawei.com/v1"

# 支持的模型列表
SUPPORTED_MODELS = [
    "Qwen2.5-72B",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
    "DeepSeek-V3",
    "Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-VL-32B-Instruct",
    "Qwen3-VL-235B-A22B-Instruct",
]


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def post_llm(prompt_text, image_paths=None, model="Qwen2.5-VL-32B-Instruct", timeout=60):
    """
    简单的 LLM 请求封装
    
    Args:
        prompt_text: 提示词（字符串或文件路径）
        image_paths: 图片路径列表（可选）
        model: 模型名称，默认使用 Qwen2.5-VL-32B-Instruct
        timeout: 请求超时时间（秒）
    
    Returns:
        dict: 包含 code、message 和 request_id 的响应
    
    Example:
        # 简单调用
        result = post_llm("描述这张图片", image_paths=["img.jpg"])
        
        # 使用提示词文件
        result = post_llm("D:/prompt.txt", image_paths=["img1.jpg", "img2.jpg"])
        
        # 指定模型
        result = post_llm("分析内容", model="Qwen3-VL-32B-Instruct")
    """
    # 验证模型
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型：{model}。支持的模型：{SUPPORTED_MODELS}")
    
    # 如果是文件路径则读取文本
    if isinstance(prompt_text, str) and prompt_text.endswith(('.txt', '.prt')):
        with open(prompt_text, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    
    # 构建 user_content
    user_content = []
    
    # 添加图片（如果提供）
    if image_paths:
        for img_path in image_paths:
            base64_img = encode_image_to_base64(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": base64_img}
            })
    
    # 添加文本提示
    user_content.append({
        "type": "text",
        "text": prompt_text
    })
    
    # 构建请求
    import uuid
    request_payload = {
        "model_type": model,
        "user_content": user_content,
        "system_content": "你是一位专业的 AI 助手",
        "service_name": "simple_client",  # necessary
        "request_id": str(uuid.uuid4()).replace('-', ''),  # necessary
    }
    
    # 发送请求
    response = requests.post(
        MODEL_URL,
        json=request_payload,
        headers={'Content-Type': 'application/json'},
        timeout=timeout
    )
    response.raise_for_status()
    
    # 解析响应
    result = response.json()
    return result


if __name__ == "__main__":
    # 测试示例：使用 qwen3-vl-235b 理解图像内容
    try:
        img_path = r"ScreenShot_20260512193810.PNG"
        result = post_llm(
            prompt_text="请详细描述这张图片的内容",
            image_paths=[img_path],
            model="Qwen3-VL-235B-A22B-Instruct"
        )
        print(f"响应：{json.dumps(result, ensure_ascii=False, indent=2)}")
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
    except Exception as e:
        print(f"发生错误：{e}")

    # 文本模型调用示例（使用纯文本模型，不支持图片）
    # result = post_llm(
    #     prompt_text="请解释什么是机器学习和深度学习，它们之间的关系是什么",
    #     model="Qwen2.5-72B"  # 纯文本模型
    # )
    # print(f"文本模型响应：{json.dumps(result, ensure_ascii=False, indent=2)}")
