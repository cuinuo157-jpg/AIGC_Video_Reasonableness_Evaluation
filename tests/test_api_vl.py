import os
from dashscope import MultiModalConversation
import dashscope

from src.mllm.dotenv_loader import load_dotenv

load_dotenv()

# 各地域配置不同，请根据实际地域修改
dashscope.base_http_api_url = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)

# 将路径替换为你本地视频的绝对路径
local_path = "../data/videos/The camera orbits around. Airpods Max, the camera circles around.-0.mp4"
video_path = f"file://{local_path}"
messages = [
    {
        "role": "user",
        "content": [
            {"video": video_path, "fps": 2},
            {"text": "分析该AI视频是否存在异常，如有，阐述异常现象"},
        ],
    }
]
response = MultiModalConversation.call(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model=os.environ.get("MLLM_MODEL", "qwen3-vl-8b-thinking"),
    messages=messages,
)
print(response.output.choices[0].message.content[0]["text"])