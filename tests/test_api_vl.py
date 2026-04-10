import os
from dashscope import MultiModalConversation
import dashscope 

# 各地域配置不同，请根据实际地域修改
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

# 将xxxx/test.mp4替换为你本地视频的绝对路径
local_path = "../data/videos/The camera orbits around. Airpods Max, the camera circles around.-0.mp4"
video_path = f"file://{local_path}"
messages = [
                {'role':'user',
                # fps参数控制视频抽帧数量，表示每隔1/fps 秒抽取一帧
                'content': [{'video': video_path,"fps":2},
                            {'text': '分析该AI视频是否存在异常，如有，阐述异常现象'}]}]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen3-vl-8b-thinking',  
    messages=messages)
print(response.output.choices[0].message.content[0]["text"])