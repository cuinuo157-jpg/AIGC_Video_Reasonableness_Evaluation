"""MLLM Prompt 模板 — 人体结构异常判定。"""

BIOLOGICAL_ANOMALY_PROMPT = """分析这组连续帧中人体局部区域是否存在 AIGC 生成异常。

这些图像是从 AI 生成视频中裁剪的人体局部区域（手部或嘴部），请仔细检查以下异常：

1. **手指融合**：两根或多根手指粘连在一起、边界模糊不清
2. **手指数量异常**：手指数量不是 5 根（多余或缺少）
3. **手指结构畸形**：手指弯折方向不自然、关节位置错误
4. **牙齿/舌头突然消失**：前一帧可见的牙齿或舌头在后续帧中突然消失
5. **嘴部附着物突变**：嘴上的食物残渣、口红等突然消失或出现
6. **面部结构扭曲**：嘴部、下巴区域出现不自然的形变或融合

请以 JSON 格式回答：
{
    "has_anomalies": true/false,
    "confidence": 0.0-1.0,
    "anomalies": [
        {
            "type": "finger_fusion/extra_finger/missing_finger/finger_deform/teeth_vanish/tongue_vanish/mouth_artifact/face_distortion",
            "description": "具体描述",
            "severity": "mild/moderate/severe"
        }
    ]
}"""
