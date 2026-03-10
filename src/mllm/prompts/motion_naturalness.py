MOTION_NATURALNESS_PROMPT = """分析这段视频中物体/人物的运动是否自然合理。

请检查以下方面：
1. 运动是否流畅连贯，有无突然的加速/减速/方向变化
2. 人物动作是否符合人体运动学（关节活动范围、肌肉协调性）
3. 物体运动轨迹是否符合惯性和动力学规律
4. 有无"运动扭捏"现象（不自然的摆动、抽搐、僵硬）

请以 JSON 格式回答：
{
    "is_natural": true/false,
    "confidence": 0.0-1.0,
    "issues": ["问题描述1", "问题描述2"],
    "severity": "none" / "mild" / "moderate" / "severe"
}"""
