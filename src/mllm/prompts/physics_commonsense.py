PHYSICS_COMMONSENSE_PROMPT = """分析这段视频是否存在违反物理常识的现象。

请检查以下方面：
1. 重力异常：物体是否违反重力方向运动（上浮、悬空等）
2. 物体穿透：刚体是否相互穿过
3. 光影矛盾：影子方向是否与光源一致
4. 物质守恒：物体是否凭空出现/消失/不合理变形
5. 流体异常：液体是否违反流体力学（水往高处流等）

请以 JSON 格式回答：
{
    "has_violations": true/false,
    "confidence": 0.0-1.0,
    "violations": [
        {"type": "gravity/penetration/shadow/conservation/fluid", "description": "描述", "severity": "mild/moderate/severe"}
    ]
}"""
