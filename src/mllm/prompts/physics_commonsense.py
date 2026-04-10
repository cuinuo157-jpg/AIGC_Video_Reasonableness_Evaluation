PHYSICS_COMMONSENSE_PROMPT_TEMPLATE = """你是一个视频物理合理性分析专家。请对这段视频进行逐步推理，判断是否存在违反物理常识的现象。

## 分析步骤

### 第一步：场景识别
识别视频中的场景类型（街道、室内、自然场景等）和主要物体/人物。

### 第二步：运动方向分析
逐一分析每个运动物体/人物的运动方向是否合理：
- 车辆是否沿正确方向行驶（是否逆行、倒退）
- 行人步行方向是否自然
- 物体下落/上升方向是否符合重力
- 多个物体是否存在不合理的同步运动（如周围物体随主体一起反向移动）

### 第三步：物理规律检查
检查是否存在以下物理违规：
- 刚体穿透（物体相互穿过）
- 悬浮异常（物体无支撑悬空）
- 光影矛盾（影子方向与光源不一致）
- 流体异常（液体违反流体力学）
- 物质守恒违反（物体凭空出现/消失/不合理变形）

### 第四步：综合判定
综合以上分析，给出 0-1 的物理合理性评分（1.0 = 完全合理，0.0 = 严重违规）。
{drift_context}
请严格以如下 JSON 格式输出：
{{
    "reasoning": "你的逐步推理过程",
    "scene_type": "场景类型",
    "has_violations": true或false,
    "physics_score": 0.0到1.0之间的浮点数,
    "violations": [
        {{
            "type": "direction_anomaly或penetration或floating或shadow或fluid或conservation",
            "description": "异常描述",
            "severity": "mild或moderate或severe",
            "confidence": 0.0到1.0之间的浮点数
        }}
    ]
}}"""

_DRIFT_CONTEXT_TEMPLATE = """
### 辅助信息
低级视觉分析检测到静态区域存在持续性单向像素漂移（平均幅度 {avg_magnitude:.2f}，方向标准差 {direction_std:.1f}°，持续 {duration_frames} 帧）。请特别关注背景物体是否存在不合理的整体平移，以及是否有物体运动方向异常。
"""


def build_physics_prompt(drift_events: list[dict] | None = None) -> str:
    """构建物理常识 CoT prompt，可选注入漂移上下文。"""
    drift_context = ""
    if drift_events:
        worst = max(drift_events, key=lambda e: e.get("avg_magnitude", 0))
        drift_context = _DRIFT_CONTEXT_TEMPLATE.format(
            avg_magnitude=worst.get("avg_magnitude", 0),
            direction_std=worst.get("direction_std", 0),
            duration_frames=worst.get("duration_frames", 0),
        )
    return PHYSICS_COMMONSENSE_PROMPT_TEMPLATE.format(drift_context=drift_context)


# 向后兼容
PHYSICS_COMMONSENSE_PROMPT = build_physics_prompt()
