# 无Prompt评测方案

## 问题背景

当前项目中部分算法（主要是`temporal_reasoning`和`aux_motion_intensity_2`模块）需要文本prompt作为输入，用于引导Grounded-SAM-2或Grounded DINO进行对象检测和分割。这限制了评测的自动化程度。

## 可行方案

### 方案1：自动标签生成 + Grounded DINO（推荐）

**原理**：使用视觉语言模型（如RAM、Tag2Text、BLIP等）自动识别视频中的对象类别，然后使用这些类别作为prompt。

**优点**：
- 完全自动化，无需人工输入
- 可以检测视频中实际存在的所有对象
- 准确度较高

**实现步骤**：
1. 从视频中采样关键帧（如第一帧或中间帧）
2. 使用RAM/Tag2Text模型生成对象标签列表
3. 过滤和清理标签（去除重复、合并相似类别）
4. 将标签组合成prompt（如"person. car. dog."）
5. 使用现有的Grounded-SAM-2流程进行检测

**参考实现**：
- `third_party/Grounded-Segment-Anything/automatic_label_simple_demo.py` 展示了使用RAM自动生成标签的示例

**依赖**：
- RAM (Recognize Anything Model) 或 Tag2Text
- 需要下载相应的预训练模型

---

### 方案2：通用对象检测模型（YOLO/DETR）

**原理**：使用通用的对象检测模型（如YOLOv8、DETR）自动检测所有常见对象，然后转换为SAM2可用的格式。

**优点**：
- 检测速度快
- 覆盖常见对象类别
- 无需文本prompt

**缺点**：
- 只能检测预定义类别（COCO数据集80类或自定义类别集）
- 可能遗漏一些特殊对象

**实现步骤**：
1. 使用YOLO/DETR在关键帧上检测对象
2. 获取检测框和类别标签
3. 将检测框直接输入SAM2进行分割（无需Grounded DINO）
4. 使用SAM2进行视频传播

**依赖**：
- YOLOv8 或 DETR
- 需要适配检测框格式到SAM2输入

---

### 方案3：通用Prompt策略

**原理**：使用通用的、覆盖范围广的prompt，如"object"、"thing"、"person. animal. vehicle. furniture."等。

**优点**：
- 实现简单，无需额外模型
- 可以检测大部分常见对象

**缺点**：
- 检测精度可能较低
- 可能检测到不相关的背景对象
- Grounded DINO对通用prompt的支持有限

**实现步骤**：
1. 定义通用prompt列表（如["person", "object", "thing"]）
2. 在检测时使用这些通用prompt
3. 通过置信度阈值过滤低质量检测

**当前代码已支持**：
- `temporal_reasoning`模块默认使用`["object"]`作为fallback
- 可以通过配置`structure_prompts`设置通用prompt

---

### 方案4：基于视觉特征的对象发现（高级）

**原理**：使用无监督或自监督方法（如特征聚类、显著性检测）发现视频中的主要对象区域。

**优点**：
- 完全无监督，不依赖预定义类别
- 可以发现任意对象

**缺点**：
- 实现复杂
- 需要额外的特征提取模型
- 可能不够稳定

**实现步骤**：
1. 使用预训练的特征提取器（如CLIP、DINO）提取帧特征
2. 使用聚类或显著性检测找到主要对象区域
3. 生成初始mask或bounding box
4. 使用SAM2进行精细化分割和传播

---

## 推荐实现方案

### 短期方案（快速实现）：方案3 + 方案1

1. **立即实现**：增强通用prompt策略
   - 扩展通用prompt列表，包含更多常见类别
   - 优化检测阈值，提高通用prompt的检测质量

2. **中期实现**：集成自动标签生成
   - 集成RAM或Tag2Text模型
   - 在关键帧上自动生成标签
   - 将自动标签作为prompt使用

### 长期方案（最佳效果）：方案1 + 方案2混合

- 使用RAM自动生成标签（覆盖特殊对象）
- 使用YOLO检测常见对象（提高速度和准确度）
- 合并两种方法的检测结果

---

## 实现细节

### 方案1实现示例

```python
# 伪代码示例
def auto_generate_prompts(video_path, frame_idx=0):
    """自动生成prompt"""
    # 1. 加载RAM模型
    ram_model = load_ram_model()
    
    # 2. 提取关键帧
    frame = extract_frame(video_path, frame_idx)
    
    # 3. 生成标签
    tags = ram_model.predict(frame)  # 返回: "person | car | dog | ..."
    class_list = tags.split(" | ")
    
    # 4. 清理和过滤标签
    filtered_classes = filter_classes(class_list)
    
    # 5. 组合成prompt
    prompt = ". ".join(filtered_classes) + "."
    
    return prompt
```

### 方案2实现示例

```python
# 伪代码示例
def detect_with_yolo(image):
    """使用YOLO检测对象"""
    # 1. 加载YOLO模型
    yolo_model = load_yolo_model()
    
    # 2. 检测对象
    results = yolo_model(image)
    
    # 3. 提取检测框和类别
    boxes = results.boxes.xyxy
    classes = results.boxes.cls
    labels = [yolo_model.names[int(c)] for c in classes]
    
    # 4. 转换为SAM2输入格式
    return boxes, labels
```

---

## 配置建议

### 在config.py中添加配置

```python
# 无prompt评测配置
class NoPromptConfig:
    enable_auto_prompt: bool = False  # 是否启用自动prompt生成
    auto_prompt_method: str = "ram"  # 方法: "ram", "yolo", "generic"
    generic_prompts: List[str] = ["person", "object", "thing"]  # 通用prompt列表
    ram_model_path: Optional[str] = None  # RAM模型路径
    yolo_model_path: Optional[str] = None  # YOLO模型路径
```

---

## 性能对比

| 方案 | 准确度 | 速度 | 实现难度 | 推荐度 |
|------|--------|------|----------|--------|
| 方案1: 自动标签生成 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 方案2: YOLO/DETR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 方案3: 通用Prompt | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| 方案4: 特征发现 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 下一步行动

1. **立即行动**：实现方案3的增强版本（通用prompt优化）
2. **短期目标**：集成RAM模型实现方案1
3. **长期目标**：实现方案1+方案2的混合方案

---

## 参考资料

- RAM模型: https://github.com/xinyu1205/recognize-anything-model
- Tag2Text: https://github.com/xinyu1205/Tag2Text
- YOLOv8: https://github.com/ultralytics/ultralytics
- Grounded-SAM自动标签示例: `third_party/Grounded-Segment-Anything/automatic_label_simple_demo.py`

