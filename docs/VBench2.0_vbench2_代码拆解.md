# VBench-2.0：`vbench2` 包模块级拆解

本文档范围仅限参考树内 **`VBench-2.0/vbench2`**（Python 包 `vbench2`），对应论文 *VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness* 的开源实现。安装说明、环境依赖、子模块编译等以仓库根目录 `VBench-2.0/README.md` 为准。

---

## 1. 与 VBench 1.x（`vbench`）的关系

| 项目 | VBench 1.x `vbench` | VBench-2.0 `vbench2` |
|------|---------------------|----------------------|
| 入口类 | `VBench` | `VBench2` |
| 缓存目录 | `~/.cache/vbench`（或 `VBENCH_CACHE_DIR`） | `~/.cache/vbench2`（或 `VBENCH2_CACHE_DIR`） |
| 评测焦点 | 技术质量、prompt 对齐等 16 维 | 人体、组合、物理/常识、多视角等 **18 维** |
| 典型依赖 | CLIP、DINO、RAFT、GRIT、ViCLIP 等 | 大量 **LLaVA-Video-7B-Qwen2** + **Qwen2.5-7B**，以及检测/跟踪/异常检测 |

两套包 **互不 import**；业务上若同时引用，需注意命名空间与依赖隔离（建议独立虚拟环境）。

---

## 2. 顶层文件与职责

`vbench2/` 根目录下与评测直接相关的模块（不含 `third_party/` 内海量代码）：

| 文件 | 维度名（`build_full_dimension_list`） |
|------|----------------------------------------|
| `__init__.py` | `VBench2` 类：拼 JSON、循环调度各维 |
| `human_anatomy.py` | `Human_Anatomy` |
| `human_identity.py` | `Human_Identity` |
| `human_clothes.py` | `Human_Clothes` |
| `diversity.py` | `Diversity` |
| `composition.py` | `Composition` |
| `dynamic_spatial_relationship.py` | `Dynamic_Spatial_Relationship` |
| `dynamic_attribute.py` | `Dynamic_Attribute` |
| `motion_order_understanding.py` | `Motion_Order_Understanding` |
| `human_interaction.py` | `Human_Interaction` |
| `complex_landscape.py` | `Complex_Landscape` |
| `complex_plot.py` | `Complex_Plot` |
| `camera_motion.py` | `Camera_Motion` |
| `motion_rationality.py` | `Motion_Rationality` |
| `instance_preservation.py` | `Instance_Preservation` |
| `mechanics.py` | `Mechanics` |
| `thermotics.py` | `Thermotics` |
| `material.py` | `Material` |
| `multi_view_consistency.py` | `Multi-View_Consistency`（导入名特殊，见下） |
| `utils.py` | 视频 IO、`load_dimension_info`、`init_submodules`、场景切分等 |
| `distributed.py` | 多进程/多卡辅助（部分维度使用） |
| `hack_registry.py` | 被检测类维度 import，用于兼容/补丁注册 |

**维度名 → Python 模块的映射**（`VBench2.evaluate`）：

- 一般：`import vbench2.<dimension.lower()>`，调用 `compute_<dimension.lower()>()`。  
  例：`Human_Anatomy` → `vbench2.human_anatomy.compute_human_anatomy`。
- **例外**：`Multi-View_Consistency` 先规范为 `Multi_View_Consistency`，再 `.lower()` → 模块 `multi_view_consistency`，函数 `compute_multi_view_consistency`。

---

## 3. `VBench2` 运行数据流

```
VBench2.evaluate(videos_path, name, dimension_list, mode, ...)
  1) init_submodules(dimension_list)   # 下载/配置权重路径
  2) build_full_info_json(...)         # 写出 {name}_full_info.json
  3) 对每个 dimension:
       import vbench2.<module>
       compute_*(cur_full_info_path, device, submodules_dict[dimension], **kwargs)
  4) save_json → {name}_eval_results.json
```

与 1.x 类似：**每个维度自包含**，没有统一的「可微评判头」。

---

## 4. `build_full_info_json` 行为摘要

- **`mode=='custom_input'`**  
  - 仅扫描目录下 **`.mp4`**。  
  - 若维度为 **`Diversity`**：按 `prompt_en` 去重，每个 prompt 构造 **20 条** 路径：`{prompt}{special_str}-{0..19}.mp4`（需文件真实存在）。  
  - 其它维度：每个文件一条 `video_list`。  
  - **`check_dimension_requires_extra_info`**：一长串维度 **不允许** custom（需官方 JSON 里的 `auxiliary_info` 等），与 1.x 类似但列表更长。

- **`mode=='vbench_standard'`（默认）**  
  - 从 `full_info_dir` 读入完整 benchmark 列表；对每条 prompt，若与 `dimension_list` 有交集，则在 `videos_path` 下找视频：  
    - **`Diversity`**：同一 prompt **20** 个索引 `0..19`。  
    - **其它**：**3** 个索引 `0..2`。  
  - 文件名：`{prompt[:180]}{special_str}-{i}{postfix}`（**prompt 截断 180 字符**）。  
  - 缺文件会 **raise**，与 1.x「仅警告」不同。

---

## 5. `load_dimension_info` 与 1.x 的差异

```python
# 逻辑要点（utils.py）
if dimension in prompt_dict['dimension'][0].lower() and 'video_list' in prompt_dict:
```

- 用 **`dimension` 字符串是否为 `prompt_dict['dimension'][0]` 小写子串** 过滤，而不是 `dimension in set(prompt_dict['dimension'])`。  
- 因此各 `compute_*` 里传入的 `dimension=` 必须与实际 JSON 第一项兼容，例如：  
  - `compute_multi_view_consistency` 使用 `dimension='multi-view_consistency'`，因 `'multi-view_consistency' in 'multi-view_consistency'`（若 JSON 写为 `Multi-View_Consistency` 转小写后仍含子串）。  
- 若存在 `auxiliary_info`，会一并传入 `prompt_dict_ls` 供 LLM/规则使用。

---

## 6. `init_submodules`：按维度的权重与外部依赖

（实现见 `vbench2/utils.py` `init_submodules`。）

| 维度 | `submodules_dict` 要点 |
|------|-------------------------|
| `Multi-View_Consistency` | RAFT `raft-things.pth` + CoTracker2（`torch.hub` `facebookresearch/co-tracker`） |
| `Camera_Motion` | 仅 CoTracker2 |
| `Human_Identity` | ArcFace 权重（Google Drive 下载） |
| `Instance_Preservation` | `gdown` 整包文件夹到本地 `model` 目录 |
| `Human_Anatomy` | YOLO-World + 三套 ViT 异常检测权重（human/face/hand，阈值写死在 dict） |
| `Human_Clothes`, `Composition`, `Dynamic_Spatial_Relationship`, `Dynamic_Attribute`, `Motion_Rationality`, `Mechanics`, `Thermotics`, `Material` | `huggingface-cli download` → `LLaVA-Video-7B-Qwen2` |
| `Complex_Landscape`, `Complex_Plot`, `Human_Interaction`, `Motion_Order_Understanding` | 上述 LLaVA + **`Qwen2.5-7B-Instruct`** |
| 其余（如 `Diversity`） | `{}` |

---

## 7. 视频与帧：常见读取方式

不同维度 **未统一**为一种 Reader，大致分几类：

1. **`utils.load_video`**  
   - `mp4`：`decord.VideoReader`，可 `num_frames` + `get_frame_indices(..., sample="middle")` 均匀采样；默认可拉 **全帧**。  
   - `gif`/`png`：PIL。  
   - 返回 `(T,C,H,W)` uint8 tensor（无 transform 时）。

2. **LLaVA 系列文件内嵌的 `load_video`**（`composition.py`、`mechanics.py` 等多处复制）  
   - `decord.VideoReader`，`max_frames_num=64`，`force_sample=True` 时在 **时间轴上均匀取 64 帧**；  
   - 拼 **时长 + 各帧时间戳字符串** 写入 prompt，再 `image_processor.preprocess` → `bfloat16` GPU。

3. **`utils.get_frames`**（**`Diversity`** 用）  
   - **OpenCV** `VideoCapture` 逐帧；Resize **512×512**，ImageNet normalize；**间隔 1** 取帧（即全帧，开销大）。

4. **整段上 GPU**（`Human_Identity`、`Camera_Motion`、`Multi-View_Consistency` 的 CoTracker 分支）  
   - `decord` `get_batch(range(len)))` 一次取全片，`permute` 成 `B,T,C,H,W` float CUDA。

5. **`Human_Anatomy`（ViTDetector）**  
   - `mmcv.VideoReader` 逐帧 + YOLO-World 检测人体/手脸 + 裁剪区域过 SimMIM ViT 异常分数（详见 `third_party/ViTDetector/detect.py`）。

6. **`multi_view_consistency` 中 RAFT 分支**  
   - OpenCV 读帧，resize **854×480**，按 fps 下采样到约 **8fps** 的帧序列算光流。

---

## 8. 各维度判定逻辑（实现语义）

以下为「代码在做什么」，分数方向以 **越大通常表示越符合该维设定** 为准（具体见各维 `video_results` 定义）。

### 8.1 `Human_Anatomy`

- **文件**：`human_anatomy.py` → `third_party/ViTDetector/detect.compute_abnormality`。  
- **逻辑**：YOLO-World 检出人体/手/脸区域 → 各区域过 **异常检测 ViT**，与阈值比较得到异常与否；汇总为视频级指标（`compute_human_anatomy` 再对 `video_results` 做平均）。  
- **依赖**：MMCV/MMDet 生态，权重由 `init_submodules` 配置。

### 8.2 `Human_Identity`

- **逻辑**：RetinaFace 检单人脸 → **ArcFace式** `resnet_face18` 特征；首帧特征为模板，后续帧余弦相似度低于阈值则计为不一致。  
- **分数**：有效帧上 **一致帧比例** `consistent_frame_count/frame_num`；有效帧不足 `mini_frame` 时该视频记为 `-1` 并不计入最终分子分母平均。

### 8.3 `Human_Clothes`

- **逻辑**：LLaVA-Video 对固定三个 yes/no 问题打分（是否仅一人、是否同人、衣着是否一致）；全 yes 则该视频 1，否则 0（见文件内循环）。

### 8.4 `Diversity`

- **逻辑**：同一 prompt 下 **多条视频**（20 条）：每条用 **VGG19** 多层特征，算 **内容 L1 + Gram 风格差异**，在视频两两之间累积，再除以经验常数 `17.712` 并 **clamp 到 [0,1]**。  
- **含义**：同 prompt 不同随机种子差异越大，分越高。

### 8.5 `Composition`

- **逻辑**：LLaVA，`auxiliary_info` 含 `question` 列表与 `judge` 两个标志位：  
  - 可先问「是否仅一个生物」；  
  - 再对各描述问是否出现在视频中；  
  - 按 `judge` 决定 **全对才 1** 还是 **按题给分**。  

### 8.6 `Dynamic_Spatial_Relationship`

- **逻辑**：两问：第一段用 **前 32 帧** 视频；第二问用 **最后一帧** 当 **image** 模态；两段都需回答 yes 才判该视频通过（`flag`）。

### 8.7 `Dynamic_Attribute`

- **逻辑**：`auxiliary_info` 为问题列表；**每一问**须输出含 `yes`，否则该视频失败（全通过才 `sco=1`）。

### 8.8 `Motion_Rationality`

- **逻辑**：多道 yes/no，**全部 yes** 则该视频 1，否则 0。

### 8.9 `Mechanics` / `Thermotics` / `Material`

- **逻辑**：与 `Dynamic_Attribute` 同型：**`auxiliary_info` 为字符串或字符串列表**；多问时往往第一道校验「现象是否成立」，第二道校验物理细节；全 yes 才通过。  
- **`Thermotics`** 的 instruction 里注明 **不纠结专有名词拼写**。

### 8.10 `Motion_Order_Understanding`

- **逻辑**：LLaVA 根据视频生成 **带编号的情节列表**；再用 **Qwen2.5**（多轮 chat + 论文式 few-shot）判断 **动作顺序/语义** 是否与 `auxiliary_info` 一致（解析回复是否以 yes 开头）。

### 8.11 `Human_Interaction`

- **逻辑**：LLaVA 描述互动；**Qwen** 两阶段：先判断是否多人，再判断与参考互动的语义是否一致（具体分支见文件内对 `auxiliary_info` 的解析）。

### 8.12 `Complex_Landscape` / `Complex_Plot`

- **逻辑**：  
  1. LLaVA 看视频 + `auxiliary_info` 中的长描述，生成 **分条 plot**；  
  2. **Qwen** 按系统 prompt 判断「LLaVA 摘要是否覆盖 ground truth 中的景观/情节要点」（先 yes/no 再理由）。  
- 与纯像素指标不同，强依赖 **VLM 文本推理**。

### 8.13 `Camera_Motion`

- **逻辑**：**CoTracker2** 在整段视频上得到网格点轨迹；根据 **上下左右边缘点位移** 映射到 `pan_left`、`tilt_up`、`zoom_in`、`static`、`orbits` 等标签集合；与 JSON **`auxiliary_info` 字符串标签** 是否命中做 **0/1**。

### 8.14 `Instance_Preservation`

- **文件**：`instance_preservation.py` → `third_party/Instance_detector/test.compute_anomaly`。  
- **逻辑**：实例级异常检测管线（权重来自 `gdown` 文件夹），输出视频级分数；细节见 `Instance_detector` 内实现。

### 8.15 `Multi-View_Consistency`

- **逻辑（概要）**：  
  1. 用 **场景检测**（`utils.split_video_into_scenes`）估第一段结束帧，判断是否存在 **360/orbit** 类相机运动（`whether_orbit`）。  
  2. **RAFT** 在降采样帧上算光流强度均值：过小（`<5`）认为不适合评测，记 `-1`。  
  3. 否则 **`PatchAutoEvaluate`**（`third_party/Dense_match`）结合光流强度做块级匹配分 `match_score`。  
  4. **最终分**：`clip(score_flow,0,10) * match_score / 10`；无效时为 `-1`；集计时对非 `-1` 视频平均。

### 8.16 `distributed.py` / `hack_registry.py`

- 部分维度或第三方库在多 GPU 下用 `distribute_list_to_rank`、`gather_list_of_dict`；`hack_registry` 在 import 检测流水线时注册兼容补丁（具体以源码为准）。

---

## 9. 业务复用时的注意点

1. **强依赖大模型**：除 `Diversity`、`Human_Identity`、`Camera_Motion`、检测类外，多数维需要 **本地 7B 级显存** 或可改源码接 API。  
2. **`custom_input` 限制严**：带 `auxiliary_info` 的维度无法直接用自定义目录模式，需自建与官方结构一致的 `full_info` JSON。  
3. **判定含大量字符串规则**：如 `"yes" in text_outputs.lower()`，对模型输出格式敏感，升级 LLaVA/Qwen 版本可能导致分布漂移。  
4. **Windows**：`init_submodules` 中 `wget`/`unzip`/`huggingface-cli` 与 torch.hub 行为与 Linux 差异大，多在 **Linux + CUDA** 上跑通。

---

## 10. 快速索引表

| 维度 | 主模块文件 | 核心信号 |
|------|------------|----------|
| Human_Anatomy | `human_anatomy.py` + ViTDetector | 检测 + 异常 ViT |
| Human_Identity | `human_identity.py` | RetinaFace + ArcFace |
| Human_Clothes | `human_clothes.py` | LLaVA 三问 |
| Diversity | `diversity.py` | VGG 内容/风格两两差异 |
| Composition | `composition.py` | LLaVA + auxiliary QA |
| Dynamic_Spatial_Relationship | `dynamic_spatial_relationship.py` | LLaVA 半段视频 + 尾帧图 |
| Dynamic_Attribute | `dynamic_attribute.py` | LLaVA 全 yes |
| Motion_Rationality | `motion_rationality.py` | LLaVA 全 yes |
| Mechanics / Thermotics / Material | 对应 `mechanics.py` 等 | LLaVA + auxiliary |
| Motion_Order_Understanding | `motion_order_understanding.py` | LLaVA 摘要 + Qwen 顺序判断 |
| Human_Interaction | `human_interaction.py` | LLaVA + Qwen |
| Complex_Landscape / Complex_Plot | `complex_landscape.py` 等 | LLaVA + Qwen 一致性 |
| Camera_Motion | `camera_motion.py` | CoTracker 运动模式分类 |
| Instance_Preservation | `instance_preservation.py` | Instance 异常检测 |
| Multi-View_Consistency | `multi_view_consistency.py` | 场景 + CoTracker + RAFT + PatchMatch |

---

*文档基于本地参考树 `参考代码/VBench/VBench-2.0/vbench2` 阅读整理；上游更新后请以官方仓库为准。*
