# VBench 参考代码拆解说明

本文档基于仓库内参考副本：`参考代码/VBench`，说明 **VBench 1.x（`vbench/` 包）** 的目录职责、运行数据流，以及各评测维度的**判定逻辑**（实现层面在算什么、分数高低含义）。**VBench-2.0**、**vbench2_beta_*** 为同仓内独立子工程，安装与入口不同，文末仅作边界说明。

---

## 1. 仓库顶层结构（与主评测流相关部分）

| 路径 | 作用 |
|------|------|
| `evaluate.py` | 命令行入口：解析参数 → 构造 `VBench` → 调用 `evaluate()` |
| `vbench/__init__.py` | 核心类 `VBench`：组装 `full_info` JSON、按维度动态 `import` 并调用 `compute_<dimension>` |
| `vbench/*.py` | 各评测维度实现，约定函数名 `compute_<dimension>(json_dir, device, submodules_list, **kwargs)` |
| `vbench/utils.py` | 视频读取、CLIP/DINO 预处理、`load_dimension_info`、`init_submodules`（模型路径与自动下载） |
| `vbench/distributed.py` | 多卡时列表分片与结果汇总 |
| `vbench/third_party/` | 内置第三方：RAFT、AMT（插帧）、UMT、GRIT、ViCLIP、Tag2Text 等 |
| `static_filter.py` | **仅服务 `temporal_flickering`**：用 RAFT 筛掉「本应几乎静止」的样本，避免与指标假设冲突 |
| `vbench/VBench_full_info.json` | 标准 benchmark：每条含 `prompt_en`、`dimension` 列表、（运行后填入的）`video_list` |
| `VBench-2.0/` | 另一套 `vbench2` 包与依赖，评测「内在忠实度」等，与本文 `vbench` 主包分离 |
| `vbench2_beta_i2v` / `vbench2_beta_long` / `vbench2_beta_trustworthiness` | VBench++ 扩展场景脚本与说明 |

---

## 2. 端到端数据流

```
evaluate.py
    → VBench(device, full_json_dir, output_path)
    → VBench.evaluate(videos_path, name, dimension_list, mode=..., ...)
         1) init_submodules(dimension_list)     # 准备各维度权重路径，必要时 wget
         2) build_full_info_json(...)          # 生成 {name}_full_info.json
         3) 对每个 dimension:
                importlib.import_module(f'vbench.{dimension}')
                compute_<dimension>(cur_full_info_path, device, submodules_dict[dimension], **kwargs)
         4) 汇总写入 {name}_eval_results.json
```

**要点：**

- 每个维度是**独立模块**，无统一「裁判网络」；逻辑从简单像素 MAE 到 VLM/检测器不等。
- `compute_*` 的**第一个参数**始终是某次运行生成的 `*_full_info.json` 路径（不是原始的 `VBench_full_info.json` 全量静态文件，除非 `build_full_info_json` 直接基于它裁剪生成）。

---

## 3. `VBench.build_full_info_json` 三种模式

| `mode` | 用途 | 生成的 `video_list` 来源 |
|--------|------|---------------------------|
| `vbench_standard` | 对齐论文标准 prompt 套件 | 读取 `full_info_dir` JSON；在 `videos_path` 下按 `{prompt}{special_str}-{i}.mp4`（或目录中首个视频的后缀）匹配 0–4 号视频 |
| `custom_input` | 用户自有视频 | 扫描目录下 `.mp4`/`.gif`；`prompt_en` 来自文件名（`get_prompt_from_filename`）或 `--prompt` / `--prompt_file`；**禁止**需要 `auxiliary_info` 的维度（见下） |
| `vbench_category` | 按内容类别 | 读 `vbench/prompts_per_category/{category}.txt` 中 prompt，文件名需以该 prompt 为前缀匹配 |

**`get_prompt_from_filename` 规则：** 去掉扩展名；若 stem 以 `-数字` 结尾则剥掉（如 `foo bar-0` → `foo bar`）。

**`custom_input` 不支持的维度**（缺少 JSON 里的结构化辅助信息则无法跑）：  
`object_class`, `multiple_objects`, `scene`, `appearance_style`, `color`, `spatial_relationship`。

---

## 4. `load_dimension_info`：维度如何取视频列表

实现见 `vbench/utils.py`：

- 遍历 `full_info` 列表中每一项；
- 若当前 `dimension` 在该条的 `prompt_dict['dimension']` 中，且存在 `video_list`，则把这些路径加入本维度的评测列表；
- 若存在 `auxiliary_info` 且含该 dimension，则 `prompt_dict_ls` 中附带 `auxiliary_info`，供 **object/scene/color** 等维度读取关键词。

---

## 5. `init_submodules`：各维度依赖的模型与缓存

逻辑集中在 `vbench/utils.py` 的 `init_submodules`。缓存根目录优先环境变量 `VBENCH_CACHE_DIR`，否则 `~/.cache/vbench`。

| 维度 | 主要依赖 | 说明 |
|------|-----------|------|
| `background_consistency` | CLIP ViT-B/32 | `local` 时下载 `.pt` 到 cache |
| `subject_consistency` | DINO ViT-B/16 | `torch.hub` 或本地 clone+权重 |
| `aesthetic_quality` | CLIP ViT-L/14 + LAION aesthetic 线性头 | |
| `imaging_quality` | PyIQA MUSIQ 权重 | 帧级技术质量 |
| `temporal_flickering` | 无深度学习模型 | OpenCV 读帧 |
| `motion_smoothness` | AMT-S（插帧网络） | `third_party/amt` + `amt-s.pth` |
| `dynamic_degree` | RAFT | `raft-things.pth` |
| `object_class` / `multiple_objects` / `color` / `spatial_relationship` | GRIT dense caption | |
| `scene` | Tag2Text | |
| `human_action` | UMT ViT-L，Kinetics-400 | |
| `overall_consistency` / `temporal_style` | ViCLIP 权重 | |
| `appearance_style` | CLIP ViT-B/32 | |

---

## 6. 各维度判定逻辑（实现语义）

以下「分数高/低」均指 **VBench 实现里返回的 `video_results` 标量或均值**，具体是否「越大越好」因维度而异。

### 6.1 时序质量类

**`temporal_flickering`**（`vbench/temporal_flickering.py`）

- 用 **OpenCV** 逐帧读取整段视频（与部分维度用 decord 不同）。
- 相邻帧计算 **全图 MAE**（`cv2.absdiff` 均值），对整段取平均 `mean_mae`。
- 单视频分数：`(255.0 - mean_mae) / 255.0`。
- **含义：** 相邻帧像素变化越大，MAE 越大，**分数越低**；变化越小则**分数越高**。  
  论文中用于刻画「静态内容上的高频闪烁/不稳定」；**设计上假设视频内容应接近静止**，故 README 要求先用 `static_filter.py` 筛样。
- **`static_filter.py`：** 同样用 RAFT 得到光流幅值，与 `dynamic_degree` 类似但阈值不同，用于判定整条视频是否「够静」，从而决定是否参与 flickering 统计。

**`motion_smoothness`**（`vbench/motion_smoothness.py`）

- 取 **每隔一帧** 的子序列（`extract_frame(..., step=2)`）。
- 用 **AMT** 在相邻帧之间做插帧，得到稠密中间帧序列。
- 将插帧结果与 **原始偶数索引帧** 在对应位置做 `absdiff` 均值，再对位置取平均得到 `vfi_score`。
- 单视频分数：`(255.0 - vfi_score) / 255.0`。
- **含义：** 若真实运动与「光流平滑假设」一致，插帧预测接近真实中间观测，diff 小，**分数高**；抖动或不合物理的运动则 diff 大，**分数低**。

**`dynamic_degree`**（`vbench/dynamic_degree.py`）

- RAFT 估计相邻帧光流；取幅值 `sqrt(u^2+v^2)`，在全图排序后取 **前 5% 像素的均值** 作为一帧对的「运动强度」。
- 阈值 `thres = 6.0 * (scale/256)`，`count_num = round(4*(帧数/16))`；若在足够多的帧对上强度超过阈值则 `infer` 返回 **True**，否则 **False**。
- 集计时对布尔值取 `np.mean`，得到数据集上「被判为有明显运动」的比例。
- **含义：** 粗粒度 **是否有足够大的位移**，不是细粒度物理合理性。

### 6.2 主体 / 背景一致性（表征相似度）

**`subject_consistency`**（`vbench/subject_consistency.py`）

- DINO 提特征，逐帧与 **前一帧**、**第一帧** 算余弦相似度，取平均 `(sim_pre + sim_fir)/2`，再对帧平均。
- **含义：** **主体外观跨帧越一致，分数越高**（不区分「真一致」与「贴图冻结」）。

**`background_consistency`**（`vbench/background_consistency.py`）

- CLIP 图像编码，与 subject 相同的 **前一帧 + 首帧** 相似度平均。
- **含义：** **背景表观越稳定，分数越高**。

### 6.3 画质与美学

**`aesthetic_quality`**（`vbench/aesthetic_quality.py`）

- CLIP ViT-L/14 提特征，过 LAION **线性美学头**，输出除以 10，对帧平均再对视频平均。
- **含义：** 与 **静态图像美学** 数据集标定相关，**非**视频专用物理指标。

**`imaging_quality`**（`vbench/imaging_quality.py`）

- **MUSIQ** 逐帧打分，视频内平均；再除以 100 归一化。
- 可通过 `imaging_quality_preprocessing_mode` 控制缩放/裁剪策略。

### 6.4 与 Prompt / 标签对齐（需 auxiliary_info 或文件名规则）

**`object_class`**（`vbench/object_class.py`）

- GRIT 对每帧生成检测/描述集合；JSON 中 `auxiliary_info.object` 为要在帧中出现的 **关键词**；统计含该词的帧比例。

**`multiple_objects`**、**`spatial_relationship`**：同类 GRIT 管线，规则在各自文件中解析 `auxiliary_info`。

**`color`**（`vbench/color.py`）

- GRIT 输出中匹配 **物体类别** 与 **颜色词**（预定义颜色表）是否同时出现。

**`scene`**（`vbench/scene.py`）

- Tag2Text 生成场景相关 caption；`auxiliary_info.scene` 中关键词需 **整句全部命中**（按空格拆词全包含）计为成功帧。

**`human_action`**（`vbench/human_action.py`）

- 从 **文件名** 解析期望类别（如含 `person is ...`）；UMT 预测 Kinetics-400 Top-5，与标签比对是否命中。

**`appearance_style`**：依赖 CLIP 与风格相关 `auxiliary_info`（实现见同目录文件）。

### 6.5 视频–文本整体一致性

**`overall_consistency`**（`vbench/overall_consistency.py`）

- **ViCLIP**：按 FPS 采样 8 帧，编码视频；将 **完整 prompt 字符串** 编码为文本特征。
- 分数为 **视频特征与文本特征的内积（logit）**（未在代码中再做 sigmoid 成概率，直接标量平均）。
- **含义：** 与 CLIP 式图文对齐类似，偏 **语义是否「像 prompt 描述的视频」**。

**`temporal_style`**：同样基于 ViCLIP，侧重时序风格维度（实现见 `temporal_style.py`）。

---

## 7. 分布式与多进程

`vbench/distributed.py` 提供 `distribute_list_to_rank`、`gather_list_of_dict` 等。各 `compute_*` 在 world_size>1 时对 **按视频列表聚合的结果** 做平均或加权平均；不同维度聚合方式略有差异（例如 `background_consistency` 用全局 `video_sim`/`cnt` 再除）。

---

## 8. 与业务评测对接时的注意点

1. **维度之间目标可能冲突**：例如主体/背景「越一致分越高」，与「露珠应微动」的直觉不一定一致；`dynamic_degree` 只问「有没有够大的光流」，不问「该不该动」。
2. **`temporal_flickering` 假设近静态内容**；对强运动视频需慎用或先过滤。
3. **`custom_input` 无法直接跑依赖 `auxiliary_info` 的维度**；若要用 object/scene/color 等，需自建与 `VBench_full_info.json` 相同结构的 meta。
4. **Windows 环境**：`init_submodules` 中大量使用 `wget`/`unzip`/`git clone`，在 Windows 上常需 WSL 或手动下载权重并改 `local` 路径。

---

## 9. VBench-2.0 与本拆解的关系

路径：`参考代码/VBench/VBench-2.0/`。  
该子项目使用 **`vbench2` 包**、独立 `requirement.txt`，维度覆盖 **Human Fidelity / Physics / Commonsense** 等，常依赖 **大模型 API 或本地 LLaVA/Qwen** 等，**代码结构不等于** 上文 `vbench.compute_*` 的简单扩展。若需要第二份「仅 VBench-2.0」的模块级拆解，可单独以 `VBench-2.0/vbench2` 为范围再写一版文档。

---

## 10. 关键文件索引（便于跳转）

| 文件 | 内容 |
|------|------|
| `vbench/__init__.py` | `VBench` 类、`build_full_info_json`、`evaluate` |
| `vbench/utils.py` | `load_dimension_info`、`init_submodules`、`load_video`、CLIP/DINO transform |
| `vbench/temporal_flickering.py` | MAE → 分数 |
| `vbench/motion_smoothness.py` | AMT 插帧 + 对齐误差 |
| `vbench/dynamic_degree.py` | RAFT + 阈值计数 |
| `vbench/subject_consistency.py` / `background_consistency.py` | DINO / CLIP 时序相似度 |
| `vbench/overall_consistency.py` | ViCLIP 视频–文本 logit |
| `static_filter.py` | temporal_flickering 前置静态筛选 |

---

*文档生成自本地参考树 `参考代码/VBench`，若上游仓库更新，以官方实现为准。*
