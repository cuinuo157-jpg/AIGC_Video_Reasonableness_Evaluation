# Grounded-SAM-2 / GroundingDINO CUDA 编译与多卡推理说明

本文档用于解释以下问题，并提供可直接执行的 Linux 方案：

- “把 CUDA 扩展编译好”是什么意思
- 为什么日志里会出现 `Failed to load custom C++ ops`
- 如何在 Linux 上正确编译 GroundingDINO 与 SAM2 扩展
- 当前项目推理需要几张卡，以及 4 张卡如何使用更高效

---

## 1. 术语说明：什么叫“把 CUDA 扩展编译好”

在 `GroundingDINO` 和 `SAM2` 中，部分算子由 C++/CUDA 自定义扩展实现。  
如果这些扩展未成功编译或未被正确加载，代码会回退到慢速路径（CPU 或普通 PyTorch 路径）。

典型日志：

- `Failed to load custom C++ ops. Running on CPU mode Only!`
- `GroundingDINO 自定义算子 _C 不可用，自动回退到 CPU 推理（速度会较慢）`

这通常**不是功能性错误**，而是**性能降级**：结果可用，但速度会明显变慢。

---

## 2. Linux 环境编译步骤（推荐按顺序执行）

以下命令默认在项目根目录执行：`AIGC_Video_Reasonableness_Evaluation`

### 2.1 检查基础环境

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
which nvcc && nvcc --version
python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
```

最基本要求：

- `torch.cuda.is_available()` 为 `True`
- `nvcc` 可用
- `CUDA_HOME` 指向有效 CUDA 安装目录（如 `/usr/local/cuda`）

### 2.2 安装编译依赖

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build
pip install -U pip setuptools wheel cython ninja
```

### 2.3 编译 GroundingDINO 扩展（关键）

```bash
cd third_party/Grounded-SAM-2/grounding_dino
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS=8
pip install -v -e .
```

说明：

- `TORCH_CUDA_ARCH_LIST` 请按实际 GPU 架构调整（不确定时可先留默认）。
- `-v` 方便看到真实编译日志，便于定位失败原因。

### 2.4 编译 SAM2 扩展

```bash
cd ../
export CUDA_HOME=/usr/local/cuda
export SAM2_BUILD_CUDA=1
export SAM2_BUILD_ALLOW_ERRORS=0
pip install -v -e .
```

说明：

- `SAM2_BUILD_ALLOW_ERRORS=0` 用于强制编译失败时直接报错，避免“静默降级”。

---

## 3. 编译成功验证

```bash
python -c "from grounding_dino.groundingdino.models.GroundingDINO import ms_deform_attn as m; print('GDINO _C:', getattr(m,'_C',None) is not None)"
python -c "import sam2._C as c; print('SAM2 _C OK')"
```

期望结果：

- GroundingDINO 输出 `GDINO _C: True`
- SAM2 可以成功 import `sam2._C`

---

## 4. Linux 仍报同样告警时的常见原因

1. **Torch 与 CUDA 版本不匹配**
   - 例如装了 CPU 版 Torch，或 Torch 编译 CUDA 版本与本机 `nvcc`/driver 组合不兼容。
2. **未在正确目录执行 editable install**
   - 必须在 `third_party/Grounded-SAM-2/grounding_dino` 与 `third_party/Grounded-SAM-2` 分别执行。
3. **历史失败缓存导致重复加载旧包**
   - 建议先卸载后重装：
   ```bash
   pip uninstall groundingdino SAM-2 -y
   ```
4. **环境变量未生效**
   - 如 `CUDA_HOME`、`TORCH_CUDA_ARCH_LIST` 未正确导出。

---

## 5. 当前项目推理的显卡需求评估

结合当前脚本实现（如 `scripts/debug_dynamics.py`、`scripts/debug_temporal_coherence.py`）：

- **单视频 / 单进程推理**：`1 张卡` 即可完成
- **吞吐优先（批量视频）**：建议 `4 张卡 = 4 个独立进程`，每进程绑定 1 张卡
- **性价比模式**：`2 张卡` 并行也能显著提速

### 5.1 为什么不是单任务吃满 4 卡

当前代码主路径是单进程单设备推理逻辑，并未在工程中实现统一 DDP/多卡切分。  
因此，最稳妥高效的方式是：**多进程分片视频列表**，而非把一个视频拆到 4 卡。

---

## 6. 4 卡部署推荐方案（生产可落地）

推荐策略：

- 用 `CUDA_VISIBLE_DEVICES` 绑定每个进程的独立 GPU
- 将待处理视频列表均分成 4 份并行执行
- 所有结果统一写入 `outputs/`，必要时再做汇总

示例思想（伪命令）：

```bash
# worker-0
CUDA_VISIBLE_DEVICES=0 python scripts/debug_dynamics.py --input <shard0_dir> --device cuda --method raft --subject
# worker-1
CUDA_VISIBLE_DEVICES=1 python scripts/debug_dynamics.py --input <shard1_dir> --device cuda --method raft --subject
# worker-2
CUDA_VISIBLE_DEVICES=2 python scripts/debug_dynamics.py --input <shard2_dir> --device cuda --method raft --subject
# worker-3
CUDA_VISIBLE_DEVICES=3 python scripts/debug_dynamics.py --input <shard3_dir> --device cuda --method raft --subject
```

---

## 7. 与当前告警的关系总结

- `bitsandbytes` 告警通常不影响本项目核心路径（多数情况下可忽略或卸载该包）。
- 真正影响速度的是 `GroundingDINO/SAM2` 自定义 CUDA 扩展未加载成功。
- 只要扩展编译成功并可 import，相关“CPU mode only / _C 不可用”告警应消失或显著减少。

