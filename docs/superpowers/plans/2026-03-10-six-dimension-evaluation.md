# 六维度评测框架实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 AIGC 视频合理性六维度评测框架，包含共享特征层、6个评测维度和统一流水线。

**Architecture:** FeatureHub 共享基础特征层 + 6个独立维度分析模块 + MLLM统一调用层 + 统一流水线集成。每个维度模块遵循现有 `__init__() → initialize() → analyze()` 模式。

**Tech Stack:** Python 3.10+, PyTorch, InsightFace, Py-Feat, MediaPipe, MiDaS, OpenCV, scipy, numpy

**Spec:** `docs/superpowers/specs/2026-03-10-six-dimension-evaluation-design.md`

---

## Chunk 1: FeatureHub 共享基础特征层

所有维度的基础依赖，必须最先实现。

### Task 1.1: FeatureHub 缓存层

**Files:**
- Create: `src/feature_hub/__init__.py`
- Create: `src/feature_hub/cache.py`
- Test: `tests/test_feature_hub_cache.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feature_hub_cache.py
import numpy as np
from src.feature_hub.cache import FeatureCache

def test_cache_store_and_retrieve():
    cache = FeatureCache()
    data = np.random.rand(10, 20)
    cache.store("optical_flow", data)
    result = cache.get("optical_flow")
    assert result is not None
    np.testing.assert_array_equal(result, data)

def test_cache_miss_returns_none():
    cache = FeatureCache()
    assert cache.get("nonexistent") is None

def test_cache_clear():
    cache = FeatureCache()
    cache.store("test", np.array([1, 2, 3]))
    cache.clear()
    assert cache.get("test") is None

def test_cache_has():
    cache = FeatureCache()
    assert not cache.has("key")
    cache.store("key", "value")
    assert cache.has("key")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_hub_cache.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 FeatureCache**

```python
# src/feature_hub/__init__.py
from .hub import FeatureHub
from .cache import FeatureCache

__all__ = ["FeatureHub", "FeatureCache"]
```

```python
# src/feature_hub/cache.py
from __future__ import annotations
from typing import Any

class FeatureCache:
    """内存特征缓存，支持按 key 存取特征数据。"""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def store(self, key: str, data: Any) -> None:
        self._store[key] = data

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def has(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> list[str]:
        return list(self._store.keys())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_feature_hub_cache.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add src/feature_hub/__init__.py src/feature_hub/cache.py tests/test_feature_hub_cache.py
git commit -m "[add] FeatureHub 缓存层"
```

### Task 1.2: FeatureHub 核心调度器

**Files:**
- Create: `src/feature_hub/hub.py`
- Test: `tests/test_feature_hub.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feature_hub.py
import pytest
from unittest.mock import MagicMock, patch
from src.feature_hub.hub import FeatureHub

def test_hub_init():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    assert hub.video_path == "test.mp4"
    assert hub.device == "cpu"

def test_hub_register_and_get_extractor():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    mock_extractor = MagicMock(return_value={"data": [1, 2, 3]})
    hub.register_extractor("test_feature", mock_extractor)
    result = hub.get("test_feature")
    assert result == {"data": [1, 2, 3]}
    mock_extractor.assert_called_once()

def test_hub_caches_result():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    call_count = 0
    def extractor(video_path, device):
        nonlocal call_count
        call_count += 1
        return {"data": "result"}
    hub.register_extractor("feat", extractor)
    hub.get("feat")
    hub.get("feat")  # 第二次应走缓存
    assert call_count == 1

def test_hub_unknown_feature_raises():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    with pytest.raises(KeyError):
        hub.get("unknown_feature")

def test_hub_available_features():
    hub = FeatureHub(video_path="test.mp4", device="cpu")
    hub.register_extractor("a", lambda vp, d: None)
    hub.register_extractor("b", lambda vp, d: None)
    assert set(hub.available_features()) == {"a", "b"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_feature_hub.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 FeatureHub**

```python
# src/feature_hub/hub.py
from __future__ import annotations
from typing import Any, Callable
from .cache import FeatureCache

ExtractorFn = Callable[[str, str], Any]

class FeatureHub:
    """共享基础特征层：懒加载 + 缓存，避免各维度重复提取特征。"""

    def __init__(self, video_path: str, device: str = "cuda") -> None:
        self.video_path = video_path
        self.device = device
        self._cache = FeatureCache()
        self._extractors: dict[str, ExtractorFn] = {}

    def register_extractor(self, feature_name: str, extractor: ExtractorFn) -> None:
        self._extractors[feature_name] = extractor

    def get(self, feature_name: str) -> Any:
        if self._cache.has(feature_name):
            return self._cache.get(feature_name)
        if feature_name not in self._extractors:
            raise KeyError(f"Unknown feature: {feature_name}. Available: {list(self._extractors.keys())}")
        result = self._extractors[feature_name](self.video_path, self.device)
        self._cache.store(feature_name, result)
        return result

    def has_cached(self, feature_name: str) -> bool:
        return self._cache.has(feature_name)

    def available_features(self) -> list[str]:
        return list(self._extractors.keys())

    def clear_cache(self) -> None:
        self._cache.clear()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_feature_hub.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add src/feature_hub/hub.py tests/test_feature_hub.py
git commit -m "[add] FeatureHub 核心调度器"
```

### Task 1.3: 光流提取器适配

**Files:**
- Create: `src/feature_hub/extractors/__init__.py`
- Create: `src/feature_hub/extractors/optical_flow.py`
- Test: `tests/test_extractors_optical_flow.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_extractors_optical_flow.py
from unittest.mock import patch, MagicMock
import numpy as np
from src.feature_hub.extractors.optical_flow import extract_optical_flow

def test_extract_optical_flow_returns_list():
    """测试光流提取器返回正确格式。"""
    fake_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    with patch("src.feature_hub.extractors.optical_flow._load_frames", return_value=fake_frames):
        with patch("src.feature_hub.extractors.optical_flow._compute_flows") as mock_compute:
            mock_compute.return_value = [
                (np.zeros((100, 100)), np.zeros((100, 100)))
                for _ in range(2)
            ]
            result = extract_optical_flow("test.mp4", "cpu")
            assert isinstance(result, list)
            assert len(result) == 2
            assert isinstance(result[0], tuple)
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现光流提取器**

```python
# src/feature_hub/extractors/__init__.py
from .optical_flow import extract_optical_flow

__all__ = ["extract_optical_flow"]
```

```python
# src/feature_hub/extractors/optical_flow.py
from __future__ import annotations
from typing import Any
import numpy as np
import cv2

def _load_frames(video_path: str) -> list[np.ndarray]:
    """从视频文件加载所有帧。"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def _compute_flows(frames: list[np.ndarray], device: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """使用 Farneback 计算光流（轻量 fallback）。可替换为 RAFT。"""
    flows = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append((flow[..., 0], flow[..., 1]))
    return flows

def extract_optical_flow(video_path: str, device: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """FeatureHub 光流提取器入口。返回 [(u, v), ...] 光流序列。"""
    frames = _load_frames(video_path)
    if len(frames) < 2:
        return []
    return _compute_flows(frames, device)
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/feature_hub/extractors/ tests/test_extractors_optical_flow.py
git commit -m "[add] FeatureHub 光流提取器"
```

### Task 1.4: 人脸特征提取器

**Files:**
- Create: `src/feature_hub/extractors/face_embedding.py`
- Test: `tests/test_extractors_face.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_extractors_face.py
from unittest.mock import patch, MagicMock
import numpy as np
from src.feature_hub.extractors.face_embedding import extract_face_embeddings

def test_extract_face_embeddings_structure():
    """测试人脸特征提取器输出结构。"""
    with patch("src.feature_hub.extractors.face_embedding._load_frames") as mock_load:
        mock_load.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)]
        with patch("src.feature_hub.extractors.face_embedding._get_face_analyzer") as mock_fa:
            mock_face = MagicMock()
            mock_face.embedding = np.random.rand(512).astype(np.float32)
            mock_face.bbox = np.array([10, 10, 50, 50])
            mock_face.det_score = 0.95
            mock_analyzer = MagicMock()
            mock_analyzer.get.return_value = [mock_face]
            mock_fa.return_value = mock_analyzer
            result = extract_face_embeddings("test.mp4", "cpu")
            assert isinstance(result, list)
            assert len(result) == 1  # 1 frame
            assert "faces" in result[0]
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现人脸特征提取器**

```python
# src/feature_hub/extractors/face_embedding.py
from __future__ import annotations
from typing import Any
import numpy as np
import cv2

_face_analyzer = None

def _load_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def _get_face_analyzer(device: str) -> Any:
    global _face_analyzer
    if _face_analyzer is None:
        from insightface.app import FaceAnalysis
        ctx_id = 0 if "cuda" in device else -1
        _face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"],
        )
        _face_analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return _face_analyzer

def extract_face_embeddings(video_path: str, device: str) -> list[dict]:
    """提取每帧的人脸检测结果和 ArcFace embedding。

    Returns:
        list[dict]: 每帧一个 dict，包含 'faces' 列表，
        每张脸有 'embedding'(512-d), 'bbox', 'det_score'。
    """
    frames = _load_frames(video_path)
    analyzer = _get_face_analyzer(device)
    results = []
    for frame in frames:
        faces_raw = analyzer.get(frame)
        faces = []
        for f in faces_raw:
            faces.append({
                "embedding": f.embedding / np.linalg.norm(f.embedding),
                "bbox": f.bbox.tolist(),
                "det_score": float(f.det_score),
            })
        results.append({"faces": faces, "num_faces": len(faces)})
    return results
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/feature_hub/extractors/face_embedding.py tests/test_extractors_face.py
git commit -m "[add] FeatureHub 人脸特征提取器（InsightFace ArcFace）"
```

### Task 1.5: 深度图提取器

**Files:**
- Create: `src/feature_hub/extractors/depth.py`
- Test: `tests/test_extractors_depth.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_extractors_depth.py
from unittest.mock import patch, MagicMock
import numpy as np
from src.feature_hub.extractors.depth import extract_depth_maps

def test_extract_depth_maps_structure():
    with patch("src.feature_hub.extractors.depth._load_frames") as mock_load:
        mock_load.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        with patch("src.feature_hub.extractors.depth._get_depth_model") as mock_model:
            mock_m = MagicMock()
            mock_m.return_value = MagicMock(squeeze=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.zeros((100, 100)))))
            )))
            mock_model.return_value = (mock_m, MagicMock(return_value=MagicMock(to=MagicMock(return_value=None))))
            result = extract_depth_maps("test.mp4", "cpu")
            assert isinstance(result, list)
            assert len(result) == 3
```

- [ ] **Step 2-3: 实现深度图提取器**

```python
# src/feature_hub/extractors/depth.py
from __future__ import annotations
import numpy as np
import cv2
import torch

_depth_model = None
_depth_transform = None

def _load_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def _get_depth_model(device: str) -> tuple:
    global _depth_model, _depth_transform
    if _depth_model is None:
        _depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        _depth_model.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _depth_transform = midas_transforms.dpt_transform
    return _depth_model, _depth_transform

def extract_depth_maps(video_path: str, device: str) -> list[np.ndarray]:
    """提取每帧深度图。返回 list[np.ndarray]，每个形状 (H, W)。"""
    frames = _load_frames(video_path)
    model, transform = _get_depth_model(device)
    depths = []
    for frame in frames:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
        depth = prediction.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        depths.append(depth)
    return depths
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/feature_hub/extractors/depth.py tests/test_extractors_depth.py
git commit -m "[add] FeatureHub 深度图提取器（MiDaS）"
```

### Task 1.6: 更新 extractors __init__ 并注册默认提取器

**Files:**
- Modify: `src/feature_hub/extractors/__init__.py`
- Modify: `src/feature_hub/hub.py` — 添加 `create_default_hub` 工厂函数

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feature_hub_factory.py
from src.feature_hub.hub import create_default_hub

def test_create_default_hub_has_extractors():
    hub = create_default_hub("test.mp4", device="cpu")
    features = hub.available_features()
    assert "optical_flow" in features
    assert "face_embedding" in features
    assert "depth" in features
```

- [ ] **Step 2-3: 实现工厂函数**

在 `hub.py` 末尾添加:

```python
def create_default_hub(video_path: str, device: str = "cuda") -> FeatureHub:
    """创建预注册所有默认提取器的 FeatureHub。"""
    from .extractors.optical_flow import extract_optical_flow
    from .extractors.face_embedding import extract_face_embeddings
    from .extractors.depth import extract_depth_maps

    hub = FeatureHub(video_path, device)
    hub.register_extractor("optical_flow", extract_optical_flow)
    hub.register_extractor("face_embedding", extract_face_embeddings)
    hub.register_extractor("depth", extract_depth_maps)
    return hub
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/feature_hub/ tests/test_feature_hub_factory.py
git commit -m "[add] FeatureHub 默认提取器注册工厂函数"
```

---

## Chunk 2: MLLM 统一调用层

维度 4 和维度 5 共用的 MLLM 推理层，需在维度实现前完成。

### Task 2.1: MLLM Client 核心

**Files:**
- Create: `src/mllm/__init__.py`
- Create: `src/mllm/client.py`
- Create: `src/mllm/config.py`
- Test: `tests/test_mllm_client.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_mllm_client.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.mllm.client import MLLMClient
from src.mllm.config import MLLMConfig

def test_mllm_client_init():
    config = MLLMConfig(backend="api", api_provider="openai", api_model="gpt-4o")
    client = MLLMClient(config)
    assert client.config.backend == "api"

def test_mllm_client_api_call():
    config = MLLMConfig(backend="api", api_provider="openai", api_model="gpt-4o", api_key="test")
    client = MLLMClient(config)
    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    with patch.object(client, "_call_api", return_value={"is_natural": True}):
        result = client.judge_video_clip(frames, "Is this natural?")
        assert result == {"is_natural": True}

def test_mllm_client_fallback():
    config = MLLMConfig(backend="hybrid", local_model="test", api_provider="openai", api_model="gpt-4o", api_key="test")
    client = MLLMClient(config)
    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    with patch.object(client, "_call_local", side_effect=RuntimeError("GPU OOM")):
        with patch.object(client, "_call_api", return_value={"fallback": True}):
            result = client.judge_with_fallback(frames, "test prompt")
            assert result == {"fallback": True}
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现 MLLM Client**

```python
# src/mllm/__init__.py
from .client import MLLMClient
from .config import MLLMConfig

__all__ = ["MLLMClient", "MLLMConfig"]
```

```python
# src/mllm/config.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class MLLMConfig:
    backend: str = "api"  # "local", "api", "hybrid"
    local_model: str = "InternVL2-8B"
    local_model_path: str | None = None
    api_provider: str = "openai"  # "openai", "anthropic"
    api_model: str = "gpt-4o"
    api_key: str | None = None
    api_base_url: str | None = None
    max_frames: int = 16
    temperature: float = 0.1
    device: str = "cuda"
```

```python
# src/mllm/client.py
from __future__ import annotations
import base64
import json
from typing import Any
import numpy as np
import cv2
from .config import MLLMConfig

class MLLMClient:
    """统一 MLLM 调用接口，支持本地模型和 API 切换。"""

    def __init__(self, config: MLLMConfig) -> None:
        self.config = config
        self._local_model = None

    def judge_video_clip(self, frames: list[np.ndarray], prompt: str) -> dict:
        if self.config.backend == "local":
            return self._call_local(frames, prompt)
        elif self.config.backend == "api":
            return self._call_api(frames, prompt)
        else:  # hybrid
            return self.judge_with_fallback(frames, prompt)

    def judge_with_fallback(self, frames: list[np.ndarray], prompt: str) -> dict:
        try:
            return self._call_local(frames, prompt)
        except Exception:
            return self._call_api(frames, prompt)

    def _call_api(self, frames: list[np.ndarray], prompt: str) -> dict:
        images_b64 = self._encode_frames(frames)
        if self.config.api_provider == "openai":
            return self._call_openai(images_b64, prompt)
        elif self.config.api_provider == "anthropic":
            return self._call_anthropic(images_b64, prompt)
        raise ValueError(f"Unknown provider: {self.config.api_provider}")

    def _call_local(self, frames: list[np.ndarray], prompt: str) -> dict:
        raise NotImplementedError("Local MLLM not yet integrated")

    def _encode_frames(self, frames: list[np.ndarray]) -> list[str]:
        step = max(1, len(frames) // self.config.max_frames)
        sampled = frames[::step][:self.config.max_frames]
        encoded = []
        for f in sampled:
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            encoded.append(base64.b64encode(buf).decode("utf-8"))
        return encoded

    def _call_openai(self, images_b64: list[str], prompt: str) -> dict:
        import openai
        client = openai.OpenAI(api_key=self.config.api_key, base_url=self.config.api_base_url)
        content = [{"type": "text", "text": prompt}]
        for img in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        resp = client.chat.completions.create(
            model=self.config.api_model,
            messages=[{"role": "user", "content": content}],
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    def _call_anthropic(self, images_b64: list[str], prompt: str) -> dict:
        import anthropic
        client = anthropic.Anthropic(api_key=self.config.api_key)
        content = []
        for img in images_b64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}})
        content.append({"type": "text", "text": prompt + "\n\n请以 JSON 格式回答。"})
        resp = client.messages.create(
            model=self.config.api_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return json.loads(resp.content[0].text)
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/mllm/ tests/test_mllm_client.py
git commit -m "[add] MLLM 统一调用层（支持 OpenAI/Anthropic API + hybrid fallback）"
```

### Task 2.2: MLLM Prompt 模板

**Files:**
- Create: `src/mllm/prompts/__init__.py`
- Create: `src/mllm/prompts/motion_naturalness.py`
- Create: `src/mllm/prompts/physics_commonsense.py`

- [ ] **Step 1: 创建 prompt 模板**

```python
# src/mllm/prompts/__init__.py
from .motion_naturalness import MOTION_NATURALNESS_PROMPT
from .physics_commonsense import PHYSICS_COMMONSENSE_PROMPT

__all__ = ["MOTION_NATURALNESS_PROMPT", "PHYSICS_COMMONSENSE_PROMPT"]
```

```python
# src/mllm/prompts/motion_naturalness.py
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
```

```python
# src/mllm/prompts/physics_commonsense.py
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
```

- [ ] **Step 2: 提交**

```bash
git add src/mllm/prompts/
git commit -m "[add] MLLM prompt 模板（运动自然度 + 物理常识）"
```

---

## Chunk 3: D1 人脸身份一致性

### Task 3.1: CSIM 评分器

**Files:**
- Create: `src/face_identity/__init__.py`
- Create: `src/face_identity/csim_scorer.py`
- Create: `src/face_identity/config.py`
- Test: `tests/test_csim_scorer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_csim_scorer.py
import numpy as np
from src.face_identity.csim_scorer import CSIMScorer

def _make_embeddings(n: int, noise: float = 0.01) -> list[np.ndarray]:
    """生成 n 个相似的 512-d 归一化向量。"""
    base = np.random.rand(512).astype(np.float32)
    base /= np.linalg.norm(base)
    embs = []
    for _ in range(n):
        e = base + np.random.randn(512).astype(np.float32) * noise
        e /= np.linalg.norm(e)
        embs.append(e)
    return embs

def test_csim_ref_high_consistency():
    embs = _make_embeddings(10, noise=0.01)
    scorer = CSIMScorer()
    result = scorer.compute(embs)
    assert result.csim_ref > 0.9

def test_csim_adj_high_consistency():
    embs = _make_embeddings(10, noise=0.01)
    scorer = CSIMScorer()
    result = scorer.compute(embs)
    assert result.csim_adj > 0.9

def test_csim_detects_drop():
    embs = _make_embeddings(10, noise=0.01)
    # 在第5帧插入一个完全不同的 embedding
    embs[5] = np.random.rand(512).astype(np.float32)
    embs[5] /= np.linalg.norm(embs[5])
    scorer = CSIMScorer()
    result = scorer.compute(embs)
    assert len(result.drop_events) > 0
    assert result.csim_min < 0.5

def test_csim_identity_score():
    embs = _make_embeddings(10, noise=0.01)
    scorer = CSIMScorer()
    result = scorer.compute(embs)
    assert 0.0 <= result.identity_score <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现 CSIMScorer**

```python
# src/face_identity/__init__.py
from .analyzer import FaceIdentityAnalyzer

__all__ = ["FaceIdentityAnalyzer"]
```

```python
# src/face_identity/config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class FaceIdentityConfig:
    csim_ref_weight: float = 0.4
    csim_adj_weight: float = 0.3
    csim_min_weight: float = 0.2
    drop_penalty_weight: float = 0.1
    drop_threshold: float = 0.3  # 相邻帧相似度下降超过此值判定为 drop
    drop_window: int = 3  # 滑动窗口大小
```

```python
# src/face_identity/csim_scorer.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .config import FaceIdentityConfig

@dataclass
class DropEvent:
    frame_idx: int
    similarity_before: float
    similarity_after: float
    drop_magnitude: float

@dataclass
class CSIMResult:
    csim_ref: float
    csim_adj: float
    csim_min: float
    drop_events: list[DropEvent]
    identity_score: float

class CSIMScorer:
    """CSIM 多模式人脸身份一致性评分器。"""

    def __init__(self, config: FaceIdentityConfig | None = None) -> None:
        self.config = config or FaceIdentityConfig()

    def compute(self, embeddings: list[np.ndarray]) -> CSIMResult:
        if len(embeddings) < 2:
            return CSIMResult(1.0, 1.0, 1.0, [], 1.0)

        ref = embeddings[0]
        # CSIM-Ref: 所有帧 vs 首帧
        ref_sims = [float(np.dot(ref, e)) for e in embeddings[1:]]
        csim_ref = float(np.mean(ref_sims)) if ref_sims else 1.0

        # CSIM-Adj: 相邻帧对
        adj_sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
        csim_adj = float(np.mean(adj_sims))

        # CSIM-Min
        all_sims = ref_sims + adj_sims
        csim_min = float(np.min(all_sims)) if all_sims else 1.0

        # CSIM-Drop: 滑动窗口检测骤降
        drop_events = self._detect_drops(adj_sims)

        # 综合评分
        c = self.config
        drop_penalty = len(drop_events) / max(len(embeddings) - 1, 1)
        identity_score = (
            c.csim_ref_weight * max(csim_ref, 0)
            + c.csim_adj_weight * max(csim_adj, 0)
            + c.csim_min_weight * max(csim_min, 0)
            - c.drop_penalty_weight * drop_penalty
        )
        identity_score = float(np.clip(identity_score, 0, 1))

        return CSIMResult(csim_ref, csim_adj, csim_min, drop_events, identity_score)

    def _detect_drops(self, adj_sims: list[float]) -> list[DropEvent]:
        drops = []
        w = self.config.drop_window
        for i in range(len(adj_sims)):
            start = max(0, i - w)
            window_mean = float(np.mean(adj_sims[start:i])) if i > 0 else 1.0
            drop_mag = window_mean - adj_sims[i]
            if drop_mag > self.config.drop_threshold:
                drops.append(DropEvent(
                    frame_idx=i + 1,
                    similarity_before=window_mean,
                    similarity_after=adj_sims[i],
                    drop_magnitude=drop_mag,
                ))
        return drops
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/face_identity/ tests/test_csim_scorer.py
git commit -m "[add] D1: CSIM 多模式人脸身份一致性评分器"
```

### Task 3.2: 人脸追踪器

**Files:**
- Create: `src/face_identity/face_tracker.py`
- Test: `tests/test_face_tracker.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_face_tracker.py
import numpy as np
from src.face_identity.face_tracker import FaceTracker, FaceTrack

def _make_frame_data(n_frames: int, n_faces: int = 1) -> list[dict]:
    """模拟 FeatureHub face_embedding 输出。"""
    base_embs = []
    for _ in range(n_faces):
        e = np.random.rand(512).astype(np.float32)
        base_embs.append(e / np.linalg.norm(e))
    frames = []
    for _ in range(n_frames):
        faces = []
        for j in range(n_faces):
            e = base_embs[j] + np.random.randn(512).astype(np.float32) * 0.01
            e /= np.linalg.norm(e)
            faces.append({"embedding": e, "bbox": [10+j*60, 10, 50+j*60, 50], "det_score": 0.95})
        frames.append({"faces": faces, "num_faces": n_faces})
    return frames

def test_tracker_single_face():
    data = _make_frame_data(10, n_faces=1)
    tracker = FaceTracker()
    tracks = tracker.track(data)
    assert len(tracks) == 1
    assert len(tracks[0].embeddings) == 10

def test_tracker_multi_face():
    data = _make_frame_data(10, n_faces=2)
    tracker = FaceTracker()
    tracks = tracker.track(data)
    assert len(tracks) == 2

def test_tracker_no_faces():
    data = [{"faces": [], "num_faces": 0} for _ in range(5)]
    tracker = FaceTracker()
    tracks = tracker.track(data)
    assert len(tracks) == 0
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现 FaceTracker**

```python
# src/face_identity/face_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class FaceTrack:
    track_id: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

class FaceTracker:
    """基于匈牙利算法的跨帧人脸身份关联。"""

    def __init__(self, match_threshold: float = 0.4) -> None:
        self.match_threshold = match_threshold

    def track(self, frame_data: list[dict]) -> list[FaceTrack]:
        tracks: list[FaceTrack] = []
        next_id = 0
        active_tracks: list[int] = []  # 当前活跃轨迹的索引

        for frame_idx, fd in enumerate(frame_data):
            faces = fd.get("faces", [])
            if not faces:
                active_tracks = []
                continue
            if not active_tracks:
                for f in faces:
                    tracks.append(FaceTrack(track_id=next_id, embeddings=[f["embedding"]], frame_indices=[frame_idx]))
                    active_tracks.append(len(tracks) - 1)
                    next_id += 1
                continue

            # 构建代价矩阵
            cost = np.zeros((len(active_tracks), len(faces)))
            for i, ti in enumerate(active_tracks):
                last_emb = tracks[ti].embeddings[-1]
                for j, f in enumerate(faces):
                    cost[i, j] = 1.0 - float(np.dot(last_emb, f["embedding"]))

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_faces = set()
            new_active = []
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - self.match_threshold):
                    ti = active_tracks[r]
                    tracks[ti].embeddings.append(faces[c]["embedding"])
                    tracks[ti].frame_indices.append(frame_idx)
                    new_active.append(ti)
                    matched_faces.add(c)

            for j, f in enumerate(faces):
                if j not in matched_faces:
                    tracks.append(FaceTrack(track_id=next_id, embeddings=[f["embedding"]], frame_indices=[frame_idx]))
                    new_active.append(len(tracks) - 1)
                    next_id += 1
            active_tracks = new_active

        return tracks
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/face_identity/face_tracker.py tests/test_face_tracker.py
git commit -m "[add] D1: 人脸跨帧追踪器（匈牙利算法）"
```

### Task 3.3: FaceIdentityAnalyzer 主入口

**Files:**
- Create: `src/face_identity/analyzer.py`
- Test: `tests/test_face_identity_analyzer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_face_identity_analyzer.py
import numpy as np
from unittest.mock import MagicMock
from src.face_identity.analyzer import FaceIdentityAnalyzer

def test_analyzer_with_faces():
    hub = MagicMock()
    base = np.random.rand(512).astype(np.float32)
    base /= np.linalg.norm(base)
    hub.get.return_value = [
        {"faces": [{"embedding": base + np.random.randn(512).astype(np.float32) * 0.01,
                     "bbox": [0,0,50,50], "det_score": 0.9}], "num_faces": 1}
        for _ in range(10)
    ]
    # normalize
    for fd in hub.get.return_value:
        for f in fd["faces"]:
            f["embedding"] /= np.linalg.norm(f["embedding"])

    analyzer = FaceIdentityAnalyzer()
    result = analyzer.analyze(hub)
    assert result.applicable is True
    assert result.identity_score > 0.5

def test_analyzer_no_faces():
    hub = MagicMock()
    hub.get.return_value = [{"faces": [], "num_faces": 0} for _ in range(10)]
    analyzer = FaceIdentityAnalyzer()
    result = analyzer.analyze(hub)
    assert result.applicable is False
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现 FaceIdentityAnalyzer**

```python
# src/face_identity/analyzer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .config import FaceIdentityConfig
from .face_tracker import FaceTracker, FaceTrack
from .csim_scorer import CSIMScorer, CSIMResult, DropEvent

@dataclass
class FaceIdentityResult:
    applicable: bool
    skip_reason: str | None = None
    face_tracks: list[FaceTrack] = field(default_factory=list)
    csim_ref: float = 0.0
    csim_adj: float = 0.0
    csim_min: float = 0.0
    drop_events: list[DropEvent] = field(default_factory=list)
    identity_score: float = 0.0

class FaceIdentityAnalyzer:
    """D1: 人脸身份一致性分析器。"""

    def __init__(self, config: FaceIdentityConfig | None = None) -> None:
        self.config = config or FaceIdentityConfig()
        self._tracker = FaceTracker()
        self._scorer = CSIMScorer(self.config)

    def analyze(self, hub: Any) -> FaceIdentityResult:
        frame_data = hub.get("face_embedding")
        total_faces = sum(fd["num_faces"] for fd in frame_data)
        if total_faces == 0:
            return FaceIdentityResult(applicable=False, skip_reason="no face detected")

        tracks = self._tracker.track(frame_data)
        if not tracks:
            return FaceIdentityResult(applicable=False, skip_reason="no face tracks")

        # 对最长轨迹计算 CSIM（主要人物）
        main_track = max(tracks, key=lambda t: len(t.embeddings))
        csim = self._scorer.compute(main_track.embeddings)

        return FaceIdentityResult(
            applicable=True,
            face_tracks=tracks,
            csim_ref=csim.csim_ref,
            csim_adj=csim.csim_adj,
            csim_min=csim.csim_min,
            drop_events=csim.drop_events,
            identity_score=csim.identity_score,
        )
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/face_identity/analyzer.py tests/test_face_identity_analyzer.py
git commit -m "[add] D1: FaceIdentityAnalyzer 主入口"
```

---

## Chunk 4: D2 表情自然度

### Task 4.1: AU 提取器（Py-Feat 封装）

**Files:**
- Create: `src/expression_naturalness/__init__.py`
- Create: `src/expression_naturalness/config.py`
- Create: `src/expression_naturalness/au_extractor.py`
- Test: `tests/test_au_extractor.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_au_extractor.py
import numpy as np
from unittest.mock import patch, MagicMock
from src.expression_naturalness.au_extractor import AUExtractor

def test_au_extractor_returns_au_dict():
    extractor = AUExtractor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch.object(extractor, "_detector") as mock_det:
        mock_result = MagicMock()
        mock_result.aus = MagicMock()
        mock_result.aus.values = [[0.5, 1.2, 0.3]]
        mock_result.aus.columns = ["AU01", "AU02", "AU04"]
        mock_det.detect_image.return_value = mock_result
        result = extractor.extract(frame)
        assert isinstance(result, dict)
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现 AUExtractor**

```python
# src/expression_naturalness/__init__.py
from .analyzer import ExpressionAnalyzer
__all__ = ["ExpressionAnalyzer"]
```

```python
# src/expression_naturalness/config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ExpressionConfig:
    au_smoothness_window: int = 5
    au_jump_threshold: float = 1.5  # AU 强度突变阈值
    flow_consistency_weight: float = 0.3
    au_combination_weight: float = 0.4
    au_smoothness_weight: float = 0.3
```

```python
# src/expression_naturalness/au_extractor.py
from __future__ import annotations
import numpy as np

class AUExtractor:
    """Py-Feat 封装的 AU 提取器。"""

    def __init__(self) -> None:
        self._detector = None

    def _ensure_detector(self) -> None:
        if self._detector is None:
            from feat import Detector
            self._detector = Detector(au_model="xgb")

    def extract(self, frame: np.ndarray) -> dict[str, float]:
        self._ensure_detector()
        result = self._detector.detect_image(frame)
        if result.aus is None or len(result.aus) == 0:
            return {}
        au_values = result.aus.values[0]
        au_names = list(result.aus.columns)
        return {name: float(val) for name, val in zip(au_names, au_values)}

    def extract_sequence(self, frames: list[np.ndarray]) -> list[dict[str, float]]:
        return [self.extract(f) for f in frames]
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/expression_naturalness/ tests/test_au_extractor.py
git commit -m "[add] D2: Py-Feat AU 提取器"
```

### Task 4.2: AU 规则库 + 时序分析 + Analyzer

**Files:**
- Create: `src/expression_naturalness/au_rules.py`
- Create: `src/expression_naturalness/temporal_analysis.py`
- Create: `src/expression_naturalness/analyzer.py`
- Test: `tests/test_expression_analyzer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_expression_analyzer.py
from src.expression_naturalness.au_rules import check_au_combination, CONFLICT_PAIRS
from src.expression_naturalness.temporal_analysis import compute_au_smoothness

def test_conflict_detection():
    # AU23 (紧唇) + AU26 (张口) 互斥
    aus = {"AU23": 2.0, "AU26": 3.0}
    violations = check_au_combination(aus)
    assert len(violations) > 0

def test_no_conflict():
    aus = {"AU06": 2.0, "AU12": 3.0}  # 真笑
    violations = check_au_combination(aus)
    assert len(violations) == 0

def test_au_smoothness_stable():
    # 稳定的 AU 序列 → 高平滑度
    sequence = [1.0, 1.1, 1.0, 0.9, 1.0, 1.1]
    score = compute_au_smoothness(sequence)
    assert score > 0.8

def test_au_smoothness_jumpy():
    # 剧烈跳变 → 低平滑度
    sequence = [0.0, 4.0, 0.0, 4.0, 0.0, 4.0]
    score = compute_au_smoothness(sequence)
    assert score < 0.5
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现规则库和时序分析**

```python
# src/expression_naturalness/au_rules.py
from __future__ import annotations
from dataclasses import dataclass

NATURAL_EXPRESSIONS = {
    "genuine_smile": {"required": ["AU06", "AU12"], "forbidden": []},
    "surprise": {"required": ["AU01", "AU02", "AU05", "AU26"], "forbidden": []},
    "frown": {"required": ["AU04"], "forbidden": ["AU12"]},
    "fear": {"required": ["AU01", "AU02", "AU04", "AU20"], "forbidden": []},
}

CONFLICT_PAIRS = [
    (["AU01", "AU02"], ["AU04"]),  # 眉毛上扬与皱眉
    (["AU23"], ["AU26"]),           # 紧唇与张口
]

AU_ACTIVATION_THRESHOLD = 1.0

@dataclass
class Violation:
    violation_type: str
    description: str
    involved_aus: list[str]

def check_au_combination(aus: dict[str, float]) -> list[Violation]:
    violations = []
    active = {k for k, v in aus.items() if v >= AU_ACTIVATION_THRESHOLD}
    for group_a, group_b in CONFLICT_PAIRS:
        a_active = any(au in active for au in group_a)
        b_active = any(au in active for au in group_b)
        if a_active and b_active:
            violations.append(Violation(
                violation_type="conflict",
                description=f"Conflicting AUs: {group_a} vs {group_b}",
                involved_aus=group_a + group_b,
            ))
    return violations
```

```python
# src/expression_naturalness/temporal_analysis.py
from __future__ import annotations
import numpy as np

def compute_au_smoothness(sequence: list[float], window: int = 3) -> float:
    if len(sequence) < 2:
        return 1.0
    arr = np.array(sequence)
    diffs = np.abs(np.diff(arr))
    max_possible_diff = 5.0  # AU 强度范围 0~5
    normalized_diffs = diffs / max_possible_diff
    smoothness = 1.0 - float(np.mean(normalized_diffs))
    return float(np.clip(smoothness, 0, 1))

def compute_all_au_smoothness(au_sequences: dict[str, list[float]]) -> dict[str, float]:
    return {au: compute_au_smoothness(seq) for au, seq in au_sequences.items() if len(seq) >= 2}
```

```python
# src/expression_naturalness/analyzer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .config import ExpressionConfig
from .au_extractor import AUExtractor
from .au_rules import check_au_combination, Violation
from .temporal_analysis import compute_all_au_smoothness

@dataclass
class ExpressionResult:
    applicable: bool
    skip_reason: str | None = None
    au_sequences: dict[str, list[float]] = field(default_factory=dict)
    combination_violations: list[Violation] = field(default_factory=list)
    temporal_smoothness: float = 0.0
    expression_score: float = 0.0

class ExpressionAnalyzer:
    """D2: 表情与肌肉运动自然度分析器。"""

    def __init__(self, config: ExpressionConfig | None = None) -> None:
        self.config = config or ExpressionConfig()
        self._extractor = AUExtractor()

    def analyze(self, hub: Any) -> ExpressionResult:
        face_data = hub.get("face_embedding")
        has_faces = any(fd["num_faces"] > 0 for fd in face_data)
        if not has_faces:
            return ExpressionResult(applicable=False, skip_reason="no face detected")

        # 需要原始帧来提取 AU（FeatureHub 应能提供）
        try:
            frames = hub.get("video_frames")
        except KeyError:
            return ExpressionResult(applicable=False, skip_reason="video_frames not available")

        au_per_frame = self._extractor.extract_sequence(frames)
        if not any(au_per_frame):
            return ExpressionResult(applicable=False, skip_reason="no AU detected")

        # 构建 AU 时序
        all_aus = set()
        for aus in au_per_frame:
            all_aus.update(aus.keys())
        au_sequences = {au: [f.get(au, 0.0) for f in au_per_frame] for au in all_aus}

        # AU 组合异常
        violations = []
        for aus in au_per_frame:
            violations.extend(check_au_combination(aus))

        # 时序平滑度
        smoothness_scores = compute_all_au_smoothness(au_sequences)
        temporal_smoothness = float(np.mean(list(smoothness_scores.values()))) if smoothness_scores else 1.0

        # 综合评分
        c = self.config
        violation_penalty = min(len(violations) / max(len(au_per_frame), 1), 1.0)
        expression_score = (
            c.au_combination_weight * (1.0 - violation_penalty)
            + c.au_smoothness_weight * temporal_smoothness
            + c.flow_consistency_weight * 1.0  # 光流一致性暂用 1.0 占位
        )
        expression_score = float(np.clip(expression_score, 0, 1))

        return ExpressionResult(
            applicable=True,
            au_sequences=au_sequences,
            combination_violations=violations,
            temporal_smoothness=temporal_smoothness,
            expression_score=expression_score,
        )
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/expression_naturalness/ tests/test_expression_analyzer.py
git commit -m "[add] D2: AU 规则库 + 时序分析 + ExpressionAnalyzer"
```

---

## Chunk 5: D3 生物特征异常

### Task 5.1: 异常规则库 + 眼睛/手指/口腔检测器

**Files:**
- Create: `src/biological_anomaly/__init__.py`
- Create: `src/biological_anomaly/config.py`
- Create: `src/biological_anomaly/anomaly_rules.py`
- Create: `src/biological_anomaly/eye_anomaly.py`
- Create: `src/biological_anomaly/hand_anomaly.py`
- Create: `src/biological_anomaly/mouth_anomaly.py`
- Test: `tests/test_biological_anomaly.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_biological_anomaly.py
from src.biological_anomaly.anomaly_rules import HAND_CONSTRAINTS, EYE_CONSTRAINTS
from src.biological_anomaly.eye_anomaly import detect_eye_anomalies
from src.biological_anomaly.hand_anomaly import detect_hand_anomalies

def test_eye_constraints_exist():
    assert "ear_blink_threshold" in EYE_CONSTRAINTS
    assert "max_no_blink_frames" in EYE_CONSTRAINTS

def test_hand_constraints_exist():
    assert "finger_count" in HAND_CONSTRAINTS
    assert HAND_CONSTRAINTS["finger_count"] == 5

def test_eye_no_blink_detected():
    # EAR 值恒定高 → 不眨眼异常
    ear_sequence = [0.35] * 100  # 100帧不眨眼
    anomalies = detect_eye_anomalies(ear_sequence, fps=30.0)
    assert any(a["type"] == "no_blink" for a in anomalies)

def test_eye_normal_blink():
    # 正常眨眼模式
    ear_seq = [0.35] * 20 + [0.15, 0.10, 0.15] + [0.35] * 20
    anomalies = detect_eye_anomalies(ear_seq, fps=30.0)
    no_blink = [a for a in anomalies if a["type"] == "no_blink"]
    assert len(no_blink) == 0

def test_hand_wrong_finger_count():
    finger_counts = [5, 5, 6, 5, 5]  # 第3帧6根手指
    anomalies = detect_hand_anomalies(finger_counts=finger_counts)
    assert any(a["type"] == "wrong_finger_count" for a in anomalies)
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现**

```python
# src/biological_anomaly/__init__.py
from .analyzer import BiologicalAnomalyAnalyzer
__all__ = ["BiologicalAnomalyAnalyzer"]
```

```python
# src/biological_anomaly/config.py
from dataclasses import dataclass

@dataclass
class BiologicalAnomalyConfig:
    ear_blink_threshold: float = 0.21
    max_no_blink_frames: int = 90
    eye_symmetry_tolerance: float = 0.15
    finger_count_expected: int = 5
    joint_angle_range: tuple[float, float] = (0, 180)
    bone_length_ratio_tolerance: float = 0.15
```

```python
# src/biological_anomaly/anomaly_rules.py
HAND_CONSTRAINTS = {
    "finger_count": 5,
    "joint_angle_range": (0, 180),
    "thumb_angle_range": (-30, 130),
    "bone_length_ratio_tolerance": 0.15,
}
EYE_CONSTRAINTS = {
    "ear_blink_threshold": 0.21,
    "max_no_blink_frames": 90,
    "symmetry_tolerance": 0.15,
}
```

```python
# src/biological_anomaly/eye_anomaly.py
from __future__ import annotations
from .anomaly_rules import EYE_CONSTRAINTS

def detect_eye_anomalies(
    ear_sequence: list[float],
    fps: float = 30.0,
    constraints: dict | None = None,
) -> list[dict]:
    c = constraints or EYE_CONSTRAINTS
    threshold = c["ear_blink_threshold"]
    max_no_blink = c["max_no_blink_frames"]
    anomalies = []
    consecutive_open = 0
    for i, ear in enumerate(ear_sequence):
        if ear > threshold:
            consecutive_open += 1
        else:
            consecutive_open = 0
        if consecutive_open >= max_no_blink:
            anomalies.append({
                "type": "no_blink",
                "frame_idx": i,
                "duration_frames": consecutive_open,
                "description": f"No blink for {consecutive_open} frames ({consecutive_open/fps:.1f}s)",
            })
            consecutive_open = 0  # 只报一次
    return anomalies
```

```python
# src/biological_anomaly/hand_anomaly.py
from __future__ import annotations
from .anomaly_rules import HAND_CONSTRAINTS

def detect_hand_anomalies(
    finger_counts: list[int] | None = None,
    joint_angles: list[list[float]] | None = None,
) -> list[dict]:
    anomalies = []
    expected = HAND_CONSTRAINTS["finger_count"]
    if finger_counts:
        for i, count in enumerate(finger_counts):
            if count != expected:
                anomalies.append({
                    "type": "wrong_finger_count",
                    "frame_idx": i,
                    "expected": expected,
                    "actual": count,
                    "description": f"Frame {i}: {count} fingers (expected {expected})",
                })
    if joint_angles:
        lo, hi = HAND_CONSTRAINTS["joint_angle_range"]
        for i, angles in enumerate(joint_angles):
            for j, angle in enumerate(angles):
                if angle < lo or angle > hi:
                    anomalies.append({
                        "type": "impossible_joint_angle",
                        "frame_idx": i,
                        "joint_idx": j,
                        "angle": angle,
                        "description": f"Frame {i}: joint {j} angle {angle:.1f}° out of range [{lo}, {hi}]",
                    })
    return anomalies
```

```python
# src/biological_anomaly/mouth_anomaly.py
from __future__ import annotations

def detect_mouth_anomalies(frames: list = None) -> list[dict]:
    """口腔异常检测占位。需要面部 landmark 的口腔区域分割。"""
    return []
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/biological_anomaly/ tests/test_biological_anomaly.py
git commit -m "[add] D3: 生物特征异常检测规则库 + 眼睛/手指检测器"
```

### Task 5.2: BiologicalAnomalyAnalyzer 主入口

**Files:**
- Create: `src/biological_anomaly/analyzer.py`
- Test: `tests/test_bio_analyzer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_bio_analyzer.py
from unittest.mock import MagicMock
from src.biological_anomaly.analyzer import BiologicalAnomalyAnalyzer

def test_bio_analyzer_no_faces_no_hands():
    hub = MagicMock()
    hub.get.return_value = [{"faces": [], "num_faces": 0} for _ in range(5)]
    analyzer = BiologicalAnomalyAnalyzer()
    result = analyzer.analyze(hub)
    assert result.applicable is False

def test_bio_analyzer_returns_score():
    hub = MagicMock()
    hub.get.side_effect = lambda key: {
        "face_embedding": [{"faces": [{"bbox": [0,0,50,50]}], "num_faces": 1} for _ in range(5)],
    }.get(key, [])
    analyzer = BiologicalAnomalyAnalyzer()
    result = analyzer.analyze(hub)
    assert 0.0 <= result.bio_quality_score <= 1.0
```

- [ ] **Step 2-3: 实现 Analyzer**

```python
# src/biological_anomaly/analyzer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .config import BiologicalAnomalyConfig
from .eye_anomaly import detect_eye_anomalies
from .hand_anomaly import detect_hand_anomalies
from .mouth_anomaly import detect_mouth_anomalies

@dataclass
class BiologicalAnomalyResult:
    applicable: bool
    skip_reason: str | None = None
    eye_anomalies: list[dict] = field(default_factory=list)
    hand_anomalies: list[dict] = field(default_factory=list)
    mouth_anomalies: list[dict] = field(default_factory=list)
    anomaly_count: int = 0
    bio_quality_score: float = 1.0

class BiologicalAnomalyAnalyzer:
    """D3: 生物特征细节异常分析器。"""

    def __init__(self, config: BiologicalAnomalyConfig | None = None) -> None:
        self.config = config or BiologicalAnomalyConfig()

    def analyze(self, hub: Any) -> BiologicalAnomalyResult:
        face_data = hub.get("face_embedding")
        has_faces = any(fd["num_faces"] > 0 for fd in face_data)
        if not has_faces:
            return BiologicalAnomalyResult(applicable=False, skip_reason="no face or hand detected")

        eye_anomalies: list[dict] = []
        hand_anomalies: list[dict] = []
        mouth_anomalies: list[dict] = []

        # 眼睛和手指异常需要 keypoint 数据
        # 这里提供框架，实际 keypoint 提取通过 FeatureHub
        try:
            keypoints = hub.get("keypoints")
            ear_seq = [kp.get("ear", 0.35) for kp in keypoints]
            eye_anomalies = detect_eye_anomalies(ear_seq, fps=30.0)
            finger_counts = [kp.get("finger_count", 5) for kp in keypoints]
            hand_anomalies = detect_hand_anomalies(finger_counts=finger_counts)
        except (KeyError, TypeError):
            pass  # keypoints 不可用时跳过

        total = len(eye_anomalies) + len(hand_anomalies) + len(mouth_anomalies)
        n_frames = len(face_data)
        anomaly_ratio = total / max(n_frames, 1)
        score = float(np.clip(1.0 - anomaly_ratio, 0, 1))

        return BiologicalAnomalyResult(
            applicable=True,
            eye_anomalies=eye_anomalies,
            hand_anomalies=hand_anomalies,
            mouth_anomalies=mouth_anomalies,
            anomaly_count=total,
            bio_quality_score=score,
        )
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/biological_anomaly/analyzer.py tests/test_bio_analyzer.py
git commit -m "[add] D3: BiologicalAnomalyAnalyzer 主入口"
```

---

## Chunk 6: D4 运动逻辑与平滑度

### Task 6.1: 平滑度增强评分器

**Files:**
- Create: `src/motion_logic/__init__.py`
- Create: `src/motion_logic/config.py`
- Create: `src/motion_logic/smoothness_scorer.py`
- Test: `tests/test_smoothness_scorer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_smoothness_scorer.py
import numpy as np
from src.motion_logic.smoothness_scorer import compute_flow_acceleration_smoothness

def test_smooth_flow():
    # 平滑光流 → 高分
    flows = [(np.ones((50, 50)) * i * 0.1, np.ones((50, 50)) * i * 0.1) for i in range(10)]
    score = compute_flow_acceleration_smoothness(flows)
    assert score > 0.8

def test_jumpy_flow():
    # 跳变光流 → 低分
    flows = []
    for i in range(10):
        mag = 10.0 if i % 2 == 0 else 0.0
        flows.append((np.ones((50, 50)) * mag, np.ones((50, 50)) * mag))
    score = compute_flow_acceleration_smoothness(flows)
    assert score < 0.5
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现**

```python
# src/motion_logic/__init__.py
from .analyzer import MotionLogicAnalyzer
__all__ = ["MotionLogicAnalyzer"]
```

```python
# src/motion_logic/config.py
from dataclasses import dataclass

@dataclass
class MotionLogicConfig:
    dynamics_weight: float = 0.3
    smoothness_weight: float = 0.4
    naturalness_weight: float = 0.3
    enable_mllm: bool = True
    smoothness_acceleration_weight: float = 0.5
    smoothness_qalign_weight: float = 0.5
```

```python
# src/motion_logic/smoothness_scorer.py
from __future__ import annotations
import numpy as np

def compute_flow_acceleration_smoothness(flows: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if len(flows) < 3:
        return 1.0
    magnitudes = []
    for u, v in flows:
        mag = float(np.mean(np.sqrt(u**2 + v**2)))
        magnitudes.append(mag)
    velocities = np.diff(magnitudes)
    accelerations = np.diff(velocities)
    if len(accelerations) == 0:
        return 1.0
    max_acc = np.max(np.abs(accelerations))
    normalized = np.abs(accelerations) / (max_acc + 1e-8)
    smoothness = 1.0 - float(np.mean(normalized))
    return float(np.clip(smoothness, 0, 1))
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/motion_logic/ tests/test_smoothness_scorer.py
git commit -m "[add] D4: 光流加速度平滑度评分器"
```

### Task 6.2: 动态度封装 + MLLM 自然度 + MotionLogicAnalyzer

**Files:**
- Create: `src/motion_logic/dynamics_scorer.py`
- Create: `src/motion_logic/naturalness_judge.py`
- Create: `src/motion_logic/analyzer.py`
- Test: `tests/test_motion_logic_analyzer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_motion_logic_analyzer.py
import numpy as np
from unittest.mock import MagicMock
from src.motion_logic.analyzer import MotionLogicAnalyzer
from src.motion_logic.config import MotionLogicConfig

def test_motion_logic_with_flows():
    hub = MagicMock()
    flows = [(np.ones((50,50))*0.5, np.ones((50,50))*0.5) for _ in range(10)]
    hub.get.side_effect = lambda k: {"optical_flow": flows}.get(k)
    config = MotionLogicConfig(enable_mllm=False)
    analyzer = MotionLogicAnalyzer(config)
    result = analyzer.analyze(hub)
    assert result.applicable is True
    assert 0.0 <= result.motion_logic_score <= 1.0

def test_motion_logic_no_motion():
    hub = MagicMock()
    hub.get.side_effect = lambda k: {"optical_flow": []}.get(k)
    config = MotionLogicConfig(enable_mllm=False)
    analyzer = MotionLogicAnalyzer(config)
    result = analyzer.analyze(hub)
    assert result.applicable is False
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现**

```python
# src/motion_logic/dynamics_scorer.py
from __future__ import annotations
import numpy as np

def compute_dynamics_score(flows: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not flows:
        return 0.0
    magnitudes = [float(np.mean(np.sqrt(u**2 + v**2))) for u, v in flows]
    return float(np.clip(np.mean(magnitudes) / 10.0, 0, 1))
```

```python
# src/motion_logic/naturalness_judge.py
from __future__ import annotations
from typing import Any
import numpy as np

def judge_naturalness_mllm(
    hub: Any,
    mllm_client: Any,
    flows: list[tuple[np.ndarray, np.ndarray]],
    smoothness_score: float,
) -> dict:
    if smoothness_score > 0.8:
        return {"skipped": True, "reason": "smoothness above threshold"}
    try:
        frames = hub.get("video_frames")
    except KeyError:
        return {"skipped": True, "reason": "no video frames"}
    from src.mllm.prompts import MOTION_NATURALNESS_PROMPT
    result = mllm_client.judge_video_clip(frames, MOTION_NATURALNESS_PROMPT)
    return result
```

```python
# src/motion_logic/analyzer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .config import MotionLogicConfig
from .smoothness_scorer import compute_flow_acceleration_smoothness
from .dynamics_scorer import compute_dynamics_score

@dataclass
class MotionLogicResult:
    applicable: bool
    skip_reason: str | None = None
    dynamics_score: float = 0.0
    smoothness_score: float = 0.0
    naturalness_score: float | None = None
    naturalness_issues: list[str] = field(default_factory=list)
    motion_logic_score: float = 0.0

class MotionLogicAnalyzer:
    """D4: 运动逻辑与平滑度分析器。"""

    def __init__(self, config: MotionLogicConfig | None = None, mllm_client: Any = None) -> None:
        self.config = config or MotionLogicConfig()
        self._mllm_client = mllm_client

    def analyze(self, hub: Any) -> MotionLogicResult:
        flows = hub.get("optical_flow")
        if not flows or len(flows) < 2:
            return MotionLogicResult(applicable=False, skip_reason="no motion detected")

        dynamics = compute_dynamics_score(flows)
        smoothness = compute_flow_acceleration_smoothness(flows)

        naturalness = None
        issues: list[str] = []
        if self.config.enable_mllm and self._mllm_client:
            from .naturalness_judge import judge_naturalness_mllm
            result = judge_naturalness_mllm(hub, self._mllm_client, flows, smoothness)
            if not result.get("skipped"):
                naturalness = 1.0 if result.get("is_natural", True) else 0.3
                issues = result.get("issues", [])

        c = self.config
        if naturalness is not None:
            score = c.dynamics_weight * dynamics + c.smoothness_weight * smoothness + c.naturalness_weight * naturalness
        else:
            total_w = c.dynamics_weight + c.smoothness_weight
            score = (c.dynamics_weight * dynamics + c.smoothness_weight * smoothness) / total_w

        return MotionLogicResult(
            applicable=True,
            dynamics_score=dynamics,
            smoothness_score=smoothness,
            naturalness_score=naturalness,
            naturalness_issues=issues,
            motion_logic_score=float(np.clip(score, 0, 1)),
        )
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/motion_logic/ tests/test_motion_logic_analyzer.py
git commit -m "[add] D4: MotionLogicAnalyzer（动态度 + 平滑度 + MLLM自然度）"
```

---

## Chunk 7: D5 物理常识

### Task 7.1: 像素漂移检测 + 重力检查 + PhysicsConsistencyAnalyzer

**Files:**
- Create: `src/physics_consistency/__init__.py`
- Create: `src/physics_consistency/config.py`
- Create: `src/physics_consistency/pixel_drift.py`
- Create: `src/physics_consistency/gravity_check.py`
- Create: `src/physics_consistency/mllm_physics_judge.py`
- Create: `src/physics_consistency/analyzer.py`
- Test: `tests/test_physics_consistency.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_physics_consistency.py
import numpy as np
from src.physics_consistency.pixel_drift import detect_pixel_drift

def test_no_drift_in_static():
    # 静态区域光流全零 → 无漂移
    flows = [(np.zeros((50, 50)), np.zeros((50, 50))) for _ in range(10)]
    mask = np.ones((50, 50), dtype=bool)  # 整帧都是静态区域
    events = detect_pixel_drift(flows, static_mask=mask)
    assert len(events) == 0

def test_detect_drift():
    # 静态区域有持续单向运动 → 漂移
    flows = [(np.ones((50, 50)) * 2.0, np.zeros((50, 50))) for _ in range(10)]
    mask = np.ones((50, 50), dtype=bool)
    events = detect_pixel_drift(flows, static_mask=mask)
    assert len(events) > 0
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现**

```python
# src/physics_consistency/__init__.py
from .analyzer import PhysicsConsistencyAnalyzer
__all__ = ["PhysicsConsistencyAnalyzer"]
```

```python
# src/physics_consistency/config.py
from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    drift_flow_threshold: float = 0.5  # 静态区域光流阈值
    drift_min_frames: int = 5  # 最少持续帧数
    drift_direction_tolerance: float = 30.0  # 方向一致性容忍度（度）
    gravity_fit_threshold: float = 0.3  # 抛物线拟合残差阈值
    enable_mllm: bool = True
    drift_weight: float = 0.4
    gravity_weight: float = 0.3
    mllm_weight: float = 0.3
```

```python
# src/physics_consistency/pixel_drift.py
from __future__ import annotations
import numpy as np

def detect_pixel_drift(
    flows: list[tuple[np.ndarray, np.ndarray]],
    static_mask: np.ndarray | None = None,
    flow_threshold: float = 0.5,
    min_frames: int = 5,
) -> list[dict]:
    if not flows:
        return []
    events = []
    h, w = flows[0][0].shape
    if static_mask is None:
        avg_mag = np.mean([np.sqrt(u**2 + v**2) for u, v in flows], axis=0)
        static_mask = avg_mag < flow_threshold

    # 检测静态区域内的持续单向运动
    directions = []
    magnitudes = []
    for u, v in flows:
        masked_u = u[static_mask]
        masked_v = v[static_mask]
        if len(masked_u) == 0:
            continue
        mean_u = float(np.mean(masked_u))
        mean_v = float(np.mean(masked_v))
        mag = np.sqrt(mean_u**2 + mean_v**2)
        direction = np.degrees(np.arctan2(mean_v, mean_u))
        directions.append(direction)
        magnitudes.append(mag)

    # 连续帧方向一致 + 幅度 > 阈值 → 漂移
    if len(directions) >= min_frames:
        dir_std = float(np.std(directions))
        avg_mag = float(np.mean(magnitudes))
        if dir_std < 30.0 and avg_mag > flow_threshold:
            events.append({
                "type": "pixel_drift",
                "avg_magnitude": avg_mag,
                "direction_std": dir_std,
                "duration_frames": len(directions),
                "description": f"Persistent drift: avg_mag={avg_mag:.2f}, dir_std={dir_std:.1f}°",
            })
    return events
```

```python
# src/physics_consistency/gravity_check.py
from __future__ import annotations
import numpy as np

def check_gravity_consistency(trajectories: list[np.ndarray]) -> list[dict]:
    """检查轨迹是否符合重力加速度方向。

    Args:
        trajectories: 每条轨迹是 (N, 2) 的数组，列为 (x, y)。
    """
    events = []
    for i, traj in enumerate(trajectories):
        if len(traj) < 5:
            continue
        t = np.arange(len(traj))
        y = traj[:, 1]
        # 二次拟合: y = at² + bt + c
        coeffs = np.polyfit(t, y, 2)
        a = coeffs[0]
        residuals = y - np.polyval(coeffs, t)
        fit_error = float(np.mean(residuals**2))
        # 在图像坐标中，重力方向 a > 0（y轴向下）
        if a < -0.1 and fit_error < 100:  # 明显反重力
            events.append({
                "type": "anti_gravity",
                "trajectory_idx": i,
                "acceleration": float(a),
                "fit_error": fit_error,
                "description": f"Trajectory {i}: upward acceleration a={a:.3f}",
            })
    return events
```

```python
# src/physics_consistency/mllm_physics_judge.py
from __future__ import annotations
from typing import Any

def judge_physics_mllm(hub: Any, mllm_client: Any) -> dict:
    try:
        frames = hub.get("video_frames")
    except KeyError:
        return {"skipped": True}
    from src.mllm.prompts import PHYSICS_COMMONSENSE_PROMPT
    return mllm_client.judge_video_clip(frames, PHYSICS_COMMONSENSE_PROMPT)
```

```python
# src/physics_consistency/analyzer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .config import PhysicsConfig
from .pixel_drift import detect_pixel_drift
from .gravity_check import check_gravity_consistency

@dataclass
class PhysicsConsistencyResult:
    applicable: bool
    skip_reason: str | None = None
    drift_events: list[dict] = field(default_factory=list)
    drift_score: float = 1.0
    gravity_violations: list[dict] = field(default_factory=list)
    gravity_score: float = 1.0
    mllm_issues: list[dict] = field(default_factory=list)
    mllm_score: float = 1.0
    physics_score: float = 1.0

class PhysicsConsistencyAnalyzer:
    """D5: 物理常识与动力学一致性分析器。"""

    def __init__(self, config: PhysicsConfig | None = None, mllm_client: Any = None) -> None:
        self.config = config or PhysicsConfig()
        self._mllm_client = mllm_client

    def analyze(self, hub: Any) -> PhysicsConsistencyResult:
        flows = hub.get("optical_flow")
        if not flows or len(flows) < 2:
            return PhysicsConsistencyResult(applicable=False, skip_reason="no motion")

        drift_events = detect_pixel_drift(flows)
        drift_score = 1.0 if not drift_events else max(0.0, 1.0 - len(drift_events) * 0.3)

        gravity_violations: list[dict] = []
        gravity_score = 1.0
        # 重力检测需要轨迹数据，若可用则检测
        try:
            tracking_data = hub.get("tracking")
            if tracking_data:
                gravity_violations = check_gravity_consistency(tracking_data)
                gravity_score = 1.0 if not gravity_violations else 0.3
        except KeyError:
            pass

        mllm_issues: list[dict] = []
        mllm_score = 1.0
        if self.config.enable_mllm and self._mllm_client:
            from .mllm_physics_judge import judge_physics_mllm
            result = judge_physics_mllm(hub, self._mllm_client)
            if not result.get("skipped"):
                mllm_issues = result.get("violations", [])
                mllm_score = 0.3 if result.get("has_violations") else 1.0

        c = self.config
        physics_score = c.drift_weight * drift_score + c.gravity_weight * gravity_score + c.mllm_weight * mllm_score

        return PhysicsConsistencyResult(
            applicable=True,
            drift_events=drift_events,
            drift_score=drift_score,
            gravity_violations=gravity_violations,
            gravity_score=gravity_score,
            mllm_issues=mllm_issues,
            mllm_score=mllm_score,
            physics_score=float(np.clip(physics_score, 0, 1)),
        )
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/physics_consistency/ tests/test_physics_consistency.py
git commit -m "[add] D5: 物理常识分析器（像素漂移 + 重力检测 + MLLM）"
```

---

## Chunk 8: D6 环境一致性

### Task 8.1: 三层检测器 + BackgroundConsistencyAnalyzer

**Files:**
- Create: `src/background_consistency/__init__.py`
- Create: `src/background_consistency/config.py`
- Create: `src/background_consistency/static_region_analysis.py`
- Create: `src/background_consistency/feature_matching.py`
- Create: `src/background_consistency/depth_consistency.py`
- Create: `src/background_consistency/analyzer.py`
- Test: `tests/test_background_consistency.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_background_consistency.py
import numpy as np
from src.background_consistency.static_region_analysis import compute_residual_score
from src.background_consistency.feature_matching import compute_homography_stability
from src.background_consistency.depth_consistency import compute_depth_consistency

def test_residual_score_identical():
    frames = [np.ones((50, 50, 3), dtype=np.uint8) * 128] * 5
    score = compute_residual_score(frames)
    assert score > 0.95

def test_residual_score_different():
    frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(5)]
    score = compute_residual_score(frames)
    assert score < 0.5

def test_depth_consistency_stable():
    depths = [np.ones((50, 50)) * 5.0 + np.random.randn(50, 50) * 0.01 for _ in range(5)]
    score = compute_depth_consistency(depths)
    assert score > 0.9

def test_depth_consistency_flickering():
    depths = [np.ones((50, 50)) * (5.0 if i % 2 == 0 else 0.5) for i in range(10)]
    score = compute_depth_consistency(depths)
    assert score < 0.5
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现**

```python
# src/background_consistency/__init__.py
from .analyzer import BackgroundConsistencyAnalyzer
__all__ = ["BackgroundConsistencyAnalyzer"]
```

```python
# src/background_consistency/config.py
from dataclasses import dataclass

@dataclass
class BackgroundConfig:
    residual_weight: float = 0.3
    homography_weight: float = 0.3
    depth_weight: float = 0.4
```

```python
# src/background_consistency/static_region_analysis.py
from __future__ import annotations
import numpy as np
import cv2

def compute_residual_score(frames: list[np.ndarray], mask: np.ndarray | None = None) -> float:
    if len(frames) < 2:
        return 1.0
    ref = frames[0].astype(np.float32)
    residuals = []
    for f in frames[1:]:
        diff = np.abs(f.astype(np.float32) - ref)
        if mask is not None:
            diff = diff[mask]
        residuals.append(float(np.mean(diff)))
    max_residual = 255.0
    avg_residual = float(np.mean(residuals))
    return float(np.clip(1.0 - avg_residual / max_residual, 0, 1))
```

```python
# src/background_consistency/feature_matching.py
from __future__ import annotations
import numpy as np
import cv2

def compute_homography_stability(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 1.0
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    stabilities = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            stabilities.append(0.5)
            continue
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            stabilities.append(0.5)
            continue
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            stabilities.append(0.5)
            continue
        identity = np.eye(3)
        deviation = float(np.mean(np.abs(H - identity)))
        stabilities.append(float(np.clip(1.0 - deviation, 0, 1)))
    return float(np.mean(stabilities)) if stabilities else 1.0
```

```python
# src/background_consistency/depth_consistency.py
from __future__ import annotations
import numpy as np

def compute_depth_consistency(depth_maps: list[np.ndarray]) -> float:
    if len(depth_maps) < 2:
        return 1.0
    correlations = []
    for i in range(len(depth_maps) - 1):
        d1 = depth_maps[i].flatten()
        d2 = depth_maps[i + 1].flatten()
        if np.std(d1) < 1e-8 or np.std(d2) < 1e-8:
            correlations.append(1.0 if np.allclose(d1, d2) else 0.0)
            continue
        corr = float(np.corrcoef(d1, d2)[0, 1])
        correlations.append(max(corr, 0.0))
    return float(np.mean(correlations))
```

```python
# src/background_consistency/analyzer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .config import BackgroundConfig
from .static_region_analysis import compute_residual_score
from .feature_matching import compute_homography_stability
from .depth_consistency import compute_depth_consistency

@dataclass
class BackgroundConsistencyResult:
    applicable: bool = True
    skip_reason: str | None = None
    residual_score: float = 1.0
    homography_stability: float = 1.0
    depth_consistency: float = 1.0
    background_score: float = 1.0

class BackgroundConsistencyAnalyzer:
    """D6: 环境一致性分析器。"""

    def __init__(self, config: BackgroundConfig | None = None) -> None:
        self.config = config or BackgroundConfig()

    def analyze(self, hub: Any) -> BackgroundConsistencyResult:
        try:
            frames = hub.get("video_frames")
        except KeyError:
            return BackgroundConsistencyResult(applicable=False, skip_reason="no video frames")

        residual = compute_residual_score(frames)
        homography = compute_homography_stability(frames)

        depth_score = 1.0
        try:
            depths = hub.get("depth")
            depth_score = compute_depth_consistency(depths)
        except KeyError:
            pass

        c = self.config
        bg_score = c.residual_weight * residual + c.homography_weight * homography + c.depth_weight * depth_score

        return BackgroundConsistencyResult(
            applicable=True,
            residual_score=residual,
            homography_stability=homography,
            depth_consistency=depth_score,
            background_score=float(np.clip(bg_score, 0, 1)),
        )
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/background_consistency/ tests/test_background_consistency.py
git commit -m "[add] D6: 环境一致性分析器（残差 + 单应性 + 深度一致性）"
```

---

## Chunk 9: 统一流水线集成

### Task 9.1: 统一评测报告数据结构 + 新版 Pipeline

**Files:**
- Create: `src/evaluation_pipeline.py`
- Test: `tests/test_evaluation_pipeline.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evaluation_pipeline.py
from unittest.mock import MagicMock, patch
import numpy as np
from src.evaluation_pipeline import EvaluationPipeline, DimensionResult, EvaluationReport

def test_pipeline_skips_inapplicable():
    pipeline = EvaluationPipeline(enable_mllm=False)
    # Mock hub 无人脸 → D1/D2/D3 跳过
    with patch.object(pipeline, "_create_hub") as mock_hub_fn:
        hub = MagicMock()
        hub.get.side_effect = lambda k: {
            "face_embedding": [{"faces": [], "num_faces": 0}] * 5,
            "optical_flow": [(np.ones((50,50)), np.ones((50,50)))] * 5,
            "video_frames": [np.zeros((50,50,3), dtype=np.uint8)] * 5,
        }.get(k, [])
        mock_hub_fn.return_value = hub
        report = pipeline.evaluate("test.mp4")
        assert isinstance(report, EvaluationReport)
        assert report.dimensions["face_identity"].applicable is False
        assert report.dimensions["background"].applicable is True
        assert len(report.active_dimensions) >= 1

def test_pipeline_weight_redistribution():
    results = {
        "d1": DimensionResult(applicable=False, score=None, weight=0.2),
        "d2": DimensionResult(applicable=True, score=0.8, weight=0.3),
        "d3": DimensionResult(applicable=True, score=0.6, weight=0.5),
    }
    from src.evaluation_pipeline import _redistribute_weights
    active, final = _redistribute_weights(results)
    assert len(active) == 2
    assert abs(sum(r.weight for r in active.values()) - 1.0) < 0.01
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现统一流水线**

```python
# src/evaluation_pipeline.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from src.feature_hub.hub import create_default_hub, FeatureHub
from src.face_identity.analyzer import FaceIdentityAnalyzer
from src.expression_naturalness.analyzer import ExpressionAnalyzer
from src.biological_anomaly.analyzer import BiologicalAnomalyAnalyzer
from src.motion_logic.analyzer import MotionLogicAnalyzer
from src.physics_consistency.analyzer import PhysicsConsistencyAnalyzer
from src.background_consistency.analyzer import BackgroundConsistencyAnalyzer

DEFAULT_WEIGHTS = {
    "face_identity": 0.20,
    "expression": 0.15,
    "biological_anomaly": 0.10,
    "motion_logic": 0.25,
    "physics": 0.15,
    "background": 0.15,
}

@dataclass
class DimensionResult:
    applicable: bool = True
    skip_reason: str | None = None
    score: float | None = None
    weight: float = 0.0
    details: Any = None

@dataclass
class EvaluationReport:
    dimensions: dict[str, DimensionResult] = field(default_factory=dict)
    active_dimensions: list[str] = field(default_factory=list)
    final_score: float = 0.0
    video_type: str = "unknown"

def _redistribute_weights(
    results: dict[str, DimensionResult],
) -> tuple[dict[str, DimensionResult], float]:
    active = {k: v for k, v in results.items() if v.applicable and v.score is not None}
    if not active:
        return {}, 0.0
    total_w = sum(v.weight for v in active.values())
    if total_w > 0:
        for v in active.values():
            v.weight = v.weight / total_w
    final = sum(v.weight * v.score for v in active.values())
    return active, float(np.clip(final, 0, 1))

class EvaluationPipeline:
    """六维度统一评测流水线。"""

    def __init__(
        self,
        device: str = "cuda",
        weights: dict[str, float] | None = None,
        enable_mllm: bool = False,
        mllm_client: Any = None,
    ) -> None:
        self.device = device
        self.weights = weights or DEFAULT_WEIGHTS
        self._mllm_client = mllm_client
        self._analyzers = {
            "face_identity": FaceIdentityAnalyzer(),
            "expression": ExpressionAnalyzer(),
            "biological_anomaly": BiologicalAnomalyAnalyzer(),
            "motion_logic": MotionLogicAnalyzer(mllm_client=mllm_client),
            "physics": PhysicsConsistencyAnalyzer(mllm_client=mllm_client),
            "background": BackgroundConsistencyAnalyzer(),
        }

    def _create_hub(self, video_path: str) -> FeatureHub:
        return create_default_hub(video_path, self.device)

    def evaluate(self, video_path: str) -> EvaluationReport:
        hub = self._create_hub(video_path)
        results: dict[str, DimensionResult] = {}

        score_attr_map = {
            "face_identity": "identity_score",
            "expression": "expression_score",
            "biological_anomaly": "bio_quality_score",
            "motion_logic": "motion_logic_score",
            "physics": "physics_score",
            "background": "background_score",
        }

        for name, analyzer in self._analyzers.items():
            try:
                raw = analyzer.analyze(hub)
                score_attr = score_attr_map[name]
                score = getattr(raw, score_attr, None) if raw.applicable else None
                results[name] = DimensionResult(
                    applicable=raw.applicable,
                    skip_reason=getattr(raw, "skip_reason", None),
                    score=score,
                    weight=self.weights.get(name, 0.0),
                    details=raw,
                )
            except Exception as e:
                results[name] = DimensionResult(
                    applicable=False,
                    skip_reason=f"error: {e}",
                    weight=self.weights.get(name, 0.0),
                )

        active, final_score = _redistribute_weights(results)

        return EvaluationReport(
            dimensions=results,
            active_dimensions=list(active.keys()),
            final_score=final_score,
        )
```

- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: 提交**

```bash
git add src/evaluation_pipeline.py tests/test_evaluation_pipeline.py
git commit -m "[add] 六维度统一评测流水线（自动跳过 + 权重重分配）"
```

### Task 9.2: 更新 requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加新依赖**

追加以下依赖到 `requirements.txt`:

```
insightface>=0.7
py-feat>=0.6
timm>=0.9
openai>=1.0
anthropic>=0.20
```

- [ ] **Step 2: 提交**

```bash
git add requirements.txt
git commit -m "[add] 新增 InsightFace/Py-Feat/MLLM 依赖"
```

---

## 依赖关系总结

```
Chunk 1 (FeatureHub) ← 所有维度依赖
Chunk 2 (MLLM)      ← Chunk 6 (D4) 和 Chunk 7 (D5) 依赖
Chunk 3 (D1)         ← 独立（仅依赖 Chunk 1）
Chunk 4 (D2)         ← 独立（仅依赖 Chunk 1）
Chunk 5 (D3)         ← 独立（仅依赖 Chunk 1）
Chunk 6 (D4)         ← 依赖 Chunk 1 + Chunk 2
Chunk 7 (D5)         ← 依赖 Chunk 1 + Chunk 2
Chunk 8 (D6)         ← 独立（仅依赖 Chunk 1）
Chunk 9 (Pipeline)   ← 依赖所有 Chunk
```

**可并行执行**: Chunk 3/4/5/8 互相独立，可并行开发。
