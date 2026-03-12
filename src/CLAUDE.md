# src 模块开发规范

## 代码隔离原则

- 每个评测维度独立为一个子包（如 `biological_anomaly/`, `perceptual_quality/`）
- 子包之间不应直接互相导入，公共工具放在 `src/` 根级别
- 每个子包须包含 `__init__.py` 并导出核心接口

## 导入规范

```python
# 正确：使用相对导入（包内部）
from .utils import compute_flow

# 正确：使用绝对导入（跨包）
from src.motion_logic.analyzer import MotionLogicAnalyzer

# 避免：通配符导入
from .utils import *  # 禁止
```

## 函数签名

- 公开函数必须添加类型注解和简要 docstring
- 内部辅助函数以 `_` 开头
