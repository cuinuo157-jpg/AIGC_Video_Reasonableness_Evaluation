# AIGC Video Reasonableness Evaluation

## 项目简介

AIGC 生成式视频合理性多维度评测框架，涵盖时序一致性、运动强度、感知质量等维度。

## 目录结构

```
├─ src/                    # 核心源码（按维度模块划分）
├─ scripts/                # 命令行脚本入口
├─ tests/                  # 单元测试与集成测试
├─ data/                   # 示例/测试视频数据
├─ outputs/                # 分析结果输出
├─ third_party/            # 第三方模型依赖
└─ .claude/                # Claude Code 配置与上下文
```

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v --tb=short

# 代码检查与格式化
ruff check src/ scripts/
ruff format src/ scripts/

# 运行统一流水线
python scripts/unified_pipeline.py --input data/ --output outputs/
```

## 开发规范

- **Python 版本**: 3.10+
- **类型注解**: 所有公开函数必须添加类型注解
- **代码风格**: 遵循 PEP 8，使用 Ruff 进行检查
- **测试**: 新增功能须附带 Pytest 测试用例
- **提交信息**: `[类型] 简要描述`，如 `[add]`, `[fix]`, `[refactor]`
