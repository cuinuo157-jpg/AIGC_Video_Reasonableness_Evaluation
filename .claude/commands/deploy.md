# 构建与部署

## 构建

```bash
# 安装依赖
pip install -r requirements.txt

# 安装为可编辑包（开发模式）
pip install -e .

# 构建发布包
python -m build
```

## 运行测试

```bash
pytest tests/ -v --tb=short
```

## 代码检查

```bash
ruff check src/ scripts/
ruff format src/ scripts/
```
