"""基础冒烟测试 - 验证项目结构与核心模块可导入。"""

import importlib
import pathlib


def test_project_structure():
    """验证关键目录存在。"""
    root = pathlib.Path(__file__).parent.parent
    for name in ["src", "scripts", "data", "outputs", "tests"]:
        assert (root / name).is_dir(), f"缺少目录: {name}"


def test_src_init_importable():
    """验证 src 包可导入。"""
    spec = importlib.util.find_spec("src")
    # src 目录存在 __init__.py 即可
    assert (pathlib.Path(__file__).parent.parent / "src" / "__init__.py").exists()


def test_scripts_unified_pipeline_exists():
    """验证评测流水线模块存在。"""
    root = pathlib.Path(__file__).parent.parent
    assert (root / "src" / "evaluation_pipeline.py").is_file()
