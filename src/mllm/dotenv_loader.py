"""集中式 .env 加载器，供所有脚本统一使用。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_repo_root(marker: str = ".git") -> Path:
    """从当前文件位置向上查找仓库根目录。"""
    p = Path(__file__).resolve().parent.parent.parent
    for _ in range(10):
        if (p / marker).exists() or (p / marker).is_dir():
            return p
        parent = p.parent
        if parent == p:
            break
        p = parent
    return Path(__file__).resolve().parent.parent.parent


def load_dotenv(dotenv_path: Optional[str | Path] = None, *, override: bool = False) -> None:
    """从 .env 文件加载环境变量（不引入 python-dotenv 依赖）。

    Args:
        dotenv_path: .env 文件路径，默认自动在仓库根目录查找。
        override: 是否覆盖已存在的环境变量（默认 False，仅填充未设置项）。
    """
    if dotenv_path is None:
        dotenv_path = find_repo_root() / ".env"
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.is_file():
        return

    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or (not override and key in os.environ):
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val
