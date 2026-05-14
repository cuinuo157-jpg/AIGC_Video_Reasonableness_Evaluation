#!/usr/bin/env python3
"""
按 name.txt 内容重命名目录

场景：
  - 文件夹 A 的每个子目录下有一个 name.txt，内容为新目录名
  - 文件夹 B 有同名子目录
  - 将 B 中对应子目录按 name.txt 的内容重命名，输出到新文件夹

用法：
  python rename_dirs_by_nametxt.py --src-name A --src-data B --out result
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Set


def read_name_txt(dir_path: Path) -> str | None:
    """读取目录下 name.txt 的第一行有效内容作为新目录名。"""
    txt = dir_path / "name.txt"
    if not txt.is_file():
        return None
    content = txt.read_text(encoding="utf-8").strip()
    # 取第一行有效内容
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def safe_name(raw: str) -> str:
    """清理目录名中的非法字符。"""
    # Windows / Linux 常见非法字符
    forbidden = '<>:"/\\|?*'
    name = raw.strip()
    for ch in forbidden:
        name = name.replace(ch, "_")
    # 去掉首尾空格和点（Windows 不允许以点结尾）
    name = name.strip(". ")
    return name or "_unnamed_"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 name.txt 内容重命名目录并输出到新文件夹",
    )
    parser.add_argument(
        "--src-name", required=True,
        help="包含 name.txt 的源文件夹 A（每个子目录下有 name.txt）",
    )
    parser.add_argument(
        "--src-data", required=True,
        help="待重命名的数据文件夹 B（子目录名与 A 对应）",
    )
    parser.add_argument(
        "--out", required=True,
        help="输出文件夹路径",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览，不实际复制",
    )
    args = parser.parse_args()

    src_name = Path(args.src_name)
    src_data = Path(args.src_data)
    out_dir = Path(args.out)

    if not src_name.is_dir():
        print(f"[error] --src-name 不是有效目录: {src_name}", file=sys.stderr)
        sys.exit(1)
    if not src_data.is_dir():
        print(f"[error] --src-data 不是有效目录: {src_data}", file=sys.stderr)
        sys.exit(1)

    # 收集 A 中的 name.txt 映射: {旧目录名 → 新目录名}
    name_map: dict[str, str] = {}
    seen_names: Set[str] = set()

    for sub in sorted(src_name.iterdir()):
        if not sub.is_dir():
            continue
        new_name = read_name_txt(sub)
        if new_name is None:
            print(f"[skip] {sub.name}: 无 name.txt 或内容为空", file=sys.stderr)
            continue

        safe = safe_name(new_name)
        if safe != new_name:
            print(f"[info] {sub.name} → '{new_name}' 含非法字符，已清理为 '{safe}'", file=sys.stderr)

        # 处理重名
        final_name = safe
        counter = 1
        while final_name in seen_names:
            final_name = f"{safe}_{counter}"
            counter += 1
        if final_name != safe:
            print(f"[warn] 重名冲突: '{safe}' → '{final_name}'", file=sys.stderr)

        seen_names.add(final_name)
        name_map[sub.name] = final_name

    if not name_map:
        print("[error] 未找到任何有效的 name.txt 映射", file=sys.stderr)
        sys.exit(1)

    print(f"\n找到 {len(name_map)} 个映射:")
    for old, new in name_map.items():
        print(f"  {old}  →  {new}")

    if args.dry_run:
        print("\n[dry-run] 未执行实际复制。去掉 --dry-run 以执行。")
        return

    # 复制 B 中对应子目录到 out，使用新名称
    if out_dir.exists():
        print(f"[warn] 输出目录已存在，将覆盖同名内容: {out_dir}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for old_name, new_name in name_map.items():
        src_sub = src_data / old_name
        dst_sub = out_dir / new_name

        if not src_sub.is_dir():
            print(f"[skip] B 中无对应目录: {old_name}", file=sys.stderr)
            skipped += 1
            continue

        if dst_sub.exists():
            print(f"[warn] 目标已存在，将被覆盖: {dst_sub}", file=sys.stderr)

        shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)
        copied += 1
        print(f"[ok] {old_name}  →  {new_name}")

    print(f"\n完成: 复制 {copied} 个目录, 跳过 {skipped} 个 → {out_dir}")


if __name__ == "__main__":
    main()
