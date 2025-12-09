# /home/esakit25/work/tensaku/tree_view.py
# -*- coding: utf-8 -*-
"""
@module: tree_view
@role  : ディレクトリの中身を再帰的に表示するユーティリティ (treeコマンド風)
@usage : python tree_view.py [対象ディレクトリ]
"""

import argparse
import os
from pathlib import Path

# 表示から除外するディレクトリ名
IGNORE_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", ".idea", ".vscode"}

def _get_size_str(path: Path) -> str:
    """ファイルサイズを人間が読みやすい形式に変換する"""
    try:
        size = path.stat().st_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    except Exception:
        return "?"

def print_tree(
    directory: Path,
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = True
):
    """ディレクトリツリーを再帰的に表示する"""
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return

    # ルートディレクトリの表示
    if is_root:
        print(f"📂 {directory.resolve().name}/")
    else:
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}📂 {directory.name}/")

    # 次のプレフィックスを決定
    if is_root:
        new_prefix = ""
    else:
        new_prefix = prefix + ("    " if is_last else "│   ")

    try:
        # ディレクトリ内のアイテムを取得してソート (ディレクトリ優先)
        items = list(directory.iterdir())
        items = sorted(items, key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        print(f"{new_prefix}└── 🚫 [Permission Denied]")
        return

    # フィルタリング
    items = [i for i in items if i.name not in IGNORE_DIRS]
    count = len(items)

    for i, item in enumerate(items):
        is_last_item = (i == count - 1)
        
        if item.is_dir():
            print_tree(item, new_prefix, is_last_item, is_root=False)
        else:
            connector = "└── " if is_last_item else "├── "
            size_str = _get_size_str(item)
            # ファイル名とサイズを表示
            print(f"{new_prefix}{connector}📄 {item.name} ({size_str})")

def main():
    parser = argparse.ArgumentParser(description="Show directory tree with file sizes.")
    parser.add_argument("dir", nargs="?", default=".", help="Target directory (default: current)")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    print(f"Target: {target_dir.resolve()}\n")
    print_tree(target_dir)

if __name__ == "__main__":
    main()