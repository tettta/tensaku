# /home/esakit25/work/tensaku/src/tensaku/config_dump.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.config_dump
@role     : 有効な設定 (cfg) を YAML で保存/表示するユーティリティ
@cli      : tensaku config-dump [-c CFG.yaml] [--set KEY=VAL ...] [--out PATH] [--stdout]
@inputs   :
  - cfg: tensaku.cli.load_config() で構成された最終的な設定 dict
@outputs  :
  - YAML ファイル:
      既定: {run.out_dir}/config_effective-{run.run_id or data.qid or "default"}.yaml
    ※ --out で明示パス指定も可能
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover - PyYAML 未インストール時の保護
    yaml = None


def _default_out_path(cfg: Dict[str, Any]) -> str:
    """run/data セクションから無難なデフォルト出力パスを決める。"""
    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}

    out_dir = (
        run_cfg.get("out_dir")
        or cfg.get("out_dir")
        or run_cfg.get("data_dir")
        or data_cfg.get("data_dir")
        or "."
    )
    out_dir = str(out_dir)

    run_id = run_cfg.get("run_id") or data_cfg.get("qid") or "default"
    if run_id:
        fname = f"config_effective-{run_id}.yaml"
    else:
        fname = "config_effective.yaml"

    return os.path.join(out_dir, fname)


def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    """
    有効な設定 cfg を YAML にして保存する。

    - tensaku.cli 側で load_config() 済みの cfg がそのまま渡される前提。
    - したがって -c / --set などでの上書き後の値がすべて含まれる。
    """
    parser = argparse.ArgumentParser(
        prog="tensaku config-dump",
        description="有効な設定(cfg)を YAML で保存/表示する。",
        add_help=True,
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="書き出し先 YAML パス。未指定なら run.out_dir 配下に自動決定。",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        default=False,
        help="YAML を標準出力にも表示する。",
    )
    ns, _rest = parser.parse_known_args(argv or [])

    if yaml is None:
        print(
            "[config-dump] ERROR: PyYAML がインストールされていないため dump できません。",
            file=sys.stderr,
        )
        return 1

    out_path = ns.out or _default_out_path(cfg)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # sort_keys=False でセクション順をなるべく保つ
    text = yaml.safe_dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[config-dump] ERROR: failed to write {out_path}: {e}", file=sys.stderr)
        return 1

    print(f"[config-dump] saved effective config to: {out_path}")

    if ns.stdout:
        print("\n# ==== Effective config (yaml) ====\n")
        print(text)

    return 0
