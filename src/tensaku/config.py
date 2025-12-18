# /home/esakit25/work/tensaku/src/tensaku/config.py
"""
@module     : tensaku.config
@role       : YAML ベースの設定読み込みとパス解決ユーティリティ
@overview   :
    - load_config(path, overrides): CLI から使う設定読み込みエントリポイント
    - make_paths_absolute(cfg, root_dir): run/out_dir など主要パスを絶対パスに正規化
    - _deep_set / _apply_overrides: "a.b.c=1" 形式の override を dict に適用
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional
import os
from pathlib import Path

import yaml


def _get_root_dir() -> str:
    """
    Tensaku プロジェクトのルートディレクトリを推定する。

    優先順位:
        1. 環境変数 TENSAKU_ROOT
        2. このファイルの 2 つ上のディレクトリ（.../tensaku → .../）
    """
    env = os.environ.get("TENSAKU_ROOT")
    if env:
        return os.path.abspath(env)
    # /home/.../tensaku/src/tensaku/config.py → parents[2] = /home/.../tensaku
    return str(Path(__file__).resolve().parents[2])


def _parse_scalar(v: str) -> Any:
    """
    "1", "true", "3.14" のような文字列を、いい感じに Python の値に変換する。

    - "true"/"false"/"yes"/"no" → bool
    - int, float を順にトライ
    - それ以外はそのまま文字列
    """
    s = v.strip()
    lower = s.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False

    # int
    try:
        return int(s)
    except ValueError:
        pass

    # float
    try:
        return float(s)
    except ValueError:
        pass

    return s


def _deep_set(d: MutableMapping[str, Any], keys: List[str], value: Any) -> None:
    """
    ネストした dict に対して "a.b.c" 形式のキーを書き込むヘルパ。

    既存の YAML の型と合わない場合は dict を新規に作成する。
    """
    x: MutableMapping[str, Any] = d
    for k in keys[:-1]:
        if k not in x or not isinstance(x[k], dict):
            # 型が dict でない場合は上書きして dict を作成（安全側）
            # print(f"[config] WARN: path override target '{'.'.join(keys)}' created dict for '{k}'")
            x[k] = {}
        x = x[k]  # type: ignore[assignment]
    x[keys[-1]] = value


def _apply_overrides(cfg: MutableMapping[str, Any], overrides: Optional[List[str]]) -> MutableMapping[str, Any]:
    """
    "section.key=value" 形式の override を cfg に適用する。

    例:
        overrides = ["run.seed=42", "data.qid=Y14_1-2_1_3"]
    """
    if not overrides:
        return cfg

    for item in overrides:
        if not item:
            continue
        if "=" not in item:
            # "foo.bar" のような形式は無視（将来の拡張余地として残す）
            continue
        key_str, raw_value = item.split("=", 1)
        keys = [k for k in key_str.split(".") if k]
        if not keys:
            continue
        value = _parse_scalar(raw_value)
        _deep_set(cfg, keys, value)

    return cfg


def make_paths_absolute(
    cfg: MutableMapping[str, Any],
    *,
    root_dir: Optional[str | os.PathLike[str]] = None,
) -> MutableMapping[str, Any]:
    """
    設定 dict の中に含まれる主要パス（run/out_dir, run/data_dir, infer/ckpt, data/input_all）を
    プロジェクトルート基準の絶対パスに変換する。

    - すでに絶対パスであればそのまま正規化のみ行う。
    - None / 空文字はそのまま返す。
    """
    if root_dir is None:
        root_dir = _get_root_dir()
    root_dir = os.fspath(root_dir)
    root_dir = os.path.abspath(root_dir)

    def abs_path(p: Any) -> Any:
        if p is None:
            return None
        if not isinstance(p, str) or p == "":
            return p
        if os.path.isabs(p):
            # 絶対パスならそのまま正規化
            return os.path.normpath(p)
        # 相対パスの場合、ROOT_DIR を基準とする（例: outputs/q-QID）
        return os.path.normpath(os.path.join(root_dir, p))

    # 主要パスを絶対化
    run_cfg = cfg.get("run")
    if isinstance(run_cfg, dict):
        if "out_dir" in run_cfg:
            run_cfg["out_dir"] = abs_path(run_cfg.get("out_dir"))
        if "data_dir" in run_cfg:
            run_cfg["data_dir"] = abs_path(run_cfg.get("data_dir"))

    infer_cfg = cfg.get("infer")
    if isinstance(infer_cfg, dict) and infer_cfg.get("ckpt"):
        infer_cfg["ckpt"] = abs_path(infer_cfg.get("ckpt"))

    data_cfg = cfg.get("data")
    if isinstance(data_cfg, dict) and data_cfg.get("input_all"):
        data_cfg["input_all"] = abs_path(data_cfg.get("input_all"))

    return cfg


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    YAML ファイルを読み込み、オプションの override を適用し、
    さらに主要パスを絶対パスにした dict を返す。

    Args:
        path: 設定 YAML へのパス。相対パスの場合は TENSAKU_ROOT または
              リポジトリルートからの相対とみなす。
        overrides: "a.b.c=value" 形式のリスト（任意）。

    Returns:
        設定 dict（run/out_dir, run/data_dir などは絶対パス）。
    """
    root_dir = _get_root_dir()
    cfg_path = path
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(root_dir, cfg_path)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # override を適用
    _apply_overrides(cfg, overrides)

    # パスを絶対パスに正規化
    make_paths_absolute(cfg, root_dir=root_dir)
    return cfg
