# /home/esakit25/work/tensaku/src/tensaku/config.py
from __future__ import annotations
from typing import Any, Dict, List
import os, yaml

def _deep_set(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    x = d
    for k in keys[:-1]:
        # 【修正】YAMLの型と合わない場合は警告を出してデフォルトの dict を設定
        if k not in x or not isinstance(x[k], dict):
            # print(f"[config] WARN: path override target '{'.'.join(keys)}' created dict for '{k}'")
            x[k] = {}
        x = x[k]
    x[keys[-1]] = value

def load_config(path: str, overrides: List[str] | None = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 1. overrides (CLI --set) を適用
    for s in (overrides or []):
        k, v = s.split("=", 1)
        _deep_set(cfg, k.split("."), yaml.safe_load(v))
    
    # 2. パスの絶対化（プロジェクトルート /home/esakit25/work/tensaku を基準とする）
    #    ※YAMLファイルパス (path) を基準にすると、実行場所によってズレるため、
    #      今回は /home/esakit25/work/tensaku をベースとする。
    #      通常は os.path.dirname(os.path.abspath(path)) を使うべきだが、
    #      ここでは tensaku CLI の制約により /home/esakit25/work/tensaku を直接使う。
    
    # 既存の絶対パス解決ロジックをコメントアウト
    # base = os.path.dirname(os.path.abspath(path))
    # def abs_path(p: str | None) -> str | None:
    #     return p if (p is None or os.path.isabs(p)) else os.path.normpath(os.path.join(base, p))

    # 【修正】プロジェクトルートをベースとする
    ROOT_DIR = "/home/esakit25/work/tensaku" # ユーザーの環境変数 $ROOT に相当
    def abs_path(p: str | None) -> str | None:
        if p is None: return None
        # 絶対パスか、相対パスか（. / .. で始まっているか）
        if os.path.isabs(p) or p.startswith("./") or p.startswith("../"):
            # 絶対パスまたは明示的な相対パスはそのまま
            return os.path.normpath(p)
        else:
            # 相対パスの場合、ROOT_DIR を基準とする（例: outputs/q-QID）
            return os.path.normpath(os.path.join(ROOT_DIR, p))

    # 主要パスを絶対化
    if "run" in cfg:
        cfg["run"]["out_dir"]  = abs_path(cfg["run"].get("out_dir"))
        cfg["run"]["data_dir"] = abs_path(cfg["run"].get("data_dir"))
    if "infer" in cfg and cfg["infer"].get("ckpt"):
        cfg["infer"]["ckpt"] = abs_path(cfg["infer"].get("ckpt"))
    if "data" in cfg and cfg["data"].get("input_all"):
        cfg["data"]["input_all"] = abs_path(cfg["data"].get("input_all"))
    
    return cfg