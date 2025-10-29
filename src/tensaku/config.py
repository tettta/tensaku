from __future__ import annotations
from typing import Any, Dict, List
import os, yaml

def _deep_set(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    x = d
    for k in keys[:-1]:
        if k not in x or not isinstance(x[k], dict):
            x[k] = {}
        x = x[k]
    x[keys[-1]] = value

def load_config(path: str, overrides: List[str] | None = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for s in (overrides or []):
        k, v = s.split("=", 1)
        _deep_set(cfg, k.split("."), yaml.safe_load(v))
    base = os.path.dirname(os.path.abspath(path))
    def abs_path(p: str | None) -> str | None:
        return p if (p is None or os.path.isabs(p)) else os.path.normpath(os.path.join(base, p))
    # 主要パスを絶対化
    if "run" in cfg:
        cfg["run"]["out_dir"]  = abs_path(cfg["run"].get("out_dir"))
        cfg["run"]["data_dir"] = abs_path(cfg["run"].get("data_dir"))
    if "infer" in cfg and cfg["infer"].get("ckpt"):
        cfg["infer"]["ckpt"] = abs_path(cfg["infer"]["ckpt"])
    return cfg
