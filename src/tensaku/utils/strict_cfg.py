# /home/esakit25/work/tensaku/src/tensaku/utils/strict_cfg.py
# -*- coding: utf-8 -*-
"""tensaku.utils.strict_cfg

Strict config access helpers.

Policy
- No fallback / no silent defaults: missing required keys raise ConfigError.
- Accept path as either:
    - single key (str/int), e.g. require_str(cfg, "qid")
    - sequence of keys, e.g. require_str(cfg, ("data","qid"))

Notes
- This module must stay lightweight (no heavy ML imports).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Type, TypeVar, Union, List, Optional

T = TypeVar("T")

class ConfigError(ValueError):
    """Raised when config is missing or invalid under strict policy."""

Key = Union[str, int]
PathLike = Union[Key, Sequence[Key]]

def _normalize_path(path: PathLike) -> Sequence[Key]:
    if isinstance(path, (str, int)):
        return (path,)
    return path

def _path_to_str(path: Sequence[Key]) -> str:
    out: List[str] = []
    for p in path:
        if isinstance(p, int):
            out.append(f"[{p}]")
        else:
            out.append(str(p))
    return ".".join(out)

def _get(cfg: Mapping[str, Any], path: Sequence[Key], *, ctx: str) -> Any:
    cur: Any = cfg
    for p in path:
        if isinstance(cur, Mapping):
            if p not in cur:
                raise ConfigError(f"[{ctx}] missing key: {_path_to_str(path)}")
            cur = cur[p]
        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
            if not isinstance(p, int):
                raise ConfigError(f"[{ctx}] expected int index at {_path_to_str(path)}")
            if p < 0 or p >= len(cur):
                raise ConfigError(f"[{ctx}] index out of range: {_path_to_str(path)}")
            cur = cur[p]
        else:
            raise ConfigError(f"[{ctx}] cannot traverse into {_path_to_str(path)} (type={type(cur)})")
    return cur

def require(cfg: Mapping[str, Any], path: PathLike, typ: Type[T], *, ctx: str = "cfg") -> T:
    p = _normalize_path(path)
    v = _get(cfg, p, ctx=ctx)
    if not isinstance(v, typ):
        raise ConfigError(f"[{ctx}] {_path_to_str(p)} must be {typ}, got {type(v)}")
    return v

def require_mapping(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> Mapping[str, Any]:
    return require(cfg, path, Mapping, ctx=ctx)

def require_list(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> List[Any]:
    v = _get(cfg, _normalize_path(path), ctx=ctx)
    # 修正: OmegaConfのListConfigなども許容するため、list型の厳密チェックではなくSequenceかつ非文字列であることを確認する
    if isinstance(v, (str, bytes)) or not isinstance(v, Sequence):
        raise ConfigError(f"[{ctx}] {_path_to_str(_normalize_path(path))} must be list-like (Sequence), got {type(v)}")
    return list(v)

def require_str(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> str:
    v = _get(cfg, _normalize_path(path), ctx=ctx)
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"[{ctx}] {_path_to_str(_normalize_path(path))} must be non-empty str")
    return v.strip()

def require_int(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> int:
    v = _get(cfg, _normalize_path(path), ctx=ctx)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"[{ctx}] {_path_to_str(_normalize_path(path))} must be int, got {type(v)}")
    return int(v)

def require_float(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> float:
    v = _get(cfg, _normalize_path(path), ctx=ctx)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"[{ctx}] {_path_to_str(_normalize_path(path))} must be float, got {type(v)}")
    return float(v)

def require_bool(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> bool:
    v = _get(cfg, _normalize_path(path), ctx=ctx)
    if not isinstance(v, bool):
        raise ConfigError(f"[{ctx}] {_path_to_str(_normalize_path(path))} must be bool, got {type(v)}")
    return bool(v)

def require_one_of(cfg: Mapping[str, Any], path: PathLike, allowed: Iterable[str], *, ctx: str = "cfg") -> str:
    v = require_str(cfg, path, ctx=ctx)
    allowed_set = set(allowed)
    if v not in allowed_set:
        raise ConfigError(f"[{ctx}] {_path_to_str(_normalize_path(path))} must be one of {sorted(allowed_set)}, got {v!r}")
    return v

def require_optional_str(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> Optional[str]:
    p = _normalize_path(path)
    try:
        v = _get(cfg, p, ctx=ctx)
    except ConfigError:
        return None
    if v is None:
        return None
    if not isinstance(v, str):
        raise ConfigError(f"[{ctx}] {_path_to_str(p)} must be str or null, got {type(v)}")
    s = v.strip()
    return s if s else None

def require_list_str(cfg: Mapping[str, Any], path: PathLike, *, ctx: str = "cfg") -> List[str]:
    xs = require_list(cfg, path, ctx=ctx)
    out: List[str] = []
    for i, x in enumerate(xs):
        if not isinstance(x, str) or not x.strip():
            raise ConfigError(f"[{ctx}] {_path_to_str(_normalize_path(path))}[{i}] must be non-empty str")
        out.append(x.strip())
    return out