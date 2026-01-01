# /home/esakit25/work/tensaku/src/tensaku/utils/memlog.py
# -*- coding: utf-8 -*-
"""tensaku.utils.memlog

メモリ使用量（RSS/HWMなど）を「軽量に」ログへ出すユーティリティ。

狙い
- 長時間実験で RSS が増え続ける場合に「どの工程で増えたか」を切り分ける。
- ログ肥大化を防ぐため、出力は小さく保つ（本文・巨大配列・DataFrame等を出さない）。

運用
- logger 名は "tensaku.mem"（Hydra job_logging でこのロガーだけ DEBUG + mem.log へ出す）
- root は INFO のまま維持する（依存ライブラリのログ爆発を防ぐ）
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


LOGGER = logging.getLogger("tensaku.mem")

# 1行が巨大になって端末・エディタが落ちるのを防ぐための上限
_MAX_STR = 200
_MAX_LIST_ITEMS = 20
_MAX_DICT_ITEMS = 30
_MAX_JSON_CHARS = 2000


@dataclass(frozen=True)
class MemSnapshot:
    ts: str
    pid: int
    rss_kb: Optional[int]
    hwm_kb: Optional[int]
    vms_kb: Optional[int]
    threads: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "pid": self.pid,
            "rss_kb": self.rss_kb,
            "hwm_kb": self.hwm_kb,
            "vms_kb": self.vms_kb,
            "threads": self.threads,
        }


def _read_proc_status(pid: int) -> Dict[str, str]:
    """Linux の /proc/<pid>/status を読む。失敗したら空 dict。"""
    path = f"/proc/{pid}/status"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return {}

    out: Dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_kb_field(v: Optional[str]) -> Optional[int]:
    if not v:
        return None
    parts = v.split()
    try:
        return int(parts[0])
    except Exception:
        return None


def _short_str(x: Any) -> str:
    s = str(x)
    if len(s) <= _MAX_STR:
        return s
    return s[: _MAX_STR] + "...(truncated)"


def _sanitize(x: Any, depth: int = 0) -> Any:
    """JSON化するための安全な縮約。巨大オブジェクトをログに出さない。"""
    if x is None or isinstance(x, (int, float, bool)):
        return x
    if isinstance(x, str):
        return _short_str(x)
    if isinstance(x, (bytes, bytearray)):
        return f"<bytes:{len(x)}>"
    if isinstance(x, dict):
        items = list(x.items())[:_MAX_DICT_ITEMS]
        out: Dict[str, Any] = {}
        for k, v in items:
            out[_short_str(k)] = _sanitize(v, depth + 1)
        if len(x) > _MAX_DICT_ITEMS:
            out["..."] = f"<dict_truncated:{len(x)}>"
        return out
    if isinstance(x, (list, tuple)):
        items = list(x)[:_MAX_LIST_ITEMS]
        out_list = [_sanitize(v, depth + 1) for v in items]
        if len(x) > _MAX_LIST_ITEMS:
            out_list.append(f"<list_truncated:{len(x)}>")
        return out_list
    # torch.Tensor / np.ndarray / pandas.DataFrame などは repr が巨大になりがち
    # ここではクラス名だけ出す
    return f"<{x.__class__.__name__}>"


def snapshot(*, event: str, extra: Optional[Dict[str, Any]] = None, pid: Optional[int] = None) -> MemSnapshot:
    """メモリスナップショットを作り、tensaku.mem へ DEBUG ログを出す。

    注意
    - DEBUG が無効なら /proc を読まない（オーバーヘッド最小）。
    - extra は縮約して出す（本文/巨大配列/DataFrameを出さない）。
    """
    if not LOGGER.isEnabledFor(logging.DEBUG):
        # 返り値互換のため、最低限だけ埋める（観測しない）
        pid_i = int(pid) if pid is not None else os.getpid()
        return MemSnapshot(
            ts=time.strftime("%F %T"),
            pid=pid_i,
            rss_kb=None,
            hwm_kb=None,
            vms_kb=None,
            threads=None,
        )

    pid_i = int(pid) if pid is not None else os.getpid()
    st = _read_proc_status(pid_i)

    snap = MemSnapshot(
        ts=time.strftime("%F %T"),
        pid=pid_i,
        rss_kb=_parse_kb_field(st.get("VmRSS")),
        hwm_kb=_parse_kb_field(st.get("VmHWM")),
        vms_kb=_parse_kb_field(st.get("VmSize")),
        threads=(int(st.get("Threads")) if st.get("Threads") and st.get("Threads").isdigit() else None),
    )

    payload: Dict[str, Any] = {"event": event, **snap.to_dict()}
    if extra:
        payload["extra"] = _sanitize(extra)

    # JSON 1行で出す（grepしやすい）。ただし肥大化したら切る。
    try:
        s = json.dumps(payload, ensure_ascii=False)
        if len(s) > _MAX_JSON_CHARS:
            s = s[:_MAX_JSON_CHARS] + "...(truncated)"
        LOGGER.debug(s)
    except Exception:
        LOGGER.debug("memlog event=%s pid=%s rss_kb=%s hwm_kb=%s", event, pid_i, snap.rss_kb, snap.hwm_kb)

    return snap
