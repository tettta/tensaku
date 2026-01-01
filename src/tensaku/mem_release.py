# /home/esakit25/work/tensaku/src/tensaku/mem_release.py
# -*- coding: utf-8 -*-
"""tensaku.mem_release

@module : tensaku.mem_release
@role   : 明示的なメモリ解放（実験運用向けの補助）

背景:
- PyTorch / NumPy / glibc のアロケータは、一度確保した大きなメモリを OS に返さず再利用することがあり、
  RSS が「リークのように」見える場合があります（高水位更新）。
- ここで行う解放は「OSへ必ず返す」ことを保証しませんが、
  参照切り + GC + malloc_trim で戻りやすくする補助として使えます。

有効化:
- 環境変数 TENSAKU_MEM_RELEASE=1 のときのみ有効です（デフォルトは無効）。
"""

from __future__ import annotations

import gc
import os

import torch

try:
    import ctypes  # Linux/glibc 前提
except Exception:
    ctypes = None  # type: ignore[assignment]


def maybe_release(tag: str = "") -> None:
    """可能であればメモリ解放を試みる。

    注意:
    - 失敗しても例外は投げません（運用補助のため）。
    """
    if os.getenv("TENSAKU_MEM_RELEASE", "0") != "1":
        return

    # Python ヒープの回収
    try:
        gc.collect()
    except Exception:
        pass

    # CUDA のキャッシュ解放（GPUメモリ向け。CPU RSS には直接効かないことが多い）
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # glibc の場合、OS に返す可能性を上げる
    try:
        if ctypes is not None:
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
    except Exception:
        pass
