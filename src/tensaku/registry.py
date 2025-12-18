# /home/esakit25/work/tensaku/src/tensaku/registry.py
"""
@module     tensaku.registry
@role       Tensaku 内の軽量レジストリ（関数/クラス/ファクトリの「名前→オブジェクト」解決）
@inputs
  - register(name: str, *, override: bool=False)(obj): デコレータで登録
  - get(name: str): 登録オブジェクトを取得（無ければ lazy import → active bootstrap）
  - create(name: str, **kw): 取得物が callable なら **kw で生成、非 callable はそのまま返す
  - list_names(): 現在登録されている名前一覧（ソート済み）
@outputs
  - 返り値として登録オブジェクト、もしくは生成済みインスタンスを返す（副作用は内部レジストリ更新のみ）
@api
  - from tensaku.registry import register, get, create, list_names
  - lazy bootstrap: 未登録 name 要求時に tensaku.confidence / tensaku.trustscore を import
  - bootstrap: import modules for side-effect registrations
@contracts
  - register(name) は同名がある場合は KeyError（override=True で上書き許可）
  - lazy/active bootstrap は import 失敗時も安全に無視（get は最終的に未登録なら KeyError）
  - スレッド非対応（現状想定: 単一プロセス/逐次実行）。必要ならロック導入で拡張可能。
@design
  - 既定の信頼度ファクトリ名: ("msp","entropy","energy","margin","mc_dropout")
  - 「プロセスを跨ぐと list_names() が空」問題に対して、lazy + active の二段構えで解消
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

_REGISTRY: Dict[str, Any] = {}

# ===== Core API ======================================================================

def register(name: str, *, override: bool = False):
    """デコレータ: 関数/クラス/ファクトリを name で登録。"""
    def _decor(obj: Any):
        if (not override) and (name in _REGISTRY):
            raise KeyError(f"'{name}' is already registered")
        _REGISTRY[name] = obj
        return obj
    return _decor

def list_names() -> List[str]:
    return sorted(_REGISTRY.keys())

# ===== Lazy bootstrap ================================================================

def _try_import(module: str) -> Optional[Any]:
    try:
        import importlib
        return importlib.import_module(module)
    except Exception:
        return None

def _lazy_bootstrap_for(name: Optional[str] = None) -> None:
    """Import modules that register estimators via decorators (best-effort).

    - This keeps registry as a simple name→object map.
    - confidence/trustscore are responsible for calling @register(...) at import time.
    """
    # Import confidence estimators (msp/entropy/margin/energy/trust, etc.)
    _try_import("tensaku.confidence")

    # Import trustscore implementation if requested (or if 'trust' is missing)
    if name == "trust" or name is None:
        _try_import("tensaku.trustscore")

# ===== Public getters ================================================================= =================================================================

def get(name: str) -> Any:
    """登録オブジェクトを取得。無ければ lazy import/登録を行い、見つからなければ KeyError。"""
    if name in _REGISTRY:
        return _REGISTRY[name]
    _lazy_bootstrap_for(name)
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise KeyError(f"'{name}' is not registered")

def create(name: str, **kwargs: Any) -> Any:
    """name で取得したオブジェクトが callable なら **kwargs を渡して生成、そうでなければオブジェクト自体を返す。"""
    obj = get(name)
    return obj(**kwargs) if callable(obj) else obj

# ===== Self test ======================================================================

if __name__ == "__main__":
    import os, sys, traceback
    BASE = os.environ.get("TensakuBase", "/home/esakit25/work/tensaku")
    if (BASE + "/src") not in sys.path:
        sys.path.insert(0, BASE + "/src")

    print("[registry] before:", list_names())
    try:
        # 1) confidence 側の既定推定器を取得（lazy で能動登録が働く）
        f_msp = create("msp")
        print("[ok] create('msp') ->", type(f_msp))
        # 2) trust も同様
        try:
            f_trust = get("trust")  # class か factory のはず
        except KeyError:
            # まだ無ければ create で起動（active bootstrap）
            f_trust = create("trust")
        print("[ok] get/create('trust') ->", f_trust)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

    print("[registry] after:", list_names()[:8], "... total", len(list_names()))
    print("[OK] registry self-test passed.")
