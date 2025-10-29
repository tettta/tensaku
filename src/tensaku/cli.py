"""
@module: cli
@role: tensaku コマンドのディスパッチャ（split/train/infer-pool/confidence/gate/...）
@inputs: YAML config（-c/--config, --set KEY=VAL）, サブコマンド引数
@outputs: 各サブコマンドの成果物（このモジュール自体は副作用なし）
@cli: tensaku {split,train,infer-pool,confidence,gate,al-sample,al-label-import,al-cycle,viz,eval}
@notes: 未実装コマンドは警告のみで終了。confidence.py 不在でも ImportError を回避。
"""

# Path: /home/esakit25/work/tensaku/src/tensaku/cli.py
from __future__ import annotations
import argparse
import sys
from typing import Dict, Any, List, Callable

from .config import load_config
from . import split as split_mod
from . import train as train_mod
from . import infer_pool as infer_pool_mod
from . import gate as gate_mod

# confidence は任意モジュール（無ければ warn のみで動作継続）
try:
    from . import confidence as confidence_mod
except Exception:
    confidence_mod = None

def _not_implemented(_cfg: Dict[str, Any]) -> None:
    print("[warn] this subcommand is not implemented yet.")

def _maybe(mod, fn_name: str, fallback: Callable[[Dict[str, Any]], None]) -> Callable[[Dict[str, Any]], None]:
    if mod is None:
        return fallback
    fn = getattr(mod, "run", None)
    return fn if callable(fn) else fallback

# ---- コマンド定義（confidence は存在チェック付き） ----
COMMANDS = {
    "split":        split_mod.run,
    "train":        train_mod.run,
    "infer-pool":   infer_pool_mod.run,
    "confidence":   _maybe(confidence_mod, "run", _not_implemented),
    "gate":         gate_mod.run,
    "al-sample":    _not_implemented,
    "al-label-import": _not_implemented,
    "al-cycle":     _not_implemented,
    "viz":          _not_implemented,
    "eval":         _not_implemented,
}

def _parse_overrides(sets: List[str]) -> Dict[str, Any]:
    """
    --set KEY=VAL ... を辞書化（VALは文字列のまま。型解決は load_config に委譲）
    """
    kv = {}
    for s in sets or []:
        if "=" not in s:
            raise SystemExit(f"[error] invalid --set entry (expect KEY=VAL): {s}")
        k, v = s.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tensaku")
    p.add_argument(
        "-c", "--config",
        default="./configs/exp_al_hitl.yaml",
        help="Path to YAML config (default: ./configs/exp_al_hitl.yaml)",
    )
    p.add_argument(
        "--set",
        metavar="KEY=VAL",
        nargs="*",
        default=[],
        help="Override config entries using dot notation (e.g., data.input_all=/path/to/all.jsonl)",
    )
    p.add_argument(
        "cmd",
        choices=list(COMMANDS.keys()),
        help="Subcommand to run",
    )
    return p

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # 設定ロード（dot-override は load_config 側で解釈）
    _ = _parse_overrides(args.set)  # 形式チェックのみ
    cfg = load_config(args.config, args.set)

    # サブコマンド実行
    fn = COMMANDS.get(args.cmd)
    if fn is None:
        print(f"[error] unknown command: {args.cmd}")
        return 2
    fn(cfg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
