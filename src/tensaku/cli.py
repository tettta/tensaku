# -*- coding: utf-8 -*-
"""
@module     tensaku.cli
@role       tensaku コマンドのディスパッチャ（split/train/infer-pool/confidence/gate/...）
@inputs     YAML config（-c/--config, --set KEY=VAL）, サブコマンド引数（未知分はサブに委譲）
@outputs    各サブコマンドの成果物（このモジュール自体は副作用なし）
@cli        tensaku {split,train,infer-pool,confidence,gate,al-sample,al-label-import,al-cycle,viz,eval} [subcmd-opts]
@notes      parse_known_argsで未知引数をサブコマンドに渡す。run(argv, cfg) や旧 run(cfg)、戻り値Noneも許容（0へ正規化）。
"""
from __future__ import annotations
import argparse
import inspect
import sys
from typing import Dict, Any, List, Callable, Optional

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

# ---- 共通：戻り値の正規化 ----
def _normalize_rc(value) -> int:
    """run() の戻りをプロセス終了コードに正規化。None/True→0, False→1, int→そのまま, それ以外→0。"""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0 if value else 1
    if isinstance(value, int):
        return value
    try:
        return int(value)  # 文字列やnp.int等にも寛容
    except Exception:
        return 0

# ---- 未実装プレースホルダ ----
def _not_implemented(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    print("[warn] this subcommand is not implemented yet.")
    return 0

def _maybe(mod, default: Callable[[Optional[List[str]], Dict[str, Any]], int]):
    """モジュールがあれば run() を (argv,cfg) で呼べるようにラップ。無ければ default。"""
    if mod is None:
        return default
    fn = getattr(mod, "run", None)
    if not callable(fn):
        return default

    def _wrapped(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())

            if "argv" in params and "cfg" in params:
                rc = fn(argv=argv, cfg=cfg)
            elif len(params) == 2:
                # 2引数だが名前不定：順序で推測
                try:
                    rc = fn(argv, cfg)  # type: ignore[misc]
                except TypeError:
                    rc = fn(cfg, argv)  # type: ignore[misc]
            elif len(params) == 1:
                # 旧API: run(cfg)
                rc = fn(cfg)  # type: ignore[misc]
            else:
                # 引数無し
                rc = fn()  # type: ignore[misc]

            return _normalize_rc(rc)

        except SystemExit as e:
            # サブコマンド内の argparse などが SystemExit を投げた場合
            return _normalize_rc(getattr(e, "code", 1))

    return _wrapped

# ---- コマンド定義 ----
COMMANDS: Dict[str, Callable[[Optional[List[str]], Dict[str, Any]], int]] = {
    "split":            _maybe(split_mod, _not_implemented),
    "train":            _maybe(train_mod, _not_implemented),
    "infer-pool":       _maybe(infer_pool_mod, _not_implemented),
    "confidence":       _maybe(confidence_mod, _not_implemented),
    "gate":             _maybe(gate_mod, _not_implemented),
    "al-sample":        _not_implemented,
    "al-label-import":  _not_implemented,
    "al-cycle":         _not_implemented,
    "viz":              _not_implemented,
    "eval":             _not_implemented,
}

# ---- 共通フラグ（トップレベル） ----
def _parse_overrides(sets: List[str]) -> Dict[str, Any]:
    """--set KEY=VAL ... を辞書化（VAL型解決は load_config 側）。"""
    kv: Dict[str, Any] = {}
    for s in sets or []:
        if "=" not in s:
            raise SystemExit(f"[error] invalid --set entry (expect KEY=VAL): {s}")
        k, v = s.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tensaku", add_help=True)
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
    argv = sys.argv[1:]
    if not argv:
        build_parser().print_help()
        return 2

    # 1) 最初に現れるサブコマンドの位置を見つけて、argvを二分
    cmd_idx = None
    for i, tok in enumerate(argv):
        if tok in COMMANDS:
            cmd_idx = i
            break
    if cmd_idx is None:
        print(f"[error] missing subcommand. choices={list(COMMANDS.keys())}")
        return 2

    head = argv[:cmd_idx]          # 上位（-c/--set だけをここで解釈）
    subcmd = argv[cmd_idx]         # 'gate' など
    subargv = argv[cmd_idx + 1:]   # サブコマンド専用引数（上位は一切触らない）

    # 2) 上位パーサは head + [subcmd] のみ解釈（subargvは渡さない）
    parser = build_parser()
    ns = parser.parse_args(head + [subcmd])

    cfg = load_config(ns.config, ns.set)

    # 3) サブコマンドを実行（subargvはそのまま渡す）
    run = COMMANDS.get(ns.cmd)
    if run is None:
        print(f"[error] unknown command: {ns.cmd}")
        return 2

    # run は (argv, cfg) を受け取る実装にしてある想定
    return int(run(argv=subargv, cfg=cfg)) if 'argv' in run.__code__.co_varnames else int(run(cfg))  # 両対応


if __name__ == "__main__":
    sys.exit(main())
