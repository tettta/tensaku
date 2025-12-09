# /home/esakit25/work/tensaku/src/tensaku/cli.py
# -*- coding: utf-8 -*-
"""
@module     : tensaku.cli
@role       : tensaku コマンドのディスパッチ
"""
from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

from .config import load_config

SubcommandFunc = Callable[..., int]

def _parse_global_anywhere(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    gp = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    gp.add_argument("-c", "--config", default="./configs/exp_al_hitl.yaml")
    gp.add_argument("--set", metavar="KEY=VAL", nargs="*", action="append", default=[])
    ns, rest = gp.parse_known_args(argv)
    flat_set: List[str] = []
    for group in ns.set or []:
        if not group: continue
        flat_set.extend(group)
    ns.set = flat_set
    return ns, rest

def _sanitize_sets(sets: List[str]) -> List[str]:
    cleaned: List[str] = []
    for s in sets or []:
        if isinstance(s, str) and ("=" in s):
            cleaned.append(s)
        else:
            print(f"[cli] WARN: ignoring invalid --set: {s}")
    return cleaned

def _lazy_runner(module_name: str, func_name: str = "run") -> SubcommandFunc:
    def _run(*, argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
        full_name = f"tensaku.{module_name}"
        try:
            mod = __import__(full_name, fromlist=[func_name])
        except Exception as e:
            # 【重要】Importエラー時は明確にエラーを報告して終了コード1を返す
            print(f"[cli] ERROR: failed to import {full_name}: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
        
        fn = getattr(mod, func_name, None)
        if fn is None:
            print(f"[cli] ERROR: {full_name} has no callable '{func_name}'", file=sys.stderr)
            return 1
            
        try:
            ret = fn(argv=argv, cfg=cfg)
            return int(ret)
        except Exception as e:
            # 【重要】実行時エラーもキャッチしてログ出力＆終了コード1
            print(f"[cli] ERROR: command '{module_name}' failed: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
            
    return _run

def _not_implemented(*, argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    print("[cli] INFO: subcommand not implemented.")
    return 0

COMMANDS: Dict[str, SubcommandFunc] = {
    "split":            _lazy_runner("split"),
    "train":            _lazy_runner("train"),
    "infer-pool":       _lazy_runner("infer_pool"),

    "confidence":       _lazy_runner("confidence"),
    "gate":             _lazy_runner("gate"),
    "eval":             _lazy_runner("eval"),
    
    "al-run":           _lazy_runner("pipelines.al", func_name="run_experiment"),

    "config-dump":      _lazy_runner("config_dump"),
    "viz":              _lazy_runner("viz"),
}

def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = list(sys.argv[1:])
    gns, rest = _parse_global_anywhere(argv)
    gns.set = _sanitize_sets(gns.set)

    subcmd: Optional[str] = None
    cmd_idx: int = -1
    for i, t in enumerate(rest):
        if not t.startswith("-"):
            subcmd = t
            cmd_idx = i
            break

    if not subcmd:
        print("Usage: tensaku [-c CFG] [--set K=V] <command> ...")
        print("Commands:", ", ".join(sorted(COMMANDS.keys())))
        return 2

    subargv = rest[cmd_idx + 1:]
    
    try:
        cfg = load_config(gns.config, gns.set)
    except Exception as e:
        print(f"[cli] ERROR: Failed to load config: {e}", file=sys.stderr)
        return 1

    run_fn = COMMANDS.get(subcmd)
    if run_fn is None:
        print(f"[cli] ERROR: unknown command: {subcmd}", file=sys.stderr)
        return 2

    return run_fn(argv=subargv, cfg=cfg)

if __name__ == "__main__":
    sys.exit(main())