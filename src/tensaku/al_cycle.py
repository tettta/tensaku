# /home/esakit25/work/tensaku/src/tensaku/al_cycle.py
# -*- coding: utf-8 -*-
"""
(ヘッダ省略: spec/docstringは既存のまま)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import infer_pool, al_sample, al_label_import

AL_HISTORY_FILENAME = "al_history.csv"

# ... (_resolve_common_info, _safe_count_lines, _infer_next_round は変更なしのため省略可能ですが、
#      完全なファイルとして機能させるため、既存のコードをそのまま利用してください) ...

def _resolve_common_info(cfg: Dict[str, Any]) -> Dict[str, str]:
    # (既存の実装)
    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    out_dir = str(run_cfg.get("out_dir") or "./outputs")
    data_dir = str(run_cfg.get("data_dir") or "./data")
    qid = run_cfg.get("qid") or data_cfg.get("qid")
    if not qid:
        qid = os.path.basename(out_dir.rstrip("/"))
    run_id = run_cfg.get("run_id") or ""
    seed_val = run_cfg.get("seed")
    if seed_val is None:
        train_cfg = cfg.get("train") or {}
        seed_val = train_cfg.get("seed")
    seed = "" if seed_val is None else str(seed_val)
    return {
        "out_dir": str(out_dir),
        "data_dir": str(data_dir),
        "qid": str(qid),
        "run_id": str(run_id),
        "seed": seed,
    }

def _safe_count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def _infer_next_round(history_path: Path, qid: str, run_id: str) -> int:
    if not history_path.exists():
        return 0
    try:
        with history_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rounds: List[int] = []
            for row in reader:
                if not row: continue
                if row.get("qid") != qid: continue
                if row.get("run_id", "") != run_id: continue
                r = row.get("round")
                try:
                    if r is not None and r != "":
                        rounds.append(int(r))
                except Exception:
                    continue
    except Exception:
        return 0
    if not rounds: return 0
    return max(rounds) + 1

def _append_al_history(cfg: Dict[str, Any], ns: argparse.Namespace) -> None:
    # (既存の実装)
    info = _resolve_common_info(cfg)
    out_dir = Path(info["out_dir"])
    data_dir = Path(info["data_dir"])
    qid = info["qid"]
    run_id = info["run_id"]
    seed = info["seed"]
    history_path = out_dir / AL_HISTORY_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    round_idx = _infer_next_round(history_path, qid, run_id)
    al_cfg = cfg.get("al") or {}
    
    # sampler名の取得ロジック改良
    sampler_cli = getattr(ns, "sampler", None)
    if sampler_cli:
        sampler = str(sampler_cli)
    else:
        sampler = (
            al_cfg.get("sampler")
            or al_cfg.get("sampler_name")
            or al_cfg.get("name")
            or ""
        )

    by_cli = getattr(ns, "by", None)
    uncertainty_key = (by_cli or al_cfg.get("by") or "msp")
    if isinstance(uncertainty_key, str):
        uncertainty_key = uncertainty_key.lower()
    budget_val = getattr(ns, "k", None)
    if budget_val is None:
        budget_val = al_cfg.get("k")
    budget = "" if budget_val is None else str(budget_val)
    n_labeled = _safe_count_lines(data_dir / "labeled.jsonl")
    n_pool = _safe_count_lines(data_dir / "pool.jsonl")
    has_label_import = bool(getattr(ns, "labels", None))
    header = [
        "qid", "run_id", "round", "seed", "sampler",
        "uncertainty_key", "budget", "n_labeled", "n_pool", "has_label_import",
    ]
    row = {
        "qid": qid, "run_id": run_id, "round": str(round_idx), "seed": seed,
        "sampler": str(sampler), "uncertainty_key": str(uncertainty_key),
        "budget": budget, "n_labeled": str(n_labeled), "n_pool": str(n_pool),
        "has_label_import": "1" if has_label_import else "0",
    }
    if not history_path.exists():
        with history_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerow(row)
    else:
        with history_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(row)

def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--by", choices=["msp", "trust"], default=None)
    parser.add_argument("--sampler", type=str, default=None)  # 追加
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--skip-infer", action="store_true")
    ns, _rest = parser.parse_known_args(argv or [])

    # 1) infer_pool
    if ns.skip_infer:
        print("[al-cycle] skip infer_pool (using existing pool_preds.csv)")
    else:
        print("[al-cycle] step 1/3: infer_pool")
        rc = infer_pool.run(argv=[], cfg=cfg)
        if rc != 0:
            print(f"[al-cycle] ERROR: infer_pool.run failed with code {rc}")
            return rc

    # 2) al_sample
    print("[al-cycle] step 2/3: al_sample")
    al_argv: List[str] = []
    if ns.k is not None:
        al_argv += ["--k", str(ns.k)]
    if ns.by is not None:
        al_argv += ["--by", ns.by]
    if ns.threshold is not None:
        al_argv += ["--threshold", str(ns.threshold)]
    if ns.sampler is not None:         # 追加
        al_argv += ["--sampler", ns.sampler]
    if ns.ascending:
        al_argv.append("--ascending")

    rc = al_sample.run(argv=al_argv, cfg=cfg)
    if rc != 0:
        print(f"[al-cycle] ERROR: al_sample.run failed with code {rc}")
        return rc

    # 3) al_label_import
    if ns.labels:
        print("[al-cycle] step 3/3: al_label_import")
        rc = al_label_import.run(argv=["--labels", ns.labels], cfg=cfg)
        if rc != 0:
            print(f"[al-cycle] ERROR: al_label_import.run failed with code {rc}")
            return rc
        print("[al-cycle] done: labeled/pool updated via al_label_import.")
    else:
        print("[al-cycle] no --labels given; skipping import.")

    _append_al_history(cfg, ns)
    return 0

if __name__ == "__main__":
    print("Run via CLI: tensaku al-cycle ...")