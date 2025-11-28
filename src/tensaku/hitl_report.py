# /home/esakit25/work/tensaku/src/tensaku/hitl_report.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.hitl_report
@role     : HITL 一周の成果を集約し hitl_summary.csv/json を生成する。
"""
from __future__ import annotations
import csv
import json
import os
import sys
import argparse
import traceback # 追加
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.metrics import f1_score

# --- 1. JSON保存用の強化エンコーダー ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def _coerce_int_or_none(v: Optional[str]) -> Optional[int]:
    if v is None: return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan": return None
    try: return int(float(s))
    except: return None

def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0: return float("nan")
    return float(np.sqrt(((p - y) ** 2).mean()))

def _cse_rate(y: np.ndarray, p: np.ndarray, cse_abs_err: int) -> float:
    if y.size == 0: return float("nan")
    bad = np.abs(p - y) >= int(cse_abs_err)
    return float(bad.mean())

def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0: return float("nan")
    return float((y == p).mean())

def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0: return float("nan")
    return float(f1_score(y, p, average='macro'))

def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0: return float("nan")
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    n_class = int(max(y_true.max(), y_pred.max())) + 1
    O = np.zeros((n_class, n_class), dtype=float)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_class and 0 <= p < n_class: O[t, p] += 1.0
    N = float(O.sum())
    if N == 0.0: return float("nan")
    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / N
    W = np.zeros((n_class, n_class), dtype=float)
    for i in range(n_class):
        for j in range(n_class):
            W[i, j] = ((i - j) ** 2) / float((n_class - 1) ** 2 if n_class > 1 else 1)
    num = (W * O).sum()
    den = (W * E).sum()
    return float(1.0 - num / den) if den != 0.0 else float("nan")

def _load_split_metrics_from_detail(detail_path: str, cse_abs_err: int) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(detail_path): return {}
    by_split: Dict[str, Dict[str, List[int]]] = {}
    with open(detail_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip()
            yt = _coerce_int_or_none(row.get("y_true"))
            yp = _coerce_int_or_none(row.get("y_pred"))
            if yt is None or yp is None: continue
            if split not in by_split: by_split[split] = {"y_true": [], "y_pred": []}
            by_split[split]["y_true"].append(yt)
            by_split[split]["y_pred"].append(yp)
    metrics = {}
    for split, vals in by_split.items():
        y_true = np.array(vals["y_true"], dtype=int)
        y_pred = np.array(vals["y_pred"], dtype=int)
        metrics[split] = {
            "N": float(y_true.shape[0]),
            "rmse": _rmse(y_true, y_pred),
            "qwk": _qwk(y_true, y_pred),
            "cse": _cse_rate(y_true, y_pred, cse_abs_err),
            "acc": _accuracy(y_true, y_pred),
            "f1": _macro_f1(y_true, y_pred),
        }
    return metrics

def _load_gate_meta(path: str) -> Dict[str, Any]:
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def _load_aurc_summary(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path): return {}
    summary = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("conf_key")
            if key: summary[key] = row
    return summary

def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tag", type=str, default=None)
    ns, _ = ap.parse_known_args(argv or [])

    run_cfg = cfg.get("run") or {}
    hitl_cfg = cfg.get("hitl") or {}
    
    out_dir = str(run_cfg.get("out_dir") or "./outputs")
    os.makedirs(out_dir, exist_ok=True)

    qid = str(run_cfg.get("qid") or os.path.basename(out_dir))
    run_id = str(ns.tag or run_cfg.get("run_id") or "")
    
    cse_abs_err = int(hitl_cfg.get("cse_abs_err", 2))

    f_detail = os.path.join(out_dir, "preds_detail.csv")
    f_gate_meta = os.path.join(out_dir, "gate_meta.json")
    f_aurc = os.path.join(out_dir, "aurc_summary.csv")

    print(f"[hitl-report] Reading from {out_dir} ...")

    split_metrics = _load_split_metrics_from_detail(f_detail, cse_abs_err)
    gate_meta = _load_gate_meta(f_gate_meta)
    aurc_summary = _load_aurc_summary(f_aurc)

    # Gate info
    conf_key = gate_meta.get("conf_key", "")
    tau = gate_meta.get("tau", "")
    
    gate_splits = gate_meta.get("splits", {})
    dev_info = gate_splits.get("dev", {})
    eval_info = gate_splits.get("test", gate_splits.get("pool", {}))

    dev_full = split_metrics.get("dev", {})
    test_full = split_metrics.get("test", {})

    aurc_data = aurc_summary.get(conf_key, {})
    if not aurc_data and aurc_summary:
        aurc_data = list(aurc_summary.values())[0]

    # CSV Row
    row = {
        "run_id": run_id, "qid": qid, "conf_key": conf_key, "tau": tau,
        # Gate
        "dev_coverage_gate": dev_info.get("coverage", ""),
        "dev_rmse_gate": dev_info.get("rmse", ""),
        "test_coverage_gate": eval_info.get("coverage", ""),
        "test_rmse_gate": eval_info.get("rmse", ""),
        # Full
        "dev_qwk_full": dev_full.get("qwk", ""),
        "dev_rmse_full": dev_full.get("rmse", ""),
        "dev_cse_full": dev_full.get("cse", ""),
        "dev_acc_full": dev_full.get("acc", ""),
        "dev_f1_full": dev_full.get("f1", ""),
        "test_qwk_full": test_full.get("qwk", ""),
        "test_rmse_full": test_full.get("rmse", ""),
        "test_cse_full": test_full.get("cse", ""),
        "test_acc_full": test_full.get("acc", ""),
        "test_f1_full": test_full.get("f1", ""),
        # AURC
        "aurc_cse": aurc_data.get("aurc_cse", ""),
        "aurc_rmse": aurc_data.get("aurc_rmse", ""),
    }

    summary_csv_path = os.path.join(out_dir, "hitl_summary.csv")
    
    try:
        file_exists = os.path.exists(summary_csv_path)
        cols = list(row.keys())
        with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists: writer.writeheader()
            writer.writerow(row)
        print(f"[hitl-report] Wrote CSV -> {summary_csv_path}")
    except Exception as e:
        print(f"[hitl-report] ERROR writing CSV: {e}", file=sys.stderr)
        traceback.print_exc() # 詳細表示
        return 1

    # JSON
    summary_json_path = os.path.join(out_dir, "hitl_summary.json")
    try:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            # --- 2. NumpyEncoder を使用 ---
            json.dump(
                {"row": row, "gate": gate_meta, "metrics": split_metrics}, 
                f, 
                indent=2,
                cls=NumpyEncoder # これでNumPy型エラーを回避
            )
        print(f"[hitl-report] Wrote JSON -> {summary_json_path}")
    except Exception as e:
        # --- 3. 詳細なエラー表示 ---
        print(f"[hitl-report] ERROR writing JSON failed: {e}", file=sys.stderr)
        traceback.print_exc()
        # エラー時は 1 を返す（hitl.sh がこれを検知して止まるようにする）
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(run(None, {"run": {}}))