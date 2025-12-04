# /home/esakit25/work/tensaku/src/tensaku/hitl_report.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.hitl_report
@role     : HITL 一周の成果を集約し hitl_summary.csv/json を生成する。
            (viz.py に依存せず、全ての指標を自力で計算する自立版)
"""
from __future__ import annotations
import csv
import json
import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.metrics import f1_score

# --- 外部ライブラリ ---
try:
    from tensaku.calibration import ece
except ImportError:
    def ece(probs, y, n_bins=15): return float("nan")

# --- 1. JSON保存用の強化エンコーダー ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- 2. 基本メトリクス計算 ---
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

# --- 3. AURC 計算ロジック (viz.py から移植・独立化) ---
def _calc_aurc_scores(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, cse_abs_err: int) -> Dict[str, float]:
    """
    Coverage-Risk 曲線を計算し、AURC (CSE/RMSE) を返す
    """
    if y_true.size == 0:
        return {"aurc_cse": float("nan"), "aurc_rmse": float("nan")}

    # 確信度降順にソート
    order = np.argsort(-conf)
    y = y_true[order].astype(float)
    p = y_pred[order].astype(float)

    N = y.shape[0]
    # coverage: 1/N, 2/N, ..., 1.0
    cov = np.arange(1, N + 1, dtype=float) / float(N)

    # RMSE Curve
    diff2 = (p - y) ** 2
    cum_diff2 = np.cumsum(diff2)
    # rmse_k = sqrt( sum(diff^2) / k )
    rmse_curve = np.sqrt(cum_diff2 / np.arange(1, N + 1, dtype=float))
    
    # CSE Curve
    bad = (np.abs(p - y) >= int(cse_abs_err)).astype(float)
    cum_bad = np.cumsum(bad)
    cse_curve = cum_bad / np.arange(1, N + 1, dtype=float)

    # AURC (台形積分)
    aurc_rmse = float(np.trapz(rmse_curve, cov))
    aurc_cse = float(np.trapz(cse_curve, cov))

    return {"aurc_cse": aurc_cse, "aurc_rmse": aurc_rmse}

# --- 4. データロード ---
def _load_split_metrics_from_detail(detail_path: str, cse_abs_err: int) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(detail_path): return {}
    by_split: Dict[str, Dict[str, List[int]]] = {}
    
    # 全データを一度読み込む (AURC計算やECEで使うため)
    # メモリ効率のため、ここではsplitごとの集計用リストだけ作る
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

def _build_probs(df: pd.DataFrame) -> Optional[np.ndarray]:
    cols = [c for c in df.columns if c.startswith("probs_")]
    if not cols: return None
    cols = sorted(cols, key=lambda x: int(x.split("_")[1]))
    return df[cols].to_numpy(dtype=float)

# --- 5. メイン実行 ---
def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--n-labeled", type=int, default=None)
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

    print(f"[hitl-report] Reading from {out_dir} ...")

    # 基本指標 (QWK, RMSE, CSE, Acc, F1)
    split_metrics = _load_split_metrics_from_detail(f_detail, cse_abs_err)
    
    # Gate情報
    gate_meta = _load_gate_meta(f_gate_meta)
    conf_key = gate_meta.get("conf_key", "")
    tau = gate_meta.get("tau", "")
    gate_splits = gate_meta.get("splits", {})
    dev_info = gate_splits.get("dev", {})
    eval_info = gate_splits.get("test", gate_splits.get("pool", {}))

    dev_full = split_metrics.get("dev", {})
    test_full = split_metrics.get("test", {})

    # --- 高度な指標の計算 (AURC, ECE) ---
    aurc_cse = float("nan")
    aurc_rmse = float("nan")
    test_ece = float("nan")
    
    try:
        if os.path.exists(f_detail):
            df_all = pd.read_csv(f_detail)
            # conf_key が指定されていれば、それを使って AURC を計算
            if conf_key and conf_key in df_all.columns:
                # Testセット (なければDev) で計算
                # 一般的にはTestでのAURCを見る
                target_split = "test"
                df_target = df_all[df_all["split"] == target_split]
                if df_target.empty:
                    target_split = "dev"
                    df_target = df_all[df_all["split"] == target_split]
                
                if not df_target.empty and "y_true" in df_target.columns:
                    df_labeled = df_target.dropna(subset=["y_true", conf_key])
                    y_true = df_labeled["y_true"].to_numpy(dtype=int)
                    y_pred = df_labeled["y_pred"].to_numpy(dtype=int)
                    conf = df_labeled[conf_key].to_numpy(dtype=float)
                    
                    # AURC 計算実行
                    aurc_res = _calc_aurc_scores(y_true, y_pred, conf, cse_abs_err)
                    aurc_cse = aurc_res["aurc_cse"]
                    aurc_rmse = aurc_res["aurc_rmse"]

            # ECE 計算 (Test)
            df_test = df_all[df_all["split"] == "test"]
            if not df_test.empty and "y_true" in df_test.columns:
                y_true_ece = df_test["y_true"].dropna().to_numpy(dtype=int)
                probs = _build_probs(df_test)
                if probs is not None and len(y_true_ece) == len(probs):
                    test_ece = ece(probs, y_true_ece)

    except Exception as e:
        print(f"[hitl-report] WARN: Advanced metrics calculation failed: {e}")
        # 致命的ではないので続行

    # CSV Row
    row = {
        "run_id": run_id, "qid": qid, "conf_key": conf_key, "tau": tau,
        "n_labeled": ns.n_labeled,
        # Gate Metrics
        "dev_coverage_gate": dev_info.get("coverage", ""),
        "dev_rmse_gate": dev_info.get("rmse", ""),
        "test_coverage_gate": eval_info.get("coverage", ""),
        "test_rmse_gate": eval_info.get("rmse", ""),
        "test_cse_gate": eval_info.get("cse", ""),
        # Full Model Metrics
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
        "test_ece_full": test_ece,
        # AURC
        "aurc_cse": aurc_cse,
        "aurc_rmse": aurc_rmse,
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
        traceback.print_exc()
        return 1

    # JSON
    summary_json_path = os.path.join(out_dir, "hitl_summary.json")
    try:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"row": row, "gate": gate_meta, "metrics": split_metrics}, 
                f, indent=2, cls=NumpyEncoder
            )
        print(f"[hitl-report] Wrote JSON -> {summary_json_path}")
    except Exception as e:
        print(f"[hitl-report] ERROR writing JSON failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(run(None, {"run": {}}))