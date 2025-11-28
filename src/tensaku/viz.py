# /home/esakit25/work/tensaku/src/tensaku/viz.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.viz
@role     : preds_detail.csv から可視化（Coverage曲線、信頼性図、ヒストグラム）を生成
@outputs  : curve_coverage_*.png, reliability_diagram*.png, conf_hist*.png, aurc_summary.csv
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .calibration import reliability_bins as _reliability_bins
except Exception:
    _reliability_bins = None


# =============================================================================
# 基本ユーティリティ
# =============================================================================

def _load_detail_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df

def _find_conf_column(df: pd.DataFrame) -> str:
    for c in ("conf_msp", "conf", "trust"):
        if c in df.columns: return c
    raise KeyError("no confidence column found")

def _build_probs_from_softmax(df: pd.DataFrame) -> Optional[np.ndarray]:
    prob_cols = [c for c in df.columns if c.startswith("probs_")]
    if not prob_cols: return None
    prob_cols_sorted = sorted(prob_cols, key=lambda x: int(x.split("_")[1]))
    probs = df[prob_cols_sorted].to_numpy(dtype=float, copy=True)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return probs / row_sum

def _build_probs_from_conf(conf: np.ndarray, y_pred: np.ndarray, n_class: int) -> np.ndarray:
    N = conf.shape[0]
    probs = np.zeros((N, n_class), dtype=float)
    idx = (np.arange(N), y_pred.astype(int))
    probs[idx] = conf
    return probs


# =============================================================================
# Risk-Coverage 計算
# =============================================================================

def compute_risk_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    *,
    cse_abs_err: int,
) -> Dict[str, np.ndarray]:
    """
    conf 降順の prefix k ごとの指標推移を計算
    """
    if y_true.size == 0:
        return {
            "coverage": np.zeros(0), "cse": np.zeros(0), 
            "rmse": np.zeros(0), "error": np.zeros(0)
        }

    order = np.argsort(-conf)
    y = y_true[order].astype(float)
    p = y_pred[order].astype(float)

    N = y.shape[0]
    cov = np.arange(1, N + 1, dtype=float) / float(N)

    # RMSE
    diff2 = (p - y) ** 2
    cum_diff2 = np.cumsum(diff2)
    rmse = np.sqrt(cum_diff2 / np.arange(1, N + 1, dtype=float))

    # CSE
    bad = (np.abs(p - y) >= int(cse_abs_err)).astype(float)
    cum_bad = np.cumsum(bad)
    cse = cum_bad / np.arange(1, N + 1, dtype=float)

    # Error Rate (1 - Accuracy)
    wrong = (p != y).astype(float)
    cum_wrong = np.cumsum(wrong)
    error = cum_wrong / np.arange(1, N + 1, dtype=float)

    return {"coverage": cov, "cse": cse, "rmse": rmse, "error": error}


def _aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    if coverage.size == 0: return float("nan")
    return float(np.trapz(risk, coverage))


# =============================================================================
# プロット関数
# =============================================================================

def plot_reliability(df: pd.DataFrame, path_out: str, conf_col: str, n_bins: int) -> None:
    if conf_col not in df.columns: return
    
    conf = df[conf_col].to_numpy(dtype=float)
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)

    probs = _build_probs_from_softmax(df)
    if probs is None:
        n_class = int(max(y_true.max(), y_pred.max())) + 1
        probs = _build_probs_from_conf(conf, y_pred, n_class)

    if _reliability_bins:
        bins = _reliability_bins(probs, y_true, n_bins=n_bins)
        centers = (bins["bin_lower"] + bins["bin_upper"]) / 2.0
        bin_acc = bins["accuracy"]
    else:
        centers = np.linspace(0, 1, n_bins)
        bin_acc = centers 

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", lw=1, color="gray")
    ax.plot(centers, bin_acc, "o-", lw=1.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability ({conf_col})")
    ax.grid(True, ls=":")
    
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_hist_multi(df: pd.DataFrame, path_out: str, cols: Sequence[str], n_bins: int) -> None:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols: return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in valid_cols:
        data = df[c].to_numpy(dtype=float)
        ax.hist(data, bins=n_bins, range=(0, 1), alpha=0.5, label=c)
    
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, ls=":")
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_rc_curves(
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    cse_abs_err: int,
    out_dir: str
) -> Dict[str, Dict[str, float]]:
    """
    CSE, RMSE, Accuracy(Error) のCoverage曲線を描画し、AURCを返す
    """
    if not series: return {}
    
    metrics = [
        ("cse", f"CSE (|err|>={cse_abs_err})", "curve_coverage_cse.png"),
        ("rmse", "RMSE", "curve_coverage_rmse.png"),
        ("error", "Error Rate (1-Acc)", "curve_coverage_error.png")
    ]
    
    aurc_summary = {}

    for key, label, fname in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, rc in series:
            cov = rc["coverage"]
            risk = rc[key]
            ax.plot(cov, risk, label=name)
            
            if name not in aurc_summary: aurc_summary[name] = {}
            aurc_summary[name][f"aurc_{key}"] = _aurc(cov, risk)

        ax.set_xlabel("Coverage")
        ax.set_ylabel(label)
        ax.set_xlim(0, 1)
        
        # 【修正】Y軸を0スタート・上限オートスケールに変更
        # これにより、値が小さい(0.05とか)場合でもグラフが見やすくなる
        ax.set_ylim(bottom=0)
        # もし値が大きければ1.0でクリップしてもよいが、オートの方が安全
        
        ax.grid(True, ls=":")
        ax.legend()
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        
    return aurc_summary


# =============================================================================
# Main
# =============================================================================

def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--detail", type=str)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--split", type=str)
    parser.add_argument("--cse-abs-err", type=int)
    ns, _ = parser.parse_known_args(argv or [])

    run_cfg = cfg.get("run") or {}
    viz_cfg = cfg.get("viz") or {}
    eval_cfg = cfg.get("eval") or {}
    out_dir = str(run_cfg.get("out_dir") or "./outputs")

    detail_path = ns.detail or viz_cfg.get("detail_path") or os.path.join(out_dir, "preds_detail.csv")
    n_bins = int(ns.bins or eval_cfg.get("n_bins", 15))
    cse_err = int(ns.cse_abs_err or eval_cfg.get("cse_abs_err", 2))
    target_split = str(ns.split or viz_cfg.get("split_for_risk", "test"))

    try:
        df = _load_detail_csv(str(detail_path))
    except FileNotFoundError:
        print(f"[viz] ERROR: {detail_path} not found", file=sys.stderr)
        return 2

    # --- Plot 1: Risk-Coverage Curves ---
    if "split" in df.columns and target_split:
        df_rc = df[df["split"] == target_split].copy()
    else:
        df_rc = df.copy()
    
    df_rc = df_rc[df_rc["y_true"].notna()]

    rc_keys = viz_cfg.get("risk_coverage_conf_keys", ["conf_msp", "conf_trust"])
    series = []
    for k in rc_keys:
        if k in df_rc.columns:
            y_true = df_rc["y_true"].to_numpy(float)
            y_pred = df_rc["y_pred"].to_numpy(float)
            conf = df_rc[k].to_numpy(float)
            rc = compute_risk_coverage(y_true, y_pred, conf, cse_abs_err=cse_err)
            series.append((k, rc))
            
    aurc_data = plot_rc_curves(series, cse_err, out_dir)
    
    # Save AURC
    if aurc_data:
        p_aurc = os.path.join(out_dir, "aurc_summary.csv")
        with open(p_aurc, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conf_key", "aurc_cse", "aurc_rmse", "aurc_error", "split", "N"])
            N = len(df_rc)
            for k, v in aurc_data.items():
                w.writerow([
                    k, 
                    v.get("aurc_cse", ""), 
                    v.get("aurc_rmse", ""), 
                    v.get("aurc_error", ""),
                    target_split, N
                ])

    # --- Plot 2: Reliability & Hist ---
    df_all = df[df["y_true"].notna()].copy()
    
    rel_keys = viz_cfg.get("reliability_conf_keys", ["conf_msp"])
    for k in rel_keys:
        plot_reliability(df_all, os.path.join(out_dir, f"reliability_diagram_{k}.png"), k, n_bins)
        
    hist_keys = viz_cfg.get("hist_multi_conf_keys", ["conf_msp", "conf_trust"])
    plot_hist_multi(df_all, os.path.join(out_dir, "conf_hist_msp_vs_trust.png"), hist_keys, n_bins)

    print(f"[viz] Done. Saved plots to {out_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(run(None, {"run": {}}))