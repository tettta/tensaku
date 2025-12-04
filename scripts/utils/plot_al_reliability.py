# scripts/utils/plot_al_reliability.py
# -*- coding: utf-8 -*-
"""
@module    : scripts/utils/plot_al_reliability.py
@role      : preds_detail.csv から信頼性評価（AURC, Reliability Diagram）を実行
@outputs   : curve_coverage_*.png, reliability_diagram*.png, conf_hist*.png, aurc_summary.csv
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse
import csv
import os
import sys
import glob 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 【修正】tensaku.calibration への依存を削除し、スタンドアロンで動作するようにする
# =============================================================================
# 1. 計算用ユーティリティ (外部モジュール依存を排除)
# =============================================================================

def _load_detail_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # 【重要修正】FileNotFoundErrorを発生させ、呼び出し元で強制終了させる
        raise FileNotFoundError(f"Detail file not found: {path}")
    return pd.read_csv(path)

def _build_probs_from_softmax(df: pd.DataFrame) -> np.ndarray | None:
    prob_cols = [c for c in df.columns if c.startswith("probs_")]
    if not prob_cols:
        return None
    prob_cols_sorted = sorted(prob_cols, key=lambda x: int(x.split("_")[1]))
    probs = df[prob_cols_sorted].to_numpy(dtype=float, copy=True)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return probs / row_sum

def _build_probs_from_conf(conf: np.ndarray, y_pred: np.ndarray, n_class: int) -> np.ndarray:
    """Softmax確率がない場合、Confをそのクラスの確率として簡易復元する"""
    N = conf.shape[0]
    probs = np.zeros((N, n_class), dtype=float)
    idx = (np.arange(N), y_pred.astype(int))
    probs[idx] = conf
    return probs

def _simple_reliability_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """簡易版 Reliability Diagram ビニング計算"""
    conf = np.max(probs, axis=1)
    y_pred = np.argmax(probs, axis=1)
    is_correct = (y_pred == y_true).astype(float)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(conf, bins) - 1
    
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_accs.append(np.mean(is_correct[mask]))
            bin_confs.append(np.mean(conf[mask]))
        else:
            # 元のコードのように nan を保持
            bin_accs.append(np.nan) 
            bin_confs.append(np.nan) 
            
    return np.array(bin_confs), np.array(bin_accs)

# =============================================================================
# 2. Risk-Coverage / AURC 計算 (変更なし)
# =============================================================================

def compute_risk_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    *,
    cse_abs_err: int,
) -> dict[str, np.ndarray]:
    """conf 降順ソート後の各指標（Risk-Coverage）推移を計算"""
    if y_true.size == 0:
        return {}

    order = np.argsort(-conf)
    y = y_true[order].astype(float)
    p = y_pred[order].astype(float)

    N = y.shape[0]
    cov = np.arange(1, N + 1, dtype=float) / float(N)

    # RMSE
    diff2 = (p - y) ** 2
    cum_diff2 = np.cumsum(diff2)
    rmse = np.sqrt(cum_diff2 / np.arange(1, N + 1, dtype=float))

    # CSE (Bound)
    bad = (np.abs(p - y) >= int(cse_abs_err)).astype(float)
    cum_bad = np.cumsum(bad)
    cse = cum_bad / np.arange(1, N + 1, dtype=float)

    # Error Rate (1 - Accuracy)
    wrong = (p != y).astype(float)
    cum_wrong = np.cumsum(wrong)
    error = cum_wrong / np.arange(1, N + 1, dtype=float)

    return {"coverage": cov, "cse": cse, "rmse": rmse, "error": error}


def _aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    if coverage.size == 0:
        return float("nan")
    return float(np.trapz(risk, coverage))

# =============================================================================
# 3. プロット処理 (変更なし)
# =============================================================================

def plot_reliability_diagram(df: pd.DataFrame, path_out: str, conf_col: str, n_bins: int) -> None:
    if conf_col not in df.columns:
        return

    conf = df[conf_col].to_numpy(dtype=float)
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)

    probs = _build_probs_from_softmax(df)
    if probs is None:
        n_class = int(max(y_true.max(), y_pred.max())) + 1
        probs = _build_probs_from_conf(conf, y_pred, n_class)

    centers, bin_acc = _simple_reliability_bins(probs, y_true, n_bins)
    
    mask = ~np.isnan(centers)
    centers = centers[mask]
    bin_acc = bin_acc[mask]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", lw=1, color="gray", label="Perfect")
    ax.plot(centers, bin_acc, "o-", lw=1.5, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability ({conf_col})")
    ax.legend()
    ax.grid(True, ls=":")

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_hist_multi(df: pd.DataFrame, path_out: str, cols: list[str], n_bins: int) -> None:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in valid_cols:
        data = df[c].to_numpy(dtype=float)
        ax.hist(data, bins=n_bins, range=(0, 1), alpha=0.5, label=c)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, ls=":")
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_rc_curves(
    series: list[tuple[str, dict[str, np.ndarray]]],
    cse_abs_err: int,
    out_dir: str,
) -> dict[str, dict[str, float]]:
    """Risk-Coverage曲線の描画とAURCの算出"""
    if not series:
        return {}

    metrics = [
        ("cse", f"CSE (|err|>={cse_abs_err})", "curve_coverage_cse.png"),
        ("rmse", "RMSE", "curve_coverage_rmse.png"),
        ("error", "Error Rate (1-Acc)", "curve_coverage_error.png"),
    ]

    aurc_summary: dict[str, dict[str, float]] = {}

    for key, label, fname in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        has_data = False
        
        for name, rc in series:
            if key not in rc: continue
            
            cov = rc["coverage"]
            risk = rc[key]
            ax.plot(cov, risk, label=name)
            has_data = True

            if name not in aurc_summary:
                aurc_summary[name] = {}
            aurc_summary[name][f"aurc_{key}"] = _aurc(cov, risk)

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Coverage")
        ax.set_ylabel(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":")
        ax.legend()
        
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)

    return aurc_summary

# =============================================================================
# Main Process
# =============================================================================

def process_experiment(exp_dir: str, out_dir: str, args):
    """単一実験ディレクトリに対する処理"""
    exp_name = os.path.basename(exp_dir.rstrip("/"))
    
    # 【修正】入力ファイルパスを exp_dir 直下から取得
    detail_path = os.path.join(exp_dir, "preds_detail.csv")
    
    if not os.path.exists(detail_path):
        print(f"[reliability] ERROR: {detail_path} not found.", file=sys.stderr)
        # 【重要修正】ファイルがない場合はエラーを発生させ、Bash側で捕捉させる
        sys.exit(1)

    print(f"[reliability] Processing: {exp_name}")
    try:
        df = _load_detail_csv(detail_path)
    except Exception as e:
        print(f"[reliability] ERROR loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # フィルタリング (split指定がある場合)
    df_target = df.copy()
    if "split" in df.columns and args.split:
        df_target = df_target[df_target["split"] == args.split].copy()
    
    df_target = df_target[df_target["y_true"].notna()]
    
    if df_target.empty:
        print(f"[reliability] WARN: Filtered data for split '{args.split}' is empty.", file=sys.stderr)
        # データが空でも、エラーではないので正常終了

    # --- 1. Risk-Coverage Curves & AURC ---
    rc_keys_default = ["conf_msp", "conf_trust"]
    rc_keys = [c for c in rc_keys_default if c in df_target.columns]
    if not rc_keys:
        rc_keys = [c for c in df_target.columns if c.startswith("conf_")]

    series = []
    for k in rc_keys:
        y_true = df_target["y_true"].to_numpy(float)
        y_pred = df_target["y_pred"].to_numpy(float)
        conf = df_target[k].to_numpy(float)
        rc = compute_risk_coverage(y_true, y_pred, conf, cse_abs_err=int(args.cse_abs_err))
        series.append((k, rc))

    aurc_data = plot_rc_curves(series, int(args.cse_abs_err), out_dir)

    # AURC CSV出力
    if aurc_data:
        p_aurc = os.path.join(out_dir, "aurc_summary.csv")
        with open(p_aurc, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conf_key", "aurc_cse", "aurc_rmse", "aurc_error", "split", "N"])
            N = len(df_target)
            for k, v in aurc_data.items():
                w.writerow([
                    k,
                    v.get("aurc_cse", ""),
                    v.get("aurc_rmse", ""),
                    v.get("aurc_error", ""),
                    args.split,
                    N
                ])
        print(f"[reliability] Saved AURC summary to: {p_aurc}")

    # --- 2. Reliability Diagram & Histograms ---
    if "conf_msp" in df_target.columns:
        rel_keys = ["conf_msp"]
    else:
        rel_keys = [c for c in df_target.columns if c.startswith("conf_")][:1]

    for k in rel_keys:
        plot_reliability_diagram(
            df_target,
            os.path.join(out_dir, f"reliability_diagram_{k}.png"),
            k,
            int(args.bins),
        )

    # Histogram
    hist_keys = [c for c in df_target.columns if c.startswith("conf_")]
    # 【修正】元の viz.py に合わせる
    hist_candidates: List[str] = []
    for c in ("conf_msp", "conf_trust", "conf_msp_temp"):
        if c in df_target.columns:
            hist_candidates.append(c)
    if not hist_candidates:
        hist_candidates = [c for c in df_target.columns if c.startswith("conf_")]
    
    hist_keys = hist_candidates[:2]
    
    if hist_keys:
        plot_hist_multi(
            df_target,
            os.path.join(out_dir, "conf_hist.png"), # 元のコードのファイル名 (conf_hist_msp_vs_trust.png) を簡略化
            hist_keys,
            int(args.bins),
        )
    
    print(f"[reliability] SUCCESS. Plots saved in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Active Learning Reliability Evaluation")
    
    # 【修正】report_al.sh から呼ばれるシングルモード用に引数を定義
    parser.add_argument("--exp-dir", type=str, help="Input experiment directory (containing preds_detail.csv)")
    parser.add_argument("--out-dir", type=str, help="Output directory for plots")
    
    # Batch mode params (従来の動作)
    parser.add_argument("--qid", type=str, default=None, help="Batch mode: Question ID")
    parser.add_argument("--root", type=str, default=".", help="Batch mode: Project root")

    # Plot params
    parser.add_argument("--bins", type=int, default=15, help="Number of bins (default: 15)")
    parser.add_argument("--split", type=str, default="test", help="Target split (default: test)")
    parser.add_argument("--cse-abs-err", type=int, default=2, help="CSE threshold (default: 2)")

    args = parser.parse_args()

    # モード判定
    target_entries = [] # List of (exp_dir, out_dir)

    if args.exp_dir and args.out_dir:
        # Single mode (report_al.sh から呼ばれるメインルート)
        target_entries.append((args.exp_dir, args.out_dir))
    
    elif args.qid and args.root:
        # Batch mode (元の viz.py の動作)
        base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
        exp_dirs = sorted(glob.glob(os.path.join(base_dir, "exp_*")))
        exp_dirs += sorted(glob.glob(os.path.join(base_dir, "test_*")))
        
        for ed in exp_dirs:
            target_entries.append((ed, os.path.join(ed, "plots"))) # バッチモードの出力は plots/ とする
    else:
        print("[reliability] ERROR: Must specify --exp-dir/--out-dir or --qid/--root.", file=sys.stderr)
        sys.exit(1)

    # 実行ループ
    for exp_d, out_d in target_entries:
        # process_experiment内でエラーを捕捉し、sys.exit(1)させる
        # このため、外側の try-except は不要
        process_experiment(exp_d, out_d, args)

if __name__ == "__main__":
    main()