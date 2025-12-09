# /home/esakit25/work/tensaku/src/tensaku/viz/cli.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import sys
from pathlib import Path


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensaku.viz import base, metrics, plots, compare

def _cmd_single(args):
    """単一実験の可視化 (Reliability, Risk-Coverage, Hist, CM)"""
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "plots"
    detail_path = exp_dir / "preds_detail.csv"
    
    if not detail_path.exists():
        print(f"[viz] ERROR: {detail_path} not found.")
        return 1
    
    print(f"[viz] Processing single experiment: {exp_dir}")
    df = pd.read_csv(detail_path)
    
    # Target Split (default: test)
    if "split" in df.columns:
        df = df[df["split"] == args.split]
    
    if df.empty:
        print(f"[viz] WARN: No data for split='{args.split}'")
        return 0

    # 1. Reliability Diagram (MSP)
    if "conf_msp" in df.columns and "y_true" in df.columns:
        y_true = df["y_true"].to_numpy(dtype=int)
        conf = df["conf_msp"].to_numpy(dtype=float)
        y_pred = df["y_pred"].to_numpy(dtype=int)
        n_class = int(max(y_true.max(), y_pred.max())) + 1
        
        probs = np.zeros((len(df), n_class))
        probs[np.arange(len(df)), y_pred] = conf
        
        fig, ax = plt.subplots()
        plots.plot_reliability_diagram(ax, probs, y_true, title=f"Reliability (MSP) - {args.split}")
        base.save_fig(fig, out_dir / "reliability_msp.png")

    # 2. Risk-Coverage (CSE) & AURC
    rc_series = []
    conf_cols = [c for c in df.columns if c.startswith("conf_")]
    
    for col in conf_cols:
        y_true = df["y_true"].to_numpy(dtype=float)
        y_pred = df["y_pred"].to_numpy(dtype=float)
        conf = df[col].to_numpy(dtype=float)
        
        rc_cse = metrics.compute_risk_coverage(y_true, y_pred, conf, risk_metric="cse", cse_abs_err=args.cse_abs_err)
        rc_series.append((col, rc_cse))
        
        aurc = metrics.compute_aurc(rc_cse["coverage"], rc_cse["risk"])
        print(f"  {col}: AURC(CSE) = {aurc:.4f}")

    if rc_series:
        fig, ax = plt.subplots()
        plots.plot_risk_coverage(ax, rc_series, ylabel=f"CSE (|err|>={args.cse_abs_err})", title=f"Risk-Coverage (CSE) - {args.split}")
        base.save_fig(fig, out_dir / "curve_rc_cse.png")

    # 3. Histogram (Confidence Distribution) ★追加
    if conf_cols:
        # 見やすくするため主要な2つ程度に絞るか、すべて出すか。ここでは全て出す。
        fig, ax = plt.subplots()
        plots.plot_histogram(ax, df, conf_cols, title=f"Confidence Distribution - {args.split}")
        base.save_fig(fig, out_dir / "conf_hist.png")

    # 4. Confusion Matrix ★追加
    if "y_true" in df.columns and "y_pred" in df.columns:
        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)
        
        fig, ax = plt.subplots(figsize=(8, 6)) # CMは少し大きめに
        plots.plot_confusion_matrix(ax, y_true, y_pred, title=f"Confusion Matrix - {args.split}")
        base.save_fig(fig, out_dir / "confusion_matrix.png")

    return 0


def _cmd_compare(args):
    """複数実験の比較 (Learning Curve)"""
    if not args.qid:
        print("[viz] ERROR: --qid is required for compare mode.")
        return 1
    
    root_dir = args.root or "."
    print(f"[viz] Collecting experiments for QID={args.qid}...")
    
    df = compare.collect_experiments(root_dir, args.qid, filter_str=args.filter)
    if df.empty:
        print("[viz] No data found.")
        return 1
    
    out_dir = Path(root_dir) / "outputs" / f"q-{args.qid}" / "summary_plots"
    
    if args.aggregate:
        print("[viz] Aggregating by sampler/uncertainty...")
        df_agg = compare.aggregate_experiments(df, group_by=args.group_by)
        
        fig, ax = plt.subplots()
        plots.plot_learning_curve_aggregate(ax, df_agg, "test_qwk_full", "QWK", "Test QWK Learning Curve")
        base.save_fig(fig, out_dir / "compare_qwk_agg.png")
        
        fig, ax = plt.subplots()
        plots.plot_learning_curve_aggregate(ax, df_agg, "test_cse_full", "CSE", "Test CSE Learning Curve")
        base.save_fig(fig, out_dir / "compare_cse_agg.png")
    else:
        print("[viz] Non-aggregated plot not fully implemented yet. Use --aggregate.")

    return 0


def run(argv: list[str] | None = None, cfg: dict[str, Any] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensaku viz")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: single
    p_single = subparsers.add_parser("single", help="Visualize single experiment result")
    p_single.add_argument("--exp-dir", required=True)
    p_single.add_argument("--out-dir", default=None)
    p_single.add_argument("--split", default="test")
    p_single.add_argument("--cse-abs-err", type=int, default=2)

    # Subcommand: compare
    p_compare = subparsers.add_parser("compare", help="Compare multiple experiments")
    p_compare.add_argument("--qid", required=True)
    p_compare.add_argument("--root", default=".")
    p_compare.add_argument("--filter", default=None)
    p_compare.add_argument("--aggregate", action="store_true")
    p_compare.add_argument("--group-by", nargs="+", default=["sampler", "uncertainty_key"])

    args = parser.parse_args(argv)
    base.setup_style()

    if args.subcommand == "single":
        return _cmd_single(args)
    elif args.subcommand == "compare":
        return _cmd_compare(args)
    
    return 0