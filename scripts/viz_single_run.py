# /home/esakit25/work/tensaku/scripts/viz_single_run.py
# -*- coding: utf-8 -*-
"""
単一実験 (1 qid / 1 exp / 1 seed) 向けの可視化スクリプト。

- 入力:
    - outputs/<qid>/<exp>/<seed>/predictions/final/preds_detail.csv
- 出力:
    - outputs/<qid>/<exp>/<seed>/plots/risk_coverage_cse_test.png
    - outputs/<qid>/<exp>/<seed>/plots/risk_coverage_rmse_test.png
    - outputs/<qid>/<exp>/<seed>/plots/confusion_matrix_test.png
    - outputs/<qid>/<exp>/<seed>/plots/conf_hist_test_<conf_key>.png

使い方例:
    cd /home/esakit25/work/tensaku

    # trust を信頼度として可視化
    python scripts/viz_single_run.py \
        --qid Y14_1-2_1_3 \
        --exp al_trust_b50 \
        --seed 42 \
        --conf-key conf_trust

    # MSP を信頼度として可視化
    python scripts/viz_single_run.py \
        --qid Y14_1-2_1_3 \
        --exp al_msp_b50 \
        --seed 42 \
        --conf-key conf_msp
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensaku.viz import base, metrics, plots


def _resolve_seed_dir(seed_arg: str) -> str:
    """
    seed の指定をディレクトリ名に正規化する。

    - "42"   -> "seed42"
    - "seed7" -> "seed7"（そのまま）
    """
    if seed_arg.isdigit():
        return f"seed{seed_arg}"
    return seed_arg


def load_test_split(exp_dir: Path, conf_key: str):
    """
    preds_detail.csv から test split の (y_true, y_pred, conf) を取り出す。
    """
    preds_path = exp_dir / "predictions" / "final" / "preds_detail.csv"
    if not preds_path.is_file():
        raise FileNotFoundError(f"preds_detail.csv not found: {preds_path}")

    df = pd.read_csv(preds_path)
    required_cols = {"split", "y_true", "y_pred", conf_key}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in preds_detail.csv: {missing}")

    df_test = df[df["split"] == "test"].copy()
    if df_test.empty:
        raise ValueError("No test rows found in preds_detail.csv (split=='test').")

    # 欠損を落とす
    df_test = df_test.dropna(subset=["y_true", "y_pred", conf_key])

    y_true = df_test["y_true"].to_numpy(dtype=np.int64)
    y_pred = df_test["y_pred"].to_numpy(dtype=np.int64)
    conf = df_test[conf_key].to_numpy(dtype=float)

    return y_true, y_pred, conf


def plot_risk_coverage_curves(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    conf_label: str,
    title_prefix: str,
) -> None:
    """CSE / RMSE の Risk-Coverage 曲線を出力する。"""
    # CSE ベース
    rc_cse = metrics.compute_risk_coverage(
        y_true=y_true,
        y_pred=y_pred,
        conf=conf,
        risk_metric="cse",
        cse_abs_err=2,
    )
    fig, ax = plt.subplots()
    plots.plot_risk_coverage(
        ax,
        series=[(conf_label, rc_cse)],
        xlabel="Coverage (test)",
        ylabel="CSE rate (|error|>=2)",
        title=f"{title_prefix} - Risk-Coverage (CSE)",
    )
    base.save_fig(fig, out_dir / "risk_coverage_cse_test.png")

    # RMSE ベース
    rc_rmse = metrics.compute_risk_coverage(
        y_true=y_true,
        y_pred=y_pred,
        conf=conf,
        risk_metric="rmse",
    )
    fig, ax = plt.subplots()
    plots.plot_risk_coverage(
        ax,
        series=[(conf_label, rc_rmse)],
        xlabel="Coverage (test)",
        ylabel="RMSE",
        title=f"{title_prefix} - Risk-Coverage (RMSE)",
    )
    base.save_fig(fig, out_dir / "risk_coverage_rmse_test.png")


def plot_confusion_matrix_test(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str,
) -> None:
    """Test データの混同行列をヒートマップで保存する。"""
    fig, ax = plt.subplots()
    plots.plot_confusion_matrix(
        ax=ax,
        y_true=y_true,
        y_pred=y_pred,
        title=f"{title_prefix} - Confusion Matrix (Test)",
    )
    base.save_fig(fig, out_dir / "confusion_matrix_test.png")


def plot_conf_hist(
    out_dir: Path,
    conf: np.ndarray,
    conf_label: str,
    title_prefix: str,
) -> None:
    """test 確信度のヒストグラムを出力する。"""
    fig, ax = plt.subplots()
    ax.hist(conf, bins=20, range=(0.0, 1.0))
    ax.set_xlabel(f"Confidence ({conf_label})")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} - Confidence Histogram (Test)")
    ax.grid(True, linestyle=":")
    base.save_fig(fig, out_dir / f"conf_hist_test_{conf_label}.png")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single AL experiment visualization (1 qid / 1 exp / 1 seed)."
    )
    parser.add_argument("--qid", required=True, help="Question ID (e.g. Y14_1-2_1_3)")
    parser.add_argument("--exp", required=True, help="Experiment name (e.g. al_trust_b50)")
    parser.add_argument(
        "--seed",
        required=True,
        help="Seed number or directory (e.g. 42 or seed42)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root (default: current directory)",
    )
    parser.add_argument(
        "--conf-key",
        default="conf_trust",
        help="Confidence column name in preds_detail.csv (e.g. conf_trust, conf_msp)",
    )

    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    seed_dir = _resolve_seed_dir(args.seed)

    exp_dir = root / "outputs" / args.qid / args.exp / seed_dir
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    plots_dir = exp_dir / "plots"

    # 共通スタイル設定
    base.setup_style()

    # データ読み込み
    y_true, y_pred, conf = load_test_split(exp_dir, args.conf_key)
    if y_true.size == 0:
        print("[viz] No test samples found. Nothing to plot.")
        return 0

    title_prefix = f"{args.qid} / {args.exp} / {seed_dir}"

    # 1) Risk-Coverage (CSE / RMSE)
    plot_risk_coverage_curves(
        out_dir=plots_dir,
        y_true=y_true,
        y_pred=y_pred,
        conf=conf,
        conf_label=args.conf_key,
        title_prefix=title_prefix,
    )

    # 2) Confusion Matrix (test)
    plot_confusion_matrix_test(
        out_dir=plots_dir,
        y_true=y_true,
        y_pred=y_pred,
        title_prefix=title_prefix,
    )

    # 3) Confidence histogram (test)
    plot_conf_hist(
        out_dir=plots_dir,
        conf=conf,
        conf_label=args.conf_key,
        title_prefix=title_prefix,
    )

    print(f"[viz] Done. Plots are saved under: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
