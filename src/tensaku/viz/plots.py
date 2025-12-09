# /home/esakit25/work/tensaku/src/tensaku/viz/plots.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.viz.plots
@role  : 具体的なグラフ描画関数群 (Learning Curve, Reliability, Risk-Coverage, Hist, CM)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensaku.viz import metrics

# --- 1. Reliability Diagram ---
def plot_reliability_diagram(
    ax: plt.Axes,
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram"
) -> None:
    bin_conf, bin_acc, _ = metrics.compute_reliability_bins(probs, y_true, n_bins)
    mask = ~np.isnan(bin_acc)
    bin_conf = bin_conf[mask]
    bin_acc = bin_acc[mask]

    ax.plot([0, 1], [0, 1], "--", lw=1, color="gray", label="Perfect")
    ax.plot(bin_conf, bin_acc, "o-", lw=1.5, label="Model", color="tab:blue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":")

# --- 2. Risk-Coverage Curve ---
def plot_risk_coverage(
    ax: plt.Axes,
    series: List[Tuple[str, Dict[str, np.ndarray]]],
    xlabel: str = "Coverage",
    ylabel: str = "Risk",
    title: str = "Risk-Coverage Curve"
) -> None:
    for name, data in series:
        cov = data.get("coverage")
        risk = data.get("risk")
        if cov is not None and risk is not None:
            ax.plot(cov, risk, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":")

# --- 3. AL Learning Curve ---
def plot_learning_curve_aggregate(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    x_col: str = "n_labeled"
) -> None:
    mean_col = f"{metric_col}_mean"
    std_col = f"{metric_col}_std"
    if mean_col not in df_agg.columns: return

    labels = df_agg["plot_label"].unique()
    colors = sns.color_palette(n_colors=len(labels))

    for i, label in enumerate(labels):
        sub = df_agg[df_agg["plot_label"] == label].sort_values(x_col)
        x = sub[x_col]
        y = sub[mean_col]
        ax.plot(x, y, label=label, color=colors[i], marker='o')
        if std_col in sub.columns:
            y_std = sub[std_col].fillna(0)
            ax.fill_between(x, y - y_std, y + y_std, color=colors[i], alpha=0.2)

    ax.set_xlabel("Number of Labeled Samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":")

# --- 4. Histogram (Confidence Distribution) ---
def plot_histogram(
    ax: plt.Axes,
    df: pd.DataFrame,
    cols: List[str],
    n_bins: int = 20,
    title: str = "Confidence Distribution"
) -> None:
    """確信度の分布ヒストグラムを描画する。"""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols: return

    for c in valid_cols:
        data = df[c].dropna().to_numpy(dtype=float)
        # alpha=0.5 で重ねて表示
        ax.hist(data, bins=n_bins, range=(0, 1), alpha=0.5, label=c)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":")

# --- 5. Confusion Matrix ---
def plot_confusion_matrix(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix"
) -> None:
    """混同行列をヒートマップで描画する。"""
    if len(y_true) == 0: return
    
    # ラベル集合（正解と予測の和集合）
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")