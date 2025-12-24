# /home/esakit25/work/tensaku/src/tensaku/viz/plots.py
# -*- coding: utf-8 -*-
"""tensaku.viz.plots

Matplotlib-only plotting helpers used by tensaku.viz.cli.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_histogram(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    title: str = "",
    bins: int = 30,
) -> None:
    vals = []
    labels = []
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        if x.size == 0:
            continue
        vals.append(x)
        labels.append(c)

    if not vals:
        ax.set_title(title or "Histogram")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    # If one series: normal hist; if multiple: overlay with alpha.
    for i, x in enumerate(vals):
        ax.hist(x, bins=bins, alpha=0.55 if len(vals) > 1 else 0.9, label=labels[i])

    ax.set_title(title or "Histogram")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    if len(vals) > 1:
        ax.legend(loc="best")


def plot_confusion_matrix(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
    labels = [int(x) for x in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        cm = cm / denom

    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("true")
    ax.set_xlabel("pred")
    ax.set_title(title)

    # annotate
    fmt = ".2f" if normalize else "d"
    thresh = (np.nanmax(cm) if cm.size else 0) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )


def plot_reliability_diagram_bins(
    ax: plt.Axes,
    bin_conf: np.ndarray,
    bin_acc: np.ndarray,
    *,
    title: str = "Reliability Diagram",
) -> None:
    bin_conf = np.asarray(bin_conf, dtype=float)
    bin_acc = np.asarray(bin_acc, dtype=float)

    m = np.isfinite(bin_conf) & np.isfinite(bin_acc)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    if m.any():
        ax.plot(bin_conf[m], bin_acc[m], marker="o", linewidth=1.5)
    else:
        ax.text(0.5, 0.5, "no bins", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("normalized confidence (min-max)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)


def plot_risk_coverage(
    ax: plt.Axes,
    series: List[Tuple[str, dict]],
    *,
    title: str = "Risk-Coverage",
    ylabel: str = "risk",
) -> None:
    for name, rc in series:
        cov = np.asarray(rc.get("coverage", []), dtype=float)
        risk = np.asarray(rc.get("risk", []), dtype=float)
        if cov.size == 0:
            continue
        ax.plot(cov, risk, label=name)
    ax.set_xlabel("coverage")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(loc="best")


def plot_learning_curve_aggregate(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    *,
    metric_col: str,
    x_col: str,
    ylabel: str = "",
    title: str = "",
) -> None:
    """Plot mean±std learning curves.

    Expected columns:
      - plot_label
      - x_col
      - f"{metric_col}__mean"
      - f"{metric_col}__std"
    """
    mean_col = f"{metric_col}__mean"
    std_col = f"{metric_col}__std"
    for col in ["plot_label", x_col, mean_col, std_col]:
        if col not in df_agg.columns:
            raise KeyError(f"plot_learning_curve_aggregate missing column: {col}. available={list(df_agg.columns)}")

    for label, g in df_agg.groupby("plot_label", dropna=False):
        g = g.sort_values(x_col)
        x = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[mean_col], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(g[std_col], errors="coerce").to_numpy(dtype=float)

        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            continue

        ax.plot(x[m], y[m], label=str(label))
        # std could be nan or 0
        s = np.where(np.isfinite(s), s, 0.0)
        ax.fill_between(x[m], (y - s)[m], (y + s)[m], alpha=0.18)

    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(title or f"Learning Curve: {metric_col}")
    ax.legend(loc="best")
