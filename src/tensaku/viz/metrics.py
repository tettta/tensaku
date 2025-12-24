# /home/esakit25/work/tensaku/src/tensaku/viz/metrics.py
# -*- coding: utf-8 -*-
"""tensaku.viz.metrics

Numerical routines for visualization:
  - Risk-Coverage curves (CSE / RMSE / error)
  - Reliability binning from generic confidence scores

Notes
- Confidence scores are treated as *monotonic* (higher=more confident), but not necessarily probabilities.
  For reliability diagrams, we min-max normalize confidence into [0, 1] per series to avoid empty bins
  when the score range is arbitrary (e.g., energy, negative-entropy).
"""

from __future__ import annotations

from typing import Literal, Tuple
import numpy as np


RiskMetric = Literal["cse", "rmse", "error"]


def compute_risk_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    *,
    risk_metric: RiskMetric = "cse",
    cse_abs_err: int = 2,
) -> dict:
    """Compute risk-coverage curve by sorting by confidence descending.

    Returns a dict with keys:
      - coverage: np.ndarray shape [n]
      - risk: np.ndarray shape [n]
      - n: int
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    conf = np.asarray(conf)

    if y_true.shape[0] == 0:
        return {"coverage": np.asarray([]), "risk": np.asarray([]), "n": 0}

    if not (y_true.shape[0] == y_pred.shape[0] == conf.shape[0]):
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, conf={conf.shape}")

    order = np.argsort(-conf)  # high -> low
    yt = y_true[order]
    yp = y_pred[order]

    n = yt.shape[0]
    k = np.arange(1, n + 1, dtype=float)
    coverage = k / float(n)

    if risk_metric == "cse":
        err = np.abs(yp - yt) >= int(cse_abs_err)
        cumsum = np.cumsum(err.astype(float))
        risk = cumsum / k
    elif risk_metric == "error":
        err = (yp != yt).astype(float)
        risk = np.cumsum(err) / k
    elif risk_metric == "rmse":
        sq = (yp - yt).astype(float) ** 2
        risk = np.sqrt(np.cumsum(sq) / k)
    else:
        raise ValueError(f"unknown risk_metric: {risk_metric}")

    return {"coverage": coverage, "risk": risk, "n": int(n)}


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if abs(hi - lo) < 1e-12:
        # Constant score; place in the middle.
        z = np.zeros_like(x, dtype=float)
        z[finite] = 0.5
        z[~finite] = np.nan
        return z
    z = (x - lo) / (hi - lo)
    z[~finite] = np.nan
    return z


def compute_reliability_bins_from_conf(
    conf: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability binning from a generic confidence score.

    Steps:
      1) min-max normalize confidence into [0,1] (per series)
      2) bin into equal-width bins on [0,1]
      3) for each bin, compute mean confidence and accuracy

    Returns:
      - bin_conf: mean normalized confidence per bin (nan if empty)
      - bin_acc : accuracy per bin (nan if empty)
      - bin_count: count per bin (0 if empty)
    """
    conf = np.asarray(conf, dtype=float)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if not (conf.shape[0] == y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(f"shape mismatch: conf={conf.shape}, y_true={y_true.shape}, y_pred={y_pred.shape}")

    conf01 = _minmax01(conf)
    correct = (y_pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # place x==1.0 into last bin
    idx = np.clip(np.digitize(conf01, edges, right=False) - 1, 0, int(n_bins) - 1)

    bin_conf = np.full(int(n_bins), np.nan, dtype=float)
    bin_acc = np.full(int(n_bins), np.nan, dtype=float)
    bin_count = np.zeros(int(n_bins), dtype=int)

    for b in range(int(n_bins)):
        m = (idx == b) & np.isfinite(conf01) & np.isfinite(correct)
        c = int(m.sum())
        bin_count[b] = c
        if c == 0:
            continue
        bin_conf[b] = float(np.nanmean(conf01[m]))
        bin_acc[b] = float(np.nanmean(correct[m]))

    return bin_conf, bin_acc, bin_count
