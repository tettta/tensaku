# /home/esakit25/work/tensaku/src/tensaku/viz/metrics.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.viz.metrics
@role  : 可視化・評価のための数値計算ロジック（AURC, ECE, Reliability Diagram等）
@notes : 描画ライブラリ (matplotlib/seaborn) には依存しない。
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

def compute_risk_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    risk_metric: str = "cse",  # "cse", "rmse", "error"
    cse_abs_err: int = 2,
) -> Dict[str, np.ndarray]:
    """
    確信度(conf)に基づく Risk-Coverage 曲線のデータを計算する。
    
    Args:
        y_true: 正解ラベル
        y_pred: 予測ラベル
        conf: 確信度 (大きいほど自信あり)
        risk_metric: リスクの種類 ("cse", "rmse", "error")
    
    Returns:
        {"coverage": [...], "risk": [...]}
    """
    if y_true.size == 0:
        return {"coverage": np.array([]), "risk": np.array([])}

    # 確信度降順にソート
    order = np.argsort(-conf)
    y = y_true[order].astype(float)
    p = y_pred[order].astype(float)

    N = y.shape[0]
    coverage = np.arange(1, N + 1, dtype=float) / float(N)

    if risk_metric == "cse":
        # 重大誤採点率 (Cumulative)
        bad = (np.abs(p - y) >= int(cse_abs_err)).astype(float)
        cum_risk = np.cumsum(bad)
    elif risk_metric == "error":
        # 誤答率 (1 - Accuracy)
        wrong = (p != y).astype(float)
        cum_risk = np.cumsum(wrong)
    elif risk_metric == "rmse":
        # RMSE
        diff2 = (p - y) ** 2
        cum_risk = np.sqrt(np.cumsum(diff2) / np.arange(1, N + 1, dtype=float))
        # RMSEはここで計算終了なので返す
        return {"coverage": coverage, "risk": cum_risk}
    else:
        raise ValueError(f"Unknown risk_metric: {risk_metric}")

    risk = cum_risk / np.arange(1, N + 1, dtype=float)
    return {"coverage": coverage, "risk": risk}


def compute_aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Risk-Coverage 曲線下の面積 (AURC) を計算する。"""
    if coverage.size == 0:
        return float("nan")
    # 台形積分
    return float(np.trapz(risk, coverage))


def compute_reliability_bins(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reliability Diagram 用のビニングを行う。
    
    Returns:
        (bin_confidence, bin_accuracy, bin_counts)
    """
    conf = np.max(probs, axis=1)
    y_pred = np.argmax(probs, axis=1)
    is_correct = (y_pred == y_true).astype(float)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(conf, bins) - 1
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = binids == i
        count = np.sum(mask)
        bin_counts.append(count)
        
        if count > 0:
            bin_accs.append(np.mean(is_correct[mask]))
            bin_confs.append(np.mean(conf[mask]))
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            
    return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) を計算する。"""
    bin_confs, bin_accs, bin_counts = compute_reliability_bins(probs, y_true, n_bins)
    
    total = np.sum(bin_counts)
    if total == 0: return 0.0
    
    # nanを除外して計算
    mask = ~np.isnan(bin_accs)
    diff = np.abs(bin_accs[mask] - bin_confs[mask])
    weights = bin_counts[mask] / total
    
    return float(np.sum(weights * diff))