# /home/esakit25/work/tensaku/src/tensaku/gate.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.gate
@role  : HITL ゲート計算 (Strict Mode)
@overview:
    - 閾値探索 (find_tau_for_constraint) と適用 (apply_tau)。
    - Strict: デフォルト設定値を持たず、Configからの明示的なパラメータ入力を要求する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


# =============================================================================
# dataclasses
# =============================================================================

@dataclass
class GateConfig:
    """
    ゲートの挙動を決める設定 (Strict)。
    デフォルト値を持たず、明示的な初期化を要求する設計とするが、
    利便性のため dataclass 上は型定義のみとし、Factory で厳格化する。
    """
    eps_cse: float
    cse_abs_err: int
    higher_is_better: bool


@dataclass
class GateInputs:
    """ゲート処理の入力 (1 split 分)。"""
    y_true: np.ndarray
    y_pred: np.ndarray
    conf: np.ndarray
    ids: Optional[np.ndarray] = None


@dataclass
class GateDevResult:
    """dev 上で tau を探索した結果。"""
    tau: Optional[float]
    coverage: float
    cse: float
    rmse: float


@dataclass
class GateApplyResult(GateDevResult):
    """ある split に tau を適用した結果。"""
    mask_auto: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    mask_human: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


# =============================================================================
# helpers
# =============================================================================

def _to_numpy_1d(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0: return 0.0
    diff = y_pred.astype(float) - y_true.astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def _cse_rate(y_true: np.ndarray, y_pred: np.ndarray, cse_abs_err: int) -> float:
    if y_true.size == 0: return 0.0
    err = np.abs(y_pred.astype(float) - y_true.astype(float))
    return float(np.mean(err >= float(cse_abs_err)))


def create_gate_config(cfg: Mapping[str, Any]) -> GateConfig:
    """
    Strict Factory: Config辞書から GateConfig を生成。
    キー欠損は KeyError となる。
    """
    # Strict: 全て必須
    if "eps_cse" not in cfg: raise KeyError("Gate config missing 'eps_cse'")
    if "cse_abs_err" not in cfg: raise KeyError("Gate config missing 'cse_abs_err'")
    if "higher_is_better" not in cfg: raise KeyError("Gate config missing 'higher_is_better'")

    return GateConfig(
        eps_cse=float(cfg["eps_cse"]),
        cse_abs_err=int(cfg["cse_abs_err"]),
        higher_is_better=bool(cfg["higher_is_better"]),
    )


# =============================================================================
# core logic
# =============================================================================

def find_tau_for_constraint(
    dev: GateInputs, 
    cfg: Mapping[str, Any] | GateConfig
) -> GateDevResult:
    """dev 上で CSE <= eps_cse を満たしつつ coverage 最大の tau を探索する。"""
    
    gc = cfg if isinstance(cfg, GateConfig) else create_gate_config(cfg)
    
    y_true = _to_numpy_1d(dev.y_true)
    y_pred = _to_numpy_1d(dev.y_pred)
    conf = _to_numpy_1d(dev.conf)

    assert y_true.shape == y_pred.shape == conf.shape, "Length mismatch"

    n = conf.size
    if n == 0:
        return GateDevResult(tau=None, coverage=0.0, cse=0.0, rmse=0.0)

    # NaN 除去
    mask_valid = np.isfinite(conf)
    if not mask_valid.all():
        y_true = y_true[mask_valid]
        y_pred = y_pred[mask_valid]
        conf = conf[mask_valid]
        n = conf.size
        if n == 0:
            return GateDevResult(tau=None, coverage=0.0, cse=0.0, rmse=0.0)

    # 全自動の場合
    full_cse = _cse_rate(y_true, y_pred, gc.cse_abs_err)
    full_rmse = _rmse(y_true, y_pred)

    if full_cse <= gc.eps_cse:
        # 全件採用 (閾値は min/max で設定して全て通過させる)
        tau = float(np.min(conf)) if gc.higher_is_better else float(np.max(conf))
        return GateDevResult(tau=tau, coverage=1.0, cse=full_cse, rmse=full_rmse)

    # ソートして探索
    # higher_is_better=True (信頼度高い順) => 降順ソート
    order = np.argsort(conf)
    if gc.higher_is_better:
        order = order[::-1]

    conf_sorted = conf[order]
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    best_tau: Optional[float] = None
    best_cov: float = 0.0
    best_cse: float = 0.0
    best_rmse: float = 0.0

    # k=1..n まで試す (k=0 は coverage=0)
    # 高速化のために累積和等を使うこともできるが、ここではナイーブ実装でOK
    # (Nが数万程度ならPythonループでも許容範囲。遅ければ Numba/Cython 化)
    
    # Strict Note: 以前のロジックを維持
    for k in range(1, n + 1):
        # top-k
        sub_true = y_true_sorted[:k]
        sub_pred = y_pred_sorted[:k]
        
        cse_k = _cse_rate(sub_true, sub_pred, gc.cse_abs_err)
        
        if cse_k <= gc.eps_cse:
            # 制約を満たすなら候補更新
            # coverage は単調増加するので、後で見つかったものが常にベストカバレッジ
            best_tau = float(conf_sorted[k - 1])
            best_cov = k / n
            best_cse = cse_k
            best_rmse = _rmse(sub_true, sub_pred)
        
        # 制約を満たさなくなっても、後ろでまた満たす可能性(分布の偏り)はあるため break しない
        # (ただし一般的には信頼度が下がるほどエラー率は上がる傾向)

    if best_tau is None:
        return GateDevResult(tau=None, coverage=0.0, cse=0.0, rmse=0.0)

    return GateDevResult(tau=best_tau, coverage=best_cov, cse=best_cse, rmse=best_rmse)


def apply_tau(
    inputs: GateInputs, 
    tau: float, 
    cfg: Mapping[str, Any] | GateConfig
) -> GateApplyResult:
    """任意の split に tau を適用する。"""
    
    gc = cfg if isinstance(cfg, GateConfig) else create_gate_config(cfg)
    
    y_true = _to_numpy_1d(inputs.y_true)
    y_pred = _to_numpy_1d(inputs.y_pred)
    conf = _to_numpy_1d(inputs.conf)

    if tau is None:
        mask_auto = np.zeros(conf.shape, dtype=bool)
        mask_human = ~mask_auto
        return GateApplyResult(
            tau=None,
            coverage=0.0,
            cse=float("nan"),   # 自動採点が0件なので定義しない
            rmse=float("nan"),  # 同上
            mask_auto=mask_auto,
            mask_human=mask_human,
        )

    if conf.size == 0:
        empty = np.zeros(0, dtype=bool)
        return GateApplyResult(tau=tau, coverage=0.0, cse=0.0, rmse=0.0, mask_auto=empty, mask_human=empty)

    if gc.higher_is_better:
        mask_auto = conf >= tau
    else:
        mask_auto = conf <= tau
    mask_human = ~mask_auto

    y_true_auto = y_true[mask_auto]
    y_pred_auto = y_pred[mask_auto]

    cov = float(np.mean(mask_auto))
    cse = _cse_rate(y_true_auto, y_pred_auto, gc.cse_abs_err)
    rmse = _rmse(y_true_auto, y_pred_auto)

    return GateApplyResult(
        tau=float(tau),
        coverage=cov,
        cse=cse,
        rmse=rmse,
        mask_auto=mask_auto,
        mask_human=mask_human,
    )