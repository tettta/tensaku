# /home/esakit25/work/tensaku/src/tensaku/gate_core.py
# -*- coding: utf-8 -*-
"""
tensaku.gate_core
=================

@module: tensaku.gate_core
@role  : HITL ゲートの「配列レベル」のコアロジックを提供するモジュール。

@overview:
    - dev 上で CSE≤eps_cse を満たしつつ coverage を最大化する閾値 tau を探索する。
    - 見つけた tau を test / pool に適用し、自動採点 vs 人手採点のマスクと
      coverage / CSE / RMSE などの指標を返す。
    - ファイル I/O は一切行わず、numpy 配列だけを扱う。

@notes:
    - 既存の tensaku.gate (CLI) からは、preds_detail.csv を読み込んで
      GateInputs を構築し、本モジュールの関数を呼び出す薄いラッパにしていく想定。
    - Phase2 では gate.conf_key を "trust" | "msp" | ... の短い名前で受け取り、
      列名 ("conf_trust" など) へのマッピングは gate 側で行う。
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
    """ゲートの挙動を決める設定。

    Attributes
    ----------
    eps_cse:
        dev 上で許容する重大誤採点率 CSE の上限 (0〜1)。
    cse_abs_err:
        重大誤採点とみなす |pred - true| の閾値。
    higher_is_better:
        True のとき conf が大きいほど「自動採点に回したい」とみなし、
        conf 降順の prefix で探索する。
    """

    eps_cse: float = 0.05
    cse_abs_err: int = 2
    higher_is_better: bool = True


@dataclass
class GateInputs:
    """ゲート処理の入力 (1 split 分)。

    y_true, y_pred, conf は shape (N,) の 1 次元配列を想定する。
    ids は任意で、caller 側で行 ID を保持したいときに利用する。
    """

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
    if y_true.size == 0:
        return 0.0
    diff = y_pred.astype(float) - y_true.astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def _cse_rate(y_true: np.ndarray, y_pred: np.ndarray, cse_abs_err: int) -> float:
    if y_true.size == 0:
        return 0.0
    err = np.abs(y_pred.astype(float) - y_true.astype(float))
    return float(np.mean(err >= float(cse_abs_err)))


def _coerce_config(cfg: Mapping[str, Any] | GateConfig) -> GateConfig:
    if isinstance(cfg, GateConfig):
        return cfg
    eps_cse = float(cfg.get("eps_cse", 0.05))
    cse_abs_err = int(cfg.get("cse_abs_err", 2))
    higher_is_better = bool(cfg.get("higher_is_better", True))
    return GateConfig(eps_cse=eps_cse, cse_abs_err=cse_abs_err, higher_is_better=higher_is_better)


# =============================================================================
# core logic
# =============================================================================


def find_tau_for_constraint(dev: GateInputs, cfg: Mapping[str, Any] | GateConfig) -> GateDevResult:
    """dev 上で CSE≤eps_cse を満たしつつ coverage 最大の tau を探索する。

    戻り値の tau が None の場合:
        - 条件 CSE≤eps_cse を満たす prefix が存在しなかったことを意味する。
        - coverage/cse/rmse は 0.0 とする（caller 側で WARN を出す想定）。
    """
    gc = _coerce_config(cfg)
    y_true = _to_numpy_1d(dev.y_true)
    y_pred = _to_numpy_1d(dev.y_pred)
    conf = _to_numpy_1d(dev.conf)

    assert y_true.shape == y_pred.shape == conf.shape, "y_true, y_pred, conf の長さが一致しません"

    n = conf.size
    if n == 0:
        return GateDevResult(tau=None, coverage=0.0, cse=0.0, rmse=0.0)

    # NaN を除去
    mask_valid = np.isfinite(conf)
    if not mask_valid.all():
        y_true = y_true[mask_valid]
        y_pred = y_pred[mask_valid]
        conf = conf[mask_valid]
        n = conf.size
        if n == 0:
            return GateDevResult(tau=None, coverage=0.0, cse=0.0, rmse=0.0)

    # 全体を自動採点に回した場合の CSE
    full_cse = _cse_rate(y_true, y_pred, gc.cse_abs_err)
    full_rmse = _rmse(y_true, y_pred)

    if full_cse <= gc.eps_cse:
        # すでに十分安全なら「全件 auto」とみなす
        return GateDevResult(
            tau=float(np.min(conf) if gc.higher_is_better else np.max(conf)),
            coverage=1.0,
            cse=full_cse,
            rmse=full_rmse,
        )

    # conf の降順 (higher_is_better=True) / 昇順 (False) に並べて prefix を走査
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

    for k in range(1, n + 1):
        auto_idx = slice(0, k)
        cse_k = _cse_rate(y_true_sorted[auto_idx], y_pred_sorted[auto_idx], gc.cse_abs_err)
        if cse_k <= gc.eps_cse:
            cov_k = k / n
            rmse_k = _rmse(y_true_sorted[auto_idx], y_pred_sorted[auto_idx])
            best_tau = float(conf_sorted[k - 1])
            best_cov = cov_k
            best_cse = cse_k
            best_rmse = rmse_k
        # 条件を満たさない場合はスキップ（他の k が条件を満たす可能性がある）

    if best_tau is None:
        return GateDevResult(tau=None, coverage=0.0, cse=0.0, rmse=0.0)

    return GateDevResult(tau=best_tau, coverage=best_cov, cse=best_cse, rmse=best_rmse)


def apply_tau(inputs: GateInputs, tau: float, cfg: Mapping[str, Any] | GateConfig) -> GateApplyResult:
    """任意の split に tau を適用し、mask_auto / 指標を計算する。"""
    gc = _coerce_config(cfg)
    y_true = _to_numpy_1d(inputs.y_true)
    y_pred = _to_numpy_1d(inputs.y_pred)
    conf = _to_numpy_1d(inputs.conf)

    assert y_true.shape == y_pred.shape == conf.shape, "y_true, y_pred, conf の長さが一致しません"

    if conf.size == 0:
        empty_mask = np.zeros(0, dtype=bool)
        return GateApplyResult(
            tau=tau,
            coverage=0.0,
            cse=0.0,
            rmse=0.0,
            mask_auto=empty_mask,
            mask_human=empty_mask,
        )

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
