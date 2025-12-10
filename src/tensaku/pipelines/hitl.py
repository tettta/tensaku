# /home/esakit25/work/tensaku/src/tensaku/pipelines/hitl.py
# -*- coding: utf-8 -*-
"""tensaku.pipelines.hitl
==========================

@module: tensaku.pipelines.hitl
@role  : 予測結果 (preds_detail 相当の DataFrame) と gate 設定から、
         gate_core を用いて HITL 閾値探索と適用を行う配列ベースのパイプライン。

@overview:
    - preds_detail 相当の DataFrame を入力として受け取る（ファイル I/O は行わない）。
    - 呼び出し元から渡された「具体的な確信度列名 (`conf_column_name`)」と
      「Gate設定 (`gate_cfg`)」に基づいて計算を行う。
    - ファイルパスや実験全体の `cfg` 構造には依存しない（Pure Logic）。

@inputs:
    - pandas.DataFrame: `split`, `y_true`, `y_pred` および指定された確信度列を含む表。
    - gate_cfg (Mapping): GateConfig を構築するための辞書 (eps_cse, cse_abs_err 等)。
    - conf_column_name (str): DataFrame 内の確信度列の名前 (例: "conf_trust")。

@outputs:
    - HitlOutputs: tau, dev/test/pool の GateResult をまとめた dataclass。

@notes:
    - Phase2 改修: 以前の `_choose_conf_column` 等のタスク固有マッピングロジックは廃止。
      列名の解決は呼び出し元 (Task層) の責務とする。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from tensaku.gate import (
    GateConfig,
    GateInputs,
    GateDevResult,
    GateApplyResult,
    find_tau_for_constraint,
    apply_tau,
)

LOGGER = logging.getLogger(__name__)


# =====================================================================
# dataclasses
# =====================================================================


@dataclass
class HitlOutputs:
    """gate_core による HITL 結果をまとめたコンテナ。"""

    conf_column_name: str
    tau: Optional[float]
    dev: GateDevResult
    test: Optional[GateApplyResult]
    pool: Optional[GateApplyResult]

    def to_summary_dict(self) -> Dict[str, Any]:
        """簡易的な要約 dict を返すユーティリティ。"""
        row: Dict[str, Any] = {
            "conf_column": self.conf_column_name,
            "tau": self.tau,
        }

        # dev
        row.update(
            {
                "dev_coverage_auto": self.dev.coverage,
                "dev_cse_auto": self.dev.cse,
                "dev_rmse_auto": self.dev.rmse,
            }
        )

        # test
        if self.test is not None:
            row.update(
                {
                    "test_coverage_auto": self.test.coverage,
                    "test_cse_auto": self.test.cse,
                    "test_rmse_auto": self.test.rmse,
                }
            )

        # pool
        if self.pool is not None:
            row.update(
                {
                    "pool_coverage_auto": self.pool.coverage,
                    "pool_cse_auto": self.pool.cse,
                    "pool_rmse_auto": self.pool.rmse,
                }
            )

        return row


# =====================================================================
# 内部ユーティリティ
# =====================================================================


def _extract_gate_inputs(df: pd.DataFrame, col_conf: str) -> GateInputs:
    """preds_detail 形式の DataFrame から GateInputs を構築する。"""
    required = {"y_true", "y_pred", col_conf}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[hitl] 必須列が不足しています: {missing} (必要: {required})")

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    conf = df[col_conf].to_numpy()

    # id がある場合は ids として渡す
    ids = df["id"].to_numpy() if "id" in df.columns else None
    return GateInputs(y_true=y_true, y_pred=y_pred, conf=conf, ids=ids)


def _coerce_gate_config(cfg: Mapping[str, Any]) -> GateConfig:
    """辞書から GateConfig を構築する。"""
    # cfg が既に GateConfig の場合もあり得るが、ここでは Mapping を想定
    eps_cse = float(cfg.get("eps_cse", 0.05))
    cse_abs_err = int(cfg.get("cse_abs_err", 2))
    higher_is_better = bool(cfg.get("higher_is_better", True))
    return GateConfig(
        eps_cse=eps_cse, cse_abs_err=cse_abs_err, higher_is_better=higher_is_better
    )


# =====================================================================
# 公開 API
# =====================================================================


def run_hitl_from_detail_df(
    df: pd.DataFrame,
    gate_cfg: Mapping[str, Any],
    conf_column_name: str,
) -> HitlOutputs:
    """preds_detail 形式の DataFrame と Gate設定から HITL ゲート処理を実行する。

    Parameters
    ----------
    df:
        preds_detail 形式の DataFrame。
        必須列: 'split', 'y_true', 'y_pred', および conf_column_name で指定される列。
    gate_cfg:
        GateConfig 用の設定辞書。
        キー例: "eps_cse", "cse_abs_err", "higher_is_better"
    conf_column_name:
        使用する確信度列の具体的な名前 (例: "conf_trust", "conf_msp")。
        呼び出し元で解決済みの名前を渡すこと。

    Returns
    -------
    HitlOutputs
        tau, dev/test/pool の GateResult をまとめたコンテナ。
    """
    if "split" not in df.columns:
        raise ValueError(
            "[hitl] DataFrame に 'split' 列が存在しません。dev/test/pool の判別に必要です。"
        )

    # 確信度列の存在確認
    if conf_column_name not in df.columns:
        raise ValueError(
            f"[hitl] 指定された確信度列 '{conf_column_name}' が DataFrame に存在しません。\n"
            f"  利用可能な列: {list(df.columns)}"
        )

    # split ごとに DataFrame を分割
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]
    pool_df = df[df["split"] == "pool"] if (df["split"] == "pool").any() else None

    if dev_df.empty:
        raise ValueError("[hitl] dev split が空です。tau 探索には dev データが必要です。")

    # GateConfig 構築
    gc = _coerce_gate_config(gate_cfg)

    # 1. dev で tau を探索
    dev_inputs = _extract_gate_inputs(dev_df, conf_column_name)
    dev_res = find_tau_for_constraint(dev_inputs, gc)

    if dev_res.tau is None:
        LOGGER.warning("[hitl] 条件を満たす tau が見つかりませんでした (dev)。")
        return HitlOutputs(
            conf_column_name=conf_column_name,
            tau=None,
            dev=dev_res,
            test=None,
            pool=None,
        )

    tau = dev_res.tau

    # 2. test への適用
    test_res: Optional[GateApplyResult] = None
    if test_df is not None and not test_df.empty:
        test_inputs = _extract_gate_inputs(test_df, conf_column_name)
        test_res = apply_tau(test_inputs, tau, gc)

    # 3. pool への適用
    pool_res: Optional[GateApplyResult] = None
    if pool_df is not None and not pool_df.empty:
        pool_inputs = _extract_gate_inputs(pool_df, conf_column_name)
        pool_res = apply_tau(pool_inputs, tau, gc)

    return HitlOutputs(
        conf_column_name=conf_column_name,
        tau=tau,
        dev=dev_res,
        test=test_res,
        pool=pool_res,
    )