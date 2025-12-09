# /home/esakit25/work/tensaku/src/tensaku/pipelines/hitl.py
# -*- coding: utf-8 -*-
"""tensaku.pipelines.hitl
==========================

@module: tensaku.pipelines.hitl
@role  : 予測結果 (preds_detail 相当の DataFrame) と gate 設定から、
         gate_core を用いて HITL 閾値探索と適用を行う配列ベースのパイプライン。

@overview:
    - preds_detail 相当の DataFrame を入力として受け取る（ファイル I/O は行わない）。
    - `split` 列で dev/test/pool を判別し、指定された conf_key に対応する
      確信度列 (例: "conf_trust") を用いて gate_core.find_tau_for_constraint を実行。
    - 得られた tau を test/pool に適用し、coverage / CSE / RMSE などの GateApplyResult を返す。
    - conf_key は Phase2 では "trust" / "msp" / "entropy" ... といった短い名前で統一し、
      実際の列名 ("conf_trust" など) へのマッピングは本モジュールで行う。

@inputs:
    - pandas.DataFrame: `split`, `y_true`, `y_pred`, `conf_*` を含む preds_detail 形式の表。
    - cfg (Mapping): exp_al_hitl.yaml を読み込んだ dict を想定。
        - gate.eps_cse: dev 上で許容する CSE の上限 (0〜1, default=0.05)
        - gate.cse_abs_err: 重大誤採点 |pred-true| の閾値 (default=2)
        - gate.conf_key: "trust" / "msp" / ... の短い名前 (default="trust")

@outputs:
    - HitlOutputs: tau, dev/test/pool の GateResult をまとめた dataclass。
    - 呼び出し側で必要に応じて hitl_summary 行や gate_assign の作成に利用する。

@notes:
    - 既存の tensaku.gate (CLI) は当面そのまま維持しつつ、
      将来的には本モジュールのロジックを中心に据える想定。
    - 本モジュールはファイルパスや OUT_DIR には依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from tensaku.gate import (
    GateConfig,
    GateInputs,
    GateDevResult,
    GateApplyResult,
    find_tau_for_constraint,
    apply_tau,
)


# =====================================================================
# dataclasses
# =====================================================================


@dataclass
class HitlOutputs:
    """gate_core による HITL 結果をまとめたコンテナ。"""

    conf_key: str
    tau: Optional[float]
    dev: GateDevResult
    test: Optional[GateApplyResult]
    pool: Optional[GateApplyResult]

    def to_summary_dict(self) -> Dict[str, Any]:
        """簡易的な要約 dict を返すユーティリティ。

        ここでは gate 関連の指標 (tau, coverage, cse, rmse) のみをまとめる。
        必要に応じて hitl_report 相当の集約で拡張する。
        """
        row: Dict[str, Any] = {
            "conf_key": self.conf_key,
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


# Phase2 では conf_key を短い名前 ("trust" / "msp" / ...) で扱い、
# 実際の DataFrame 列名へのマッピングはここで行う。
CONF_COLUMN_MAP: Dict[str, str] = {
    "trust": "conf_trust",
    "msp": "conf_msp",
    "entropy": "conf_entropy",
    "margin": "conf_margin",
    "energy": "conf_energy",
}


def _choose_conf_column(df: pd.DataFrame, conf_key: str) -> str:
    """conf_key から DataFrame 上の確信度列名を決定する。

    - conf_key: "trust" / "msp" / ... を想定。
    - 対応する列が存在しない場合は ValueError を投げる。
    """
    key = conf_key.lower()
    col = CONF_COLUMN_MAP.get(key)
    if col is None:
        raise ValueError(f"[hitl] 未対応の conf_key が指定されました: {conf_key!r}")
    if col not in df.columns:
        raise ValueError(
            f"[hitl] conf_key={conf_key!r} に対応する列 {col!r} が DataFrame に存在しません。\n"
            f"  利用可能な列: {list(df.columns)}"
        )
    return col


def _extract_gate_inputs(df: pd.DataFrame, col_conf: str) -> GateInputs:
    """preds_detail 形式の DataFrame から GateInputs を構築する。"""
    required = {"y_true", "y_pred", col_conf}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[hitl] 必須列が不足しています: {missing} (必要: {required})")

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    conf = df[col_conf].to_numpy()

    # id がある場合は ids として渡す（使うかどうかは caller 次第）
    ids = df["id"].to_numpy() if "id" in df.columns else None
    return GateInputs(y_true=y_true, y_pred=y_pred, conf=conf, ids=ids)


def _coerce_gate_config(cfg: Mapping[str, Any]) -> GateConfig:
    """cfg['gate'] から GateConfig を構築する。"""
    gate_cfg = cfg.get("gate", {}) if isinstance(cfg, Mapping) else {}
    eps_cse = float(gate_cfg.get("eps_cse", 0.05))
    cse_abs_err = int(gate_cfg.get("cse_abs_err", 2))
    higher_is_better = bool(gate_cfg.get("higher_is_better", True))
    return GateConfig(eps_cse=eps_cse, cse_abs_err=cse_abs_err, higher_is_better=higher_is_better)


# =====================================================================
# 公開 API
# =====================================================================


def run_hitl_from_detail_df(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    conf_key: Optional[str] = None,
) -> HitlOutputs:
    """preds_detail 形式の DataFrame と cfg から HITL ゲート処理を実行する。

    Parameters
    ----------
    df:
        preds_detail 形式の DataFrame。最低限、以下の列を含むことを想定する:
          - 'split': 'dev' / 'test' / 'pool' 等の split 種別
          - 'y_true': 正解スコア
          - 'y_pred': 予測スコア
          - 'conf_*': 確信度列 (例: 'conf_trust')
    cfg:
        exp_al_hitl.yaml を読み込んだ dict。
        gate.eps_cse, gate.cse_abs_err, gate.higher_is_better を参照する。
    conf_key:
        使用する確信度の短い名前。None の場合は、
        gate.conf_key → run.conf_key → 'trust' の優先順位で決定する。

    Returns
    -------
    HitlOutputs
        tau (しきい値), dev/test/pool の GateResult をまとめたコンテナ。
        dev_result.tau が None の場合、test/pool には tau を適用せず None を返す。
    """
    if "split" not in df.columns:
        raise ValueError("[hitl] DataFrame に 'split' 列が存在しません。dev/test/pool の判別に必要です。")

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg, Mapping) else {}
    run_cfg = cfg.get("run", {}) if isinstance(cfg, Mapping) else {}

    if conf_key is None:
        conf_key = (
            str(gate_cfg.get("conf_key")).lower()
            if gate_cfg.get("conf_key") is not None
            else None
        )
        if not conf_key or conf_key in {"", "none", "null"}:
            # run.conf_key のほうを見て、それも無ければ Phase2 では 'trust' をデフォルトとする
            rk = run_cfg.get("conf_key")
            conf_key = str(rk).lower() if rk is not None else "trust"

    col_conf = _choose_conf_column(df, conf_key)

    # split ごとに DataFrame を分割
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]
    pool_df = df[df["split"] == "pool"] if (df["split"] == "pool").any() else None

    if dev_df.empty:
        raise ValueError("[hitl] dev split が空です。tau 探索には dev データが必要です。")

    gc = _coerce_gate_config(cfg)

    # dev で tau を探索
    dev_inputs = _extract_gate_inputs(dev_df, col_conf)
    dev_res = find_tau_for_constraint(dev_inputs, gc)

    if dev_res.tau is None:
        # 条件を満たす tau が見つからなかった場合は test/pool への適用を行わない。
        return HitlOutputs(conf_key=conf_key, tau=None, dev=dev_res, test=None, pool=None)

    tau = dev_res.tau

    # test への適用
    test_res: Optional[GateApplyResult]
    if test_df is not None and not test_df.empty:
        test_inputs = _extract_gate_inputs(test_df, col_conf)
        test_res = apply_tau(test_inputs, tau, gc)
    else:
        test_res = None

    # pool への適用（存在すれば）
    pool_res: Optional[GateApplyResult]
    if pool_df is not None and not pool_df.empty:
        pool_inputs = _extract_gate_inputs(pool_df, col_conf)
        pool_res = apply_tau(pool_inputs, tau, gc)
    else:
        pool_res = None

    return HitlOutputs(conf_key=conf_key, tau=tau, dev=dev_res, test=test_res, pool=pool_res)
