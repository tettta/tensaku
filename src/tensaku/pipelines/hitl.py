# /home/esakit25/work/tensaku/src/tensaku/pipelines/hitl.py
# -*- coding: utf-8 -*-
"""tensaku.pipelines.hitl

HITL pipeline primitive: given a detail DataFrame (preds + conf), find tau on dev and
apply it to other splits (test/pool).

This module is still a *lower* module, but is intentionally lightweight and
in-memory oriented:
- Inputs/outputs are DataFrames and small numpy arrays.
- Saving is handled by upper orchestrators (tasks/pipelines) via layout/fs_core.

Strict notes
- Required columns must exist; missing columns raise immediately.
- No cfg mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from tensaku.gate import GateApplyResult, GateDevResult, GateInputs, apply_tau, create_gate_config, find_tau_for_constraint


@dataclass
class HitlOutputs:
    dev: GateDevResult
    test: Optional[GateApplyResult]
    pool: Optional[GateApplyResult]

    def to_summary_dict(self, *, prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        p = prefix or ""
        out[p + "tau"] = self.dev.tau
        out[p + "dev.coverage"] = self.dev.coverage
        out[p + "dev.cse"] = self.dev.cse
        out[p + "dev.rmse"] = self.dev.rmse

        if self.test is not None:
            out[p + "test.coverage"] = self.test.coverage
            out[p + "test.cse"] = self.test.cse
            out[p + "test.rmse"] = self.test.rmse
        if self.pool is not None:
            out[p + "pool.coverage"] = self.pool.coverage
            out[p + "pool.cse"] = self.pool.cse
            out[p + "pool.rmse"] = self.pool.rmse
        return out

    def assign_df(self) -> pd.DataFrame:
        """Return pool assignment as a DataFrame with columns [is_auto, is_human]."""
        if self.pool is None:
            return pd.DataFrame(columns=["is_auto", "is_human"])
        return pd.DataFrame(
            {
                "is_auto": self.pool.mask_auto.astype(bool),
                "is_human": self.pool.mask_human.astype(bool),
            }
        )


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"hitl: missing required columns: {missing}")
    

def build_gate_assign_df(*, pool_ids: List[Any], conf_key: str, hitl_out: HitlOutputs) -> pd.DataFrame:
    """
    Build long-format gate assignment DF for pool.
    Columns: id, split, conf_key, tau, is_auto, is_human

    Important: normalize tau None -> np.nan (float) to avoid pandas concat FutureWarning.
    """
    assign = hitl_out.assign_df()
    if len(assign) != len(pool_ids):
        raise RuntimeError(f"Gate assign length mismatch: assign={len(assign)} pool_ids={len(pool_ids)}")

    tau = hitl_out.dev.tau
    tau_val = float(tau) if tau is not None else float("nan")  # <-- 핵: None を nan に

    out = assign.copy()
    out.insert(0, "id", pool_ids)
    out.insert(1, "split", "pool")
    out.insert(2, "conf_key", conf_key)
    out.insert(3, "tau", tau_val)

    # dtype 安定化（将来のpandas差分にも強くする）
    out["tau"] = out["tau"].astype(float)
    out["is_auto"] = out["is_auto"].astype(bool)
    out["is_human"] = out["is_human"].astype(bool)
    return out


def create_empty_gate_assign_df() -> pd.DataFrame:
    """Gate割り当て結果の空スキーマを返す。"""
    return pd.DataFrame(columns=["id", "split", "conf_key", "tau", "is_auto", "is_human"])  


def run_hitl_from_detail_df(*, df: pd.DataFrame, gate_cfg: Mapping[str, Any], conf_column_name: str) -> HitlOutputs:
    """Run HITL (dev->tau, apply to test/pool).

    Expected df columns:
      - split: str in {labeled, dev, test, pool}
      - y_true: int (dev/test) (pool may be NaN)
      - y_pred: int
      - conf_*: float confidence column

    gate_cfg: Mapping with required GateConfig keys:
      - eps_cse, cse_abs_err, higher_is_better, steps
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    _require_cols(df, ["split", "y_pred", conf_column_name])
    # dev/test require y_true; pool may be NaN but column must exist for gate functions
    _require_cols(df, ["y_true"])

    cfg_gate = create_gate_config(gate_cfg)

    # ----------------
    # DEV: search tau
    # ----------------
    df_dev = df[df["split"] == "dev"]
    if len(df_dev) == 0:
        raise ValueError("hitl requires at least one dev sample to search tau")

    y_true_dev = df_dev["y_true"].to_numpy()
    y_pred_dev = df_dev["y_pred"].to_numpy()
    conf_dev = df_dev[conf_column_name].to_numpy()

    dev_in = GateInputs(y_true=y_true_dev, y_pred=y_pred_dev, conf=conf_dev, ids=None)
    dev_out = find_tau_for_constraint(dev=dev_in, cfg=cfg_gate)

    # ----------------
    # TEST: apply tau
    # ----------------
    df_test = df[df["split"] == "test"]
    test_out: Optional[GateApplyResult] = None
    if len(df_test) > 0:
        y_true_test = df_test["y_true"].to_numpy()
        y_pred_test = df_test["y_pred"].to_numpy()
        conf_test = df_test[conf_column_name].to_numpy()
        te_in = GateInputs(y_true=y_true_test, y_pred=y_pred_test, conf=conf_test, ids=None)
        test_out = apply_tau(inputs=te_in, cfg=cfg_gate, tau=dev_out.tau)

    # ----------------
    # POOL: apply tau (no gold; only masks are meaningful)
    # ----------------
    df_pool = df[df["split"] == "pool"]
    pool_out: Optional[GateApplyResult] = None
    if len(df_pool) > 0:
        conf_pool = df_pool[conf_column_name].to_numpy()
        # dummy arrays for shape compatibility
        y_dummy = np.zeros_like(conf_pool, dtype=int)
        po_in = GateInputs(y_true=y_dummy, y_pred=y_dummy, conf=conf_pool, ids=None)
        pool_out = apply_tau(inputs=po_in, cfg=cfg_gate, tau=dev_out.tau)
        pool_out.cse = float("nan")
        pool_out.rmse = float("nan")

    return HitlOutputs(dev=dev_out, test=test_out, pool=pool_out)
