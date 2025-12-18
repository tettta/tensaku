# /home/esakit25/work/tensaku/src/tensaku/contracts.py
# -*- coding: utf-8 -*-
"""tensaku.contracts

Pipelines/tasks shared *contracts* (schemas + validators).

Design principles (MUST)
- No fallback: contract violations raise ValueError immediately.
- Pipelines/tasks depend on contracts and config keys, not lower-level implementations.
- Contracts specify *what* must exist, not *how* it is produced.

Notes
- We intentionally keep this module small and stable.
- Do NOT import heavy ML libs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


class ContractError(ValueError):
    """Raised when an artifact violates a contract."""


ALLOWED_SPLITS: Tuple[str, ...] = ("labeled", "dev", "test", "pool")


def _as_set(xs: Iterable[Any]) -> Set[Any]:
    return set(xs)


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, ctx: str) -> None:
    missing = _as_set(cols) - _as_set(df.columns)
    if missing:
        raise ContractError(f"[{ctx}] missing required columns: {sorted(missing)}. columns={list(df.columns)}")


def validate_preds_detail(
    df: pd.DataFrame,
    *,
    required_conf_cols: Sequence[str] = (),
    require_y_true_splits: Sequence[str] = ("labeled", "dev", "test"),
    allow_missing_y_true_splits: Sequence[str] = ("pool",),
    ctx: str = "preds_detail",
) -> None:
    """Validate `preds_detail` DataFrame.

    Minimal contract:
    - columns: id, split, y_pred are required
    - y_true is required only for `require_y_true_splits`
    - conf columns required by caller must exist (e.g., conf_msp, trust, ...)
    """
    if not isinstance(df, pd.DataFrame):
        raise ContractError(f"[{ctx}] must be pandas.DataFrame, got {type(df)}")

    require_columns(df, ("id", "split", "y_pred"), ctx=ctx)

    # split validation
    bad = sorted(_as_set(df["split"].astype(str).tolist()) - _as_set(ALLOWED_SPLITS))
    if bad:
        raise ContractError(f"[{ctx}] invalid split values: {bad}. allowed={list(ALLOWED_SPLITS)}")

    # id validation
    if df["id"].isna().any():
        raise ContractError(f"[{ctx}] id contains NaN")
    if df["id"].astype(str).str.len().eq(0).any():
        raise ContractError(f"[{ctx}] id contains empty string")

    # y_true requirement by split
    splits_need_true = set(require_y_true_splits)
    splits_allow_missing = set(allow_missing_y_true_splits)
    if not splits_need_true.issubset(set(ALLOWED_SPLITS)):
        raise ContractError(f"[{ctx}] require_y_true_splits has invalid split(s): {sorted(splits_need_true)}")
    if not splits_allow_missing.issubset(set(ALLOWED_SPLITS)):
        raise ContractError(f"[{ctx}] allow_missing_y_true_splits has invalid split(s): {sorted(splits_allow_missing)}")

    if "y_true" not in df.columns:
        # If any rows require y_true, this is a hard error.
        need = df["split"].astype(str).isin(list(splits_need_true)).any()
        if need:
            raise ContractError(f"[{ctx}] missing required column 'y_true' for splits={sorted(splits_need_true)}")
    else:
        # rows requiring y_true must be non-null
        mask_need = df["split"].astype(str).isin(list(splits_need_true))
        if mask_need.any() and df.loc[mask_need, "y_true"].isna().any():
            raise ContractError(f"[{ctx}] y_true has NaN in required splits={sorted(splits_need_true)}")

    # required confidence columns
    for c in required_conf_cols:
        if c not in df.columns:
            raise ContractError(f"[{ctx}] missing required confidence column: {c}")
        v = df[c].to_numpy()
        if np.asarray(v).size != len(df):
            raise ContractError(f"[{ctx}] conf column '{c}' length mismatch")

    # y_pred type check (soft check: allow numpy ints)
    # We do not coerce here to keep strictness: producer must output correct dtype.
    # But we validate it's integer-like.
    yp = df["y_pred"].to_numpy()
    if not np.issubdtype(np.asarray(yp).dtype, np.integer):
        raise ContractError(f"[{ctx}] y_pred must be integer dtype, got {np.asarray(yp).dtype}")



def validate_gate_assign(df: pd.DataFrame, *, ctx: str = "gate_assign") -> None:
    """
    Gate assignment DataFrame (pool split) の整合性を検証する。
    
    Required columns:
      - id, split, is_auto, is_human
    Optional columns:
      - conf_key: 複数指標の場合の識別子
      - tau: 適用された閾値
    
    Rules:
      1. 必須カラムが存在すること
      2. is_auto と is_human は排他的であること (True/True や False/False は不可)
      3. tau は (conf_key がある場合はそのキーごとに) 一意であること
    """
    # 1. 必須カラムチェック
    required = ["id", "split", "is_auto", "is_human"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(f"[{ctx}] missing columns: {missing}")

    # 2. 排他性チェック (is_auto vs is_human)
    # bool型変換してから比較
    auto = df["is_auto"].astype(bool)
    human = df["is_human"].astype(bool)
    
    # auto == human (両方True or 両方False) な行があってはならない
    invalid_mask = (auto == human)
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        raise ContractError(f"[{ctx}] is_auto and is_human must be mutually exclusive. Found {n_invalid} invalid rows.")

    # 3. Tau 一意性チェック (ここを修正)
    if "tau" in df.columns:
        if "conf_key" in df.columns:
            # conf_key がある場合: キーごとに tau が一意かチェック
            for key, group in df.groupby("conf_key"):
                # dropna() しているのは、NaN (未定義) は除外して「値が入っているなら一意」とするため
                unique_taus = group["tau"].dropna().unique()
                if len(unique_taus) > 1:
                    raise ContractError(
                        f"[{ctx}] tau must be unique per conf_key. "
                        f"Key '{key}' has multiple taus: {unique_taus}"
                    )
        else:
            # conf_key がない場合 (Single mode): 全体で tau は1つのみ
            unique_taus = df["tau"].dropna().unique()
            if len(unique_taus) > 1:
                raise ContractError(f"[{ctx}] tau must be a single value for the artifact, got {unique_taus}")

    # (必要であればIDの重複チェックなどをここに追加)

def validate_pool_scores(
    pool_scores: Mapping[str, np.ndarray],
    *,
    required_keys: Sequence[str],
    pool_size: int,
    ctx: str = "pool_scores",
) -> None:
    """Validate pool_scores mapping (id -> score), stored as 1D arrays aligned with pool_ids."""
    if not isinstance(pool_scores, Mapping):
        raise ContractError(f"[{ctx}] must be Mapping[str, np.ndarray], got {type(pool_scores)}")

    for k in required_keys:
        if k not in pool_scores:
            raise ContractError(f"[{ctx}] missing required score key: {k}. available={list(pool_scores.keys())}")
        v = np.asarray(pool_scores[k])
        if v.ndim != 1 or v.shape[0] != pool_size:
            raise ContractError(f"[{ctx}] score '{k}' must be 1d length={pool_size}, got shape={v.shape}")
