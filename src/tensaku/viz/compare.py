# /home/esakit25/work/tensaku/src/tensaku/viz/compare.py
# -*- coding: utf-8 -*-
"""tensaku.viz.compare

Load and aggregate learning curves across experiments using the experiment index.

Index location:
  outputs/<qid>/_index/experiments.jsonl

Curve source:
  Usually exp_dir/metrics/al_learning_curve.csv, but this can be configured.

This module does NOT scan directories heuristically by default; it relies on the index
to stay robust when output layouts evolve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

from tensaku.viz import base


@dataclass(frozen=True)
class CurveBundle:
    df: pd.DataFrame
    exp_uid: str
    meta: Dict[str, Any]


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"index not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"failed to parse jsonl at {path}:{ln} ({e})") from e
    return rows


def index_path(project_root: Path, qid: str) -> Path:
    return (project_root / "outputs" / qid / "_index" / "experiments.jsonl").resolve()


def load_index(project_root: Path, qid: str) -> List[dict]:
    return _read_jsonl(index_path(project_root, qid))


def _match_opt(value: Optional[Sequence], x) -> bool:
    if value is None:
        return True
    return x in set(value)


def _get_curve_rel_path(rec: dict, curve_key: str) -> str:
    # If caller passed a literal relative path, use it.
    if "/" in curve_key or curve_key.endswith(".csv"):
        return curve_key
    # Else treat as a key in rec["paths"].
    paths = rec.get("paths") or {}
    if curve_key in paths:
        return str(paths[curve_key])
    # Common alias
    if curve_key == "al_learning_curve" and "al_learning_curve" in paths:
        return str(paths["al_learning_curve"])
    raise KeyError(f"curve_key not found in index record paths: key={curve_key} available={list(paths.keys())}")


def load_learning_curves(
    *,
    project_root: Path,
    qid: str,
    status: Sequence[str] = ("success", "backfilled"),
    tag: Optional[Sequence[str]] = None,
    sampler: Optional[Sequence[str]] = None,
    seed: Optional[Sequence[int]] = None,
    curve_key: str = "al_learning_curve",
    require_files: bool = True,
) -> List[CurveBundle]:
    recs = load_index(project_root, qid)
    out: List[CurveBundle] = []

    for rec in recs:
        st = rec.get("status")
        if st not in set(status):
            continue

        run = rec.get("run") or {}
        if not _match_opt(tag, run.get("tag")):
            continue
        if not _match_opt(sampler, run.get("sampler")):
            continue
        if seed is not None:
            try:
                sd = int(run.get("seed"))
            except Exception:
                continue
            if sd not in set(seed):
                continue

        exp_dir = Path(str(run.get("exp_dir"))).resolve()
        if not exp_dir.exists():
            if require_files:
                raise FileNotFoundError(f"exp_dir not found: {exp_dir}")
            continue

        try:
            rel = _get_curve_rel_path(rec, curve_key)  # 修正: 引数名変更
        except Exception as e:
            if require_files:
                raise
            continue

        curve_path = exp_dir / rel
        if not curve_path.exists():
            if require_files:
                raise FileNotFoundError(f"curve csv not found: {curve_path}")
            continue

        df = base.read_csv(curve_path, ctx="learning_curve")
        # Basic required columns for plotting
        base.require_columns(df, ["round"], ctx="learning_curve")
        if "n_labeled" not in df.columns:
            # Some old runs may have only coverage/n_pool; allow but warn in caller.
            pass

        exp_uid = str(rec.get("uid") or rec.get("exp_uid") or "")
        meta = {
            "meta_tag": run.get("tag"),
            "meta_sampler": run.get("sampler"),
            "meta_seed": run.get("seed"),
            "meta_sampler_conf_key": (rec.get("meta") or {}).get("al", {}).get("sampler", {}).get("conf_key"),
            "meta_budget": (rec.get("meta") or {}).get("al", {}).get("budget"),
            "meta_rounds": (rec.get("meta") or {}).get("al", {}).get("rounds"),
            "meta_n_train": (rec.get("meta") or {}).get("split", {}).get("n_train"),
            "meta_eps_cse": (rec.get("meta") or {}).get("gate", {}).get("eps_cse"),
            "meta_cse_abs_err": (rec.get("meta") or {}).get("gate", {}).get("cse_abs_err"),
            "exp_dir": str(exp_dir),
        }

        out.append(CurveBundle(df=df, exp_uid=exp_uid, meta=meta))

    return out


def concat_curves(bundles: Sequence[CurveBundle]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for b in bundles:
        df = b.df.copy()
        df["exp_uid"] = b.exp_uid
        for k, v in b.meta.items():
            df[k] = v
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def aggregate_curves(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_cols: Sequence[str],
    group_keys: Sequence[str],
) -> pd.DataFrame:
    """Aggregate curves into mean/std/count for each plot_label and x value.

    Output columns:
      - group keys...
      - x_col
      - for each y: y__mean, y__std, y__count
    """
    base.require_columns(df, list(group_keys) + [x_col] + list(y_cols), ctx="aggregate_curves")

    gcols = list(group_keys) + [x_col]
    # Avoid including group keys in aggregation targets (prevents reset_index duplicates)
    agg_targets = [c for c in y_cols if c not in set(gcols)]
    if not agg_targets:
        raise ValueError("no y_cols left to aggregate after excluding group keys")

    agg = (
        df.groupby(gcols, dropna=False)[list(agg_targets)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten MultiIndex columns: (y, stat) -> y__stat
    new_cols: List[str] = []
    for c in agg.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                new_cols.append(c[0])
            else:
                new_cols.append(f"{c[0]}__{c[1]}")
        else:
            new_cols.append(c)

    # Ensure unique
    if len(new_cols) != len(set(new_cols)):
        # This should not happen, but when it does, it is better to crash with context.
        raise RuntimeError(f"aggregate produced duplicate columns: {new_cols}")

    agg.columns = new_cols
    return agg


SUMMARY_DIRNAME = "summary"


def get_summary_dir(project_root: Path, qid: str) -> Path:
    """Directory for visualization summaries.

    We standardize the directory name to "summary" because outputs include
    not only plots but also tables and metadata.

    Path:
      outputs/<qid>/summary
    """
    return base.ensure_dir((project_root / "outputs" / qid / SUMMARY_DIRNAME).resolve())


def get_summary_plots_dir(project_root: Path, qid: str) -> Path:
    """Backward-compatible alias.

    Older code used "summary_plots"; keep the API name, but write to the
    standardized directory "summary".
    """
    return get_summary_dir(project_root, qid)


def get_by_sampler_plots_dir(project_root: Path, qid: str, sampler: str) -> Path:
    return base.ensure_dir((get_summary_dir(project_root, qid) / "by_sampler" / str(sampler)).resolve())