# /home/esakit25/work/tensaku/src/tensaku/viz/tau_sweep.py
# -*- coding: utf-8 -*-
"""
tensaku.viz.tau_sweep

@role
  - preds_detail.csv（dev/test を含む）から、HITL の tau 探索証跡（curve）を作る。
  - 最低限 CSV を出力し、後から論文図/表にできる形にする。

@strict
  - cse_abs_err と eps_cse は、(A) run_meta.json から読む か (B) CLI 引数で明示。
  - conf の方向（higher_is_better）は、既知の尺度のみ自動判定。
    それ以外は明示指定が必要（黙って推定しない）。

@outputs
  - tau_sweep_{split}_{conf}.csv:
      columns: [split, conf_key, k, coverage, tau, cse, rmse, meets_eps]
  - tau_selected_{split}_{conf}.json:
      selected_tau, selected_coverage, selected_k
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def _coerce_float_1d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    return a


def _coerce_int_1d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=int).reshape(-1)
    return a


def _conf_higher_is_better(conf_key: str, direction_map: Optional[Dict[str, bool]] = None) -> bool:
    """Determine if higher confidence score is better (monotonicity).
    
    Order of precedence:
    1. direction_map (from explicit metadata/config)
    2. Heuristic rules (Strictly defined for known keys only)
    3. ValueError
    """
    if direction_map is not None:
        if conf_key in direction_map:
            return direction_map[conf_key]
    
    k = conf_key.lower()
    if "entropy" in k: return True
    if "energy" in k: return True
    if "loss" in k: return True
    if "std" in k: return True
    
    if "msp" in k: return True
    if "margin" in k: return True
    if "trust" in k: return True
    if "prob" in k and "margin" in k: return True
    
    raise ValueError(
        f"Cannot infer direction for conf_key={conf_key!r}. "
        "Provide explicit direction via run_meta.json or CLI args."
    )


def _cse_rate(abs_err: np.ndarray, cse_abs_err: int) -> np.ndarray:
    return (abs_err >= int(cse_abs_err)).astype(float)


def _rmse_from_cumsums(cum_sqerr: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.sqrt(cum_sqerr / np.maximum(k, 1))


@dataclass(frozen=True)
class TauSweepConfig:
    eps_cse: float
    cse_abs_err: int
    higher_is_better: bool


@dataclass(frozen=True)
class TauSelected:
    conf_key: str
    split: str
    eps_cse: float
    cse_abs_err: int
    higher_is_better: bool
    selected_tau: float
    selected_coverage: float
    selected_k: int


def compute_tau_sweep_curve(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    cfg: TauSweepConfig,
) -> Tuple[pd.DataFrame, TauSelected]:
    y_true = _coerce_int_1d(y_true)
    y_pred = _coerce_int_1d(y_pred)
    conf = _coerce_float_1d(conf)

    if not (y_true.shape == y_pred.shape == conf.shape):
        raise ValueError(f"Length mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, conf={conf.shape}")

    n = int(len(conf))
    if n == 0:
        raise ValueError("Empty arrays are not allowed")

    # order
    order = np.argsort(conf)
    if cfg.higher_is_better:
        order = order[::-1]

    conf_s = conf[order]
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]

    abs_err = np.abs(y_pred_s - y_true_s).astype(int)
    sq_err = (y_pred_s - y_true_s).astype(float) ** 2

    # cumulative
    k = np.arange(1, n + 1, dtype=int)
    cum_cse = np.cumsum(_cse_rate(abs_err, cfg.cse_abs_err))
    cse = cum_cse / k
    cum_sq = np.cumsum(sq_err)
    rmse = _rmse_from_cumsums(cum_sq, k)
    coverage = k / float(n)

    # tau threshold at k: choose conf value at k-th item
    tau = conf_s  # per position (k aligns with tau[k-1])

    meets = cse <= float(cfg.eps_cse)

    df = pd.DataFrame(
        {
            "k": k,
            "coverage": coverage,
            "tau": tau,
            "cse": cse,
            "rmse": rmse,
            "meets_eps": meets.astype(int),
        }
    )

    # select best coverage under eps (last True in meets)
    if not np.any(meets):
        # Strict: no feasible tau
        sel_k = 0
        sel_cov = 0.0
        sel_tau = float("nan")
    else:
        sel_k = int(np.where(meets)[0][-1] + 1)
        sel_cov = float(sel_k) / float(n)
        sel_tau = float(tau[sel_k - 1])

    selected = TauSelected(
        conf_key="",
        split="",
        eps_cse=float(cfg.eps_cse),
        cse_abs_err=int(cfg.cse_abs_err),
        higher_is_better=bool(cfg.higher_is_better),
        selected_tau=sel_tau,
        selected_coverage=sel_cov,
        selected_k=sel_k,
    )
    return df, selected


def load_gate_params_from_run_meta(run_meta_path: Path) -> Dict[str, Any]:
    meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    # flexible: allow either {"gate": {...}} or {"hitl": {"gate": {...}}}
    if "gate" in meta and isinstance(meta["gate"], dict):
        return dict(meta["gate"])
    hitl = meta.get("hitl")
    if isinstance(hitl, dict) and "gate" in hitl and isinstance(hitl["gate"], dict):
        return dict(hitl["gate"])
    return {}


def export_tau_sweep(
    *,
    preds_detail_csv: Path,
    out_dir: Path,
    conf_keys: List[str],
    cse_abs_err: Optional[int],
    eps_cse: Optional[float],
    run_meta_json: Optional[Path],
    direction: Optional[Dict[str, bool]],
) -> List[Path]:
    if not preds_detail_csv.exists():
        raise FileNotFoundError(f"preds_detail_csv not found: {preds_detail_csv}")
    # Split outputs to keep directory readable.
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    meta_dir = out_dir / "meta"
    tables_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_detail_csv)
    required_cols = {"y_true", "y_pred"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"preds_detail.csv must contain {sorted(required_cols)} (Strict)")

    if "split" not in df.columns:
        # Strict: without split, we cannot separate dev/test reliably
        raise KeyError("preds_detail.csv must contain 'split' column (Strict)")

    gate_params: Dict[str, Any] = {}
    if run_meta_json is not None:
        if not run_meta_json.exists():
            raise FileNotFoundError(f"run_meta_json not found: {run_meta_json}")
        gate_params = load_gate_params_from_run_meta(run_meta_json)

    eff_cse_abs_err = cse_abs_err if cse_abs_err is not None else gate_params.get("cse_abs_err")
    eff_eps_cse = eps_cse if eps_cse is not None else gate_params.get("eps_cse")

    if eff_cse_abs_err is None:
        raise ValueError("cse_abs_err is required (provide --cse-abs-err or ensure run_meta has gate.cse_abs_err)")
    if eff_eps_cse is None:
        raise ValueError("eps_cse is required (provide --eps-cse or ensure run_meta has gate.eps_cse)")

    paths: List[Path] = []
    for split_name in ["dev", "test"]:
        df_s = df[df["split"] == split_name]
        if len(df_s) == 0:
            continue
        y_true = df_s["y_true"].to_numpy()
        y_pred = df_s["y_pred"].to_numpy()

        for ck in conf_keys:
            if ck not in df_s.columns:
                raise KeyError(f"confidence column not found: {ck}. available={list(df.columns)}")
            hib = _conf_higher_is_better(ck, direction_map=direction)
            conf = pd.to_numeric(df_s[ck], errors="raise").to_numpy(dtype=float)

            cfg = TauSweepConfig(eps_cse=float(eff_eps_cse), cse_abs_err=int(eff_cse_abs_err), higher_is_better=hib)
            curve, selected = compute_tau_sweep_curve(y_true=y_true, y_pred=y_pred, conf=conf, cfg=cfg)
            curve.insert(0, "conf_key", ck)
            curve.insert(0, "split", split_name)

            selected = TauSelected(
                conf_key=ck,
                split=split_name,
                eps_cse=selected.eps_cse,
                cse_abs_err=selected.cse_abs_err,
                higher_is_better=selected.higher_is_better,
                selected_tau=selected.selected_tau,
                selected_coverage=selected.selected_coverage,
                selected_k=selected.selected_k,
            )

            out_csv = tables_dir / f"tau_sweep_{split_name}_{ck}.csv"
            curve.to_csv(out_csv, index=False, encoding="utf-8")
            paths.append(out_csv)

            out_json = meta_dir / f"tau_selected_{split_name}_{ck}.json"
            out_json.write_text(json.dumps(asdict(selected), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            paths.append(out_json)

    return paths
