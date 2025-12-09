# /home/esakit25/work/tensaku/src/tensaku/confidence.py
# -*- coding: utf-8 -*-
"""
@module     tensaku.confidence
@role       確信度推定（MSP / entropy / energy / margin / MC-Dropout）の薄い共通層
@inputs     - logits: ndarray | torch.Tensor, shape (N, C)
           - probs:  ndarray | torch.Tensor, shape (N, C)
           - temperature (T): float
           - MC-Dropout系: model, dataloader
@outputs    - conf: ndarray, shape (N,)
@cli        tensaku confidence -c CFG.yaml
@api        create_estimator(name:str, **kw) -> Callable[..., np.ndarray]
@notes      registry への登録名: "msp", "entropy", "energy", "margin", "mc_dropout"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Iterable, Dict, Any, Union
from contextlib import nullcontext

import numpy as np
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)

# ---- Optional torch dependency -------------------------------------------------------------------

try:
    import torch
    from torch import nn, Tensor as TorchTensor
except Exception:  # pragma: no cover
    torch = None
    nn = None

    class _TorchTensorStub:
        pass

    TorchTensor = _TorchTensorStub  # type: ignore[misc,assignment]

# ---- Registry hook (soft dependency) -------------------------------------------------------------

try:
    from .registry import register  # type: ignore
except Exception:
    def register(name: str):
        def _decor(x):
            return x
        return _decor


# ---- 共通ユーティリティ -------------------------------------------------------------------------

ArrayLike = Union[npt.NDArray[np.floating], TorchTensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(x, TorchTensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax_np(logits: np.ndarray, T: float = 1.0, axis: int = -1) -> np.ndarray:
    z = logits / max(T, 1e-8)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=axis, keepdims=True), 1e-12, None)


def _to_probs(
    logits: Optional[ArrayLike],
    probs: Optional[ArrayLike],
    T: float,
) -> np.ndarray:
    if probs is not None:
        return _to_numpy(probs)
    if logits is None:
        raise ValueError("Either logits or probs must be provided.")
    return _softmax_np(_to_numpy(logits), T=T)


def _top2(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    top1_idx = np.argmax(probs, axis=1)
    top1 = probs[np.arange(probs.shape[0]), top1_idx]
    tmp = probs.copy()
    tmp[np.arange(tmp.shape[0]), top1_idx] = -np.inf
    top2 = np.max(tmp, axis=1)
    return top1, top2


def _entropy_np(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def _logsumexp_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(logits - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _sigmoid01(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-k * x))


# ---- ベースクラス --------------------------------------------------------------------------------

class ConfidenceEstimator:
    name: str = "base"

    def __call__(
        self,
        *,
        logits: Optional[ArrayLike] = None,
        probs: Optional[ArrayLike] = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        raise NotImplementedError


# ---- MSP -----------------------------------------------------------------------------------------

@register("msp")
class MSP(ConfidenceEstimator):
    name = "msp"

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        return p.max(axis=1)


# ---- Entropy -------------------------------------------------------------------------------------

@register("entropy")
class OneMinusEntropy(ConfidenceEstimator):
    name = "entropy"

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        H = _entropy_np(p)
        C = p.shape[1]
        H_max = math.log(max(C, 2))
        conf = 1.0 - (H / H_max)
        return np.clip(conf, 0.0, 1.0)


# ---- Energy --------------------------------------------------------------------------------------

@register("energy")
class Energy(ConfidenceEstimator):
    name = "energy"

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        if logits is None:
            p = _to_probs(None, probs, T=1.0)
            logits_np = np.log(np.clip(p, 1e-12, 1.0))
        else:
            logits_np = _to_numpy(logits) / max(temperature, 1e-8)

        lse = _logsumexp_np(logits_np)
        energy = -lse
        return _sigmoid01(-energy)


# ---- Margin --------------------------------------------------------------------------------------

@register("margin")
class Margin(ConfidenceEstimator):
    name = "margin"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        t1, t2 = _top2(p)
        margin = t1 - t2
        return _sigmoid01(margin, k=self.alpha)


# ---- MC-Dropout ----------------------------------------------------------------------------------

@dataclass
class MCDropoutConfig:
    n_passes: int = 20
    temperature: float = 1.0
    use_predictive_entropy: bool = True


def _enable_dropout(model: Any) -> None:
    if nn is None:
        return
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _predict_logits_once(
    model: Any,
    dataloader: Iterable,
    device: str = "auto",
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for MC-Dropout.")
    dev = (
        torch.device("cuda")
        if device == "auto" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(dev)
    model.eval()
    _enable_dropout(model)

    logits_list = []
    no_grad = torch.no_grad
    autocast = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext

    with no_grad():
        with autocast():
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    attention_mask = batch.get("attention_mask")
                    tk = {
                        k: v.to(dev, non_blocking=True)
                        for k, v in [
                            ("input_ids", input_ids),
                            ("attention_mask", attention_mask),
                        ]
                        if v is not None
                    }
                    out = model(**tk)
                    logits = out.logits if hasattr(out, "logits") else out[0]
                else:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                        input_ids = batch[0].to(dev, non_blocking=True)
                        attention_mask = (
                            batch[1].to(dev, non_blocking=True) if len(batch) >= 2 else None
                        )
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = out.logits if hasattr(out, "logits") else out[0]
                    else:
                        raise ValueError("Unsupported batch format for MC-Dropout.")
                logits_list.append(logits.detach().cpu())
    return torch.cat(logits_list, dim=0).numpy()


@register("mc_dropout")
class MCDropout(ConfidenceEstimator):
    name = "mc_dropout"

    def __init__(self, cfg: Optional[MCDropoutConfig] = None):
        self.cfg = cfg or MCDropoutConfig()

    def run_with_dataloader(
        self,
        model: Any,
        dataloader: Iterable,
        device: str = "auto",
    ) -> np.ndarray:
        logits_stack = []
        for _ in range(max(1, self.cfg.n_passes)):
            logits_np = _predict_logits_once(model, dataloader, device=device)
            logits_stack.append(logits_np / max(self.cfg.temperature, 1e-8))
        L = np.stack(logits_stack, axis=0)
        P = _softmax_np(L, T=1.0, axis=-1)
        P_mean = P.mean(axis=0)

        if self.cfg.use_predictive_entropy:
            H = _entropy_np(P_mean)
            C = P_mean.shape[1]
            conf = 1.0 - H / math.log(max(C, 2))
            return np.clip(conf, 0.0, 1.0)
        else:
            return P_mean.max(axis=1)

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        if self.cfg.use_predictive_entropy:
            H = _entropy_np(p)
            C = p.shape[1]
            conf = 1.0 - H / math.log(max(C, 2))
            return np.clip(conf, 0.0, 1.0)
        else:
            return p.max(axis=1)


# ---- 簡易ファクトリ -------------------------------------------------------------------------------

_ESTIMATORS: Dict[str, Callable[..., ConfidenceEstimator]] = {
    "msp": MSP,
    "entropy": OneMinusEntropy,
    "energy": Energy,
    "margin": Margin,
    "mc_dropout": MCDropout,
}


def create_estimator(name: str, **kwargs) -> ConfidenceEstimator:
    name_l = name.lower()
    if name_l not in _ESTIMATORS:
        raise KeyError(f"Unknown estimator: {name}")
    return _ESTIMATORS[name_l](**kwargs)


# ---- CLI entry point: tensaku confidence --------------------------------------------


def run(argv=None, cfg=None) -> int:
    """
    @cli   : tensaku confidence -c CFG.yaml [--out PATH]
    """
    import argparse
    import math as _math
    import os as _os
    import pandas as _pd
    import numpy as _np

    # コンソール出力も LOGGER 経由にするために basicConfig (CLI単体実行時用)
    # すでに他所で設定済みなら触らない
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )



    parser = argparse.ArgumentParser(prog="tensaku confidence", add_help=True)
    parser.add_argument(
        "--out",
        "--detail-out",
        dest="out_path",
        default=None,
        help="書き出す preds_detail.csv のパス",
    )
    args = parser.parse_args(argv or [])

    cfg = cfg or {}
    run_cfg = cfg.get("run", {}) or {}
    conf_cfg = cfg.get("confidence", {}) or {}
    out_dir = run_cfg.get("out_dir")
    
    if not out_dir:
        LOGGER.error("run.out_dir is not set in config")
        return 1
    out_dir = str(out_dir)

    raw_ests = conf_cfg.get("estimators", []) or []
    est_names: list[str] = []
    for est in raw_ests:
        if isinstance(est, dict):
            name = est.get("name")
        else:
            name = str(est)
        if not name:
            continue
        est_names.append(str(name).strip().lower())

    def _load_split(split_name: str) -> "_pd.DataFrame | None":
        path = _os.path.join(out_dir, f"{split_name}_preds.csv")
        if not _os.path.exists(path):
            LOGGER.warning(f"Missing preds CSV for split='{split_name}': {path}")
            return None
        df = _pd.read_csv(path)
        if "id" not in df.columns:
            raise KeyError(f"missing 'id' column in {path}")
        df.insert(0, "split", split_name)

        if "y_true" not in df.columns:
            df["y_true"] = _np.nan

        if "conf_msp" not in df.columns:
            raise KeyError(f"missing 'conf_msp' column in {path}")

        return df

    dfs = []
    for split_name in ("dev", "pool", "test"):
        df = _load_split(split_name)
        if df is not None:
            dfs.append(df)

    if not dfs:
        LOGGER.error("No preds CSV found in out_dir")
        return 1

    df_all = _pd.concat(dfs, ignore_index=True)

    est_to_col = {
        "msp": "conf_msp",
        "entropy": "conf_entropy",
        "energy": "conf_energy",
        "margin": "conf_margin",
        "prob_margin": "conf_prob_margin",
        "mc_dropout": "conf_mcdo",
        "trust": "conf_trust",
        "conf_trust": "conf_trust",
    }

    need_logits = {"msp", "entropy", "energy", "margin", "prob_margin", "mc_dropout"}
    logits_cache: dict[str, "_np.ndarray | None"] = {}

    def _load_logits(split_name: str) -> "_np.ndarray | None":
        if split_name in logits_cache:
            return logits_cache[split_name]
        path = _os.path.join(out_dir, f"{split_name}_logits.npy")
        if not _os.path.exists(path):
            LOGGER.warning(f"Missing logits for split='{split_name}': {path}")
            logits_cache[split_name] = None
            return None
        arr = _np.load(path)
        logits_cache[split_name] = arr
        return arr

    splits_present = list(dict.fromkeys(df_all["split"].tolist()))

    for est_name in est_names:
        col = est_to_col.get(est_name)
        if not col:
            LOGGER.warning(f"Unknown estimator name in config: {est_name}")
            continue

        if col in df_all.columns:
            continue

        if est_name in {"trust", "conf_trust"}:
            if "conf_trust" not in df_all.columns:
                LOGGER.warning("'conf_trust' column is not present; skipping trust estimator")
            continue

        if est_name == "msp":
            continue

        if est_name == "mc_dropout":
            LOGGER.warning("'mc_dropout' estimator is not yet supported in CLI; skipping.")
            continue

        if est_name in need_logits:
            try:
                est = create_estimator(est_name)
            except Exception as e:
                LOGGER.warning(f"Failed to create estimator '{est_name}': {e}; skip")
                continue

            df_all[col] = _np.nan

            for split_name in splits_present:
                mask = df_all["split"] == split_name
                if not mask.any():
                    continue
                logits = _load_logits(split_name)
                if logits is None:
                    continue
                n_rows = int(mask.sum())
                if logits.shape[0] != n_rows:
                    LOGGER.warning(
                        f"Logits length mismatch for split='{split_name}' "
                        f"(csv_rows={n_rows}, logits_rows={logits.shape[0]}); skipping."
                    )
                    continue
                try:
                    conf = est(logits=logits)
                except Exception as e:
                    LOGGER.warning(f"Estimator '{est_name}' failed on split='{split_name}': {e}; skip")
                    continue
                
                conf = _np.asarray(conf).reshape(-1)
                df_all.loc[mask, col] = conf

    base_cols = [
        "split",
        "id",
        "y_true",
        "y_pred",
        "conf_msp",
        "conf_msp_temp",
        "conf_trust",
        "conf_entropy",
        "conf_margin",
        "conf_prob_margin",
        "conf_energy",
        "conf_mcdo",
    ]
    other_cols = [c for c in df_all.columns if c not in base_cols]
    ordered_cols = [c for c in base_cols if c in df_all.columns] + other_cols
    df_all = df_all[ordered_cols]

    out_path = args.out_path or _os.path.join(out_dir, "preds_detail.csv")
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    df_all.to_csv(out_path, index=False)
    LOGGER.info(f"Wrote preds_detail.csv -> {out_path} (n={len(df_all)})")

    dev_mask = df_all["split"] == "dev"
    if dev_mask.any():
        dev = df_all[dev_mask]
        try:
            y_true = dev["y_true"].to_numpy(dtype=float)
            y_pred = dev["y_pred"].to_numpy(dtype=float)
            err = y_pred - y_true
            rmse = float(_math.sqrt(_np.mean(err ** 2)))
            cse = float(_np.mean(_np.abs(err) >= 2.0))
            LOGGER.info(f"dev summary: RMSE={rmse:.4f}, CSE(|err|>=2)={cse:.4f}")
        except Exception as e:
            LOGGER.warning(f"Failed to compute dev summary metrics: {e}")

    return 0