# /home/esakit25/work/tensaku/src/tensaku/confidence.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.confidence
@role  : 信頼度スコア計算ロジック (Strict Mode)

Design notes
- Task側は estimator.apply(df, raw) で呼び出す。
- raw は infer_core(return_raw_outputs=True) の返り値を想定:
    raw[split] = {"logits":..., "embs":..., "labels":..., "y_pred":..., "ids":[...]}
- 返り値は df と同じ行数の 1D スコア（np.ndarray）を基本とする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from tensaku.registry import register
from tensaku.utils.strict_cfg import ConfigError


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if torch.is_tensor(x):
        return x
    return torch.tensor(x)


def _require_raw_split(raw: Mapping[str, Any], split: str, key: str) -> Any:
    if split not in raw:
        raise ConfigError(f"raw missing split '{split}'")
    d = raw[split]
    if not isinstance(d, Mapping):
        raise ConfigError(f"raw['{split}'] must be a mapping, got {type(d)}")
    if key not in d:
        raise ConfigError(f"raw['{split}'] missing key '{key}'")
    return d[key]


def _align_scores_by_id(
    df_ids: np.ndarray,
    raw_ids: Any,
    scores: np.ndarray,
    *,
    where: str,
) -> np.ndarray:
    raw_ids_arr = np.asarray(raw_ids)
    if raw_ids_arr.ndim != 1:
        raise ConfigError(f"{where}: raw ids must be 1D")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ConfigError(f"{where}: scores must be 1D")
    if len(raw_ids_arr) != len(scores):
        raise ConfigError(f"{where}: len(raw_ids) != len(scores) ({len(raw_ids_arr)} != {len(scores)})")

    # Build mapping and align strictly
    m: Dict[Any, float] = {}
    for i, rid in enumerate(raw_ids_arr.tolist()):
        m[rid] = float(scores[i])

    out = np.empty(len(df_ids), dtype=float)
    for i, did in enumerate(df_ids.tolist()):
        if did not in m:
            raise ConfigError(f"{where}: id '{did}' not found in raw ids")
        out[i] = m[did]
    return out


@dataclass(frozen=True)
class _EstimatorBase:
    cfg: Mapping[str, Any]

    def apply(self, *, df: Any, raw: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError


class _LogitsEstimator(_EstimatorBase):
    """logits -> scalar confidence"""

    def _score_from_logits(self, logits: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, *, df: Any, raw: Mapping[str, Any]) -> np.ndarray:
        if "split" not in df.columns or "id" not in df.columns:
            raise ConfigError("df must contain 'split' and 'id'")
        out = np.empty(len(df), dtype=float)

        # per split align by id for safety
        for split_name, df_part in df.groupby("split", sort=False):
            logits = np.asarray(_require_raw_split(raw, split_name, "logits"))
            raw_ids = _require_raw_split(raw, split_name, "ids")
            scores = self._score_from_logits(logits)
            aligned = _align_scores_by_id(
                df_part["id"].to_numpy(),
                raw_ids,
                scores,
                where=f"{self.__class__.__name__}({split_name})",
            )
            out[df_part.index.to_numpy()] = aligned
        return out


@register("msp", override=True)
class MSPEstimator(_LogitsEstimator):
    """Maximum Softmax Probability"""

    def _score_from_logits(self, logits: np.ndarray) -> np.ndarray:
        temperature = float(self.cfg.get("temperature", 1.0))
        t_logits = _to_tensor(logits) / temperature
        probs = F.softmax(t_logits, dim=-1)
        conf, _ = torch.max(probs, dim=-1)
        return conf.detach().cpu().numpy()


@register("entropy", override=True)
class EntropyEstimator(_LogitsEstimator):
    """Negative entropy (higher is better)"""

    def _score_from_logits(self, logits: np.ndarray) -> np.ndarray:
        temperature = float(self.cfg.get("temperature", 1.0))
        t_logits = _to_tensor(logits) / temperature
        probs = F.softmax(t_logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        return (-ent).detach().cpu().numpy()


@register("margin", override=True)
class MarginEstimator(_LogitsEstimator):
    """Top1 - Top2 probability margin"""

    def _score_from_logits(self, logits: np.ndarray) -> np.ndarray:
        temperature = float(self.cfg.get("temperature", 1.0))
        t_logits = _to_tensor(logits) / temperature
        probs = F.softmax(t_logits, dim=-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        margin = top2[:, 0] - top2[:, 1]
        return margin.detach().cpu().numpy()


@register("energy", override=True)
class EnergyEstimator(_LogitsEstimator):
    """Energy score (higher is better)"""

    def _score_from_logits(self, logits: np.ndarray) -> np.ndarray:
        temperature = float(self.cfg.get("temperature", 1.0))
        t_logits = _to_tensor(logits) / temperature
        # energy = T * logsumexp(logits/T)
        energy = temperature * torch.logsumexp(t_logits, dim=-1)
        return energy.detach().cpu().numpy()


@register("trust", override=True)
class TrustEstimator(_EstimatorBase):
    """kNN Trust Score based on embeddings and predicted class.

    Contract:
    - Reference split is fixed to 'labeled' (Strict; if missing -> error)
    - Uses raw['labeled']['embs'] and raw['labeled']['labels'] as training set.
    - Scores each split using its own (embs, y_pred).
    """

    def apply(self, *, df: Any, raw: Mapping[str, Any]) -> np.ndarray:
        if "split" not in df.columns or "id" not in df.columns:
            raise ConfigError("df must contain 'split' and 'id'")
        from tensaku.trustscore import TrustScorer

        ref_split = "labeled"
        Xtr = np.asarray(_require_raw_split(raw, ref_split, "embs"), dtype=float)
        ytr = np.asarray(_require_raw_split(raw, ref_split, "labels"), dtype=int)

        if Xtr.ndim != 2:
            raise ConfigError("trust: labeled embs must be 2D [N,D]")
        if ytr.ndim != 1 or len(ytr) != len(Xtr):
            raise ConfigError("trust: labeled labels must be 1D and match embs length")

        scorer = TrustScorer(
            version=str(self.cfg.get("version", "v2")),
            metric=str(self.cfg.get("metric", "cosine")),
            k=int(self.cfg.get("k", 1)),
            k_list=self.cfg.get("k_list", [1, 3, 5]),
            agg=str(self.cfg.get("agg", "median")),
            trim_q=float(self.cfg.get("trim_q", 0.1)),
            normalize=str(self.cfg.get("normalize", "zscore")),
        ).fit(Xtr, ytr)

        out = np.empty(len(df), dtype=float)
        for split_name, df_part in df.groupby("split", sort=False):
            Xte = np.asarray(_require_raw_split(raw, split_name, "embs"), dtype=float)
            y_pred = np.asarray(_require_raw_split(raw, split_name, "y_pred"), dtype=int)
            raw_ids = _require_raw_split(raw, split_name, "ids")

            if Xte.ndim != 2:
                raise ConfigError(f"trust: {split_name} embs must be 2D")
            if y_pred.ndim != 1 or len(y_pred) != len(Xte):
                raise ConfigError(f"trust: {split_name} y_pred must be 1D and match embs")

            scores = scorer.score(Xte, y_pred)
            aligned = _align_scores_by_id(
                df_part["id"].to_numpy(),
                raw_ids,
                scores,
                where=f"TrustEstimator({split_name})",
            )
            out[df_part.index.to_numpy()] = aligned
        return out


def create_confidence_estimator(name: str, *, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Any:
    """Create a confidence estimator by name (STRICT).

    The registry `tensaku.registry.create(name, ...)` will call class constructors.
    This helper is provided for direct use if needed.
    """
    key = str(name).strip().lower()
    # prefer registry registrations
    from tensaku.registry import get

    obj = get(key)
    if callable(obj):
        return obj(cfg=cfg or {}, **kwargs)
    return obj


# Backward-compatible alias
create_estimator = create_confidence_estimator
