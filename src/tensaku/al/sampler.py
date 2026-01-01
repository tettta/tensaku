# /home/esakit25/work/tensaku/src/tensaku/al/sampler.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.al.sampler
@role  : Active Learning (AL) におけるサンプル選択ロジック（Sampler）の定義
@policy:
  - フォールバック（サイレントに別サンプラーへ切替）はしない。必要条件が満たされない場合は例外で停止。
  - kmeans/hybrid は「ファイルパスを知らない」純粋ロジック。埋め込みは pipeline/task 側から features/feature_ids として渡す。
  - config 上は "uncertainty" を基本的に使わず、"msp" / "trust" / "entropy" を選ぶ前提。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tensaku.al.state import ALState
from tensaku.registry import register as _register_global, create as _registry_create

LOGGER = logging.getLogger(__name__)

SAMPLER_REGISTRY_PREFIX = "sampler/"


def _infer_higher_is_better_from_conf_key(conf_key: str) -> bool:
    """Infer score direction from a confidence/uncertainty key.

    Policy:
      - confidence-like (msp/trust/margin/prob_margin): low => more informative => higher_is_better=False
      - uncertainty-like (entropy/energy): high => more informative => higher_is_better=True

    NOTE:
      - This is used only when cfg does not explicitly set higher_is_better.
      - If you introduce a new conf key, prefer setting higher_is_better explicitly in cfg.
    """
    k = (conf_key or "").strip().lower()
    if k in {"entropy", "energy"}:
        return True
    # default: choose low-score items first
    return False


def register_sampler(name: str, *, override: bool = False):
    """
    Sampler 用のレジストリデコレータ。
    - registry 上のキーは 'sampler/<name>' という名前空間付きにする。
    - 使い方: @register_sampler("random") / @register_sampler("hybrid") など
    """
    key = SAMPLER_REGISTRY_PREFIX + str(name)
    return _register_global(key, override=override)


# Scikit-learn (KMeans) の利用可能性チェック
try:
    from sklearn.cluster import KMeans  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Core Samplers
# =============================================================================


@dataclass
class BaseSampler:
    """
    AL サンプル選択のベースクラス。

    NOTE:
      - kmeans/hybrid など埋め込みが必要なサンプラーは、features/feature_ids を pipeline/task 側から受け取る。
      - loop 側がまだ features を渡していない段階でも API 互換性を保つため、引数は optional にしているが、
        必要な場合は例外を投げる。
    """

    name: str = "base"
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        raise NotImplementedError


@dataclass
class RandomSampler(BaseSampler):
    """プールから一様ランダムにサンプルを選択する Sampler。"""

    name: str = "random"

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if budget <= 0 or state.n_pool <= 0:
            return []

        # 決定論的挙動のために sort→shuffle（pool_ids の元順序依存を排除）
        pool_ids = sorted(list(state.pool_ids), key=str)
        self._rng.shuffle(pool_ids)
        selected = pool_ids[: min(budget, len(pool_ids))]

        LOGGER.info("[RandomSampler] selected=%d (pool=%d, budget=%d)", len(selected), state.n_pool, budget)
        return selected


@dataclass
class UncertaintySampler(BaseSampler):
    """
    scores に基づきサンプルを選択する Sampler（名称は互換のため UncertaintySampler のまま）。

    higher_is_better:
      - True  : スコアが大きいほど「優先して選びたい」（例: entropy/energy のような “不確実性” 指標）
      - False : スコアが小さいほど「優先して選びたい」（例: msp/trust のような “確信度” 指標で、低確信度を選ぶ）
    """

    name: str = "uncertainty"
    by: str = "score"
    higher_is_better: bool = False

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        
        if scores is None:
            raise ValueError(
                f"UncertaintySampler requires 'scores' but got None. "
                "Check if Task returns pool_scores or if Scheduler is misconfigured (e.g. using Uncertainty for Round 0)."
            )
        if budget <= 0 or state.n_pool <= 0:
            return []

        if scores is None or not scores:
            raise RuntimeError(
                f"[UncertaintySampler] scores is empty (by={self.by}). "
                "Task/pipeline side must provide pool_scores for this sampler."
            )

        pool_scores: List[Tuple[Any, float]] = []
        missing = 0
        for pid in state.pool_ids:
            if pid in scores:
                pool_scores.append((pid, float(scores[pid])))
            else:
                missing += 1

        if not pool_scores:
            raise RuntimeError(
                f"[UncertaintySampler] no matching scores for current pool (by={self.by}). "
                "Check that pool_scores are keyed by pool 'id'."
            )

        if missing > 0:
            LOGGER.warning(
                "[UncertaintySampler] missing scores for %d pool items (by=%s). Proceeding with %d scored items.",
                missing,
                self.by,
                len(pool_scores),
            )

        # 低確信度を選ぶ（higher_is_better=False）なら昇順、entropy 等なら降順
        pool_scores.sort(key=lambda x: x[1], reverse=self.higher_is_better)

        selected = [pid for pid, _ in pool_scores[: min(budget, len(pool_scores))]]
        LOGGER.info(
            "[UncertaintySampler] by=%s higher_is_better=%s selected=%d (pool=%d, budget=%d, scored=%d)",
            self.by,
            self.higher_is_better,
            len(selected),
            state.n_pool,
            budget,
            len(pool_scores),
        )
        return selected


@dataclass
class DensityWeightedUncertaintySampler(BaseSampler):
    """Density-weighted uncertainty sampler.

    代表例: "density-weighted uncertainty" (uncertainty × density).

    直感:
      - 不確実性 top-k は分布の端に偏りやすい
      - そこで、近傍が多い（=dense / 代表的）な点を優先して選び、
        "ラベル効率" を上げる狙い

    要件:
      - scores（pool_scores）と features（埋め込み）が必要
      - higher_is_better=False のときは "低いほど良い" score を自動で反転して扱う
    """

    name: str = "unc_density"
    k_nn: int = 10
    metric: str = "cosine"  # 'cosine'|'euclidean'
    alpha: float = 1.0  # uncertainty weight
    beta: float = 1.0  # density weight
    higher_is_better: bool = False

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if budget <= 0 or state.n_pool <= 0:
            return []

        if scores is None or not scores:
            raise RuntimeError("[DensityWeightedUncertaintySampler] scores is required (provide pool_scores).")
        if features is None or feature_ids is None:
            raise RuntimeError("[DensityWeightedUncertaintySampler] requires 'features' and 'feature_ids'.")

        feat, ids = _normalize_features(features, feature_ids)
        id_to_idx = {ids[i]: i for i in range(len(ids))}
        target_ids = [pid for pid in state.pool_ids if pid in id_to_idx and pid in scores]
        if not target_ids:
            raise RuntimeError("[DensityWeightedUncertaintySampler] no usable ids (need intersection of pool_ids, feature_ids, scores).")

        X = feat[[id_to_idx[i] for i in target_ids], :].astype(float)
        n = len(target_ids)
        k = min(int(budget), n)
        k_nn = max(1, min(int(self.k_nn), max(1, n - 1)))

        metric = (self.metric or "cosine").lower()
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"[DensityWeightedUncertaintySampler] unsupported metric={self.metric!r}")

        if not SKLEARN_AVAILABLE:
            raise RuntimeError("[DensityWeightedUncertaintySampler] scikit-learn is required (NearestNeighbors), but unavailable.")
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        # Density via average similarity to kNN
        if metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            X_use = X / norms
        else:
            X_use = X

        nn = NearestNeighbors(n_neighbors=min(k_nn + 1, n), metric=metric)
        nn.fit(X_use)
        dist, _ = nn.kneighbors(X_use, return_distance=True)
        # drop self (distance 0)
        dist = dist[:, 1:]
        if metric == "cosine":
            sim = 1.0 - dist
            density = sim.mean(axis=1)
        else:
            # convert distance to a bounded similarity (avoid division by zero)
            density = 1.0 / (1.0 + dist.mean(axis=1))

        # Uncertainty term
        s = np.array([float(scores[pid]) for pid in target_ids], dtype=float)
        unc = s if self.higher_is_better else (-s)

        def _z(x: np.ndarray) -> np.ndarray:
            mu = float(x.mean())
            sd = float(x.std())
            if sd == 0.0:
                return x - mu
            return (x - mu) / sd

        comb = float(self.alpha) * _z(unc) + float(self.beta) * _z(density)
        order = np.argsort(-comb)  # higher better
        selected = [target_ids[int(i)] for i in order[:k]]
        selected = list(dict.fromkeys(selected))
        if len(selected) > budget:
            selected = selected[:budget]

        LOGGER.info(
            "[DensityWeightedUncertaintySampler] metric=%s k_nn=%d hib=%s selected=%d",
            metric,
            k_nn,
            bool(self.higher_is_better),
            len(selected),
        )
        return selected


@dataclass
class NotImplementedSampler(BaseSampler):
    """Placeholder sampler for future extensions.

    This sampler is intentionally strict: selecting with it raises NotImplementedError.
    Use it as a reminder to implement a sampler once the required artifacts exist.
    """

    name: str = "not_implemented"
    reason: str = ""

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        raise NotImplementedError(f"Sampler '{self.name}' is not implemented yet. {self.reason}")


@dataclass
class ClusteringSampler(BaseSampler):
    """
    K-Means クラスタリングに基づく多様性サンプラー（Diversity Sampler）。

    - features: (N, D) 埋め込み（pool もしくは pool を含む集合）
    - feature_ids: 長さ N の ID 列（features の行と対応）
    """

    name: str = "kmeans"
    k: Optional[int] = None  # クラスタ数（Noneなら budget）

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if budget <= 0 or state.n_pool <= 0:
            return []

        if not SKLEARN_AVAILABLE:
            raise RuntimeError("[ClusteringSampler] scikit-learn is required for kmeans sampler, but unavailable.")

        if features is None or feature_ids is None:
            raise RuntimeError(
                "ClusteringSampler.select requires 'features' and 'feature_ids'. "
                "Provide embeddings from the task/pipeline side."
            )

        feat, ids = _normalize_features(features, feature_ids)

        # 現在の pool に残っている id のみ対象にする
        id_to_idx = {ids[i]: i for i in range(len(ids))}
        target_ids = [pid for pid in state.pool_ids if pid in id_to_idx]

        if not target_ids:
            raise RuntimeError("[ClusteringSampler] no intersection between pool_ids and feature_ids.")

        X = feat[[id_to_idx[i] for i in target_ids], :]

        n = len(target_ids)
        n_clusters = self.k if self.k is not None else budget
        n_clusters = max(1, min(int(n_clusters), n))

        LOGGER.info("[ClusteringSampler] round=%d pool=%d target=%d k=%d budget=%d",
                    state.round_index, state.n_pool, n, n_clusters, budget)

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init="auto", max_iter=300)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        selected: List[Any] = []
        # 各クラスタ中心に最も近い点を1つ
        for c in range(n_clusters):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            sub = X[idxs]
            d = np.linalg.norm(sub - centers[c][None, :], axis=1)
            best_local = int(d.argmin())
            selected.append(target_ids[int(idxs[best_local])])

        # 念のため重複排除
        selected = list(dict.fromkeys(selected))
        if len(selected) > budget:
            self._rng.shuffle(selected)
            selected = selected[:budget]

        LOGGER.info("[ClusteringSampler] selected=%d", len(selected))
        return selected


@dataclass
class KCenterSampler(BaseSampler):
    """k-center greedy による多様性サンプラー（core-set / coverage）。

    目的:
      - 選択集合が pool 全体をできるだけカバーする（最遠点を順に追加）

    実装方針（最小・堅牢）:
      - metric='cosine' or 'euclidean' を選べる
      - 初期点は seed により決定論的（random）
    """

    name: str = "kcenter"
    metric: str = "cosine"  # 'cosine' | 'euclidean'

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if budget <= 0 or state.n_pool <= 0:
            return []

        if features is None or feature_ids is None:
            raise RuntimeError(
                "KCenterSampler.select requires 'features' and 'feature_ids'. Provide embeddings from the task/pipeline side."
            )

        feat, ids = _normalize_features(features, feature_ids)
        id_to_idx = {ids[i]: i for i in range(len(ids))}
        target_ids = [pid for pid in state.pool_ids if pid in id_to_idx]
        if not target_ids:
            raise RuntimeError("[KCenterSampler] no intersection between pool_ids and feature_ids.")

        X = feat[[id_to_idx[i] for i in target_ids], :].astype(float)

        # metric preparation
        metric = (self.metric or "cosine").lower()
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"[KCenterSampler] unsupported metric={self.metric!r} (expected 'cosine'|'euclidean')")

        if metric == "cosine":
            # normalize rows to unit length; avoid div-by-zero
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            Xn = X / norms
            # distance = 1 - cosine_similarity
            def _dist_to_set(x: np.ndarray, S: np.ndarray) -> np.ndarray:
                # returns min distance to set for each row in x
                sim = x @ S.T
                d = 1.0 - sim
                return d.min(axis=1)

            X_use = Xn
        else:
            def _dist_to_set(x: np.ndarray, S: np.ndarray) -> np.ndarray:
                # ||x - s|| for each s; take min
                # shapes: x(N,D), S(k,D)
                dif = x[:, None, :] - S[None, :, :]
                d = np.linalg.norm(dif, axis=2)
                return d.min(axis=1)

            X_use = X

        n = len(target_ids)
        k = min(int(budget), n)

        # initial point: deterministic random over target_ids
        idxs = list(range(n))
        self._rng.shuffle(idxs)
        first = idxs[0]
        selected_idx = [first]

        # maintain min-distance to selected set
        S = X_use[np.array(selected_idx, dtype=int)]
        min_d = _dist_to_set(X_use, S)

        while len(selected_idx) < k:
            # pick farthest point
            cand = int(np.argmax(min_d))
            if cand in selected_idx:
                # numerical tie/degenerate; pick next best
                order = np.argsort(-min_d)
                cand = next(int(i) for i in order if int(i) not in selected_idx)
            selected_idx.append(cand)
            S = X_use[np.array(selected_idx, dtype=int)]
            min_d = np.minimum(min_d, _dist_to_set(X_use, X_use[np.array([cand], dtype=int)]))

        selected = [target_ids[i] for i in selected_idx]
        selected = list(dict.fromkeys(selected))
        if len(selected) > budget:
            selected = selected[:budget]

        LOGGER.info("[KCenterSampler] metric=%s selected=%d", metric, len(selected))
        return selected


@dataclass
class HybridSampler(BaseSampler):
    """
    ハイブリッドサンプリング（Uncertainty + Diversity）。

    1) scores で候補を絞り込み（sub_budget_ratio 倍）
    2) 候補の features で KMeans を回し、多様性を確保して budget 件返す
    """

    name: str = "hybrid"
    sub_budget_ratio: float = 5.0
    k: Optional[int] = None
    higher_is_better: Optional[bool] = None  # None のときは推定（後方互換）

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if budget <= 0 or state.n_pool <= 0:
            return []

        if not SKLEARN_AVAILABLE:
            raise RuntimeError("[HybridSampler] scikit-learn is required for hybrid sampler, but unavailable.")

        if scores is None or not scores:
            raise RuntimeError("[HybridSampler] scores is required (provide pool_scores from task/pipeline side).")

        if features is None or feature_ids is None:
            raise RuntimeError(
                "HybridSampler.select requires 'features' and 'feature_ids'. Provide embeddings from the task/pipeline side."
            )

        feat, ids = _normalize_features(features, feature_ids)
        id_to_idx = {ids[i]: i for i in range(len(ids))}

        # (A) scores を pool 上で整列（ここでは “低確信度を優先” を基本とする）
        #     - msp/trust は低いほど優先（higher_is_better=False）
        #     - entropy は高いほど優先（higher_is_better=True）
        #   どれかは scores の意味次第なので、ここは "mode 推定" のみにしておく。
        #   ※ pipeline側が「選びたい順に大きい値」へ変換する設計でもOK。その場合は higher_is_better=True を cfg で渡すのが理想。
        #   いまは保守的に「msp/trust は低いほど、entropy は高いほど」とする。
        # Prefer explicit direction; else keep backward-compat default (False: low scores first).
        hib = self.higher_is_better if self.higher_is_better is not None else False
        mode = {"higher_is_better": bool(hib)}

        pool_scores: List[Tuple[Any, float]] = []
        for pid in state.pool_ids:
            if pid in scores:
                pool_scores.append((pid, float(scores[pid])))

        if not pool_scores:
            raise RuntimeError("[HybridSampler] no matching scores for current pool.")

        pool_scores.sort(key=lambda x: x[1], reverse=mode["higher_is_better"])

        cand_budget = min(int(max(1, round(budget * self.sub_budget_ratio))), len(pool_scores))
        cand_ids = [pid for pid, _ in pool_scores[:cand_budget]]

        # (B) cand_ids の features を抽出
        rows = [id_to_idx[i] for i in cand_ids if i in id_to_idx]
        if not rows:
            raise RuntimeError("[HybridSampler] no candidate ids found in feature_ids (cannot run KMeans).")

        X = feat[np.array(rows, dtype=int)]
        ids_X = [cand_ids[i] for i in range(len(cand_ids)) if cand_ids[i] in id_to_idx]

        n = len(ids_X)
        n_clusters = self.k if self.k is not None else budget
        n_clusters = max(1, min(int(n_clusters), n))

        LOGGER.info(
            "[HybridSampler] round=%d pool=%d candidates=%d k=%d budget=%d (higher_is_better=%s)",
            state.round_index,
            state.n_pool,
            n,
            n_clusters,
            budget,
            mode["higher_is_better"],
        )

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init="auto", max_iter=300)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        selected: List[Any] = []
        for c in range(n_clusters):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            sub = X[idxs]
            d = np.linalg.norm(sub - centers[c][None, :], axis=1)
            best_local = int(d.argmin())
            selected.append(ids_X[int(idxs[best_local])])

        selected = list(dict.fromkeys(selected))
        if len(selected) > budget:
            self._rng.shuffle(selected)
            selected = selected[:budget]

        LOGGER.info("[HybridSampler] selected=%d", len(selected))
        return selected


@dataclass
class DivThenUncertaintySampler(BaseSampler):
    """Diversity -> Uncertainty (D2U).

    手順:
      1) features で KMeans を回してクラスタを作る（多様性確保）
      2) 各クラスタから 1 点ずつ、scores に基づいて選ぶ（不確実性/低確信度など）

    用途:
      - kmeans が random に負けるケースで、"中心"ではなく"境界/難所"を拾いたい
      - uncertainty top-k が偏るケースで、diversity を強制したい
    """

    name: str = "d2u"
    k: Optional[int] = None
    higher_is_better: Optional[bool] = None

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
        *,
        features: Optional[np.ndarray] = None,
        feature_ids: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if budget <= 0 or state.n_pool <= 0:
            return []

        if not SKLEARN_AVAILABLE:
            raise RuntimeError("[DivThenUncertaintySampler] scikit-learn is required, but unavailable.")
        if scores is None or not scores:
            raise RuntimeError("[DivThenUncertaintySampler] scores is required (provide pool_scores).")
        if features is None or feature_ids is None:
            raise RuntimeError("[DivThenUncertaintySampler] requires 'features' and 'feature_ids'.")

        feat, ids = _normalize_features(features, feature_ids)
        id_to_idx = {ids[i]: i for i in range(len(ids))}
        target_ids = [pid for pid in state.pool_ids if pid in id_to_idx and pid in scores]
        if not target_ids:
            raise RuntimeError("[DivThenUncertaintySampler] no usable ids (need intersection of pool_ids, feature_ids, scores).")

        X = feat[[id_to_idx[i] for i in target_ids], :]
        n = len(target_ids)
        n_clusters = self.k if self.k is not None else budget
        n_clusters = max(1, min(int(n_clusters), n))

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init="auto", max_iter=300)
        labels = kmeans.fit_predict(X)

        # direction
        hib = self.higher_is_better
        if hib is None:
            # infer from any key; task extracted score already; best effort: default False
            hib = False

        selected: List[Any] = []
        for c in range(n_clusters):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            # pick best score within cluster
            cand_ids = [target_ids[int(i)] for i in idxs]
            cand_ids.sort(key=lambda pid: float(scores[pid]), reverse=bool(hib))
            selected.append(cand_ids[0])

        selected = list(dict.fromkeys(selected))
        if len(selected) > budget:
            self._rng.shuffle(selected)
            selected = selected[:budget]

        LOGGER.info(
            "[DivThenUncertaintySampler] k=%d higher_is_better=%s selected=%d",
            n_clusters,
            bool(hib),
            len(selected),
        )
        return selected


# =============================================================================
# Helpers
# =============================================================================


def _normalize_features(features: np.ndarray, feature_ids: Sequence[Any]) -> Tuple[np.ndarray, List[Any]]:
    if not isinstance(features, np.ndarray):
        raise TypeError(f"features must be np.ndarray, got {type(features)}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2D array (N,D), got shape={features.shape}")
    if len(feature_ids) != features.shape[0]:
        raise ValueError(
            f"feature_ids length must match features rows: len(feature_ids)={len(feature_ids)} vs N={features.shape[0]}"
        )
    ids = list(feature_ids)
    return features, ids


@register_sampler("random")
def _factory_random_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return RandomSampler(seed=seed)


@register_sampler("msp")
def _factory_msp_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="msp", higher_is_better=False)


@register_sampler("trust")
def _factory_trust_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="trust", higher_is_better=False)


@register_sampler("entropy")
def _factory_entropy_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    # entropy: higher => more uncertain
    return UncertaintySampler(seed=seed, by="entropy", higher_is_better=True)


# =============================================================================
# Extended naming (extensible)
#
# - Prefer *concept first* naming:
#     unc_* : uncertainty/confidence based
#     div_* : diversity based
#     u2d_* : uncertainty -> diversity
#     d2u_* : diversity -> uncertainty
# - Keep legacy names (msp/trust/entropy/kmeans/hybrid) working.
# - For future keys, use generic 'unc' / 'hyb_*' + explicit conf_key.
# =============================================================================


@register_sampler("unc")
def _factory_unc_generic(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    # Strict: conf_key must be provided (either directly or derived upstream).
    conf_key = sampler_cfg.get("conf_key", sampler_cfg.get("by", None))
    if conf_key is None:
        raise ValueError("sampler 'unc' requires sampler_cfg.conf_key (e.g. 'msp', 'margin', 'entropy').")
    conf_key = str(conf_key)
    hib = sampler_cfg.get("higher_is_better", None)
    higher_is_better = bool(hib) if isinstance(hib, bool) else _infer_higher_is_better_from_conf_key(conf_key)
    return UncertaintySampler(seed=seed, by=conf_key, higher_is_better=higher_is_better)


@register_sampler("unc_msp")
def _factory_unc_msp(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="msp", higher_is_better=False)


@register_sampler("unc_trust")
def _factory_unc_trust(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="trust", higher_is_better=False)


@register_sampler("unc_margin")
def _factory_unc_margin(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="margin", higher_is_better=False)


@register_sampler("unc_prob_margin")
def _factory_unc_prob_margin(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="prob_margin", higher_is_better=False)


@register_sampler("unc_entropy")
def _factory_unc_entropy(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="entropy", higher_is_better=True)


@register_sampler("unc_energy")
def _factory_unc_energy(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return UncertaintySampler(seed=seed, by="energy", higher_is_better=True)


@register_sampler("unc_density")
def _factory_unc_density(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    k_nn = int(sampler_cfg.get("k_nn", 10))
    metric = str(sampler_cfg.get("metric", "cosine"))
    alpha = float(sampler_cfg.get("alpha", 1.0))
    beta = float(sampler_cfg.get("beta", 1.0))
    hib_raw = sampler_cfg.get("higher_is_better", None)
    hib = bool(hib_raw) if isinstance(hib_raw, bool) else None
    if hib is None:
        conf_key = sampler_cfg.get("conf_key", sampler_cfg.get("by", None))
        if conf_key is not None:
            hib = _infer_higher_is_better_from_conf_key(str(conf_key))
        else:
            hib = False
    return DensityWeightedUncertaintySampler(
        seed=seed,
        k_nn=k_nn,
        metric=metric,
        alpha=alpha,
        beta=beta,
        higher_is_better=bool(hib),
    )


@register_sampler("div_kmeans")
def _factory_div_kmeans(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return _factory_kmeans_sampler(cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)


@register_sampler("kcenter")
def _factory_kcenter(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    metric = str(sampler_cfg.get("metric", "cosine"))
    return KCenterSampler(seed=seed, metric=metric)


@register_sampler("div_kcenter")
def _factory_div_kcenter(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return _factory_kcenter(cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)


@register_sampler("d2u_kmeans")
def _factory_d2u_kmeans(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    k = sampler_cfg.get("k", None)
    k = int(k) if k is not None else None
    hib_raw = sampler_cfg.get("higher_is_better", None)
    hib = bool(hib_raw) if isinstance(hib_raw, bool) else None
    # If conf_key is available, infer direction as a default.
    if hib is None:
        conf_key = sampler_cfg.get("conf_key", sampler_cfg.get("by", None))
        if conf_key is not None:
            hib = _infer_higher_is_better_from_conf_key(str(conf_key))
    return DivThenUncertaintySampler(seed=seed, k=k, higher_is_better=hib)


@register_sampler("u2d_kmeans")
def _factory_u2d_kmeans(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    # alias of HybridSampler (uncertainty -> kmeans diversity)
    return _factory_hybrid_sampler(cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)


@register_sampler("hyb_u2d_kmeans")
def _factory_hyb_u2d_kmeans(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return _factory_u2d_kmeans(cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)


@register_sampler("hyb_d2u_kmeans")
def _factory_hyb_d2u_kmeans(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return _factory_d2u_kmeans(cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)


# Backward compat alias (previous patch name)
@register_sampler("cluster_uncertainty")
def _factory_cluster_uncertainty(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return _factory_d2u_kmeans(cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)


# ----------------------------------------------------------------------------
# Future extensions (placeholders)
# ----------------------------------------------------------------------------


@register_sampler("qbc")
def _factory_qbc_placeholder(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return NotImplementedSampler(
        seed=seed,
        name="qbc",
        reason="Query-by-Committee requires committee disagreement scores (e.g., vote entropy) to be provided as pool_scores.",
    )


@register_sampler("badge")
def _factory_badge_placeholder(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    return NotImplementedSampler(
        seed=seed,
        name="badge",
        reason="BADGE requires gradient embeddings (gradients of loss w.r.t. last-layer) for each pool sample.",
    )



# 後方互換（基本は使わない方針）
@register_sampler("uncertainty")
def _factory_uncertainty_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    # 旧設定: uncertainty_strategy / higher_is_better を明示できるようにしておく
    by = str(sampler_cfg.get("by", sampler_cfg.get("uncertainty_strategy", "score")))
    hib = bool(sampler_cfg.get("higher_is_better", False))
    return UncertaintySampler(seed=seed, by=by, higher_is_better=hib)


@register_sampler("kmeans")
def _factory_kmeans_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    k = sampler_cfg.get("k", None)
    k = int(k) if k is not None else None
    return ClusteringSampler(seed=seed, k=k)


@register_sampler("hybrid")
def _factory_hybrid_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    k = sampler_cfg.get("k", None)
    k = int(k) if k is not None else None
    ratio = float(sampler_cfg.get("sub_budget_ratio", 5.0))
    return HybridSampler(seed=seed, k=k, sub_budget_ratio=ratio)


# =============================================================================
# Public factory (cfg -> sampler)
# =============================================================================


def create_sampler(cfg: Mapping[str, Any]) -> BaseSampler:
    """
    cfg から Sampler を構築するファクトリ関数（フォールバック禁止・厳格）。

    注意:
      - round0 フォールバックは完全撤廃。埋め込み等の前提が満たされない場合は例外で停止。

    対応する設定形式:
      1) 文字列形式
         al:
           sampler: "msp"

      2) 辞書形式
         al:
           sampler:
             name: "kmeans"
             k: 32
    """
    al_cfg = cfg.get("al")
    if not isinstance(al_cfg, Mapping):
        raise ValueError("cfg['al'] が未設定/不正です。al セクションを設定してください。")

    sampler_cfg_raw = al_cfg.get("sampler")
    if sampler_cfg_raw is None:
        raise ValueError("cfg['al']['sampler'] が未設定です。サンプラーを明示指定してください。")

    if isinstance(sampler_cfg_raw, str):
        name = sampler_cfg_raw.strip().lower()
        sampler_cfg: Mapping[str, Any] = {}
        if not name:
            raise ValueError("cfg['al']['sampler'] が空文字です。")
    elif isinstance(sampler_cfg_raw, Mapping):
        if "name" not in sampler_cfg_raw:
            raise ValueError("cfg['al']['sampler']['name'] が未設定です。")
        name = str(sampler_cfg_raw["name"]).strip().lower()
        if not name:
            raise ValueError("cfg['al']['sampler']['name'] が空文字です。")
        sampler_cfg = sampler_cfg_raw
        if "round0" in sampler_cfg_raw:
            raise ValueError("cfg['al']['sampler'] に 'round0' は指定できません（フォールバック完全撤廃）。設定から削除してください。")
    else:
        raise TypeError(f"cfg['al']['sampler'] は str か Mapping が必要です: got {type(sampler_cfg_raw)}")

    seed = al_cfg.get("seed")
    seed = seed if isinstance(seed, int) else None

    key = SAMPLER_REGISTRY_PREFIX + name
    try:
        sampler = _registry_create(key, cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)
    except KeyError as exc:
        raise ValueError(f"Unknown sampler name: {name!r}") from exc

    if not isinstance(sampler, BaseSampler):
        raise TypeError(f"Sampler factory '{key}' must return BaseSampler, got {type(sampler)}")

    LOGGER.info("Created sampler: name=%s class=%s seed=%s", name, type(sampler).__name__, seed)
    return sampler




def create_sampler_from_spec(cfg: Mapping[str, Any], sampler_spec: Any) -> BaseSampler:
    """Create sampler from a *spec* (used by schedule rules).

    sampler_spec formats:
      - str: "random" / "msp" / ...
      - Mapping: {"name": "kmeans", ...}  (optionally includes "seed")

    Seed policy (STRICT-ish):
      - If sampler_spec provides 'seed', use it.
      - Else, require cfg['run']['seed'] (already required elsewhere) and use it for determinism.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be Mapping")
    run_cfg = cfg.get("run")
    if not isinstance(run_cfg, Mapping) or "seed" not in run_cfg:
        raise ValueError("cfg['run']['seed'] is required for deterministic sampler creation.")

    seed_default = run_cfg["seed"]
    if isinstance(seed_default, bool) or not isinstance(seed_default, int):
        raise ValueError("cfg['run']['seed'] must be int")

    # Normalize spec -> (name, sampler_cfg, seed)
    if isinstance(sampler_spec, str):
        name = sampler_spec.strip().lower()
        if not name:
            raise ValueError("sampler_spec is empty string")
        sampler_cfg: Mapping[str, Any] = {}
        seed = int(seed_default)
    elif isinstance(sampler_spec, Mapping):
        if "name" not in sampler_spec:
            raise ValueError("sampler_spec mapping missing 'name'")
        name = str(sampler_spec["name"]).strip().lower()
        if not name:
            raise ValueError("sampler_spec['name'] is empty")
        sampler_cfg = {k: v for k, v in sampler_spec.items() if k not in ("name", "seed")}
        seed_val = sampler_spec.get("seed", seed_default)
        if isinstance(seed_val, bool) or not isinstance(seed_val, int):
            raise ValueError("sampler_spec['seed'] (or run.seed) must be int")
        seed = int(seed_val)
    else:
        raise TypeError(f"sampler_spec must be str or Mapping, got {type(sampler_spec)}")

    key = SAMPLER_REGISTRY_PREFIX + name
    try:
        sampler = _registry_create(key, cfg=cfg, sampler_cfg=sampler_cfg, seed=seed)
    except KeyError as exc:
        raise ValueError(f"Unknown sampler name: {name!r}") from exc

    if not isinstance(sampler, BaseSampler):
        raise TypeError(f"Sampler factory '{key}' must return BaseSampler, got {type(sampler)}")

    LOGGER.info("Created sampler(from_spec): name=%s class=%s seed=%s", name, type(sampler).__name__, seed)
    return sampler

# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1) Minimal ALState mock
    class _DummyState:
        def __init__(self) -> None:
            self.round_index = 0
            self.pool_ids = {"a", "b", "c", "d"}
            self.n_pool = len(self.pool_ids)

    st = _DummyState()  # type: ignore[assignment]

    # 2) Random
    s = RandomSampler(seed=42)
    print("random:", s.select(st, scores=None, budget=2))

    # 3) MSP (low score => selected)
    scores = {"a": 0.9, "b": 0.1, "c": 0.3, "d": 0.2}
    s2 = UncertaintySampler(seed=42, by="msp", higher_is_better=False)
    print("msp(low first):", s2.select(st, scores=scores, budget=2))

    # 5) kmeans with embeddings (simulate round>=1)
    st.round_index = 1
    s3 = ClusteringSampler(seed=42, k=2)
    feat_ids = ["a", "b", "c", "d"]
    feat = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.0], [9.9, 10.2]], dtype=float)
    print("kmeans:", s3.select(st, scores=None, budget=2, features=feat, feature_ids=feat_ids))


"""
# ---- Minimal smoke (bash) ----
python -m tensaku.al.sampler

# ---- Module evaluation checklist ----
- SRP: Sampler は「選択ロジック」のみ。ファイルI/O/パス解決はしない（features は外から注入）。
- API契約: select(state, scores, budget, *, features, feature_ids) を全 Sampler で統一。
- データ契約: kmeans/hybrid は features/feature_ids 必須。scores 必須は msp/trust/entropy/hybrid。
- 依存: kmeans/hybrid は sklearn 必須（無ければ例外で停止）。
- 再現性: seed により Random/KMeans の決定論性を担保。
- エラー処理: フォールバック禁止。条件不足は RuntimeError/ValueError で停止。
- ロギング: logging.getLogger(__name__) + INFO/WARNING/ERROR を統一。
- 性能: numpy ベースで O(N) 前処理 + KMeans。大規模 N の場合は pipeline 側で候補圧縮推奨。
- 改善アクション（<=3）:
  1) Hybrid の higher_is_better を cfg から明示的に渡せるようにする（いまは保守的推定）。
  2) feature_ids→index の辞書生成をキャッシュできる（同一 round で複数回呼ぶなら）。
"""
