# /home/esakit25/work/tensaku/src/tensaku/al/sampler.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.al.sampler
@role  : Active Learning (AL) におけるサンプル選択ロジック（Sampler）の定義
@policy:
  - フォールバック（サイレントに別サンプラーへ切替）はしない。必要条件が満たされない場合は例外で停止。
  - kmeans/hybrid は「ファイルパスを知らない」純粋ロジック。埋め込みは pipeline/task 側から features/feature_ids として渡す。
  - config 上は "uncertainty" を基本的に使わず、"msp" / "trust" / "entropy" / "random" を選ぶ前提。
  - round=0 の挙動は kmeans/hybrid 側の round0 で明示指定できる（"random"/"msp"/"trust"/"entropy"）。
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
class ClusteringSampler(BaseSampler):
    """
    K-Means クラスタリングに基づく多様性サンプラー（Diversity Sampler）。

    - features: (N, D) 埋め込み（pool もしくは pool を含む集合）
    - feature_ids: 長さ N の ID 列（features の行と対応）

    round=0 など埋め込みが未提供のケースのために round0 を持つ。
      round0 in {"random","msp","trust","entropy"}（基本方針として "uncertainty" は使わない）
    """

    name: str = "kmeans"
    k: Optional[int] = None  # クラスタ数（Noneなら budget）
    round0: str = "random"   # "random" / "msp" / "trust" / "entropy"

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

        # round=0 は埋め込み無し運用を許容（明示設定で分岐）
        if state.round_index == 0 and (features is None or feature_ids is None):
            return _select_round0(mode=self.round0, seed=self.seed, state=state, scores=scores, budget=budget)

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
class HybridSampler(BaseSampler):
    """
    ハイブリッドサンプリング（Uncertainty + Diversity）。

    1) scores で候補を絞り込み（sub_budget_ratio 倍）
    2) 候補の features で KMeans を回し、多様性を確保して budget 件返す

    round0:
      - round=0 は埋め込み無しでも動くように、round0 で明示指定したサンプラーを使う。
      - round0 in {"random","msp","trust","entropy"}（基本方針として "uncertainty" は使わない）
    """

    name: str = "hybrid"
    sub_budget_ratio: float = 5.0
    k: Optional[int] = None
    round0: str = "msp"  # round0 は “低確信度” を取るケースが多い想定で msp をデフォルト

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

        # round=0 は埋め込み無し運用を許容（明示設定で分岐）
        if state.round_index == 0 and (features is None or feature_ids is None):
            return _select_round0(mode=self.round0, seed=self.seed, state=state, scores=scores, budget=budget)

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
        mode = _infer_score_mode_from_cfg_like(scores_hint_name=None)

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


def _select_round0(
    *,
    mode: str,
    seed: Optional[int],
    state: ALState,
    scores: Optional[Mapping[Any, float]],
    budget: int,
) -> List[Any]:
    """
    round=0 用の明示的分岐。
    config 上は "uncertainty" を使わず、mode として "msp"/"trust"/"entropy"/"random" を想定。
    """
    m = (mode or "random").lower()

    if m == "random":
        LOGGER.info("[round0] mode=random -> RandomSampler")
        return RandomSampler(seed=seed).select(state=state, scores=scores, budget=budget)

    if m in {"msp", "trust"}:
        # 低確信度を優先（昇順）
        LOGGER.info("[round0] mode=%s -> UncertaintySampler(by=%s, higher_is_better=False)", m, m)
        return UncertaintySampler(seed=seed, by=m, higher_is_better=False).select(
            state=state, scores=scores, budget=budget
        )

    if m in {"entropy", "energy"}:
        # 高不確実性を優先（降順）
        LOGGER.info("[round0] mode=%s -> UncertaintySampler(by=%s, higher_is_better=True)", m, m)
        return UncertaintySampler(seed=seed, by=m, higher_is_better=True).select(
            state=state, scores=scores, budget=budget
        )

    raise ValueError(f"[round0] unsupported mode={mode!r} (expected 'random'/'msp'/'trust'/'entropy')")


def _infer_score_mode_from_cfg_like(scores_hint_name: Optional[str]) -> Dict[str, Any]:
    """
    現段階では sampler 側が conf_key を知らないこともあるため、
    “超保守的” な推定関数として残している（将来は cfg から higher_is_better を渡すのが理想）。

    今回はデフォルトを「低確信度優先（昇順）」に寄せる。
    """
    # 将来: scores_hint_name が "entropy" 等なら True にするなど
    return {"higher_is_better": False}


# =============================================================================
# Registry factories (cfg -> sampler)
# =============================================================================


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
    return UncertaintySampler(seed=seed, by="entropy", higher_is_better=False)


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
    round0 = str(sampler_cfg.get("round0", "random"))
    return ClusteringSampler(seed=seed, k=k, round0=round0)


@register_sampler("hybrid")
def _factory_hybrid_sampler(*, cfg: Mapping[str, Any], sampler_cfg: Mapping[str, Any], seed: Optional[int] = None) -> BaseSampler:
    k = sampler_cfg.get("k", None)
    k = int(k) if k is not None else None
    ratio = float(sampler_cfg.get("sub_budget_ratio", 5.0))
    round0 = str(sampler_cfg.get("round0", "msp"))
    return HybridSampler(seed=seed, k=k, sub_budget_ratio=ratio, round0=round0)


# =============================================================================
# Public factory (cfg -> sampler)
# =============================================================================


def create_sampler(cfg: Mapping[str, Any]) -> BaseSampler:
    """
    cfg から Sampler を構築するファクトリ関数（フォールバック禁止・厳格）。

    対応する設定形式:
      1) 文字列形式
         al:
           sampler: "msp"

      2) 辞書形式
         al:
           sampler:
             name: "kmeans"
             k: 32
             round0: "msp"
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

    # 4) kmeans round0 fallback (explicit)
    s3 = ClusteringSampler(seed=42, k=2, round0="msp")
    print("kmeans round0 via msp:", s3.select(st, scores=scores, budget=2))

    # 5) kmeans with embeddings (simulate round>=1)
    st.round_index = 1
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
- 拡張性: round0 の mode を追加しやすい（_select_round0 を拡張）。
- 性能: numpy ベースで O(N) 前処理 + KMeans。大規模 N の場合は pipeline 側で候補圧縮推奨。
- 改善アクション（<=3）:
  1) Hybrid の higher_is_better を cfg から明示的に渡せるようにする（いまは保守的推定）。
  2) feature_ids→index の辞書生成をキャッシュできる（同一 round で複数回呼ぶなら）。
  3) round0 で scores が欠ける場合の診断ログをもう少し詳しくする。
"""
