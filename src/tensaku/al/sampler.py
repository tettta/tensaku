# /home/esakit25/work/tensaku/src/tensaku/al/sampler.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.al.sampler
@role  : Active Learning (AL) におけるサンプル選択ロジック（Sampler）の定義
@overview:
    - RandomSampler: ランダム選択
    - UncertaintySampler: 不確実性スコアに基づく選択 (Least Confidence等)
    - ClusteringSampler: 埋め込みベクトルを用いた K-Means クラスタリングによる多様性選択
    - HybridSampler: 不確実性で候補を絞り込み、クラスタリングで多様性を確保する選択
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd

from tensaku.al.state import ALState
from tensaku.experiments.layout import ExperimentLayout

LOGGER = logging.getLogger(__name__)

# Scikit-learn (KMeans) の利用可能性チェック
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    LOGGER.warning("Scikit-learn (KMeans) is not installed. Clustering and Hybrid samplers will be disabled.")
    SKLEARN_AVAILABLE = False


@dataclass
class BaseSampler:
    """AL サンプル選択のベースクラス。"""

    name: str = "base"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
    ) -> List[Any]:
        """プールから budget 件のサンプル ID を選択して返す。"""
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
    ) -> List[Any]:
        # scores は使用しない
        if budget <= 0 or state.n_pool == 0:
            return []
        
        # 決定論的な挙動のためにソートしてからシャッフル（state.pool_ids の順序依存を排除）
        pool_ids = sorted(list(state.pool_ids), key=str)
        
        # seed が指定されていれば再現可能
        self._rng.shuffle(pool_ids)
        
        return pool_ids[:budget]


@dataclass
class UncertaintySampler(BaseSampler):
    """
    Task から返された不確実性スコア (scores) に基づいてサンプルを選択する Sampler。
    スコアが高いもの（＝不確実性が高いと仮定）を優先して選択する。
    """
    name: str = "uncertainty"
    strategy: str = "least_confidence"  # 現状はソート順のラベルとしてのみ使用

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
    ) -> List[Any]:
        if budget <= 0 or state.n_pool == 0:
            return []
            
        if scores is None or not scores:
            LOGGER.warning("[UncertaintySampler] scores is empty. Fallback to Random.")
            rs = RandomSampler(seed=self.seed)
            return rs.select(state, scores, budget)

        pool_ids_set = set(state.pool_ids)
        
        # Pool に存在する ID のスコアのみ抽出
        # (Task から返される scores は全データを含んでいる可能性があるため)
        pool_scores = []
        for pid in state.pool_ids:
            if pid in scores:
                pool_scores.append((pid, scores[pid]))
        
        if not pool_scores:
            LOGGER.warning("[UncertaintySampler] No matching scores for pool items. Fallback to Random.")
            rs = RandomSampler(seed=self.seed)
            return rs.select(state, scores, budget)

        # スコアの降順（大きい順）にソート
        # 不確実性指標（Entropy, 1-MaxProb 等）は通常大きいほど不確実
        # もし小さいほど不確実な指標（Margin等）を使う場合は、Task側で符号反転するか、ここで制御が必要
        sorted_items = sorted(
            pool_scores, 
            key=lambda item: item[1], 
            reverse=True 
        )
        
        # 上位 budget 件を選択
        selected_ids = [item[0] for item in sorted_items[:budget]]
        return selected_ids


@dataclass
class ClusteringSampler(BaseSampler):
    """
    K-Means クラスタリングに基づく多様性サンプラー (Diversity Sampler)。
    前のラウンドで Task が出力した埋め込みファイル (pool_embs.npy) を使用する。
    """
    name: str = "clustering"
    out_dir: str = ""       # ExperimentLayout 構築用
    k: Optional[int] = None # クラスタ数 (Noneなら budget と同じ)

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
    ) -> List[Any]:
        if not SKLEARN_AVAILABLE:
            LOGGER.error("ClusteringSampler requires scikit-learn. Falling back to RandomSampler.")
            return RandomSampler(seed=self.seed).select(state, scores, budget)

        if budget <= 0 or state.n_pool == 0:
            return []

        # ラウンド0 (初期状態) では前ラウンドの埋め込みが存在しないためランダム
        if state.round_index == 0:
            LOGGER.info("[ClusteringSampler] Round 0: No embeddings available. Fallback to Random.")
            return RandomSampler(seed=self.seed).select(state, scores, budget)

        # 前ラウンド (round_index - 1) の推論結果パスを特定
        prev_round = state.round_index - 1
        layout = ExperimentLayout(cfg={"run": {"out_dir": self.out_dir}})
        # ExperimentLayout の API に従いパスを解決 (inferディレクトリ想定)
        # infer_pool.py は rounds/round_XXX/infer/pool_embs.npy に出力している
        infer_dir = layout.root / "rounds" / f"round_{prev_round:03d}" / "infer"
        
        pool_embs_path = infer_dir / "pool_embs.npy"
        pool_preds_path = infer_dir / "pool_preds.csv"

        try:
            if not pool_embs_path.exists() or not pool_preds_path.exists():
                raise FileNotFoundError(f"Embeddings or preds not found in {infer_dir}")

            # データのロード
            pool_embs = np.load(pool_embs_path)
            df_preds = pd.read_csv(pool_preds_path)
            
            # ID の整合性確認 (現在の pool と一致するものだけフィルタリング)
            # AL の過程で pool は減っていくため、前ラウンドの pool 全体ではなく
            # 「現在も pool に残っているデータ」の埋め込みを使う必要がある
            current_pool_ids = set(state.pool_ids)
            
            valid_indices = []
            valid_ids = []
            
            # preds.csv の 'id' 列を見てフィルタリング
            for idx, row_id in enumerate(df_preds["id"]):
                if row_id in current_pool_ids:
                    valid_indices.append(idx)
                    valid_ids.append(row_id)
            
            if not valid_indices:
                raise ValueError("No matching IDs between current pool and previous embeddings.")

            target_embs = pool_embs[valid_indices]
            target_ids = valid_ids

        except Exception as e:
            LOGGER.error(f"[ClusteringSampler] Data load failed: {e}. Fallback to Random.", exc_info=True)
            return RandomSampler(seed=self.seed).select(state, scores, budget)

        # K-Means 実行
        n_clusters = self.k if self.k is not None else budget
        n_clusters = min(n_clusters, len(target_embs))
        
        if n_clusters <= 0:
            return []

        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.seed, 
            n_init="auto", 
            max_iter=300
        )
        kmeans.fit(target_embs)
        
        # 各クラスタの中心に最も近いサンプルを選択
        # transform は各クラスタ中心までの距離を返す (N, n_clusters)
        distances = kmeans.transform(target_embs)
        selected_ids: List[Any] = []
        
        for cluster_id in range(n_clusters):
            # このクラスタに属するサンプルのインデックス
            indices_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
            if len(indices_in_cluster) == 0:
                continue
            
            # その中で、このクラスタ中心までの距離が最小のものを選ぶ
            dists = distances[indices_in_cluster, cluster_id]
            min_idx_relative = np.argmin(dists)
            min_idx_absolute = indices_in_cluster[min_idx_relative]
            
            selected_ids.append(target_ids[min_idx_absolute])
        
        # 重複排除して budget 件まで返す
        # (通常 K-Means では重複しないが念のため)
        unique_selected = list(dict.fromkeys(selected_ids))
        return unique_selected[:budget]


@dataclass
class HybridSampler(BaseSampler):
    """
    ハイブリッドサンプリング (Uncertainty + Diversity)。
    1. UncertaintySampler で pool から候補を絞り込む (sub_budget_ratio倍)。
    2. ClusteringSampler のロジックで候補の中から多様性を考慮して最終選択を行う。
    """
    name: str = "hybrid"
    out_dir: str = ""
    uncertainty_strategy: str = "least_confidence"
    sub_budget_ratio: float = 5.0  # 候補集合のサイズ倍率
    k_cluster: Optional[int] = None

    def select(
        self,
        state: ALState,
        scores: Optional[Mapping[Any, float]],
        budget: int,
    ) -> List[Any]:
        if not SKLEARN_AVAILABLE:
            LOGGER.error("HybridSampler requires scikit-learn. Falling back to RandomSampler.")
            return RandomSampler(seed=self.seed).select(state, scores, budget)

        if budget <= 0 or state.n_pool == 0:
            return []
            
        # 1. 不確実性による候補絞り込み
        #    scores が無い場合は Hybrid は成立しない -> Random へ
        if scores is None:
            LOGGER.warning("[HybridSampler] Scores is None. Fallback to Random.")
            return RandomSampler(seed=self.seed).select(state, scores, budget)

        candidate_budget = min(
            int(budget * self.sub_budget_ratio), 
            state.n_pool
        )
        
        # UncertaintySampler を委譲利用
        u_sampler = UncertaintySampler(seed=self.seed, strategy=self.uncertainty_strategy)
        candidate_ids = u_sampler.select(state, scores, candidate_budget)
        
        if not candidate_ids:
            return []

        # 2. 候補集合 (candidate_ids) に対してクラスタリング
        #    ClusteringSampler のロジックを再利用したいが、対象 ID が全 pool ではなく
        #    candidate_ids に限定されるため、同様のロジックをここで実行する。
        
        round_index = state.round_index
        if round_index == 0:
            # Round 0 は埋め込みがないため、Uncertainty の結果をそのまま返す (Top-K)
            return candidate_ids[:budget]

        prev_round = round_index - 1
        layout = ExperimentLayout(cfg={"run": {"out_dir": self.out_dir}})
        infer_dir = layout.root / "rounds" / f"round_{prev_round:03d}" / "infer"
        pool_embs_path = infer_dir / "pool_embs.npy"
        pool_preds_path = infer_dir / "pool_preds.csv"

        try:
            pool_embs = np.load(pool_embs_path)
            df_preds = pd.read_csv(pool_preds_path)
            
            # candidate_ids に含まれるものだけを抽出
            candidate_set = set(candidate_ids)
            valid_indices = []
            valid_ids = []
            
            for idx, row_id in enumerate(df_preds["id"]):
                if row_id in candidate_set:
                    valid_indices.append(idx)
                    valid_ids.append(row_id)
            
            if not valid_indices:
                # 何らかの理由でマッチしない場合は Uncertainty の結果を返す
                LOGGER.warning("[HybridSampler] No matching embeddings for candidates. Returning Top-K.")
                return candidate_ids[:budget]

            target_embs = pool_embs[valid_indices]
            target_ids = valid_ids

        except Exception as e:
            LOGGER.error(f"[HybridSampler] Data load failed: {e}. Fallback to Uncertainty Top-K.", exc_info=True)
            return candidate_ids[:budget]

        # 3. K-Means 実行
        n_clusters = self.k_cluster if self.k_cluster is not None else budget
        n_clusters = min(n_clusters, len(target_embs))
        
        if n_clusters <= 0:
            return []

        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.seed, 
            n_init="auto",
            max_iter=300
        )
        kmeans.fit(target_embs)
        
        distances = kmeans.transform(target_embs)
        selected_ids: List[Any] = []
        
        for cluster_id in range(n_clusters):
            indices_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
            if len(indices_in_cluster) == 0: continue
            
            dists = distances[indices_in_cluster, cluster_id]
            min_idx_rel = np.argmin(dists)
            min_idx_abs = indices_in_cluster[min_idx_rel]
            
            selected_ids.append(target_ids[min_idx_abs])
            
        unique_selected = list(dict.fromkeys(selected_ids))
        return unique_selected[:budget]


# ======================================================================
# Factory
# ======================================================================

def create_sampler(cfg: Mapping[str, Any]) -> BaseSampler:
    """cfg から Sampler インスタンスを構築するファクトリ関数。"""
    al_cfg = cfg.get("al", {})
    sampler_cfg = al_cfg.get("sampler", {}) if isinstance(al_cfg, Mapping) else {}
    
    # 名前解決
    name_raw = sampler_cfg.get("name", "random")
    name = str(name_raw).lower()

    # シード
    seed = al_cfg.get("seed")
    if not isinstance(seed, int):
        seed = None
        
    # 出力ディレクトリ (埋め込みロード用)
    out_dir = cfg.get("run", {}).get("out_dir", "")
    
    # 個別パラメータ
    strat = sampler_cfg.get("uncertainty_strategy", "least_confidence")
    k = sampler_cfg.get("k", None)
    ratio = sampler_cfg.get("sub_budget_ratio", 5.0)

    # インスタンス生成
    if name == "random":
        return RandomSampler(seed=seed)

    if name == "uncertainty":
        return UncertaintySampler(seed=seed, strategy=strat)

    if name == "clustering":
        return ClusteringSampler(seed=seed, out_dir=out_dir, k=k)

    if name == "hybrid":
        return HybridSampler(
            seed=seed, 
            out_dir=out_dir, 
            uncertainty_strategy=strat, 
            sub_budget_ratio=ratio,
            k_cluster=k
        )

    LOGGER.warning(
        f"Unknown sampler name '{name}'. Falling back to RandomSampler."
    )
    return RandomSampler(seed=seed)