import numpy as np
from .base import Sampler
from ..registry import register

@register("hybrid")
class HybridSampler:
    def __init__(self, alpha: float = 0.7, k_center: int = 200, uncertainty: str = "entropy"):
        self.alpha = alpha
        self.k_center = k_center
        self.uncertainty = uncertainty

    def select(self, pool_rows, K, embeddings, uncertainties, state):
        n = len(pool_rows)
        if n == 0 or K <= 0:
            return []
        rng = np.random.default_rng(state.get("seed", 42))
        # 不確実性スコア（高いほど優先）
        rank_unc = np.argsort(-uncertainties)
        score_unc = np.zeros(n)
        score_unc[rank_unc] = np.linspace(1.0, 0.0, n, endpoint=True)
        # 多様性（k-centerの粗い近似：ランダム代表との最短距離）
        reps_idx = rng.choice(n, size=min(self.k_center, n), replace=False)
        dist = np.linalg.norm(embeddings[:, None, :] - embeddings[reps_idx][None, :, :], axis=-1).min(axis=1)
        # 0-1 正規化
        score_div = (dist - dist.min()) / (dist.ptp() + 1e-8)
        score = self.alpha * score_unc + (1 - self.alpha) * score_div
        topk = np.argsort(-score)[:K]
        return [pool_rows[i]["id"] for i in topk]
