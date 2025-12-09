# -*- coding: utf-8 -*-
"""
@module     tensaku.trustscore
@role       kNNベースの Trust Score 推定器（v1=従来, v2=ロバスト集約）。fit→score と 1-shot の両APIを提供。
@inputs
  - train_feats: np.ndarray (N_train, D)      # 埋め込み（CLSなど）
  - train_labels: np.ndarray (N_train,)       # 学習/開発ラベル（int: 0..C-1）
  - test_feats:  np.ndarray (N_test, D)       # 評価対象の埋め込み
  - test_pred:   np.ndarray (N_test,)         # 予測クラス（argmaxなど）
  - params (任意):
      version: Literal["v1","v2"]="v1"       # v1: 単一k, v2: k_list＋ロバスト集約
      metric:  Literal["cosine","euclidean"]="cosine"
      k: int=1                               # v1 用
      k_list: list[int]=[1,3,5]              # v2 用
      agg: Literal["median","trimmed_mean"]="median"
      trim_q: float=0.1                      # v2 の両端除外率（0.0–0.45推奨）
      normalize: Optional[Literal["zscore"]]=None  # 学習統計で z-score するか
      eps: float=1e-12
      robust: Optional[bool]=None            # 互換用：True なら v2 推奨設定を既定化
      return_components: bool=False          # True で (trust, dp, dc) を返す
@outputs
  - trust: np.ndarray (N_test,) in [0,1]
  - （option）dp, dc: np.ndarray (N_test,)    # 代表距離（v2は集約済み）
@contracts
  - クラスラベルは 0..C-1 の整数想定。cosine のとき内部で L2 正規化（安定性向上）。
  - 例外安全：同クラス学習データが空なら dp=+inf（→ trust≈0）、他クラスが空なら dc=+inf（→ trust≈1）。
@api
  - 2段階:   TrustScorer(...).fit(train_feats, train_labels).score(test_feats, test_pred, return_components=False)
  - 1-shot:   TrustScorer(...).score(train_feats, train_labels, test_feats, test_pred, return_components=False)
  - 関数版:   trustscore(train_feats, train_labels, test_feats, test_pred, **kwargs)
@notes
  - registry 名は "trust"（gate/registry から `name: trust` で呼び出し可能）。
  - `robust=True` を指定した場合、version="v2", k_list=[1,3,5], agg="median" を既定化（後方互換フラグ）。
"""

from __future__ import annotations

from typing import Optional, Literal, Tuple, Dict, Any, List, overload
import numpy as np

# ---- Registry（任意依存：無ければno-op） -----------------------------------------------------------
try:
    from tensaku.registry import register  # type: ignore
except Exception:  # pragma: no cover
    def register(name: str, **_kw):
        def _decor(x): return x
        return _decor


# ---- ユーティリティ ------------------------------------------------------------------------------
def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def _zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=False)
    sd = x.std(axis=0, keepdims=False)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd

def _zscore_transform(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd

def _pairwise_distances(A: np.ndarray, B: np.ndarray, metric: str = "cosine", eps: float = 1e-12) -> np.ndarray:
    """
    A: (n, d), B: (m, d) -> D: (n, m)
    cosine の場合は 1 - cos_sim（∈[0,2]）を距離として返す。
    """
    if metric == "euclidean":
        aa = np.sum(A * A, axis=1, keepdims=True)          # (n,1)
        bb = np.sum(B * B, axis=1, keepdims=True).T        # (1,m)
        D2 = aa + bb - 2.0 * (A @ B.T)
        np.maximum(D2, 0.0, out=D2)
        return np.sqrt(D2, dtype=A.dtype)
    elif metric == "cosine":
        A_n = _l2_normalize_rows(A, eps)
        B_n = _l2_normalize_rows(B, eps)
        sim = A_n @ B_n.T
        np.clip(sim, -1.0, 1.0, out=sim)
        return 1.0 - sim
    else:
        raise ValueError(f"unsupported metric: {metric}")

def _kth_pool(values: np.ndarray, k: int) -> float:
    """最小 k 個のうち第k番目を返す（values は1次元）。"""
    k = max(1, int(k))
    if values.size == 0:
        return float("inf")
    if k >= values.size:
        return float(np.max(values))
    idx = np.argpartition(values, k - 1)
    return float(values[idx[k - 1]])

def _robust_aggregate_per_k(d_arr: np.ndarray, k_list: List[int], agg: str, trim_q: float) -> float:
    """
    例: k_list=[1,3,5] なら、各kで第k番目の距離を取り、その集合をロバスト集約。
    """
    vals = np.array([_kth_pool(d_arr, k) for k in k_list], dtype=float)
    if agg == "median":
        return float(np.median(vals))
    elif agg == "trimmed_mean":
        if not (0.0 <= trim_q < 0.5):
            raise ValueError("trim_q must be in [0.0, 0.5)")
        lo = int(np.floor(trim_q * vals.size))
        hi = int(np.ceil((1.0 - trim_q) * vals.size))
        if hi <= lo:
            return float(np.mean(vals))
        vals.sort()
        return float(vals[lo:hi].mean())
    else:
        raise ValueError(f"unsupported agg: {agg}")

def _class_index_map(labels: np.ndarray) -> Dict[int, np.ndarray]:
    lab = labels.astype(int, copy=False)
    classes = np.unique(lab)
    return {int(c): np.nonzero(lab == c)[0] for c in classes}


# ---- メイン実装 -----------------------------------------------------------------------------------
@register("trust")
class TrustScorer:
    """
    @role  Trust Score 推定器（v1/v2両対応）
    - fit(): 学習統計（zscore用）とクラス別インデックスを準備
    - score(): 
        * 二段階: score(test_feats, test_pred, return_components=False)
        * 一発  : score(train_feats, train_labels, test_feats, test_pred, return_components=False)
    """

    name = "trust"

    def __init__(
        self,
        *,
        version: Literal["v1", "v2"] = "v1",
        metric: Literal["cosine", "euclidean"] = "cosine",
        k: int = 1,
        k_list: Optional[List[int]] = None,
        agg: Literal["median", "trimmed_mean"] = "median",
        trim_q: float = 0.1,
        normalize: Optional[Literal["zscore"]] = None,
        eps: float = 1e-12,
        robust: Optional[bool] = None,  # 後方互換：robust=True で v2 推奨既定を有効化
    ):
        # robust フラグで推奨設定を一括適用（他指定があればそれを優先）
        if robust is True:
            version = "v2" if version == "v1" else version
            if k_list is None:
                k_list = [1, 3, 5]
            agg = "median" if agg not in ("median", "trimmed_mean") else agg

        self.version = version
        self.metric = metric
        self.k = int(k)
        self.k_list = k_list if k_list is not None else [1, 3, 5]
        self.agg = agg
        self.trim_q = float(trim_q)
        self.normalize = normalize
        self.eps = float(eps)

        # 学習時に決まる
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.train_feats_: Optional[np.ndarray] = None
        self.train_labels_: Optional[np.ndarray] = None
        self.class2idx_: Dict[int, np.ndarray] = {}

    # -------------------- API --------------------
    def fit(self, train_feats: np.ndarray, train_labels: np.ndarray) -> "TrustScorer":
        X = np.asarray(train_feats, dtype=float)
        y = np.asarray(train_labels, dtype=int)

        if self.normalize == "zscore":
            mu, sd = _zscore_fit(X)
            X = _zscore_transform(X, mu, sd)
            self.mu_, self.sd_ = mu, sd

        if self.metric == "cosine":
            X = _l2_normalize_rows(X, eps=self.eps)

        self.train_feats_ = X
        self.train_labels_ = y
        self.class2idx_ = _class_index_map(y)
        return self

    # 2-way 互換API： (test_feats, test_pred) または (train_feats, train_labels, test_feats, test_pred)
    def score(
        self,
        *args: Any,
        return_components: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(args) == 2:
            test_feats, test_pred = args
            return self._score_core(np.asarray(test_feats, float),
                                    np.asarray(test_pred, int),
                                    return_components=return_components)
        elif len(args) == 4:
            train_feats, train_labels, test_feats, test_pred = args
            self.fit(np.asarray(train_feats, float), np.asarray(train_labels, int))
            return self._score_core(np.asarray(test_feats, float),
                                    np.asarray(test_pred, int),
                                    return_components=return_components)
        else:
            raise TypeError("score() expects (test_feats, test_pred) or (train_feats, train_labels, test_feats, test_pred)")

    # 実体
    def _score_core(
        self,
        test_feats: np.ndarray,
        test_pred: np.ndarray,
        *,
        return_components: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_fitted()
        Xtr = self.train_feats_
        ytr = self.train_labels_
        assert Xtr is not None and ytr is not None

        Xt = test_feats
        yp = test_pred

        # 同じ前処理を適用
        if self.normalize == "zscore" and (self.mu_ is not None):
            Xt = _zscore_transform(Xt, self.mu_, self.sd_)  # type: ignore[arg-type]
        if self.metric == "cosine":
            Xt = _l2_normalize_rows(Xt, eps=self.eps)

        N = Xt.shape[0]
        dp_arr = np.zeros(N, dtype=float)
        dc_arr = np.zeros(N, dtype=float)

        all_idx = np.arange(Xtr.shape[0])

        for i in range(N):
            c = int(yp[i])
            same_idx = self.class2idx_.get(c, np.array([], dtype=int))
            other_idx = np.setdiff1d(all_idx, same_idx, assume_unique=True)

            # 同クラス距離
            if same_idx.size > 0:
                Dp = _pairwise_distances(Xt[i:i+1, :], Xtr[same_idx, :], metric=self.metric, eps=self.eps).ravel()
                dp = self._aggregate_dist(Dp)
            else:
                dp = float("inf")

            # 他クラス距離
            if other_idx.size > 0:
                Dc = _pairwise_distances(Xt[i:i+1, :], Xtr[other_idx, :], metric=self.metric, eps=self.eps).ravel()
                dc = self._aggregate_dist(Dc)
            else:
                dc = float("inf")

            dp_arr[i] = dp
            dc_arr[i] = dc

        trust = dc_arr / (dp_arr + dc_arr + self.eps)
        trust = np.clip(trust, 0.0, 1.0)

        if return_components:
            return trust, dp_arr, dc_arr
        return trust

    # -------------------- 内部ヘルパ --------------------
    def _aggregate_dist(self, d: np.ndarray) -> float:
        if self.version == "v1":
            k = max(1, self.k)
            if k == 1:
                return float(np.min(d)) if d.size else float("inf")
            return _kth_pool(d, k)
        elif self.version == "v2":
            return _robust_aggregate_per_k(d, self.k_list, self.agg, self.trim_q)
        else:
            raise ValueError(f"unsupported version: {self.version}")

    def _check_fitted(self) -> None:
        if self.train_feats_ is None or self.train_labels_ is None:
            raise RuntimeError("TrustScorer is not fitted. Call fit(train_feats, train_labels) or use 1-shot score(...).")


# ---- 直接関数インタフェース ------------------------------------------------------------------------
def trustscore(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_pred: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """
    1-shot の簡易版。例：
      trust = trustscore(Xtr, ytr, Xte, y_pred, version="v2", metric="cosine", k_list=[1,3,5], agg="median")
    """
    scorer = TrustScorer(**kwargs).fit(train_feats, train_labels)
    return scorer.score(test_feats, test_pred, return_components=False)  # type: ignore[return-value]


# ---- セルフテスト（軽量） --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # 二つのガウスクラスタ（0/1）
    Ntr, Nte, D = 120, 40, 16
    X0 = rng.normal(loc=0.0, scale=1.0, size=(Ntr//2, D))
    X1 = rng.normal(loc=2.5, scale=1.0, size=(Ntr//2, D))
    Xtr = np.vstack([X0, X1])
    ytr = np.array([0]*(Ntr//2) + [1]*(Ntr//2), dtype=int)

    Xt0 = rng.normal(loc=0.0, scale=1.0, size=(Nte//2, D))
    Xt1 = rng.normal(loc=2.5, scale=1.0, size=(Nte//2, D))
    Xte = np.vstack([Xt0, Xt1])
    y_pred = np.array([0]*(Nte//2) + [1]*(Nte//2), dtype=int)

    # v1（二段階）
    ts_v1 = TrustScorer(version="v1", metric="cosine", k=1).fit(Xtr, ytr).score(Xte, y_pred)
    print("[v1] trust (head):", np.round(ts_v1[:6], 3))

    # v2（推奨）— 二段階
    ts_v2 = TrustScorer(version="v2", metric="cosine", k_list=[1,3,5], agg="median").fit(Xtr, ytr).score(Xte, y_pred)
    print("[v2-2step] trust (head):", np.round(ts_v2[:6], 3))

    # v2（推奨）— 1-shot
    ts_v2_1shot = TrustScorer(version="v2", metric="cosine", k_list=[1,3,5]).score(Xtr, ytr, Xte, y_pred)
    print("[v2-1shot] trust (head):", np.round(ts_v2_1shot[:6], 3))
