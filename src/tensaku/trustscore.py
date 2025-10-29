# /home/esakit25/work/tensaku/src/tensaku/trustscore.py
"""
@module: tensaku.trustscore
@role: kNN-based Trust Score（分類用）— v1（従来）と v2（ロバスト拡張）の両対応
@inputs:
  - train_feats: np.ndarray (N_train, D)   埋め込み（CLSなど）
  - train_labels: np.ndarray (N_train,)    学習/開発ラベル（int: 0..C-1）
  - test_feats:  np.ndarray (N_test, D)    評価対象の埋め込み
  - test_pred:   np.ndarray (N_test,)      予測クラス（argmaxなど）
  - params (任意):
      version: Literal["v1","v2"] = "v1"
      metric:  Literal["cosine","euclidean"] = "cosine"
      k: int = 1                      # v1 用
      k_list: list[int] = [1,3,5]     # v2 用（複数kの集約）
      agg: Literal["median","trimmed_mean"] = "median"  # v2 用
      trim_q: float = 0.1             # v2: 両端除外率（0.0–0.45推奨）
      normalize: Optional[Literal["zscore"]] = None     # 学習統計で z-score
      eps: float = 1e-12
      return_components: bool = False # True で (trust, dp, dc) を返す
@outputs:
  - trust: np.ndarray (N_test,) in [0,1]
  - （option）dp, dc: np.ndarray (N_test,)  # 代表距離（v2は集約済み）
@notes:
  - 定義（基本）：dp = “予測クラスと同一クラス”への距離代表値、dc = “他クラス”への距離代表値。
                   trust = dc / (dp + dc) ∈ (0,1]（大きいほど「正しい」と信じやすい）
  - v1: k=1 の最短距離 min を用いる従来の Trust Score。
  - v2: 複数 k の集合（k_list）で、同/異クラスそれぞれ距離を計算し、agg（median か trimmed_mean）
        でロバスト集約。metric は cosine/欧距離を選択可。cosine のときは自動で L2 正規化。
  - 例外安全：同クラス学習データが空なら dp=+inf（→ trust≈0）、他クラスが空なら dc=+inf（→ trust≈1）。
  - registry: "trust" 名で tensaku.registry に登録（gate から `name: trust` で呼べる）。
"""

from __future__ import annotations

from typing import Optional, Literal, Tuple, Dict, Any, List
import numpy as np

# ---- Registry（任意依存） -------------------------------------------------------------------------
try:
    from .registry import register  # type: ignore
except Exception:
    def register(name: str, **_kw):
        def _decor(x): return x
        return _decor


# ---- ユーティリティ --------------------------------------------------------------------------------

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
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        aa = np.sum(A * A, axis=1, keepdims=True)  # (n,1)
        bb = np.sum(B * B, axis=1, keepdims=True).T  # (1,m)
        D2 = aa + bb - 2.0 * (A @ B.T)
        np.maximum(D2, 0.0, out=D2)
        return np.sqrt(D2, dtype=A.dtype)
    elif metric == "cosine":
        A_n = _l2_normalize_rows(A, eps)
        B_n = _l2_normalize_rows(B, eps)
        sim = A_n @ B_n.T
        # 数値誤差をクリップ
        np.clip(sim, -1.0, 1.0, out=sim)
        return 1.0 - sim
    else:
        raise ValueError(f"unsupported metric: {metric}")


def _kth_pool(values: np.ndarray, k: int) -> float:
    """最小 k 個のうち最後（= 第k番目）を返す。values は 1次元。"""
    k = max(1, int(k))
    if values.size == 0:
        return float("inf")
    if k >= values.size:
        return float(np.max(values))
    # パーシャルソートで O(n)
    idx = np.argpartition(values, k - 1)
    kth = values[idx[k - 1]]
    return float(kth)


def _robust_aggregate_per_k(d_arr: np.ndarray, k_list: List[int], agg: str, trim_q: float) -> float:
    """
    d_arr: 近傍距離（昇順でなくてOK）
    k_list: 例 [1,3,5] なら、各kの「最小kのうちの最大（第k番目）」を求め、その集合をロバスト集約。
    """
    vals = []
    for k in k_list:
        vals.append(_kth_pool(d_arr, k))
    arr = np.asarray(vals, dtype=float)
    if agg == "median":
        return float(np.median(arr))
    elif agg == "trimmed_mean":
        if not (0.0 <= trim_q < 0.5):
            raise ValueError("trim_q must be in [0.0, 0.5)")
        lo = int(np.floor(trim_q * arr.size))
        hi = int(np.ceil((1.0 - trim_q) * arr.size))
        if hi <= lo:
            return float(np.mean(arr))
        arr_s = np.sort(arr)
        return float(arr_s[lo:hi].mean())
    else:
        raise ValueError(f"unsupported agg: {agg}")


def _class_index_map(labels: np.ndarray) -> Dict[int, np.ndarray]:
    """クラス -> 学習インデックス配列"""
    lab = labels.astype(int, copy=False)
    classes = np.unique(lab)
    out: Dict[int, np.ndarray] = {}
    for c in classes:
        idx = np.nonzero(lab == c)[0]
        out[int(c)] = idx
    return out


# ---- メイン実装 -----------------------------------------------------------------------------------

@register("trust")
class TrustScorer:
    """
    @role: Trust Score 推定器（v1/v2両対応）
    - fit(): 学習統計（zscore用）とクラス別インデックスを準備
    - score(): 入力 test_feats/test_pred に対して trust を返す
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
    ):
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

        # cosine の場合は内部で L2 正規化をかける（距離の安定性向上）
        if self.metric == "cosine":
            X = _l2_normalize_rows(X, eps=self.eps)

        self.train_feats_ = X
        self.train_labels_ = y
        self.class2idx_ = _class_index_map(y)
        return self

    def score(
        self,
        test_feats: np.ndarray,
        test_pred: np.ndarray,
        *,
        return_components: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """test_feats/test_pred に対して trust を返す。"""
        self._check_fitted()
        Xtr = self.train_feats_
        ytr = self.train_labels_
        assert Xtr is not None and ytr is not None

        Xt = np.asarray(test_feats, dtype=float)
        yp = np.asarray(test_pred, dtype=int)

        # 同じ前処理を適用
        if self.normalize == "zscore" and (self.mu_ is not None):
            Xt = _zscore_transform(Xt, self.mu_, self.sd_)  # type: ignore[arg-type]
        if self.metric == "cosine":
            Xt = _l2_normalize_rows(Xt, eps=self.eps)

        N = Xt.shape[0]
        dp_arr = np.zeros(N, dtype=float)
        dc_arr = np.zeros(N, dtype=float)

        # クラスごとに学習集合を参照
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
            # 最小値（k=1）か、k指定があれば第k番目
            k = max(1, self.k)
            if k == 1:
                return float(np.min(d)) if d.size else float("inf")
            return _kth_pool(d, k)
        elif self.version == "v2":
            # k_list をロバスト集約
            return _robust_aggregate_per_k(d, self.k_list, self.agg, self.trim_q)
        else:
            raise ValueError(f"unsupported version: {self.version}")

    def _check_fitted(self) -> None:
        if self.train_feats_ is None or self.train_labels_ is None:
            raise RuntimeError("TrustScorer is not fitted. Call fit(train_feats, train_labels) first.")


# ---- 直接関数インタフェース（簡便） ----------------------------------------------------------------

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


# ---- セルフテスト ----------------------------------------------------------------------------------

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
    # 疑似予測（実際はモデルの argmax を入れる）
    y_pred = np.array([0]*(Nte//2) + [1]*(Nte//2), dtype=int)

    # v1
    ts_v1 = TrustScorer(version="v1", metric="cosine", k=1).fit(Xtr, ytr).score(Xte, y_pred)
    print("[v1] trust (head):", np.round(ts_v1[:6], 3))

    # v2（推奨設定例）
    ts_v2 = TrustScorer(version="v2", metric="cosine", k_list=[1,3,5], agg="median").fit(Xtr, ytr).score(Xte, y_pred)
    print("[v2] trust (head):", np.round(ts_v2[:6], 3))

    # コンポーネント確認
    ts_v2_c, dp, dc = TrustScorer(version="v2", metric="cosine", k_list=[1,3,5]).fit(Xtr, ytr).score(
        Xte, y_pred, return_components=True
    )
    print("[v2] dp/dc (head):", np.round(dp[:3], 3), np.round(dc[:3], 3))
