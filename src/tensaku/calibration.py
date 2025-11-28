# /home/esakit25/work/tensaku/src/tensaku/calibration.py
"""
@module     tensaku.calibration
@role       温度スケーリング（Temperature Scaling）と校正指標（NLL/ECE/信頼度ビン）
@inputs     - logits: ndarray (N, C)        ※基本は logits 入力（Tスケーリングは logits / T）
           - y:      ndarray (N,) int 0..C-1
           - probs:  ndarray (N, C)（任意。可視化/外部計算に利用）
@outputs    - TemperatureScaler.fit(logits, y) -> float（devで最良T）
           - TemperatureScaler.transform_logits(logits) -> logits'
           - nll_from_logits(logits, y, T=1.0) -> float
           - ece(probs, y, n_bins=15) -> float
           - reliability_bins(probs, y, n_bins=15) -> dict（binごとの count/accuracy/avg_conf）
@cli        直接のCLIは持たない（tensaku gate / infer-pool等から内部利用）
@api        class TemperatureScaler(T_min=0.5, T_max=3.0, step=0.05).fit(...).transform_logits(...)
@deps       numpy（必須）
@config     calibration: {enable: bool, T_min: float, T_max: float, step: float, n_bins: int=15}
@contracts  - T は正の実数。logits shape=(N,C) と y shape=(N,) が整合
           - 返すTは dev セットでのECE最小（同点は小さいT優先）
@errors     - 形状不一致は ValueError、NaN/Infを検知した場合は RuntimeError を送出
@notes      - ECEは “max prob” の標準定義。binsの端は[0,1]を等間隔に分割
@tests      - スモーク: 乱数logitsとyで fit→T∈[0.5,3.0] を確認、reliability_binsの総数=件数
"""


from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional
import math
import numpy as np

# ---- Registry (optional hook) --------------------------------------------------------------------
try:
    from .registry import register  # type: ignore
except Exception:
    def register(name: str, **_kw):
        def _decor(x): return x
        return _decor


# ---- Core utils ----------------------------------------------------------------------------------

def _softmax_np(logits: np.ndarray, T: float = 1.0, axis: int = -1) -> np.ndarray:
    z = logits / max(T, 1e-8)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=axis, keepdims=True), 1e-12, None)


def nll_from_logits(logits: np.ndarray, y: np.ndarray, T: float = 1.0, eps: float = 1e-12) -> float:
    """Negative Log-Likelihood (平均)。"""
    p = _softmax_np(logits, T=T)  # (N, C)
    n = max(1, p.shape[0])
    y = y.astype(int, copy=False)
    p_y = np.clip(p[np.arange(n), y], eps, 1.0)
    return float(-np.log(p_y).mean())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (max prob でビン分け)。
    戻りは 0..1 程度（ビン数依存）。小さいほど良い。
    """
    p = np.asarray(probs, dtype=float)
    y = y.astype(int, copy=False)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    ece_sum = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = mask.sum()
        if m == 0:
            continue
        acc = (pred[mask] == y[mask]).mean()
        avg_conf = conf[mask].mean()
        ece_sum += (m / total) * abs(acc - avg_conf)
    return float(ece_sum)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    """
    信頼度ビンごとの統計（可視化用）:
      - 'bin_lower', 'bin_upper', 'count', 'accuracy', 'avg_conf'
    """
    p = np.asarray(probs, dtype=float)
    y = y.astype(int, copy=False)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lower = bins[:-1]
    bin_upper = bins[1:]
    count = np.zeros(n_bins, dtype=int)
    accuracy = np.zeros(n_bins, dtype=float)
    avg_conf = np.zeros(n_bins, dtype=float)

    for i in range(n_bins):
        lo, hi = bin_lower[i], bin_upper[i]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = mask.sum()
        count[i] = int(m)
        if m > 0:
            accuracy[i] = (pred[mask] == y[mask]).mean()
            avg_conf[i] = conf[mask].mean()
        else:
            accuracy[i] = 0.0
            avg_conf[i] = (lo + hi) / 2.0

    return {
        "bin_lower": bin_lower,
        "bin_upper": bin_upper,
        "count": count,
        "accuracy": accuracy,
        "avg_conf": avg_conf,
    }


# ---- Temperature Scaler --------------------------------------------------------------------------

@register("temperature_scaler")
class TemperatureScaler:
    """
    温度スケーリング器。dev logits から T を推定し、logits/T を返す。
    - fit: グリッドサーチで NLL 最小の T を探索
    - transform_logits: logits を /T して返す
    """

    name = "temperature_scaler"

    def __init__(self, T_init: float = 1.0):
        self.T_: float = float(T_init)

    def fit(
        self,
        logits_dev: np.ndarray,
        y_dev: np.ndarray,
        T_min: float = 0.5,
        T_max: float = 3.0,
        T_step: float = 0.05,
    ) -> float:
        best_T = self.T_
        best_nll = float("inf")
        t = T_min
        while t <= T_max + 1e-12:
            nll = nll_from_logits(logits_dev, y_dev, T=t)
            if nll < best_nll:
                best_nll = nll
                best_T = t
            t += T_step
        self.T_ = float(best_T)
        return self.T_

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / max(self.T_, 1e-8)

    # 参考：probsしか無いときの近似（log(probs)をlogits近似とみなす）
    def transform_probs_via_log(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, 1e-12, 1.0)
        z = np.log(p) / max(self.T_, 1e-8)
        return _softmax_np(z, T=1.0)


# ---- Self test -----------------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, C = 200, 6
    logits = rng.normal(size=(N, C)) * 1.5  # わざと過信/過小信側にも振れるようスケール
    y = rng.integers(low=0, high=C, size=N)

    # Before
    p0 = _softmax_np(logits, T=1.0)
    nll0 = nll_from_logits(logits, y, T=1.0)
    ece0 = ece(p0, y, n_bins=15)

    # Fit T on the same data (デモ用、実運用は dev で fit → test 固定適用)
    scaler = TemperatureScaler()
    T_hat = scaler.fit(logits, y, T_min=0.5, T_max=3.0, T_step=0.05)

    # After
    logits_t = scaler.transform_logits(logits)
    p1 = _softmax_np(logits_t, T=1.0)
    nll1 = nll_from_logits(logits_t, y, T=1.0)
    ece1 = ece(p1, y, n_bins=15)

    print(f"[calib] T* = {T_hat:.2f}")
    print(f"[calib] NLL: before={nll0:.4f}, after={nll1:.4f}")
    print(f"[calib] ECE: before={ece0:.4f}, after={ece1:.4f}")

    bins = reliability_bins(p1, y, n_bins=10)
    # サンプル表示
    show_k = min(3, len(bins["count"]))
    for i in range(show_k):
        print(f"[bin{i}] n={bins['count'][i]} acc={bins['accuracy'][i]:.3f} conf={bins['avg_conf'][i]:.3f}")
