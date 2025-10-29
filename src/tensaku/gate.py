# /home/esakit25/work/tensaku/src/tensaku/gate.py
"""
@module: tensaku.gate
@role: 薄いオーケストレータ。Confidence / Calibration /（任意で Trust）を呼び出し、HITLゲートを決定する。
@notes:
  - devで T と τ を推定 → pool/testに固定適用（AL×HITLの原則）
  - 直接実行（python gate.py）とモジュール実行（python -m tensaku.gate）の両方に対応
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import os, csv, sys
import numpy as np

# --- import shim: 直接実行でも tensaku.* を解決 ---
try:
    # パッケージ実行（推奨）
    from .calibration import TemperatureScaler, _softmax_np, nll_from_logits, ece  # type: ignore
    from . import registry as R  # type: ignore
    try:
        from .confidence import create_estimator as _create_estimator  # type: ignore
    except Exception:
        _create_estimator = None
except Exception:
    # 直接実行時のフォールバック
    ROOT = os.path.dirname(os.path.dirname(__file__))  # /.../src/tensaku -> /.../src
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from tensaku.calibration import TemperatureScaler, _softmax_np, nll_from_logits, ece
    from tensaku import registry as R
    try:
        from tensaku.confidence import create_estimator as _create_estimator
    except Exception:
        _create_estimator = None

# ====== Public API =================================================================================

def fit_temperature_on_dev(dev_logits: np.ndarray, dev_labels: np.ndarray,
                           T_min: float = 0.5, T_max: float = 3.0, T_step: float = 0.05):
    scaler = TemperatureScaler()
    T = scaler.fit(dev_logits, dev_labels, T_min=T_min, T_max=T_max, T_step=T_step)
    return scaler, T

def compute_confidences(logits: np.ndarray, estimators: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for econf in estimators:
        name = econf.get("name")
        if not name: continue
        kwargs = {k: v for k, v in econf.items() if k != "name"}
        est = _create_estimator(name, **kwargs) if _create_estimator is not None else R.create(name, **kwargs)
        try: val = est(logits=logits)
        except TypeError: val = est(logits)
        val = np.asarray(val)
        if val.ndim != 1: raise ValueError(f"confidence '{name}' must return shape (N,), got {val.shape}")
        out[str(name)] = val
    return out

def find_tau_for_constraint(dev_pred: np.ndarray, dev_true: np.ndarray, dev_conf: np.ndarray, *,
                            metric: str = "cse", abs_err: int = 2, epsilon: float = 0.05,
                            higher_is_better: bool = True, steps: int = 201):
    idx = np.argsort(dev_conf)[::-1] if higher_is_better else np.argsort(dev_conf)
    conf_sorted = dev_conf[idx]; pred_sorted = dev_pred[idx]; true_sorted = dev_true[idx]
    N = len(conf_sorted); best_cov = 0.0; best_tau: Optional[float] = None
    for k in np.linspace(1, N, num=min(steps, N), dtype=int):
        m = np.zeros(N, dtype=bool); m[:k] = True
        cse = np.mean(np.abs(pred_sorted[m] - true_sorted[m]) >= abs_err)
        cov = m.mean()
        if cse <= epsilon and cov > best_cov + 1e-12:
            best_cov = cov; best_tau = float(conf_sorted[k-1])
    return best_tau, float(best_cov)

def decide_mask(conf: np.ndarray, tau: Optional[float], *, higher_is_better: bool = True) -> np.ndarray:
    if tau is None: return np.zeros_like(conf, dtype=bool)
    return (conf >= tau) if higher_is_better else (conf <= tau)

def save_gate_csv(path: str, y_true: Optional[np.ndarray], y_pred: np.ndarray,
                  conf: Dict[str, np.ndarray], mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(conf.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["y_true", "y_pred"] + [f"conf_{k}" for k in keys] + ["auto_accept"]
        w.writerow(header)
        N = y_pred.shape[0]
        yt = y_true if y_true is not None else np.array([""] * N, dtype=object)
        for i in range(N):
            row = [yt[i], y_pred[i]] + [float(conf[k][i]) for k in keys] + [bool(mask[i])]
            w.writerow(row)

def score_trust(train_feats: np.ndarray, train_labels: np.ndarray, test_feats: np.ndarray, test_pred: np.ndarray, **trust_kwargs: Any) -> np.ndarray:
    try:
        from .trustscore import TrustScorer  # 遅延import
    except Exception:
        from tensaku.trustscore import TrustScorer
    scorer = TrustScorer(**trust_kwargs).fit(train_feats, train_labels)
    return scorer.score(test_feats, test_pred, return_components=False)  # type: ignore[return-value]

# ====== Self test / Demo =========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    Nd, Nt, C = 180, 120, 6
    dev_logits = rng.normal(size=(Nd, C)) * 1.4
    test_logits = rng.normal(size=(Nt, C)) * 1.4
    dev_labels = rng.integers(0, C, size=Nd)

    scaler, T = fit_temperature_on_dev(dev_logits, dev_labels, 0.5, 3.0, 0.05)
    dev_logits_t = scaler.transform_logits(dev_logits)
    test_logits_t = scaler.transform_logits(test_logits)

    ests = [{"name": "msp"}, {"name": "entropy"}]
    dev_conf_map = compute_confidences(dev_logits_t, ests)
    test_conf_map = compute_confidences(test_logits_t, ests)

    dev_probs = _softmax_np(dev_logits_t); dev_pred = dev_probs.argmax(axis=1)
    tau, cov = find_tau_for_constraint(dev_pred, dev_labels, dev_conf_map["msp"],
                                       metric="cse", abs_err=2, epsilon=0.05, higher_is_better=True)
    mask = decide_mask(test_conf_map["msp"], tau, higher_is_better=True)

    BASE = os.environ.get("TensakuBase") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_csv = os.path.join(BASE, "outputs", "gate_decision.csv")
    test_pred = _softmax_np(test_logits_t).argmax(axis=1)
    save_gate_csv(out_csv, None, test_pred, test_conf_map, mask)

    summary = {k: {"dev_mean": float(vd.mean()), "test_mean": float(test_conf_map[k].mean())}
               for k, vd in dev_conf_map.items()}
    print(f"[gate] T*={T:.2f}, tau={None if tau is None else round(tau,4)}, cov_dev={round(cov,3)}")
    for k, s in summary.items():
        print(f"[{k}] dev_mean={s['dev_mean']:.3f} test_mean={s['test_mean']:.3f}")
    print("[gate] saved:", out_csv)
