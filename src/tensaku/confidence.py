# /home/esakit25/work/tensaku/src/tensaku/confidence.py
"""
@module: tensaku.confidence
@role: Confidence estimators (MSP / entropy / energy / margin / MC-Dropout) with a lightweight registry hook.
@inputs:
  - logits: ndarray | torch.Tensor, shape (N, C)  ※一部推定器（MC-Dropout）は model & dataloader を直接受け取るユーティリティを提供
  - probs:  ndarray | torch.Tensor, shape (N, C)  ※logits があれば内部で softmax(T) を適用
  - temperature (T): float, optional       ※温度スケーリング後の信頼度算出に利用
@outputs:
  - conf: ndarray, shape (N,)  各サンプルの信頼度（大きいほど確信が高い想定）
@cli: （直接のCLIは持たない。tensaku gate / tensaku infer-pool から内部利用）
@notes:
  - すべて **純関数的** に扱える logits/probs ベースの推定器と、**再推論を伴う** MC-Dropout を同一レイヤに整理。
  - 出力スケールは以下を原則とする：
      MSP, 1-Entropy(norm), Energy(sigmoid), Margin(sigmoid) は **[0,1]** に正規化（単調変換）。
    ※ Energy と Margin はランク保存を優先し、単調写像（sigmoid）で 0–1 に収める。
  - registry への登録名： "msp", "entropy", "energy", "margin", "mc_dropout"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Iterable, Dict, Any, Union

import math
import numpy as np
import numpy.typing as npt
from contextlib import nullcontext

# ---- Optional torch dependency -------------------------------------------------------------------

try:
    import torch
    from torch import nn, Tensor as TorchTensor
except Exception:  # pragma: no cover
    torch = None
    nn = None

    class _TorchTensorStub:  # Pylance/typing 用のフォールバック
        pass

    TorchTensor = _TorchTensorStub  # type: ignore[misc,assignment]

# ---- Registry hook (soft dependency) -------------------------------------------------------------

try:
    # 期待API: @register("name")(cls_or_fn)
    from .registry import register  # type: ignore
except Exception:  # フォールバック（registry未実装でも壊れないように）
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
    # 最大値と2番目を高速に求める
    top1_idx = np.argmax(probs, axis=1)
    top1 = probs[np.arange(probs.shape[0]), top1_idx]
    # 2番目を求めるために最大位置を -inf にして再取得
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
    # 数値安定化
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-k * x))


# ---- ベースクラス --------------------------------------------------------------------------------

class ConfidenceEstimator:
    """推定器の最小インタフェース。"""

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
    """Maximum Softmax Probability."""

    name = "msp"

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        return p.max(axis=1)  # [0,1]


# ---- Entropy (1 - normalized entropy) ------------------------------------------------------------

@register("entropy")
class OneMinusEntropy(ConfidenceEstimator):
    """1 - H(p)/log(C): 0..1（1が高確信）"""

    name = "entropy"

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        H = _entropy_np(p)
        C = p.shape[1]
        H_max = math.log(max(C, 2))
        conf = 1.0 - (H / H_max)
        return np.clip(conf, 0.0, 1.0)


# ---- Energy (log-sum-exp) ------------------------------------------------------------------------

@register("energy")
class Energy(ConfidenceEstimator):
    """
    Energy score: E(x) = -logsumexp(logits / T)
    ここでは **単調写像** σ(-E) を返し、0..1 に正規化（大きいほど高確信）。
    """

    name = "energy"

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        if logits is None:
            # probs しか無い場合は logits ~ log(p) とみなす（温度無視の近似）
            p = _to_probs(None, probs, T=1.0)
            logits_np = np.log(np.clip(p, 1e-12, 1.0))
        else:
            logits_np = _to_numpy(logits) / max(temperature, 1e-8)
        lse = _logsumexp_np(logits_np)  # logsumexp
        energy = -lse
        # スケール不変な単調変換で 0..1 に：
        return _sigmoid01(energy)


# ---- Margin (top1 - top2) ------------------------------------------------------------------------

@register("margin")
class Margin(ConfidenceEstimator):
    """margin = top1 - top2 を単調写像 σ(α * margin) で 0..1 に写す。"""

    name = "margin"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha  # 小さすぎると潰れる／大きすぎると0/1に張り付く

    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        t1, t2 = _top2(p)
        margin = t1 - t2  # 0..1
        return _sigmoid01(margin, k=self.alpha)


# ---- MC-Dropout ----------------------------------------------------------------------------------

@dataclass
class MCDropoutConfig:
    n_passes: int = 20
    temperature: float = 1.0
    use_predictive_entropy: bool = True  # True: 1 - H( E[p] ), False: E[ max(p) ]


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
    _enable_dropout(model)  # Dropoutのみtrainにする

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
    """
    MC-Dropout based confidence.

    2通りの代表的な集約を提供：
      - use_predictive_entropy=True:   conf = 1 - H( mean(probs) ) / log(C)
      - use_predictive_entropy=False:  conf = mean( max(probs) )
    """

    name = "mc_dropout"

    def __init__(self, cfg: Optional[MCDropoutConfig] = None):
        self.cfg = cfg or MCDropoutConfig()

    def run_with_dataloader(
        self,
        model: Any,
        dataloader: Iterable,
        device: str = "auto",
    ) -> np.ndarray:
        """dataloader を再推論して信頼度を返す（0..1）。"""
        logits_stack = []
        for _ in range(max(1, self.cfg.n_passes)):
            logits_np = _predict_logits_once(model, dataloader, device=device)
            logits_stack.append(logits_np / max(self.cfg.temperature, 1e-8))
        # shape: (T, N, C)
        L = np.stack(logits_stack, axis=0)
        P = _softmax_np(L, T=1.0, axis=-1)  # 温度適用済みlogitsにsoftmax
        P_mean = P.mean(axis=0)  # (N, C)

        if self.cfg.use_predictive_entropy:
            H = _entropy_np(P_mean)
            C = P_mean.shape[1]
            conf = 1.0 - H / math.log(max(C, 2))
            return np.clip(conf, 0.0, 1.0)
        else:
            return P_mean.max(axis=1)  # mean of max(p)

    # logits/probs からの簡易版（MCを擬似的に適用しない。互換のためのプレースホルダ）
    def __call__(self, *, logits=None, probs=None, temperature: float = 1.0, **_) -> np.ndarray:
        p = _to_probs(logits, probs, T=temperature)
        if self.cfg.use_predictive_entropy:
            H = _entropy_np(p)
            C = p.shape[1]
            conf = 1.0 - H / math.log(max(C, 2))
            return np.clip(conf, 0.0, 1.0)
        else:
            return p.max(axis=1)


# ---- 簡易ファクトリ（直接利用用） -----------------------------------------------------------------

_ESTIMATORS: Dict[str, Callable[..., ConfidenceEstimator]] = {
    "msp": MSP,
    "entropy": OneMinusEntropy,
    "energy": Energy,
    "margin": Margin,
    "mc_dropout": MCDropout,
}


def create_estimator(name: str, **kwargs) -> ConfidenceEstimator:
    """
    直接 import して使える軽量ファクトリ。
    registry.create がある場合はそちらの利用を推奨。
    """
    name_l = name.lower()
    if name_l not in _ESTIMATORS:
        raise KeyError(f"Unknown estimator: {name}")
    return _ESTIMATORS[name_l](**kwargs)


# ---- モジュール内セルフテスト（任意実行） ---------------------------------------------------------

if __name__ == "__main__":  # 簡易動作確認（ユニットテストの代替）
    rng = np.random.default_rng(0)
    N, C = 8, 6
    logits = rng.normal(size=(N, C))
    print("[msp]", MSP()(logits=logits)[:3])
    print("[entropy]", OneMinusEntropy()(logits=logits)[:3])
    print("[energy]", Energy()(logits=logits)[:3])
    print("[margin]", Margin(alpha=8.0)(logits=logits)[:3])
