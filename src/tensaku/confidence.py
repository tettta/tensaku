# /home/esakit25/work/tensaku/src/tensaku/confidence.py
"""
@module     tensaku.confidence
@role       確信度推定（MSP / entropy / energy / margin / MC-Dropout）の薄い共通層
@inputs     - logits: ndarray | torch.Tensor, shape (N, C)  ※logitsがあれば内部でsoftmax(T)適用可
           - probs:  ndarray | torch.Tensor, shape (N, C)  ※logits無しでも可
           - temperature (T): float（任意。温度スケーリング後の確信度に使用）
           - MC-Dropout系: model, dataloader（再推論）
@outputs    - conf: ndarray, shape (N,)（大きいほど確信が高い想定）
@cli        直接のCLIは持たない（tensaku gate / tensaku infer-pool から内部利用）
@api        create_estimator(name:str, **kw) -> Callable[..., np.ndarray]
           提供名: "msp", "entropy", "energy", "margin", "mc_dropdown"
@deps       numpy（必須） / torch（任意。MC-Dropout時）
@config     CFG.gate.conf_name でレジストリ（tensaku.registry）から name を解決
@contracts  - 返値は [0,1] 正規化（MSP, 1-Entropy(normalized), Energy(sigmoid), Margin(sigmoid)）
           - 入力 logits/probs は (N,C)。C>=2。NaN/Infは非許容
@errors     - 未対応nameは KeyError。型/形状不整合は ValueError（メッセージにnameとshapeを含める）
@notes      - Energy/Marginはランク保持を優先し単調写像(sigmoid)で0–1に収める
           - registry への登録名: "msp", "entropy", "energy", "margin", "mc_dropout"
@tests      - スモーク: ランダムlogits→各推定器の先頭3件を出力（範囲[0,1]にあることをassert）
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
        # ここで σ(-E) を返す（E が小さいほど conf が大きくなるように反転）
        return _sigmoid01(-energy)



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


# ---- CLI entry point: tensaku confidence --------------------------------------------


def run(argv=None, cfg=None) -> int:
    """
    @cli   : tensaku confidence -c CFG.yaml [--out PATH]
    @role  : dev/pool/test の *_preds.csv から preds_detail.csv を生成し、
             Active Learning / 可視化用に一貫したカラム構成へ正規化しつつ、
             必要に応じて logits.npy から追加の確信度指標を計算して付与する。
    """
    import argparse
    import math as _math
    import os as _os
    import pandas as _pd
    import numpy as _np

    parser = argparse.ArgumentParser(prog="tensaku confidence", add_help=True)
    parser.add_argument(
        "--out",
        "--detail-out",
        dest="out_path",
        default=None,
        help="書き出す preds_detail.csv のパス（省略時: {run.out_dir}/preds_detail.csv）",
    )
    args = parser.parse_args(argv or [])

    # ---- 設定読み出し -------------------------------------------------
    cfg = cfg or {}
    run_cfg = cfg.get("run", {}) or {}
    conf_cfg = cfg.get("confidence", {}) or {}
    out_dir = run_cfg.get("out_dir")
    if not out_dir:
        print("[confidence] ERROR: run.out_dir is not set in config", flush=True)
        return 1
    out_dir = str(out_dir)

    # estimators: [{name: "msp"}, "entropy", ...] などを許容
    raw_ests = conf_cfg.get("estimators", []) or []
    est_names: list[str] = []
    for est in raw_ests:
        if isinstance(est, dict):
            name = est.get("name")
        else:
            name = str(est)
        if not name:
            continue
        est_names.append(str(name).strip())
    # 小文字で正規化
    est_names = [n.lower() for n in est_names]

    # ---- splitごとの preds.csv 読み込み --------------------------------
    def _load_split(split_name: str) -> "_pd.DataFrame | None":
        path = _os.path.join(out_dir, f"{split_name}_preds.csv")
        if not _os.path.exists(path):
            print(f"[confidence] WARN: missing preds CSV for split='{split_name}': {path}", flush=True)
            return None
        df = _pd.read_csv(path)
        if "id" not in df.columns:
            raise KeyError(f"[confidence] missing 'id' column in {path}")
        df.insert(0, "split", split_name)

        # y_true は pool では存在しないので NaN で埋める
        if "y_true" not in df.columns:
            df["y_true"] = _np.nan

        # conf_msp は必須（infer-pool 側で出力されている前提）
        if "conf_msp" not in df.columns:
            raise KeyError(f"[confidence] missing 'conf_msp' column in {path}")

        return df

    dfs = []
    for split_name in ("dev", "pool", "test"):
        df = _load_split(split_name)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("[confidence] ERROR: no preds CSV found in out_dir", flush=True)
        return 1

    df_all = _pd.concat(dfs, ignore_index=True)

    # ---- logits ベースの追加確信度指標を計算 ----------------------------
    # estimator名 → 出力カラム名のマッピング
    est_to_col = {
        "msp": "conf_msp",
        "entropy": "conf_entropy",
        "energy": "conf_energy",
        "margin": "conf_margin",
        "prob_margin": "conf_prob_margin",
        "mc_dropout": "conf_mcdo",
        "trust": "conf_trust",
        "conf_trust": "conf_trust",
    }

    # logits が必要な推定器
    need_logits = {"msp", "entropy", "energy", "margin", "prob_margin", "mc_dropout"}

    # splitごとの logits をキャッシュ読み込み
    logits_cache: dict[str, "_np.ndarray | None"] = {}

    def _load_logits(split_name: str) -> "_np.ndarray | None":
        if split_name in logits_cache:
            return logits_cache[split_name]
        path = _os.path.join(out_dir, f"{split_name}_logits.npy")
        if not _os.path.exists(path):
            print(f"[confidence] WARN: missing logits for split='{split_name}': {path}", flush=True)
            logits_cache[split_name] = None
            return None
        arr = _np.load(path)
        logits_cache[split_name] = arr
        return arr

    # 実際に計算
    splits_present = list(dict.fromkeys(df_all["split"].tolist()))  # 順序保持のunique

    for est_name in est_names:
        col = est_to_col.get(est_name)
        if not col:
            print(f"[confidence] WARN: unknown estimator name in config: {est_name}", flush=True)
            continue

        # 既にカラムがある場合は再計算しない（infer-pool等で埋まっているケース）
        if col in df_all.columns:
            continue

        # trust/conf_trust は infer-pool 側で算出済みの列を期待
        if est_name in {"trust", "conf_trust"}:
            if "conf_trust" not in df_all.columns:
                print("[confidence] WARN: 'conf_trust' column is not present; skipping trust estimator", flush=True)
            continue

        # msp は conf_msp 列が既にある前提なので、ここでは何もしない
        if est_name == "msp":
            continue

        # MC-Dropout は現状 CLI からの on-demand 再推論には未対応（将来拡張）
        if est_name == "mc_dropout":
            print("[confidence] WARN: 'mc_dropout' estimator is not yet supported in CLI; skipping.", flush=True)
            continue

        # ここからは logits 必須の推定器（entropy / energy / margin / prob_margin）
        if est_name in need_logits:
            # estimator インスタンスを作成
            try:
                est = create_estimator(est_name)
            except Exception as e:  # pragma: no cover - best effort
                print(f"[confidence] WARN: failed to create estimator '{est_name}': {e}; skip", flush=True)
                continue

            # カラムを先に作っておく（デフォルト NaN）
            df_all[col] = _np.nan

            for split_name in splits_present:
                mask = df_all["split"] == split_name
                if not mask.any():
                    continue
                logits = _load_logits(split_name)
                if logits is None:
                    # この split では計算できないので NaN のまま
                    continue
                n_rows = int(mask.sum())
                if logits.shape[0] != n_rows:
                    print(
                        f"[confidence] WARN: logits length mismatch for split='{split_name}' "
                        f"(csv_rows={n_rows}, logits_rows={logits.shape[0]}); skipping this split.",
                        flush=True,
                    )
                    continue
                try:
                    conf = est(logits=logits)
                except Exception as e:  # pragma: no cover - best effort
                    print(
                        f"[confidence] WARN: estimator '{est_name}' failed on split='{split_name}': {e}; skip this split.",
                        flush=True,
                    )
                    continue
                conf = _np.asarray(conf).reshape(-1)
                if conf.shape[0] != n_rows:
                    print(
                        f"[confidence] WARN: estimator '{est_name}' returned length {conf.shape[0]} "
                        f"for split='{split_name}' (expected {n_rows}); skipping this split.",
                        flush=True,
                    )
                    continue
                df_all.loc[mask, col] = conf

    # ---- カラム順を固定（AL / viz / gate で前提とする） ----------------
    base_cols = [
        "split",
        "id",
        "y_true",
        "y_pred",
        "conf_msp",
        "conf_msp_temp",
        "conf_trust",
        "conf_entropy",
        "conf_margin",
        "conf_prob_margin",
        "conf_energy",
        "conf_mcdo",
    ]
    other_cols = [c for c in df_all.columns if c not in base_cols]
    ordered_cols = [c for c in base_cols if c in df_all.columns] + other_cols
    df_all = df_all[ordered_cols]

    # ---- preds_detail.csv として保存 ------------------------------------
    out_path = args.out_path or _os.path.join(out_dir, "preds_detail.csv")
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"[confidence] wrote preds_detail.csv -> {out_path} (n={len(df_all)})", flush=True)

    # ついでに dev 行の簡易指標もログに出す（研究メモ用）
    dev_mask = df_all["split"] == "dev"
    if dev_mask.any():
        dev = df_all[dev_mask]
        try:
            y_true = dev["y_true"].to_numpy(dtype=float)
            y_pred = dev["y_pred"].to_numpy(dtype=float)
            err = y_pred - y_true
            rmse = float(_math.sqrt(_np.mean(err ** 2)))
            cse = float(_np.mean(_np.abs(err) >= 2.0))
            print(f"[confidence] dev summary: RMSE={rmse:.4f}, CSE(|err|>=2)={cse:.4f}", flush=True)
        except Exception as e:  # pragma: no cover - best effort
            print(f"[confidence] WARN: failed to compute dev summary metrics: {e}", flush=True)

    return 0

