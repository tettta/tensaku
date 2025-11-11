# -*- coding: utf-8 -*-
"""
@module     tensaku.gate
@role       HITLゲートの薄いオーケストレータ。
@overview   devで温度校正→信頼度→τ探索（CSE<=ε を満たし coverage 最大）→pool/testへ固定適用。
@inputs     - YAML設定: data_dir, outputs, gate.{conf_name,cse_abs_err,eps_list,accept_policy,pseudo_label_thresh}, calibration, confidence
            - {outputs}/dev_detail.csv        : y_true, y_pred, conf_* [任意: logits_*]
            - {outputs}/preds_detail.csv      : id, y_pred, conf_*     [任意: logits_*]
@outputs    - {outputs}/hitl_summary.csv      : eps, tau, coverage, CSE, RMSE, QWK, ...
            - {outputs}/accept.csv, hold.csv  : id, y_pred, conf, ...
            - {outputs}/curve_coverage_rmse.png
            - {outputs}/curve_coverage_cse_margin.png
@cli        tensaku gate -c /home/esakit25/work/tensaku/configs/exp_al_hitl.yaml [--conf msp|trust ...]
@notes      依存モジュールが未実装でも動くよう、内部実装にフォールバックを用意（calibration/trust 無しでも実行可）。
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import time
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# 可能ならプロジェクト内ユーティリティを使う（無ければ内部フォールバック）
try:
    from tensaku import metrics as _metrics
except Exception:  # フォールバック
    _metrics = None

try:
    from tensaku import calibration as _calib
except Exception:
    _calib = None

try:
    from tensaku import trustscore as _trust
except Exception:
    _trust = None

# 🔧 1) 先頭付近（importの下）にヘルパを追加（重複があればスキップ可）
CONF_PREFIX = "conf_"

def _subset_accept_hold(df_pool, mask, conf_prefix: str = CONF_PREFIX):
    """mask(1=accept/0=hold) で DataFrame を二分し、保存用の最小列に整形する。"""
    if not isinstance(mask, (list, tuple, np.ndarray, pd.Series)):
        raise RuntimeError("mask must be sequence-like (0/1).")
    mask = np.array(mask).astype(int)
    if len(mask) != len(df_pool):
        raise RuntimeError(f"mask length mismatch: mask={len(mask)} df={len(df_pool)}")
    cols_conf = [c for c in df_pool.columns if c.startswith(conf_prefix)]
    cols_base = [c for c in ["id", "y_pred"] if c in df_pool.columns]
    cols = cols_base + cols_conf
    df_accept = df_pool[mask == 1][cols].copy()
    df_hold   = df_pool[mask == 0][cols].copy()
    return df_accept, df_hold

def _load_pool_preds(out_dir: str) -> pd.DataFrame:
    """preds_detail.csv が無ければ pool_preds.csv を探す後方互換ローダ。"""
    p1 = os.path.join(out_dir, "preds_detail.csv")
    p2 = os.path.join(out_dir, "pool_preds.csv")
    for p in (p1, p2):
        if os.path.isfile(p):
            df = pd.read_csv(p)
            if "y_pred" not in df.columns:
                raise RuntimeError(f"missing column y_pred in {p}")
            return df
    raise RuntimeError(f"not found preds file: {p1} or {p2}")



# ---------------------------
# 小さなユーティリティ
# ---------------------------
def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _qwk(pred: np.ndarray, y: np.ndarray, n_class: Optional[int] = None) -> float:
    # フォールバックの簡易QWK（sklearn）
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y, pred, weights="quadratic"))
    except Exception:
        return float("nan")


def _cse_rate(pred: np.ndarray, y: np.ndarray, abs_err: int) -> float:
    return float(np.mean(np.abs(pred - y) >= abs_err))


def _softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick_conf_column(df: pd.DataFrame, conf_name: str) -> str:
    """
    conf_name='msp' なら conf_msp を優先、存在しなければ conf っぽい列から最大値を選ぶ。
    """
    key = f"conf_{conf_name}"
    if key in df.columns:
        return key
    # ゆるいフォールバック
    candidates = [c for c in df.columns if c.startswith("conf_")]
    if len(candidates) == 1:
        return candidates[0]
    # 最後のフォールバック
    return key  # たとえ無くてもこの名前で後段がわかる


# ---------------------------
# API: Temperature on dev
# ---------------------------
def fit_temperature_on_dev(dev_df: pd.DataFrame, logits_prefix: str = "logits_", label_col: str = "y_true") -> Optional[float]:
    """
    dev上のlogits_* から温度Tを推定。logits列が無い場合は None を返す。
    """
    logit_cols = [c for c in dev_df.columns if c.startswith(logits_prefix)]
    if not logit_cols or label_col not in dev_df.columns:
        return None

    logits = dev_df[logit_cols].to_numpy(dtype=np.float64)
    labels = dev_df[label_col].to_numpy(dtype=int)

    if _calib and hasattr(_calib, "tune_temperature"):
        # プロジェクト実装がある場合
        return float(_calib.tune_temperature(logits, labels))
    else:
        # フォールバック: 0.5～3.0 を0.1刻みでECE最小を探索（簡易）
        def ece_of_T(T: float) -> float:
            p = _softmax(logits, T=T)
            conf = p.max(axis=1)
            pred = p.argmax(axis=1)
            bins = np.linspace(0.0, 1.0, 16)
            idx = np.digitize(conf, bins, right=True)
            ece = 0.0
            for b in range(len(bins)):
                m = idx == b
                if not np.any(m):
                    continue
                acc = np.mean(pred[m] == labels[m])
                gap = abs(acc - np.mean(conf[m]))
                ece += gap * (np.sum(m) / len(conf))
            return ece

        Ts = np.arange(0.5, 3.01, 0.1)
        eces = [ece_of_T(float(T)) for T in Ts]
        return float(Ts[int(np.argmin(eces))])


# ---------------------------
# API: compute_confidences
# ---------------------------
def compute_confidences(df: pd.DataFrame, conf_name: str, T: Optional[float] = None,
                        logits_prefix: str = "logits_") -> np.ndarray:
    """
    conf列があればそれを使用。なければ logits から (温度T付きで) MSP を計算。
    """
    col = _pick_conf_column(df, conf_name)
    if col in df.columns:
        return df[col].to_numpy(dtype=float)

    # フォールバック: logits_* から MSP を作る
    logit_cols = [c for c in df.columns if c.startswith(logits_prefix)]
    if logit_cols:
        logits = df[logit_cols].to_numpy(dtype=np.float64)
        p = _softmax(logits, 1.0 if T is None else float(T))
        return p.max(axis=1)

    # すべて無い場合は定数（受け入れゼロ回避のため）※実務では警告
    return np.zeros(len(df), dtype=float)


# ---------------------------
# API: score_trust
# ---------------------------
def score_trust(df_pool: pd.DataFrame) -> Optional[np.ndarray]:
    """
    TrustScore を計算（プロジェクトの trustscore 実装があればそれを使用）。
    ここでは conf_trust 列があればそれを返し、無ければ None。
    """
    if "conf_trust" in df_pool.columns:
        return df_pool["conf_trust"].to_numpy(dtype=float)
    # 将来: _trust を使って CLS埋め込みから算出
    return None


# ---------------------------
# API: find_tau_for_constraint
# ---------------------------
def find_tau_for_constraint(y_pred: np.ndarray, y_true: np.ndarray, conf: np.ndarray,
                            eps: float, cse_abs_err: int, higher_is_better: bool = True) -> Tuple[float, float]:
    """
    dev上で CSE ≤ eps を満たしつつ coverage 最大の τ を探索。
    戻り値: (best_coverage, tau)。満たせない場合は (0.0, +inf / -inf) を返す。
    """
    n = len(conf)
    order = np.argsort(conf * (1 if higher_is_better else -1))[::-1]  # 降順（higher=True）/昇順（False）
    y_pred_sorted = y_pred[order]
    y_true_sorted = y_true[order]
    conf_sorted = conf[order]

    best_cov = 0.0
    best_tau = -math.inf if higher_is_better else math.inf

    for k in range(1, n + 1):
        cse = _cse_rate(y_pred_sorted[:k], y_true_sorted[:k], cse_abs_err)
        cov = k / n
        if cse <= eps and cov >= best_cov:
            best_cov = cov
            best_tau = conf_sorted[k - 1]

    if best_cov == 0.0:
        return 0.0, (math.inf if higher_is_better else -math.inf)
    return best_cov, float(best_tau)


# ---------------------------
# API: decide_mask
# ---------------------------
def decide_mask(conf: np.ndarray, tau: float, higher_is_better: bool = True) -> np.ndarray:
    """
    conf と τ から accept(1)/hold(0) のマスクを返す。
    """
    if higher_is_better:
        return (conf >= tau).astype(np.int32)
    else:
        return (conf <= tau).astype(np.int32)


# ---------------------------
# API: save_gate_csv
# ---------------------------
def save_gate_csv(out_dir: str, summary_row: dict,
                  df_accept: pd.DataFrame, df_hold: pd.DataFrame) -> None:
    _ensure_dir(out_dir)
    # hitl_summary.csv へ追記
    path_sum = os.path.join(out_dir, "hitl_summary.csv")
    row = {**summary_row}
    row["timestamp"] = _now_str()
    if os.path.exists(path_sum):
        pd.DataFrame([row]).to_csv(path_sum, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(path_sum, index=False)

    # accept/hold 明細
    df_accept.to_csv(os.path.join(out_dir, "accept.csv"), index=False)
    df_hold.to_csv(os.path.join(out_dir, "hold.csv"), index=False)


# ---------------------------
# 可視化（最小版）
# ---------------------------
def _save_curves(out_dir: str, y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, cse_abs_err: int) -> None:
    """
    coverage–RMSE / coverage–CSE を描く最小版。既存 plots があれば置き換えてください。
    """
    import matplotlib.pyplot as plt

    order = np.argsort(conf)[::-1]  # 高確信度から順に採用
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]
    conf_s = conf[order]

    covs, rmses, cses = [], [], []
    for k in range(1, len(conf_s) + 1):
        covs.append(k / len(conf_s))
        rmses.append(_rmse(y_pred_s[:k], y_true_s[:k]))
        cses.append(_cse_rate(y_pred_s[:k], y_true_s[:k], cse_abs_err))

    # RMSE曲線
    plt.figure()
    plt.plot(covs, rmses)
    plt.xlabel("coverage")
    plt.ylabel("RMSE")
    plt.title("coverage–RMSE")
    _ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, "curve_coverage_rmse.png"))
    plt.close()

    # CSE曲線
    plt.figure()
    plt.plot(covs, cses)
    plt.axhline(0.02, linestyle="--")
    plt.axhline(0.05, linestyle="--")
    plt.xlabel("coverage")
    plt.ylabel(f"CSE(|err|≥{cse_abs_err})")
    plt.title("coverage–CSE")
    plt.savefig(os.path.join(out_dir, "curve_coverage_cse_margin.png"))
    plt.close()


# ---------------------------
# メイン実行
# ---------------------------
@dataclasses.dataclass
class GateConfig:
    conf_name: str = "msp"
    cse_abs_err: int = 2
    eps_list: Tuple[float, ...] = (0.02, 0.05)
    accept_policy: str = "tau"      # 未来拡張用
    pseudo_label_thresh: Optional[float] = None
    higher_is_better: bool = True


def _load_io(cfg: dict) -> Tuple[str, str, pd.DataFrame, pd.DataFrame, GateConfig]:
    data_dir = cfg.get("data_dir") or cfg.get("DATA_DIR")
    out_dir = cfg.get("outputs") or cfg.get("OUT_DIR") or os.path.join(os.getcwd(), "outputs")

    gate_cfg = cfg.get("gate", {}) or {}
    g = GateConfig(
        conf_name=gate_cfg.get("conf_name", "msp"),
        cse_abs_err=int(gate_cfg.get("cse_abs_err", 2)),
        eps_list=tuple(gate_cfg.get("eps_list", [0.02, 0.05])),
        accept_policy=str(gate_cfg.get("accept_policy", "tau")),
        pseudo_label_thresh=gate_cfg.get("pseudo_label_thresh", None),
        higher_is_better=True,  # MSP/Trust は高いほど良い
    )

    dev_path = os.path.join(out_dir, "dev_detail.csv")
    pool_path = os.path.join(out_dir, "preds_detail.csv")
    if not os.path.isfile(dev_path):
        raise RuntimeError(f"missing: {dev_path}")
    if not os.path.isfile(pool_path):
        raise RuntimeError(f"missing: {pool_path}")

    dev_df = pd.read_csv(dev_path)
    pool_df = pd.read_csv(pool_path)
    return out_dir, data_dir, dev_df, pool_df, g


def run(argv: Optional[Iterable[str]] = None, cfg: Optional[dict] = None) -> int:
    """
    CLIエントリ。例:
      tensaku gate -c /home/esakit25/work/tensaku/configs/exp_al_hitl.yaml --conf msp
    """
    parser = argparse.ArgumentParser(prog="tensaku gate", description="HITL gate (devでτ探索→pool/testへ適用)")
    parser.add_argument("-c", "--config", type=str, required=(cfg is None), help="YAML config path")
    parser.add_argument("--conf", type=str, choices=["msp", "trust", "entropy", "energy", "margin"], default=None)
    parser.add_argument("--eps", type=float, nargs="*", default=None, help="CSE上限の候補（例: 0.02 0.05）")
    parser.add_argument("--cse-abs-err", type=int, default=None)
    parser.add_argument("--no-calib", action="store_true", help="温度校正をスキップ")
    parser.add_argument("--save-fig", action="store_true")
    # ★ 追加
    parser.add_argument("--no-infer", action="store_true", help="内部推論を一切行わず、既存ファイルのみ使用する")
    parser.add_argument("--preds", type=str, default=None, help="pool予測CSV（preds_detail.csv互換）を明示指定")
    args, _ = parser.parse_known_args(list(argv) if argv is not None else None)

    # 設定ロード
    yml = {} if cfg is None else cfg
    if cfg is None:
        yml = _read_yaml(args.config)
    out_dir, data_dir, dev_df, pool_df, g = _load_io(yml)

    # 引数で上書き
    if args.conf:
        g.conf_name = args.conf
    if args.eps:
        g.eps_list = tuple(float(x) for x in args.eps)
    if args.cse_abs_err is not None:
        g.cse_abs_err = int(args.cse_abs_err)

    # 1) 温度推定（任意）
    T = None
    if not args.no_calib:
        T = fit_temperature_on_dev(dev_df)  # 失敗(None)でも続行OK

    # 2) dev/pool の信頼度列の決定
    conf_dev = compute_confidences(dev_df, g.conf_name, T=T)

    # ★ poolの確定：--preds > pool_df > （最後の手段として既定ファイル探索）
    if args.preds:
        df_pool = pd.read_csv(args.preds)
    elif pool_df is not None and len(pool_df) > 0:
        df_pool = pool_df
    else:
        # 既定の出力場所から拾う（存在しなければ明示エラー）
        try:
            df_pool = _load_pool_preds(out_dir)
        except Exception as e:
            raise RuntimeError(
                f"[gate] pool predictions not found. "
                f"先に infer-pool を実行するか、--preds で明示してください: {e}"
            )

    # 再推論の完全抑止
    if args.no_infer:
        print("[gate] --no-infer: 内部推論は行いません（既存CSVのみ使用）")

    # pool側の信頼度
    conf_pool = compute_confidences(df_pool, g.conf_name, T=T)
    if g.conf_name == "trust":
        tr = score_trust(df_pool)
        if tr is not None:
            conf_pool = tr

    # devの真値・予測
    y_true_dev = dev_df["y_true"].to_numpy(int)
    y_pred_dev = dev_df["y_pred"].to_numpy(int)

    # poolの予測
    if "y_pred" not in df_pool.columns:
        raise RuntimeError("df_pool に y_pred 列がありません（preds_detail互換のCSVを指定してください）")
    y_pred_pool = df_pool["y_pred"].to_numpy(int)

    # 集計
    all_rows = []
    best_for_plot = None

    for eps in g.eps_list:
        cov_dev, tau = find_tau_for_constraint(
            y_pred_dev, y_true_dev, conf_dev,
            eps=eps, cse_abs_err=g.cse_abs_err,
            higher_is_better=g.higher_is_better
        )

        # --- τで二分（pool/test） ---
        mask_te = decide_mask(conf_pool, tau, higher_is_better=g.higher_is_better)
        df_accept, df_hold = _subset_accept_hold(df_pool, mask_te, conf_prefix=CONF_PREFIX)

        # --- dev側も τを適用して“受け入れサブセット”品質 ---
        mask_dev = decide_mask(conf_dev, tau, higher_is_better=g.higher_is_better)
        if mask_dev.sum() > 0:
            y_pred_dev_acc = y_pred_dev[mask_dev == 1]
            y_true_dev_acc = y_true_dev[mask_dev == 1]
            cse_at_tau = _cse_rate(y_pred_dev_acc, y_true_dev_acc, g.cse_abs_err)
            rmse_at_tau = _rmse(y_pred_dev_acc, y_true_dev_acc)
            n_class = int(max(y_true_dev.max(), y_pred_dev.max()) + 1)
            qwk_at_tau = _qwk(y_pred_dev_acc, y_true_dev_acc, n_class=n_class)
            coverage_dev = float(mask_dev.mean())
        else:
            cse_at_tau = float("nan")
            rmse_at_tau = float("nan")
            qwk_at_tau = float("nan")
            coverage_dev = 0.0

        row = dict(
            eps=float(eps),
            tau=float(tau),
            coverage=float(float(mask_te.mean())),  # pool/test 側 coverage
            CSE=float(cse_at_tau),                  # dev受け入れサブセットのCSE
            RMSE=float(rmse_at_tau),                # dev受け入れサブセットのRMSE
            QWK=float(qwk_at_tau),                  # dev受け入れサブセットのQWK
            coverage_dev=float(coverage_dev),       # dev側 coverage（参考）
            conf_name=g.conf_name,
            cse_abs_err=int(g.cse_abs_err),
        )

        cols_conf = [c for c in df_accept.columns if c.startswith(CONF_PREFIX)]
        save_gate_csv(
            out_dir,
            row,
            df_accept[["id", "y_pred"] + cols_conf] if "id" in df_accept.columns else df_accept,
            df_hold  [["id", "y_pred"] + cols_conf] if "id" in df_hold.columns   else df_hold,
        )

        all_rows.append(row)
        if best_for_plot is None:
            best_for_plot = (y_true_dev, y_pred_dev, conf_dev)

    if args.save_fig and best_for_plot is not None:
        _save_curves(out_dir, *best_for_plot, cse_abs_err=g.cse_abs_err)

    print(f"[gate] done. summary rows: {len(all_rows)}  -> {os.path.join(out_dir,'hitl_summary.csv')}")
    return 0



if __name__ == "__main__":
    sys.exit(run())
