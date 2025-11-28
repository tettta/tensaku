# /home/esakit25/work/tensaku/src/tensaku/gate.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.gate
@role     : dev の予測CSVから閾値 τ を推定し、pool/test に仮想 HITL を適用して gate_* を保存する
@inputs   :
  - cfg["run"].out_dir   : 出力ディレクトリ（例: /.../outputs/q-Y14_1-2_1_3）
  - cfg["run"].conf_key  : 使用する確信度列名（例: "conf_msp", "conf_msp_temp", "trust"）
  - cfg["gate"].eps_cse      : dev 上で許容する重大誤採点率 CSE の上限（例: 0.10）
  - cfg["gate"].cse_abs_err  : 重大誤採点とみなす |pred-true| の閾値（例: 4）
  - CLI 引数（優先度: CLI > cfg > デフォルト）
      --conf-key STR       : 使用する確信度列名
      --eps-cse FLOAT      : dev 上の CSE 上限
      --cse-abs-err INT    : CSE 判定用の絶対誤差
@files_in :
  - {out_dir}/preds_detail.csv  : split,id,y_true,y_pred,conf_* を含む CSV
    （無い場合は dev/pool/test_preds.csv から組み立てを試みる）
@outputs  :
  - {out_dir}/gate_assign.csv   :
        split,id,route,y_pred,<conf列...>,y_true(あれば),abs_err,severe_err
        route は "auto"（conf>=τ） / "human"（それ以外）
  - {out_dir}/gate_meta.json    :
        {
          "conf_key": ...,
          "tau": ...,
          "cse_abs_err": ...,
          "eps_cse": ...,
          "primary_split": "test"|"pool"|"dev",
          "coverage": ...,
          "rmse": ...,
          "cse": ...,
          "splits": {
            "dev":  {"n":..., "n_labeled":..., "coverage":..., "rmse":..., "cse":...},
            "pool": {...},
            "test": {...}
          }
        }
@cli     : tensaku gate -c configs/exp_al_hitl.yaml [--conf-key conf_msp] [--eps-cse 0.10] [--cse-abs-err 4]
@deps    : pandas, numpy
@notes   :
  - 本モジュールは解析専用であり、data_sas/splits 以下の labeled/pool/test には一切触れない。
  - 閾値 τ は dev のみから決定し、pool/test にはそのまま適用する（リーク防止）。
  - coverage/RMSE/CSE は「auto ルートに流したサンプル」に対してのみ計算する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import argparse
import json
import os

import numpy as np
import pandas as pd


# =====================================================================================
# 小さなユーティリティ
# =====================================================================================


@dataclass
class GateConfig:
    out_dir: str
    conf_key: str
    eps_cse: float
    cse_abs_err: int


def _load_preds_detail(out_dir: str) -> pd.DataFrame:
    """preds_detail.csv をロード（無ければ dev/pool/test_preds.csv から組み立て）。

    期待カラム（最低限）:
      - split: dev|pool|test
      - id
      - y_pred
      - conf_*  （conf_key で指定された列）
      - y_true （評価可能なら）
    """
    path = os.path.join(out_dir, "preds_detail.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "split" not in df.columns:
            raise RuntimeError(f"[gate] ERROR: 'split' column missing in {path}")
        return df

    # レガシー互換: dev_preds.csv / pool_preds.csv / test_preds.csv から統合を試みる
    print(f"[gate] info: {path} not found. Falling back to *_preds.csv files.")
    frames: List[pd.DataFrame] = []
    for split in ("dev", "pool", "test"):
        p = os.path.join(out_dir, f"{split}_preds.csv")
        if not os.path.exists(p):
            continue
        df_s = pd.read_csv(p)
        df_s["split"] = split
        frames.append(df_s)
    if not frames:
        raise FileNotFoundError(
            f"[gate] ERROR: neither preds_detail.csv nor *_preds.csv found in {out_dir}"
        )
    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def _cse_rate(y_true: np.ndarray, y_pred: np.ndarray, abs_err_th: int) -> float:
    """重大誤採点率（CSE）を計算する。

    CSE = mean(|pred - true| >= abs_err_th)
    サンプル数 0 のときは 0.0 を返す。
    """
    if y_true.size == 0:
        return 0.0
    err = np.abs(y_pred - y_true)
    return float(np.mean(err >= abs_err_th))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE を計算（サンプル数 0 のときは 0.0）。"""
    if y_true.size == 0:
        return 0.0
    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))


def _find_tau_for_cse_constraint_prefix(
    conf: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps_cse: float,
    abs_err_th: int,
) -> Tuple[Optional[float], float, float, float]:
    """dev 上で CSE≤eps_cse を満たしつつ coverage 最大となる τ を探索（prefix 走査）。

    手順:
      - conf の NaN を除外
      - まず「全体の CSE」がすでに eps_cse 以下なら、全件 auto として即終了
      - conf 降順にソートし、上位 k 件を auto とみなす
      - k=1..n について CSE を評価し、条件 CSE≤eps_cse を満たす最大 k を採用
      - τ は「採用された集合の中で最も低い conf 値」とする

    戻り値: (tau, coverage, cse, rmse)
      - tau が None の場合は「条件を満たす τ が存在しない」ことを意味する
        （coverage=0, cse=0, rmse=0 とする）。
    """
    # NaN を除外
    mask_valid = ~np.isnan(conf)
    conf = conf[mask_valid]
    y_true = y_true[mask_valid]
    y_pred = y_pred[mask_valid]

    n = conf.size
    if n == 0:
        return None, 0.0, 0.0, 0.0

    # まず「全体の CSE」がすでに eps_cse 以下なら、全件 auto にして終了
    cse_all = _cse_rate(y_true, y_pred, abs_err_th)
    if cse_all <= eps_cse:
        tau = float(conf.min())  # conf >= tau で全件 auto
        cov = 1.0
        rmse_all = _rmse(y_true, y_pred)
        return tau, cov, cse_all, rmse_all

    # conf 降順にソート
    order = np.argsort(conf)[::-1]
    conf_s = conf[order]
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]

    best_k = 0
    best_cov = 0.0
    best_cse = 0.0
    best_rmse = 0.0

    for k in range(1, n + 1):
        idx = slice(0, k)
        yt = y_true_s[idx]
        yp = y_pred_s[idx]
        cse = _cse_rate(yt, yp, abs_err_th)
        if cse > eps_cse:
            continue
        cov = k / n
        rmse = _rmse(yt, yp)
        if cov > best_cov:
            best_cov = cov
            best_k = k
            best_cse = cse
            best_rmse = rmse

    if best_k == 0:
        # 条件を満たす τ が存在しなかった場合：coverage=0 で返す
        return None, 0.0, 0.0, 0.0

    tau = float(conf_s[best_k - 1])
    return tau, best_cov, best_cse, best_rmse


def _summarize_split(
    df: pd.DataFrame,
    conf_key: str,
    tau: Optional[float],
    cse_abs_err: int,
) -> Dict[str, Any]:
    """特定 split に対する coverage/RMSE/CSE を要約する。"""
    n_total = int(len(df))
    if n_total == 0:
        return {
            "n": 0,
            "n_labeled": 0,
            "coverage": 0.0,
            "rmse": 0.0,
            "cse": 0.0,
        }

    has_label = "y_true" in df.columns and df["y_true"].notna().any()
    if has_label:
        df_l = df[df["y_true"].notna()].copy()
    else:
        df_l = df.iloc[0:0].copy()  # 空
    n_labeled = int(len(df_l))

    if tau is None or n_labeled == 0:
        return {
            "n": n_total,
            "n_labeled": n_labeled,
            "coverage": 0.0,
            "rmse": 0.0,
            "cse": 0.0,
        }

    conf = df_l[conf_key].to_numpy(dtype=float)
    y_true = df_l["y_true"].to_numpy(dtype=int)
    y_pred = df_l["y_pred"].to_numpy(dtype=int)

    mask_auto = conf >= tau
    if mask_auto.sum() == 0:
        cov = 0.0
        cse = 0.0
        rmse = 0.0
    else:
        cov = float(mask_auto.mean())
        yt_a = y_true[mask_auto]
        yp_a = y_pred[mask_auto]
        cse = _cse_rate(yt_a, yp_a, cse_abs_err)
        rmse = _rmse(yt_a, yp_a)

    return {
        "n": n_total,
        "n_labeled": n_labeled,
        "coverage": cov,
        "rmse": rmse,
        "cse": cse,
    }


def _choose_primary_split(summaries: Dict[str, Dict[str, Any]]) -> str:
    """gate_meta.json の coverage/cse/rmse をどの split から取るか決める。

    優先順位: test -> pool -> dev
    """
    for key in ("test", "pool", "dev"):
        s = summaries.get(key)
        if s and s.get("n_labeled", 0) > 0:
            return key
    # すべてラベルなしなら、何かしら存在する split を返す（なければ dev）
    for key in ("test", "pool", "dev"):
        if key in summaries:
            return key
    return "dev"


def _build_gate_config(argv: Optional[List[str]], cfg: Dict[str, Any]) -> GateConfig:
    """CLI 引数と cfg から GateConfig を組み立てる。"""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--conf-key", dest="conf_key", type=str, default=None)
    ap.add_argument("--eps-cse", dest="eps_cse", type=float, default=None)
    ap.add_argument("--cse-abs-err", dest="cse_abs_err", type=int, default=None)
    ns, _rest = ap.parse_known_args(argv or [])

    run_cfg = cfg.get("run") or {}
    gate_cfg = cfg.get("gate") or {}

    out_dir = run_cfg.get("out_dir") or "./outputs"
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # conf_key の優先順位: CLI > run.conf_key > gate.conf_key > "conf_msp"
    conf_key = ns.conf_key or run_cfg.get("conf_key") or gate_cfg.get("conf_key") or "conf_msp"

    # eps_cse の優先順位: CLI > gate.eps_cse > デフォルト 0.05
    eps_cse = ns.eps_cse if ns.eps_cse is not None else gate_cfg.get("eps_cse", 0.05)

    # cse_abs_err の優先順位: CLI > gate.cse_abs_err > デフォルト 2
    cse_abs_err = ns.cse_abs_err if ns.cse_abs_err is not None else gate_cfg.get("cse_abs_err", 2)

    print(f"[gate] out_dir     : {out_dir}")
    print(f"[gate] conf_key    : {conf_key}")
    print(f"[gate] eps_cse     : {eps_cse}")
    print(f"[gate] cse_abs_err : {cse_abs_err}")

    return GateConfig(out_dir=out_dir, conf_key=conf_key, eps_cse=eps_cse, cse_abs_err=cse_abs_err)


# =====================================================================================
# メイン処理
# =====================================================================================


def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    """tensaku gate メインエントリ。

    手順:
      1) preds_detail.csv を読み取り、dev/pool/test に分割
      2) dev の conf_key を用いて閾値 τ を探索（CSE≤eps_cse で coverage 最大）
      3) τ を pool/test に適用し、gate_assign.csv と gate_meta.json を出力
    """
    gc = _build_gate_config(argv, cfg)

    # preds_detail を読み込み
    try:
        df = _load_preds_detail(gc.out_dir)
    except Exception as e:
        print(f"[gate] ERROR: failed to load predictions: {e}")
        return 1

    if gc.conf_key not in df.columns:
        print(f"[gate] ERROR: column '{gc.conf_key}' not found in preds detail.")
        print("[gate]        Available columns:", list(df.columns))
        return 1

    # dev 部分を抽出（ラベル付きのみ）
    df_dev = df[df["split"] == "dev"].copy()
    if "y_true" not in df_dev.columns or not df_dev["y_true"].notna().any():
        print("[gate] ERROR: no labeled rows for dev split='dev'")
        return 1
    df_dev = df_dev[df_dev["y_true"].notna()].copy()

    conf_dev = df_dev[gc.conf_key].to_numpy(dtype=float)
    y_true_dev = df_dev["y_true"].to_numpy(dtype=int)
    y_pred_dev = df_dev["y_pred"].to_numpy(dtype=int)

    print(f"[gate] dev: n={len(df_dev)} (labeled) for threshold search")

    # 1st: prefix 走査で τ を探索
    tau, cov_dev, cse_dev, rmse_dev = _find_tau_for_cse_constraint_prefix(
        conf_dev, y_true_dev, y_pred_dev, gc.eps_cse, gc.cse_abs_err
    )

    # 2nd: それでも τ が見つからない場合、dev 全体で CSE≤eps_cse なら「全件 auto」にフォールバック
    if tau is None:
        cse_all = _cse_rate(y_true_dev, y_pred_dev, gc.cse_abs_err)
        rmse_all = _rmse(y_true_dev, y_pred_dev)
        if cse_all <= gc.eps_cse:
            tau = float(np.nanmin(conf_dev))  # dev 全体を auto にする最小 conf
            cov_dev = 1.0
            cse_dev = cse_all
            rmse_dev = rmse_all
            print(
                f"[gate] INFO: prefix search found no tau, but CSE_all={cse_all:.4f}<=eps_cse."
                " Fallback to tau=min(conf_dev) (coverage=1.0)."
            )
        else:
            print(
                f"[gate] WARN: no tau satisfies CSE<=eps_cse (eps={gc.eps_cse}). "
                "         Falling back to coverage=0 (all samples -> human)."
            )
            tau = float("inf")  # conf>=inf は常に False → coverage=0
            cov_dev = 0.0
            cse_dev = 0.0
            rmse_dev = 0.0

    print(f"[gate] selected tau={tau:.6f}  (dev coverage={cov_dev:.3f}, CSE={cse_dev:.4f}, RMSE={rmse_dev:.4f})")

    # split ごとの要約
    summaries: Dict[str, Dict[str, Any]] = {}
    for split in ("dev", "pool", "test"):
        df_s = df[df["split"] == split].copy()
        if len(df_s) == 0:
            continue
        summaries[split] = _summarize_split(df_s, gc.conf_key, tau, gc.cse_abs_err)
        s = summaries[split]
        print(
            f"[gate] {split}: n={s['n']}, n_labeled={s['n_labeled']}, "
            f"coverage={s['coverage']:.3f}, CSE={s['cse']:.4f}, RMSE={s['rmse']:.4f}"
        )

    # gate_assign.csv を作成（pool/test を対象）
    df_assign_src = df[df["split"].isin(["pool", "test"])].copy()
    if df_assign_src.empty:
        print("[gate] WARN: no pool/test rows found. gate_assign.csv will contain 0 rows.")
    conf = df_assign_src[gc.conf_key].to_numpy(dtype=float)
    route = np.where(conf >= tau, "auto", "human")
    df_assign = pd.DataFrame(
        {
            "split": df_assign_src["split"].to_list(),
            "id": df_assign_src["id"].to_list(),
            "route": route,
            "y_pred": df_assign_src["y_pred"].to_list(),
            gc.conf_key: conf,
        }
    )

    # 他の conf_* 列や y_true があれば付け足す（解析用）
    extra_cols = [
        c
        for c in df_assign_src.columns
        if c not in df_assign.columns and c not in {"split", "id"}
    ]
    for col in extra_cols:
        df_assign[col] = df_assign_src[col].to_list()

    # 評価用に絶対誤差と severe_err フラグ（ラベルがあれば）
    if "y_true" in df_assign.columns:
        yt = pd.to_numeric(df_assign["y_true"], errors="coerce")
        yp = pd.to_numeric(df_assign["y_pred"], errors="coerce")
        abs_err = (yp - yt).abs()
        df_assign["abs_err"] = abs_err
        df_assign["severe_err"] = abs_err >= gc.cse_abs_err

    out_assign = os.path.join(gc.out_dir, "gate_assign.csv")
    df_assign.to_csv(out_assign, index=False)
    print(f"[gate] wrote gate_assign.csv -> {out_assign} (n={len(df_assign)})")

    # gate_meta.json を作成
    primary = _choose_primary_split(summaries) if summaries else "dev"
    primary_summary = summaries.get(primary, {"coverage": 0.0, "rmse": 0.0, "cse": 0.0})

    meta: Dict[str, Any] = {
        "conf_key": gc.conf_key,
        "tau": float(tau),
        "cse_abs_err": int(gc.cse_abs_err),
        "eps_cse": float(gc.eps_cse),
        "primary_split": primary,
        "coverage": float(primary_summary.get("coverage", 0.0)),
        "rmse": float(primary_summary.get("rmse", 0.0)),
        "cse": float(primary_summary.get("cse", 0.0)),
        "splits": summaries,
    }

    out_meta = os.path.join(gc.out_dir, "gate_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[gate] wrote gate_meta.json -> {out_meta}")

    return 0


if __name__ == "__main__":  # 直叩き用（開発時向け）
    import sys
    from tensaku.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--conf-key", dest="conf_key", type=str, default=None)
    parser.add_argument("--eps-cse", dest="eps_cse", type=float, default=None)
    parser.add_argument("--cse-abs-err", dest="cse_abs_err", type=int, default=None)
    args, rest = parser.parse_known_args()

    cfg = load_config(args.config, [])
    cli_argv: List[str] = []
    if args.conf_key is not None:
        cli_argv.extend(["--conf-key", args.conf_key])
    if args.eps_cse is not None:
        cli_argv.extend(["--eps-cse", str(args.eps_cse)])
    if args.cse_abs_err is not None:
        cli_argv.extend(["--cse-abs-err", str(args.cse_abs_err)])

    cli_argv.extend(rest)
    sys.exit(run(cli_argv, cfg))
