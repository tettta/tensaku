# /home/esakit25/work/tensaku/src/tensaku/al_sample.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.al_sample
@role     : Active Learning 用のサンプル選定
@inputs   : preds_detail.csv, pool_embs.npy (任意)
@outputs  : al_sample_ids.txt, al_sample.csv, al_sample_meta.json
@samplers :
  - "uncertainty" : 不確実性スコア（MSP/Trust等）でソートしてTop-K
  - "random"      : 完全ランダム
  - "clustering"  : 全体からK-Meansクラスタリング（Diversity重視）
  - "hybrid"      : 不確実性で候補を絞ってからK-Means（Uncertainty + Diversity）
"""

from __future__ import annotations

import argparse
import json
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# クラスタリング用 (scikit-learn必須)
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    SKLEARN_AVAIL = True
except ImportError:
    SKLEARN_AVAIL = False


# =============================================================================
# 設定・ユーティリティ
# =============================================================================


@dataclass
class SamplerConfig:
    out_dir: str
    detail_path: str
    sampler_name: str
    uncertainty: str
    ascending: bool
    budget: int
    threshold: Optional[float]
    hybrid_alpha: float = 3.0  # Hybrid用: 予算の何倍をPre-filterするか


def _load_preds_detail(detail_path: Path) -> pd.DataFrame:
    if not detail_path.exists():
        raise FileNotFoundError(f"preds_detail not found: {detail_path}")
    df = pd.read_csv(detail_path)
    if "split" not in df.columns or "id" not in df.columns:
        raise ValueError("preds_detail.csv must contain 'id' and 'split'.")
    return df


def _load_embs_pool(out_dir: Path) -> Optional[np.ndarray]:
    """pool_embs.npy を読み込む。無ければ None。"""
    path = out_dir / "pool_embs.npy"
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception as e:
        print(f"[al-sample] WARN: failed to load embs: {e}", file=sys.stderr)
        return None


def _resolve_budget(ns: argparse.Namespace, al_cfg: Dict[str, Any]) -> int:
    for key in ("budget", "k"):
        v = getattr(ns, key, None)
        if v is not None: return int(v)
    return int(al_cfg.get("budget") or al_cfg.get("k") or 50)


def _resolve_ascending(ns: argparse.Namespace, default_ascending: bool) -> bool:
    if getattr(ns, "ascending", False): return True
    if getattr(ns, "descending", False): return False
    return default_ascending


def _guess_conf_column(df: pd.DataFrame, unc_name: str) -> Tuple[str, bool]:
    """指標名からカラムと昇順/降順を推定。"""
    cols = df.columns
    unc = unc_name.lower()
    
    # マッピング定義 (name -> (col, ascending))
    # ascending=True: 小さいほど不確実 (MSP, Trust)
    # ascending=False: 大きいほど不確実 (Entropy)
    mapping = {
        "msp": ("conf_msp", True),
        "trust": ("conf_trust", True),
        "entropy": ("conf_entropy", False),
        "energy": ("conf_energy", True), # Energyは通常小さいほど安定だが、Confidenceとしては大きい方が確信が高いよう正規化されている前提
        "margin": ("conf_margin", True),
    }
    
    # 1. 既知のマッピング
    for key, (col, asc) in mapping.items():
        if unc == key or unc == col:
            if col in cols: return col, asc
            # conf_msp_temp などの派生対応
            if key == "msp" and "conf_msp_temp" in cols: return "conf_msp_temp", True

    # 2. そのままのカラム名があれば、デフォルトは昇順（小さい＝不確実）と仮定
    if unc in cols:
        return unc, True

    # エラー
    msg = f"Uncertainty metric '{unc_name}' not found in columns: {list(cols)}"
    # Entropyなど計算が必要なものがカラムにない場合はここでエラーにする
    raise ValueError(msg)


# =============================================================================
# サンプリング戦略の実装
# =============================================================================


def _run_kmeans_selection(
    df: pd.DataFrame,
    embs: np.ndarray,
    budget: int,
    seed: int
) -> pd.DataFrame:
    """共通処理: KMeansによる代表点選択"""
    if len(df) <= budget:
        return df

    # DataFrameの並び順と embs の並び順が一致していることを前提とする
    # df は reset_index されている必要がある
    indices = df.index.to_numpy()
    target_embs = embs[indices]

    kmeans = KMeans(n_clusters=budget, random_state=seed, n_init=10)
    kmeans.fit(target_embs)

    # 各クラスタ重心に最も近いサンプルを見つける
    closest_indices_in_subset, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, target_embs
    )
    
    # 全体インデックスに戻す
    final_indices = indices[closest_indices_in_subset]
    
    # 重複除去（稀に重心が重なると減る可能性があるため不足分を埋める処理を入れても良いが、今回はシンプルに）
    final_indices = np.unique(final_indices)
    
    return df.loc[final_indices].copy()


def _select_random(df: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    if len(df) <= budget: return df
    return df.sample(n=budget, random_state=seed).copy()


def _select_uncertainty(
    df: pd.DataFrame,
    conf_col: str,
    budget: int,
    ascending: bool,
    threshold: Optional[float]
) -> pd.DataFrame:
    """Top-K Uncertainty"""
    # 閾値フィルタ
    if threshold is not None:
        if ascending: df = df[df[conf_col] <= threshold]
        else: df = df[df[conf_col] >= threshold]
    
    if df.empty: return df
    
    df = df.sort_values(conf_col, ascending=ascending)
    return df.head(budget).copy()


def _select_clustering(
    df: pd.DataFrame,
    embs: np.ndarray,
    budget: int,
    seed: int
) -> pd.DataFrame:
    """K-Means only (Diversity Sampling)"""
    print(f"[al-sample] Clustering on {len(df)} samples...")
    return _run_kmeans_selection(df, embs, budget, seed)


def _select_hybrid(
    df: pd.DataFrame,
    embs: np.ndarray,
    conf_col: str,
    budget: int,
    ascending: bool,
    alpha: float,
    seed: int
) -> pd.DataFrame:
    """Uncertainty Pre-filter -> K-Means"""
    n_pre = int(budget * alpha)
    print(f"[al-sample] Hybrid: pre-selecting {n_pre} by uncertainty...")
    
    # 1. Pre-filter (Uncertainty)
    df_cand = df.sort_values(conf_col, ascending=ascending).head(n_pre)
    
    # 2. Clustering
    return _run_kmeans_selection(df_cand, embs, budget, seed)


# =============================================================================
# メインロジック
# =============================================================================


def main_impl(cfg: Dict[str, Any], ns: argparse.Namespace) -> int:
    run_cfg = cfg.get("run") or {}
    al_cfg = cfg.get("al") or {}
    sampler_cfg = al_cfg.get("sampler") or {}

    out_dir = Path(run_cfg.get("out_dir") or "./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # パラメータ解決
    # CLI --sampler > YAML al.sampler.name > "uncertainty"
    sampler_name = getattr(ns, "sampler", None)
    if not sampler_name:
        sampler_name = sampler_cfg.get("name") or "uncertainty"
    sampler_name = str(sampler_name).lower()

    # CLI --by > YAML al.by > YAML al.sampler.uncertainty > "msp"
    uncertainty_key = getattr(ns, "uncertainty", None)
    if not uncertainty_key:
        uncertainty_key = al_cfg.get("by") or sampler_cfg.get("uncertainty") or "msp"
    
    budget = _resolve_budget(ns, al_cfg)
    threshold = getattr(ns, "threshold", None) # TODO: YAML対応
    
    # Seed
    seed = int(getattr(ns, "seed", None) or al_cfg.get("seed") or run_cfg.get("seed") or 42)
    random.seed(seed)
    np.random.seed(seed)

    print(f"[al-sample] Start: sampler={sampler_name}, budget={budget}, by={uncertainty_key}")

    # データロード
    detail_path = getattr(ns, "detail", None) or (out_dir / "preds_detail.csv")
    try:
        df_all = _load_preds_detail(Path(detail_path))
    except Exception as e:
        print(f"[al-sample] ERROR: {e}", file=sys.stderr)
        return 1

    df_pool = df_all[df_all["split"] == "pool"].copy()
    # インデックスをリセットして 0..N にする (埋め込みとの対応のため必須)
    df_pool = df_pool.reset_index(drop=True)
    
    if df_pool.empty:
        print("[al-sample] WARN: pool is empty.")
        return 0

    # 埋め込みが必要な場合ロード
    embs = None
    if sampler_name in ("clustering", "hybrid"):
        if not SKLEARN_AVAIL:
            print("[al-sample] ERROR: scikit-learn not installed. Cannot use clustering/hybrid.", file=sys.stderr)
            return 1
        embs = _load_embs_pool(out_dir)
        if embs is None:
            print("[al-sample] WARN: pool_embs.npy not found. Fallback to 'uncertainty'.")
            sampler_name = "uncertainty"
        elif len(embs) != len(df_pool):
            print(f"[al-sample] WARN: embs shape {embs.shape} != pool size {len(df_pool)}. Fallback to 'uncertainty'.")
            sampler_name = "uncertainty"

    # サンプリング実行
    df_selected = pd.DataFrame()
    conf_col_used = None
    ascending_used = True

    # --- 1. Random ---
    if sampler_name == "random":
        df_selected = _select_random(df_pool, budget, seed)

    # --- 2. Clustering (Diversity only) ---
    elif sampler_name == "clustering":
        assert embs is not None
        df_selected = _select_clustering(df_pool, embs, budget, seed)

    # --- 3. Hybrid / Uncertainty ---
    else:
        # 不確実性カラムの特定
        try:
            conf_col, asc_default = _guess_conf_column(df_pool, uncertainty_key)
            ascending = _resolve_ascending(ns, asc_default)
            conf_col_used = conf_col
            ascending_used = ascending
        except ValueError as e:
            print(f"[al-sample] ERROR: {e}", file=sys.stderr)
            return 1

        if sampler_name == "hybrid":
            assert embs is not None
            alpha = float(sampler_cfg.get("alpha", 3.0))
            df_selected = _select_hybrid(df_pool, embs, conf_col, budget, ascending, alpha, seed)
        else:
            # uncertainty (default)
            df_selected = _select_uncertainty(df_pool, conf_col, budget, ascending, threshold)

    # 結果保存
    if df_selected.empty:
        print("[al-sample] WARN: No candidates selected.")
        (out_dir / "al_sample_ids.txt").write_text("", encoding="utf-8")
        return 0

    # rank付与 (便宜的)
    df_selected = df_selected.copy()
    df_selected["al_rank"] = np.arange(1, len(df_selected) + 1)

    # 1. IDs
    ids = [str(x) for x in df_selected["id"].tolist()]
    (out_dir / "al_sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    # 2. CSV
    out_cols = ["id", "split", "y_true", "y_pred", "al_rank"]
    if conf_col_used:
        out_cols.append(conf_col_used)
    
    # 存在するカラムだけ
    final_cols = [c for c in out_cols if c in df_selected.columns]
    df_selected[final_cols].to_csv(out_dir / "al_sample.csv", index=False)

    # 3. Meta
    meta = {
        "sampler": sampler_name,
        "uncertainty": uncertainty_key,
        "conf_column": conf_col_used,
        "budget": budget,
        "n_pool": len(df_pool),
        "n_selected": len(df_selected),
        "seed": seed
    }
    (out_dir / "al_sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[al-sample] Done. Selected {len(df_selected)} samples ({sampler_name}).")
    return 0


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    if cfg is None: raise ValueError("cfg required")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--detail", type=str)
    parser.add_argument("--budget", "--k", dest="budget", type=int)
    parser.add_argument("--uncertainty", "--by", dest="uncertainty", type=str)
    parser.add_argument("--sampler", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--seed", type=int)
    ns, _ = parser.parse_known_args(argv or [])
    return main_impl(cfg, ns)


if __name__ == "__main__":
    print("Run via tensaku al-sample")