# /home/esakit25/work/tensaku/scripts/utils/plot_al_select.py
# -*- coding: utf-8 -*-
"""
@role: Active Learning の各ラウンドで「選択されたサンプル」の分布を可視化する。
       True Score × Confidence Bin（MSP / Trust / Entropy）のヒートマップを出力する。
@notes:
    - preds_detail.csv（split,id,y_true,y_pred,conf_*）と
      oracle_labels_round*.csv（id, score or y_true）を突き合わせて可視化する。
    - AL 実験の設計上、preds_detail.csv は最後のラウンドで上書きされている前提。
      → 常に「最後のモデル」での conf_* を使う。
    - どのラウンドを集計するかは --round-mode / --round-k / --round-range で指定する。
"""

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.figsize": (10, 8)})


# ----------------------------------------------------------------------
# ユーティリティ
# ----------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[plot_sel] {msg}")


def load_preds_pool(exp_dir: str) -> Optional[pd.DataFrame]:
    """
    exp_dir 配下の preds_detail.csv を読み込み、split=='pool' の行のみ返す。
    必須カラム: id, y_true, conf_*（最低どれか1つ）
    """
    preds_path = os.path.join(exp_dir, "preds_detail.csv")
    if not os.path.exists(preds_path):
        _log(f"WARN: preds_detail.csv not found: {preds_path}")
        return None

    try:
        preds = pd.read_csv(preds_path)
    except Exception as e:  # pragma: no cover - ログ用
        _log(f"WARN: failed to read preds_detail.csv: {e}")
        return None

    if "id" not in preds.columns:
        _log("WARN: preds_detail.csv has no 'id' column")
        return None

    # pool のみ対象（split が無い場合はそのまま）
    if "split" in preds.columns:
        preds_pool = preds[preds["split"] == "pool"].copy()
    else:
        preds_pool = preds.copy()

    if preds_pool.empty:
        _log("WARN: preds_detail.csv has no pool rows")
        return None

    # 重複 id は先勝ちで1行に潰す
    preds_pool = preds_pool.drop_duplicates(subset=["id"])

    return preds_pool


def load_oracle_labels_by_round(exp_dir: str) -> Dict[int, pd.DataFrame]:
    """
    oracle_labels_round*.csv を round -> DataFrame の dict にして返す。
    - 探索パス: exp_dir/rounds/oracle_labels_round*.csv, exp_dir/oracle_labels_round*.csv
    - 必須カラム: id, (y_true or score)
    """
    label_files: List[str] = []
    label_files.extend(
        glob.glob(os.path.join(exp_dir, "rounds", "oracle_labels_round*.csv"))
    )
    label_files.extend(glob.glob(os.path.join(exp_dir, "oracle_labels_round*.csv")))

    if not label_files:
        _log(f"WARN: No oracle_labels_round*.csv found in {exp_dir}")
        return {}

    round_dfs: Dict[int, pd.DataFrame] = {}
    for fpath in sorted(label_files):
        fname = os.path.basename(fpath)
        m = re.search(r"round(\d+)\.csv", fname)
        if not m:
            continue
        r = int(m.group(1))

        try:
            df = pd.read_csv(fpath)
        except Exception as e:  # pragma: no cover - ログ用
            _log(f"WARN: failed to read {fpath}: {e}")
            continue

        if "id" not in df.columns:
            _log(f"WARN: 'id' column not found in {fpath}, skip")
            continue

        # y_true 列名の正規化
        if "y_true" in df.columns:
            pass
        elif "score" in df.columns:
            df = df.rename(columns={"score": "y_true"})
        else:
            _log(f"WARN: neither 'y_true' nor 'score' in {fpath}, skip")
            continue

        round_dfs[r] = df[["id", "y_true"]].copy()

    if not round_dfs:
        _log(f"WARN: No valid oracle labels with (id,y_true) in {exp_dir}")

    return round_dfs


# ----------------------------------------------------------------------
# プロット
# ----------------------------------------------------------------------


def make_selection_plot(df: pd.DataFrame, out_path_base: str, title: str) -> None:
    """
    df: 選択されたサンプル（複数 round 分をまとめたもの）
        必須カラム: y_true, conf_msp / conf_trust / conf_entropy のうちあるもの
    out_path_base: "selection_heatmap_round-xxx.png" のようなベースパス
                   実際には conf_key ごとに suffix を付けて保存する
    """
    if df.empty:
        _log("WARN: empty df passed to make_selection_plot; skip")
        return

    base_dir = os.path.dirname(out_path_base)
    base_name, ext = os.path.splitext(os.path.basename(out_path_base))
    if not ext:
        ext = ".png"

    os.makedirs(base_dir, exist_ok=True)

    # このスクリプトで見る conf の候補
    conf_keys = ["conf_msp", "conf_trust", "conf_entropy"]

    for conf_key in conf_keys:
        if conf_key not in df.columns:
            continue

        df_plot = df.copy()

        # entropy は「小さいほど自信あり」なので -entropy にして「大きいほど自信あり」に揃える
        if conf_key == "conf_entropy":
            df_plot["_conf_value"] = -df_plot[conf_key].astype(float)
            xlabel = "Confidence Bin (-entropy)"
        else:
            df_plot["_conf_value"] = df_plot[conf_key].astype(float)
            xlabel = f"Confidence Bin ({conf_key})"

        df_plot = df_plot.dropna(subset=["_conf_value", "y_true"])
        if df_plot.empty:
            _log(f"WARN: no valid rows for {conf_key}; skip plot")
            continue

        vmin = df_plot["_conf_value"].min()
        vmax = df_plot["_conf_value"].max()

        # conf がほぼ一定値のときの保険
        if np.isclose(vmin, vmax):
            edges = np.array([vmin - 1e-6, vmax + 1e-6])
        else:
            # 0〜1 の範囲に収まっていれば 0.0〜1.0 を10分割、それ以外は min〜max を10分割
            if vmin >= 0.0 and vmax <= 1.0:
                edges = np.linspace(0.0, 1.0, 11)
            else:
                edges = np.linspace(vmin, vmax, 11)

        labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
        df_plot["conf_bin"] = pd.cut(
            df_plot["_conf_value"],
            bins=edges,
            labels=labels,
            include_lowest=True,
        )

        # y軸（スコア）のビン決め
        min_score = df_plot["y_true"].min()
        max_score = df_plot["y_true"].max()

        if pd.api.types.is_integer_dtype(df_plot["y_true"]) and (max_score - min_score) < 30:
            df_plot["label_bin"] = df_plot["y_true"].astype(int).astype(str)
            y_label = "True Score"
        else:
            n_bins_label = min(15, max(1, int(max_score - min_score) + 1))
            df_plot["label_bin"] = pd.cut(
                df_plot["y_true"],
                bins=n_bins_label,
                include_lowest=True,
            )
            y_label = "True Score Bin"

        pivot = df_plot.pivot_table(
            index="label_bin",
            columns="conf_bin",
            aggfunc="size",
            fill_value=0,
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar=False)
        plt.xlabel(xlabel)
        plt.ylabel(y_label)
        plt.title(f"Selected Sample Distribution: {title} [{conf_key}]")

        out_path = os.path.join(base_dir, f"{base_name}_{conf_key}{ext}")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        _log(f"Saved: {out_path}")


# ----------------------------------------------------------------------
# メイン処理（1実験分）
# ----------------------------------------------------------------------


def process_experiment(
    exp_dir: str,
    exp_name: str,
    args: argparse.Namespace,
    out_dir_base: Optional[str] = None,
) -> None:
    """
    1) preds_detail.csv (split=pool) を読み込み
    2) oracle_labels_round*.csv と id で突き合わせ
    3) --round-mode / --round-k / --round-range で指定した round のみ集計
    4) conf_msp / conf_trust / conf_entropy でヒートマップを出力
    """
    _log(f"Start experiment: {exp_name} ({exp_dir})")

    preds_pool = load_preds_pool(exp_dir)
    if preds_pool is None:
        _log("WARN: skip because preds_pool is None")
        return

    round_dfs = load_oracle_labels_by_round(exp_dir)
    if not round_dfs:
        _log("WARN: skip because no oracle labels")
        return

    available_rounds = sorted(round_dfs.keys())
    if not available_rounds:
        _log("WARN: no available rounds")
        return

    # --- ラウンドの選択ロジック ---
    mode = args.round_mode
    k = args.round_k

    if mode == "all":
        target_rounds = available_rounds
        round_desc = "rounds_all"
    elif mode == "first":
        target_rounds = available_rounds[:k]
        round_desc = f"first_{k}_rounds"
    elif mode == "last":
        target_rounds = available_rounds[-k:]
        round_desc = f"last_{k}_rounds"
    elif mode == "range" and args.round_range:
        try:
            start_str, end_str = args.round_range.split(":")
            start_r = int(start_str)
            end_r = int(end_str)
            target_rounds = [r for r in available_rounds if start_r <= r <= end_r]
            round_desc = f"range_{start_r}_{end_r}"
        except Exception as e:  # pragma: no cover - ログ用
            _log(f"WARN: invalid --round-range '{args.round_range}': {e}")
            target_rounds = available_rounds
            round_desc = "rounds_all"
    else:
        target_rounds = available_rounds
        round_desc = "rounds_all"

    if not target_rounds:
        _log(f"WARN: No rounds selected (mode={mode})")
        return

    _log(f"Use rounds: {target_rounds} ({round_desc})")

    # --- 選択サンプルと preds_detail を id で突き合わせ ---
    df_list: List[pd.DataFrame] = []
    for r in target_rounds:
        df_label = round_dfs[r].copy()

        merged = df_label.merge(
            preds_pool,
            on="id",
            how="left",
            suffixes=("", "_pred"),
        )

        # preds 側の y_true が混ざった場合は oracle を優先
        if "y_true_pred" in merged.columns:
            merged = merged.drop(columns=["y_true_pred"])

        # conf_* が一つも無ければスキップ
        if not any(
            col in merged.columns
            for col in ("conf_msp", "conf_trust", "conf_entropy")
        ):
            _log(f"WARN: round {r} has no conf_* columns; skip")
            continue

        merged["round"] = r
        df_list.append(merged)

    if not df_list:
        _log("WARN: No samples with conf_* for selected rounds")
        return

    df_all = pd.concat(df_list, ignore_index=True)

    # --- 出力先の決定 ---
    if out_dir_base:
        out_dir = out_dir_base
    else:
        # 既定は exp_dir/analysis
        out_dir = os.path.join(exp_dir, "analysis")

    os.makedirs(out_dir, exist_ok=True)

    out_path_base = os.path.join(out_dir, f"selection_heatmap_{mode}.png")
    full_title = f"{exp_name} ({round_desc})"
    make_selection_plot(df_all, out_path_base, full_title)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AL で選択されたサンプルの True Score × Confidence 分布を可視化する"
    )
    parser.add_argument("--qid", type=str, default=None, help="Question ID (batch mode)")
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Input experiment directory to process (single mode)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (single mode only; default: <exp-dir>/plots)",
    )

    # どのラウンドを集計するか
    parser.add_argument(
        "--round-mode",
        type=str,
        default="all",
        choices=["all", "first", "last", "range"],
        help="Which rounds to aggregate: all / first / last / range",
    )
    parser.add_argument(
        "--round-k",
        type=int,
        default=5,
        help="K rounds for round-mode=first/last",
    )
    parser.add_argument(
        "--round-range",
        type=str,
        default=None,
        help="Round range 'start:end' (0-based, inclusive) when round-mode=range",
    )

    args = parser.parse_args()

    if args.exp_dir:
        # 単一モード
        if os.path.exists(args.exp_dir):
            exp_name = os.path.basename(args.exp_dir.rstrip(os.sep))
            out_d = args.out_dir if args.out_dir else os.path.join(args.exp_dir, "plots")
            _log(f"Processing single directory: {args.exp_dir}")
            process_experiment(args.exp_dir, exp_name, args, out_d)
        else:
            _log(f"ERROR: Directory not found: {args.exp_dir}")
            sys.exit(1)
    elif args.qid:
        # バッチモード
        base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
        if not os.path.exists(base_dir):
            _log(f"ERROR: base_dir not found: {base_dir}")
            sys.exit(1)

        # まず exp_* を探す。無ければ直下のディレクトリを全部対象とする。
        exp_dirs = sorted(
            d for d in glob.glob(os.path.join(base_dir, "exp_*")) if os.path.isdir(d)
        )
        if not exp_dirs:
            exp_dirs = sorted(
                d
                for d in glob.glob(os.path.join(base_dir, "*"))
                if os.path.isdir(d)
            )

        if not exp_dirs:
            _log(f"ERROR: No experiment dirs under {base_dir}")
            sys.exit(1)

        for exp_dir in exp_dirs:
            out_d = os.path.join(exp_dir, "analysis")
            exp_name = os.path.basename(exp_dir.rstrip(os.sep))
            _log(f"Processing exp_dir (batch): {exp_dir}")
            process_experiment(exp_dir, exp_name, args, out_d)
    else:
        print("Usage: --exp-dir (single) or --qid (batch)")
        sys.exit(1)


if __name__ == "__main__":
    main()
