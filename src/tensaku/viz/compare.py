# /home/esakit25/work/tensaku/src/tensaku/viz/compare.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.viz.compare
@role  : 複数のAL実験結果 (al_learning_curve.csv) を収集・集約するロジック
"""
from __future__ import annotations
import glob
import os
import re
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np

def collect_experiments(
    root_dir: Union[str, Path],
    qid: str,
    filename: str = "al_learning_curve.csv",
    filter_str: Optional[str] = None,
    exclude_str: Optional[str] = None
) -> pd.DataFrame:
    """
    指定された QID 以下の全実験ディレクトリから学習曲線を収集・結合する。
    
    Args:
        root_dir: プロジェクトルート
        qid: 問題ID (例: "Y14_1-2_1_3")
        filename: 収集対象のCSVファイル名
        filter_str: 実験名にこの文字列を含むもののみ対象 (部分一致)
        exclude_str: 実験名にこの文字列を含むものを除外
    
    Returns:
        結合された DataFrame ("exp_name" 列が追加される)
    """
    base_dir = Path(root_dir) / "outputs" / f"q-{qid}"
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    # 再帰的ではなく、直下の実験フォルダ(exp_*)を探すのが基本だが、
    # 柔軟性のため glob で探索する
    search_pattern = base_dir / "*" / filename
    files = glob.glob(str(search_pattern))
    
    if not files:
        return pd.DataFrame()

    df_list = []
    for fpath in files:
        exp_name = os.path.basename(os.path.dirname(fpath))
        
        if filter_str and (filter_str not in exp_name):
            continue
        if exclude_str and (exclude_str in exp_name):
            continue
            
        try:
            df = pd.read_csv(fpath)
            df["exp_name"] = exp_name
            
            # 数値型への強制変換 (エラー回避)
            cols_numeric = [
                "n_labeled", "budget", "round",
                "test_qwk_full", "test_rmse_full", "test_cse_full", 
                "test_acc_full", "test_f1_full", "aurc_cse"
            ]
            for c in cols_numeric:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    
            df_list.append(df)
        except Exception as e:
            print(f"[compare] WARN: Failed to read {fpath}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def aggregate_experiments(
    df: pd.DataFrame,
    group_by: List[str] = ["sampler", "uncertainty_key"],
    x_axis: str = "n_labeled"
) -> pd.DataFrame:
    """
    収集した実験データをグループ化し、平均と標準偏差を計算する。
    描画時に帯グラフ (Mean ± Std) を出すために使用。
    """
    if df.empty:
        return pd.DataFrame()

    # 表示用ラベルの生成
    def _make_label(row):
        keys = [str(row.get(k, "?")) for k in group_by]
        return "-".join(keys)

    df["plot_label"] = df.apply(_make_label, axis=1)
    
    # 数値カラムのみ集約対象にする
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # x軸自体はグループ化キーに含める必要がある
    keys = ["plot_label", x_axis]
    
    # 平均と標準偏差を一括計算
    agg_df = df.groupby(keys)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
    
    # カラム名をフラットにする (例: test_qwk_full_mean, test_qwk_full_std)
    new_cols = []
    for c in agg_df.columns:
        if isinstance(c, tuple):
            if c[1] == "": new_cols.append(c[0])
            else: new_cols.append(f"{c[0]}_{c[1]}")
        else:
            new_cols.append(c)
            
    agg_df.columns = new_cols
    return agg_df