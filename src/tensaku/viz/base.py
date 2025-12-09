# /home/esakit25/work/tensaku/src/tensaku/viz/base.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.viz.base
@role  : 可視化のスタイル設定と共通ユーティリティ
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns

def setup_style():
    """共通のプロットスタイルを適用する。"""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
        'lines.linewidth': 2,
        'grid.linestyle': ':',
        'grid.alpha': 0.7,
    })

def save_fig(fig: plt.Figure, path: Union[str, Path], dpi: int = 300) -> None:
    """図を保存する共通関数（ディレクトリ自動作成）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    print(f"[viz] Saved plot: {p}")
    plt.close(fig)