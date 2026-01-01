# /home/esakit25/work/tensaku/src/tensaku/viz/base.py
# -*- coding: utf-8 -*-
"""tensaku.viz.base

Small utilities shared by visualization modules.

Principles
- Keep matplotlib configuration centralized.
- Raise clear errors for missing required columns / files.
- Avoid heavy dependencies (no seaborn).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import re

import pandas as pd


def ensure_split_dirs(root: Path) -> dict:
    """Ensure standard output subdirectories under a root directory.

    Rationale: keep outputs readable by separating file types.

    Layout:
      <root>/figures : image files (png, pdf, ...)
      <root>/tables  : csv/tsv tables
      <root>/meta    : json and other metadata

    Returns a dict with keys: root, figures, tables, meta.
    """
    root = ensure_dir(Path(root))
    figures = ensure_dir(root / "figures")
    tables = ensure_dir(root / "tables")
    meta = ensure_dir(root / "meta")
    return {"root": root, "figures": figures, "tables": tables, "meta": meta}


def setup_style() -> None:
    """Set a conservative matplotlib style without extra dependencies."""
    import matplotlib.pyplot as plt

    # Default matplotlib is fine; we only set a few readability knobs.
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 11,
        }
    )


def require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"[{ctx}] " if ctx else ""
        raise KeyError(f"{prefix}missing required columns: {missing}. available={list(df.columns)}")


def read_csv(path: Path, *, ctx: str = "") -> pd.DataFrame:
    if not path.exists():
        prefix = f"[{ctx}] " if ctx else ""
        raise FileNotFoundError(f"{prefix}csv not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        prefix = f"[{ctx}] " if ctx else ""
        raise RuntimeError(f"{prefix}failed to read csv: {path} ({e})") from e


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_token(s: str) -> str:
    """Filename-safe token (keeps alnum, '-', '_', '.', '=')."""
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^0-9A-Za-z\-_.=]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:180] if len(s) > 180 else s


def join_tokens(xs: Iterable[str], *, empty: str = "") -> str:
    ys = [sanitize_token(x) for x in xs if x is not None and str(x).strip() != ""]
    return empty if not ys else "_".join(ys)
