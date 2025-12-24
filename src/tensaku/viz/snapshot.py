# /home/esakit25/work/tensaku/src/tensaku/viz/snapshot.py
# -*- coding: utf-8 -*-
"""tensaku.viz.snapshot

@role
  - "途中の時点"（round / n_labeled）に対応する preds_detail を決定する。
  - viz 側のスナップショット指定のための小ユーティリティ。

@strict
  - 指定 round / n_labeled に対応するファイルが無ければ例外で落とす（黙って代替しない）。
  - --nearest を明示した場合のみ、最も近い時点へ丸める。

@outputs
  - 返すのは Path（preds_detail の CSV）と、選ばれた round / n_labeled。

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd


@dataclass(frozen=True)
class SnapshotPick:
    preds_detail_csv: Path
    round: int
    n_labeled: Optional[int]
    tag: str  # directory-friendly identifier


_ROUND_RE = re.compile(r"round_(\d+)_preds_detail\.csv$")


def list_round_preds(exp_dir: Path) -> Dict[int, Path]:
    """Return mapping round -> preds_detail.csv path."""
    rounds_dir = exp_dir / "predictions" / "rounds"
    out: Dict[int, Path] = {}
    if not rounds_dir.exists():
        return out
    for p in rounds_dir.glob("round_*_preds_detail.csv"):
        m = _ROUND_RE.search(p.name)
        if not m:
            continue
        r = int(m.group(1))
        out[r] = p
    return out


def find_latest_preds_detail(exp_dir: Path) -> SnapshotPick:
    """Deterministic selection used when snapshot is not requested."""
    cand_final = exp_dir / "predictions" / "final" / "preds_detail.csv"
    if cand_final.exists():
        return SnapshotPick(preds_detail_csv=cand_final, round=-1, n_labeled=None, tag="final")
    rounds = list_round_preds(exp_dir)
    if not rounds:
        raise FileNotFoundError(f"preds_detail not found under: {exp_dir}")
    r = max(rounds.keys())
    return SnapshotPick(preds_detail_csv=rounds[r], round=r, n_labeled=None, tag=f"round_{r:03d}")


def _load_learning_curve(exp_dir: Path) -> pd.DataFrame:
    path = exp_dir / "metrics" / "al_learning_curve.csv"
    if not path.exists():
        raise FileNotFoundError(
            "snapshot by n_labeled requires metrics/al_learning_curve.csv (missing): " + str(path)
        )
    df = pd.read_csv(path)
    for col in ("round", "n_labeled"):
        if col not in df.columns:
            raise KeyError(f"learning_curve missing required column: {col} in {path}")
    return df


def _pick_round_by_n_labeled(
    *,
    exp_dir: Path,
    n_labeled: int,
    nearest: bool,
) -> Tuple[int, Optional[int]]:
    df = _load_learning_curve(exp_dir)
    # normalize
    df = df[["round", "n_labeled"]].dropna()
    if df.empty:
        raise ValueError(f"learning_curve is empty after dropna: {exp_dir}/metrics/al_learning_curve.csv")

    df["round"] = df["round"].astype(int)
    df["n_labeled"] = df["n_labeled"].astype(int)

    # exact
    m = df[df["n_labeled"] == int(n_labeled)]
    if not m.empty:
        # if multiple, pick the latest round
        r = int(m["round"].max())
        return r, int(n_labeled)

    if not nearest:
        avail = sorted(set(int(x) for x in df["n_labeled"].tolist()))
        raise ValueError(f"n_labeled={n_labeled} not found. available n_labeled (unique): {avail}")

    # nearest (explicit): choose floor if possible; otherwise smallest greater.
    le = df[df["n_labeled"] <= int(n_labeled)]
    if not le.empty:
        row = le.sort_values(["n_labeled", "round"], ascending=[True, True]).iloc[-1]
    else:
        row = df.sort_values(["n_labeled", "round"], ascending=[True, True]).iloc[0]
    return int(row["round"]), int(row["n_labeled"])


def pick_snapshot(
    *,
    exp_dir: Path,
    snapshot_round: Optional[int],
    snapshot_n_labeled: Optional[int],
    nearest: bool,
) -> SnapshotPick:
    """Pick preds_detail for the requested snapshot."""
    exp_dir = exp_dir.resolve()
    if snapshot_round is not None and snapshot_n_labeled is not None:
        raise ValueError("snapshot_round and snapshot_n_labeled are mutually exclusive")

    if snapshot_round is None and snapshot_n_labeled is None:
        return find_latest_preds_detail(exp_dir)

    rounds = list_round_preds(exp_dir)
    if not rounds:
        raise FileNotFoundError(f"predictions/rounds not found or empty: {exp_dir}/predictions/rounds")

    if snapshot_round is not None:
        req = int(snapshot_round)
        if req in rounds:
            return SnapshotPick(preds_detail_csv=rounds[req], round=req, n_labeled=None, tag=f"round_{req:03d}")
        if not nearest:
            avail = sorted(rounds.keys())
            raise ValueError(f"round={req} not found. available rounds: {avail}")
        # nearest (explicit): pick max <= req
        cand = [r for r in rounds.keys() if r <= req]
        if not cand:
            avail = sorted(rounds.keys())
            raise ValueError(f"round={req} not found and no rounds <= req. available rounds: {avail}")
        r = max(cand)
        return SnapshotPick(preds_detail_csv=rounds[r], round=r, n_labeled=None, tag=f"round_{r:03d}")

    assert snapshot_n_labeled is not None
    r, picked_n = _pick_round_by_n_labeled(exp_dir=exp_dir, n_labeled=int(snapshot_n_labeled), nearest=nearest)
    if r not in rounds:
        avail = sorted(rounds.keys())
        raise FileNotFoundError(
            f"round file for round={r} not found under predictions/rounds. available rounds: {avail}"
        )
    return SnapshotPick(
        preds_detail_csv=rounds[r],
        round=r,
        n_labeled=picked_n,
        tag=f"n_labeled_{int(picked_n)}_round_{r:03d}",
    )
