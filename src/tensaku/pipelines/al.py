# /home/esakit25/work/tensaku/src/tensaku/pipelines/al.py
# -*- coding: utf-8 -*-
"""tensaku.pipelines.al

@role:
  - Run Active Learning (AL) loop.
  - Keep orchestration thin: delegate dataset I/O to adapter, training/inference/confidence to task,
    and selection+label acquisition to tensaku.al.loop.

Design:
  - Lower modules (adapter/task/al.loop) receive full cfg (Mapping) and read their needed sections.
  - No fallback: required cfg keys must exist.
"""

from __future__ import annotations

import csv
import logging
import math
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from tensaku.data.base import create_adapter
from tensaku.al.loop import run_one_step
from tensaku.al.label_acquisition import acquire_labels
from tensaku.al.schedule import create_scheduler
from tensaku.al.state import ALState, init_state_from_split
from tensaku.experiments.layout import ExperimentLayout
from tensaku.experiments.index import append_experiment_index, utc_now_iso
from tensaku.utils.strict_cfg import require_int, require_mapping, require_str
from tensaku.utils.cleaner import clean_round_end, clean_experiment_end
from tensaku.tasks.base import create_task


LOGGER = logging.getLogger(__name__)


def run_experiment(cfg: Mapping[str, Any], *, layout: ExperimentLayout) -> int:
    """Run Active Learning experiment.

    Notes
    -----
    - `layout` is created by `main` (root path decision is a main responsibility).
    - This function is strict: required cfg keys must exist, and errors are raised.
    """
    cfg_dict = dict(cfg)

    # Adapter / Task
    adapter = create_adapter(cfg_dict)

    id_key = require_str(cfg_dict, ("data", "id_key"), ctx="cfg.data")
    label_key = require_str(cfg_dict, ("data", "label_key"), ctx="cfg.data")

    split0 = adapter.make_initial_split()
    task = create_task(cfg=cfg_dict, adapter=adapter, layout=layout)

    al_cfg = require_mapping(cfg_dict, ("al",), ctx="cfg")
    rounds = int(al_cfg["rounds"])
    budget = int(al_cfg["budget"])

    state: ALState = init_state_from_split(split0, id_key=id_key, round_index=0)
    scheduler = create_scheduler(al_cfg)

    _init_al_history(layout, state=state)

    last_round = -1
    for r in range(rounds):
        last_round = r
        LOGGER.info("=== [Round %d] Start ===", r)

        # 1) Build split for this round from state (Universe-aware)
        split_r = _split_from_state(split0, state=state, id_key=id_key)

        # 2) Label injection: labeled rows must have labels (oracle as source of truth).
        all_labeled_ids = list(state.labeled_ids)
        labels_map = acquire_labels(adapter=adapter, ids=all_labeled_ids)

        missing_in_oracle: List[Any] = []
        mismatched: List[Any] = []

        for row in split_r.labeled:
            rid = row.get(id_key)
            if rid is None:
                raise RuntimeError(f"[Label Injection] missing id_key='{id_key}' in a labeled row")
            if rid not in labels_map:
                missing_in_oracle.append(rid)
                continue

            if label_key in row:
                if int(row[label_key]) != int(labels_map[rid]):
                    mismatched.append((rid, row[label_key], labels_map[rid]))
            else:
                row[label_key] = labels_map[rid]

        if missing_in_oracle:
            raise RuntimeError(
                f"[Label Injection] oracle missing {len(missing_in_oracle)} ids. example={missing_in_oracle[:5]}"
            )
        if mismatched:
            raise RuntimeError(
                f"[Label Injection] labeled vs oracle label mismatch {len(mismatched)} rows. "
                f"example(id, labeled, oracle)={mismatched[:5]}"
            )

        # 3) Task executes training/inference/confidence and writes artifacts via layout
        out = task.run_round(round_index=r, split=split_r)

        # 3.5) Record model-only learning curve (computed from preds_detail.csv written by task)
        _append_al_learning_curve(layout, cfg=cfg_dict, round_index=r, state_before=state)

        # 4) Select next samples
        sampler = scheduler.get_sampler_for_round(round_idx=r, cfg=cfg_dict)

        state, selected_ids = run_one_step(
            state=state,
            adapter=adapter,
            sampler=sampler,
            budget=budget,
            scores=out.pool_scores if out.pool_scores else None,
            features=out.pool_features,
            feature_ids=out.pool_feature_ids,
            as_new_round=True,
        )

        # Early stop: pool exhausted
        if not selected_ids and state.n_pool == 0:
            LOGGER.info("AL Loop finished early (pool exhausted).")
            break

        _save_selected_ids(layout, round_index=r, selected_ids=selected_ids)

        selected_rows = _rows_by_ids(split0.pool, id_key=id_key, ids=selected_ids)
        _save_selected_rows(layout, round_index=r, selected_rows=selected_rows)
        _append_selected_all(layout, selected_rows=selected_rows)
        # 早期に参照を切る（大きい列がある場合のメモリ高水位の抑制）
        del selected_rows

        _append_al_history(layout, state=state, added=len(selected_ids))

        LOGGER.info(
            "=== [Round %d] Done: added=%d n_labeled=%d n_pool=%d coverage=%.4f ===",
            r,
            len(selected_ids),
            state.n_labeled,
            state.n_pool,
            state.coverage,
        )

        # Round-end cleanup (ledger + config driven; immediate deletion; no delayed cleanup)
        _ = clean_round_end(layout=layout, cfg=cfg_dict, round_index=r)

    if last_round >= 0:
        _finalize_experiment(layout, last_round=last_round)

    # Experiment-end cleanup (optional; config-driven)
    _ = clean_experiment_end(layout=layout, cfg=cfg_dict)

    return 0

def _rows_by_ids(rows: List[Mapping[str, Any]], *, id_key: str, ids: List[Any]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    wanted = set(ids)
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        if r.get(id_key) in wanted:
            # 浅いコピーを返すことで、後続のラベル注入などが元データ(split0)を汚染しないようにする
            out.append(dict(r))
    return out


def _split_from_state(base_split: Any, *, state: ALState, id_key: str) -> Any:
    # 【修正】Universe対応: 初期データ(labeled)と追加データ(pool)の両方から探す
    universe = base_split.labeled + base_split.pool
    
    labeled = _rows_by_ids(universe, id_key=id_key, ids=list(state.labeled_ids))
    pool = _rows_by_ids(universe, id_key=id_key, ids=list(state.pool_ids))
    
    # Dev/Test は固定
    dev = _rows_by_ids(base_split.dev, id_key=id_key, ids=list(state.dev_ids))
    test = _rows_by_ids(base_split.test, id_key=id_key, ids=list(state.test_ids))
    
    from tensaku.data.base import DatasetSplit
    return DatasetSplit(labeled=labeled, dev=dev, test=test, pool=pool)


def _read_preds_detail_csv(path: Path) -> pd.DataFrame:
    # Required columns in preds_detail: split, y_true, y_pred
    df = pd.read_csv(path)
    required = {"split", "y_true", "y_pred"}
    miss = required.difference(df.columns)
    if miss:
        raise RuntimeError(f"preds_detail.csv missing columns: {sorted(miss)}")
    return df


def _compute_split_metrics(df: pd.DataFrame, *, split: str, cse_abs_err: int) -> dict:
    sub = df[df["split"] == split]
    if sub.empty:
        return {"n": 0, "qwk": None, "rmse": None, "cse": None}
    # y_true may be NaN for pool; exclude
    sub = sub.dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        return {"n": 0, "qwk": None, "rmse": None, "cse": None}
    y_true = sub["y_true"].astype(int).to_numpy()
    y_pred = sub["y_pred"].astype(int).to_numpy()
    n = int(len(y_true))
    # QWK: requires at least 2 samples and at least 2 unique labels to be meaningful
    qwk = None
    if n >= 2 and (len(set(y_true.tolist())) >= 2 or len(set(y_pred.tolist())) >= 2):
        try:
            qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        except Exception:
            qwk = None
    rmse = float(math.sqrt(float(np.mean((y_pred - y_true) ** 2)))) if n else None
    cse = float(np.mean(np.abs(y_pred - y_true) >= int(cse_abs_err))) if n else None
    return {"n": n, "qwk": qwk, "rmse": rmse, "cse": cse}


def _write_csv_overwrite(file, *, header, rows) -> None:
    # record only on first creation to avoid duplicate ledger entries
    file.save_csv(rows=rows, header=header, record=(not file.exists()))


def _append_al_learning_curve(layout: ExperimentLayout, *, cfg: dict, round_index: int, state_before: ALState) -> None:
    """Write metrics/al_learning_curve.csv (model-only) for each round.

    We compute metrics from the per-round preds_detail.csv that task already wrote.
    This keeps pipeline thin and avoids coupling to TaskOutputs schema.
    """
    # cse_abs_err is optional; if absent we record the used value explicitly.
    gate = cfg.get("gate") if isinstance(cfg.get("gate"), dict) else {}
    cse_abs_err = int(gate.get("cse_abs_err", 2))

    pred_f = layout.predictions_round_detail(round=round_index)
    if not pred_f.exists():
        # If task didn't write, do nothing (but this indicates a bug in task).
        LOGGER.warning("[al_learning_curve] preds_detail missing for round=%d: %s", round_index, pred_f.path)
        return

    df = _read_preds_detail_csv(pred_f.path)
    dev = _compute_split_metrics(df, split="dev", cse_abs_err=cse_abs_err)
    te = _compute_split_metrics(df, split="test", cse_abs_err=cse_abs_err)

    f = layout.al_learning_curve
    header = [
        "round",
        "n_labeled",
        "n_pool",
        "coverage",
        "cse_abs_err",
        "n_dev",
        "dev_qwk",
        "dev_rmse",
        "dev_cse",
        "n_test",
        "test_qwk",
        "test_rmse",
        "test_cse",
    ]

    row = {
        "round": int(round_index),
        "n_labeled": int(state_before.n_labeled),
        "n_pool": int(state_before.n_pool),
        "coverage": float(state_before.coverage),
        "cse_abs_err": int(cse_abs_err),
        "n_dev": dev["n"],
        "dev_qwk": dev["qwk"],
        "dev_rmse": dev["rmse"],
        "dev_cse": dev["cse"],
        "n_test": te["n"],
        "test_qwk": te["qwk"],
        "test_rmse": te["rmse"],
        "test_cse": te["cse"],
    }

    rows = []
    if f.exists():
        old = pd.read_csv(f.path)
        if "round" in old.columns:
            old = old[old["round"].astype(int) != int(round_index)]
        rows = old.values.tolist()

    new_row = [row.get(k) for k in header]
    rows.append(new_row)
    _write_csv_overwrite(f, header=header, rows=rows)


def _init_al_history(layout: ExperimentLayout, *, state: ALState) -> None:
    f = layout.al_history
    header = ["round", "n_labeled", "n_pool", "added", "coverage"]
    # create empty file with header once
    if not f.exists():
        _write_csv_overwrite(f, header=header, rows=[])
    _append_al_history(layout, state=state, added=0)


def _append_al_history(layout: ExperimentLayout, *, state: ALState, added: int) -> None:
    f = layout.al_history
    header = ["round", "n_labeled", "n_pool", "added", "coverage"]

    rows = []
    if f.exists():
        try:
            old = pd.read_csv(f.path)
            # remove existing same round (upsert)
            if "round" in old.columns:
                old = old[old["round"].astype(int) != int(state.round_index)]
            rows = old.values.tolist()
        except Exception:
            rows = []

    rows.append([
        int(state.round_index),
        int(state.n_labeled),
        int(state.n_pool),
        int(added),
        float(state.coverage),
    ])

    _write_csv_overwrite(f, header=header, rows=rows)


def _save_selected_ids(layout: ExperimentLayout, *, round_index: int, selected_ids: List[Any]) -> None:
    f = layout.selection_round_sample_ids(round=round_index)
    text = "".join(str(sid) + "\n" for sid in selected_ids)
    f.save_text(text, record=(not f.exists()))

def _save_selected_rows(layout: ExperimentLayout, *, round_index: int, selected_rows: List[Dict[str, Any]]) -> None:
    f = layout.selection_round_samples(round=round_index)
    df = pd.DataFrame(selected_rows)
    f.save_csv(rows=df.values.tolist(), header=list(df.columns), record=(not f.exists()))


def _append_selected_all(layout: ExperimentLayout, *, selected_rows: List[Dict[str, Any]]) -> None:
    """選択済みサンプル（行データ）を「全体」ファイルへ追記する。

    目的
    ---
    - 以前は全round分の selected_rows を in-memory に蓄積して最後に一括保存していたが、
      SASの行は本文・根拠配列などが重く、round数が増えるとRSSが増え続ける原因になりやすい。
    - ここでは **メモリに蓄積せず**、roundごとに `selection/al_samples_all.csv` へ追記する。

    仕様
    ---
    - 初回のみヘッダ付きで作成（layoutの記録=ledgerもこの時点で行う）。
    - 2回目以降はファイル末尾へ追記する（ヘッダは再出力しない）。
    - ヘッダ（列名）が変わった場合はサイレントに吸収せず、即エラー。
    """
    if not selected_rows:
        return

    f = layout.selection_all_samples
    df = pd.DataFrame(selected_rows)
    cols = list(df.columns)

    # 初回は layout 経由で作成して ledger に記録
    if not f.exists():
        f.save_csv(rows=df.values.tolist(), header=cols, record=True)
        return

    # 既存ヘッダを1行だけ読む（全体を pandas で読むと巨大化し得るため避ける）
    try:
        with f.path.open("r", encoding="utf-8", newline="") as rf:
            reader = csv.reader(rf)
            header = next(reader, None)
    except Exception as e:
        raise RuntimeError(f"failed to read header: {f.path}") from e

    if not header:
        raise RuntimeError(f"selection_all_samples is empty or header missing: {f.path}")

    if header != cols:
        raise RuntimeError(
            "selection_all_samples header mismatch. "
            f"file={f.path} file_header={header} current_cols={cols}"
        )

    # 追記
    with f.path.open("a", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerows(df.values.tolist())


def _finalize_experiment(layout: ExperimentLayout, *, last_round: int) -> None:
    # Best-effort: if task already wrote final artifacts, do nothing.
    try:
        if not layout.preds_detail_final.exists():
            src_pred = layout.predictions_round_detail(round=last_round)
            if src_pred.exists():
                df = _read_preds_detail_csv(src_pred.path)
                layout.preds_detail_final.save_csv(rows=df.values.tolist(), header=list(df.columns), record=True)
    except Exception:
        pass

    try:
        if not layout.gate_assign_final.exists():
            src_gate = layout.predictions_round_gate_assign(round=last_round)
            if src_gate.exists():
                df = pd.read_csv(src_gate.path)
                layout.gate_assign_final.save_csv(rows=df.values.tolist(), header=list(df.columns), record=True)
    except Exception:
        pass
