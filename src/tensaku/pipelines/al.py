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

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tensaku.data.base import create_adapter
from tensaku.al.loop import run_one_step
from tensaku.al.label_acquisition import acquire_labels
from tensaku.al.schedule import create_scheduler
from tensaku.al.state import ALState, init_state_from_split
from tensaku.experiments.layout import ExperimentLayout
from tensaku.utils.strict_cfg import require_mapping, require_str
from tensaku.utils.cleaner import clean_round_end, clean_experiment_end
from tensaku.tasks.base import create_task


LOGGER = logging.getLogger(__name__)


def run_experiment(cfg: Mapping[str, Any]) -> int:
    cfg_dict = dict(cfg)

    out_root = Path(require_str(cfg_dict, ("run", "out_dir"), ctx="cfg.run"))
    out_root.mkdir(parents=True, exist_ok=True)
    layout = ExperimentLayout(root=out_root)

    adapter = create_adapter(cfg_dict)

    qid = require_str(cfg_dict, ("data", "qid"), ctx="cfg.data")
    id_key = require_str(cfg_dict, ("data", "id_key"), ctx="cfg.data")
    # ラベル注入のためにキー名を取得
    label_key = require_str(cfg_dict, ("data", "label_key"), ctx="cfg.data")

    split0 = adapter.make_initial_split()

    task = create_task(cfg=cfg_dict, adapter=adapter, layout=layout)

    al_cfg = require_mapping(cfg_dict, ("al",), ctx="cfg")
    rounds = int(al_cfg["rounds"])
    budget = int(al_cfg["budget"])

    state: ALState = init_state_from_split(split0, id_key=id_key, round_index=0)
    
    scheduler = create_scheduler(al_cfg)

    _init_al_history(layout, state=state)

    all_selected_rows: List[Dict[str, Any]] = []

    for r in range(rounds):
        LOGGER.info("=== [Round %d] Start ===", r)

        # 1. Split構築 (Universe対応版)
        split_r = _split_from_state(split0, state=state, id_key=id_key)

        # 2. ラベル注入 (Label Injection)
        # Pool 由来の行には label_key が無い（＝未ラベル）ので、Oracle で補完する。
        # 既に label_key を持つ行は上書きしない。不一致があればデータ不整合なので即エラー。
        all_labeled_ids = list(state.labeled_ids)
        labels_map = acquire_labels(adapter=adapter, ids=all_labeled_ids)

        missing_in_oracle = []
        mismatched = []

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
                f"[Label Injection] oracle missing {len(missing_in_oracle)} ids. "
                f"example={missing_in_oracle[:5]}"
            )
        if mismatched:
            raise RuntimeError(
                f"[Label Injection] labeled vs oracle label mismatch {len(mismatched)} rows. "
                f"example(id, labeled, oracle)={mismatched[:5]}"
            )
        out = task.run_round(round_index=r, split=split_r)

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
        if not selected_ids and state.n_pool == 0:
            LOGGER.info("AL Loop finished early (pool exhausted).")
            break

        _save_selected_ids(layout, round_index=r, selected_ids=selected_ids)
        
        # 選択された行を保存する際も Universe から探す必要があるため _rows_by_ids の呼び出し元(split0)に注意
        # ここでは split0.pool から探すので、初期Poolにあったデータなら見つかるはず
        selected_rows = _rows_by_ids(split0.pool, id_key=id_key, ids=selected_ids)
        _save_selected_rows(layout, round_index=r, selected_rows=selected_rows)
        all_selected_rows.extend(selected_rows)

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

    _save_selected_all(layout, selected_rows=all_selected_rows)
    _finalize_experiment(layout, last_round=rounds-1)

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


def _init_al_history(layout: ExperimentLayout, *, state: ALState) -> None:
    f = layout.al_history
    f.ensure_parent()
    if not f.exists():
        with f.open("w", encoding="utf-8") as w:
            w.write("round,n_labeled,n_pool,added,coverage\n")
    _append_al_history(layout, state=state, added=0)


def _append_al_history(layout: ExperimentLayout, *, state: ALState, added: int) -> None:
    f = layout.al_history
    with f.open("a", encoding="utf-8") as w:
        w.write(f"{state.round_index},{state.n_labeled},{state.n_pool},{added},{state.coverage:.6f}\n")


def _save_selected_ids(layout: ExperimentLayout, *, round_index: int, selected_ids: List[Any]) -> None:
    f = layout.selection_round_sample_ids(round=round_index)
    f.ensure_parent()
    with f.open("w", encoding="utf-8") as w:
        for sid in selected_ids:
            w.write(str(sid) + "\n")


def _save_selected_rows(layout: ExperimentLayout, *, round_index: int, selected_rows: List[Dict[str, Any]]) -> None:
    f = layout.selection_round_samples(round=round_index)
    f.ensure_parent()
    import pandas as pd
    pd.DataFrame(selected_rows).to_csv(f.path, index=False)


def _save_selected_all(layout: ExperimentLayout, *, selected_rows: List[Dict[str, Any]]) -> None:
    f = layout.selection_all_samples
    f.ensure_parent()
    import pandas as pd
    pd.DataFrame(selected_rows).to_csv(f.path, index=False)


def _finalize_experiment(layout: ExperimentLayout, *, last_round: int) -> None:
    # Copy last-round prediction artifacts to stable "final" locations (best-effort).
    try:
        src_pred = layout.predictions_round_detail(round=last_round)
        dst_pred = layout.preds_detail_final
        if src_pred.exists():
            dst_pred.ensure_parent()
            shutil.copy(src_pred.path, dst_pred.path)
    except Exception:
        pass

    try:
        src_gate = layout.predictions_round_gate_assign(round=last_round)
        dst_gate = layout.gate_assign_final
        if src_gate.exists():
            dst_gate.ensure_parent()
            shutil.copy(src_gate.path, dst_gate.path)
    except Exception:
        pass