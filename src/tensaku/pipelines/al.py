# /home/esakit25/work/tensaku/src/tensaku/pipelines/al.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.pipelines.al
@role  : Active Learning (AL) 実験パイプラインのエントリポイント
@overview:
    - cfg（YAML + --set 済み）を受け取り、1 本の AL 実験を実行する。
    - データ層 / タスク層 / AL層 を組み合わせて、ラウンドごとにループを回す。
    - Task から返された pool_scores を Sampler に渡すことで、
      不確実性サンプリング等を実現する。
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml  # PyYAML

from tensaku.experiments.layout import ExperimentLayout
from tensaku.data.base import DatasetSplit, create_adapter
from tensaku.tasks.base import BaseTask, TaskOutputs, create_task
from tensaku.al.state import ALState, init_state_from_split
from tensaku.al.sampler import create_sampler
from tensaku.al.loop import run_one_step


LOGGER_NAME = "tensaku.al.pipeline"


# ======================================================================
# ロガー
# ======================================================================


def _setup_logger(log_path: Path, verbose: bool = True) -> logging.Logger:
    """パイプライン全体のロガーをセットアップする。"""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # 既存ハンドラがあればクリア（重複防止）
    if logger.handlers:
        for h in logger.handlers:
            logger.removeHandler(h)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File Handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stream Handler
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


# ======================================================================
# config / meta ダンプ
# ======================================================================


def _dump_exp_config(layout: ExperimentLayout, cfg: Mapping[str, Any]) -> None:
    """有効設定 cfg を YAML として保存する。"""
    path = layout.path_exp_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)


def _dump_run_meta(layout: ExperimentLayout, cfg: Mapping[str, Any]) -> None:
    """実行時メタ情報を JSON として保存する。"""
    run_cfg = cfg.get("run", {})
    run_id: Optional[str] = run_cfg.get("run_id")

    meta: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "out_dir": str(layout.root),
    }

    path = layout.path_run_meta()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ======================================================================
# DatasetSplit / ALState 変換ユーティリティ
# ======================================================================


def _build_record_index(
    split: DatasetSplit,
    id_key: str = "id",
) -> Dict[Any, Mapping[str, Any]]:
    """DatasetSplit 全体から id -> record のインデックスを構築する。"""
    index: Dict[Any, Mapping[str, Any]] = {}

    def _add(records: Sequence[Mapping[str, Any]]) -> None:
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            if id_key in rec:
                index[rec[id_key]] = rec

    _add(split.labeled)
    _add(split.dev)
    _add(split.test)
    _add(split.pool)

    return index


def _split_from_state(
    state: ALState,
    index: Mapping[Any, Mapping[str, Any]],
) -> DatasetSplit:
    """ALState の ID 集合から、DatasetSplit を再構築する。"""
    def _lookup(ids: Sequence[Any]):
        return [index[i] for i in ids if i in index]

    labeled = _lookup(state.labeled_ids)
    dev = _lookup(state.dev_ids)
    test = _lookup(state.test_ids)
    pool = _lookup(state.pool_ids)

    return DatasetSplit(
        labeled=labeled,
        dev=dev,
        test=test,
        pool=pool,
    )


# ======================================================================
# al_history.csv の追記
# ======================================================================


def _append_al_history(
    layout: ExperimentLayout,
    round_index: int,
    prev_state: ALState,
    new_state: ALState,
    metrics: Dict[str, Any],
) -> None:
    """
    metrics/al_history.csv に 1 行追記する。
    Task から返された metrics も統合して記録する。
    """
    path = layout.path_metrics_al_history()
    path.parent.mkdir(parents=True, exist_ok=True)

    added = new_state.n_labeled - prev_state.n_labeled
    coverage = new_state.coverage

    # 基本カラム
    base_row = {
        "round": round_index,
        "n_labeled": new_state.n_labeled,
        "n_pool": new_state.n_pool,
        "added": added,
        "coverage": f"{coverage:.6f}",
    }
    
    # Task metrics をマージ (競合時は Task 優先)
    row = {**base_row, **metrics}

    file_exists = path.exists()
    fieldnames = list(row.keys())
    
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # 既存ファイルがある場合、ヘッダ不一致のリスクがあるが今回は許容
        writer.writerow(row)


# ======================================================================
# メイン: AL 実験 1 本を実行
# ======================================================================


def run_experiment(
    cfg: Mapping[str, Any],
    argv: Optional[List[str]] = None,  # CLI互換用 (未使用)
) -> int:
    """
    Active Learning 実験を 1 本実行するメイン関数。

    Args:
        cfg: 実行設定 (YAMLロード済み)
        argv: CLI引数 (cli.py から渡されるが本関数では cfg を優先するため未使用)

    Returns:
        int: 終了コード (0: 成功, 1: 失敗)
    """
    # 1) レイアウト構築 & ディレクトリ作成
    layout = ExperimentLayout.from_cfg(cfg)
    layout.ensure_all_dirs()

    # 2) ロガー準備
    logger = _setup_logger(layout.path_log_pipeline())
    logger.info("=== Tensaku AL pipeline (Phase2) ===")
    run_cfg = cfg.get("run", {})
    run_id = run_cfg.get("run_id")
    if run_id is not None:
        logger.info("run_id = %s", run_id)
    logger.info("out_dir = %s", layout.root)

    # 3) 設定 & メタ情報のダンプ
    _dump_exp_config(layout, cfg)
    _dump_run_meta(layout, cfg)

    # 4) DatasetAdapter を構築し、初期 DatasetSplit を取得
    try:
        adapter = create_adapter(cfg)
        logger.info("DatasetAdapter: %s", adapter.__class__.__name__)
        split0: DatasetSplit = adapter.make_initial_split()
    except Exception:
        logger.exception("Dataset initialization failed.")
        return 1

    logger.info(
        "Initial DatasetSplit sizes: labeled=%d, dev=%d, test=%d, pool=%d",
        len(split0.labeled),
        len(split0.dev),
        len(split0.test),
        len(split0.pool),
    )

    # 5) ALState と id->record index を構築
    id_key = getattr(adapter, "id_key", "id")
    state = init_state_from_split(split0, id_key=id_key, round_index=0)
    index = _build_record_index(split0, id_key=id_key)
    logger.info(
        "ALState initialized: n_labeled=%d, n_pool=%d, total_n=%d",
        state.n_labeled,
        state.n_pool,
        state.total_n,
    )

    # 6) Task と Sampler を構築
    try:
        task: BaseTask = create_task(cfg=cfg, adapter=adapter, layout=layout)
        logger.info("Task: %s (name=%s)", task.__class__.__name__, getattr(task, "name", "?"))
        
        sampler = create_sampler(cfg)
        logger.info("Sampler: %s (name=%s)", sampler.__class__.__name__, getattr(sampler, "name", "?"))
    except Exception:
        logger.exception("Task or Sampler initialization failed.")
        return 1

    al_cfg = cfg.get("al", {})
    rounds = int(al_cfg.get("rounds", 1))
    budget = int(al_cfg.get("budget", 0))
    logger.info("AL rounds = %d, budget per round = %d", rounds, budget)

    # 7) ラウンドループ
    for r in range(rounds):
        logger.info("=== [round %d] ===", r)
        
        # (a) Task 実行
        split_r = _split_from_state(state, index)
        try:
            out: TaskOutputs = task.run_round(round_index=r, split=split_r)
        except Exception:
            logger.exception("Task.run_round failed at round %d", r)
            return 1

        if out.metrics:
            logger.info("  Task metrics: %s", out.metrics)

        # (b) Sampling (pool -> labeled)
        #     Task から返された pool_scores を sampler に渡す
        new_state, selected_ids = run_one_step(
            state=state,
            sampler=sampler,
            budget=budget,
            scores=out.pool_scores,  # ★不確実性スコアを渡す
            as_new_round=True,
        )

        logger.info(
            "  AL Step: selected=%d samples. Coverage: %.3f -> %.3f",
            len(selected_ids),
            state.coverage,
            new_state.coverage,
        )

        # (c) 履歴保存
        #     Task の metrics も一緒に al_history.csv に残す
        _append_al_history(
            layout, 
            round_index=r, 
            prev_state=state, 
            new_state=new_state, 
            metrics=out.metrics
        )
        
        # 選択されたIDの保存
        if selected_ids:
            sel_path = layout.path_selection_round_sample_ids(r)
            sel_path.parent.mkdir(parents=True, exist_ok=True)
            with sel_path.open("w", encoding="utf-8") as f:
                for sid in selected_ids:
                    f.write(f"{sid}\n")

        state = new_state

    logger.info("AL pipeline finished successfully.")
    return 0