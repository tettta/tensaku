# -*- coding: utf-8 -*-
"""
@module: tensaku.utils.cleaner
@role  : 実験中の中間ファイル（チェックポイント、推論結果など）を削除し、ディスク容量を管理する。
@note  : 削除タイミングには「即時 (Immediate)」と「遅延 (Delayed)」の2種類がある。
         - Immediate: タスク終了直後。サンプラーが使用しない巨大ファイル（ckpt等）を即消す。
         - Delayed  : サンプリング終了後。サンプラーが参照し終えた「前ラウンド」のファイルを消す。
"""

import shutil
import logging
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)


def cleanup_round_immediate(cfg: Mapping[str, Any], layout: Any, round_index: int) -> None:
    """
    【即時削除 (Immediate Cleanup)】
    設定キー: run.cleanup.immediate
    """
    run_cfg = cfg.get("run", {})
    # ★変更: per_round ではなく immediate を読む
    cleanup_cfg = run_cfg.get("cleanup", {}).get("immediate", {})

    if not cleanup_cfg:
        return

    # 1. Checkpoints
    if cleanup_cfg.get("checkpoints", False):
        ckpt_dir = layout.path_models_round_dir(round_index) / "checkpoints_min"
        if ckpt_dir.exists():
            try:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                LOGGER.debug(f"[Cleanup-Immediate] Removed checkpoints: {ckpt_dir}")
            except OSError as e:
                LOGGER.warning(f"[Cleanup-Immediate] Failed to remove {ckpt_dir}: {e}")

    # 2. Temp Data
    if cleanup_cfg.get("temp_data", False):
        temp_dir = layout.path_temp_round_dir(round_index)
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                LOGGER.debug(f"[Cleanup-Immediate] Removed temp dir: {temp_dir}")
            except OSError as e:
                LOGGER.warning(f"[Cleanup-Immediate] Failed to remove {temp_dir}: {e}")


def cleanup_round_delayed(cfg: Mapping[str, Any], layout: Any, round_index: int) -> None:
    """
    【遅延削除 (Delayed Cleanup)】
    設定キー: run.cleanup.delayed
    """
    target_round_index = round_index - 1
    if target_round_index < 0:
        return

    run_cfg = cfg.get("run", {})
    cleanup_cfg = run_cfg.get("cleanup", {}).get("delayed", {})

    if not cleanup_cfg:
        return

    LOGGER.info(f"[Cleanup-Delayed] Processing artifacts for previous round {target_round_index}...")

    # 3. Infer Arrays
    if cleanup_cfg.get("infer_arrays", False):
        infer_dir = layout.path_rounds_infer_dir(target_round_index)
        if infer_dir.exists():
            try:
                shutil.rmtree(infer_dir, ignore_errors=True)
                LOGGER.debug(f"  Removed infer arrays: {infer_dir}")
            except OSError as e:
                LOGGER.warning(f"  Failed to remove {infer_dir}: {e}")

    # 4. Global Arrays
    if cleanup_cfg.get("arrays", False):
        arrays_dir = getattr(layout, "arrays_rounds_dir", None)
        if arrays_dir and arrays_dir.exists():
            round_name = layout.round_name(target_round_index)
            prefix = f"{round_name}_"
            count = 0
            for p in arrays_dir.glob(f"{prefix}*.npy"):
                try:
                    p.unlink()
                    count += 1
                except OSError:
                    pass
            if count > 0:
                LOGGER.debug(f"  Removed {count} array files for {round_name}")


def cleanup_post_experiment(cfg: Mapping[str, Any], layout: Any) -> None:
    """
    【実験終了後削除 (Post-Experiment Cleanup)】
    
    実行タイミング:
        al.py のループ脱出後、終了処理の直前。
        
    目的:
        実験中に保持していたが、分析が終われば不要になる全ファイルを一括削除する。
        
    対象 (run.cleanup.post_experiment):
        - all_checkpoints: 全ラウンドの checkpoints
        - infer_arrays   : 全ラウンドの infer ディレクトリ
        - temp_dirs      : temp_data ディレクトリ全体
    """
    run_cfg = cfg.get("run", {})
    cleanup_cfg = run_cfg.get("cleanup", {}).get("post_experiment", {})

    if not cleanup_cfg:
        return

    LOGGER.info("[Cleanup-Post] Starting post-experiment cleanup...")
    out_dir = layout.root

    # 1. 全Checkpoints削除 (models/*/checkpoints_min)
    if cleanup_cfg.get("all_checkpoints", False):
        # models/round_*/checkpoints_min を探索
        count = 0
        for p in out_dir.glob("models/*/checkpoints_min"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                count += 1
        if count > 0:
            LOGGER.info(f"  Removed checkpoints in {count} directories.")

    # 2. 全Infer Arrays削除 (rounds/*/infer)
    if cleanup_cfg.get("infer_arrays", False):
        count = 0
        for p in out_dir.glob("rounds/*/infer"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                count += 1
        if count > 0:
            LOGGER.info(f"  Removed infer arrays in {count} directories.")

    # 3. Tempディレクトリ自体の削除
    if cleanup_cfg.get("temp_dirs", False):
        # temp_data は通常 out_dir 直下にあると想定
        temp_root = out_dir / "temp_data"
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            LOGGER.info(f"  Removed temp root dir: {temp_root}")