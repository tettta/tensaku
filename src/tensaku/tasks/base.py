# /home/esakit25/work/tensaku/src/tensaku/tasks/base.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.tasks.base
@role  : タスク実行の基底クラス (Template Method パターン)
@overview:
    - BaseTask: 最小限のインターフェース
    - TrainInferHitlTask: 標準的な AL ループのテンプレート (Train -> Infer -> Confidence -> HITL)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from tensaku.data.base import BaseDatasetAdapter, DatasetSplit
from tensaku.experiments.layout import ExperimentLayout

LOGGER = logging.getLogger(__name__)


@dataclass
class TaskOutputs:
    """各ラウンドの Task 実行結果。"""
    metrics: Dict[str, Any] = field(default_factory=dict)
    pool_scores: Dict[Any, float] = field(default_factory=dict)


class BaseTask:
    """すべての Task の基底クラス。"""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: BaseDatasetAdapter,
        layout: ExperimentLayout,
    ) -> None:

        self.cfg = cfg
        self.adapter = adapter
        self.layout = layout

    def run_round(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        """1 ラウンド分の処理を実行し、評価指標とサンプリング用スコアを返す。"""
        raise NotImplementedError


class TrainInferHitlTask(BaseTask):
    """
    標準的な教師あり学習 AL のテンプレートタスク。
    run_round で「学習 -> 推論 -> 信頼度計算 -> HITL判定」の順にメソッドを呼ぶ。
    """

    def run_round(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        # 1. Train (学習)
        self.step_train(round_index, split)

        # 2. Infer (推論)
        self.step_infer(round_index, split)
        
        # 3. Confidence (信頼度計算) - ★新規追加
        #    step_infer で得られた結果に対して、追加の指標(TrustScore等)を付与する
        self.step_confidence(round_index, split)

        # 4. HITL & Return (判定と結果返却)
        return self.step_hitl(round_index, split)

    def step_train(self, round_index: int, split: DatasetSplit) -> None:
        """学習を実行し、モデルを作成・保存するステップ。"""
        raise NotImplementedError

    def step_infer(self, round_index: int, split: DatasetSplit) -> None:
        """推論を実行し、予測結果を保持するステップ。"""
        raise NotImplementedError

    def step_confidence(self, round_index: int, split: DatasetSplit) -> None:
        """推論結果に対して信頼度・不確実性を計算・付与するステップ。"""
        # 実装はサブクラスに任せるが、何もしない(pass)も許容
        pass

    def step_hitl(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        """予測結果に基づき HITL 判定を行い、Metrics と Pool Scores を返すステップ。"""
        raise NotImplementedError


# ======================================================================
# Factory
# ======================================================================

def create_task(
    cfg: Mapping[str, Any],
    adapter: BaseDatasetAdapter,
    layout: ExperimentLayout,
) -> BaseTask:
    task_cfg = cfg.get("task", {})
    name = str(task_cfg.get("name", "base")).lower()

    if name == "sas_total_score":
        from tensaku.tasks.sas_total_score import SasTotalScoreTask
        return SasTotalScoreTask(cfg, adapter, layout)

    LOGGER.warning(f"Unknown task '{name}'. Falling back to BaseTask (will crash).")
    return BaseTask(cfg, adapter, layout)