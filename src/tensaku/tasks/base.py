# /home/esakit25/work/tensaku/src/tensaku/tasks/base.py
# -*- coding: utf-8 -*-
"""tensaku.tasks.base
=================================
@module: tensaku.tasks.base
@role  : Active Learning / HITL パイプラインで利用する Task 抽象クラスと
         典型フロー用テンプレートクラス、および最小実装 (DummyTask) を提供する。

@overview:
    - Task は「与えられた DatasetSplit を使って 1 ラウンド分の処理
      （学習・推論・評価など）を行う」役割を持つ。
    - Active Learning パイプライン (tensaku.pipelines.al) からは、
      BaseTask.run_round(...) を通じて呼び出される。

@inputs (for BaseTask.run_round):
    - round_index: int
    - split      : tensaku.data.base.DatasetSplit

@outputs:
    - TaskOutputs: metrics/artifacts/pool_scores を格納したコンテナ。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import logging

from tensaku.data.base import DatasetSplit, BaseDatasetAdapter
from tensaku.experiments.layout import ExperimentLayout

LOGGER = logging.getLogger(__name__)


# ======================================================================
# TaskOutputs: Task の戻り値コンテナ
# ======================================================================


@dataclass
class TaskOutputs:
    """Task が 1 ラウンドの処理結果として返す出力。

    Attributes
    ----------
    metrics : Dict[str, Any]
        1 ラウンドの要約指標（coverage / rmse / qwk / cse 等）。
        上位レイヤ（例: al_history）の集計対象。
    artifacts : Dict[str, Any]
        ログ出力や後処理で参照したい追加情報。
        例: preds_detail のパス、HITL の詳細結果、図表ファイル名など。
    pool_scores : Optional[Mapping[Any, float]]
        Pool データに対する「優先度スコア」（ID -> float）。
        Active Learning のサンプラーが利用する。
        - None の場合: ランダムサンプリング等が適用される。
        - 値がある場合: 不確実性サンプリング等で利用される。
    """

    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    pool_scores: Optional[Mapping[Any, float]] = None


# ======================================================================
# BaseTask 抽象クラス
# ======================================================================


class BaseTask:
    """AL / HITL パイプラインから呼び出される Task の抽象基底クラス（極薄版）。"""

    #: 識別名（ログや config で利用）
    name: str = "base"

    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: BaseDatasetAdapter,
        layout: ExperimentLayout,
    ) -> None:
        """cfg / adapter / layout を保持するだけの極薄基底クラス。"""
        self.cfg: Mapping[str, Any] = cfg
        self.adapter: BaseDatasetAdapter = adapter
        self.layout: ExperimentLayout = layout

    # サブクラスで override する想定
    def run_round(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        """1 ラウンド分の処理を実行する。

        Args:
            round_index: ラウンド番号（0-origin）
            split      : 現在の labeled/dev/test/pool を表す DatasetSplit

        Returns:
            TaskOutputs: 指標や成果物。
        """
        raise NotImplementedError


# ======================================================================
# 典型フロー用テンプレート: Train→Infer→Confidence→HITL
# ======================================================================


class TrainInferHitlTask(BaseTask):
    """典型的な「train→infer→confidence→HITL」フローを持つ Task のテンプレ。

    派生クラスは必要に応じて以下の hook を override する:

        - step_train
        - step_infer
        - step_confidence
        - step_hitl  (戻り値: TaskOutputs)

    run_round(...) はこれらを順に呼び出す Template Method として実装される。
    """

    def run_round(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        LOGGER.info("Task[%s]: round=%d を開始します。", self.name, round_index)

        self.step_train(round_index, split)
        self.step_infer(round_index, split)
        self.step_confidence(round_index, split)
        outputs = self.step_hitl(round_index, split)

        LOGGER.info("Task[%s]: round=%d を終了しました。", self.name, round_index)
        return outputs

    # ---- hook 群（必要に応じて派生クラスで実装） -------------------

    def step_train(self, round_index: int, split: DatasetSplit) -> None:  # pragma: no cover - デフォルト実装
        """学習ステップ。

        典型的には tensaku.train を呼び出し、チェックポイントを更新する。
        デフォルト実装は何もしない（学習不要タスク用）。
        """
        LOGGER.debug("TrainInferHitlTask.step_train: round=%d (no-op)", round_index)

    def step_infer(self, round_index: int, split: DatasetSplit) -> None:  # pragma: no cover - デフォルト実装
        """推論ステップ。

        典型的には tensaku.infer_pool を呼び出し、preds_detail を生成する。
        デフォルト実装は何もしない。
        """
        LOGGER.debug("TrainInferHitlTask.step_infer: round=%d (no-op)", round_index)

    def step_confidence(self, round_index: int, split: DatasetSplit) -> None:  # pragma: no cover - デフォルト実装
        """確信度計算ステップ。

        典型的には tensaku.confidence を呼び出し、conf_* カラムを preds_detail に付与する。
        デフォルト実装は何もしない。
        """
        LOGGER.debug("TrainInferHitlTask.step_confidence: round=%d (no-op)", round_index)

    def step_hitl(self, round_index: int, split: DatasetSplit) -> TaskOutputs:  # pragma: no cover - デフォルト実装
        """HITL ステップ。

        典型的には tensaku.pipelines.hitl を用いて preds_detail から HitlOutputs を得て、
        TaskOutputs にまとめて返す。
        デフォルト実装は NotImplementedError を送出する。
        """
        raise NotImplementedError(
            "TrainInferHitlTask.step_hitl を派生クラスで実装してください。"
        )


# ======================================================================
# DummyTask（開発・デバッグ用）
# ======================================================================


class DummyTask(BaseTask):
    """何もしない Task 実装（開発・デバッグ用）。

    metrics には単にデータ件数を格納する。
    """

    name: str = "dummy"

    def run_round(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        n_labeled = len(split.labeled)
        n_dev = len(split.dev)
        n_test = len(split.test)
        n_pool = len(split.pool)

        LOGGER.info(
            "DummyTask: round=%d  labeled=%d dev=%d test=%d pool=%d",
            round_index,
            n_labeled,
            n_dev,
            n_test,
            n_pool,
        )

        metrics: Dict[str, float] = {
            "round": float(round_index),
            "n_labeled": float(n_labeled),
            "n_dev": float(n_dev),
            "n_test": float(n_test),
            "n_pool": float(n_pool),
        }

        artifacts: Dict[str, Any] = {
            "layout_root": str(self.layout.root),
        }

        # ダミーの pool_scores
        pool_scores: Dict[Any, float] = {}

        return TaskOutputs(metrics=metrics, artifacts=artifacts, pool_scores=pool_scores)


# ======================================================================
# Task Factory
# ======================================================================


def create_task(
    cfg: Mapping[str, Any],
    adapter: BaseDatasetAdapter,
    layout: ExperimentLayout,
    initial_split: Optional[DatasetSplit] = None,  # noqa: ARG001 - 将来拡張用
) -> BaseTask:
    """cfg から Task インスタンスを構築するファクトリ関数。

    Args:
        cfg          : 実験全体の config。
        adapter      : DatasetAdapter 実装インスタンス。
        layout       : ExperimentLayout（出力レイアウト）。
        initial_split: 初期の DatasetSplit（必要に応じて Task 側で利用）。

    Returns:
        BaseTask インスタンス。
    """
    task_cfg: Mapping[str, Any] = cfg.get("task", {}) if isinstance(cfg, Mapping) else {}
    name_raw = task_cfg.get("name")

    if not name_raw:
        # 互換のため run.task_name も見る
        run_cfg: Mapping[str, Any] = cfg.get("run", {}) if isinstance(cfg, Mapping) else {}
        name_raw = run_cfg.get("task_name", "dummy")

    name = str(name_raw).lower() if name_raw is not None else "dummy"

    # タスク名に応じて適切な Task クラスを返す
    if name in {"sas_total_score", "sas_total"}:
        LOGGER.info("Task 'sas_total_score' を使用します。")
        from tensaku.tasks.sas_total_score import SasTotalScoreTask  # type: ignore

        return SasTotalScoreTask(cfg=cfg, adapter=adapter, layout=layout)

    if name in {"dummy", "", None}:
        LOGGER.info("Task 'dummy' を使用します。")
        return DummyTask(cfg=cfg, adapter=adapter, layout=layout)

    LOGGER.warning(
        "未知の Task 名 '%s' が指定されました。DummyTask にフォールバックします。",
        name,
    )
    return DummyTask(cfg=cfg, adapter=adapter, layout=layout)