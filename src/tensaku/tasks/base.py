# /home/esakit25/work/tensaku/src/tensaku/tasks/base.py
# -*- coding: utf-8 -*-
"""tensaku.tasks.base

Base task interface and factory.

Design principles (MUST)
- No fallback: unknown task name is an error.
- Tasks are orchestrators: they may call lower-level modules, but pipelines must not depend on task internals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from tensaku.experiments.layout import ExperimentLayout
from tensaku.data.base import BaseDatasetAdapter, DatasetSplit
from tensaku.utils.strict_cfg import ConfigError, require_mapping, require_str

LOGGER = logging.getLogger(__name__)


@dataclass
class TaskOutputs:
    """Return type for Task.run_round.

    pool_scores:
        Mapping from score-key to {id -> score} mapping. (Implementation chooses actual representation;
        pipeline will consume per sampler requirements.)
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    pool_scores: Dict[Any, float] = field(default_factory=dict)
    pool_features: Optional[Any] = None
    pool_feature_ids: Optional[List[Any]] = None


class BaseTask:
    """Base class for all tasks."""

    name: str = "base"

    def __init__(self, cfg: Mapping[str, Any], adapter: BaseDatasetAdapter, layout: ExperimentLayout) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.layout = layout

    def run_round(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        raise NotImplementedError



def create_task(cfg: Mapping[str, Any], adapter: BaseDatasetAdapter, layout: ExperimentLayout) -> BaseTask:
    """Factory for tasks (STRICT)."""
    task_cfg = require_mapping(cfg, ("task",), ctx="cfg")
    name = require_str(task_cfg, ("name",), ctx="cfg.task").lower()

    if name == "sas_total_score":
        from tensaku.tasks.sas_total_score import SasTotalScoreTask
        # ★ keyword-only 対応
        return SasTotalScoreTask(cfg=cfg, adapter=adapter, layout=layout)

    if name == "standard":
        from tensaku.tasks.standard import StandardSupervisedAlTask
        # ★ keyword-only 対応
        return StandardSupervisedAlTask(cfg=cfg, adapter=adapter, layout=layout)

    raise ConfigError(f"Unknown task name: '{name}'. (no fallback allowed)")

