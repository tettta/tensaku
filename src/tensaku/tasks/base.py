# /home/esakit25/work/tensaku/src/tensaku/tasks/base.py
# -*- coding: utf-8 -*-
"""tensaku.tasks.base

Base task interface.

Design principles (MUST)
- No fallback: unknown task name is an error.
- Tasks are orchestrators: they may call lower-level modules, but pipelines must not depend on task internals.

NOTE:
- Task factory has been moved to `tensaku.tasks.factory`.
- `create_task` remains here only as a *compat wrapper* (imports at call-time to avoid circular imports).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from tensaku.experiments.layout import ExperimentLayout
from tensaku.data.base import BaseDatasetAdapter, DatasetSplit

LOGGER = logging.getLogger(__name__)


@dataclass
class TaskOutputs:
    metrics: Dict[str, Any] = field(default_factory=dict)
    pool_scores: Dict[Any, float] = field(default_factory=dict)
    pool_features: Optional[Any] = None
    pool_feature_ids: Optional[List[Any]] = None
    detail_df: Optional[Any] = None

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
    """Compatibility wrapper. Prefer `tensaku.tasks.factory.create_task`."""
    from tensaku.tasks.factory import create_task as _create_task

    return _create_task(cfg=cfg, adapter=adapter, layout=layout)
