# /home/esakit25/work/tensaku/src/tensaku/tasks/factory.py
# -*- coding: utf-8 -*-
"""tensaku.tasks.factory

Task factory (STRICT).

- No fallback: unknown task name is an error.
- Keeps a stable import path for pipelines:
    from tensaku.tasks.factory import create_task, create_task_patched
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from tensaku.experiments.layout import ExperimentLayout
from tensaku.data.base import BaseDatasetAdapter
from tensaku.utils.strict_cfg import ConfigError, require_mapping, require_str

LOGGER = logging.getLogger(__name__)


def create_task(cfg: Mapping[str, Any], adapter: BaseDatasetAdapter, layout: ExperimentLayout):
    """Create a task instance from cfg (STRICT)."""
    task_cfg = require_mapping(cfg, ("task",), ctx="cfg")
    name = require_str(task_cfg, ("name",), ctx="cfg.task").lower()

    if name == "sas_total_score":
        from tensaku.tasks.sas_total_score import SasTotalScoreTask

        return SasTotalScoreTask(cfg=cfg, adapter=adapter, layout=layout)

    if name == "standard":
        from tensaku.tasks.standard import StandardSupervisedAlTask

        return StandardSupervisedAlTask(cfg=cfg, adapter=adapter, layout=layout)

    raise ConfigError(f"Unknown task name: '{name}'. (no fallback allowed)")


# Backward-compat alias used by older pipeline code
create_task_patched = create_task

__all__ = ["create_task", "create_task_patched"]
