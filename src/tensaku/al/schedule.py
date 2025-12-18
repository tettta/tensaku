# /home/esakit25/work/tensaku/src/tensaku/al/schedule.py
# -*- coding: utf-8 -*-
"""tensaku.al.schedule

Sampler scheduling (round -> sampler spec).

Design principles (STRICT)
- No fallback / no silent defaults:
  - `al.schedule` rules require `start` and `sampler`.
  - `al.sampler` is mutually exclusive with `al.schedule`.
- Upper orchestrators pass the full cfg Mapping; this module reads only `al.*`.
- This module must NOT mutate cfg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, List

import logging

from tensaku.al.sampler import BaseSampler, create_sampler_from_spec

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class ScheduleRule:
    """One rule for [start, end] inclusive; end=None means no upper bound."""
    start: int
    end: Optional[int]
    sampler_spec: Any  # str or Mapping

    def matches(self, round_idx: int) -> bool:
        if round_idx < self.start:
            return False
        if self.end is None:
            return True
        return round_idx <= self.end

class SamplerScheduler:
    """Resolve which sampler to use at each round."""

    def __init__(self, al_cfg: Mapping[str, Any]) -> None:
        if not isinstance(al_cfg, Mapping):
            raise TypeError("al_cfg must be Mapping")

        has_schedule = "schedule" in al_cfg
        has_sampler = "sampler" in al_cfg
        if has_schedule and has_sampler:
            raise KeyError("Config 'al' must not contain both 'schedule' and 'sampler' (ambiguous).")

        rules: List[ScheduleRule] = []

        if has_schedule:
            sched_list = al_cfg["schedule"]
            if not isinstance(sched_list, list) or not sched_list:
                raise ValueError("al.schedule must be a non-empty list of rules.")

            for i, rule in enumerate(sched_list):
                if not isinstance(rule, Mapping):
                    raise TypeError(f"al.schedule[{i}] must be Mapping, got {type(rule)}")
                if "start" not in rule:
                    raise KeyError(f"al.schedule[{i}] missing required key 'start'")
                if "sampler" not in rule:
                    raise KeyError(f"al.schedule[{i}] missing required key 'sampler'")
                start = int(rule["start"])
                end_val = rule.get("end")
                end = int(end_val) if end_val is not None else None
                if start < 0:
                    raise ValueError(f"al.schedule[{i}].start must be >=0, got {start}")
                if end is not None and end < start:
                    raise ValueError(f"al.schedule[{i}].end must be >= start, got end={end} start={start}")
                rules.append(ScheduleRule(start=start, end=end, sampler_spec=rule["sampler"]))

            # Strict: enforce non-overlapping (to avoid ambiguous matches)
            rules_sorted = sorted(rules, key=lambda r: (r.start, 10**9 if r.end is None else r.end))
            for a, b in zip(rules_sorted, rules_sorted[1:]):
                a_end = 10**9 if a.end is None else a.end
                if b.start <= a_end:
                    raise ValueError(
                        "al.schedule rules overlap: "
                        f"[{a.start},{a.end}] and [{b.start},{b.end}] (start must be > previous end)"
                    )
            self._rules = rules_sorted
            LOGGER.info("Initialized SamplerScheduler with %d scheduled rules.", len(self._rules))
            return

        if has_sampler:
            sampler_spec = al_cfg["sampler"]
            self._rules = [ScheduleRule(start=0, end=None, sampler_spec=sampler_spec)]
            LOGGER.info("Initialized SamplerScheduler with single sampler.")
            return

        raise KeyError("Config 'al' missing both 'schedule' and 'sampler'. One is required.")

    def get_sampler_for_round(self, round_idx: int, *, cfg: Mapping[str, Any]) -> BaseSampler:
        if not isinstance(round_idx, int) or round_idx < 0:
            raise ValueError(f"round_idx must be non-negative int, got {round_idx!r}")
        for rule in self._rules:
            if rule.matches(round_idx):
                return create_sampler_from_spec(cfg=cfg, sampler_spec=rule.sampler_spec)
        # unreachable if rules include start=0..∞, but keep strict guard
        raise RuntimeError(f"No schedule rule matched round {round_idx}. Check al.schedule.")

def create_scheduler(al_cfg: Mapping[str, Any]) -> SamplerScheduler:
    """Factory wrapper for compatibility."""
    return SamplerScheduler(al_cfg)
