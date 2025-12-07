# /home/esakit25/work/tensaku/src/tensaku/al/loop.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.al.loop
@role  : Active Learning (AL) の「ラウンド進行ロジック」を提供する層
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from tensaku.al.state import ALState, move_from_pool_to_labeled
from tensaku.al.sampler import BaseSampler


LOGGER = logging.getLogger(__name__)


def run_one_step(
    state: ALState,
    sampler: BaseSampler,
    budget: int,
    scores: Optional[Mapping[Any, float]] = None,
    as_new_round: bool = True,
) -> Tuple[ALState, List[Any]]:
    if budget <= 0 or state.n_pool == 0:
        new_state = ALState(
            round_index=state.round_index + 1 if as_new_round else state.round_index,
            labeled_ids=list(state.labeled_ids),
            pool_ids=list(state.pool_ids),
            dev_ids=list(state.dev_ids),
            test_ids=list(state.test_ids),
        )
        return new_state, []

    selected_ids = sampler.select(state=state, scores=scores, budget=budget)
    new_state = move_from_pool_to_labeled(
        state=state,
        selected_ids=selected_ids,
        as_new_round=as_new_round,
    )
    return new_state, list(selected_ids)


def run_loop(
    initial_state: ALState,
    sampler: BaseSampler,
    rounds: int,
    budget: int,
    score_provider: Optional[Callable[[ALState], Mapping[Any, float]]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    log = logger or LOGGER

    state = initial_state
    history: List[Dict[str, Any]] = []

    for r in range(rounds):
        if score_provider is not None:
            scores = score_provider(state)
        else:
            scores = None

        new_state, selected_ids = run_one_step(
            state=state,
            sampler=sampler,
            budget=budget,
            scores=scores,
            as_new_round=True,
        )

        summary = new_state.to_dict()
        summary.update(
            {
                "round": r,
                "selected_count": len(selected_ids),
            }
        )
        history.append(summary)

        log.info(
            "[AL loop] round=%d, n_labeled=%d, n_pool=%d, selected=%d, coverage=%.3f",
            r,
            new_state.n_labeled,
            new_state.n_pool,
            len(selected_ids),
            new_state.coverage,
        )

        state = new_state

    return history
