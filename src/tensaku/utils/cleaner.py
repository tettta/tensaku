# /home/esakit25/work/tensaku/src/tensaku/utils/cleaner.py
# -*- coding: utf-8 -*-
"""tensaku.utils.cleaner

@role:
  - Round-end / experiment-end file cleanup driven by:
      (1) Layout ledger (what artifacts were produced)
      (2) Config (which artifact kinds to delete)

@design:
  - No silent fallback for config keys: missing cleaner config is an error.
  - Layout must expose a readable ledger interface; otherwise error with actionable message.
  - Deleting a file that is already missing is NOT an error (missing_ok) because
    external processes or previous cleanups may have removed it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union
import logging
import shutil

from tensaku.utils.strict_cfg import ConfigError, require_mapping, require_list, require_bool, require_str

logger = logging.getLogger(__name__)


def clean_round_end(*, layout: Any, cfg: Mapping[str, Any], round_index: int) -> Dict[str, Any]:
    """Delete artifacts at round end.

    Required config:
      cleaner.round_end.enabled: bool
      cleaner.round_end.kinds: list[str]
    """
    cleaner_cfg = require_mapping(cfg, "cleaner")
    round_cfg = require_mapping(cleaner_cfg, "round_end")
    enabled = require_bool(round_cfg, "enabled")
    kinds = require_list(round_cfg, "kinds")
    kinds = [str(k) for k in kinds]

    if not enabled or not kinds:
        return {"enabled": enabled, "deleted": 0, "kinds": kinds}

    entries = list(_iter_round_entries(layout=layout, round_index=round_index))
    deleted = 0
    for e in entries:
        kind = _get_kind(e)
        if kind not in kinds:
            continue
        p = _get_path(e)
        if p is None:
            continue
        deleted += _delete_path(p)

    logger.info("[cleaner] round_end: round=%d kinds=%s deleted=%d", round_index, kinds, deleted)
    return {"enabled": enabled, "deleted": deleted, "kinds": kinds, "round": int(round_index)}


def clean_experiment_end(*, layout: Any, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Optional experiment-end cleanup.

    Required config:
      cleaner.experiment_end.enabled: bool
      cleaner.experiment_end.kinds: list[str]
    """
    cleaner_cfg = require_mapping(cfg, "cleaner")
    exp_cfg = require_mapping(cleaner_cfg, "experiment_end")
    enabled = require_bool(exp_cfg, "enabled")
    kinds = require_list(exp_cfg, "kinds")
    kinds = [str(k) for k in kinds]

    if not enabled or not kinds:
        return {"enabled": enabled, "deleted": 0, "kinds": kinds}

    entries = list(_iter_all_entries(layout=layout))
    deleted = 0
    for e in entries:
        kind = _get_kind(e)
        if kind not in kinds:
            continue
        p = _get_path(e)
        if p is None:
            continue
        deleted += _delete_path(p)

    logger.info("[cleaner] experiment_end: kinds=%s deleted=%d", kinds, deleted)
    return {"enabled": enabled, "deleted": deleted, "kinds": kinds}


# --------------------------------------------------------------------
# Ledger access (Layout-dependent)
# --------------------------------------------------------------------

def _iter_round_entries(*, layout: Any, round_index: int) -> Iterator[Any]:
    """Read ledger entries for a given round.

    We support several common Layout APIs. If none are found, raise ConfigError.
    """
    # Preferred: list_artifacts(round_index=...)
    if hasattr(layout, "list_artifacts") and callable(getattr(layout, "list_artifacts")):
        return iter(layout.list_artifacts(round_index=round_index))  # type: ignore

    # Alternative: iter_artifacts(round_index=...)
    if hasattr(layout, "iter_artifacts") and callable(getattr(layout, "iter_artifacts")):
        return iter(layout.iter_artifacts(round_index=round_index))  # type: ignore

    # Alternative: ledger_round(round_index)
    if hasattr(layout, "ledger_round") and callable(getattr(layout, "ledger_round")):
        return iter(layout.ledger_round(round_index))  # type: ignore

    # Alternative: ledger attribute that is list/dict with per-round mapping
    if hasattr(layout, "ledger"):
        led = getattr(layout, "ledger")
        if callable(led):
            # ledger(round_index=...)
            try:
                return iter(led(round_index=round_index))
            except TypeError:
                pass
        # dict-like: ledger[round_index]
        try:
            if isinstance(led, dict) and round_index in led:
                return iter(led[round_index])
        except Exception:
            pass

    raise ConfigError(
        "Layout does not expose a readable ledger interface required by cleaner. "
        "Expected one of: layout.list_artifacts(round_index=...), layout.iter_artifacts(round_index=...), "
        "layout.ledger_round(round_index), or layout.ledger. "
        "Please implement one of these in layout so cleaner can delete artifacts by kind."
    )


def _iter_all_entries(*, layout: Any) -> Iterator[Any]:
    """Read all ledger entries (best effort, but API must exist)."""
    if hasattr(layout, "list_artifacts") and callable(getattr(layout, "list_artifacts")):
        return iter(layout.list_artifacts())  # type: ignore
    if hasattr(layout, "iter_artifacts") and callable(getattr(layout, "iter_artifacts")):
        return iter(layout.iter_artifacts())  # type: ignore
    if hasattr(layout, "ledger"):
        led = getattr(layout, "ledger")
        if callable(led):
            try:
                return iter(led())
            except TypeError:
                pass
        if isinstance(led, list):
            return iter(led)
        if isinstance(led, dict):
            # flatten
            def _flat():
                for v in led.values():
                    if isinstance(v, list):
                        for e in v:
                            yield e
            return iter(list(_flat()))
    raise ConfigError(
        "Layout does not expose a readable ledger interface required by cleaner (all entries). "
        "Implement list_artifacts()/iter_artifacts()/ledger in layout."
    )


def _get_kind(entry: Any) -> str:
    if isinstance(entry, dict):
        k = entry.get("kind")
        return str(k) if k is not None else ""
    # object with attribute
    return str(getattr(entry, "kind", "") or "")


def _get_path(entry: Any) -> Optional[Path]:
    raw = None
    if isinstance(entry, dict):
        raw = entry.get("path") or entry.get("p") or entry.get("file")
    else:
        raw = getattr(entry, "path", None) or getattr(entry, "p", None) or getattr(entry, "file", None)

    if raw is None:
        return None

    # layout may store wrapper objects with `.path` attribute
    if hasattr(raw, "path"):
        try:
            raw2 = getattr(raw, "path")
            if isinstance(raw2, (str, Path)):
                return Path(raw2)
        except Exception:
            pass

    if isinstance(raw, Path):
        return raw
    return Path(str(raw))


def _delete_path(p: Path) -> int:
    """Delete file or directory. Return 1 if something was deleted, else 0."""
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            return 1
        # file
        p.unlink(missing_ok=True)
        return 1 if not p.exists() else 0
    except Exception as e:
        raise RuntimeError(f"Failed to delete artifact path: {p} ({e})") from e


# Backward compatibility alias (old name)
def cleanup_after_round(*, layout: Any, cfg: Mapping[str, Any], round_index: int) -> Dict[str, Any]:
    return clean_round_end(layout=layout, cfg=cfg, round_index=round_index)
