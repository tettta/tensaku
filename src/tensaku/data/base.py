# /home/esakit25/work/tensaku/src/tensaku/data/base.py
# -*- coding: utf-8 -*-
"""tensaku.data.base

Dataset adapter layer.

Design principles (STRICT)
- No fallback / no silent defaults: missing required config keys are errors.
- Pipelines/Tasks depend only on this Adapter contract, not on file formats.
- Pool is treated as *unlabeled* (label key should not exist in pool records).

Contract summary
- Adapter must provide:
  - make_initial_split() -> DatasetSplit (labeled/dev/test/pool records)
  - oracle_labels(ids) -> Mapping[id, label]  (external label acquisition)

Important NOTE (ID uniqueness)
- If cfg.data.input_all contains multiple QIDs with overlapping IDs, IDs are NOT globally unique.
  In that case, this adapter MUST scope the oracle index to cfg.data.qid using the 'qid' field
  in the oracle file, otherwise label injection becomes corrupted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import re

from tensaku.utils.strict_cfg import require_mapping, require_str


def _coerce_int_strict(v, *, ctx: str) -> int:
    """Coerce label to int with NO silent rounding.

    Accepts: int, float that is integer-valued, numeric strings ("14", "14.0").
    Raises: ValueError on anything else.
    """
    if isinstance(v, bool):
        raise ValueError(f"{ctx}: bool is not a valid integer label")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"{ctx}: non-integer float label: {v}")
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.0+", s):
            return int(float(s))
        raise ValueError(f"{ctx}: non-integer string label: {v!r}")
    raise ValueError(f"{ctx}: unsupported label type {type(v).__name__}")


@dataclass
class DatasetSplit:
    labeled: List[Dict[str, Any]] = field(default_factory=list)
    dev: List[Dict[str, Any]] = field(default_factory=list)
    test: List[Dict[str, Any]] = field(default_factory=list)
    pool: List[Dict[str, Any]] = field(default_factory=list)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise TypeError(f"JSONL row must be object/dict: {path}")
            rows.append(obj)
    return rows


class BaseDatasetAdapter:
    """Dataset adapter contract (STRICT)."""

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

        data_cfg = require_mapping(cfg, ("data",), ctx="cfg")
        self.id_key = require_str(data_cfg, ("id_key",), ctx="cfg.data")
        self.label_key = require_str(data_cfg, ("label_key",), ctx="cfg.data")

    def make_initial_split(self) -> DatasetSplit:
        raise NotImplementedError

    def oracle_labels(self, ids: Sequence[Any]) -> Mapping[Any, Any]:
        """Return {id -> label} for given ids.

        This is REQUIRED by contract. If not overridden, it is treated as unimplemented.
        """
        raise NotImplementedError("oracle_labels must be implemented by dataset adapter")


class StandardAdapter(BaseDatasetAdapter):
    """Standard adapter for jsonl splits.

    Expected (STRICT)
    - cfg.run.data_dir: directory containing split jsonl files
    - cfg.data.files: mapping with keys labeled/dev/test/pool pointing to jsonl filenames
    - cfg.data.input_all: absolute path to full dataset (oracle source)

    Additional STRICT requirement (to prevent silent label corruption)
    - cfg.data.qid: current question id
    - If cfg.data.input_all contains overlapping IDs across QIDs, oracle rows must include 'qid'.
    """

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__(cfg)
        data_cfg = require_mapping(cfg, ("data",), ctx="cfg")
        # NOTE: this project runs per-qid; require qid to prevent cross-qid label contamination.
        self.qid = require_str(data_cfg, ("qid",), ctx="cfg.data")
        self.input_all = require_str(data_cfg, ("input_all",), ctx="cfg.data")
        self._oracle_index: Optional[Dict[Any, Any]] = None

    def make_initial_split(self) -> DatasetSplit:
        cfg = self.cfg
        run_cfg = require_mapping(cfg, ("run",), ctx="cfg")
        data_cfg = require_mapping(cfg, ("data",), ctx="cfg")
        files = require_mapping(data_cfg, ("files",), ctx="cfg.data")

        data_dir = Path(require_str(run_cfg, ("data_dir",), ctx="cfg.run"))
        if not data_dir.exists():
            raise FileNotFoundError(f"run.data_dir not found: {data_dir}")

        labeled_name = require_str(files, ("labeled",), ctx="cfg.data.files")
        dev_name = require_str(files, ("dev",), ctx="cfg.data.files")
        test_name = require_str(files, ("test",), ctx="cfg.data.files")
        pool_name = require_str(files, ("pool",), ctx="cfg.data.files")

        labeled = _read_jsonl(data_dir / labeled_name)
        dev = _read_jsonl(data_dir / dev_name)
        test = _read_jsonl(data_dir / test_name)
        pool = _read_jsonl(data_dir / pool_name)


        # STRICT: validate split meta against cfg to prevent mixing different sources
        meta_path = data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"split meta.json not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        meta_qid = meta.get("qid")
        if meta_qid != self.qid:
            raise ValueError(f"split meta qid mismatch: meta={meta_qid!r} cfg={self.qid!r}")

        meta_input_all = meta.get("input_all")
        if not meta_input_all:
            raise KeyError("split meta missing 'input_all'")
        if Path(meta_input_all).resolve() != Path(self.input_all).resolve():
            raise ValueError(
                "split meta input_all mismatch: "
                f"meta={Path(meta_input_all).resolve()} cfg={Path(self.input_all).resolve()}"
            )

        meta_label_key = meta.get("label_key")
        if meta_label_key != self.label_key:
            raise ValueError(f"split meta label_key mismatch: meta={meta_label_key!r} cfg={self.label_key!r}")

        # STRICT: pool must be unlabeled
        for rec in pool:
            if self.label_key in rec:
                raise ValueError(
                    f"pool must be unlabeled (no '{self.label_key}') but found in record id={rec.get(self.id_key)}"
                )

        return DatasetSplit(labeled=labeled, dev=dev, test=test, pool=pool)

    def _build_oracle_index(self) -> Dict[Any, Any]:
        cfg = self.cfg
        data_cfg = require_mapping(cfg, ("data",), ctx="cfg")
        input_all = Path(require_str(data_cfg, ("input_all",), ctx="cfg.data"))
        if not input_all.is_absolute():
            raise ValueError(f"cfg.data.input_all must be absolute in strict mode: {input_all}")

        rows = _read_jsonl(input_all)

        # First pass: detect duplicates by id_key (regardless of qid)
        seen: Dict[Any, Any] = {}
        dup_ids: List[Any] = []
        for r in rows:
            if self.id_key not in r:
                raise KeyError(f"oracle record missing id_key '{self.id_key}'")
            if self.label_key not in r:
                raise KeyError(f"oracle record missing label_key '{self.label_key}'")
            rid = r[self.id_key]
            if rid in seen:
                dup_ids.append(rid)
            else:
                seen[rid] = True

        # If oracle contains overlapping IDs across multiple qids, scope by qid to avoid contamination.
        # We treat this as STRICT: if duplicates exist, 'qid' must exist in oracle rows.
        if dup_ids:
            # Require 'qid' field in oracle records when IDs are not globally unique
            missing_qid = [r for r in rows[:50] if "qid" not in r]  # cheap probe
            if missing_qid:
                preview = dup_ids[:10]
                raise RuntimeError(
                    "oracle id_key is not globally unique, but oracle rows do not contain 'qid'. "
                    "This would silently corrupt label injection.\n"
                    f"- qid (cfg.data.qid) = {self.qid!r}\n"
                    f"- example duplicate ids = {preview}\n"
                    "Fix: either (A) include 'qid' in cfg.data.input_all rows, or (B) make IDs unique per record (e.g., 'qid:id')."
                )

            # Filter to current qid
            rows_q = [r for r in rows if r.get("qid") == self.qid]
            if not rows_q:
                raise RuntimeError(
                    f"oracle contains duplicate ids across qids, but no rows matched qid={self.qid!r} in {input_all}"
                )
            rows = rows_q

        idx: Dict[Any, Any] = {}
        conflicts: List[tuple] = []
        for r in rows:
            if self.id_key not in r:
                raise KeyError(f"oracle record missing id_key '{self.id_key}'")
            if self.label_key not in r:
                raise KeyError(f"oracle record missing label_key '{self.label_key}'")
            rid = r[self.id_key]
            y = r[self.label_key]
            try:
                y_int = _coerce_int_strict(y, ctx=f'input_all[{rid}].{self.label_key}')
            except Exception as e:
                raise TypeError(f"label is not int-castable for id={rid}: {y}") from e

            if rid in idx and idx[rid] != y_int:
                conflicts.append((rid, idx[rid], y_int))
            else:
                idx[rid] = y_int

        if conflicts:
            sample = conflicts[:10]
            raise RuntimeError(
                "oracle index has conflicting labels for the same id within the scoped oracle set.\n"
                f"qid={self.qid!r} id_key={self.id_key!r} label_key={self.label_key!r}\n"
                f"examples(id, old, new)={sample}"
            )

        if len(idx) == 0:
            raise ValueError(f"oracle index is empty after scoping: {input_all} (qid={self.qid!r})")

        return idx

    def oracle_labels(self, ids: Sequence[Any]) -> Mapping[Any, Any]:
        if self._oracle_index is None:
            self._oracle_index = self._build_oracle_index()

        out: Dict[Any, Any] = {}
        missing: List[Any] = []
        for i in ids:
            if i in self._oracle_index:
                out[i] = self._oracle_index[i]
            else:
                missing.append(i)

        if missing:
            preview = ", ".join(map(str, missing[:10]))
            more = "" if len(missing) <= 10 else f" (+{len(missing)-10} more)"
            raise KeyError(f"oracle missing labels for ids: {preview}{more}")

        return out


def create_adapter(cfg: Mapping[str, Any]) -> BaseDatasetAdapter:
    """Strict factory for dataset adapter."""
    # Currently we only support StandardAdapter, and we do not provide fallbacks.
    return StandardAdapter(cfg)
