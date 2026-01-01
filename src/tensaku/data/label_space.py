# /home/esakit25/work/tensaku/src/tensaku/data/label_space.py
# -*- coding: utf-8 -*-
"""tensaku.data.label_space

Label-space utilities for SAS experiments.

We intentionally separate:

* **Global label space** (per QID): derived from the oracle/master dataset for
  that QID (e.g., all.jsonl filtered by qid).
* **Observed labels**: labels actually present in a specific split or a
  specific AL round.

Historically these were conflated, which produced bugs like:
  - meta.json num_labels shrinking to the observed subset in early AL rounds
  - cfg.model.num_labels being injected from the shrunk value
  - training failing once a higher label appeared later

This module provides a small, well-typed representation and strict parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def coerce_int_label_strict(v: Any, *, label_key: str, ctx: str) -> int:
    """Coerce label to int without rounding.

    Accepts: int, integer-valued float, integer-like string.
    Rejects: bool, non-integer float, empty string, other types.
    """
    if v is None:
        raise ValueError(f"{ctx}: label '{label_key}' is None")
    if isinstance(v, bool):
        raise ValueError(f"{ctx}: label '{label_key}' must be int, got bool")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        raise ValueError(f"{ctx}: label '{label_key}' must be an integer-valued float, got {v}")
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError(f"{ctx}: label '{label_key}' is empty string")
        try:
            fv = float(s)
        except Exception as e:
            raise ValueError(f"{ctx}: label '{label_key}' cannot parse '{v}'") from e
        if fv.is_integer():
            return int(fv)
        raise ValueError(f"{ctx}: label '{label_key}' must be integer-like, got '{v}'")
    raise ValueError(f"{ctx}: label '{label_key}' has unsupported type: {type(v).__name__}")


@dataclass(frozen=True)
class LabelSpace:
    """Global label space representation."""

    label_min: int
    label_max: int
    num_labels: int
    unique_count: int
    unique_labels: List[int]

    @staticmethod
    def from_labels(labels: Sequence[int], *, require_start_at_zero: bool = True, ctx: str = "label_space") -> "LabelSpace":
        if not labels:
            raise ValueError(f"{ctx}: labels is empty")

        u = sorted(set(int(x) for x in labels))
        mn = u[0]
        mx = u[-1]

        if require_start_at_zero and mn != 0:
            raise ValueError(f"{ctx}: labels must start at 0 (min=0), got min={mn}")

        # We *do not* require contiguity here. Some datasets may skip labels.
        num = mx + 1
        return LabelSpace(label_min=mn, label_max=mx, num_labels=num, unique_count=len(u), unique_labels=u)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_min": int(self.label_min),
            "label_max": int(self.label_max),
            "num_labels": int(self.num_labels),
            "unique_count": int(self.unique_count),
            "unique_labels": list(self.unique_labels),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, ctx: str = "label_space") -> "LabelSpace":
        try:
            mn = int(d["label_min"])
            mx = int(d["label_max"])
            num = int(d["num_labels"])
            ucnt = int(d["unique_count"])
            u = [int(x) for x in list(d["unique_labels"])]
        except Exception as e:
            raise ValueError(f"{ctx}: invalid LabelSpace dict: {d}") from e
        if num != mx + 1:
            raise ValueError(f"{ctx}: num_labels must equal label_max+1, got num_labels={num}, label_max={mx}")
        if ucnt != len(u):
            raise ValueError(f"{ctx}: unique_count mismatch: {ucnt} != {len(u)}")
        return LabelSpace(label_min=mn, label_max=mx, num_labels=num, unique_count=ucnt, unique_labels=u)

    def contains_observed(self, observed: "LabelSpace") -> bool:
        """Return True if `observed` labels are a subset of this label space."""
        return set(observed.unique_labels).issubset(set(self.unique_labels))


def compute_label_space_from_records(
    records: Iterable[Mapping[str, Any]], *, label_key: str, ctx: str
) -> LabelSpace:
    labels: List[int] = []
    for i, obj in enumerate(records):
        if label_key not in obj:
            raise ValueError(f"{ctx}: missing label_key '{label_key}' at record index {i}")
        labels.append(coerce_int_label_strict(obj[label_key], label_key=label_key, ctx=ctx))
    return LabelSpace.from_labels(labels, ctx=ctx)


def compute_label_space_from_jsonl(
    *, path: str, label_key: str, qid: Optional[str] = None, qid_key: str = "qid", ctx: str
) -> LabelSpace:
    """Compute label space from a JSONL file.

    If `qid` is provided, the function filters by `obj[qid_key] == qid`.

    Strictness:
      - Requires label_key presence for included records.
      - If `qid` is provided, included records are those with matching qid.
        Missing qid_key will simply not match and will be excluded.
    """
    import json

    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if qid is not None:
                if str(obj.get(qid_key)) != str(qid):
                    continue
            if label_key not in obj:
                raise ValueError(f"{ctx}: missing label_key '{label_key}' at {path}:{ln}")
            labels.append(coerce_int_label_strict(obj[label_key], label_key=label_key, ctx=f"{ctx}:{path}:{ln}"))

    return LabelSpace.from_labels(labels, ctx=ctx)
