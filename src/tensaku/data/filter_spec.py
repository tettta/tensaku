# /home/esakit25/work/tensaku/src/tensaku/data/filter_spec.py
# -*- coding: utf-8 -*-
"""tensaku.data.filter_spec

Canonical specification for selecting *one* dataset slice from a master oracle.

In Tensaku, the master dataset may contain multiple QIDs. Split generation must
*always* apply the same filter, otherwise label-space mismatch will happen
(e.g., label_max coming from a different QID).

This spec is stored in split meta.json and used both by split.py and
experiments/bootstrap.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class FilterSpec:
    """Filter spec for selecting records."""

    qid: str
    qid_key: str = "qid"

    def to_dict(self) -> Dict[str, Any]:
        return {"qid": str(self.qid), "qid_key": str(self.qid_key)}

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, ctx: str = "filter_spec") -> "FilterSpec":
        if not isinstance(d, Mapping):
            raise ValueError(f"{ctx}: must be a mapping")
        qid = d.get("qid")
        qid_key = d.get("qid_key", "qid")
        if qid is None or str(qid).strip() == "":
            raise ValueError(f"{ctx}: missing qid")
        if qid_key is None or str(qid_key).strip() == "":
            raise ValueError(f"{ctx}: missing qid_key")
        return FilterSpec(qid=str(qid), qid_key=str(qid_key))
