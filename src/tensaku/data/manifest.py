# /home/esakit25/work/tensaku/src/tensaku/data/manifest.py
# -*- coding: utf-8 -*-
"""tensaku.data.manifest

Split manifest (meta.json) definition.

Design goals
---
* Make split generation and bootstrap share the *same* primitives.
* Store **global label space** (from oracle) and not overwrite it with
  observed-labeled subsets in early AL rounds.
* Provide a **stable signature** for skip/overwrite decisions.
* Keep backward compatibility with existing meta.json as much as possible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from tensaku.data.filter_spec import FilterSpec
from tensaku.data.label_space import LabelSpace


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_stat_fingerprint(path: str) -> Dict[str, Any]:
    """Cheap, stable-enough fingerprint for strict skip decisions.

    We use file size + mtime_ns; this avoids re-hashing large files for each
    multirun job while still catching most content updates.
    """
    p = Path(path)
    st = p.stat()
    return {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}


def compute_signature(payload: Mapping[str, Any]) -> str:
    """Compute sha256 signature of a mapping payload."""
    s = _stable_json_dumps(payload)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    """Canonical split manifest stored as meta.json."""

    schema_version: int
    qid: str
    data_dir: str
    input_all: str
    label_key: str
    id_key: str
    filter_spec: FilterSpec
    label_space: LabelSpace
    split: Dict[str, Any]
    counts: Dict[str, Any]
    input_all_fingerprint: Dict[str, Any]
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "qid": str(self.qid),
            "data_dir": str(self.data_dir),
            "input_all": str(self.input_all),
            "label_key": str(self.label_key),
            "id_key": str(self.id_key),
            "filter_spec": self.filter_spec.to_dict(),
            "label_space": self.label_space.to_dict(),
            # Backward compatibility: keep the old key name used by main.py
            "label_stats": self.label_space.to_dict(),
            "split": dict(self.split),
            "counts": dict(self.counts),
            "input_all_fingerprint": dict(self.input_all_fingerprint),
            "signature": str(self.signature),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, meta_path: Optional[str] = None) -> "SplitManifest":
        ctx = meta_path or "meta.json"
        if not isinstance(d, Mapping):
            raise ValueError(f"{ctx}: meta must be a mapping")

        schema_version = int(d.get("schema_version", 1))
        qid = str(d.get("qid"))
        data_dir = str(d.get("data_dir"))
        input_all = str(d.get("input_all"))
        label_key = str(d.get("label_key"))
        id_key = str(d.get("id_key", "id"))

        # filter_spec: if absent, reconstruct from qid and default qid_key
        fs_raw = d.get("filter_spec")
        if isinstance(fs_raw, Mapping):
            fs = FilterSpec.from_dict(fs_raw, ctx=f"{ctx}.filter_spec")
        else:
            fs = FilterSpec(qid=qid)

        # label_space: prefer new key, fallback to old label_stats
        ls_raw = d.get("label_space")
        if not isinstance(ls_raw, Mapping):
            ls_raw = d.get("label_stats")
        if not isinstance(ls_raw, Mapping):
            raise ValueError(f"{ctx}: missing label_space/label_stats")
        ls = LabelSpace.from_dict(ls_raw, ctx=f"{ctx}.label_space")

        split = dict(d.get("split") or {})
        counts = dict(d.get("counts") or {})
        fp = dict(d.get("input_all_fingerprint") or {})

        # signature: if absent, compute from canonical payload (best-effort)
        sig = d.get("signature")
        if not isinstance(sig, str) or not sig:
            payload = build_signature_payload(
                qid=qid,
                data_dir=data_dir,
                input_all=input_all,
                label_key=label_key,
                id_key=id_key,
                filter_spec=fs,
                split=split,
                input_all_fingerprint=fp if fp else file_stat_fingerprint(input_all),
            )
            sig = compute_signature(payload)

        return SplitManifest(
            schema_version=schema_version,
            qid=qid,
            data_dir=data_dir,
            input_all=input_all,
            label_key=label_key,
            id_key=id_key,
            filter_spec=fs,
            label_space=ls,
            split=split,
            counts=counts,
            input_all_fingerprint=fp if fp else file_stat_fingerprint(input_all),
            signature=str(sig),
        )


def build_signature_payload(
    *,
    qid: str,
    data_dir: str,
    input_all: str,
    label_key: str,
    id_key: str,
    filter_spec: FilterSpec,
    split: Mapping[str, Any],
    input_all_fingerprint: Mapping[str, Any],
) -> Dict[str, Any]:
    """The only fields that affect split reuse decisions."""
    return {
        "qid": str(qid),
        "data_dir": str(data_dir),
        "input_all": str(input_all),
        "label_key": str(label_key),
        "id_key": str(id_key),
        "filter_spec": filter_spec.to_dict(),
        "split": dict(split),
        "input_all_fingerprint": dict(input_all_fingerprint),
    }


def expected_manifest_from_cfg(cfg: Mapping[str, Any]) -> SplitManifest:
    """Build an *expected* manifest from cfg (without writing files)."""
    run_cfg = cfg.get("run")
    data_cfg = cfg.get("data")
    split_cfg = cfg.get("split")
    if not isinstance(run_cfg, Mapping) or not isinstance(data_cfg, Mapping) or not isinstance(split_cfg, Mapping):
        raise ValueError("cfg must contain run/data/split mappings (Strict)")

    data_dir = str(run_cfg["data_dir"])
    qid = str(data_cfg["qid"])
    input_all = str(data_cfg["input_all"])
    label_key = str(data_cfg["label_key"])
    id_key = str(data_cfg.get("id_key", "id"))

    fs = FilterSpec(qid=qid)
    fp = file_stat_fingerprint(input_all)

    # Keep split signature identical to what split.py stores
    seed = int(split_cfg["seed"])
    stratify = bool(split_cfg["stratify"])
    ratio_cfg = split_cfg.get("ratio")
    n_train = split_cfg.get("n_train")
    if not isinstance(ratio_cfg, Mapping):
        raise ValueError("split.ratio must be a mapping (Strict)")

    if n_train is None:
        mode = "ratio"
        ratio_sig = {
            "test": float(ratio_cfg["test"]),
            "dev": float(ratio_cfg["dev"]),
            "labeled": float(ratio_cfg.get("labeled", ratio_cfg.get("train"))),
            "pool": float(ratio_cfg["pool"]),
        }
        split_sig: Dict[str, Any] = {
            "seed": seed,
            "stratify": stratify,
            "mode": mode,
            "n_train": None,
            "ratio": ratio_sig,
        }
    else:
        mode = "n_train"
        split_sig = {
            "seed": seed,
            "stratify": stratify,
            "mode": mode,
            "n_train": int(n_train),
            "ratio": {"test": float(ratio_cfg["test"]), "dev": float(ratio_cfg["dev"])},
        }

    payload = build_signature_payload(
        qid=qid,
        data_dir=data_dir,
        input_all=input_all,
        label_key=label_key,
        id_key=id_key,
        filter_spec=fs,
        split=split_sig,
        input_all_fingerprint=fp,
    )
    sig = compute_signature(payload)

    # label_space and counts are unknown here (need to read data), fill placeholders.
    # bootstrap/split should not compare these for skip decisions.
    empty_ls = LabelSpace.from_labels([0], ctx="expected_manifest_placeholder")

    return SplitManifest(
        schema_version=2,
        qid=qid,
        data_dir=data_dir,
        input_all=input_all,
        label_key=label_key,
        id_key=id_key,
        filter_spec=fs,
        label_space=empty_ls,
        split=split_sig,
        counts={},
        input_all_fingerprint=fp,
        signature=sig,
    )


def read_manifest(meta_path: Path) -> SplitManifest:
    d = json.loads(meta_path.read_text(encoding="utf-8"))
    return SplitManifest.from_dict(d, meta_path=str(meta_path))


def write_manifest(meta_path: Path, manifest: SplitManifest) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    text = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass
    tmp.replace(meta_path)
