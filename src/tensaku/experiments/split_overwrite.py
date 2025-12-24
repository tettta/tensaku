# /home/esakit25/work/tensaku/src/tensaku/experiments/split_overwrite.py
# -*- coding: utf-8 -*-
"""
tensaku.experiments.split_overwrite

@role
  - 既存 split（data_dir）を安全に一括 regenerate（overwrite）する。
  - Strict:
      * 実行対象は明示的に指定（--qid / --qid-file）。
      * overwrite 実行は --yes が無い限り行わない。
      * 設定は引数で明示（seed/stratify/ratio or n_train 等）。

@features
  - dry-run: 何を上書きするか、既存 meta.json の label_stats を表示
  - report: overwrite 前後の label_stats を CSV に出力
  - optional index append: 任意の JSONL に実行記録を追記

@notes
  - split の実体生成は tensaku.split.run を呼び出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from tensaku.split import run as split_run


@dataclass(frozen=True)
class BeforeAfter:
    qid: str
    data_dir: str
    had_existing: bool
    before_label_min: Optional[int]
    before_label_max: Optional[int]
    before_num_labels: Optional[int]
    before_unique_count: Optional[int]
    before_is_contiguous: Optional[bool]
    before_missing_labels: Optional[str]

    after_label_min: Optional[int]
    after_label_max: Optional[int]
    after_num_labels: Optional[int]
    after_unique_count: Optional[int]
    after_is_contiguous: Optional[bool]
    after_missing_labels: Optional[str]

    changed: bool


def _read_meta_label_stats(meta_path: Path) -> Dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ls = meta.get("label_stats", {})
    if not isinstance(ls, dict):
        raise ValueError(f"meta.label_stats must be dict: {meta_path}")
    # optional extras (contiguity info)
    return ls


def _extract_optional(ls: Mapping[str, Any], key: str) -> Optional[Any]:
    v = ls.get(key, None)
    return v if v is not None else None


def _resolve_qids(ns) -> List[str]:
    qids: List[str] = []
    if ns.qid:
        qids.extend(ns.qid)
    if ns.qid_file:
        qpath = Path(ns.qid_file)
        if not qpath.exists():
            raise FileNotFoundError(f"qid-file not found: {qpath}")
        qids.extend([ln.strip() for ln in qpath.read_text(encoding="utf-8").splitlines() if ln.strip()])
    qids = [q for q in qids if q]
    if not qids:
        raise ValueError("No QID specified. Use --qid or --qid-file (Strict)")
    # de-dup preserve order
    out: List[str] = []
    seen = set()
    for q in qids:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _build_cfg_for_split(*, qid: str, ns) -> Dict[str, Any]:
    # Strict: required args were validated in argparse
    run_cfg = {
        "qid": qid,
        "data_dir": str((Path(ns.data_root) / f"q-{qid}" / ns.split_subdir).resolve()),
        "on_exist": "overwrite",  # overwrite mode explicitly
    }
    data_cfg = {"input_all": str(Path(ns.input_all).resolve())}

    split_cfg: Dict[str, Any] = {"seed": int(ns.seed), "stratify": bool(ns.stratify)}

    if ns.n_train is not None:
        # n_train mode requires base test/dev ratios
        if ns.base_test is None or ns.base_dev is None:
            raise ValueError("--n-train requires --base-test and --base-dev (Strict)")
        split_cfg["n_train"] = int(ns.n_train)
        split_cfg["ratio"] = {"test": float(ns.base_test), "dev": float(ns.base_dev)}
    else:
        if ns.ratio_json is None:
            raise ValueError("Either --n-train or --ratio-json is required. (Strict)")
        ratio = json.loads(ns.ratio_json)
        if not isinstance(ratio, dict):
            raise ValueError("--ratio-json must be a JSON object")
        split_cfg["ratio"] = ratio

    return {"run": run_cfg, "data": data_cfg, "split": split_cfg}


def _append_index_jsonl(index_jsonl: Path, rec: Dict[str, Any]) -> None:
    index_jsonl.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    with index_jsonl.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_overwrite(ns) -> int:
    input_all = Path(ns.input_all)
    if not input_all.exists():
        raise FileNotFoundError(f"input-all not found: {input_all}")
    data_root = Path(ns.data_root)
    report_csv = Path(ns.report_csv)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    qids = _resolve_qids(ns)

    plan_rows = []
    for qid in qids:
        data_dir = (data_root / f"q-{qid}" / ns.split_subdir).resolve()
        meta_path = data_dir / "meta.json"
        had = meta_path.exists()
        if had:
            ls = _read_meta_label_stats(meta_path)
            before = dict(
                before_label_min=_extract_optional(ls, "label_min"),
                before_label_max=_extract_optional(ls, "label_max"),
                before_num_labels=_extract_optional(ls, "num_labels"),
                before_unique_count=_extract_optional(ls, "unique_count"),
                before_is_contiguous=_extract_optional(ls, "is_contiguous"),
                before_missing_labels=json.dumps(_extract_optional(ls, "missing_labels"), ensure_ascii=False)
                if _extract_optional(ls, "missing_labels") is not None
                else None,
            )
        else:
            before = dict(
                before_label_min=None,
                before_label_max=None,
                before_num_labels=None,
                before_unique_count=None,
                before_is_contiguous=None,
                before_missing_labels=None,
            )
        plan_rows.append((qid, str(data_dir), had, before))

    print("[split_overwrite] plan:")
    for qid, ddir, had, before in plan_rows:
        print(f"  - qid={qid} data_dir={ddir} existing={had} before.num_labels={before['before_num_labels']}")

    if not ns.yes:
        print("[split_overwrite] dry-run only. Add --yes to execute overwrite.")
        # still write an empty-ish report for traceability
        df = pd.DataFrame(
            [
                dict(
                    qid=qid,
                    data_dir=ddir,
                    had_existing=had,
                    before_num_labels=before["before_num_labels"],
                    before_label_max=before["before_label_max"],
                    before_unique_count=before["before_unique_count"],
                    before_is_contiguous=before["before_is_contiguous"],
                    before_missing_labels=before["before_missing_labels"],
                    after_num_labels=None,
                    after_label_max=None,
                    after_unique_count=None,
                    after_is_contiguous=None,
                    after_missing_labels=None,
                    changed=None,
                )
                for (qid, ddir, had, before) in plan_rows
            ]
        )
        df.to_csv(report_csv, index=False, encoding="utf-8")
        print(f"[split_overwrite] wrote report (dry-run): {report_csv}")
        return 0

    # Execute
    out_rows: List[BeforeAfter] = []
    for qid, ddir, had, before in plan_rows:
        cfg = _build_cfg_for_split(qid=qid, ns=ns)
        rc = split_run(argv=None, cfg=cfg)
        if rc != 0:
            print(f"[split_overwrite] ERROR: split failed for qid={qid} rc={rc}", file=sys.stderr)
            return 2

        meta_path = Path(ddir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found after split: {meta_path}")

        ls2 = _read_meta_label_stats(meta_path)
        after = dict(
            after_label_min=_extract_optional(ls2, "label_min"),
            after_label_max=_extract_optional(ls2, "label_max"),
            after_num_labels=_extract_optional(ls2, "num_labels"),
            after_unique_count=_extract_optional(ls2, "unique_count"),
            after_is_contiguous=_extract_optional(ls2, "is_contiguous"),
            after_missing_labels=json.dumps(_extract_optional(ls2, "missing_labels"), ensure_ascii=False)
            if _extract_optional(ls2, "missing_labels") is not None
            else None,
        )

        changed = any(
            [
                before["before_label_min"] != after["after_label_min"],
                before["before_label_max"] != after["after_label_max"],
                before["before_num_labels"] != after["after_num_labels"],
                before["before_unique_count"] != after["after_unique_count"],
                before["before_is_contiguous"] != after["after_is_contiguous"],
                before["before_missing_labels"] != after["after_missing_labels"],
            ]
        )

        out_rows.append(
            BeforeAfter(
                qid=qid,
                data_dir=str(ddir),
                had_existing=bool(had),
                **before,
                **after,
                changed=bool(changed),
            )
        )
        print(f"[split_overwrite] done: qid={qid} changed={changed} after.num_labels={after['after_num_labels']}")

    df = pd.DataFrame([asdict(r) for r in out_rows])
    df.to_csv(report_csv, index=False, encoding="utf-8")
    print(f"[split_overwrite] wrote report: {report_csv}")

    if ns.index_jsonl:
        index_path = Path(ns.index_jsonl)
        rec = {
            "type": "split_overwrite",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_all": str(input_all.resolve()),
            "data_root": str(data_root.resolve()),
            "split_subdir": ns.split_subdir,
            "seed": int(ns.seed),
            "stratify": bool(ns.stratify),
            "mode": "n_train" if ns.n_train is not None else "ratio",
            "report_csv": str(report_csv.resolve()),
            "qids": qids,
        }
        _append_index_jsonl(index_path, rec)
        print(f"[split_overwrite] appended index: {index_path}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tensaku-split-overwrite", add_help=True)
    p.add_argument("--input-all", type=str, required=True, help="Path to master all.jsonl")
    p.add_argument("--data-root", type=str, required=True, help="Root directory of q-{qid}/... splits")
    p.add_argument("--split-subdir", type=str, required=True, help="Split subdir under q-{qid} (e.g., base_split)")
    p.add_argument("--qid", type=str, action="append", default=None, help="Target QID (repeatable)")
    p.add_argument("--qid-file", type=str, default=None, help="Text file with QIDs (one per line)")

    # split args
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stratify", action="store_true", help="Use stratified split")
    p.add_argument("--ratio-json", type=str, default=None, help='Ratio mode: JSON like {"labeled":0.2,"pool":0.5,"test":0.2,"dev":0.1}')
    p.add_argument("--n-train", type=int, default=None, help="n_train mode: labeled size (requires --base-test/--base-dev)")
    p.add_argument("--base-test", type=float, default=None)
    p.add_argument("--base-dev", type=float, default=None)

    # safety
    p.add_argument("--yes", action="store_true", help="Actually perform overwrite (otherwise dry-run)")
    p.add_argument("--report-csv", type=str, required=True, help="Output report CSV (before/after)")
    p.add_argument("--index-jsonl", type=str, default=None, help="Optional JSONL to append execution record")

    ns = p.parse_args(list(argv) if argv is not None else None)
    return run_overwrite(ns)


if __name__ == "__main__":
    raise SystemExit(main())
