# /home/esakit25/work/tensaku/src/tensaku/experiments/preflight.py
# -*- coding: utf-8 -*-
"""
tensaku.experiments.preflight

@role
  - 入力マスタ (JSONL) を QID ごとに走査し、実験前に「回らない原因」を可視化する。
  - Strict: 入力パスや出力パス等の必須引数は必ず指定。黙って推定しない。

@output
  - out_csv: QID ごとの統計（件数、ラベル統計、欠損/不正カウント等）
  - (optional) summary_json: 全体要約

@notes
  - 本コマンドは「修復」はしない（Strict）。あくまで診断。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class QidStats:
    qid: str
    n_total: int
    n_bad_json: int
    n_missing_id: int
    n_missing_text: int
    n_missing_score: int
    n_invalid_score: int

    label_min: Optional[int]
    label_max: Optional[int]
    num_labels: Optional[int]
    unique_count: int
    is_contiguous: Optional[bool]
    missing_labels: str  # JSON list string for CSV


def _iter_jsonl_lines(path: Path) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield ln, s


def _parse_json_obj(line_no: int, s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"bad json at line {line_no}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"json must be object at line {line_no}, got {type(obj).__name__}")
    return obj


def _coerce_score(v: Any) -> int:
    # Strict診断：変換できなければ例外（呼び出し側でカウント）
    if isinstance(v, bool):
        raise ValueError("score must not be bool")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError("score float must be integer-valued")
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            raise ValueError("score string empty")
        # "7.0" なども拒否（Strict）
        if "." in s:
            raise ValueError("score string contains '.'")
        return int(s)
    raise ValueError(f"unsupported score type: {type(v).__name__}")


def _finalize_label_stats(labels: List[int]) -> Tuple[Optional[int], Optional[int], Optional[int], int, Optional[bool], List[int]]:
    if not labels:
        return None, None, None, 0, None, []
    u = sorted(set(labels))
    mn = u[0]
    mx = u[-1]
    num = mx + 1  # 0..max を想定（欠番は missing_labels に出す）
    missing = [k for k in range(mn, mx + 1) if k not in set(u)]
    is_cont = len(missing) == 0
    return mn, mx, num, len(u), is_cont, missing


def compute_preflight(
    *,
    input_all: Path,
    out_csv: Path,
    qids: Optional[List[str]],
    id_key: str,
    text_key: str,
    score_key: str,
    fail_on_issues: bool,
) -> int:
    if not input_all.exists():
        raise FileNotFoundError(f"input_all not found: {input_all}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # per-qid accumulators
    acc: Dict[str, Dict[str, Any]] = {}

    def ensure(qid: str) -> Dict[str, Any]:
        if qid not in acc:
            acc[qid] = dict(
                n_total=0,
                n_bad_json=0,
                n_missing_id=0,
                n_missing_text=0,
                n_missing_score=0,
                n_invalid_score=0,
                labels=[],
            )
        return acc[qid]

    qid_filter = set(qids) if qids else None

    for ln, s in _iter_jsonl_lines(input_all):
        try:
            obj = _parse_json_obj(ln, s)
        except Exception:
            # qid が取れないので全体に加算する仕組みは持たない（Strict）
            # 代わりに "_GLOBAL" に集約する
            st = ensure("_GLOBAL")
            st["n_bad_json"] += 1
            continue

        qid = obj.get("qid")
        if not isinstance(qid, str) or not qid.strip():
            qid = "_MISSING_QID"

        if qid_filter is not None and qid not in qid_filter:
            continue

        st = ensure(qid)
        st["n_total"] += 1

        if obj.get(id_key, None) is None:
            st["n_missing_id"] += 1
        if obj.get(text_key, None) is None:
            st["n_missing_text"] += 1
        if obj.get(score_key, None) is None:
            st["n_missing_score"] += 1
        else:
            try:
                y = _coerce_score(obj[score_key])
                st["labels"].append(y)
            except Exception:
                st["n_invalid_score"] += 1

    rows: List[QidStats] = []
    for qid, st in sorted(acc.items(), key=lambda kv: kv[0]):
        labels = list(st["labels"])
        mn, mx, num, ucnt, is_cont, missing = _finalize_label_stats(labels)
        rows.append(
            QidStats(
                qid=qid,
                n_total=int(st["n_total"]),
                n_bad_json=int(st["n_bad_json"]),
                n_missing_id=int(st["n_missing_id"]),
                n_missing_text=int(st["n_missing_text"]),
                n_missing_score=int(st["n_missing_score"]),
                n_invalid_score=int(st["n_invalid_score"]),
                label_min=mn,
                label_max=mx,
                num_labels=num,
                unique_count=int(ucnt),
                is_contiguous=is_cont,
                missing_labels=json.dumps(missing, ensure_ascii=False),
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[preflight] wrote: {out_csv} (rows={len(df)})")

    if fail_on_issues:
        # issue: any qid has missing required keys or invalid score or bad json
        bad = df[
            (df["n_bad_json"] > 0)
            | (df["n_missing_id"] > 0)
            | (df["n_missing_text"] > 0)
            | (df["n_missing_score"] > 0)
            | (df["n_invalid_score"] > 0)
        ]
        if len(bad) > 0:
            print(f"[preflight] FAIL: found issues in {len(bad)} qids. See: {out_csv}", file=sys.stderr)
            return 2

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tensaku-preflight", add_help=True)
    p.add_argument("--input-all", type=str, required=True, help="Path to all.jsonl (master)")
    p.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    p.add_argument("--qid", type=str, action="append", default=None, help="Filter QID (repeatable)")
    p.add_argument("--qid-file", type=str, default=None, help="Text file of QIDs (one per line)")
    p.add_argument("--id-key", type=str, required=True, help="Record ID key (Strict)")
    p.add_argument("--text-key", type=str, required=True, help="Text key (Strict)")
    p.add_argument("--score-key", type=str, required=True, help="Score key (Strict)")
    p.add_argument("--fail-on-issues", action="store_true", help="Exit non-zero if issues are detected")
    ns = p.parse_args(argv)

    qids: Optional[List[str]] = ns.qid
    if ns.qid_file:
        qpath = Path(ns.qid_file)
        if not qpath.exists():
            raise FileNotFoundError(f"qid-file not found: {qpath}")
        q_from_file = [ln.strip() for ln in qpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        qids = (qids or []) + q_from_file

    return compute_preflight(
        input_all=Path(ns.input_all),
        out_csv=Path(ns.out_csv),
        qids=qids,
        id_key=ns.id_key,
        text_key=ns.text_key,
        score_key=ns.score_key,
        fail_on_issues=bool(ns.fail_on_issues),
    )


if __name__ == "__main__":
    raise SystemExit(main())
