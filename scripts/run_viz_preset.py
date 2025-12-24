# /home/esakit25/work/tensaku/scripts/run_viz_preset.py
# -*- coding: utf-8 -*-
"""Run the stable viz preset (standard + by-sampler).

Minimal usage:
  python scripts/run_viz_preset.py --qid Y14_1-2_1_3

It reads outputs/<qid>/_index/experiments.jsonl to discover tag/sampler/seed,
then calls `python -m tensaku.viz.cli compare --preset --aux-by-sampler --aggregate-seeds --allow-missing`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Tuple

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"index not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--qid", required=True)
    ap.add_argument("--tag", default=None, help="comma separated (optional)")
    ap.add_argument("--sampler", default=None, help="comma separated (optional)")
    ap.add_argument("--seed", default=None, help="comma separated (optional)")
    ap.add_argument("--n-labeled-min", type=float, default=None, help="optional: restrict compare x=n_labeled to >= this")
    ap.add_argument("--n-labeled-max", type=float, default=None, help="optional: restrict compare x=n_labeled to <= this")
    ap.add_argument("--round-min", type=float, default=None, help="optional: restrict compare x=round to >= this")
    ap.add_argument("--round-max", type=float, default=None, help="optional: restrict compare x=round to <= this")
    ap.add_argument("--strict", action="store_true", help="do not allow missing files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    idx_path = project_root / "outputs" / args.qid / "_index" / "experiments.jsonl"

    # If filters not provided, discover from index.
    tag = args.tag
    sampler = args.sampler
    seed = args.seed

    if tag is None or sampler is None or seed is None:
        recs = _read_jsonl(idx_path)
        tags: Set[str] = set()
        samplers: Set[str] = set()
        seeds: Set[str] = set()
        for rec in recs:
            st = rec.get("status")
            if st not in {"success", "backfilled"}:
                continue
            run = rec.get("run") or {}
            if tag is None and run.get("tag") is not None:
                tags.add(str(run.get("tag")))
            if sampler is None and run.get("sampler") is not None:
                samplers.add(str(run.get("sampler")))
            if seed is None and run.get("seed") is not None:
                seeds.add(str(run.get("seed")))
        if tag is None:
            tag = ",".join(sorted(tags)) if tags else None
        if sampler is None:
            sampler = ",".join(sorted(samplers)) if samplers else None
        if seed is None:
            seed = ",".join(sorted(seeds)) if seeds else None

    cmd = [
        "python",
        "-m",
        "tensaku.viz.cli",
        "compare",
        "--project-root",
        str(project_root),
        "--qid",
        args.qid,
        "--preset",
        "--aux-by-sampler",
        "--aggregate-seeds",
    ]
    if not args.strict:
        cmd.append("--allow-missing")
    if tag:
        cmd += ["--tag", tag]
    if sampler:
        cmd += ["--sampler", sampler]
    if seed:
        cmd += ["--seed", seed]

    # Optional x-range restrictions (passed to compare preset)
    if args.n_labeled_min is not None:
        cmd += ["--x-min-n-labeled", str(args.n_labeled_min)]
    if args.n_labeled_max is not None:
        cmd += ["--x-max-n-labeled", str(args.n_labeled_max)]
    if args.round_min is not None:
        cmd += ["--x-min-round", str(args.round_min)]
    if args.round_max is not None:
        cmd += ["--x-max-round", str(args.round_max)]

    print("[run]", " ".join(cmd))
    if args.dry_run:
        return 0

    p = subprocess.run(cmd)
    return int(p.returncode)

if __name__ == "__main__":
    raise SystemExit(main())
