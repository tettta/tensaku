#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /home/esakit25/work/tensaku/scripts/backfill_experiment_index.py
"""Backfill per-QID experiment index.

This script is for experiments executed before the pipeline started writing the
index (or for legacy experiments whose directories were moved/renamed later).

Two explicit truth modes:
- --truth dir  : identity (qid/tag/sampler/seed) comes from directory structure
- --truth hydra: identity comes from .hydra/config.yaml (legacy/override style)

Default is --truth dir, which matches the post-migration policy:
  outputs/<qid>/<tag>/<sampler>/seed=<seed>/...

The resulting index record uses the *v3* schema written by
`tensaku.experiments.index.build_experiment_record`, and it additionally stores
flags about mismatches between `.hydra` and path-derived identity.

Usage:
  python scripts/backfill_experiment_index.py --project-root . --qid Y14_1-2_1_3
  python scripts/backfill_experiment_index.py --project-root . --qid Y14_1-2_1_3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from tensaku.experiments.discovery import discover_from_fs
from tensaku.experiments.index import (
    append_experiment_index,
    normalize_record,
    qid_index_path_from,
)


def _read_yaml_mapping(p: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{p}: expected YAML mapping, got {type(obj)}")
    return obj


def _cfg_for_index(project_root: Path, qid: str) -> Dict[str, Any]:
    """Minimal cfg used only to resolve index path.

    We purposely avoid relying on run.outputs_root or run.tag here.
    """

    return {
        "project_root": str(project_root),
        "data": {"qid": qid},
        "run": {"seed": 0},
        "al": {"sampler": {"name": "_"}},
        "gate": {"mode": "_", "eps_cse": 0.0, "cse_abs_err": 0.0, "estimators": []},
        "confidence": {"estimators": []},
        "split": {"stratify": True},
        "train": {"epochs": 0},
        "model": {"name": "_"},
    }


def _load_existing_uids(index_path: Path) -> set:
    if not index_path.exists():
        return set()
    uids = set()
    for ln, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            print(f"[warn] index parse error: {index_path}:{ln}", file=sys.stderr)
            continue
        if isinstance(obj, dict):
            try:
                uids.add(normalize_record(obj).get("uid"))
            except Exception:
                continue
    return {u for u in uids if isinstance(u, str) and u}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--qid", default=None, help="If set, only backfill this qid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-duplicates", action="store_true")
    ap.add_argument(
        "--require-learning-curve",
        action="store_true",
        help="If set, only index experiments that have metrics/al_learning_curve.csv",
    )
    ap.add_argument(
        "--truth",
        choices=["dir", "hydra"],
        default="dir",
        help="Which identity source is treated as truth for run.tag/sampler/seed.",
    )
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()

    # Discover experiments from FS; identity comes from directory structure.
    exps = discover_from_fs(
        project_root=project_root,
        qid=args.qid,
        require_learning_curve=args.require_learning_curve,
    )

    if not exps:
        qid_msg = "(all qids)" if args.qid is None else args.qid
        print(f"[warn] no experiments found under {project_root / 'outputs'} for {qid_msg}")
        return 0

    scanned = 0
    appended = 0

    # De-dup cache per qid
    uid_cache: Dict[str, set] = {}

    for ref in exps:
        scanned += 1

        exp_dir = ref.exp_dir
        hydra_cfg = exp_dir / ".hydra" / "config.yaml"
        if not hydra_cfg.exists():
            print(f"[skip] missing .hydra/config.yaml: {hydra_cfg}", file=sys.stderr)
            continue

        try:
            cfg = _read_yaml_mapping(hydra_cfg)
        except Exception as e:
            print(f"[skip] cannot read hydra config: {hydra_cfg}: {e}", file=sys.stderr)
            continue

        # Decide which identity is truth. Directory identity is always available (ref.*).
        qid_dir, tag_dir, sampler_dir, seed_dir = ref.qid, ref.tag, ref.sampler, ref.seed

        if args.truth == "dir":
            # Ensure cfg has required sections used by build_experiment_record, but do not force run.tag.
            # We set run.tag from directory by injecting it into cfg *only for indexing*.
            cfg_for_index = dict(cfg)
            run = dict(cfg_for_index.get("run") or {})
            run["tag"] = tag_dir
            cfg_for_index["run"] = run
            # Also keep sampler/seed consistent (directory truth)
            try:
                run["seed"] = seed_dir
                cfg_for_index["al"]["sampler"]["name"] = sampler_dir
                cfg_for_index["data"]["qid"] = qid_dir
            except Exception:
                # If cfg doesn't have these paths, we don't silently create them; skip.
                print(f"[skip] cfg missing required sections for indexing: {hydra_cfg}", file=sys.stderr)
                continue

            cfg_use = cfg_for_index
        else:
            # hydra truth: require run.tag exists in cfg
            try:
                _ = cfg["run"]["tag"]
            except Exception:
                print(f"[skip] truth=hydra but missing run.tag in {hydra_cfg}", file=sys.stderr)
                continue
            
            cfg_use = cfg
            
        cfg_use = dict(cfg_use)
        cfg_use["project_root"] = str(project_root)

        run_map = dict(cfg_use.get("run") or {})
        run_map["outputs_root"] = str((project_root / "outputs").resolve())
        cfg_use["run"] = run_map

        # Prepare de-dup
        qid_use = qid_dir if args.truth == "dir" else str(cfg_use.get("data", {}).get("qid", ""))
        if not qid_use:
            print(f"[skip] cannot determine qid for {exp_dir}", file=sys.stderr)
            continue
        index_path = qid_index_path_from(project_root, qid_use)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.allow_duplicates:
            if qid_use not in uid_cache:
                uid_cache[qid_use] = _load_existing_uids(index_path)

        # Append record via library (ensures consistent schema)
        if args.dry_run:
            print(
                f"[dry-run] would index: qid={qid_use} tag={tag_dir} sampler={sampler_dir} seed={seed_dir} exp={exp_dir}"
            )
            continue

        # Add mismatch flags to metrics_summary (safe place) so downstream can filter.
        mismatch = ref.hydra_mismatch
        metrics_summary = {
            "flags": {
                "hydra_mismatch": bool(mismatch),
                "source_truth": args.truth,
            }
        }

        appended_ok = append_experiment_index(
            cfg=cfg_use,
            exp_dir=exp_dir,
            status="backfilled",
            exit_code=0,
            error=None,
            metrics_summary=metrics_summary,
            allow_duplicates=args.allow_duplicates,
        )

        if appended_ok:
            appended += 1
            if not args.allow_duplicates:
                # refresh cache by reading the just-created record uid
                uid_cache[qid_use] = _load_existing_uids(index_path)

    print(f"[done] scanned={scanned} appended={appended} dry_run={args.dry_run} truth={args.truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
