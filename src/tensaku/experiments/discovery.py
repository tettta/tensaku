# /home/esakit25/work/tensaku/src/tensaku/experiments/discovery.py
# -*- coding: utf-8 -*-
"""tensaku.experiments.discovery

Experiment discovery with explicit "source of truth".

Your project moved from "override-driven" identity (Hydra overrides) to
"directory-driven" identity where the run is identified by:

  outputs/{qid}/{tag}/{sampler}/seed={seed}/...

Legacy experiments can end up with mismatched `.hydra/*` vs the *current* path
(e.g., you moved directories after the fact). This module makes that explicit by
supporting *two* modes:

- `mode="index"`: use the recorded `_index/experiments.jsonl` as truth.
- `mode="fs"`: treat the *filesystem path* (directory structure) as truth.

No silent fallbacks:
- If required keys are missing from a chosen source, we error.
- In `mode="fs"`, we only accept explicit, documented patterns for `seed=`.

The output is a list of `ExperimentRef` that is stable enough for viz/report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import yaml

from tensaku.experiments.index import load_index, filter_records, qid_index_path_from

DiscoveryMode = Literal["index", "fs"]


@dataclass(frozen=True)
class ExperimentRef:
    """Resolved experiment reference.

    Fields
    ------
    exp_dir:
        Experiment root directory (contains `.hydra/`).
    qid, tag, sampler, seed:
        Run identity. In fs-mode these come from the directory structure.
    source:
        "index" or "fs".
    hydra_mismatch:
        True if `.hydra/config.yaml` disagrees with the identity derived from the
        filesystem path. (Not an error; just a flag.)
    meta:
        Extra information for diagnostics.
    """

    exp_dir: Path
    qid: str
    tag: str
    sampler: str
    seed: int
    source: DiscoveryMode
    hydra_mismatch: bool
    meta: Dict[str, Any]


def _read_yaml_mapping(p: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{p}: expected YAML mapping, got {type(obj)}")
    return obj


def _require(cfg: Mapping[str, Any], keys: Sequence[str], ctx: str) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            raise KeyError(f"missing key: {ctx} -> {'.'.join(keys)} (failed at '{k}')")
        cur = cur[k]
    return cur


def _as_str(x: Any, ctx: str) -> str:
    if isinstance(x, str) and x.strip():
        return x.strip()
    raise TypeError(f"{ctx}: expected non-empty str, got {type(x)}")


def _as_int(x: Any, ctx: str) -> int:
    if isinstance(x, bool):
        raise TypeError(f"{ctx}: expected int, got bool")
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return int(s)
    raise TypeError(f"{ctx}: expected int, got {type(x)} ({x!r})")


def _parse_seed_component(comp: str) -> int:
    """Parse seed component like 'seed=42', 'seed_42', 'seed42'."""
    if comp.startswith("seed="):
        return _as_int(comp.split("=", 1)[1], ctx="seed component")
    if comp.startswith("seed_"):
        return _as_int(comp.split("_", 1)[1], ctx="seed component")
    if comp.startswith("seed") and comp[4:].isdigit():
        return int(comp[4:])
    raise ValueError(
        "unsupported seed directory name. Supported: seed=<int>, seed_<int>, seed<int>. "
        f"got: {comp!r}"
    )


def _resolve_exp_dir_from_hydra_config(hydra_cfg_path: Path) -> Path:
    # .../<exp_dir>/.hydra/config.yaml
    if hydra_cfg_path.name != "config.yaml" or hydra_cfg_path.parent.name != ".hydra":
        raise ValueError(f"not a hydra config.yaml path: {hydra_cfg_path}")
    return hydra_cfg_path.parent.parent


def derive_identity_from_path(project_root: Union[str, Path], exp_dir: Path) -> Tuple[str, str, str, int]:
    """Derive (qid, tag, sampler, seed) from exp_dir path.

    Contract (explicit):
        <project_root>/outputs/<qid>/<tag>/<sampler>/<seed_dir>/...
    where seed_dir is one of: seed=<int>, seed_<int>, seed<int>.

    This is used for fs-mode and also for legacy backfill where `.hydra` may be stale.
    """

    project_root = Path(project_root).resolve()
    outputs_root = (project_root / "outputs").resolve()
    exp_dir = exp_dir.resolve()

    try:
        rel = exp_dir.relative_to(outputs_root)
    except Exception as e:
        raise ValueError(f"exp_dir is not under outputs root: {exp_dir} (outputs_root={outputs_root})") from e

    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(
            "exp_dir must be at least outputs/<qid>/<tag>/<sampler>/<seed_dir>/... ; "
            f"got: outputs/{'/'.join(parts)}"
        )

    qid = parts[0]
    tag = parts[1]
    sampler = parts[2]
    seed = _parse_seed_component(parts[3])
    return qid, tag, sampler, seed


def _hydra_identity(exp_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """Read qid/sampler/tag/seed from `.hydra/config.yaml` when present."""
    cfg_path = exp_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None, None, None, None
    cfg = _read_yaml_mapping(cfg_path)

    qid = None
    sampler = None
    tag = None
    seed = None

    try:
        qid = _as_str(_require(cfg, ["data", "qid"], "cfg"), "data.qid")
    except Exception:
        qid = None

    try:
        sampler = _as_str(_require(cfg, ["al", "sampler", "name"], "cfg"), "al.sampler.name")
    except Exception:
        sampler = None

    # run.tag is optional in your newer configs
    try:
        tag = _as_str(_require(cfg, ["run", "tag"], "cfg"), "run.tag")
    except Exception:
        tag = None

    try:
        seed = _as_int(_require(cfg, ["run", "seed"], "cfg"), "run.seed")
    except Exception:
        seed = None

    return qid, sampler, tag, seed


def discover_from_fs(
    *,
    project_root: Union[str, Path],
    qid: Optional[str] = None,
    require_learning_curve: bool = False,
) -> List[ExperimentRef]:
    """Discover experiments by scanning the filesystem.

    Truth source: directory structure under outputs.

    Experiment root is a directory containing `.hydra/config.yaml`.
    """

    project_root = Path(project_root).resolve()
    outputs_root = project_root / "outputs"
    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs not found: {outputs_root}")

    hydra_cfg_paths = sorted(outputs_root.rglob(".hydra/config.yaml"))
    out: List[ExperimentRef] = []

    for cfg_path in hydra_cfg_paths:
        exp_dir = _resolve_exp_dir_from_hydra_config(cfg_path)

        try:
            qid_d, tag_d, sampler_d, seed_d = derive_identity_from_path(project_root, exp_dir)
        except Exception as e:
            # Not an experiment under canonical outputs layout; skip explicitly.
            continue

        if qid is not None and qid_d != qid:
            continue

        if require_learning_curve and not (exp_dir / "metrics" / "al_learning_curve.csv").exists():
            continue

        qid_h, sampler_h, tag_h, seed_h = _hydra_identity(exp_dir)
        mismatch = False
        if qid_h is not None and qid_h != qid_d:
            mismatch = True
        if sampler_h is not None and sampler_h != sampler_d:
            mismatch = True
        if seed_h is not None and seed_h != seed_d:
            mismatch = True
        # tag_h is optional; if present and differs, it's also a mismatch.
        if tag_h is not None and tag_h != tag_d:
            mismatch = True

        meta = {
            "dir": {"qid": qid_d, "tag": tag_d, "sampler": sampler_d, "seed": seed_d},
            "hydra": {"qid": qid_h, "tag": tag_h, "sampler": sampler_h, "seed": seed_h},
        }

        out.append(
            ExperimentRef(
                exp_dir=exp_dir,
                qid=qid_d,
                tag=tag_d,
                sampler=sampler_d,
                seed=seed_d,
                source="fs",
                hydra_mismatch=mismatch,
                meta=meta,
            )
        )

    return out


def discover_from_index(
    *,
    project_root: Union[str, Path],
    qid: str,
    tag: Optional[Sequence[str]] = None,
    sampler: Optional[Sequence[str]] = None,
    seed: Optional[Sequence[int]] = None,
    status: Optional[Sequence[str]] = ("success", "backfilled"),
    include_mismatch: bool = True,
) -> List[ExperimentRef]:
    """Discover experiments by reading the per-QID index."""

    project_root = Path(project_root).resolve()
    index_path = qid_index_path_from(project_root, qid)
    records = load_index(index_path)
    records = filter_records(records, status=status, tag=tag, sampler=sampler, seed=seed)

    out: List[ExperimentRef] = []
    for r in records:
        run = r.get("run")
        if not isinstance(run, Mapping):
            continue
        exp_dir_s = run.get("exp_dir")
        if not isinstance(exp_dir_s, str) or not exp_dir_s:
            continue
        exp_dir = Path(exp_dir_s).resolve()

        try:
            seed_i = int(run.get("seed"))
        except Exception:
            continue

        mismatch = bool(r.get("meta", {}).get("hydra_mismatch", False)) if isinstance(r.get("meta"), Mapping) else False
        if not include_mismatch and mismatch:
            continue

        out.append(
            ExperimentRef(
                exp_dir=exp_dir,
                qid=str(run.get("qid")),
                tag=str(run.get("tag")),
                sampler=str(run.get("sampler")),
                seed=seed_i,
                source="index",
                hydra_mismatch=mismatch,
                meta={"record": dict(r)},
            )
        )
    return out


def discover_experiments(
    *,
    project_root: Union[str, Path],
    mode: DiscoveryMode,
    qid: Optional[str] = None,
    tag: Optional[Sequence[str]] = None,
    sampler: Optional[Sequence[str]] = None,
    seed: Optional[Sequence[int]] = None,
    require_learning_curve: bool = False,
    status: Optional[Sequence[str]] = ("success", "backfilled"),
    include_mismatch: bool = True,
) -> List[ExperimentRef]:
    """Unified discovery entrypoint."""

    if mode == "fs":
        return discover_from_fs(project_root=project_root, qid=qid, require_learning_curve=require_learning_curve)

    if mode == "index":
        if qid is None:
            raise ValueError("mode='index' requires qid")
        return discover_from_index(
            project_root=project_root,
            qid=qid,
            tag=tag,
            sampler=sampler,
            seed=seed,
            status=status,
            include_mismatch=include_mismatch,
        )

    raise ValueError(f"unknown discovery mode: {mode!r}")
