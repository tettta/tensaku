# /home/esakit25/work/tensaku/src/tensaku/experiments/index.py
# -*- coding: utf-8 -*-
"""tensaku.experiments.index

Per-QID experiment index (JSONL).

The index allows offline tools (viz, reports) to discover experiments without
hard-coding directory structures.

Canonical index location:
  outputs/<qid>/_index/experiments.jsonl

Schema notes
-----------
This project evolved. As a result, two record shapes exist in the wild:

A) Legacy flat records (produced by early backfill):
   {"qid":..., "tag":..., "sampler":..., "seed":..., "exp_dir_abs":..., "paths":...}

B) Current structured records (produced by pipelines / new backfill):
   {"schema_version": 3, "uid":..., "run": {...}, "meta": {...}, "paths": {...}, ...}

Tools should prefer the structured schema, but must be able to read legacy
records. This module provides `normalize_record()` for that.

Source of truth
--------------
- New runs identify `tag` via the directory structure (outputs/{qid}/{tag}/...).
- Some older runs stored tag in Hydra overrides and may have stale `.hydra/*`
  after directory moves.

We make this explicit:
- `run.tag` is the *effective* tag used by tooling (directory-derived if needed)
- `run.tag_cfg` is the value in `.hydra/config.yaml` if present
- `run.tag_dir` is the value derived from the directory structure
- `meta.hydra_mismatch` flags disagreements (not fatal).

No silent fallbacks:
- `qid`, `sampler`, `seed` MUST be present in `.hydra/config.yaml`.
- `tag` MUST be resolvable either from cfg (`run.tag`) or from directory layout.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from tensaku.experiments.layout import ExperimentLayout, QidLayout

Status = Literal["success", "failed", "backfilled"]

SCHEMA_VERSION = 3


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()


def _cfg_get(cfg: Mapping[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping):
            raise KeyError(f"cfg path {'.'.join(keys)}: '{k}' parent is not a mapping")
        if k not in cur:
            raise KeyError(f"cfg missing required key: {'.'.join(keys)}")
        cur = cur[k]
    return cur


def _cfg_maybe(cfg: Mapping[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping):
            return None
        if k not in cur:
            return None
        cur = cur[k]
    return cur


def _as_int(v: Any, *, ctx: str) -> int:
    if isinstance(v, bool):
        raise TypeError(f"{ctx}: expected int, got bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    raise TypeError(f"{ctx}: expected int, got {type(v).__name__}: {v!r}")


def _as_float(v: Any, *, ctx: str) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception as e:
            raise TypeError(f"{ctx}: expected float-like str, got {v!r}") from e
    raise TypeError(f"{ctx}: expected float, got {type(v).__name__}: {v!r}")



def _as_list_float(v: Any, *, ctx: str) -> List[float]:
    if isinstance(v, (list, tuple)):
        out: List[float] = []
        for i, x in enumerate(v):
            out.append(_as_float(x, ctx=f"{ctx}[{i}]"))
        if len(out) == 0:
            raise ValueError(f"{ctx}: list must be non-empty")
        return out
    raise TypeError(f"{ctx}: expected list[float], got {type(v).__name__}: {v!r}")

def _as_str(v: Any, *, ctx: str) -> str:
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError(f"{ctx}: empty string")
        return s
    raise TypeError(f"{ctx}: expected str, got {type(v).__name__}: {v!r}")


def _as_list_str(v: Any, *, ctx: str) -> List[str]:
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for i, x in enumerate(v):
            out.append(_as_str(x, ctx=f"{ctx}[{i}]") )
        return out
    raise TypeError(f"{ctx}: expected list[str], got {type(v).__name__}: {v!r}")


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


def derive_identity_from_path(project_root: Union[str, Path], exp_dir: Union[str, Path]) -> Tuple[str, str, str, int]:
    """Derive (qid, tag, sampler, seed) from exp_dir path.

    Contract (explicit):
        <project_root>/outputs/<qid>/<tag>/<sampler>/<seed_dir>/...
    where seed_dir is one of: seed=<int>, seed_<int>, seed<int>.

    Raises if exp_dir is not under outputs root or does not match the contract.
    """

    project_root = Path(project_root).resolve()
    outputs_root = (project_root / "outputs").resolve()
    exp_dir_p = Path(exp_dir).resolve()

    try:
        rel = exp_dir_p.relative_to(outputs_root)
    except Exception as e:
        raise ValueError(f"exp_dir is not under outputs root: {exp_dir_p} (outputs_root={outputs_root})") from e

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


def qid_index_path(cfg: Mapping[str, Any]) -> Path:
    """Return outputs/<qid>/_index/experiments.jsonl.

    Strict resolution order for outputs root:
    1) run.outputs_root (explicit)
    2) <project_root>/outputs (project_root is required if outputs_root is absent)
    """

    qid = _as_str(_cfg_get(cfg, ["data", "qid"]), ctx="data.qid")

    outputs_root_val = _cfg_maybe(cfg, ["run", "outputs_root"])
    if outputs_root_val is not None:
        outputs_root = Path(_as_str(outputs_root_val, ctx="run.outputs_root")).resolve()
    else:
        project_root = Path(_as_str(_cfg_get(cfg, ["project_root"]), ctx="project_root")).resolve()
        outputs_root = (project_root / "outputs").resolve()

    return QidLayout(root=outputs_root / qid).experiments_index.path


def qid_index_path_from(project_root: Union[str, Path], qid: str) -> Path:
    """Return <project_root>/outputs/<qid>/_index/experiments.jsonl."""

    project_root = Path(project_root).resolve()
    qid = _as_str(qid, ctx="qid")
    return QidLayout(root=project_root / "outputs" / qid).experiments_index.path


def _extract_confidence_meta(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    c = _cfg_get(cfg, ["confidence"])
    if not isinstance(c, Mapping):
        raise TypeError("cfg.confidence must be a mapping")

    # Two possible shapes have been used in configs.
    # 1) confidence.estimators: list[dict]
    # 2) confidence.estimators: list[str]
    est = _cfg_get(c, ["estimators"])
    if isinstance(est, (list, tuple)) and all(isinstance(x, str) for x in est):
        estimators = _as_list_str(est, ctx="confidence.estimators")
    else:
        # keep raw for later analysis
        estimators = est

    out: Dict[str, Any] = {"estimators": estimators}

    trust = _cfg_maybe(c, ["trust"])
    if isinstance(trust, Mapping):
        out["trust"] = dict(trust)

    mcd = _cfg_maybe(c, ["mc_dropout"])
    if isinstance(mcd, Mapping):
        out["mc_dropout"] = dict(mcd)

    return out


def _extract_gate_meta(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    g = _cfg_get(cfg, ["gate"])
    if not isinstance(g, Mapping):
        raise TypeError("cfg.gate must be a mapping")

    eps_raw = _cfg_get(g, ["eps_cse"])
    eps_cse: Optional[float]
    eps_cse_list: Optional[List[float]]

    if isinstance(eps_raw, (list, tuple)):
        eps_cse_list = _as_list_float(eps_raw, ctx="gate.eps_cse")
        # Strict: do not choose a representative value silently when multiple are given.
        eps_cse = eps_cse_list[0] if len(eps_cse_list) == 1 else None
    else:
        eps_cse = _as_float(eps_raw, ctx="gate.eps_cse")
        eps_cse_list = None

    out: Dict[str, Any] = {
        "mode": _as_str(_cfg_get(g, ["mode"]), ctx="gate.mode"),
        "eps_cse": eps_cse,
        "eps_cse_list": eps_cse_list,
        "cse_abs_err": _as_float(_cfg_get(g, ["cse_abs_err"]), ctx="gate.cse_abs_err"),
        "estimators": _cfg_get(g, ["estimators"]),
    }
    if not isinstance(out["estimators"], (list, tuple)):
        raise TypeError("gate.estimators must be a list")
    return out


def _extract_al_meta(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    a = _cfg_get(cfg, ["al"])
    if not isinstance(a, Mapping):
        raise TypeError("cfg.al must be a mapping")

    sampler = _cfg_get(a, ["sampler"])
    if not isinstance(sampler, Mapping):
        raise TypeError("cfg.al.sampler must be a mapping")

    out: Dict[str, Any] = {
        "rounds": _as_int(_cfg_get(a, ["rounds"]), ctx="al.rounds"),
        "budget": _as_int(_cfg_get(a, ["budget"]), ctx="al.budget"),
        "sampler": {
            "name": _as_str(_cfg_get(sampler, ["name"]), ctx="al.sampler.name"),
        },
    }

    # Optional sampler knobs (if present, keep them)
    for k in ["conf_key", "k_cluster", "mix", "rand_ratio", "seed_mode"]:
        v = _cfg_maybe(sampler, [k])
        if v is not None:
            out["sampler"][k] = v

    return out


def _extract_split_meta(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    s = _cfg_get(cfg, ["split"])
    if not isinstance(s, Mapping):
        raise TypeError("cfg.split must be a mapping")

    out: Dict[str, Any] = {
        "stratify": bool(_cfg_get(s, ["stratify"])),
    }
    n_train = _cfg_maybe(s, ["n_train"])
    ratio = _cfg_maybe(s, ["ratio"])
    if n_train is not None:
        out["n_train"] = _as_int(n_train, ctx="split.n_train")
    if ratio is not None:
        out["ratio"] = ratio
    return out


def _extract_train_meta(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    t = _cfg_get(cfg, ["train"])
    if not isinstance(t, Mapping):
        raise TypeError("cfg.train must be a mapping")

    out: Dict[str, Any] = {
        "epochs": _as_int(_cfg_get(t, ["epochs"]), ctx="train.epochs"),
    }
    for k in ["max_len", "batch_size", "lr", "weight_decay", "lr_full", "lr_frozen"]:
        v = _cfg_maybe(t, [k])
        if v is not None:
            out[k] = v
    return out


def _extract_model_meta(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    m = _cfg_get(cfg, ["model"])
    if not isinstance(m, Mapping):
        raise TypeError("cfg.model must be a mapping")
    out: Dict[str, Any] = {
        "name": _as_str(_cfg_get(m, ["name"]), ctx="model.name"),
    }
    for k in ["task", "head", "speed_mode", "num_labels", "speed", "dropout"]:
        v = _cfg_maybe(m, [k])
        if v is not None:
            out[k] = v
    return out


def _effective_tag(
    *,
    cfg: Mapping[str, Any],
    exp_dir: Path,
    project_root: Path,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve effective tag.

    Returns (tag_effective, tag_cfg, tag_dir).
    """

    tag_cfg: Optional[str] = None
    try:
        tag_cfg = _as_str(_cfg_get(cfg, ["run", "tag"]), ctx="run.tag")
    except Exception:
        tag_cfg = None

    tag_dir: Optional[str] = None
    try:
        _, tag_dir, _, _ = derive_identity_from_path(project_root, exp_dir)
    except Exception:
        tag_dir = None

    if tag_cfg is not None:
        return tag_cfg, tag_cfg, tag_dir
    if tag_dir is not None:
        return tag_dir, tag_cfg, tag_dir

    raise KeyError(
        "tag is required but missing from cfg (run.tag) AND exp_dir does not match outputs/<qid>/<tag>/..."
    )


def build_experiment_record(
    *,
    cfg: Mapping[str, Any],
    exp_dir: Union[str, Path],
    status: Status,
    exit_code: int = 0,
    error: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    metrics_summary: Optional[Mapping[str, Any]] = None,
    allow_missing_layout_files: bool = True,
) -> Dict[str, Any]:
    """Build a single experiment record.

    Required (cfg):
      - project_root OR run.outputs_root
      - data.qid
      - al.sampler.name
      - run.seed
      - run.tag OR a directory-derivable tag via outputs/{qid}/{tag}/...

    Required (filesystem):
      - <exp_dir>/.hydra/config.yaml
      - <exp_dir>/.hydra/hydra.yaml

    Notes
    -----
    - Paths are stored *relative* to exp_dir and are defined by ExperimentLayout.
    - `run.tag` is the *effective* tag used by tooling.
    - We also store `run.*_cfg` and `run.*_dir` when available.
    """

    exp_dir_p = Path(exp_dir).resolve()
    layout = ExperimentLayout(root=exp_dir_p)

    outputs_root_val = _cfg_maybe(cfg, ["run", "outputs_root"])
    if outputs_root_val is not None:
        outputs_root = Path(_as_str(outputs_root_val, ctx="run.outputs_root")).resolve()
        project_root = outputs_root.parent.resolve()  # best effort; not used for anything strict
    else:
        project_root = Path(_as_str(_cfg_get(cfg, ["project_root"]), ctx="project_root")).resolve()
        outputs_root = (project_root / "outputs").resolve()

    qid_cfg = _as_str(_cfg_get(cfg, ["data", "qid"]), ctx="data.qid")
    sampler_cfg = _as_str(_cfg_get(cfg, ["al", "sampler", "name"]), ctx="al.sampler.name")
    seed_cfg = _as_int(_cfg_get(cfg, ["run", "seed"]), ctx="run.seed")

    tag_eff, tag_cfg, tag_dir = _effective_tag(cfg=cfg, exp_dir=exp_dir_p, project_root=project_root)

    # Directory-derived identity (if applicable)
    qid_dir: Optional[str] = None
    sampler_dir: Optional[str] = None
    seed_dir: Optional[int] = None
    try:
        qid_dir, _, sampler_dir, seed_dir = derive_identity_from_path(project_root, exp_dir_p)
    except Exception:
        qid_dir, sampler_dir, seed_dir = None, None, None

    hydra_mismatch = False
    mismatch_detail: Dict[str, Any] = {}

    if qid_dir is not None and qid_dir != qid_cfg:
        hydra_mismatch = True
        mismatch_detail["qid"] = {"cfg": qid_cfg, "dir": qid_dir}
    if sampler_dir is not None and sampler_dir != sampler_cfg:
        hydra_mismatch = True
        mismatch_detail["sampler"] = {"cfg": sampler_cfg, "dir": sampler_dir}
    if seed_dir is not None and seed_dir != seed_cfg:
        hydra_mismatch = True
        mismatch_detail["seed"] = {"cfg": seed_cfg, "dir": seed_dir}
    if tag_cfg is not None and tag_dir is not None and tag_cfg != tag_dir:
        hydra_mismatch = True
        mismatch_detail["tag"] = {"cfg": tag_cfg, "dir": tag_dir}

    uid = _sha1(str(exp_dir_p))

    def _rel(p: Path) -> str:
        return str(p.relative_to(exp_dir_p))

    hydra_dir = exp_dir_p / ".hydra"
    hydra_config = hydra_dir / "config.yaml"
    hydra_hydra = hydra_dir / "hydra.yaml"
    hydra_overrides = hydra_dir / "overrides.yaml"

    if not hydra_config.exists():
        raise FileNotFoundError(f"missing .hydra/config.yaml: {hydra_config}")
    if not hydra_hydra.exists():
        raise FileNotFoundError(f"missing .hydra/hydra.yaml: {hydra_hydra}")

    # layout-defined relative paths
    def _maybe_rel(p: Path, *, name: str) -> Optional[str]:
        if p.exists() or allow_missing_layout_files:
            return _rel(p)
        raise FileNotFoundError(f"missing required artifact '{name}': {p}")

    paths: Dict[str, str] = {
        "run_meta": _rel(layout.run_meta.path),
        "hydra_config": _rel(hydra_config),
        "hydra_hydra": _rel(hydra_hydra),
        "pipeline_log": _rel(layout.pipeline_log.path),
        "al_history": _rel(layout.al_history.path),
        "al_learning_curve": _rel(layout.al_learning_curve.path),
        "hitl_summary_final": _rel(layout.hitl_summary_final.path),
        "hitl_summary_rounds": _rel(layout.hitl_summary_rounds.path),
        "preds_detail_final": _rel(layout.preds_detail_final.path),
        "gate_assign_final": _rel(layout.gate_assign_final.path),
        "selection_all_samples": _rel(layout.selection_all_samples.path),
    }
    if hydra_overrides.exists():
        paths["hydra_overrides"] = _rel(hydra_overrides)

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "uid": uid,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "extra": ({} if extra is None else dict(extra)),
        "status": status,
        "exit_code": int(exit_code),
        "error": (None if error is None else str(error)),
        "run": {
            "qid": qid_cfg,
            "tag": tag_eff,
            "sampler": sampler_cfg,
            "seed": seed_cfg,
            "outputs_root": str(outputs_root),
            "exp_dir": str(exp_dir_p),
            "qid_cfg": qid_cfg,
            "sampler_cfg": sampler_cfg,
            "seed_cfg": seed_cfg,
            "tag_cfg": tag_cfg,
            "qid_dir": qid_dir,
            "sampler_dir": sampler_dir,
            "seed_dir": seed_dir,
            "tag_dir": tag_dir,
        },
        "meta": {
            "hydra_mismatch": hydra_mismatch,
            "hydra_mismatch_detail": mismatch_detail,
            "al": _extract_al_meta(cfg),
            "split": _extract_split_meta(cfg),
            "gate": _extract_gate_meta(cfg),
            "confidence": _extract_confidence_meta(cfg),
            "train": _extract_train_meta(cfg),
            "model": _extract_model_meta(cfg),
        },
        "paths": paths,
    }

    if metrics_summary is not None:
        record["metrics_summary"] = dict(metrics_summary)

    return record


def load_index(index_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load JSONL index records."""

    p = Path(index_path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception as e:
            raise ValueError(f"Index JSON parse error at {p}:{ln}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"Index record must be a JSON object at {p}:{ln}")
        records.append(obj)
    return records


def normalize_record(r: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize both legacy and current record shapes.

    Returns a dict that always has:
      - uid (str)
      - run: {qid, tag, sampler, seed, exp_dir}
      - paths: dict
      - meta.hydra_mismatch (bool, optional)

    This is for tooling; pipelines should only write schema_version>=3.
    """

    if "run" in r and isinstance(r.get("run"), Mapping):
        out = dict(r)
        out.setdefault("schema_version", r.get("schema_version", 0))
        out.setdefault("uid", r.get("uid") or _sha1(str(out["run"].get("exp_dir", ""))))
        meta = out.get("meta")
        if not isinstance(meta, Mapping):
            out["meta"] = {}
        else:
            out["meta"] = dict(meta)
        out["meta"].setdefault("hydra_mismatch", False)
        out.setdefault("paths", {})
        return out

    # legacy flat
    qid = r.get("qid")
    tag = r.get("tag")
    sampler = r.get("sampler")
    seed = r.get("seed")
    exp_dir = r.get("exp_dir_abs") or r.get("exp_dir")

    out: Dict[str, Any] = {
        "schema_version": 1,
        "uid": r.get("uid") or _sha1(str(exp_dir)),
        "status": r.get("status", "unknown"),
        "run": {
            "qid": qid,
            "tag": tag,
            "sampler": sampler,
            "seed": seed,
            "exp_dir": exp_dir,
        },
        "paths": dict(r.get("paths") or {}),
        "meta": {"hydra_mismatch": False},
    }
    return out


def append_experiment_index(
    *,
    cfg: Mapping[str, Any],
    exp_dir: Union[str, Path],
    status: Status = "success",
    exit_code: int = 0,
    error: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    metrics_summary: Optional[Mapping[str, Any]] = None,
    allow_duplicates: bool = False,
) -> bool:
    """Append a record to per-QID index.

    Returns
    - True: appended
    - False: skipped (duplicate uid) if allow_duplicates=False
    """

    index_path = qid_index_path(cfg)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    rec = build_experiment_record(
        cfg=cfg,
        exp_dir=exp_dir,
        status=status,
        exit_code=exit_code,
        error=error,
        started_at=started_at,
        ended_at=ended_at,
        extra=extra,
        metrics_summary=metrics_summary,
    )

    if not allow_duplicates:
        existing = load_index(index_path)
        uids = {normalize_record(r).get("uid") for r in existing}
        if rec["uid"] in uids:
            return False

    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def filter_records(
    records: Sequence[Mapping[str, Any]],
    *,
    status: Optional[Sequence[str]] = ("success",),
    tag: Optional[Sequence[str]] = None,
    sampler: Optional[Sequence[str]] = None,
    seed: Optional[Sequence[int]] = None,
    include_hydra_mismatch: bool = True,
) -> List[Dict[str, Any]]:
    """Filter records by common fields.

    Note: This is a pure filter; it does not validate record schema.
    """

    st = None if status is None else set(status)
    tg = None if tag is None else set(tag)
    sp = None if sampler is None else set(sampler)
    sd = None if seed is None else set(int(x) for x in seed)

    out: List[Dict[str, Any]] = []
    for r0 in records:
        r = normalize_record(r0)
        run = r.get("run")
        if not isinstance(run, Mapping):
            continue
        if st is not None and r.get("status") not in st:
            continue
        if tg is not None and run.get("tag") not in tg:
            continue
        if sp is not None and run.get("sampler") not in sp:
            continue
        if sd is not None:
            try:
                s = int(run.get("seed"))
            except Exception:
                continue
            if s not in sd:
                continue
        if not include_hydra_mismatch:
            meta = r.get("meta")
            if isinstance(meta, Mapping) and bool(meta.get("hydra_mismatch")):
                continue
        out.append(r)
    return out


def utc_now_iso() -> str:
    """UTC now in ISO8601 (timezone-aware)."""

    return _utc_now_iso()
