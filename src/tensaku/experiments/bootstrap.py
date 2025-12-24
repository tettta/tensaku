# /home/esakit25/work/tensaku/src/tensaku/experiments/bootstrap.py
# -*- coding: utf-8 -*-
"""Split bootstrap (Strict).

Responsibility: ensure split artifacts exist and match the current config.
- Canonical location is `run.data_dir`.
- If `data.base_dir` is set, it must equal `run.data_dir`.
- In Strict mode, `run.on_exist=skip` is allowed only when the signature matches.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tensaku.split import run as split_run
from tensaku.utils.strict_cfg import ConfigError, require_bool, require_int, require_mapping, require_str


def _atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    """Atomic JSON write (best-effort) for meta repair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass
    tmp.replace(path)


def _coerce_int_label_strict(v: Any, *, label_key: str) -> int:
    if v is None:
        raise ValueError(f"label '{label_key}' is None")
    if isinstance(v, bool):
        raise ValueError(f"label '{label_key}' must be int, got bool")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        raise ValueError(f"label '{label_key}' must be an integer-valued float, got {v}")
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError(f"label '{label_key}' is empty string")
        fv = float(s)
        if fv.is_integer():
            return int(fv)
        raise ValueError(f"label '{label_key}' must be integer-like, got '{v}'")
    raise ValueError(f"label '{label_key}' has unsupported type: {type(v)}")


def _compute_label_stats_from_jsonl(paths: Sequence[Path], *, label_key: str) -> Dict[str, Any]:
    labels: list[int] = []
    for p in paths:
        if not p.exists():
            raise RuntimeError(f"label stats: missing required split file: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if label_key not in obj:
                    raise RuntimeError(f"label stats: missing label_key '{label_key}' in {p.name}")
                labels.append(_coerce_int_label_strict(obj[label_key], label_key=label_key))
    if not labels:
        raise RuntimeError("label stats: no labeled data found in split files")

    label_min = min(labels)
    label_max = max(labels)
    unique_labels = sorted(set(labels))
    expected = list(range(label_min, label_max + 1))
    if unique_labels != expected:
        raise RuntimeError(
            f"labels must be contiguous ints. got unique_labels={unique_labels} (min={label_min}, max={label_max})"
        )
    if label_min != 0:
        raise RuntimeError(f"labels must start at 0 (Strict). got min={label_min}, max={label_max}")
    num_labels = label_max + 1

    return {
        "label_min": label_min,
        "label_max": label_max,
        "num_labels": num_labels,
        "unique_count": len(unique_labels),
        "unique_labels": unique_labels,
    }
def _missing_split_files(data_dir: Path) -> Sequence[Path]:
    required = [
        data_dir / "labeled.jsonl",
        data_dir / "dev.jsonl",
        data_dir / "test.jsonl",
        data_dir / "pool.jsonl",
        data_dir / "meta.json",
    ]
    return [p for p in required if not p.exists()]


def _pool_contains_label(pool_path: Path, label_key: str, max_lines: int = 200) -> bool:
    """Quick corruption check: pool must not contain label_key."""
    if not pool_path.exists():
        return False
    try:
        with pool_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if label_key in obj:
                    return True
        return False
    except Exception:
        return True


def _collect_expected_signature(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    run_cfg = require_mapping(cfg, ("run",), ctx="cfg")
    data_cfg = require_mapping(cfg, ("data",), ctx="cfg")
    split_cfg = require_mapping(cfg, ("split",), ctx="cfg")

    data_dir = require_str(run_cfg, ("data_dir",), ctx="cfg.run")

    base_dir = data_cfg.get("base_dir")
    if base_dir is not None and str(base_dir) != str(data_dir):
        raise ConfigError(
            f"data.base_dir ({base_dir}) != run.data_dir ({data_dir}). "
            "(Strict: keep a single canonical split location)"
        )

    qid = require_str(data_cfg, ("qid",), ctx="cfg.data")
    input_all = require_str(data_cfg, ("input_all",), ctx="cfg.data")
    label_key = require_str(data_cfg, ("label_key",), ctx="cfg.data")

    seed = require_int(split_cfg, ("seed",), ctx="cfg.split")
    stratify = require_bool(split_cfg, ("stratify",), ctx="cfg.split")

    ratio_cfg = require_mapping(split_cfg, ("ratio",), ctx="cfg.split")
    n_train = split_cfg.get("n_train")

    if n_train is None:
        for k in ("test", "dev", "pool"):
            if k not in ratio_cfg:
                raise ConfigError(f"split.ratio.{k} is required in ratio mode (Strict)")
        if ("labeled" not in ratio_cfg) and ("train" not in ratio_cfg):
            raise ConfigError("split.ratio.labeled (or train) is required in ratio mode (Strict)")
        labeled_v = ratio_cfg.get("labeled", ratio_cfg.get("train"))
        ratio_sig: Dict[str, Any] = {
            "test": float(ratio_cfg["test"]),
            "dev": float(ratio_cfg["dev"]),
            "labeled": float(labeled_v),
            "pool": float(ratio_cfg["pool"]),
        }
        mode = "ratio"
        n_train_sig: Optional[int] = None
    else:
        if ("test" not in ratio_cfg) or ("dev" not in ratio_cfg):
            raise ConfigError("split.n_train requires split.ratio.test and split.ratio.dev (Strict)")
        ratio_sig = {"test": float(ratio_cfg["test"]), "dev": float(ratio_cfg["dev"])}
        mode = "n_train"
        n_train_sig = int(n_train)

    return {
        "qid": str(qid),
        "data_dir": str(data_dir),
        "input_all": str(input_all),
        "label_key": str(label_key),
        "split": {
            "seed": int(seed),
            "stratify": bool(stratify),
            "mode": mode,
            "n_train": n_train_sig,
            "ratio": ratio_sig,
        },
    }


def _collect_actual_signature(meta: Mapping[str, Any]) -> Dict[str, Any]:
    split_meta = meta.get("split") or {}
    return {
        "qid": str(meta.get("qid")),
        "data_dir": str(meta.get("data_dir")),
        "input_all": str(meta.get("input_all")),
        "label_key": str(meta.get("label_key")),
        "split": {
            "seed": int(split_meta.get("seed")),
            "stratify": bool(split_meta.get("stratify")),
            "mode": str(split_meta.get("mode")),
            "n_train": split_meta.get("n_train"),
            "ratio": split_meta.get("ratio"),
        },
    }


def _diff_signature(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    diffs: Dict[str, Tuple[Any, Any]] = {}

    def walk(prefix: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a.keys()) | set(b.keys())
            for k in sorted(keys):
                walk(f"{prefix}{k}.", a.get(k), b.get(k))
            return
        if a != b:
            diffs[prefix[:-1]] = (a, b)

    walk("", expected, actual)
    return diffs


def ensure_split_for_qid(
    cfg: Mapping[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
) -> Path:
    log = logger or logging.getLogger(__name__)

    run_cfg = require_mapping(cfg, ("run",), ctx="cfg")
    on_exist = str(run_cfg.get("on_exist", "skip"))
    if on_exist not in {"skip", "overwrite", "error"}:
        raise ConfigError(f"run.on_exist must be one of skip|overwrite|error, got {on_exist!r}")

    expected = _collect_expected_signature(cfg)
    data_dir = Path(expected["data_dir"]).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    missing = _missing_split_files(data_dir)
    pool_bad = _pool_contains_label(data_dir / "pool.jsonl", expected["label_key"]) if (data_dir / "pool.jsonl").exists() else False

    # missing / pool汚染は「既存splitが壊れている」ので on_exist に関わらず再生成
    if missing or pool_bad:
        if pool_bad:
            log.warning("[bootstrap] pool.jsonl contains label_key -> invalid split; regenerating")
        if missing:
            log.info("[bootstrap] missing split files: %s", [p.name for p in missing])
        argv = ["--dry-run"] if dry_run else []
        rc = split_run(argv=argv, cfg=dict(cfg))
        if rc != 0:
            raise RuntimeError(f"split failed with rc={rc}")

    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = _collect_actual_signature(meta)

    diffs = _diff_signature(expected, actual)
    if diffs:
        msg = "[bootstrap] split meta.json signature mismatch:\n" + "\n".join(
            [f"  - {k}: expected={v[0]!r} actual={v[1]!r}" for k, v in diffs.items()]
        )
        if on_exist == "overwrite":
            log.warning(msg)
            log.info("[bootstrap] on_exist=overwrite -> resplitting")
            argv = ["--dry-run"] if dry_run else []
            rc = split_run(argv=argv, cfg=dict(cfg))
            if rc != 0:
                raise RuntimeError(f"split failed with rc={rc}")
        elif on_exist == "error":
            raise RuntimeError(msg)
        else:
            # 重要：skip は「一致している時のみreuse」を強制（ここが場所食い違い防止の要）
            raise RuntimeError(msg + "\n(run.on_exist=skip disallows mismatch in Strict mode; set overwrite or fix config)")
        
    else:
        log.info("[bootstrap] signature match -> skipping generation")

    if _pool_contains_label(data_dir / "pool.jsonl", expected["label_key"]):
        raise RuntimeError("pool.jsonl contains label_key (contract violation)")

    # --- label stats (Strict) ---
    computed_stats = _compute_label_stats_from_jsonl(
        [data_dir / "labeled.jsonl", data_dir / "dev.jsonl", data_dir / "test.jsonl"],
        label_key=expected["label_key"],
    )
    meta_stats = meta.get("label_stats")
    if meta_stats is None:
        log.info("[bootstrap] meta.json missing label_stats -> computing & updating")
        meta["label_stats"] = computed_stats
        _atomic_write_json(meta_path, meta)
    elif isinstance(meta_stats, dict):
        # If meta exists but disagrees with actual files, treat as corruption.
        if dict(meta_stats) != computed_stats:
            raise RuntimeError(
                "meta.json label_stats mismatch with split files (possible corruption). "
                "Set run.on_exist=overwrite to regenerate. "
                f"meta={meta_stats} computed={computed_stats}"
            )
    else:
        raise RuntimeError("meta.json label_stats must be a mapping")

    # Optional: validate model.num_labels against computed stats.
    # - If cfg.model.num_labels is None, treat as "not specified" and skip validation.
    #   (Recommended: set null and rely on split meta.json label_stats.num_labels.)
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, Mapping) and ("num_labels" in model_cfg):
        v = model_cfg.get("num_labels")
        if v is None:
            log.info("[bootstrap] cfg.model.num_labels is None -> skip validation (use split label_stats)")
        else:
            cfg_num_labels = require_int(model_cfg, ("num_labels",), ctx="cfg.model")
            if cfg_num_labels != int(computed_stats["num_labels"]):
                raise ConfigError(
                    f"cfg.model.num_labels={cfg_num_labels} does not match split label_stats.num_labels={computed_stats['num_labels']} "
                    "(Strict: fix config or regenerate split)"
                )

    log.info("[bootstrap] split ok: %s", data_dir)
    return data_dir
