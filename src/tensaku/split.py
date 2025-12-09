# /home/esakit25/work/tensaku/src/tensaku/split.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.split
@role     : all.jsonl（マスタ）から単一 QID の {labeled, dev, test, pool} を生成する
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# ===================== 基本ユーティリティ =====================


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _ensure_id(rows: List[dict], prefix: str) -> None:
    n = len(rows)
    width = len(str(max(n, 1)))
    cnt = 0
    for r in rows:
        if "id" in r and r["id"] not in (None, ""):
            continue
        cnt += 1
        r["id"] = f"{prefix}{cnt:0{width}d}"


def _infer_all_path(run_data_dir: str, data_cfg: Dict[str, Any]) -> str:
    inp = data_cfg.get("input_all")
    if not inp:
        return os.path.join(run_data_dir, "all.jsonl")
    inp = str(inp)
    if os.path.isdir(inp):
        return os.path.join(inp, "all.jsonl")
    return inp


# ===================== 分割ロジック =====================


def _normalize_ratio(raw: Dict[str, float]) -> Dict[str, float]:
    default_ratio = {"test": 0.2, "dev": 0.1, "labeled": 0.2, "pool": 0.5}
    if not raw:
        return default_ratio

    labeled_val = raw.get("labeled")
    if labeled_val is None:
        labeled_val = raw.get("train")

    r = {
        "test": float(raw.get("test", default_ratio["test"])),
        "dev": float(raw.get("dev", default_ratio["dev"])),
        "labeled": float(labeled_val if labeled_val is not None else default_ratio["labeled"]),
        "pool": float(raw.get("pool", default_ratio["pool"])),
    }
    s = sum(r.values())
    if s <= 0:
        return default_ratio
    return {k: v / s for k, v in r.items()}


def _round_alloc(n_total: int, ratio: Dict[str, float]) -> Dict[str, int]:
    if n_total <= 0:
        return {k: 0 for k in ("test", "dev", "labeled", "pool")}

    keys = ["test", "dev", "labeled", "pool"]
    r = {k: float(ratio.get(k, 0.0)) for k in keys}
    s = sum(r.values()) or 1.0
    desired = {k: n_total * r[k] / s for k in keys}
    base = {k: int(desired[k]) for k in keys}
    used = sum(base.values())
    remain = max(0, n_total - used)

    frac = {k: desired[k] - base[k] for k in keys}
    frac_sorted = sorted(keys, key=lambda k: frac[k], reverse=True)
    for k in frac_sorted:
        if remain <= 0:
            break
        base[k] += 1
        remain -= 1
    return base


def _split_indices_stratified(
    labels: List[int],
    ratio: Dict[str, float],
    seed: int,
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_label[int(y)].append(idx)

    splits = {"test": [], "dev": [], "labeled": [], "pool": []}
    for lab, idxs in by_label.items():
        if not idxs:
            continue
        rng.shuffle(idxs)
        alloc = _round_alloc(len(idxs), ratio)
        i = 0
        for split_name in ("test", "dev", "labeled", "pool"):
            k = alloc[split_name]
            if k <= 0:
                continue
            splits[split_name].extend(idxs[i : i + k])
            i += k
    return splits


def _split_indices_simple(
    n_total: int,
    ratio: Dict[str, float],
    seed: int,
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    alloc = _round_alloc(n_total, ratio)

    splits = {}
    cur = 0
    for split_name in ("test", "dev", "labeled", "pool"):
        k = alloc[split_name]
        splits[split_name] = idxs[cur : cur + k]
        cur += k
    return splits


# ===================== エントリポイント =====================


def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    ns, _rest = parser.parse_known_args(argv or [])

    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    split_cfg = cfg.get("split") or {}

    data_dir = run_cfg.get("data_dir") or data_cfg.get("data_dir")
    if not data_dir:
        LOGGER.error("run.data_dir or data.data_dir must be set.")
        return 2
    data_dir = str(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    qid = data_cfg.get("qid") or run_cfg.get("qid")
    if not qid:
        LOGGER.error("data.qid or run.qid must be set.")
        return 2
    qid = str(qid)

    label_key = data_cfg.get("label_key") or "score"
    seed = int(split_cfg.get("seed") or 42)
    stratify_cfg = split_cfg.get("stratify")
    stratify = True if stratify_cfg is None else bool(stratify_cfg)

    all_path = _infer_all_path(data_dir, data_cfg)
    if not os.path.exists(all_path):
        LOGGER.error(f"all.jsonl not found: {all_path}")
        return 2

    ratio_raw = split_cfg.get("ratio") or {}
    ratio = _normalize_ratio(ratio_raw)

    n_train_cfg = split_cfg.get("n_train")
    n_train: Optional[int] = None
    if n_train_cfg is not None:
        try:
            n_train = int(n_train_cfg)
        except Exception:
            LOGGER.warning(f"invalid split.n_train={n_train_cfg!r} (ignored)")
            n_train = None

    all_rows = _read_jsonl(all_path)
    rows = [r for r in all_rows if str(r.get("qid")) == qid]
    if not rows:
        LOGGER.error(f"no rows found for qid={qid!r} in {all_path}")
        return 2

    _ensure_id(rows, prefix=f"{qid}_")

    labels: List[int] = []
    label_values: List[float] = []
    all_int_like = True

    for r in rows:
        raw = r.get(label_key, 0)
        try:
            v = float(raw)
        except Exception:
            v = 0.0
        label_values.append(v)

        iv = int(round(v))
        if abs(v - iv) > 1e-9:
            all_int_like = False
        labels.append(iv)

    if label_values:
        label_min = min(label_values)
        label_max = max(label_values)
    else:
        label_min = None
        label_max = None
    label_type = "int" if all_int_like else "float"

    num_labels: Optional[int] = None
    is_classification = False
    if (
        label_min is not None
        and label_max is not None
        and all_int_like
        and label_min >= 0
    ):
        try:
            max_int = int(round(label_max))
        except Exception:
            max_int = None
        else:
            if max_int >= 0:
                is_classification = True
                num_labels = max_int + 1

    n_total = len(rows)
    mode = "ratio"
    if n_train is not None and n_total > 0:
        base_test = float(ratio.get("test", 0.0))
        base_dev = float(ratio.get("dev", 0.0))
        sum_td = base_test + base_dev

        if sum_td >= 1.0:
            LOGGER.error(
                f"split.ratio.test + dev = {sum_td:.3f} >= 1.0; no room for n_train."
            )
            return 2

        approx_n_test = int(round(n_total * base_test))
        approx_n_dev = int(round(n_total * base_dev))
        max_n_train = max(0, n_total - approx_n_test - approx_n_dev)

        if n_train < 0:
            LOGGER.warning(f"split.n_train={n_train} < 0; clipped to 0.")
            n_train = 0
        if n_train > max_n_train:
            LOGGER.warning(
                f"split.n_train={n_train} > max_available={max_n_train}; clipped."
            )
            n_train = max_n_train

        labeled_ratio = float(n_train) / float(n_total) if n_total > 0 else 0.0
        pool_ratio = max(0.0, 1.0 - base_test - base_dev - labeled_ratio)

        ratio = {
            "labeled": labeled_ratio,
            "test": base_test,
            "dev": base_dev,
            "pool": pool_ratio,
        }
        mode = "n_train"

    ratio_log = {k: round(float(v), 3) for k, v in ratio.items()}

    LOGGER.info(f"qid={qid} total={n_total} all_path={all_path}")
    LOGGER.info(f"mode={mode} n_train={n_train}")
    LOGGER.info(f"ratio={ratio_log} seed={seed} stratify={stratify}")

    if stratify:
        idx_splits = _split_indices_stratified(labels, ratio, seed=seed)
    else:
        idx_splits = _split_indices_simple(n_total, ratio, seed=seed)

    labeled_rows = [rows[i] for i in idx_splits["labeled"]]
    dev_rows = [rows[i] for i in idx_splits["dev"]]
    test_rows = [rows[i] for i in idx_splits["test"]]
    pool_rows = [rows[i] for i in idx_splits["pool"]]

    counts = {
        "labeled": len(labeled_rows),
        "dev": len(dev_rows),
        "test": len(test_rows),
        "pool": len(pool_rows),
    }
    LOGGER.info(f"counts: {counts}")

    if ns.dry_run:
        LOGGER.info("dry-run: no files written.")
        return 0

    path_labeled = os.path.join(data_dir, "labeled.jsonl")
    path_dev = os.path.join(data_dir, "dev.jsonl")
    path_test = os.path.join(data_dir, "test.jsonl")
    path_pool = os.path.join(data_dir, "pool.jsonl")

    _write_jsonl(path_labeled, labeled_rows)
    _write_jsonl(path_dev, dev_rows)
    _write_jsonl(path_test, test_rows)
    _write_jsonl(path_pool, pool_rows)

    meta = {
        "qid": qid,
        "data_dir": data_dir,
        "all_path": all_path,
        "label_key": label_key,
        "seed": seed,
        "mode": mode,
        "n_train": n_train,
        "ratio": ratio,
        "stratify": stratify,
        "counts": counts,
        "label_min": label_min,
        "label_max": label_max,
        "label_type": label_type,
        "is_classification": is_classification,
        "num_labels": num_labels,
    }

    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    LOGGER.info(f"Wrote split files and meta.json to {data_dir}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.info("Run via CLI: tensaku split -c <CFG.yaml>")
