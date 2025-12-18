# /home/esakit25/work/tensaku/src/tensaku/split.py
# -*- coding: utf-8 -*-
"""tensaku.split

@role
  - マスタデータ (all.jsonl) から {labeled, dev, test, pool} を生成する。

@design (Strict)
  - フォールバック/暗黙デフォルト禁止：必須キー欠落は即エラー。
  - split は「データ分割」だけを知る。

@io
  - input : data.input_all (JSONL)
  - output: run.data_dir/{labeled,dev,test,pool}.jsonl + meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_jsonl(path: str, rows: List[dict]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    _atomic_write_text(path, text)


def _coerce_int_label_strict(v: Any, *, label_key: str, qid: str) -> int:
    """Coerce label to int **without rounding**.

    RIKEN SAS の総得点は整数である前提。
    float/str でも整数表現 (e.g., 14.0, "7") は許容するが、
    14.7 のような値は **データ不整合**として即エラーにする。
    """
    if v is None:
        raise ValueError(f"[{qid}] label '{label_key}' is None")
    # bool is subclass of int -> reject explicitly
    if isinstance(v, bool):
        raise ValueError(f"[{qid}] label '{label_key}' must be int, got bool")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        raise ValueError(
            f"[{qid}] label '{label_key}' must be an integer-valued float, got {v}"
        )
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            raise ValueError(f"[{qid}] label '{label_key}' is empty string")
        try:
            fv = float(s)
        except Exception as e:
            raise ValueError(f"[{qid}] label '{label_key}' cannot parse '{v}'") from e
        if fv.is_integer():
            return int(fv)
        raise ValueError(
            f"[{qid}] label '{label_key}' must be integer-like, got '{v}'"
        )
    raise ValueError(f"[{qid}] label '{label_key}' has unsupported type: {type(v)}")


def _ensure_id(rows: List[dict], prefix: str) -> None:
    n = len(rows)
    width = len(str(max(n, 1)))
    cnt = 0
    for r in rows:
        if "id" in r and r["id"] not in (None, ""):
            continue
        cnt += 1
        r["id"] = f"{prefix}{cnt:0{width}d}"


def _validate_and_normalize_ratio(raw: Dict[str, Any]) -> Dict[str, float]:
    """Strict: keys(test,dev,pool,labeled(or train)) 必須 & 正規化"""
    if not raw:
        raise ValueError("split.ratio is empty (Strict: defaults disabled)")

    missing = []
    for k in ("test", "dev", "pool"):
        if k not in raw:
            missing.append(k)
    if ("labeled" not in raw) and ("train" not in raw):
        missing.append("labeled")
    if missing:
        raise ValueError(f"split.ratio is missing required keys: {missing}. (Strict)")

    labeled_val = raw.get("labeled", raw.get("train"))
    ratio = {
        "test": float(raw["test"]),
        "dev": float(raw["dev"]),
        "labeled": float(labeled_val),
        "pool": float(raw["pool"]),
    }
    for k, v in ratio.items():
        if v < 0:
            raise ValueError(f"split.ratio.{k} must be >= 0, got {v}")

    total = sum(ratio.values())
    if total <= 0:
        raise ValueError(f"Sum of split ratios must be positive: {ratio}")

    return {k: (v / total) for k, v in ratio.items()}


def _round_alloc(n_total: int, ratio: Dict[str, float]) -> Dict[str, int]:
    if n_total <= 0:
        return {k: 0 for k in ("test", "dev", "labeled", "pool")}
    keys = ["test", "dev", "labeled", "pool"]
    desired = {k: n_total * ratio[k] for k in keys}
    base = {k: int(desired[k]) for k in keys}
    used = sum(base.values())
    remain = max(0, n_total - used)
    frac = {k: desired[k] - base[k] for k in keys}
    for k in sorted(keys, key=lambda k: frac[k], reverse=True):
        if remain <= 0:
            break
        base[k] += 1
        remain -= 1
    return base


def _split_indices_stratified(labels: List[int], ratio: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_label[int(y)].append(idx)

    splits = {"test": [], "dev": [], "labeled": [], "pool": []}
    for _, idxs in by_label.items():
        if not idxs:
            continue
        rng.shuffle(idxs)
        alloc = _round_alloc(len(idxs), ratio)
        i = 0
        for split_name in ("test", "dev", "labeled", "pool"):
            k = alloc[split_name]
            splits[split_name].extend(idxs[i : i + k])
            i += k
    return splits


def _split_indices_simple(n_total: int, ratio: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    alloc = _round_alloc(n_total, ratio)
    splits: Dict[str, List[int]] = {}
    cur = 0
    for split_name in ("test", "dev", "labeled", "pool"):
        k = alloc[split_name]
        splits[split_name] = idxs[cur : cur + k]
        cur += k
    return splits


def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    ns, _ = parser.parse_known_args(argv or [])

    if "run" not in cfg or "data" not in cfg or "split" not in cfg:
        raise ValueError("cfg must contain 'run', 'data', 'split' sections (Strict)")

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]

    data_dir = str(run_cfg["data_dir"])
    qid = str(data_cfg["qid"])
    label_key = str(data_cfg["label_key"])
    all_path = str(data_cfg["input_all"])

    seed = int(split_cfg["seed"])
    stratify = bool(split_cfg["stratify"])

    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(all_path):
        LOGGER.error("all.jsonl not found: %s", all_path)
        return 2

    ratio_raw = split_cfg.get("ratio")
    n_train_cfg = split_cfg.get("n_train")
    if (not ratio_raw) and (n_train_cfg is None):
        raise ValueError("Either split.ratio or split.n_train must be set. (Strict)")

    all_rows = _read_jsonl(all_path)
    rows = [r for r in all_rows if str(r.get("qid")) == qid]
    if not rows:
        LOGGER.error("no rows found for qid=%r in %s", qid, all_path)
        return 2

    _ensure_id(rows, prefix=f"{qid}_")

    labels: List[int] = []
    for r in rows:
        if label_key not in r:
            raise KeyError(f"label_key '{label_key}' is missing in a record (qid={qid})")
        labels.append(_coerce_int_label_strict(r[label_key], label_key=label_key, qid=qid))

    # label stats (Strict): classification labels must be contiguous.
    label_min = min(labels)
    label_max = max(labels)
    unique_labels = sorted(set(labels))
    expected = list(range(label_min, label_max + 1))
    if unique_labels != expected:
        raise ValueError(
            f"[{qid}] labels must be contiguous ints. "
            f"got unique_labels={unique_labels} (min={label_min}, max={label_max})"
        )
    if label_min != 0:
        raise ValueError(
            f"[{qid}] labels must start at 0 (Strict). got min={label_min}, max={label_max}"
        )
    num_labels = label_max + 1

    n_total = len(rows)

    mode = "ratio"
    ratio: Dict[str, float]
    ratio_sig: Dict[str, float]
    n_train_sig: Optional[int] = None

    if n_train_cfg is not None:
        mode = "n_train"
        n_train = int(n_train_cfg)
        if not ratio_raw or ("test" not in ratio_raw) or ("dev" not in ratio_raw):
            raise ValueError("split.n_train requires split.ratio.test and split.ratio.dev. (Strict)")

        base_test = float(ratio_raw["test"])
        base_dev = float(ratio_raw["dev"])

        approx_test = int(round(n_total * base_test))
        approx_dev = int(round(n_total * base_dev))
        if n_train + approx_test + approx_dev > n_total:
            raise ValueError(
                f"Impossible split: n_train={n_train} + approx(test/dev)={approx_test}/{approx_dev} > total={n_total}"
            )

        l_ratio = n_train / n_total if n_total else 0.0
        p_ratio = max(0.0, 1.0 - base_test - base_dev - l_ratio)
        ratio = {"labeled": l_ratio, "pool": p_ratio, "test": base_test, "dev": base_dev}

        ratio_sig = {"test": base_test, "dev": base_dev}
        n_train_sig = n_train
    else:
        ratio = _validate_and_normalize_ratio(ratio_raw)
        ratio_sig = dict(ratio)

    if stratify:
        idx_splits = _split_indices_stratified(labels, ratio, seed)
    else:
        idx_splits = _split_indices_simple(n_total, ratio, seed)

    labeled = [rows[i] for i in idx_splits["labeled"]]
    dev = [rows[i] for i in idx_splits["dev"]]
    test = [rows[i] for i in idx_splits["test"]]
    pool_rows = [rows[i] for i in idx_splits["pool"]]

    pool: List[dict] = []
    for r in pool_rows:
        rr = dict(r)
        rr.pop(label_key, None)
        pool.append(rr)

    if ns.dry_run:
        return 0

    _write_jsonl(os.path.join(data_dir, "labeled.jsonl"), labeled)
    _write_jsonl(os.path.join(data_dir, "dev.jsonl"), dev)
    _write_jsonl(os.path.join(data_dir, "test.jsonl"), test)
    _write_jsonl(os.path.join(data_dir, "pool.jsonl"), pool)

    meta = {
        "qid": qid,
        "data_dir": os.path.abspath(data_dir),
        "input_all": os.path.abspath(all_path),
        "label_key": label_key,
        "label_stats": {
            "label_min": label_min,
            "label_max": label_max,
            "num_labels": num_labels,
            "unique_count": len(unique_labels),
            "unique_labels": unique_labels,
        },
        "split": {
            "seed": seed,
            "stratify": stratify,
            "mode": mode,
            "n_train": n_train_sig,
            "ratio": ratio_sig,  # n_train modeでは test/dev のみ
        },
        "counts": {"labeled": len(labeled), "dev": len(dev), "test": len(test), "pool": len(pool)},
    }
    _atomic_write_text(os.path.join(data_dir, "meta.json"), json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return 0
