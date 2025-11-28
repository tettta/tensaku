# /home/esakit25/work/tensaku/src/tensaku/split.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.split
@role     : all.jsonl（マスタ）から単一 QID の {labeled, dev, test, pool} を生成する
@inputs   :
  - YAML:
      split.seed      : 乱数シード（既定 42）
      split.ratio     : 各 split の比率 dict
      split.stratify  : True なら score ベースの層化分割（既定 True）
      split.n_train   : (任意) 初期 labeled の固定件数。指定時は ratio.labeled を無視し、残りを pool へ。
  - CLI:
      --n-train N     : split.n_train を上書きする
@outputs  :
  - {run.data_dir}/{labeled,dev,test,pool}.jsonl
  - {run.data_dir}/meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    if s <= 0.0:
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
    frac_sorted = sorted(keys, key=lambda k: desired[k] - base[k], reverse=True)
    for k in frac_sorted:
        if remain <= 0: break
        base[k] += 1
        remain -= 1
    return base


def _split_indices_stratified(
    labels: List[int],
    ratio: Dict[str, float],
    seed: int,
    n_train_fixed: Optional[int] = None,
) -> Dict[str, List[int]]:
    """score による層化分割（n_train_fixed 対応版）。"""
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_label[int(y)].append(idx)

    splits = {"test": [], "dev": [], "labeled": [], "pool": []}
    total_n = len(labels)

    for lab, idxs in by_label.items():
        if not idxs:
            continue
        rng.shuffle(idxs)
        n_local = len(idxs)

        if n_train_fixed is not None:
            # --- 固定件数モード ---
            # 全体に対するこのラベルの比率に基づいて n_labeled を決定
            # 例: 全体100件中、ラベルAが30件(30%)。n_train=10なら、ラベルAは 3件確保。
            if total_n > 0:
                target_labeled = int(round(n_train_fixed * (n_local / total_n)))
            else:
                target_labeled = 0
            
            # test/dev は比率で確保（評価データは固定したい）
            n_test = int(n_local * ratio.get("test", 0.2))
            n_dev  = int(n_local * ratio.get("dev", 0.1))
            
            # labeled は target_labeled 分だけ確保（ただし在庫を超えない）
            remaining_after_eval = n_local - n_test - n_dev
            n_labeled = min(target_labeled, remaining_after_eval)
            n_labeled = max(0, n_labeled) # 安全策
            
            # 残りすべてを pool へ
            n_pool = n_local - n_test - n_dev - n_labeled
            
            alloc = {"test": n_test, "dev": n_dev, "labeled": n_labeled, "pool": n_pool}
        else:
            # --- 通常（比率）モード ---
            alloc = _round_alloc(n_local, ratio)

        # 配分に従ってインデックスを取り出す
        curr = 0
        for split_name in ("test", "dev", "labeled", "pool"):
            k = alloc[split_name]
            if k > 0:
                splits[split_name].extend(idxs[curr : curr + k])
                curr += k

    return splits


def _split_indices_simple(
    n_total: int,
    ratio: Dict[str, float],
    seed: int,
    n_train_fixed: Optional[int] = None,
) -> Dict[str, List[int]]:
    """全体のみで分割（層化なし、n_train_fixed 対応版）。"""
    rng = random.Random(seed)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    
    splits = {"test": [], "dev": [], "labeled": [], "pool": []}
    
    if n_train_fixed is not None:
        # 固定件数モード
        n_test = int(n_total * ratio.get("test", 0.2))
        n_dev  = int(n_total * ratio.get("dev", 0.1))
        
        remain = n_total - n_test - n_dev
        n_labeled = min(n_train_fixed, remain)
        n_labeled = max(0, n_labeled)
        n_pool = remain - n_labeled
        
        alloc = {"test": n_test, "dev": n_dev, "labeled": n_labeled, "pool": n_pool}
    else:
        # 通常モード
        alloc = _round_alloc(n_total, ratio)

    curr = 0
    for split_name in ("test", "dev", "labeled", "pool"):
        k = alloc[split_name]
        if k > 0:
            splits[split_name] = idxs[curr : curr + k]
            curr += k
            
    return splits


# ===================== メインエントリ =====================


def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    """
    all.jsonl から {labeled, dev, test, pool}.jsonl を生成。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    # 【追加】固定件数指定用
    parser.add_argument("--n-train", type=int, default=None, help="Fixed number of initial labeled data")
    ns, _rest = parser.parse_known_args(argv or [])

    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    split_cfg = cfg.get("split") or {}

    data_dir = run_cfg.get("data_dir") or data_cfg.get("data_dir")
    if not data_dir:
        print("[split] ERROR: run.data_dir or data.data_dir must be set.", file=sys.stderr)
        return 2
    qid = data_cfg.get("qid")
    if not qid:
        print("[split] ERROR: data.qid must be set.", file=sys.stderr)
        return 2

    data_dir = str(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    all_path = _infer_all_path(data_dir, data_cfg)
    if not os.path.exists(all_path):
        print(f"[split] ERROR: all.jsonl not found: {all_path}", file=sys.stderr)
        return 2

    label_key = str(data_cfg.get("label_key", "score"))
    seed = int(split_cfg.get("seed", 42))
    stratify = bool(split_cfg.get("stratify", True))
    ratio_raw = split_cfg.get("ratio") or {}
    ratio = _normalize_ratio(ratio_raw)

    # n_train の取得 (CLI優先 > YAML)
    n_train_fixed = ns.n_train
    if n_train_fixed is None:
        n_train_fixed = split_cfg.get("n_train") # YAML: split.n_train

    # ---- all.jsonl 読み込み & qid でフィルタ ----
    all_rows = _read_jsonl(all_path)
    rows = [r for r in all_rows if str(r.get("qid")) == str(qid)]
    if not rows:
        print(f"[split] ERROR: no rows found for qid={qid!r} in {all_path}", file=sys.stderr)
        return 2

    _ensure_id(rows, prefix=f"{qid}_")

    # ラベル取得
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

    # ラベル統計
    label_min = min(label_values) if label_values else None
    label_max = max(label_values) if label_values else None
    label_type = "int" if all_int_like else "float"

    num_labels: Optional[int] = None
    is_classification = False
    if label_min is not None and label_max is not None and all_int_like and label_min >= 0:
        try:
            max_int = int(round(label_max))
            if max_int >= 0:
                is_classification = True
                num_labels = max_int + 1
        except: pass

    n_total = len(rows)
    print(f"[split] qid={qid}  total={n_total}  all_path={all_path}")
    print(f"[split] ratio={ratio}  seed={seed}  stratify={stratify}")
    if n_train_fixed is not None:
        print(f"[split] MODE: Fixed n_train = {n_train_fixed}")
    else:
        print(f"[split] MODE: Ratio based")

    # 分割実行
    if stratify:
        idx_splits = _split_indices_stratified(
            labels, ratio, seed=seed, n_train_fixed=n_train_fixed
        )
    else:
        idx_splits = _split_indices_simple(
            n_total, ratio, seed=seed, n_train_fixed=n_train_fixed
        )

    # index -> rows
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
    print(f"[split] counts: {counts}")

    if ns.dry_run:
        print("[split] dry-run: no files written.")
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
        "ratio": ratio,
        "n_train_fixed": n_train_fixed,
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
    print(f"[split] wrote: labeled/dev/test/pool/meta -> {data_dir}")

    return 0


if __name__ == "__main__":
    print("Run via CLI: tensaku split ...")