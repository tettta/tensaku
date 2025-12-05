# /home/esakit25/work/tensaku/src/tensaku/split.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.split
@role     : all.jsonl（マスタ）から単一 QID の {labeled, dev, test, pool} を生成する（研究モード用スプリット）
@inputs   :
  - YAML:
      run.data_dir    : 出力先ディレクトリ（例: /.../data_sas/q-Y14_1-2_1_3）
      data.qid        : 対象とする QID（例: "Y14_1-2_1_3"）
      data.input_all  : all.jsonl の所在（ファイル or ディレクトリ）。未指定時は run.data_dir/all.jsonl
      data.label_key  : ラベルキー名（既定 "score"）
      split.seed      : 乱数シード（既定 42）
      split.ratio     : 各 split の比率 dict（keys: labeled/dev/test/pool or train/dev/test/pool）
      split.stratify  : True なら score ベースの層化分割（既定 True）
  - ファイル:
      {DATA}/all.jsonl : QID 混在のマスタ。各行は {qid, text系, score, id?} など。
@outputs  :
  - {run.data_dir}/labeled.jsonl
  - {run.data_dir}/dev.jsonl
  - {run.data_dir}/test.jsonl
  - {run.data_dir}/pool.jsonl
  - {run.data_dir}/meta.json  （qid/seed/ratio/件数/層化の有無/元 all.jsonl パス/ラベル統計 等）
 （qid/seed/ratio/件数/層化の有無/元 all.jsonl パス 等）
@cli      : tensaku split -c CFG.yaml --set data.qid="Y14_1-2_1_3"
@api      : run(argv: Optional[list[str]], cfg: dict[str, Any]) -> int
@deps     : 標準ライブラリのみ
@contracts:
  - all.jsonl が存在しない / qid 一致行が 0 件のときは安全にエラー終了（コード 2）。
  - score は 0..N の int を想定（層化時）。
  - id が無い行には `_ensure_id` で一意 ID を付与（qid_000001 のような形式）。
@errors   :
  - ファイル未発見や qid 不一致は stderr に詳細を記録し、終了コード 2。
  - split.ratio が欠落/ゼロ合計の場合は素朴な既定比率 {test:0.2, dev:0.1, labeled:0.2, pool:0.5} を使用。
@notes    :
  - 研究モード仕様書 5.1 に対応：test を先に hold-out し、残りを labeled/dev/pool に振り分ける。
  - PL なし前提：pool は「未ラベル集合」として扱い、学習には使用しない。
  - ratio の "labeled" と "train" はエイリアス扱い。内部では "labeled" に正規化する。
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
                # 壊れ行は警告せずスキップ（研究用途なので寛容に扱う）
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
    """data.input_all から all.jsonl のパスを推定。未指定時は run.data_dir/all.jsonl。"""
    inp = data_cfg.get("input_all")
    if not inp:
        return os.path.join(run_data_dir, "all.jsonl")
    inp = str(inp)
    if os.path.isdir(inp):
        return os.path.join(inp, "all.jsonl")
    return inp  # ファイルパス前提


# ===================== 分割ロジック =====================


def _normalize_ratio(raw: Dict[str, float]) -> Dict[str, float]:
    """
    ratio dict を labeled/dev/test/pool に正規化する。
    - "train" があれば "labeled" の別名として扱う。
    - 値の合計が 0 または key が無ければ既定比率にフォールバック。
    """
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
    """
    合計 n_total を ratio で丸め配分。
    - 小数点以下は最大の余りから順に配分。
    - keys: "test", "dev", "labeled", "pool"
    """
    if n_total <= 0:
        return {k: 0 for k in ("test", "dev", "labeled", "pool")}

    keys = ["test", "dev", "labeled", "pool"]
    r = {k: float(ratio.get(k, 0.0)) for k in keys}
    s = sum(r.values()) or 1.0
    desired = {k: n_total * r[k] / s for k in keys}
    base = {k: int(desired[k]) for k in keys}
    used = sum(base.values())
    remain = max(0, n_total - used)

    # 余りを大きい順に振る
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
    """score による層化分割。"""
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
    """層化なしの単純ランダム分割。"""
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
    """
    all.jsonl（マスタ）から data.qid を持つ行のみ抽出し、
    {labeled, dev, test, pool}.jsonl に分割して run.data_dir に保存する。

    モード:
      - ratio モード: split.ratio に従って比率で分割（従来どおり）
      - n_train モード: split.n_train 件を labeled に固定し、残りを ratio の test/dev/pool 比率で分割
        （本プロジェクトではこちらをデフォルト運用とする）
    """
    # --- CLI 引数（最小限） ---
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    ns, _rest = parser.parse_known_args(argv or [])

    # --- YAML セクション ---
    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    split_cfg = cfg.get("split") or {}

    # --- 出力先ディレクトリ ---
    data_dir = run_cfg.get("data_dir") or data_cfg.get("data_dir")
    if not data_dir:
        print("[split] ERROR: run.data_dir or data.data_dir must be set.", file=sys.stderr)
        return 2
    data_dir = str(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # --- QID / ラベルキー等 ---
    qid = data_cfg.get("qid") or run_cfg.get("qid")
    if not qid:
        print("[split] ERROR: data.qid or run.qid must be set.", file=sys.stderr)
        return 2
    qid = str(qid)

    label_key = data_cfg.get("label_key") or "score"

    seed = int(split_cfg.get("seed") or 42)
    stratify_cfg = split_cfg.get("stratify")
    stratify = True if stratify_cfg is None else bool(stratify_cfg)

    # --- all.jsonl のパス推定 ---
    all_path = _infer_all_path(data_dir, data_cfg)
    if not os.path.exists(all_path):
        print(f"[split] ERROR: all.jsonl not found: {all_path}", file=sys.stderr)
        return 2

    # --- 比率 & n_train の取得 ---
    ratio_raw = split_cfg.get("ratio") or {}
    ratio = _normalize_ratio(ratio_raw)

    n_train_cfg = split_cfg.get("n_train")
    n_train: Optional[int] = None
    if n_train_cfg is not None:
        try:
            n_train = int(n_train_cfg)
        except Exception:
            print(f"[split] WARN: invalid split.n_train={n_train_cfg!r} (ignored)", file=sys.stderr)
            n_train = None

    # ---- all.jsonl 読み込み & qid でフィルタ ----
    all_rows = _read_jsonl(all_path)
    rows = [r for r in all_rows if str(r.get("qid")) == qid]
    if not rows:
        print(f"[split] ERROR: no rows found for qid={qid!r} in {all_path}", file=sys.stderr)
        return 2

    _ensure_id(rows, prefix=f"{qid}_")

    # ラベル取得（層化用 + メタ統計）
    labels: List[int] = []          # 層化用（int）
    label_values: List[float] = []  # 統計用（float）
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

    # ラベル統計（メタ用）
    if label_values:
        label_min = min(label_values)
        label_max = max(label_values)
    else:
        label_min = None
        label_max = None

    label_type = "int" if all_int_like else "float"

    # 分類として扱えそうなら num_labels を定義
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

        # --- n_train モードの比率調整 ---
    n_total = len(rows)
    mode = "ratio"
    if n_train is not None and n_total > 0:
        # 1) 元データに対して dev/test を ratio 通りに確保
        base_test = float(ratio.get("test", 0.0))
        base_dev = float(ratio.get("dev", 0.0))
        sum_td = base_test + base_dev

        # 境界条件①: dev+test が 1.0 を超えるのは論理破綻 → ERROR で落とす
        if sum_td >= 1.0:
            print(
                "[split] ERROR: split.ratio.test + split.ratio.dev "
                f"= {sum_td:.3f} >= 1.0; n_train/pool を割り当てる余地がありません。"
                " split.ratio を見直してください。",
                file=sys.stderr,
            )
            return 2

        # dev/test の近似件数（丸めの都合で ±1 程度はずれることがある）
        approx_n_test = int(round(n_total * base_test))
        approx_n_dev = int(round(n_total * base_dev))
        max_n_train = max(0, n_total - approx_n_test - approx_n_dev)

        # 2) n_train のクリップ（境界条件②は WARN にしてクリップ）
        if n_train < 0:
            print(
                f"[split] WARN: split.n_train={n_train} < 0; 0 にクリップします。",
                file=sys.stderr,
            )
            n_train = 0
        if n_train > max_n_train:
            print(
                "[split] WARN: split.n_train={n_train} が dev/test で確保した残りよりも大きいため "
                f"{max_n_train} にクリップします。",
                file=sys.stderr,
            )
            n_train = max_n_train

        labeled_ratio = float(n_train) / float(n_total) if n_total > 0 else 0.0

        # 3) pool は「dev/test/labeled を引いた残り」
        pool_ratio = max(0.0, 1.0 - base_test - base_dev - labeled_ratio)

        # n_train モードでは dev/test の比率は元 ratio をそのまま使う
        ratio = {
            "labeled": labeled_ratio,
            "test": base_test,
            "dev": base_dev,
            "pool": pool_ratio,
        }
        mode = "n_train"


    # --- ログ出力（設定サマリ） ---
    # 表示用に丸めた ratio（小数第 3 位まで。2 桁にしたければ round(v, 2) に）
    ratio_log = {k: round(float(v), 3) for k, v in ratio.items()}

    print(f"[split] qid={qid}  total={n_total}  all_path={all_path}")
    if mode == "n_train":
        print(f"[split] mode=n_train  n_train={n_train}")
    else:
        print("[split] mode=ratio (split.n_train not set)")
    print(f"[split] ratio={ratio_log}  seed={seed}  stratify={stratify}")
    print(
        f"[split] label_min={label_min} label_max={label_max} "
        f"label_type={label_type} num_labels={num_labels} is_classification={is_classification}"
    )


    # --- インデックス分割 ---
    if stratify:
        idx_splits = _split_indices_stratified(labels, ratio, seed=seed)
    else:
        idx_splits = _split_indices_simple(n_total, ratio, seed=seed)

    # index -> rows へ変換
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

    # ---- 書き出し ----
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
        # ラベル統計（将来の回帰タスクも考慮）
        "label_min": label_min,
        "label_max": label_max,
        "label_type": label_type,              # "int" / "float"
        "is_classification": is_classification,
        "num_labels": num_labels,              # 分類として扱える場合のみ int、それ以外は null
    }

    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[split] wrote: labeled/dev/test/pool/meta -> {data_dir}")

    return 0


if __name__ == "__main__":
    print("Run via CLI: tensaku split -c /path/to/cfg.yaml --set data.qid=...")
