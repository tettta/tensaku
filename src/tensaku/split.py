"""
@module: split
@role: "all.jsonl"（score有/無 混在可）から学習/検証/テスト/プールを作るデータ分割ユーティリティ
@inputs: {run.data_dir}/{data.files.*} または data.input_all（JSONL: id, mecab|text, score?）
@outputs: {data_dir}/labeled.jsonl, {data_dir}/dev.jsonl, {data_dir}/test.jsonl, {data_dir}/pool.jsonl
@cli: tensaku split
@notes: freeze_dev_test=True のとき dev/test を固定維持。未採点は pool に落とす。
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, json, math, random

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 壊れ行はスキップ
                pass
    return rows

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _ensure_id(rows: List[Dict[str, Any]], id_key: str = "id") -> None:
    seen = set()
    for i, r in enumerate(rows):
        if id_key not in r or r[id_key] in (None, ""):
            r[id_key] = f"auto-{i}"
        # 衝突があればナンバリング
        x = r[id_key]
        j = 1
        while x in seen:
            x = f"{r[id_key]}-{j}"; j += 1
        r[id_key] = x
        seen.add(x)

def _text_of(r: Dict[str, Any], pri: str, fb: str) -> str | None:
    v = r.get(pri)
    if v is None or (isinstance(v, str) and not v.strip()):
        v = r.get(fb)
    if isinstance(v, list): v = " ".join(map(str, v))
    if isinstance(v, str): return v.strip()
    return None

def _split_stratified(labeled: List[Dict[str, Any]], label_key: str,
                      ratios: Dict[str, float], seed: int) -> Tuple[List, List, List]:
    # 層化：scoreごとにシャッフルして比率配分（端数は切り上げで最低1件を目指す）
    rng = random.Random(seed)
    by_cls: Dict[int, List[Dict[str, Any]]] = {}
    for r in labeled:
        by_cls.setdefault(int(r[label_key]), []).append(r)
    train, dev, test = [], [], []
    for c, items in by_cls.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = max(0, int(round(n * ratios.get("train", 0.7))))
        n_dv = max(0, int(round(n * ratios.get("dev", 0.15))))
        # 調整して総数合わせ
        n_te = max(0, n - n_tr - n_dv)
        # 少数クラス保護：各集合が0にならないよう微調整（可能な範囲で）
        if n >= 3:
            if n_tr == 0: n_tr, n_te = 1, max(0, n_te-1)
            if n_dv == 0 and n >= 2: n_dv, n_tr = 1, max(0, n_tr-1)
            if n_te == 0 and n >= 2: n_te, n_tr = 1, max(0, n_tr-1)
        train += items[:n_tr]
        dev   += items[n_tr:n_tr+n_dv]
        test  += items[n_tr+n_dv:]
    return train, dev, test

def run(cfg: Dict[str, Any]) -> None:
    run = cfg["run"]; data = cfg["data"]; split = cfg["split"]
    data_dir = run["data_dir"]
    pri, fb = data.get("text_key_primary","mecab"), data.get("text_key_fallback","text")
    label_key, id_key = data.get("label_key","score"), data.get("id_key","id")
    seed = run.get("seed", 42)

    input_all = data.get("input_all")
    paths = {k: os.path.join(data_dir, v) for k, v in data["files"].items()}

    if input_all:  # 未分割 all.jsonl → labeled/pool/dev/test を生成
        src = input_all if os.path.isabs(input_all) else os.path.join(data_dir, input_all)
        rows = _read_jsonl(src)
        # 正規化と振り分け
        labeled, pool = [], []
        for r in rows:
            r[id_key] = r.get(id_key)  # あとで埋め直す
            txt = _text_of(r, pri, fb)
            if txt is None: continue  # テキスト無しは除外
            r[pri] = txt  # 正規化して主キーへ寄せる
            if label_key in r and r[label_key] is not None:
                try:
                    r[label_key] = int(r[label_key])
                    if r[label_key] < 0: r[label_key] = 0
                    labeled.append(r)
                except Exception:
                    pool.append(r)  # ラベル不正は未採点扱い
            else:
                pool.append(r)

        _ensure_id(labeled, id_key); _ensure_id(pool, id_key)

        # 層化分割（既存 dev/test を保持したいなら freeze_dev_test=True を尊重）
        tr, dv, te = _split_stratified(labeled, label_key, split.get("ratios", {}), seed)
        _write_jsonl(paths["labeled"], tr)
        if not split.get("freeze_dev_test", True) or not os.path.exists(paths["dev"]):
            _write_jsonl(paths["dev"], dv)
        if not split.get("freeze_dev_test", True) or not os.path.exists(paths["test"]):
            _write_jsonl(paths["test"], te)
        _write_jsonl(paths["pool"], pool)

        print(f"[split] from {os.path.relpath(src, data_dir)} -> "
              f"labeled={len(tr)}, dev={len(dv)}, test={len(te)}, pool={len(pool)}")
    else:
        # 既存ファイルの存在確認のみ
        miss = [k for k,p in paths.items() if not os.path.exists(p)]
        if miss:
            print(f"[split] missing files: {miss}. "
                  f"Provide data.input_all or create files under {data_dir}.")
        else:
            print(f"[split] files exist under {data_dir}: {list(paths.values())}")
