# /home/esakit25/work/tensaku/scripts/make_all_from_raw.py
# -*- coding: utf-8 -*-
"""
@module     scripts.make_all_from_raw
@role       raw配下（Y14, Y15 など）の各問題ファイルから全レコードを収集し、単一の {all.json, all.jsonl} を生成する
@inputs     RAW_DIRS=[/home/esakit25/work/tensaku/data_sas/raw/Y14, /home/esakit25/work/tensaku/data_sas/raw/Y15]
@outputs    /home/esakit25/work/tensaku/data_sas/all.json（JSON配列）, /home/esakit25/work/tensaku/data_sas/all.jsonl（JSON Lines）
@cli        python /home/esakit25/work/tensaku/scripts/make_all_from_raw.py \
              --raw /home/esakit25/work/tensaku/data_sas/raw/Y14 /home/esakit25/work/tensaku/data_sas/raw/Y15 \
              --out-dir /home/esakit25/work/tensaku/data_sas
@api        main(argv=None) -> int
@contracts  各入力ファイルの拡張子は .json / .jsonl を想定。各レコードに qid を【ファイル名（拡張子除去）】で追加する
@errors     不正JSONはスキップ（警告表示）。必須キーは規定しない（集約のみ）。id未設定は qid連番で自動採番
@notes      split側の互換のため all.json と all.jsonl を**両方**出力（既存のどちらの実装でも読めるように）
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

DEF_OUT_DIR = "/home/esakit25/work/tensaku/data_sas"
DEF_RAW_DIRS = [
    "/home/esakit25/work/tensaku/data_sas/raw/Y14",
    "/home/esakit25/work/tensaku/data_sas/raw/Y15",
]

def _iter_files(dirs: List[str]) -> Iterable[Tuple[str, str]]:
    """対象ディレクトリから .json / .jsonl を列挙して (path, qid) を返す。qidはファイル名（拡張子除去）。"""
    exts = {".json", ".jsonl"}
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[warn] skip: not a dir: {d}")
            continue
        for name in sorted(os.listdir(d)):
            base, ext = os.path.splitext(name)
            if ext.lower() in exts:
                yield os.path.join(d, name), base  # qid = base

def _load_any_json_records(path: str) -> List[Dict[str, Any]]:
    """JSON/JSONL両対応。配列JSON or JSON Lines（1行=1JSON）。不正行はスキップ。"""
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if not txt:
            return rows
        if txt[0] == "[":  # JSON array
            arr = json.loads(txt)
            if isinstance(arr, list):
                for i, r in enumerate(arr):
                    if isinstance(r, dict):
                        rows.append(r)
                    else:
                        print(f"[warn] non-dict element (index {i}) in array: {path}")
            else:
                print(f"[warn] top-level is not list: {path}")
        else:
            # JSONL
            with open(path, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict):
                            rows.append(obj)
                        else:
                            print(f"[warn] non-dict line#{ln}: {path}")
                    except Exception as e:
                        print(f"[warn] bad json line#{ln}: {path} ({e})")
    except FileNotFoundError:
        print(f"[warn] not found: {path}")
    except Exception as e:
        print(f"[warn] read error: {path} ({e})")
    return rows

def _ensure_ids(rows: List[Dict[str, Any]], qid: str) -> None:
    """id未設定のレコードに qid_000001 のようなIDを付与（既存idは温存）。"""
    w = len(str(max(1, len(rows))))
    n = 0
    for r in rows:
        if "id" not in r or r["id"] in (None, "", []):
            r["id"] = f"{qid}_{str(n).zfill(w)}"
            n += 1

def build_all(raw_dirs: List[str]) -> List[Dict[str, Any]]:
    """rawディレクトリ群から全レコードを収集し、qidを付与して返す。"""
    all_rows: List[Dict[str, Any]] = []
    for path, qid in _iter_files(raw_dirs):
        rows = _load_any_json_records(path)
        if not rows:
            print(f"[info] empty/invalid: {path}")
            continue
        # すべてにqidを付与（既にあっても上書きで “ファイル名由来” を優先）
        for r in rows:
            r["qid"] = qid
        _ensure_ids(rows, qid)
        all_rows.extend(rows)
        print(f"[load] {qid}: +{len(rows)} from {os.path.basename(path)}")
    return all_rows

def _write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="*", default=DEF_RAW_DIRS, help="Raw dirs to scan (Y14, Y15 ...)")
    ap.add_argument("--out-dir", default=DEF_OUT_DIR, help="Output directory for all.json / all.jsonl")
    ns = ap.parse_args(argv)

    rows = build_all(ns.raw)
    out_json  = os.path.join(ns.out_dir, "all.json")
    out_jsonl = os.path.join(ns.out_dir, "all.jsonl")

    _write_json(out_json, rows)
    _write_jsonl(out_jsonl, rows)

    print(f"[make_all] wrote: {out_json}  ({len(rows)} rows)")
    print(f"[make_all] wrote: {out_jsonl} ({len(rows)} rows)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
