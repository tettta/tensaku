# /home/esakit25/work/tensaku/src/tensaku/al_oracle_labels.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.al_oracle_labels
@role     : Active Learning 研究モード向け Oracle ラベラ
            （pool.jsonl の gold ラベルから、al_label_import 用 CSV を自動生成する）
@inputs   :
  - cfg["run"].data_dir : 分割済みデータディレクトリ（labeled.jsonl, pool.jsonl 等を含む）
  - cfg["run"].out_dir  : 実験出力ディレクトリ（al_sample_ids.txt 等を含む）
  - cfg["data"].files.pool : pool ファイル名（既定 "pool.jsonl"）
  - cfg["data"].id_key     : ID キー（既定 "id"）
  - cfg["data"].label_key  : ラベルキー（既定 "score"）
  - CLI 引数:
      --ids PATH : al_sample_ids.txt（1 行 1 ID）のパス
                   省略時は {out_dir}/al_sample_ids.txt
      --out PATH : 生成する labels CSV のパス
                   省略時は {out_dir}/oracle_labels.csv
@files   :
  - 入力:
      {out_dir}/al_sample_ids.txt      : 1 行 1 ID（AL が「採点候補」として選んだ pool の ID 群）
      {data_dir}/{pool}.jsonl          : gold ラベル付き pool レコード
  - 出力:
      {out_dir}/oracle_labels.csv      : 列 "id","score" を持つ CSV（al_label_import 用）
        ※ --out で任意のパス（例: oracle_labels_round0.csv）に変更可
@api      :
  - main_impl(cfg, ids_path: Optional[Path], out_csv: Optional[Path]) -> int
  - run(argv: Optional[list[str]], cfg: Optional[dict]) -> int
@cli      :
  - Python モジュールとして:
      python -m tensaku.al_oracle_labels -c configs/exp_al_hitl.yaml
      python -m tensaku.al_oracle_labels -c CFG.yaml --ids .../al_sample_ids.txt --out .../oracle_labels_round0.csv
  - tensaku CLI からは `tensaku al-oracle-labels` サブコマンド経由で run(cfg=..., argv=...) が呼ばれる想定
@notes    :
  - ここでは「Oracle = gold ラベルを知っている人」とみなし、al_label_import に渡す CSV を
    自動生成する役割のみに徹する。
  - al_label_import 側は「(id,score) CSV しか知らない」設計のまま維持し、
    人間 HITL と Oracle を同じインターフェースで扱えるようにしている。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import load_config  # type: ignore[import]


# =============================================================================
# dataclass / ユーティリティ
# =============================================================================


@dataclass
class OracleStats:
    """Oracle ラベル生成の統計値。"""

    total_ids: int = 0         # al_sample_ids.txt に含まれる ID 数
    found_in_pool: int = 0     # pool からラベルを取得できた ID 数
    missing_in_pool: int = 0   # pool 側に存在しなかった ID 数
    missing_label: int = 0     # pool レコードに label_key が無かった ID 数
    malformed_label: int = 0   # ラベルが int に変換できなかった ID 数
    written_rows: int = 0      # CSV に実際に書き込んだ行数


def _read_jsonl(path: Path) -> List[dict]:
    """シンプルな JSONL ローダ（壊れ行は静かにスキップ）。"""
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 壊れ行は静かにスキップ（研究モード前提のため寛容に扱う）
                continue
    return rows


def _load_pool_index(pool_rows: List[dict], id_key: str) -> Dict[str, dict]:
    """pool のレコードから id -> record のインデックスを構築する。"""
    index: Dict[str, dict] = {}
    for rec in pool_rows:
        rid = rec.get(id_key)
        if rid is None:
            continue
        index[str(rid)] = rec
    return index


def _read_sample_ids(path: Path) -> List[str]:
    """al_sample_ids.txt から ID リストを読み込む。"""
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s)
    return ids


def _extract_label(rec: dict, label_key: str) -> Optional[int]:
    """レコードから label_key を取り出し、int へ変換（失敗時は None）。"""
    if label_key not in rec:
        return None
    val = rec[label_key]
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _write_labels_csv(path: Path, rows: List[Tuple[str, int]]) -> int:
    """(id,score) row 群を CSV に書き出し、書き込んだ行数を返す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for rid, score in rows:
            w.writerow([rid, str(score)])
    return len(rows)


# =============================================================================
# 本体
# =============================================================================


def main_impl(
    cfg: Dict[str, Any],
    ids_path: Optional[Path] = None,
    out_csv: Optional[Path] = None,
) -> int:
    """
    al_sample_ids.txt ＋ pool.jsonl から (id,score) CSV を生成する本体処理。

    - ids_path が None の場合は {out_dir}/al_sample_ids.txt を仮定。
    - out_csv が None の場合は {out_dir}/oracle_labels.csv を仮定。
    """
    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    files_cfg: Dict[str, Any] = data_cfg.get("files") or {}

    data_dir = Path(run_cfg["data_dir"])
    out_dir = Path(run_cfg["out_dir"])

    pool_name = files_cfg.get("pool", "pool.jsonl")
    pool_path = data_dir / pool_name

    if ids_path is None:
        ids_path = out_dir / "al_sample_ids.txt"
    else:
        ids_path = Path(ids_path)

    if out_csv is None:
        out_csv = out_dir / "oracle_labels.csv"
    else:
        out_csv = Path(out_csv)

    id_key = data_cfg.get("id_key", "id")
    label_key = data_cfg.get("label_key", "score")

    print(f"[al-oracle-labels] data_dir={data_dir}")
    print(f"[al-oracle-labels] out_dir={out_dir}")
    print(f"[al-oracle-labels] pool_path={pool_path}")
    print(f"[al-oracle-labels] ids_path={ids_path}")
    print(f"[al-oracle-labels] out_csv={out_csv}")
    print(f"[al-oracle-labels] id_key={id_key} label_key={label_key}")

    if not pool_path.exists():
        print(f"[al-oracle-labels] ERROR: pool file not found: {pool_path}")
        return 1
    if not ids_path.exists():
        print(f"[al-oracle-labels] ERROR: sample IDs file not found: {ids_path}")
        return 1

    # 1) pool / al_sample_ids を読み込む
    pool_rows = _read_jsonl(pool_path)
    print(f"[al-oracle-labels] pool_rows={len(pool_rows)}")

    pool_index = _load_pool_index(pool_rows, id_key=id_key)
    print(f"[al-oracle-labels] pool_index size={len(pool_index)}")

    sample_ids = _read_sample_ids(ids_path)
    print(f"[al-oracle-labels] n_sample_ids={len(sample_ids)}")

    stats = OracleStats(total_ids=len(sample_ids))

    # 2) sample_ids について pool から score を引き出す
    label_rows: List[Tuple[str, int]] = []
    for sid in sample_ids:
        rec = pool_index.get(sid)
        if rec is None:
            stats.missing_in_pool += 1
            continue
        score = _extract_label(rec, label_key=label_key)
        if score is None:
            # ラベルキーが無いか、int 化できない
            if label_key not in rec:
                stats.missing_label += 1
            else:
                stats.malformed_label += 1
            continue
        label_rows.append((sid, score))
        stats.found_in_pool += 1

    # 3) CSV に書き出し
    if not label_rows:
        print(
            "[al-oracle-labels] WARNING: no valid labels extracted; "
            "oracle CSV will not be created."
        )
        return 1

    written = _write_labels_csv(out_csv, label_rows)
    stats.written_rows = written

    # 4) 統計を表示
    print(
        "[al-oracle-labels] result:"
        f" total_ids={stats.total_ids}"
        f" found_in_pool={stats.found_in_pool}"
        f" missing_in_pool={stats.missing_in_pool}"
        f" missing_label={stats.missing_label}"
        f" malformed_label={stats.malformed_label}"
        f" written_rows={stats.written_rows}"
    )
    print(f"[al-oracle-labels] wrote labels CSV to {out_csv}")

    if stats.written_rows == 0:
        return 1
    return 0


# =============================================================================
# CLI エントリ
# =============================================================================


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    CLI エントリポイント。

    - 単体実行 (`python -m tensaku.al_oracle_labels -c ...`) の場合:
        cfg は None で、ここで YAML を読み込む。
    - tensaku CLI から呼び出される場合:
        cfg は tensaku.cli が構築した dict を受け取り、
        argv にはサブコマンド以降の引数（例: ["--ids", "...", "--out", "..."]）が入る。
    """
    if cfg is None:
        # 単体実行モード
        parser = argparse.ArgumentParser(prog="tensaku al-oracle-labels")
        parser.add_argument(
            "-c",
            "--config",
            dest="config",
            default="configs/exp_al_hitl.yaml",
            help="path to YAML config (default: configs/exp_al_hitl.yaml)",
        )
        parser.add_argument(
            "--ids",
            dest="ids",
            default=None,
            help="path to al_sample_ids.txt "
            "(default: {run.out_dir}/al_sample_ids.txt)",
        )
        parser.add_argument(
            "--out",
            dest="out",
            default=None,
            help="path to output labels CSV "
            "(default: {run.out_dir}/oracle_labels.csv)",
        )
        ns, _rest = parser.parse_known_args(argv)
        cfg = load_config(ns.config)
        ids_path = Path(ns.ids) if ns.ids is not None else None
        out_csv = Path(ns.out) if ns.out is not None else None
        return main_impl(cfg, ids_path=ids_path, out_csv=out_csv)


    else:
        # tensaku CLI 経由
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--ids",
            dest="ids",
            default=None,
            help="path to al_sample_ids.txt "
            "(default: {run.out_dir}/al_sample_ids.txt)",
        )
        parser.add_argument(
            "--out",
            dest="out",
            default=None,
            help="path to output labels CSV "
            "(default: {run.out_dir}/oracle_labels.csv)",
        )
        ns, _rest = parser.parse_known_args(argv)
        ids_path = Path(ns.ids) if ns.ids is not None else None
        out_csv = Path(ns.out) if ns.out is not None else None
        return main_impl(cfg, ids_path=ids_path, out_csv=out_csv)



if __name__ == "__main__":
    raise SystemExit(run())
