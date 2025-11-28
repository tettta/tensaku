# /home/esakit25/work/tensaku/src/tensaku/al_label_import.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.al_label_import
@role     : Active Learning サイクルで選ばれた pool レコードを labeled へ移し替える
            （研究モードでは「既存 gold ラベルの解禁」）インポータ。
@inputs   :
  - cfg["run"].data_dir : 分割済みデータディレクトリ（labeled.jsonl, pool.jsonl 等を含む）
  - cfg["run"].out_dir  : 実験出力ディレクトリ（al_sample_ids.txt などを含む）
  - cfg["data"].files.labeled : labeled ファイル名（既定 "labeled.jsonl"）
  - cfg["data"].files.pool    : pool ファイル名（既定 "pool.jsonl"）
  - cfg["data"].id_key        : ID キー（既定 "id"）
  - cfg["data"].label_key     : ラベルキー（既定 "score"）
  - CLI 引数:
      --labels PATH : new_labels.csv（列 "id","score" を含む）
@files   :
  - 入力:
      {out_dir}/al_sample_ids.txt   : 1 行 1 ID（AL で「ラベル解禁」すべき候補）
      {labels_csv}                  : 列 id,score を含む CSV（人手 or オラクルラベル）
      {data_dir}/{labeled}.jsonl    : 既存 labeled_k
      {data_dir}/{pool}.jsonl       : 既存 pool_k
  - 出力:
      {data_dir}/{labeled}.jsonl    : labeled_{k+1}（al_sample_ids ∩ labels の分だけ追記）
      {data_dir}/{pool}.jsonl       : pool_{k+1}（追記分の ID を除外した残り）
      {out_dir}/al_label_import.log : 取り込み統計のログ
@notes   :
  - PL（擬似ラベル）は扱わず、labels_csv は gold or 人手ラベルのみを前提とする。
  - labeled/pool は JSONL ベースで管理し、ID は cfg["data"].id_key で指定される。
  - al_sample_ids.txt に含まれていても、labels_csv に score が無い ID は
    「まだラベルされていない」と見なし、pool に残す。
  - 既に labeled 側に存在する ID は二重追加を避けるが、pool 側に残っていた場合は
    取り除く（inconsistency 自動修正）。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .config import load_config  # type: ignore[import]


# =============================================================================
# dataclass / ユーティリティ
# =============================================================================


@dataclass
class ImportStats:
    """al-label-import の統計値。"""

    added: int = 0                # labeled に新規追加した件数
    already_labeled: int = 0      # 既に labeled に存在していた ID
    missing_in_pool: int = 0      # al_sample_ids にあるが pool 側に無い ID
    missing_in_labels: int = 0    # al_sample_ids にあるが labels_csv に無い ID
    malformed_csv: int = 0        # CSV の壊れ行数（列欠損・score 非整数など）
    removed_from_pool: int = 0    # pool から除外された ID 件数


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
                # 壊れ行はスキップ（ここで例外を投げると全体が止まるため）
                continue
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """rows を JSONL 形式で**上書き**し、書き込んだ件数を返す。"""
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """rows を JSONL 形式で追記し、追記件数を返す。"""
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _load_pool_index(pool_rows: List[dict], id_key: str) -> Dict[str, dict]:
    """pool_rows から ID -> レコード の辞書を構築する。"""
    index: Dict[str, dict] = {}
    for rec in pool_rows:
        i = rec.get(id_key)
        if i is None:
            continue
        index[str(i)] = rec
    return index


def _load_labeled_ids(labeled_rows: List[dict], id_key: str) -> Set[str]:
    """既存 labeled に含まれる ID 集合を返す。"""
    ids: Set[str] = set()
    for rec in labeled_rows:
        i = rec.get(id_key)
        if i is None:
            continue
        ids.add(str(i))
    return ids


def _read_labels_csv(path: Path, stats: ImportStats) -> List[Tuple[str, int]]:
    """labels CSV を読み込んで (id, score) のリストを返す。

    前提:
      - 1 行目はヘッダ。
      - 必須列: "id", "score"。
      - score は整数（str -> int）に変換可能なもののみ採用。
    """
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        if "id" not in fields or "score" not in fields:
            raise ValueError(
                f"[al-label-import] labels CSV must have 'id' and 'score' columns. got={fields}"
            )
        for row in r:
            raw_id = row.get("id")
            raw_score = row.get("score")
            if raw_id is None or raw_id == "" or raw_score is None or raw_score == "":
                stats.malformed_csv += 1
                continue
            try:
                score = int(raw_score)
            except Exception:
                stats.malformed_csv += 1
                continue
            rows.append((str(raw_id), score))
    return rows

def _read_oracle_labels_from_pool(
    pool_index: Dict[str, dict],
    sample_ids_path: Path,
    stats: Optional[ImportStats] = None,
) -> List[tuple[str, int]]:
    """研究モードオラクル用:
    out_dir/al_sample_ids.txt と pool_index から (id, score) のリストを生成する。

    - al_sample_ids.txt: 1 行 1 id（文字列）
    - pool_index       : id -> レコード（少なくとも "score" を含むことを期待）
    - stats            : 渡された場合は missing_in_pool / missing_score_pool / malformed_csv をここで更新する。
    """
    rows: List[tuple[str, int]] = []

    if not sample_ids_path.exists():
        raise FileNotFoundError(
            f"[al-label-import] oracle mode: al_sample_ids.txt not found: {sample_ids_path}"
        )

    with sample_ids_path.open("r", encoding="utf-8") as f:
        for line in f:
            rid = line.strip()
            if not rid or rid.startswith("#"):
                # 空行やコメント行はスキップ
                continue

            rec = pool_index.get(rid)
            if rec is None:
                if stats is not None:
                    stats.missing_in_pool += 1
                continue

            if "score" not in rec:
                if stats is not None:
                    stats.missing_score_pool += 1
                continue

            try:
                score = int(rec["score"])
            except Exception:
                # pool 側の score が整数に解釈できない場合
                if stats is not None:
                    stats.malformed_csv += 1
                continue

            rows.append((rid, score))

    return rows



def _read_al_sample_ids(path: Path) -> List[str]:
    """al_sample_ids.txt から ID リストを読み込む（空行はスキップ）。"""
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s)
    return ids


def _log_stats(out_dir: Path, stats: ImportStats, n_sample_ids: int, n_label_rows: int) -> None:
    """out_dir/al_label_import.log に簡易ログを追記する。"""
    log_path = out_dir / "al_label_import.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"n_sample_ids={n_sample_ids}"
        f" n_label_rows={n_label_rows}"
        f" added={stats.added}"
        f" removed_from_pool={stats.removed_from_pool}"
        f" already_labeled={stats.already_labeled}"
        f" missing_in_pool={stats.missing_in_pool}"
        f" missing_in_labels={stats.missing_in_labels}"
        f" malformed_csv={stats.malformed_csv}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# =============================================================================
# 本体
# =============================================================================


def main_impl(cfg: Dict[str, Any], labels_csv: Path) -> int:
    """al_sample_ids.txt ＋ labels_csv をもとに labeled/pool を更新する本体処理。"""
    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    files_cfg: Dict[str, Any] = data_cfg.get("files") or {}

    data_dir = Path(run_cfg["data_dir"])
    out_dir = Path(run_cfg["out_dir"])

    id_key = data_cfg.get("id_key", "id")
    label_key = data_cfg.get("label_key", "score")

    labeled_name = files_cfg.get("labeled", "labeled.jsonl")
    pool_name = files_cfg.get("pool", "pool.jsonl")

    labeled_path = data_dir / labeled_name
    pool_path = data_dir / pool_name
    sample_ids_path = out_dir / "al_sample_ids.txt"

    print(f"[al-label-import] data_dir={data_dir}")
    print(f"[al-label-import] out_dir={out_dir}")
    print(f"[al-label-import] labeled={labeled_path}")
    print(f"[al-label-import] pool={pool_path}")
    print(f"[al-label-import] al_sample_ids={sample_ids_path}")
    print(f"[al-label-import] labels_csv={labels_csv}")

    # 存在チェック
    if not pool_path.exists():
        print(f"[al-label-import] ERROR: {pool_path} not found.")
        return 1
    if not sample_ids_path.exists():
        print(f"[al-label-import] ERROR: {sample_ids_path} not found (run al-sample first).")
        return 1
    if not labels_csv.exists():
        print(f"[al-label-import] ERROR: {labels_csv} not found.")
        return 1

    # 1) データ読み込み
    pool_rows = _read_jsonl(pool_path)
    labeled_rows = _read_jsonl(labeled_path)
    print(f"[al-label-import] pool_rows={len(pool_rows)}")
    print(f"[al-label-import] labeled_rows={len(labeled_rows)}")

    pool_index = _load_pool_index(pool_rows, id_key=id_key)
    labeled_ids = _load_labeled_ids(labeled_rows, id_key=id_key)
    print(f"[al-label-import] pool_index size={len(pool_index)}")
    print(f"[al-label-import] labeled_ids size={len(labeled_ids)}")

    # 2) al_sample_ids / labels_csv を読み込み
    stats = ImportStats()
    try:
        sample_ids = _read_al_sample_ids(sample_ids_path)
    except Exception as e:
        print(f"[al-label-import] ERROR: failed to read {sample_ids_path}: {e}")
        return 1

    try:
        label_rows = _read_labels_csv(labels_csv, stats=stats)
    except ValueError as e:
        print(str(e))
        return 1

    print(f"[al-label-import] n_sample_ids={len(sample_ids)}")
    print(f"[al-label-import] n_label_rows={len(label_rows)}")
    if stats.malformed_csv:
        print(f"[al-label-import] malformed_csv rows={stats.malformed_csv}")

    # 3) labels_csv を id -> score の辞書へ変換（後勝ち）
    label_map: Dict[str, int] = {}
    for rid, score in label_rows:
        label_map[rid] = score

    # 4) labeled に追加すべきレコード＆ pool から除外すべき ID を決定
    to_append: List[dict] = []
    to_remove_ids: Set[str] = set()

    for sid in sample_ids:
        rec = pool_index.get(sid)
        if rec is None:
            stats.missing_in_pool += 1
            continue

        # 既に labeled に存在する ID
        if sid in labeled_ids:
            stats.already_labeled += 1
            to_remove_ids.add(sid)  # pool 側に残っているなら除外する
            continue

        score = label_map.get(sid)
        if score is None:
            # まだラベルされていない → labeled に追加せず pool に残す
            stats.missing_in_labels += 1
            continue

        # labeled に追加するレコードを構築
        new_rec = dict(rec)
        new_rec[label_key] = score
        to_append.append(new_rec)
        labeled_ids.add(sid)
        to_remove_ids.add(sid)

    # 5) labeled.jsonl に追記
    if to_append:
        added = _append_jsonl(labeled_path, to_append)
        stats.added = added
    else:
        stats.added = 0

    # 6) pool.jsonl を更新（除外対象を取り除く）
    if to_remove_ids:
        new_pool_rows: List[dict] = []
        for rec in pool_rows:
            rid = rec.get(id_key)
            if rid is None:
                new_pool_rows.append(rec)
                continue
            if str(rid) in to_remove_ids:
                continue
            new_pool_rows.append(rec)
        stats.removed_from_pool = len(to_remove_ids)
        written = _write_jsonl(pool_path, new_pool_rows)
        print(
            f"[al-label-import] pool updated: removed_ids={len(to_remove_ids)}"
            f" -> pool_rows {len(pool_rows)} -> {written}"
        )
    else:
        print("[al-label-import] pool unchanged (no ids to remove).")

    # 7) ログ出力
    _log_stats(out_dir, stats, n_sample_ids=len(sample_ids), n_label_rows=len(label_rows))

    # 8) 結果メッセージ
    print(
        "[al-label-import] result:"
        f" added={stats.added}"
        f" removed_from_pool={stats.removed_from_pool}"
        f" already_labeled={stats.already_labeled}"
        f" missing_in_pool={stats.missing_in_pool}"
        f" missing_in_labels={stats.missing_in_labels}"
        f" malformed_csv={stats.malformed_csv}"
    )

    if stats.added == 0:
        print(
            "[al-label-import] NOTE: no new records appended to labeled."
            " (ids may be missing in pool/labels or already labeled)"
        )
    else:
        print(f"[al-label-import] DONE: appended {stats.added} records to {labeled_path}")

    return 0


# =============================================================================
# run / CLI エントリポイント
# =============================================================================


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    CLI エントリポイント。

    - tensaku CLI から呼び出される場合:
        cfg は tensaku.cli が構築した dict を受け取り、
        argv にはサブコマンド以降の引数（例: ["--labels", "foo.csv"]）が入る。
    - 単体実行 (`python -m tensaku.al_label_import -c ... --labels ...`) の場合:
        cfg は None で、ここで YAML を読み込む。
    """
    if cfg is None:
        # 単体実行モード（旧来互換）
        parser = argparse.ArgumentParser(prog="tensaku al-label-import")
        parser.add_argument(
            "-c",
            "--config",
            dest="config",
            default="configs/exp_al_hitl.yaml",
            help="YAML config file path (default: configs/exp_al_hitl.yaml)",
        )
        parser.add_argument(
            "--labels",
            required=True,
            help="CSV file containing columns 'id' and 'score' (oracle or human labels).",
        )
        args = parser.parse_args(argv)
        cfg_path = Path(args.config)
        cfg = load_config(str(cfg_path), [])
        labels_csv = Path(args.labels)
        return main_impl(cfg, labels_csv)
    else:
        # tensaku CLI 経由
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--labels",
            required=True,
            help="CSV file containing columns 'id' and 'score' (oracle or human labels).",
        )
        ns, _rest = parser.parse_known_args(argv or [])
        labels_csv = Path(ns.labels)
        return main_impl(cfg, labels_csv)


if __name__ == "__main__":
    # 単体スモークテスト用エントリ
    # 例:
    #   python -m tensaku.al_label_import -c configs/exp_al_hitl.yaml --labels _al_label_smoke/new_labels.csv
    raise SystemExit(run())
