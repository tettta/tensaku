# /home/esakit25/work/tensaku/src/tensaku/al_log.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.al_log
@role     : Active Learning ラウンドのメタ情報を OUT_DIR/al_history.csv に1行追記する薄いユーティリティ
@inputs   :
  - cfg["run"].out_dir : 実験出力ディレクトリ（例: /home/esakit25/work/tensaku/outputs/q-Y14_1-2_1_3）
  - cfg["data"].qid    : 問題ID（あれば）
  - cfg["run"].run_id  : 実験ID（あれば）
  - cfg["train"].seed  : 乱数シード（あれば）
@files   :
  - 出力:
      {out_dir}/al_history.csv
        - ラウンドごとの履歴を追記していくログ。
        - カラム:
            qid, run_id, round, sampler, uncertainty, budget,
            n_labeled, n_pool, note, seed, timestamp
@api     :
  - append_round_history(
        cfg: dict,
        round_idx: int,
        sampler: str,
        uncertainty: str,
        budget: int,
        n_labeled: int,
        n_pool: int,
        note: str | None = None,
    ) -> None
@cli     :
  - 原則として他モジュール (al_sample, al_label_import, al.sh 経由など) から
    append_round_history を呼び出す想定。
  - 単体で python -m tensaku.al_log を叩いた場合はヘルプメッセージのみを表示。
@notes   :
  - 役割を「1行追記」に限定し、集約・分析は別モジュール/ノートブックで行う。
  - 書き込み先は cfg["run"].out_dir 固定とし、/tmp など外部ディレクトリは使用しない。
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional


AL_HISTORY_FILENAME = "al_history.csv"


def _get_out_dir(cfg: Dict[str, Any]) -> Path:
    """cfg から out_dir を取得（なければ ./outputs を既定とする）."""
    run_cfg = cfg.get("run") or {}
    out_dir_raw = run_cfg.get("out_dir", "./outputs")
    out_dir = Path(str(out_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def append_round_history(
    cfg: Dict[str, Any],
    round_idx: int,
    sampler: str,
    uncertainty: str,
    budget: int,
    n_labeled: int,
    n_pool: int,
    note: str | None = None,
) -> None:
    """
    OUT_DIR/al_history.csv に1行追記する。

    Parameters
    ----------
    cfg:
        tensaku 全体の YAML 設定を dict 化したもの。
    round_idx:
        0 始まりのラウンド番号。
    sampler:
        サンプラ名（例: "topk", "random", "hybrid" など）。
    uncertainty:
        不確実性指標のキー（例: "msp", "trust", "entropy" など）。
    budget:
        そのラウンドで新たに追加したサンプル数 K。
    n_labeled:
        ラウンド終了時点での labeled.jsonl の件数。
    n_pool:
        ラウンド終了時点での pool.jsonl の件数。
    note:
        任意のメモ。実験条件や備考に利用可能。
    """
    out_dir = _get_out_dir(cfg)
    history_path = out_dir / AL_HISTORY_FILENAME

    data_cfg = cfg.get("data") or {}
    run_cfg = cfg.get("run") or {}
    train_cfg = cfg.get("train") or {}

    qid = str(data_cfg.get("qid", "")) if data_cfg.get("qid") is not None else ""
    run_id = str(run_cfg.get("run_id", "")) if run_cfg.get("run_id") is not None else ""
    seed = str(train_cfg.get("seed", "")) if train_cfg.get("seed") is not None else ""

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")

    fieldnames: List[str] = [
        "qid",
        "run_id",
        "round",
        "sampler",
        "uncertainty",
        "budget",
        "n_labeled",
        "n_pool",
        "note",
        "seed",
        "timestamp",
    ]

    file_exists = history_path.exists()

    with history_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {
            "qid": qid,
            "run_id": run_id,
            "round": int(round_idx),
            "sampler": sampler,
            "uncertainty": uncertainty,
            "budget": int(budget),
            "n_labeled": int(n_labeled),
            "n_pool": int(n_pool),
            "note": note or "",
            "seed": seed,
            "timestamp": timestamp,
        }
        writer.writerow(row)

    print(f"[al-log] append round={round_idx} to {history_path}")


# -----------------------------------------------------------------------------
# 単体実行用の軽いエントリ
# -----------------------------------------------------------------------------

def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    単体実行時のエントリポイント。

    研究モードでは al.sh / 他モジュールから append_round_history を呼ぶ前提のため、
    ここでは簡単なヘルプメッセージのみを表示する。
    """
    _ = argv  # unused
    _ = cfg   # unused
    msg = (
        "[al-log] This module is intended to be used via append_round_history(cfg, ...).\n"
        "Example:\n"
        "  from tensaku.config import load_config\n"
        "  from tensaku.al_log import append_round_history\n\n"
        "  cfg = load_config('configs/exp_al_hitl.yaml', [])\n"
        "  append_round_history(cfg, round_idx=0, sampler='topk',\n"
        "                       uncertainty='msp', budget=10,\n"
        "                       n_labeled=50, n_pool=790,\n"
        "                       note='oracle/msp/topk-K=10')\n"
    )
    print(msg)
    return 0


if __name__ == "__main__":
    # python -m tensaku.al_log したとき用
    raise SystemExit(run())
