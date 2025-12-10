# /home/esakit25/work/tensaku/scripts/run_split_all.py
# -*- coding: utf-8 -*-
"""
@role: data_sas/all.jsonl から全 QID を抽出し、tensaku split を QID ごとに一括実行するヘルパースクリプト
@usage:
    cd /home/esakit25/work/tensaku

    # デフォルト設定で実行（config: configs/exp_al_hitl.yaml, all.jsonl / splits もデフォルト）
    python scripts/run_split_all.py

    # 別の config や all.jsonl を使いたい場合
    python scripts/run_split_all.py \
        --config configs/exp_al_hitl.yaml \
        --root . \
        --all-path data_sas/all.jsonl \
        --splits-root data_sas/splits
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _load_qids_from_all(all_path: Path) -> List[str]:
    """all.jsonl から qid のユニーク集合を抽出する。"""
    qids: Set[str] = set()
    if not all_path.exists():
        raise FileNotFoundError(f"all.jsonl not found: {all_path}")

    with all_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("qid")
            if qid is None:
                continue
            qids.add(str(qid))

    return sorted(qids)


def _run_split_for_qid(
    cfg_path: Path,
    root: Path,
    qid: str,
    splits_root: Path,
) -> Tuple[int, str]:
    """
    単一 QID について tensaku split を実行する。

    Returns:
        (exit_code, message)
    """
    # 出力先ディレクトリ: <root>/<splits_root>/<qid>
    out_dir = (root / splits_root / qid).resolve()

    cmd = [
        "tensaku",
        "split",
        "-c",
        str(cfg_path),
        "--set",
        f"data.qid={qid}",
        "--set",
        f"run.data_dir={out_dir}",
    ]

    print(f"[split-all] QID={qid}  out_dir={out_dir}")
    print(f"[split-all]   cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        return 1, f"subprocess error: {e}"

    if proc.returncode != 0:
        # 失敗時は標準出力・エラーを少しだけ出す
        print(f"[split-all]   ERROR: exit_code={proc.returncode}")
        if proc.stdout:
            print("[split-all]   stdout (tail):")
            for line in proc.stdout.strip().splitlines()[-5:]:
                print("      " + line)
        if proc.stderr:
            print("[split-all]   stderr (tail):")
            for line in proc.stderr.strip().splitlines()[-5:]:
                print("      " + line)
        return proc.returncode, "failed"

    print(f"[split-all]   OK")
    return 0, "success"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/exp_al_hitl.yaml",
        help="YAML config for tensaku split",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root (TENSAKU_ROOT と揃える)",
    )
    parser.add_argument(
        "--all-path",
        type=str,
        default="data_sas/all.jsonl",
        help="Master all.jsonl path (relative to root)",
    )
    parser.add_argument(
        "--splits-root",
        type=str,
        default="data_sas/splits",
        help="Splits root dir (relative to root). Each QID will be placed under here.",
    )
    parser.add_argument(
        "--qid",
        type=str,
        nargs="*",
        help="If specified, only these QIDs will be split. Otherwise, all QIDs in all.jsonl.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg_path = (root / args.config).resolve()
    all_path = (root / args.all_path).resolve()
    splits_root = Path(args.splits_root)

    print(f"[split-all] root       = {root}")
    print(f"[split-all] config     = {cfg_path}")
    print(f"[split-all] all_path   = {all_path}")
    print(f"[split-all] splits_root= {splits_root}")

    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    if not all_path.exists():
        raise FileNotFoundError(f"all.jsonl not found: {all_path}")

    # QID リストを決定
    if args.qid:
        qids = sorted(set(args.qid))
        print(f"[split-all] Using QIDs from CLI: {qids}")
    else:
        qids = _load_qids_from_all(all_path)
        print(f"[split-all] Detected QIDs from all.jsonl: {qids}")

    # 実行 & 結果記録
    results: List[Dict[str, str]] = []
    n_success = 0
    n_error = 0

    for qid in qids:
        exit_code, msg = _run_split_for_qid(
            cfg_path=cfg_path,
            root=root,
            qid=qid,
            splits_root=splits_root,
        )
        status = "success" if exit_code == 0 else "error"
        if status == "success":
            n_success += 1
        else:
            n_error += 1
        results.append(
            {
                "qid": qid,
                "status": status,
                "exit_code": str(exit_code),
                "message": msg,
            }
        )

    # ステータスを CSV に保存
    out_status_path = root / "outputs" / "split_status.csv"
    out_status_path.parent.mkdir(parents=True, exist_ok=True)

    with out_status_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qid", "status", "exit_code", "message"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(
        f"[split-all] Summary: success={n_success}, error={n_error}, "
        f"status_csv={out_status_path}"
    )


if __name__ == "__main__":
    main()
