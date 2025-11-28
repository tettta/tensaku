# scripts/aggregate_al_rounds.py
# -*- coding: utf-8 -*-
"""
@role: 指定ディレクトリ内の hitl_summary_round*.csv を集約し、al_learning_curve.csv を生成する
"""
import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict, Any

def get_round_idx(filename: str) -> int:
    # hitl_summary_round12.csv -> 12
    m = re.search(r"round(\d+)", filename)
    return int(m.group(1)) if m else -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Target OUT_DIR")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    if not out_dir.exists():
        print(f"[aggregate] ERROR: Directory not found: {out_dir}")
        return

    # ラウンドごとのサマリファイルを探索
    summary_files = list(out_dir.glob("hitl_summary_round*.csv"))
    if not summary_files:
        print(f"[aggregate] WARN: No hitl_summary_round*.csv found in {out_dir}")
        return

    # ラウンド順にソート
    summary_files.sort(key=lambda p: get_round_idx(p.name))

    # al_history.csv (メタ情報) もあれば読む
    history_map = {}
    history_path = out_dir / "al_history.csv"
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r_idx = row.get("round")
                if r_idx is not None:
                    history_map[int(r_idx)] = row

    aggregated_rows = []
    
    for p in summary_files:
        r_idx = get_round_idx(p.name)
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 1ファイル1行想定だが、念のため全行読む
            for row in reader:
                # history側のメタ情報を結合（n_labeled, budgetなど）
                if r_idx in history_map:
                    hist = history_map[r_idx]
                    # 重複しないキーだけ追加
                    for k, v in hist.items():
                        if k not in row:
                            row[k] = v
                else:
                    # historyが無い場合の補完
                    row["round"] = str(r_idx)
                
                aggregated_rows.append(row)

    if not aggregated_rows:
        return

    # CSV書き出し
    out_path = out_dir / "al_learning_curve.csv"
    fieldnames = list(aggregated_rows[0].keys())
    
    # fieldnamesの並び順を整える（主要項目を前に）
    priority_cols = ["qid", "run_id", "round", "n_labeled", "dev_qwk_full", "dev_rmse_full", "test_qwk_full", "test_rmse_full"]
    sorted_fieldnames = [k for k in priority_cols if k in fieldnames] + [k for k in fieldnames if k not in priority_cols]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"[aggregate] Merged {len(aggregated_rows)} rounds -> {out_path}")

if __name__ == "__main__":
    main()