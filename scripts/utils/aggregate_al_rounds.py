# scripts/utils/aggregate_al_rounds.py
# -*- coding: utf-8 -*-
"""
@role: 指定ディレクトリ（またはその下の rounds/）内の hitl_summary_round*.csv を集約する
       (n_labeled の強制補完機能付き)
"""
import argparse
import csv
import re
import json
import os
from pathlib import Path
from typing import List, Dict, Any

def get_round_from_fname(filename: str) -> int:
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

    # 1. ファイル収集 (直下 および rounds/ サブディレクトリ)
    # 【修正】roundsフォルダの中も探すように変更
    summary_files = list(out_dir.glob("hitl_summary_round*.csv")) + \
                    list(out_dir.glob("rounds/hitl_summary_round*.csv"))
    
    if not summary_files:
        print(f"[aggregate] WARN: No hitl_summary_round*.csv found in {out_dir} (checked ./ and ./rounds/)")
        return

    summary_files.sort(key=lambda p: get_round_from_fname(p.name))

    # 2. 履歴ファイル (al_history.csv) の読み込み
    history_map = {}
    history_path = out_dir / "al_history.csv"
    
    budget_detected = 50  
    
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r_idx = row.get("round")
                    if r_idx is not None:
                        history_map[int(r_idx)] = row
                        b = int(row.get("budget", 0))
                        if b > 0: budget_detected = b
        except Exception as e:
            print(f"[aggregate] WARN: Failed to read history: {e}")

    initial_size = 50
    if 0 in history_map:
        try:
            n_r0 = int(history_map[0].get("n_labeled", 0))
            if n_r0 > budget_detected:
                initial_size = n_r0 - budget_detected
        except: pass

    aggregated_rows = []

    for p in summary_files:
        r_idx = get_round_from_fname(p.name)
        
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged = row.copy()
                merged["round"] = str(r_idx)
                
                if r_idx in history_map:
                    hist = history_map[r_idx]
                    for k, v in hist.items():
                        if k not in merged:
                            merged[k] = v
                
                if "n_labeled" not in merged or not merged["n_labeled"]:
                    theoretical_n = initial_size + (r_idx * budget_detected)
                    merged["n_labeled"] = str(theoretical_n)

                # 補完ロジック
                if "sampler" not in merged or not merged["sampler"]:
                    dir_name = out_dir.name
                    if "random" in dir_name: merged["sampler"] = "random"; merged["uncertainty_key"] = "msp"
                    elif "trust" in dir_name: merged["sampler"] = "uncertainty"; merged["uncertainty_key"] = "trust"
                    elif "entropy" in dir_name: merged["sampler"] = "uncertainty"; merged["uncertainty_key"] = "entropy"
                    elif "msp" in dir_name: merged["sampler"] = "uncertainty"; merged["uncertainty_key"] = "msp"
                    elif "hybrid" in dir_name: merged["sampler"] = "hybrid"; merged["uncertainty_key"] = "trust"
                    elif "clustering" in dir_name: merged["sampler"] = "clustering"; merged["uncertainty_key"] = "none"
                    merged["budget"] = str(budget_detected)

                aggregated_rows.append(merged)

    if not aggregated_rows:
        return

    out_path = out_dir / "al_learning_curve.csv"
    
    all_keys = set().union(*(d.keys() for d in aggregated_rows))
    priority = ["qid", "run_id", "round", "n_labeled", "test_qwk_full", "test_rmse_full"]
    fieldnames = [k for k in priority if k in all_keys] + [k for k in sorted(all_keys) if k not in priority]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"[aggregate] Merged {len(aggregated_rows)} rounds -> {out_path}")

if __name__ == "__main__":
    main()