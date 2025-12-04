# scripts/utils/plot_al_confmat.py
# -*- coding: utf-8 -*-
"""
@role: 最終ラウンドの混同行列を描画する (単一実験対応版)
"""
import argparse
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys

def process_experiment(exp_dir, exp_name, split, out_dir_base=None):
    detail_path = os.path.join(exp_dir, "preds_detail.csv")
    if not os.path.exists(detail_path):
        print(f"[plot_cm] WARN: {detail_path} not found.")
        return

    try:
        df = pd.read_csv(detail_path)
        
        # 【修正】splitカラムがない場合のガード
        if "split" not in df.columns:
            # split列がない場合は全データを対象とする（またはWARNを出して終了）
            print(f"[plot_cm] WARN: 'split' column not found in {detail_path}. Using all data.")
            df_tgt = df
        else:
            df_tgt = df[df["split"] == split]

        if df_tgt.empty:
            print(f"[plot_cm] WARN: No data for split '{split}' in {exp_name}")
            return
        
        y_true = df_tgt["y_true"].astype(int)
        y_pred = df_tgt["y_pred"].astype(int)
        labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix ({split}) - {exp_name}")
        plt.ylabel("True"); plt.xlabel("Pred")
        
        # 出力先の設定
        if not out_dir_base:
            out_dir_base = os.path.join(exp_dir, "plots")
            
        os.makedirs(out_dir_base, exist_ok=True)
        save_path = os.path.join(out_dir_base, f"cm_{split}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[plot_cm] Saved: {save_path}")

    except Exception as e:
        print(f"[plot_cm] WARN: Failed {exp_name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=False, help="Question ID")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--split", default="test", help="Split to analyze")
    
    # 入力元を exp-dir, 出力先を out-dir に分離
    parser.add_argument("--exp-dir", type=str, default=None, help="Input experiment directory (containing preds_detail.csv)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    if args.exp_dir:
        # 単一モード
        out_d = args.out_dir if args.out_dir else os.path.join(args.exp_dir, "plots")
        # 【修正】末尾スラッシュを除去してから basename を取得
        exp_name = os.path.basename(args.exp_dir.rstrip(os.sep))
        process_experiment(args.exp_dir, exp_name, args.split, out_d)
    elif args.qid:
        # バッチモード
        base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
        exp_dirs = sorted(glob.glob(os.path.join(base_dir, "exp_*")))
        for exp_dir in exp_dirs:
            out_d = os.path.join(exp_dir, "analysis")
            exp_name = os.path.basename(exp_dir.rstrip(os.sep))
            process_experiment(exp_dir, exp_name, args.split, out_d)
    else:
        print("Usage: --exp-dir (single) or --qid (batch)")

if __name__ == "__main__":
    main()