# scripts/utils/plot_al_select.py
# -*- coding: utf-8 -*-
"""
@role: 各ラウンドで選ばれたデータの正解ラベル分布をヒートマップで可視化する
       (単体実験モード対応版)
"""
import argparse
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.figsize': (10, 8)})

def make_selection_plot(df: pd.DataFrame, out_path: str, title: str):
    if df.empty: return

    n_bins = 10
    if 'conf_msp' in df.columns:
        df['conf_bin'] = pd.cut(df['conf_msp'], bins=n_bins, labels=[f'{i/n_bins:.1f}-{(i+1)/n_bins:.1f}' for i in range(n_bins)], include_lowest=True)
    else:
        df['conf_bin'] = "N/A"
    
    min_score = df['y_true'].min()
    max_score = df['y_true'].max()
    
    if df['y_true'].dtype in ['int64', 'int32'] and (max_score - min_score) < 15:
        df['label_bin'] = df['y_true'].astype(str)
        y_label = "True Score"
    else:
        df['label_bin'] = pd.cut(df['y_true'], bins=n_bins, include_lowest=True)
        y_label = "True Score Bin"

    pivot = df.pivot_table(index="label_bin", columns="conf_bin", aggfunc="size", fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar=False)
    plt.xlabel("Confidence Bin (conf_msp)")
    plt.ylabel(y_label)
    plt.title(f"Selected Sample Distribution: {title}")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plot_sel] Saved: {out_path}")

def process_experiment(exp_dir, exp_name, out_dir_base=None):
    # oracle_labels_round*.csv を探す (rounds/ 以下または直下)
    label_files = glob.glob(os.path.join(exp_dir, "rounds", "oracle_labels_round*.csv"))
    label_files += glob.glob(os.path.join(exp_dir, "oracle_labels_round*.csv"))
    
    if not label_files:
        print(f"[plot_sel] WARN: No oracle labels found in {exp_dir}")
        return

    df_list = []
    for fpath in label_files:
        try:
            df = pd.read_csv(fpath) 
            if "score" in df.columns:
                df = df.rename(columns={"score": "y_true"})
            if "y_true" not in df.columns:
                continue
            
            if "conf_msp" not in df.columns:
                df["conf_msp"] = 0.5 
            df_list.append(df)
        except: pass

    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
        
        # 出力先の決定
        if not out_dir_base:
             out_dir_base = os.path.join(exp_dir, "plots")
             
        os.makedirs(out_dir_base, exist_ok=True)
        out_path = os.path.join(out_dir_base, "selection_heatmap.png")
        
        make_selection_plot(df_all, out_path, exp_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=False, help="Question ID")
    parser.add_argument("--root", default=".", help="Project root")
    
    # 入力元を exp-dir, 出力先を out-dir に分離
    parser.add_argument("--exp-dir", type=str, default=None, help="Input experiment directory to process")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    if args.exp_dir:
        # 単一モード
        if os.path.exists(args.exp_dir):
            # 【修正】末尾スラッシュを除去してから basename を取得
            exp_name = os.path.basename(args.exp_dir.rstrip(os.sep))
            out_d = args.out_dir if args.out_dir else os.path.join(args.exp_dir, "plots")
            print(f"[plot_sel] Processing single directory: {args.exp_dir}")
            process_experiment(args.exp_dir, exp_name, out_d)
        else:
            print(f"[plot_sel] ERROR: Directory not found: {args.exp_dir}")
            sys.exit(1)
    elif args.qid:
        # バッチモード
        base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
        exp_dirs = sorted(glob.glob(os.path.join(base_dir, "exp_*")))
        for exp_dir in exp_dirs:
            out_d = os.path.join(exp_dir, "analysis")
            exp_name = os.path.basename(exp_dir.rstrip(os.sep))
            process_experiment(exp_dir, exp_name, out_d)
    else:
        print("Usage: --exp-dir (single) or --qid (batch)")

if __name__ == "__main__":
    main()