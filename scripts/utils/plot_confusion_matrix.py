# scripts/utils/plot_confusion_matrix.py
# -*- coding: utf-8 -*-
"""
@role: 最終ラウンドの混同行列を描画する (個別保存)
"""
import argparse
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True, help="Question ID")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--split", default="test", help="Split to analyze")
    args = parser.parse_args()

    base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
    
    exp_dirs = sorted(glob.glob(os.path.join(base_dir, "exp_*")))
    
    print(f"[plot-cm] Generating matrices for split='{args.split}'...")

    count = 0
    for exp_path in exp_dirs:
        exp_name = os.path.basename(exp_path)
        detail_path = os.path.join(exp_path, "preds_detail.csv")
        
        if not os.path.exists(detail_path): continue

        try:
            df = pd.read_csv(detail_path)
            df_tgt = df[df["split"] == args.split]
            if df_tgt.empty: continue
            
            y_true = df_tgt["y_true"].astype(int)
            y_pred = df_tgt["y_pred"].astype(int)
            labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
            
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", 
                        xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix ({args.split})") # タイトルをシンプルに
            plt.ylabel("True"); plt.xlabel("Pred")
            
            # 【修正】保存先を 'exp_path/analysis/' に変更
            exp_analysis_dir = os.path.join(exp_path, "analysis")
            os.makedirs(exp_analysis_dir, exist_ok=True)
            
            save_path = os.path.join(exp_analysis_dir, f"cm_{args.split}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved CM analysis to: {save_path}")
            count += 1

        except Exception as e:
            print(f"  -> WARN: Failed {exp_name}: {e}")

    print(f"[plot-cm] Done. Generated {count} matrices.")

if __name__ == "__main__":
    main()