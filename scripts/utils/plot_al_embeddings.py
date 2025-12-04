# scripts/utils/plot_al_embeddings.py
# -*- coding: utf-8 -*-
"""
@role: 各ラウンドで選ばれたデータをt-SNEで2次元に圧縮して可視化する
"""
import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# t-SNEは重いので、UMAPがあればそちらを推奨（今回はsklearn標準のt-SNE）
from sklearn.manifold import TSNE

def get_round_from_fname(fname):
    m = re.search(r"round(\d+)", fname)
    return int(m.group(1)) if m else -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True, help="Question ID")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--round", type=int, default=0, help="Round to visualize")
    args = parser.parse_args()

    base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
    
    # 実験フォルダを走査
    exp_dirs = glob.glob(os.path.join(base_dir, "exp_*"))
    if not exp_dirs:
        print("No experiments found.")
        return

    print(f"Generating t-SNE plots for Round {args.round}...")

    for exp_path in exp_dirs:
        exp_name = os.path.basename(exp_path)
        
        # 1. 埋め込みベクトルのロード (pool_embs.npy)
        # ※ infer-pool で保存されたもの。無い場合は再推論が必要だが今回はある前提
        embs_path = os.path.join(exp_path, "pool_embs.npy")
        if not os.path.exists(embs_path):
            print(f"Skipping {exp_name}: pool_embs.npy not found.")
            continue
        
        # 2. 選ばれたIDのロード
        # oracle_labels_roundX.csv からIDを取得
        oracle_path = os.path.join(exp_path, f"oracle_labels_round{args.round}.csv")
        if not os.path.exists(oracle_path):
            print(f"Skipping {exp_name}: Round {args.round} labels not found.")
            continue
            
        # 3. IDの紐付け用データ (pool_preds.csv)
        # pool_preds.csv の行順序と pool_embs.npy の行順序は一致している前提
        # ただし、ラウンドが進むと pool が減っていくので、
        # 「そのラウンド開始時点のPool」の埋め込みが必要。
        # ALの仕様上、pool_embs.npy は "最新の推論結果" で上書きされている可能性が高い。
        # 厳密にやるなら "all_embs.npy" を最初に作っておくのがベストだが、
        # ここでは簡易的に "現在残っているPool + 選ばれたデータ" で可視化を試みる。
        
        # 簡易実装: 全データの埋め込み(all.jsonl由来)があればベストだが、
        # ここでは「pool_embs.npy」を使って、その分布を見る。
        
        pool_embs = np.load(embs_path)
        
        # t-SNE実行 (データが多いと時間がかかります)
        # 毎回やると重いので、5000件程度にサンプリングしてもよい
        print(f"  Running t-SNE for {exp_name} (N={len(pool_embs)})...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        embs_2d = tsne.fit_transform(pool_embs)
        
        df_plot = pd.DataFrame(embs_2d, columns=["dim1", "dim2"])
        
        # 選ばれたデータをハイライトしたいが、indexの対応付けが難しい。
        # 今回は「Pool全体の形状」と「Conf値のグラデーション」を見る図にする
        
        # preds_detail.csv から conf を取得して色付け
        preds_path = os.path.join(exp_path, "preds_detail.csv")
        if os.path.exists(preds_path):
            df_preds = pd.read_csv(preds_path)
            df_pool_preds = df_preds[df_preds["split"] == "pool"].reset_index(drop=True)
            
            if len(df_pool_preds) == len(df_plot):
                if "conf_trust" in df_pool_preds.columns:
                    df_plot["confidence"] = df_pool_preds["conf_trust"]
                else:
                    df_plot["confidence"] = df_pool_preds["conf_msp"]

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_plot, x="dim1", y="dim2", hue="confidence", palette="viridis_r", s=10, alpha=0.6)
        plt.title(f"Pool Embeddings & Uncertainty (Darker=Uncertain) - {exp_name}")
        
        save_dir = os.path.join(exp_path, "plots")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "tsne_pool_uncertainty.png"))
        plt.close()
        print(f"  -> Saved t-SNE plot.")

if __name__ == "__main__":
    main()