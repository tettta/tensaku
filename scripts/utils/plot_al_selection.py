# /home/esakit25/work/tensaku/scripts/utils/plot_al_selection.py (修正版)

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.figsize': (10, 8)})

def make_selection_plot(df: pd.DataFrame, out_path: str, title: str):
    """
    選択されたデータのスコア分布をヒートマップで描画する
    """
    if df.empty:
        print(f"[plot_sel] WARN: No data for {title}")
        return

    # 1. ビニング
    n_bins = 10
    
    # 信頼度 (conf_msp) でビニング
    df['conf_bin'] = pd.cut(df['conf_msp'], bins=n_bins, labels=[f'{i/n_bins:.1f}-{(i+1)/n_bins:.1f}' for i in range(n_bins)], include_lowest=True)
    
    # 正解スコア (y_true) でビニング
    min_score = df['y_true'].min()
    max_score = df['y_true'].max()
    
    if df['y_true'].dtype in ['int64', 'int32'] and (max_score - min_score) < 15:
        df['label_bin'] = df['y_true'].astype(str)
        y_label = "True Score (y_true)"
    else:
        df['label_bin'] = pd.cut(df['y_true'], bins=n_bins, include_lowest=True)
        y_label = "True Score Bin (y_true)"

    # 2. ピボットテーブルを作成 (クロス集計)
    pivot = df.pivot_table(index="label_bin", columns="conf_bin", aggfunc="size", fill_value=0)

    # 3. 描画
    plt.figure(figsize=(12, 8))
    
    # 💡 修正箇所: fmt="d" を fmt=".0f" に変更してfloatを許容する
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar=False) 
    
    plt.xlabel("Confidence Bin (conf_msp)")
    plt.ylabel(y_label)
    plt.title(f"Selected Sample Distribution: {title}")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plot_sel] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True, help="Question ID")
    parser.add_argument("--root", default=".", help="Project root")
    args = parser.parse_args()

    base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
    plots_dir = os.path.join(base_dir, "summary_plots", "selections")

    search_pattern = os.path.join(base_dir, "exp_*")
    exp_dirs = glob.glob(search_pattern)

    if not exp_dirs:
        print(f"[plot_sel] ERROR: No experiment folders found in {base_dir}")
        return

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        
        selection_files = glob.glob(os.path.join(exp_dir, "al_selected_round*.csv"))
        
        if not selection_files:
            continue
        
        df_list = []
        for fpath in selection_files:
            try:
                df_list.append(pd.read_csv(fpath))
            except Exception as e:
                print(f"[plot_sel] WARN: Skipping {fpath}: {e}")
                continue

        if not df_list:
            continue

        df_all_selections = pd.concat(df_list, ignore_index=True)
        
        out_path = os.path.join(plots_dir, f"{exp_name}_selection_heatmap.png")
        make_selection_plot(df_all_selections, out_path, exp_name)

if __name__ == "__main__":
    main()