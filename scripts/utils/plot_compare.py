# scripts/utils/plot_al_compare.py
# -*- coding: utf-8 -*-
"""
@role: 実験結果を集計・比較描画する（メタデータ活用・集約機能・フォルダ自動整理付き）
@metrics: QWK, RMSE, CSE, Accuracy, Coverage, Macro F1, AURC
"""
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# スタイル設定
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True, help="Question ID")
    parser.add_argument("--root", default=".", help="Project root")
    
    # フィルタリング・グルーピング
    parser.add_argument("--query", type=str, default=None, 
                        help="Pandas query string (e.g. 'budget == 50')")
    parser.add_argument("--filter", type=str, default=None, help="Simple string filter for experiment name")
    parser.add_argument("--exclude", type=str, default=None, help="Simple string exclude for experiment name")
    
    # 集約オプション
    parser.add_argument("--aggregate", action="store_true", 
                        help="Aggregate experiments with same group keys (mean + std)")
    parser.add_argument("--group-by", nargs="+", default=["sampler", "uncertainty_key"],
                        help="Columns to group by (default: sampler uncertainty_key)")
    
    # 表示オプション
    parser.add_argument("--label-format", type=str, default=None,
                        help="Format string for legend")
    parser.add_argument("--title-suffix", type=str, default="", help="Suffix for title/filename")
    
    args = parser.parse_args()

    base_dir = os.path.join(args.root, "outputs", f"q-{args.qid}")
    
    # 保存先ディレクトリの決定 (summary_plots/{suffix})
    # suffixがあればサブフォルダを掘り、なければ "all" フォルダへ
    summary_plots_dir = os.path.join(base_dir, "summary_plots")
    
    if args.title_suffix:
        folder_name = args.title_suffix.strip("_")
        if not folder_name: folder_name = "custom"
    else:
        folder_name = "all"
    
    out_dir = os.path.join(summary_plots_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. データ読み込み
    search_pattern = os.path.join(base_dir, "*", "al_learning_curve.csv")
    files = glob.glob(search_pattern)

    if not files:
        print(f"[plot] ERROR: No csv found in {base_dir}")
        return

    df_list = []
    for fpath in files:
        exp_name = os.path.basename(os.path.dirname(fpath))
        
        # 簡易フィルタ（文字列マッチ）
        if args.filter and (args.filter not in exp_name): continue
        if args.exclude and (args.exclude in exp_name): continue

        try:
            df = pd.read_csv(fpath)
            df["exp_name"] = exp_name
            
            # 数値変換 (エラー回避)
            cols_to_numeric = [
                "n_labeled", "budget", "round", 
                "test_qwk_full", "test_rmse_full", "test_cse_full", 
                "test_acc_full", "test_f1_full", "test_coverage_gate", "aurc_cse"
            ]
            for c in cols_to_numeric:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            
            df_list.append(df)
        except Exception as e:
            print(f"[plot] WARN: Skipping {fpath}: {e}")

    if not df_list:
        print("[plot] No data matches the filter.")
        return

    df_all = pd.concat(df_list, ignore_index=True)

    # 2. クエリによるフィルタリング
    if args.query:
        try:
            print(f"[plot] Filtering by query: {args.query}")
            df_all = df_all.query(args.query).copy()
        except Exception as e:
            print(f"[plot] ERROR: Query failed: {e}")
            return

    if df_all.empty:
        print("[plot] No data matches the query.")
        return

    # 3. 表示用ラベル生成
    def make_label(row):
        if args.label_format:
            try:
                return args.label_format.format(**row.to_dict())
            except KeyError:
                return row["exp_name"]
        else:
            if args.aggregate:
                keys = [str(row.get(k, "?")) for k in args.group_by]
                return "-".join(keys)
            else:
                return row["exp_name"]

    df_all["plot_label"] = df_all.apply(make_label, axis=1)

    # 4. コンソールへのサマリ表示
    print(f"{'Label':<30} | {'Count':<5} | {'QWK':<6} | {'CSE':<6} | {'Cov':<6} | {'AURC':<6}")
    print("-" * 80)
    
    # 最新ラウンドの平均値を表示
    last_round_idx = df_all["round"].max()
    df_last = df_all[df_all["round"] == last_round_idx]
    
    # グループごとに平均
    if not df_last.empty:
        grouped = df_last.groupby("plot_label")[["test_qwk_full", "test_cse_full", "test_coverage_gate", "aurc_cse"]].mean()
        counts = df_last.groupby("plot_label").size()
        
        for label in grouped.index:
            row = grouped.loc[label]
            cnt = counts.loc[label]
            print(f"{label:<30} | {cnt:<5} | {row['test_qwk_full']:.4f} | {row['test_cse_full']:.4f} | {row['test_coverage_gate']:.4f} | {row['aurc_cse']:.4f}")

    # 5. プロット関数
    # マーカーリスト
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

    def plot_metric(metric_col, ylabel, title, fname_base):
        plt.figure(figsize=(10, 6))
        
        df_plot = df_all.dropna(subset=[metric_col])
        if df_plot.empty:
            plt.close(); return

        unique_labels = sorted(df_plot["plot_label"].unique())

        if args.aggregate:
            # 集約モード (Seabornの機能で平均+帯を表示)
            sns.lineplot(
                data=df_plot, x="n_labeled", y=metric_col,
                hue="plot_label", style="plot_label",
                markers=True, dashes=False, errorbar='sd', alpha=0.8
            )
            title_prefix = "Aggregated AL Curve"
        else:
            # 個別モード (重なり防止のためJitterを手動適用)
            # SeabornのlineplotだとJitterが難しいので、ループで描画
            for i, label in enumerate(unique_labels):
                sub = df_plot[df_plot["plot_label"] == label].sort_values("n_labeled")
                
                # Jitter: X軸を少しずらす (-1% ~ +1%程度)
                jitter_amount = (i - len(unique_labels)/2) * (sub["n_labeled"].max() * 0.003)
                x = sub["n_labeled"] + jitter_amount
                
                plt.plot(
                    x, sub[metric_col],
                    label=label,
                    marker=markers[i % len(markers)],
                    alpha=0.7, linewidth=2, markersize=8
                )
            
            title_prefix = "AL Curve"

        plt.xlabel("Number of Labeled Samples")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} ({title}) - {args.qid} {args.title_suffix}")
        
        # 凡例を枠外へ
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        # ファイル名
        fname = f"{fname_base}{args.title_suffix}.png"
        save_path = os.path.join(out_dir, fname)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

    # 6. 実行 (全指標)
    print(f"\n[plot] Generating plots to {out_dir}...")
    
    metrics = [
        ("test_qwk_full", "QWK", "QWK", "curve_compare_qwk"),
        ("test_rmse_full", "RMSE", "RMSE", "curve_compare_rmse"),
        ("test_cse_full", "CSE", "CSE", "curve_compare_cse"),
        ("test_acc_full", "Accuracy", "Accuracy", "curve_compare_acc"),
        ("test_f1_full", "Macro F1", "Macro F1", "curve_compare_f1"),
        ("test_coverage_gate", "Coverage", "Coverage", "curve_compare_coverage"),
        ("aurc_cse", "AURC", "AURC(CSE)", "curve_compare_aurc"),
    ]

    for col, ylab, tit, fn in metrics:
        if col in df_all.columns:
            plot_metric(col, ylab, tit, fn)

if __name__ == "__main__":
    main()