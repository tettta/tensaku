#!/bin/bash
# scripts/refresh_plots.sh
#
# 既存の実験結果から CSE/Acc 等を含めた集計をやり直すスクリプト
# (ディレクトリ構成変更 core/utils に対応版)

set -euo pipefail

ROOT="${ROOT:-/home/esakit25/work/tensaku}"
QID="${QID:-Y14_1-2_1_3}"
BASE_DIR="${ROOT}/outputs/q-${QID}"
CFG="${ROOT}/configs/exp_al_hitl.yaml"

# 集計・描画スクリプトのパス (utils/ に移動済み)
AGG_SCRIPT="${ROOT}/scripts/utils/aggregate_al_rounds.py"
PLOT_SCRIPT="${ROOT}/scripts/utils/plot_al_curves.py"

echo "Refreshing summaries for QID=${QID}..."

# 実験フォルダを走査
for exp_dir in "${BASE_DIR}"/exp_*; do
  if [[ ! -d "${exp_dir}" ]]; then continue; fi
  EXP_NAME=$(basename "${exp_dir}")
  echo ">>> Processing ${EXP_NAME}..."

  # 1. 最新の hitl-report を実行して指標 (CSE, Acc等) を計算し直す
  #    (preds_detail.csv が残っている最終ラウンド分のみ有効)
  if [[ -f "${exp_dir}/preds_detail.csv" ]]; then
      # hitl_summary.csv を生成
      tensaku hitl-report -c "${CFG}" --set "run.out_dir=${exp_dir}" --set "run.qid=${QID}" > /dev/null
      
      # 生成された hitl_summary.csv を、最終ラウンドのファイルとしてリネーム保存する処理が必要だが、
      # 既存の hitl_summary_roundX.csv とのマージは複雑なため、
      # ここでは「al_learning_curve.csv の再生成」に主眼を置く。
      # (注意: 過去ラウンドの preds_detail.csv は消えているため、過去分のCSE等は復元できません)
  fi
  
  # 2. 集計しなおし (al_learning_curve.csv を更新)
  if [[ -f "${AGG_SCRIPT}" ]]; then
      python "${AGG_SCRIPT}" --dir "${exp_dir}"
  else
      echo "ERROR: ${AGG_SCRIPT} not found."
      exit 1
  fi
done

# 3. グラフ再描画
echo ">>> Plotting curves..."
if [[ -f "${PLOT_SCRIPT}" ]]; then
    python "${PLOT_SCRIPT}" --qid "${QID}" --root "${ROOT}"
else
    echo "ERROR: ${PLOT_SCRIPT} not found."
    exit 1
fi

echo "Done."