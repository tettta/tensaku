#!/bin/bash
# /home/esakit25/work/tensaku/scripts/tools/refresh_plots.sh
# 全実験のレポートとプロットを再生成する

set -euo pipefail
ROOT="${ROOT:-/home/esakit25/work/tensaku}"
QID="${QID:-Y14_1-2_1_3}" # デフォルトのQID
BASE_DIR="${ROOT}/outputs/q-${QID}"

REPORT_SCRIPT="${ROOT}/scripts/utils/report_al.sh"

if [[ ! -f "${REPORT_SCRIPT}" ]]; then
    echo "ERROR: Report script not found at ${REPORT_SCRIPT}" >&2
    exit 1
fi

# 1. 各実験フォルダを走査してレポート更新
for exp_dir in "${BASE_DIR}"/exp_*; do
  if [[ -d "${exp_dir}" ]]; then
      echo "--- Refreshing ${exp_dir} ---"
      # 【変更】 report_al.sh を呼び出す
      bash "${REPORT_SCRIPT}" "${exp_dir}" "${QID}"
  fi
done

# 2. 最後に全体比較グラフを更新
PLOT_CURVE="${ROOT}/scripts/utils/plot_al_curves.py"
if [[ -f "${PLOT_CURVE}" ]]; then
    echo "--- Generating overall comparison plots ---"
    python "${PLOT_CURVE}" --qid "${QID}" --root "${ROOT}"
fi