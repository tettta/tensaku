#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/run_batch_qids.sh
# -*- coding: utf-8 -*-
#
# Master script to run AL experiments for a list of QIDs.
# Usage:
#   bash scripts/run_batch_qids.sh <LIST_FILE> [QID1] [QID2] ...

# 実行時の環境変数例:
# env ROUNDS=1 BUDGET=50 START_SIZE=50 bash scripts/run_batch.sh ...

set -euo pipefail
IFS=$'\n\t' # 改行とタブのみを区切り文字とする (安全性の確保)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. 第1引数: 実験リストファイルを取得
EXPERIMENTS_FILE="${1:-${ROOT}/scripts/lists/experiments.txt}"

# 2. 引数を1つずらす (EXPERIMENTS_FILE を捨てる)
if [[ $# -ge 1 ]]; then
    shift
fi

# ==============================================================================
# [CONFIG] 実行対象のQIDリスト（定義）
# ==============================================================================

# コマンドライン引数の残り($@)があればそれを採用、なければデフォルトリストを使用
if [[ $# -gt 0 ]]; then
    echo "--- ⚠️ QIDリストがコマンドライン引数で上書きされました ---"
    # 引数配列をそのままコピー (スペース区切りがそのまま配列になる)
    QIDS=("$@")
else
    # デフォルトQIDリスト (実行したいQIDのみを記述)
    QIDS=(

        "Y14_1-2_1_3" 
        #"Y14_1-2_1_3.sub" 
        "Y14_1-2_2_4" 
        #"Y14_1-2_2_4.sub" 
        "Y14_2-1_1_5" 
        #"Y14_2-1_1_5.sub" 
        "Y14_2-1_2_3" 
        #"Y14_2-1_2_3.sub" 
        "Y14_2-2_1_4" 
        #"Y14_2-2_1_4.sub" 
        "Y14_2-2_2_3" 
        "Y14_2-2_2_3.sub" 
        "Y15_1-1_1_4.0" 
        "Y15_1-1_1_4.1" 
        "Y15_1-1_1_6.0" 
        "Y15_1-1_1_6.1" 
        "Y15_1-3_1_2.0" 
        "Y15_1-3_1_2.1" 
        "Y15_1-3_1_5.0" 
        "Y15_1-3_1_5.1" 
        "Y15_2-2_1_3.0"
        "Y15_2-2_1_3.1" 
        "Y15_2-2_1_5.0" 
        "Y15_2-2_1_5.1" 
        "Y15_2-2_2_4.0" 
        "Y15_2-2_2_4.1" 
        "Y15_2-2_2_5.0" 
        "Y15_2-2_2_5.1" 
        "Y15_2-3_1_5" 
        "Y15_2-3_2_2" 
        "Y15_2-3_2_4" 
    )
fi

# ==============================================================================
# [MAIN] QIDリストの実行
# ==============================================================================

if [[ ${#QIDS[@]} -eq 0 ]]; then
    echo "ERROR: QIDS list is empty. Please define QIDS in the script or pass them as arguments."
    exit 1
fi

echo "Starting batch run for ${#QIDS[@]} QIDs."
echo "Using experiment list: ${EXPERIMENTS_FILE}"
echo "QIDs to run:"
printf " - %s\n" "${QIDS[@]}" # 配列を一つずつ改行して表示
echo "----------------------------------------------------"

for QID in "${QIDS[@]}"; do
    echo ""
    echo ">>> PROCESSING QID: ${QID} <<<"
    
    LOG_DIR="${ROOT}/outputs/q-${QID}"
    mkdir -p "${LOG_DIR}"
    QID_LOG="${LOG_DIR}/run_batch.log"
    
    echo "    Log: ${QID_LOG}"

    # tee を使って「ファイル保存」と「画面表示」を同時に行う
    # 実行: nohup bash ... > master.log & で実行
    # 監視: tail -f master.log

    QID="${QID}" bash "${ROOT}/scripts/run_qid.sh" "${EXPERIMENTS_FILE}" 2>&1 | tee -a "${QID_LOG}" || {
        echo "‼️ WARNING: run_qid failed for QID ${QID}. Check log for details." >&2
        continue
    }

    echo "--- QID ${QID} finished. ---"
done

echo ""
echo ">>> BATCH RUN COMPLETE <<<"