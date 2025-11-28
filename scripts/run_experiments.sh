#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/run_experiments.sh
# -*- coding: utf-8 -*-
#
# ==============================================================================
#  TENSAKU EXPERIMENT RUNNER
#  このファイルの [CONFIG] セクションを編集するだけで実験を回せます。
# ==============================================================================

set -euo pipefail

# ==============================================================================
# [CONFIG] 実験設定エリア (ここを変更してください)
# ==============================================================================

# 1. 対象データと基本設定
export QID="Y14_1-2_2_4"           # 問題ID
export AL_ROUNDS="10"              # ALラウンド数
export AL_K="50"                   # 1ラウンドあたりの追加件数

# 2. 学習パラメータ (軽量化やチューニング)
#    例: "--set train.batch_size=16 --set train.epochs=3"
export CFG_OVERRIDE="--set train.batch_size=16 --set train.epochs=5"

# 3. 実行したい実験リスト
#    形式: "実験名  指標(AL_BY)  戦略(AL_SAMPLER)"
#    ※コメントアウト(#)することで実行対象から外せます
EXPERIMENTS=(
    # --- Baseline ---
    "exp_al_random       msp       random"

    # --- Uncertainty Sampling ---
    "exp_al_msp          msp       uncertainty"
    "exp_al_entropy      entropy   uncertainty"
    "exp_al_trust        trust     uncertainty"

    # --- Diversity / Hybrid ---
    # "exp_al_clustering   msp       clustering"
    "exp_al_hybrid       trust     hybrid"
)

# 4. 実行オプション
#    0: 新規実行 (フォルダ削除して開始)
#    1: 再開モード (既存の結果を維持し、途中から再開)
AL_RESUME=0

# ==============================================================================
# [SYSTEM] 以下はロジック部分です (通常は変更不要)
# ==============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# パス定義
OUT_BASE="${ROOT}/outputs/q-${QID}"
DATA_BASE="${ROOT}/data_sas/q-${QID}"
INPUT_ALL="${ROOT}/data_sas/all.jsonl"
if [[ ! -f "${INPUT_ALL}" ]]; then INPUT_ALL="${ROOT}/data_sas/q-${QID}/all.jsonl"; fi

echo "============================================================"
echo " STARTING EXPERIMENTS"
echo " QID: ${QID}, Rounds: ${AL_ROUNDS}, Budget: ${AL_K}"
echo " Override: ${CFG_OVERRIDE}"
echo " Resume Mode: ${AL_RESUME}"
echo "============================================================"

# 1. Base Split (共通初期データの作成)
BASE_SPLIT_DIR="${DATA_BASE}/base_split"

if [[ "${AL_RESUME}" == "0" || ! -d "${BASE_SPLIT_DIR}" ]]; then
    echo "[runner] Creating/Resetting Base Split..."
    if [[ -d "${BASE_SPLIT_DIR}" ]]; then rm -rf "${BASE_SPLIT_DIR}"; fi
    mkdir -p "${BASE_SPLIT_DIR}"
    
    # 最初の実験の設定を使ってSplitする
    FIRST_CFG="${ROOT}/configs/exp_al_hitl.yaml"
    
    tensaku split -c "${FIRST_CFG}" \
      --set "data.qid=${QID}" \
      --set "run.data_dir=${BASE_SPLIT_DIR}" \
      --set "data.input_all=${INPUT_ALL}"
else
    echo "[runner] Using existing Base Split (Resume Mode)."
fi

# 2. Experiment Loop
for EXP_LINE in "${EXPERIMENTS[@]}"; do
    # コメントアウトされた行などはスキップ
    if [[ "${EXP_LINE}" =~ ^#.* ]] || [[ -z "${EXP_LINE}" ]]; then continue; fi

    read -r NAME BY SAMPLER <<< "${EXP_LINE}"
    
    # 空白除去
    NAME=$(echo "${NAME}" | xargs)
    BY=$(echo "${BY}" | xargs)
    SAMPLER=$(echo "${SAMPLER}" | xargs)

    OUT_DIR="${OUT_BASE}/${NAME}"
    DATA_DIR="${DATA_BASE}/${NAME}"
    CFG="${ROOT}/configs/exp_al_hitl.yaml"

    echo "------------------------------------------------------------"
    echo "[runner] >>> Experiment: ${NAME}"
    echo "         >>> Strategy: ${SAMPLER} (by ${BY})"
    echo "------------------------------------------------------------"

    # ディレクトリ準備 (Resumeでなければリセット)
    if [[ "${AL_RESUME}" == "1" && -d "${OUT_DIR}" ]]; then
        echo "[runner] Resuming..."
    else
        if [[ -d "${OUT_DIR}" ]]; then rm -rf "${OUT_DIR}"; fi
        if [[ -d "${DATA_DIR}" ]]; then rm -rf "${DATA_DIR}"; fi
        mkdir -p "${DATA_DIR}" "${OUT_DIR}"

        # データのコピー & リンク
        cp "${BASE_SPLIT_DIR}/labeled.jsonl" "${DATA_DIR}/"
        cp "${BASE_SPLIT_DIR}/pool.jsonl"    "${DATA_DIR}/"
        ln -s "${BASE_SPLIT_DIR}/dev.jsonl"  "${DATA_DIR}/"
        ln -s "${BASE_SPLIT_DIR}/test.jsonl" "${DATA_DIR}/"
        ln -s "${BASE_SPLIT_DIR}/meta.json"  "${DATA_DIR}/"
    fi

    # AL実行 (core/al.sh 呼び出し)
    # 環境変数を渡して実行
    QID="${QID}" CFG="${CFG}" \
    DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" \
    AL_ROUNDS="${AL_ROUNDS}" AL_K="${AL_K}" \
    AL_BY="${BY}" AL_SAMPLER="${SAMPLER}" \
    AL_CLEAN_CKPT_AFTER=1 \
    AL_RESUME="${AL_RESUME}" \
    CFG_OVERRIDE="${CFG_OVERRIDE}" \
    bash "${ROOT}/scripts/core/al.sh" || {
        echo "‼️ [runner] ERROR: Experiment ${NAME} failed. Skipping to next." >&2
        continue
    }

    echo "[runner] FINISHED: ${NAME}"
    echo
done

# 3. Final Plotting
echo "============================================================"
echo " ALL DONE. Generating Summary Plots..."
echo "============================================================"

PLOT_SCRIPT="${ROOT}/scripts/utils/plot_al_curves.py"
if [[ -f "${PLOT_SCRIPT}" ]]; then
    python "${PLOT_SCRIPT}" --qid "${QID}" --root "${ROOT}"
    echo "[runner] Plots saved to ${OUT_BASE}/summary_plots/"
else
    echo "[runner] WARN: Plot script not found at ${PLOT_SCRIPT}"
fi
