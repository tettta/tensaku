#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/run_al_generic.sh
# -*- coding: utf-8 -*-
#
# Run multiple AL experiments defined in experiments.txt

set -euo pipefail
# IFS=$'\n\t' <-- スペース区切りのため無効化

ROOT="${ROOT:-/home/esakit25/work/tensaku}"
QID="${QID:-Y14_1-2_1_3}"

# 実験リストのパス
EXPERIMENTS_FILE="${1:-${ROOT}/scripts/lists/experiments.txt}"

# AL 共通設定
AL_ROUNDS="${AL_ROUNDS:-10}"
AL_K="${AL_K:-50}"

cd "${ROOT}"

echo "[run_al_generic] QID    = ${QID}"
echo "[run_al_generic] ROUNDS = ${AL_ROUNDS}"
echo "[run_al_generic] K      = ${AL_K}"
echo "[run_al_generic] LIST   = ${EXPERIMENTS_FILE}"

# マスタデータの場所
INPUT_ALL="${ROOT}/data_sas/all.jsonl"
if [[ ! -f "${INPUT_ALL}" ]]; then INPUT_ALL="${ROOT}/data_sas/q-${QID}/all.jsonl"; fi

# ベース出力先
OUT_BASE="${ROOT}/outputs/q-${QID}"
DATA_BASE="${ROOT}/data_sas/q-${QID}"

# 1. Base Split (共通データ作成)
# 最初の有効な行を取得
FIRST_EXP_LINE=$(grep -v '^#' "${EXPERIMENTS_FILE}" | grep -v '^\s*$' | head -n 1)
FIRST_EXP_LINE=$(echo "${FIRST_EXP_LINE}" | tr -d '\r')
FIRST_CFG="${ROOT}/configs/exp_al_hitl.yaml"

BASE_SPLIT_DIR="${DATA_BASE}/base_split"
if [[ -d "${BASE_SPLIT_DIR}" ]]; then rm -rf "${BASE_SPLIT_DIR}"; fi
mkdir -p "${BASE_SPLIT_DIR}"

echo "------------------------------------------------------------"
echo "[run_al_generic] Creating Base Split..."
echo "------------------------------------------------------------"
tensaku split -c "${FIRST_CFG}" \
  --set "data.qid=${QID}" --set "run.data_dir=${BASE_SPLIT_DIR}" --set "data.input_all=${INPUT_ALL}"

# =============================================================================
# 【修正】ユーザー指定の環境変数（ドライラン設定など）を保存
# =============================================================================
CFG_OVERRIDE_GLOBAL="${CFG_OVERRIDE:-}"
if [[ -n "${CFG_OVERRIDE_GLOBAL}" ]]; then
  echo "[run_al_generic] ⚠️  Global Override Active: ${CFG_OVERRIDE_GLOBAL}"
fi

# 2. Experiment Loop
while read -r line; do
  line=$(echo "${line}" | tr -d '\r')
  [[ "${line}" =~ ^#.*$ ]] && continue
  [[ -z "${line}" ]] && continue

  # 4つ目の要素（EXTRA_ARGS）まで読み込む
  read -r NAME BY SAMPLER EXTRA <<< "${line} _NONE_"

  NAME=$(echo "${NAME}" | xargs)
  BY=$(echo "${BY}" | xargs)
  SAMPLER=$(echo "${SAMPLER}" | xargs)
  
  if [[ "${EXTRA}" == "_NONE_" ]]; then EXTRA=""; else EXTRA=$(echo "${EXTRA}" | xargs); fi

  CFG="${ROOT}/configs/exp_al_hitl.yaml"
  OUT_DIR="${OUT_BASE}/${NAME}"
  DATA_DIR="${DATA_BASE}/${NAME}"
  
  echo "------------------------------------------------------------"
  echo "[run_al_generic] START: ${NAME}"
  echo "  > BY: ${BY}, SAMPLER: ${SAMPLER}"
  echo "  > EXTRA: ${EXTRA}"
  echo "------------------------------------------------------------"

  # Resume Check
  if [[ "${AL_RESUME:-0}" == "1" && -d "${OUT_DIR}" ]]; then
      echo "[run_al_generic] 🔄 RESUME MODE: Skipping cleanup and data init."
  else
      if [[ -d "${OUT_DIR}" ]]; then rm -rf "${OUT_DIR}"; fi
      if [[ -d "${DATA_DIR}" ]]; then rm -rf "${DATA_DIR}"; fi
      mkdir -p "${DATA_DIR}" "${OUT_DIR}"

      cp "${BASE_SPLIT_DIR}/labeled.jsonl" "${DATA_DIR}/"
      cp "${BASE_SPLIT_DIR}/pool.jsonl"    "${DATA_DIR}/"
      ln -s "${BASE_SPLIT_DIR}/dev.jsonl"  "${DATA_DIR}/"
      ln -s "${BASE_SPLIT_DIR}/test.jsonl" "${DATA_DIR}/"
      ln -s "${BASE_SPLIT_DIR}/meta.json"  "${DATA_DIR}/"
  fi

  # ===========================================================================
  # 【修正】グローバル設定と個別設定を結合
  # ===========================================================================
  ALL_OVERRIDES="${CFG_OVERRIDE_GLOBAL} ${EXTRA}"

  # Run al.sh
  QID="${QID}" CFG="${CFG}" DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" \
  AL_ROUNDS="${AL_ROUNDS}" AL_K="${AL_K}" \
  AL_BY="${BY}" AL_SAMPLER="${SAMPLER}" \
  AL_CLEAN_CKPT_AFTER=1 \
  AL_RESUME="${AL_RESUME:-0}" \
  CFG_OVERRIDE="${ALL_OVERRIDES}" \
  bash "${ROOT}/scripts/core/al.sh" || {
      echo "‼️ [run_al_generic] WARNING: Experiment ${NAME} failed. Skipping..." >&2
      continue
  }

  echo "[run_al_generic] FINISHED: ${NAME}"
  echo

done < "${EXPERIMENTS_FILE}"

# 3. Final Plotting
echo ">>> ALL DONE. Plotting curves..."
if [[ -f "${ROOT}/scripts/utils/plot_al_curves.py" ]]; then
  python "${ROOT}/scripts/utils/plot_al_curves.py" --qid "${QID}" --root "${ROOT}"
else
  echo "Plot script not found."
fi