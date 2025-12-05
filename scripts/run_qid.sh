#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/run_qid.sh
# -*- coding: utf-8 -*-

set -euo pipefail

ROOT="${ROOT:-/home/esakit25/work/tensaku}"
QID="${QID:-}"

EXPERIMENTS_FILE="${1:-${ROOT}/scripts/lists/experiments.txt}"
EXP_LIST_NAME=$(basename "${EXPERIMENTS_FILE}" .txt)

ROUNDS="${ROUNDS:-}"
BUDGET="${BUDGET:-}"
START_SIZE="${START_SIZE:-}"

cd "${ROOT}"

echo "[run_qid] QID        = ${QID:-'(Auto/YAML)'}"
echo "[run_qid] ROUNDS     = ${ROUNDS:-'(Auto/YAML)'}"
echo "[run_qid] BUDGET     = ${BUDGET:-'(Auto/YAML)'}"
echo "[run_qid] START_SIZE = ${START_SIZE:-'(Auto/YAML)'}"
echo "[run_qid] LIST       = ${EXPERIMENTS_FILE}"
echo "------------------------------------------------------------"

if [[ -z "${QID}" ]]; then
  echo "ERROR: QID is not set."
  exit 1
fi

# master all.jsonl の候補
INPUT_ALL="${ROOT}/data_sas/all.jsonl"
if [[ ! -f "${INPUT_ALL}" ]]; then INPUT_ALL="${ROOT}/data_sas/q-${QID}/all.jsonl"; fi

OUT_BASE="${ROOT}/outputs/q-${QID}"
DATA_BASE="${ROOT}/data_sas/q-${QID}"

# 1. Base Split
FIRST_EXP_LINE=$(grep -v '^#' "${EXPERIMENTS_FILE}" | grep -v '^\s*$' | head -n 1)
FIRST_EXP_LINE=$(echo "${FIRST_EXP_LINE}" | tr -d '\r')
FIRST_CFG="${ROOT}/configs/exp_al_hitl.yaml"

BASE_SPLIT_DIR="${DATA_BASE}/base_split"
if [[ -d "${BASE_SPLIT_DIR}" ]]; then rm -rf "${BASE_SPLIT_DIR}"; fi
mkdir -p "${BASE_SPLIT_DIR}"

echo "------------------------------------------------------------"
echo "[run_qid] Creating Base Split (n_train=${START_SIZE:-'(Auto/YAML)'})..."
echo "------------------------------------------------------------"

SPLIT_ARGS=(
  split -c "${FIRST_CFG}"
  --set "data.qid=${QID}"
  --set "run.data_dir=${BASE_SPLIT_DIR}"
  --set "data.input_all=${INPUT_ALL}"
)

if [[ -n "${START_SIZE}" ]]; then
  SPLIT_ARGS+=(--set "split.n_train=${START_SIZE}")
fi

tensaku "${SPLIT_ARGS[@]}"


CFG_OVERRIDE_GLOBAL="${CFG_OVERRIDE:-}"
if [[ -n "${CFG_OVERRIDE_GLOBAL}" ]]; then
  echo "[run_qid] ⚠️  Global Override Active: ${CFG_OVERRIDE_GLOBAL}"
fi

# 2. Experiment Loop
while read -r line; do
  line=$(echo "${line}" | tr -d '\r')
  line="${line%%#*}"
  if [[ -z "${line}" || "${line}" =~ ^[[:space:]]*$ ]]; then continue; fi

  read -r NAME BY SAMPLER EXTRA <<< "${line} _NONE_"
  NAME=$(echo "${NAME}" | xargs)
  BY=$(echo "${BY}" | xargs)
  SAMPLER=$(echo "${SAMPLER}" | xargs)
  if [[ "${EXTRA}" == "_NONE_" ]]; then EXTRA=""; else EXTRA=$(echo "${EXTRA}" | xargs); fi

  CFG="${ROOT}/configs/exp_al_hitl.yaml"
  OUT_DIR="${OUT_BASE}/${NAME}"
  DATA_DIR="${DATA_BASE}/${NAME}"
  
  echo "------------------------------------------------------------"
  echo "[run_qid] START: ${NAME}"
  echo "  > BY: ${BY}, SAMPLER: ${SAMPLER}"
  echo "------------------------------------------------------------"

  if [[ "${AL_RESUME:-0}" == "1" && -d "${OUT_DIR}" ]]; then
      echo "[run_qid] 🔄 RESUME MODE: Skipping cleanup and data init."
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

  ALL_OVERRIDES="${CFG_OVERRIDE_GLOBAL} ${EXTRA}"

  # (A) 実験実行
  QID="${QID}" CFG="${CFG}" DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" \
  ROUNDS="${ROUNDS}" BUDGET="${BUDGET}" \
  AL_BY="${BY}" AL_SAMPLER="${SAMPLER}" \
  AL_CLEAN_CKPT_AFTER=1 \
  AL_RESUME="${AL_RESUME:-0}" \
  START_SIZE="${START_SIZE}" \
  CFG_OVERRIDE="${ALL_OVERRIDES}" \
  bash "${ROOT}/scripts/core/run_al.sh" || {
      echo "‼️ [run_qid] WARNING: Experiment ${NAME} failed during AL loop." >&2
  }

  echo "[run_qid] FINISHED: ${NAME}"
  echo

done < "${EXPERIMENTS_FILE}"

# 3. Final Plotting (全体比較)
echo ">>> ALL DONE. Plotting curves..."
PLOT_COMP="${ROOT}/scripts/utils/plot_compare.py"

if [[ -f "${PLOT_COMP}" ]]; then
  python "${PLOT_COMP}" \
    --qid "${QID}" \
    --root "${ROOT}" \
    --title-suffix "_${EXP_LIST_NAME}"
    
  echo "[run_qid] Plots saved to summary_plots/${EXP_LIST_NAME}/"
else
  echo "Plot script not found: ${PLOT_COMP}"
fi

echo ">>> QID BATCH RUN COMPLETE <<<"
