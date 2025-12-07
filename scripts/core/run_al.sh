#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/core/run_al.sh
# -*- coding: utf-8 -*-
# Active Learning Loop (Self-contained)

set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${CFG:-${ROOT}/configs/exp_al_hitl.yaml}"

# --- 設定読み込み (Python連携) ---
PY_OUT="$(python - <<'PY' "${CFG}"
import sys, yaml, os, typing as t

def get_str(cfg: t.Mapping, path: t.Sequence[str]) -> str:
    cur: t.Any = cfg
    for key in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
        if cur is None:
            return ""
    try:
        return str(cur).strip()
    except Exception:
        return str(cur)

try:
    with open(sys.argv[1], "r") as f:
        c = yaml.safe_load(f) or {}
except Exception:
    c = {}

print(get_str(c, ("run", "out_dir")))
print(get_str(c, ("run", "data_dir")))
print(get_str(c, ("data", "qid")))
print(get_str(c, ("al", "rounds")))
print(get_str(c, ("al", "budget")))
print(get_str(c, ("al", "start_size")))
PY
)"
YAML_OUT="$(echo "${PY_OUT}" | sed -n '1p')"
YAML_DATA="$(echo "${PY_OUT}" | sed -n '2p')"
YAML_QID="$(echo "${PY_OUT}" | sed -n '3p')"
YAML_ROUNDS="$(echo "${PY_OUT}" | sed -n '4p')"
YAML_BUDGET="$(echo "${PY_OUT}" | sed -n '5p')"
YAML_START_SIZE="$(echo "${PY_OUT}" | sed -n '6p')"

OUT_DIR="${OUT_DIR:-${YAML_OUT:-./outputs/default_exp}}"
DATA_DIR="${DATA_DIR:-${YAML_DATA:-./data/default_exp}}"
QID="${QID:-${YAML_QID:-}}"

if [[ -z "${QID}" ]]; then
  QID="$(basename "$(dirname "${OUT_DIR}")")"
  if [[ "${QID}" == "." || "${QID}" == "/" ]]; then QID="unknown_qid"; fi
fi

# --- パラメータ設定 ---
START_SIZE="${START_SIZE:-${YAML_START_SIZE:-50}}"
ROUNDS="${ROUNDS:-${YAML_ROUNDS:-10}}"
BUDGET="${BUDGET:-${YAML_BUDGET:-50}}"
echo "[run_al] Settings: START_SIZE=${START_SIZE}, ROUNDS=${ROUNDS}, BUDGET=${BUDGET}"

# --- ディレクトリとログファイルの準備 ---
mkdir -p "${OUT_DIR}"
CKPT_DIR="${OUT_DIR}/checkpoints_min"
LOG_FILE="${OUT_DIR}/run_al.log"
TIME_FILE="${OUT_DIR}/time.txt"

AL_BY="${AL_BY:-trust}"
AL_SAMPLER="${AL_SAMPLER:-uncertainty}"
AL_ASC="${AL_ASC:-}"
AL_CLEAN_CKPT_AFTER="${AL_CLEAN_CKPT_AFTER:-0}"

# ★ run_id を QID_実験名 の形式で生成 ★
EXP_NAME=$(basename "${OUT_DIR}")
RUN_ID="${QID}_${EXP_NAME}"
echo "[run_al] RUN_ID set to: ${RUN_ID}"

COMMON_ARGS=(
  --set "run.out_dir=${OUT_DIR}"
  --set "run.data_dir=${DATA_DIR}"
  --set "data.qid=${QID}"
  --set "run.run_id=${RUN_ID}"  # ここでPython側に渡す
)
if [[ -n "${CFG_OVERRIDE:-}" ]]; then
  read -ra ADDR <<< "${CFG_OVERRIDE}"
  COMMON_ARGS+=("${ADDR[@]}")
fi

# 環境情報の保存
ENV_LOG="${OUT_DIR}/env_info.txt"
{
  echo "Date: $(date)"
  echo "Config: ${CFG}"
  echo "Override: ${CFG_OVERRIDE:-}"
  echo "Run_ID: ${RUN_ID}"
  pip freeze | grep -E "torch|transformers|numpy|pandas" || echo ""
} > "${ENV_LOG}"


DUMP_OUT="${OUT_DIR}/exp_config.yaml"
DUMP_ARGS=(
  config-dump -c "${CFG}"
  "${COMMON_ARGS[@]}"
  --set "split.n_train=${START_SIZE}"
  --set "al.rounds=${ROUNDS}"
  --set "al.budget=${BUDGET}"
  --set "al.sampler.name=${AL_SAMPLER}"
  --set "al.sampler.uncertainty=${AL_BY}"
  --out "${DUMP_OUT}"
)

tensaku "${DUMP_ARGS[@]}"


# --- データ準備 ---
mkdir -p "${OUT_DIR}" "${DATA_DIR}"
MISSING_FILES=0
for f in labeled.jsonl pool.jsonl dev.jsonl test.jsonl; do
  if [[ ! -f "${DATA_DIR}/$f" ]]; then MISSING_FILES=1; break; fi
done
if [[ "${MISSING_FILES}" -eq 1 ]]; then
  echo "[run_al] Data missing. Attempting auto-split..."
  INPUT_ALL="${INPUT_ALL:-}"
  if [[ -z "${INPUT_ALL}" ]]; then
    if [[ -f "${ROOT}/data_sas/all.jsonl" ]]; then INPUT_ALL="${ROOT}/data_sas/all.jsonl";
    elif [[ -f "${ROOT}/data_sas/q-${QID}/all.jsonl" ]]; then INPUT_ALL="${ROOT}/data_sas/q-${QID}/all.jsonl";
    fi
  fi
  if [[ -f "${INPUT_ALL}" ]]; then
    echo "[run_al] Found master data: ${INPUT_ALL}"
    SPLIT_ARGS=(
      split -c "${CFG}"
      "${COMMON_ARGS[@]}"
      --set "data.input_all=${INPUT_ALL}"
    )
    if [[ -n "${START_SIZE}" ]]; then
      SPLIT_ARGS+=(--n-train "${START_SIZE}")
    fi
    tensaku "${SPLIT_ARGS[@]}"
  else
    echo "[run_al] ERROR: Data files missing and 'all.jsonl' not found."
    exit 1
  fi
else
  echo "[run_al] Data ready in ${DATA_DIR}."
fi

# ==========================================
# メイン処理ブロック
# ==========================================
{
  # 全体の開始時刻の記録
  START_TIME_SEC=$(date +%s)
  START_DATE_STR=$(date '+%Y-%m-%d %H:%M:%S')

  echo "=========================================================================="
  echo "=== Active Learning Loop Start at ${START_DATE_STR} ==="
  echo "=== Experiment ID (Dir): ${OUT_DIR} ==="
  echo "=== Run ID: ${RUN_ID} ==="
  echo "=========================================================================="

  start_round=0
  HISTORY_FILE="${OUT_DIR}/al_history.csv"

  if [[ "${AL_RESUME:-0}" == "1" && -f "${HISTORY_FILE}" ]]; then
      LAST_R=$(python -c "import csv,sys; r=[int(d['round']) for d in csv.DictReader(open(sys.argv[1])) if d['round'].isdigit()] or [-1]; print(max(r))" "${HISTORY_FILE}")
      if [[ "${LAST_R}" != "-1" ]]; then
          start_round=$((LAST_R + 1))
          echo "[run_al] 🔄 Resuming from Round ${start_round}..."
      fi
  fi
  round=${start_round}

  # --- ラウンドループ ---
  while [[ "${round}" -lt "${ROUNDS}" ]]; do

    # 各ラウンド開始時に時刻を記録
    echo ""
    echo "--------------------------------------------------------------------------"
    echo "--- Round ${round}/${ROUNDS} STARTED at $(date '+%Y-%m-%d %H:%M:%S') ---"
    echo "--------------------------------------------------------------------------"

    POOL_FILE="${DATA_DIR}/pool.jsonl"
    if [[ ! -f "${POOL_FILE}" ]] || [[ $(grep -c . "${POOL_FILE}") -eq 0 ]]; then
      echo "[run_al] Pool is empty. Stopping AL loop gracefully."
      break
    fi

    N_LABELED=0
    if [[ -f "${DATA_DIR}/labeled.jsonl" ]]; then
        N_LABELED=$(grep -c . "${DATA_DIR}/labeled.jsonl" || echo 0)
    fi

    # HITL実行 (学習・推論)
    CFG="${CFG}" QID="${QID}" DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" \
    AL_ROUND_IDX="${round}" CFG_OVERRIDE="${CFG_OVERRIDE:-}" \
    AL_N_LABELED="${N_LABELED}" \
      bash "${ROOT}/scripts/core/run_hitl.sh"

    # サンプリング
    SAMPLE_CMD=(tensaku al-sample -c "${CFG}" "${COMMON_ARGS[@]}" --budget "${BUDGET}" --uncertainty "${AL_BY}")
    if [[ -n "${AL_SAMPLER}" ]]; then SAMPLE_CMD+=(--sampler "${AL_SAMPLER}"); fi
    if [[ "${AL_ASC}" == "1" ]]; then SAMPLE_CMD+=(--ascending);
    elif [[ "${AL_ASC}" == "0" ]]; then SAMPLE_CMD+=(--descending); fi

    echo "[run_al] Running sampler..."
    "${SAMPLE_CMD[@]}"

    if [[ ! -f "${OUT_DIR}/al_sample_ids.txt" ]] || ! grep -q '.' "${OUT_DIR}/al_sample_ids.txt"; then
      echo "[run_al] No samples selected. Break."
      break
    fi

    # ラベル付与 (Oracle)
    ORACLE_CSV="${OUT_DIR}/oracle_labels_round${round}.csv"
    tensaku al-oracle-labels -c "${CFG}" "${COMMON_ARGS[@]}" --out "${ORACLE_CSV}"
    if [[ ! -f "${ORACLE_CSV}" ]]; then
        echo "[run_al] ERROR: Oracle CSV generation failed."
        exit 1
    fi
    tensaku al-label-import -c "${CFG}" "${COMMON_ARGS[@]}" --labels "${ORACLE_CSV}"

    # 履歴記録 (Python)
    LOG_SAMPLER="${AL_SAMPLER:-uncertainty}"
    python - <<PY
from tensaku.config import load_config
from tensaku.al_log import append_round_history
import os
def cnt(p): return sum(1 for _ in open(p)) if os.path.exists(p) else 0
cfg = load_config("${CFG}", ["run.out_dir=${OUT_DIR}","run.data_dir=${DATA_DIR}","data.qid=${QID}"])
append_round_history(cfg, ${round}, "${LOG_SAMPLER}", "${AL_BY}", ${BUDGET},
    cnt("${DATA_DIR}/labeled.jsonl"), cnt("${DATA_DIR}/pool.jsonl"), "oracle")
PY

    # ファイル整理
    ROUNDS_DIR="${OUT_DIR}/rounds"
    mkdir -p "${ROUNDS_DIR}"
    mv "${OUT_DIR}"/*_round${round}.* "${ROUNDS_DIR}/" 2>/dev/null || true
    round=$((round + 1))
  done

  # --- 終了処理 ---
  REPORT_SCRIPT="${ROOT}/scripts/utils/report_al.sh"
  if [[ -f "${REPORT_SCRIPT}" ]]; then
    echo "[run_al] Running final report process..."
    bash "${REPORT_SCRIPT}" "${OUT_DIR}" "${QID}"
  fi

  if [[ "${AL_CLEAN_CKPT_AFTER}" == "1" && -d "${CKPT_DIR}" ]]; then
    rm -rf -- "${CKPT_DIR}"
    echo "[run_al] Checkpoints cleaned."
  fi

  # 全体の所要時間計算と記録 (time.txt)
  END_TIME_SEC=$(date +%s)
  DURATION=$((END_TIME_SEC - START_TIME_SEC))

  echo "Duration_Seconds: ${DURATION}" > "${TIME_FILE}"
  echo "Start_Date: ${START_DATE_STR}" >> "${TIME_FILE}"
  echo "End_Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "${TIME_FILE}"
  echo "=========================================================================="
  echo "[run_al] All done. Total execution time: ${DURATION} seconds."
  echo "=========================================================================="

} 2>&1 | tee "${LOG_FILE}"
