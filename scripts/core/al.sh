#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/core/al.sh
# -*- coding: utf-8 -*-
#
# Active Learning Loop (Self-contained)

set -euo pipefail
IFS=$'\n\t'

# ROOT計算 (scripts/core/al.sh なので 2階層上)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${CFG:-${ROOT}/configs/exp_al_hitl.yaml}"

# --- QID, DATA_DIR, OUT_DIR の解決 ---
PY_OUT="$(python - <<'PY' "${CFG}"
import sys, yaml, os
try:
    with open(sys.argv[1], "r") as f: c = yaml.safe_load(f) or {}
    print(c.get("run", {}).get("out_dir", "").strip())
    print(c.get("run", {}).get("data_dir", "").strip())
    print(c.get("data", {}).get("qid", "").strip())
except: pass
PY
)"

YAML_OUT="$(echo "${PY_OUT}" | sed -n '1p')"
YAML_DATA="$(echo "${PY_OUT}" | sed -n '2p')"
YAML_QID="$(echo "${PY_OUT}" | sed -n '3p')"

OUT_DIR="${OUT_DIR:-${YAML_OUT:-./outputs/default_exp}}"
DATA_DIR="${DATA_DIR:-${YAML_DATA:-./data/default_exp}}"
QID="${QID:-${YAML_QID:-}}"

if [[ -z "${QID}" ]]; then
  QID="$(basename "$(dirname "${OUT_DIR}")")"
  if [[ "${QID}" == "." || "${QID}" == "/" ]]; then QID="unknown_qid"; fi
fi

CKPT_DIR="${OUT_DIR}/checkpoints_min"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ALパラメータ
ROUNDS="${AL_ROUNDS:-1}"
AL_K="${AL_K:-50}"
AL_BY="${AL_BY:-trust}"
AL_SAMPLER="${AL_SAMPLER:-}"
AL_ASC="${AL_ASC:-}" 
AL_CLEAN_CKPT_AFTER="${AL_CLEAN_CKPT_AFTER:-0}"

LOG_FILE="${LOG_DIR}/al_${QID}_${ROUNDS}rnd.log"

# tensakuコマンドへの共通引数
COMMON_ARGS=(
  --set "run.out_dir=${OUT_DIR}"
  --set "run.data_dir=${DATA_DIR}"
  --set "data.qid=${QID}"
)

# CFG_OVERRIDE があれば COMMON_ARGS に結合
if [[ -n "${CFG_OVERRIDE:-}" ]]; then
  read -ra ADDR <<< "${CFG_OVERRIDE}"
  COMMON_ARGS+=("${ADDR[@]}")
fi

# =============================================================================
# X. 環境情報の保存
# =============================================================================
ENV_LOG="${OUT_DIR}/env_info.txt"
mkdir -p "${OUT_DIR}"
{
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "QID: ${QID}"
  echo "Config: ${CFG}"
  echo "Override: ${CFG_OVERRIDE:-}"
  echo "--------------------------------------------------"
  echo "Git Commit:"
  git rev-parse HEAD 2>/dev/null || echo "No git repo"
  git diff --shortstat 2>/dev/null || echo ""
  echo "--------------------------------------------------"
  echo "Pip Freeze:"
  pip freeze | grep -E "torch|transformers|numpy|pandas|scikit-learn" || echo "pip freeze failed"
} > "${ENV_LOG}"

# =============================================================================
# 1. データの自動準備 (Self-Setup)
# =============================================================================
mkdir -p "${OUT_DIR}" "${DATA_DIR}"

MISSING_FILES=0
for f in labeled.jsonl pool.jsonl dev.jsonl test.jsonl; do
  if [[ ! -f "${DATA_DIR}/$f" ]]; then MISSING_FILES=1; break; fi
done

if [[ "${MISSING_FILES}" -eq 1 ]]; then
  echo "[al.sh] Data missing in ${DATA_DIR}. Attempting auto-split..."
  INPUT_ALL="${INPUT_ALL:-}"
  if [[ -z "${INPUT_ALL}" ]]; then
    if [[ -f "${ROOT}/data_sas/all.jsonl" ]]; then INPUT_ALL="${ROOT}/data_sas/all.jsonl";
    elif [[ -f "${ROOT}/data_sas/q-${QID}/all.jsonl" ]]; then INPUT_ALL="${ROOT}/data_sas/q-${QID}/all.jsonl";
    fi
  fi

  if [[ -f "${INPUT_ALL}" ]]; then
    echo "[al.sh] Found master data: ${INPUT_ALL}"
    tensaku split -c "${CFG}" "${COMMON_ARGS[@]}" --set "data.input_all=${INPUT_ALL}"
  else
    echo "[al.sh] ERROR: Data files missing and 'all.jsonl' not found."
    exit 1
  fi
else
  echo "[al.sh] Data ready in ${DATA_DIR}."
fi

# =============================================================================
# 2. AL ループ本体
# =============================================================================
{
  echo "=== Active Learning Loop Start ==="
  echo " CFG: ${CFG}"
  echo " QID: ${QID}"
  echo " OUT: ${OUT_DIR}"
  echo " DATA: ${DATA_DIR}"

  start_round=0
  HISTORY_FILE="${OUT_DIR}/al_history.csv"
  if [[ "${AL_RESUME:-0}" == "1" && -f "${HISTORY_FILE}" ]]; then
      LAST_R=$(python -c "import csv,sys; r=[int(d['round']) for d in csv.DictReader(open('${HISTORY_FILE}')) if d['round'].isdigit()] or [-1]; print(max(r))")
      if [[ "${LAST_R}" != "-1" ]]; then
          start_round=$((LAST_R + 1))
          echo "[al.sh] 🔄 Found history. Resuming from Round ${start_round}..."
      fi
  fi

  round=${start_round}

  while [[ "${round}" -lt "${ROUNDS}" ]]; do
    echo ""
    echo "--- Round ${round}/${ROUNDS} ---"

    # 1. HITL
    CFG="${CFG}" QID="${QID}" DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" \
    AL_ROUND_IDX="${round}" CFG_OVERRIDE="${CFG_OVERRIDE:-}" \
      bash "${ROOT}/scripts/core/hitl.sh"

    # 2. Sampling
    SAMPLE_CMD=(tensaku al-sample -c "${CFG}" "${COMMON_ARGS[@]}" \
                --budget "${AL_K}" --uncertainty "${AL_BY}")
    
    if [[ -n "${AL_SAMPLER}" ]]; then SAMPLE_CMD+=(--sampler "${AL_SAMPLER}"); fi
    if [[ "${AL_ASC}" == "1" ]]; then SAMPLE_CMD+=(--ascending); 
    elif [[ "${AL_ASC}" == "0" ]]; then SAMPLE_CMD+=(--descending); fi
    
    echo "[al.sh] Running sampler..."
    "${SAMPLE_CMD[@]}"

    if [[ ! -f "${OUT_DIR}/al_sample_ids.txt" ]] || ! grep -q '.' "${OUT_DIR}/al_sample_ids.txt"; then
      echo "[al.sh] No samples selected. AL finished."
      break
    fi

    # 3. Oracle Labeling
    ORACLE_CSV="${OUT_DIR}/oracle_labels_round${round}.csv"
    echo "[al.sh] Generating Oracle Labels -> ${ORACLE_CSV}"
    tensaku al-oracle-labels -c "${CFG}" "${COMMON_ARGS[@]}" --out "${ORACLE_CSV}"

    if [[ ! -f "${ORACLE_CSV}" ]]; then
        echo "[al.sh] ERROR: Oracle CSV generation failed."
        exit 1
    fi

    # 4. Import Labels
    tensaku al-label-import -c "${CFG}" "${COMMON_ARGS[@]}" --labels "${ORACLE_CSV}"

# 5. Log History
    # 【修正】固定文字 "al_sample" ではなく、実際の戦略名 (AL_SAMPLER) を記録する
    # AL_SAMPLER が空の場合はデフォルトの "uncertainty" とする
    LOG_SAMPLER="${AL_SAMPLER:-uncertainty}"

    python - <<PY
from tensaku.config import load_config
from tensaku.al_log import append_round_history
import os
def cnt(p): return sum(1 for _ in open(p)) if os.path.exists(p) else 0
cfg = load_config("${CFG}", ["run.out_dir=${OUT_DIR}","run.data_dir=${DATA_DIR}","data.qid=${QID}"])

# 第3引数を修正
append_round_history(cfg, ${round}, "${LOG_SAMPLER}", "${AL_BY}", ${AL_K},
    cnt("${DATA_DIR}/labeled.jsonl"), cnt("${DATA_DIR}/pool.jsonl"), "oracle")
PY

    round=$((round + 1))
  done


  # 6. Aggregate & Cleanup
  AGG_SCRIPT="${ROOT}/scripts/utils/aggregate_al_rounds.py"
  if [[ -f "${AGG_SCRIPT}" ]]; then
    echo "[al.sh] Aggregating learning curves..."
    python "${AGG_SCRIPT}" --dir "${OUT_DIR}"
    
    if [[ -f "${OUT_DIR}/al_learning_curve.csv" ]]; then
      echo "[al.sh] Cleaning up intermediate round summaries..."
      rm -f "${OUT_DIR}/hitl_summary_round"*.csv
      rm -f "${OUT_DIR}/hitl_summary_round"*.json
    fi
  fi

  # 7. Cleanup Checkpoints (モデルはフラグ次第で消す)
  if [[ "${AL_CLEAN_CKPT_AFTER}" == "1" && -d "${CKPT_DIR}" ]]; then
    rm -rf -- "${CKPT_DIR}"
    echo "[al.sh] Checkpoints cleaned."
  fi

  # 8. Cleanup Heavy Intermediate Files (【変更】中間ファイルは常に消す)
  #    logits.npy や embs.npy は事後分析で使う頻度が低く、容量を食うため
  if ls "${OUT_DIR}"/*.npy >/dev/null 2>&1; then
      rm -f "${OUT_DIR}"/*.npy
      echo "[al.sh] Heavy numpy files (*.npy) cleaned."
  fi

# 8. Individual Analysis Plots (個別分析)
echo "[al.sh] Generating individual analysis plots..."

# 選択データの分布 (Heatmap)
PLOT_SEL="${ROOT}/scripts/utils/plot_al_selection.py"
if [[ -f "${PLOT_SEL}" ]]; then
    python "${PLOT_SEL}" --qid "${QID}" --root "${ROOT}"
fi

# 混同行列 (Confusion Matrix)
PLOT_CM="${ROOT}/scripts/utils/plot_confusion_matrix.py"
if [[ -f "${PLOT_CM}" ]]; then
    python "${PLOT_CM}" --qid "${QID}" --root "${ROOT}" --split test
fi

echo "[al.sh] All done for this experiment."

} 2>&1 | tee "${LOG_FILE}"