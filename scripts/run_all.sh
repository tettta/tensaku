#!/usr/bin/env bash
# ==============================================================================
# run_all.sh (v7 - Strict Logging Edition)
# ==============================================================================
set -uo pipefail

# --- Defaults ---
MODE="prod"
PYTHON_BIN="python"
MAIN="src/main.py"

# Default grids
SAMPLERS_DEFAULT=("random" "msp" "trust" "kmeans" "hybrid")
SEEDS_DEFAULT=(42 43 44)
QIDS_DEFAULT=("Y14_1-2_2_4" "Y14_2-1_1_5" "Y14_2-1_2_3" "Y14_2-2_1_4" "Y14_2-2_2_3")
# QIDS_DEFAULT=( "Y14_1-2_2_3" )

# Params
ON_EXIST="skip"
LOG_ROOT="logs"
DRY_RUN=0
NOHUP_LOG=""      # ここには詳細ログ(stdout/stderr)が入る
PID_FILE=""       # バッチ全体のPIDファイルパス
STOP_ON_FAIL=0
HYDRA_FULL_ERROR=0

CUSTOM_BASE=""
EXTRA_OVERRIDES=""

# --- Helpers ---
usage() {
  echo "Usage: bash run_all.sh [options] [hydra_overrides...]"
  exit 1
}

die() { echo "ERROR: $*" >&2; exit 2; }

split_csv_to_array() {
  local csv="$1"; local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<<"$csv"
}

# --- Arg Parse ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --qids) QIDS_CSV="$2"; shift 2;;
    --samplers) SAMPLERS_CSV="$2"; shift 2;;
    --seeds) SEEDS_CSV="$2"; shift 2;;
    --on-exist) ON_EXIST="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --main) MAIN="$2"; shift 2;;
    --base) CUSTOM_BASE="$2"; shift 2;;
    --nohup-log) NOHUP_LOG="$2"; shift 2;;
    --pid-file) PID_FILE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift 1;;
    --stop-on-fail) STOP_ON_FAIL=1; shift 1;;
    --hydra-full-error) HYDRA_FULL_ERROR=1; shift 1;;
    -h|--help) usage;;
    *) 
      # 未知の引数はHydraへの上書きとして扱う
      EXTRA_OVERRIDES+=" $1"
      shift 1
      ;;
  esac
done

# --- Setup Grids ---
SAMPLERS=("${SAMPLERS_DEFAULT[@]}")
SEEDS=("${SEEDS_DEFAULT[@]}")
QIDS=("${QIDS_DEFAULT[@]}")
if [[ -n "${SAMPLERS_CSV:-}" ]]; then split_csv_to_array "$SAMPLERS_CSV" SAMPLERS; fi
if [[ -n "${SEEDS_CSV:-}" ]]; then split_csv_to_array "$SEEDS_CSV" SEEDS; fi
if [[ -n "${QIDS_CSV:-}" ]]; then split_csv_to_array "$QIDS_CSV" QIDS; fi

# --- Base Params ---
if [[ "$MODE" == "debug" ]]; then
  BASE_PARAMS="run=debug train=debug al=few_cycle cleaner=debug"
elif [[ "$MODE" == "prod" ]]; then
  BASE_PARAMS="run=base train=standard al=base cleaner=base"
else
  [[ -n "$CUSTOM_BASE" ]] || die "custom mode requires --base"
  BASE_PARAMS="$CUSTOM_BASE"
fi
COMMON_PARAMS="run.on_exist=${ON_EXIST}"
if [[ "$HYDRA_FULL_ERROR" -eq 1 ]]; then export HYDRA_FULL_ERROR=1; fi

# --- Logging Setup ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}"
mkdir -p "$LOG_DIR"

# master.log: 実験の進行状況まとめ（重複させないためコンソールには出さない）
MASTER_LOG="${LOG_DIR}/master.log"

# PID File (Batch Process ID)
if [[ -n "$PID_FILE" ]]; then
  mkdir -p "$(dirname "$PID_FILE")"
  echo "$$" > "$PID_FILE"
fi

# Redirect stdout/stderr to nohup log (Detailed Log)
if [[ -n "$NOHUP_LOG" ]]; then
  mkdir -p "$(dirname "$NOHUP_LOG")"
  # exec > ... してしまうと master.log への書き込みと競合したりややこしくなるので
  # 今回は「詳細ログはリダイレクト済み」という前提で、echoなどは明示的に使い分ける
  exec > >(tee -a "$NOHUP_LOG") 2>&1
fi

TOTAL_COUNT=$((${#QIDS[@]} * ${#SAMPLERS[@]} * ${#SEEDS[@]}))

# --- Header to Master Log ---
{
  echo "==================================================================="
  echo " Batch Experiment Summary (${MODE})"
  echo " Date: $(date)"
  echo " Batch PID: $$"
  echo " Total Jobs: ${TOTAL_COUNT}"
  echo " Base Params: ${BASE_PARAMS}"
  echo " Overrides: ${EXTRA_OVERRIDES}"
  echo "==================================================================="
} >> "$MASTER_LOG"

# --- Run Loop ---
job_i=0
for qid in "${QIDS[@]}"; do
  for sampler in "${SAMPLERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      job_i=$((job_i+1))
      
      # ログファイル名（詳細ログ用）
      QID_LOG="${LOG_DIR}/run_${qid}_${sampler}_${seed}.log"

      # コマンド構築
      CMD=("$PYTHON_BIN" "-u" "$MAIN" "data.qid=${qid}" "al/sampler=${sampler}" "run.seed=${seed}")
      # shellcheck disable=SC2206
      CMD+=($BASE_PARAMS)
      # shellcheck disable=SC2206
      CMD+=($COMMON_PARAMS)
      if [[ -n "$EXTRA_OVERRIDES" ]]; then
        # shellcheck disable=SC2206
        CMD+=($EXTRA_OVERRIDES)
      fi
      
      # Master Logへの開始記録 (echoのみ)
      echo "[$(date +%H:%M:%S)] (${job_i}/${TOTAL_COUNT}) START ${qid}:${sampler}:${seed}" >> "$MASTER_LOG"
      
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [DRY-RUN] ${CMD[*]}" >> "$MASTER_LOG"
        continue
      fi

      # 【重要】バックグラウンド実行してPIDを取得し、waitで待つ
      "${CMD[@]}" > "$QID_LOG" 2>&1 &
      JOB_PID=$!
      
      # PIDを記録
      echo "  -> PID: ${JOB_PID} (Logging to $(basename "$QID_LOG"))" >> "$MASTER_LOG"

      # プロセスの終了を待つ
      wait $JOB_PID
      EXIT_CODE=$?

      # 結果判定
      STATUS="SUCCESS"
      if [[ $EXIT_CODE -ne 0 ]]; then
        STATUS="FAILED"
      elif grep -q "\[AL_STATUS\] SKIPPED" "$QID_LOG"; then
        STATUS="SKIPPED"
      fi

      # 結果をMaster Logに追記
      if [[ "$STATUS" == "SUCCESS" ]]; then
        echo "  -> ✅ SUCCESS" >> "$MASTER_LOG"
      elif [[ "$STATUS" == "SKIPPED" ]]; then
        echo "  -> ⏭️  SKIPPED" >> "$MASTER_LOG"
      else
        echo "  -> ❌ FAILED (Exit: $EXIT_CODE)" >> "$MASTER_LOG"
        if [[ "$STOP_ON_FAIL" -eq 1 ]]; then
           echo "!!! Stop on fail triggered !!!" >> "$MASTER_LOG"
           exit 1
        fi
      fi
      
      # コンソール(nohup.log)には最低限の進捗だけ出す
      echo "(${job_i}/${TOTAL_COUNT}) ${STATUS} - ${qid}:${sampler}:${seed}"

    done
  done
done

echo "===================================================================" >> "$MASTER_LOG"
echo " Batch Finished at $(date)" >> "$MASTER_LOG"
echo "===================================================================" >> "$MASTER_LOG"

exit 0