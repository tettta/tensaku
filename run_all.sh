#!/usr/bin/env bash
# ==============================================================================
# run_all.sh (v6)
# Batch runner for Tensaku experiments.
#
# Fixes:
# - Added missing arg parsing for --nohup-log, --print-log-dir, --pid-file
# - Support for Explicit Tagging via --tag
# ==============================================================================

set -uo pipefail

# -----------------------
# Defaults
# -----------------------
MODE="prod"
PYTHON_BIN="python"
MAIN="src/main.py"

# Default grids
SAMPLERS_DEFAULT=("random" "msp" "trust" "kmeans" "hybrid")
SEEDS_DEFAULT=(42 43 44)
QIDS_DEFAULT=("Y14_1-2_1_3")

# Params
ON_EXIST="skip"
LOG_ROOT="logs"
TAG="default"  # Default exp_tag
DRY_RUN=0
NOHUP_LOG=""
PRINT_LOG_DIR=0
PID_FILE=""

STOP_ON_FAIL=0
HYDRA_FULL_ERROR=0

CUSTOM_BASE=""
EXTRA_OVERRIDES=""

# -----------------------
# Helpers
# -----------------------
usage() {
  cat <<'USAGE'
Usage:
  bash run_all.sh [options]

Options:
  --mode <debug|prod|custom>     Execution mode (default: prod)
  --tag <name>                   Experiment tag (default: default)
  --qids <qid1,...>              QIDs to run
  --samplers <s1,...>            Samplers
  --seeds <n1,...>               Seeds
  --on-exist <skip|overwrite|error>
  --log-root <dir>               Log directory root
  --extra "<k=v ...>"            Hydra overrides
  --dry-run                      Print commands only
  --stop-on-fail                 Stop on first error
  --hydra-full-error             Show full stacktrace
  --nohup-log <path>             Mirror output to file
  --print-log-dir                Print log dir early
  --pid-file <path>              Write PID to file

Examples:
  bash run_all.sh --tag v1_budget50 --qids Y14_1-2_1_3 --extra "al.budget=50"
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

split_csv_to_array() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<<"$csv"
}

join_cmd() {
  local -n arr="$1"
  local s=""
  for x in "${arr[@]}"; do
    if [[ "$x" == *" "* ]]; then s+="\"$x\" "; else s+="$x "; fi
  done
  echo "${s% }"
}

# -----------------------
# Arg parse (FIXED)
# -----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --qids) QIDS_CSV="$2"; shift 2;;
    --samplers) SAMPLERS_CSV="$2"; shift 2;;
    --seeds) SEEDS_CSV="$2"; shift 2;;
    --on-exist) ON_EXIST="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --main) MAIN="$2"; shift 2;;
    --extra) EXTRA_OVERRIDES="$2"; shift 2;;
    --base) CUSTOM_BASE="$2"; shift 2;;
    
    # ▼▼▼ 追加した箇所 ▼▼▼
    --nohup-log) NOHUP_LOG="$2"; shift 2;;
    --print-log-dir) PRINT_LOG_DIR=1; shift 1;;
    --pid-file) PID_FILE="$2"; shift 2;;
    # ▲▲▲ 追加した箇所 ▲▲▲

    --dry-run) DRY_RUN=1; shift 1;;
    --stop-on-fail) STOP_ON_FAIL=1; shift 1;;
    --hydra-full-error) HYDRA_FULL_ERROR=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done

# Build grids
SAMPLERS=("${SAMPLERS_DEFAULT[@]}")
SEEDS=("${SEEDS_DEFAULT[@]}")
QIDS=("${QIDS_DEFAULT[@]}")

if [[ -n "${SAMPLERS_CSV:-}" ]]; then split_csv_to_array "$SAMPLERS_CSV" SAMPLERS; fi
if [[ -n "${SEEDS_CSV:-}" ]]; then
  tmp=()
  IFS=',' read -r -a tmp <<<"$SEEDS_CSV"
  SEEDS=()
  for s in "${tmp[@]}"; do
    [[ "$s" =~ ^[0-9]+$ ]] || die "Seed must be int (got: $s)"
    SEEDS+=("$s")
  done
fi
if [[ -n "${QIDS_CSV:-}" ]]; then split_csv_to_array "$QIDS_CSV" QIDS; fi

# Base params
BASE_PARAMS=""
if [[ "$MODE" == "debug" ]]; then
  echo "⚠️  RUNNING IN DEBUG MODE"
  BASE_PARAMS="run=debug train=debug al=few_cycle cleaner=debug"
elif [[ "$MODE" == "prod" ]]; then
  echo "🚀 RUNNING IN PROD MODE"
  BASE_PARAMS="run=base train=standard al=base cleaner=base"
else
  [[ -n "$CUSTOM_BASE" ]] || die "custom mode requires --base"
  echo "🧩 RUNNING IN CUSTOM MODE"
  BASE_PARAMS="$CUSTOM_BASE"
fi

COMMON_PARAMS="run.on_exist=${ON_EXIST}"

# Env
if [[ "$HYDRA_FULL_ERROR" -eq 1 ]]; then export HYDRA_FULL_ERROR=1; fi

# Logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# バッチごとにフォルダを分ける運用にするなら下記推奨（現状は指定フォルダ直下）
LOG_DIR="${LOG_ROOT}"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/master.log"
SUMMARY_CSV="${LOG_DIR}/summary.csv"

# Redirect for nohup
if [[ -n "$NOHUP_LOG" ]]; then
  mkdir -p "$(dirname "$NOHUP_LOG")"
  exec > >(tee -a "$NOHUP_LOG") 2>&1
fi

# Write PID
if [[ -n "$PID_FILE" ]]; then
  mkdir -p "$(dirname "$PID_FILE")"
  echo "$$" > "$PID_FILE"
fi

# Print Log Dir
if [[ "$PRINT_LOG_DIR" -eq 1 ]]; then
  echo "[RUN_ALL] LOG_DIR=${LOG_DIR}"
  echo "[RUN_ALL] MASTER_LOG=${MASTER_LOG}"
  echo "[RUN_ALL] SUMMARY_CSV=${SUMMARY_CSV}"
fi

TOTAL_COUNT=$((${#QIDS[@]} * ${#SAMPLERS[@]} * ${#SEEDS[@]}))

# Header
{
  echo "=== Batch Experiment (${MODE}) Started at $(date) ==="
  echo "Tag: ${TAG}"
  echo "Total Jobs: ${TOTAL_COUNT}"
  echo "---------------------------------------------------"
} >> "$MASTER_LOG"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "qid,sampler,seed,status,exit_code,log_path" > "$SUMMARY_CSV"
fi

# Run Loop
job_i=0
for qid in "${QIDS[@]}"; do
  for sampler in "${SAMPLERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      job_i=$((job_i+1))
      LABEL="${qid}:${sampler}:${seed}"
      echo "[$(date +%H:%M:%S)] (${job_i}/${TOTAL_COUNT}) Starting: ${LABEL}" | tee -a "$MASTER_LOG"

      QID_LOG="${LOG_DIR}/run_${qid}_${sampler}_${seed}.log"

      CMD=("$PYTHON_BIN" "-u" "$MAIN"
           "data.qid=${qid}"
           "al/sampler=${sampler}"
           "run.seed=${seed}"
      )

      # shellcheck disable=SC2206
      CMD+=($BASE_PARAMS)
      # shellcheck disable=SC2206
      CMD+=($COMMON_PARAMS)
      if [[ -n "$EXTRA_OVERRIDES" ]]; then
        # shellcheck disable=SC2206
        CMD+=($EXTRA_OVERRIDES)
      fi

      CMD_STR="$(join_cmd CMD)"
      echo "CMD: $CMD_STR" >> "$MASTER_LOG"

      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] $CMD_STR" | tee -a "$MASTER_LOG"
        continue
      fi

      "${CMD[@]}" >"$QID_LOG" 2>&1
      EXIT_CODE=$?

      STATUS="SUCCESS"
      if [[ $EXIT_CODE -ne 0 ]]; then
        STATUS="FAILED"
      elif grep -q "\[AL_STATUS\] SKIPPED" "$QID_LOG"; then
        STATUS="SKIPPED"
      fi

      echo "${qid},${sampler},${seed},${STATUS},${EXIT_CODE},${QID_LOG}" >> "$SUMMARY_CSV"

      if [[ "$STATUS" == "SUCCESS" ]]; then
        echo "  -> SUCCESS" | tee -a "$MASTER_LOG"
      elif [[ "$STATUS" == "SKIPPED" ]]; then
        echo "  -> SKIPPED" | tee -a "$MASTER_LOG"
      else
        echo "  -> FAILED (Exit Code: $EXIT_CODE). Check $QID_LOG" | tee -a "$MASTER_LOG"
        if [[ "$STOP_ON_FAIL" -eq 1 ]]; then
          echo "Stopping on fail." | tee -a "$MASTER_LOG"
          exit 1
        fi
      fi
      echo "---------------------------------------------------" >> "$MASTER_LOG"
    done
  done
done

echo "Batch finished. Summary: $SUMMARY_CSV" | tee -a "$MASTER_LOG"
exit 0