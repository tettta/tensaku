#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/core/hitl.sh
set -euo pipefail

# =============================================================================
# HITL 一連パイプライン実行スクリプト
# =============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG_REL="${CFG:-configs/exp_al_hitl.yaml}"

if [[ "${CFG_REL}" = /* ]]; then CFG_ABS="${CFG_REL}"; else CFG_ABS="${ROOT}/${CFG_REL}"; fi
if [[ ! -f "${CFG_ABS}" ]]; then echo "[hitl.sh] ERROR: config not found: ${CFG_ABS}" >&2; exit 1; fi

QID="${QID:-}"
AL_ROUND_IDX="${AL_ROUND_IDX:-}"

# --- 1. ディレクトリ決定 ---
PY_OUT="$(python - << 'PY' "${CFG_ABS}"
import sys, yaml
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
EFFECTIVE_QID="${QID:-${YAML_QID:-}}"

if [[ -n "${DATA_DIR:-}" ]]; then :
elif [[ -n "${QID}" ]]; then DATA_DIR="${ROOT}/data_sas/splits/q-${QID}"
else DATA_DIR="${YAML_DATA}"; fi

if [[ -n "${OUT_DIR:-}" ]]; then :
elif [[ -n "${QID}" ]]; then OUT_DIR="${ROOT}/outputs/q-${QID}"
else OUT_DIR="${YAML_OUT}"; fi

if [[ -z "${DATA_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "[hitl.sh] ERROR: DATA_DIR/OUT_DIR undetermined." >&2; exit 1
fi

mkdir -p "${OUT_DIR}"
echo "[hitl.sh] QID=${EFFECTIVE_QID} / ROUND=${AL_ROUND_IDX:-'(none)'}"
echo "[hitl.sh] DATA=${DATA_DIR}"
echo "[hitl.sh] OUT =${OUT_DIR}"

COMMON_SET_ARGS=(--set "run.data_dir=${DATA_DIR}" --set "run.out_dir=${OUT_DIR}")

# CFG_OVERRIDE があれば結合
if [[ -n "${CFG_OVERRIDE:-}" ]]; then
  read -ra OVERRIDE_ARR <<< "${CFG_OVERRIDE}"
  COMMON_SET_ARGS+=("${OVERRIDE_ARR[@]}")
fi

# --- 2. split ---
SPLIT_NEEDED=0
for f in labeled.jsonl dev.jsonl test.jsonl pool.jsonl; do
  if [[ ! -f "${DATA_DIR}/$f" ]]; then SPLIT_NEEDED=1; break; fi
done

if [[ "${SPLIT_NEEDED}" -eq 1 ]]; then
  echo "[hitl.sh] Running split..."
  SPLIT_OPTS=("${COMMON_SET_ARGS[@]}" --set "data.qid=${EFFECTIVE_QID}")
  if [[ -n "${INPUT_ALL:-}" ]]; then SPLIT_OPTS+=(--set "data.input_all=${INPUT_ALL}"); fi
  tensaku split -c "${CFG_ABS}" "${SPLIT_OPTS[@]}"
fi

# --- 3. Train & Infer ---
CKPT_PATH="${INFER_CKPT:-${OUT_DIR}/checkpoints_min/best.pt}"

echo "[hitl.sh] STEP 1: train"
tensaku train -c "${CFG_ABS}" "${COMMON_SET_ARGS[@]}"

echo "[hitl.sh] STEP 2: infer-pool (ckpt=${CKPT_PATH})"
tensaku infer-pool -c "${CFG_ABS}" --trust "${COMMON_SET_ARGS[@]}" \
  --set "infer.ckpt=${CKPT_PATH}"

echo "[hitl.sh] STEP 3: confidence"
tensaku confidence -c "${CFG_ABS}" "${COMMON_SET_ARGS[@]}"


echo "[hitl.sh] STEP 4: gate"
tensaku gate -c "${CFG_ABS}" --conf-key conf_trust "${COMMON_SET_ARGS[@]}"

echo "[hitl.sh] STEP 5: hitl-report"
REPORT_ARGS=("${COMMON_SET_ARGS[@]}")
if [[ -n "${AL_N_LABELED:-}" ]]; then
  REPORT_ARGS+=(--n-labeled "${AL_N_LABELED}")
fi
tensaku hitl-report -c "${CFG_ABS}" "${REPORT_ARGS[@]}"

# --- 4. Rename Summary ---
if [[ "${AL_ROUND_IDX}" != "" ]]; then
  SUMMARY_CSV="${OUT_DIR}/hitl_summary.csv"
  SUMMARY_JSON="${OUT_DIR}/hitl_summary.json"
  
  if [[ -f "${SUMMARY_CSV}" ]]; then
    NEW_CSV="${OUT_DIR}/hitl_summary_round${AL_ROUND_IDX}.csv"
    mv "${SUMMARY_CSV}" "${NEW_CSV}"
    echo "[hitl.sh] Renamed summary: $(basename "${SUMMARY_CSV}") -> $(basename "${NEW_CSV}")"
  fi
  
  if [[ -f "${SUMMARY_JSON}" ]]; then
    NEW_JSON="${OUT_DIR}/hitl_summary_round${AL_ROUND_IDX}.json"
    mv "${SUMMARY_JSON}" "${NEW_JSON}"
    echo "[hitl.sh] Renamed json: $(basename "${SUMMARY_JSON}") -> $(basename "${NEW_JSON}")"
  else
    echo "[hitl.sh] WARN: hitl_summary.json not found (skipping rename)."
  fi
fi

echo "[hitl.sh] DONE."