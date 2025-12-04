#!/usr/bin/env bash
#/home/esakit25/work/tensaku/scripts/utils/report_al.sh
# -*- coding: utf-8 -*-

# =============================================================================
# Report Generation Script for Active Learning Experiments
# =============================================================================

set -euo pipefail

EXP_DIR="$1"
QID="${2:-}"

if [[ -z "${EXP_DIR}" || ! -d "${EXP_DIR}" ]]; then
    echo "Usage: bash report_al.sh <EXP_DIR> [QID]"
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${QID}" ]]; then
    PARENT_DIR=$(dirname "${EXP_DIR}")
    QID=$(basename "${PARENT_DIR}" | sed 's/^q-//')
fi

PLOTS_DIR="${EXP_DIR}/plots"
mkdir -p "${PLOTS_DIR}"

echo "[report_al] Processing ${EXP_DIR} (QID=${QID})..."
echo "[report_al] Artifacts will be saved to: ${PLOTS_DIR}"

# -----------------------------------------------------------------------------
# 1. 数値集計 (Aggregation) - Critical
# -----------------------------------------------------------------------------
AGG_SCRIPT="${ROOT}/scripts/utils/aggregate_al_rounds.py"
if [[ -f "${AGG_SCRIPT}" ]]; then
    echo "[report_al] [1/4] Aggregating logs..."
    python "${AGG_SCRIPT}" --dir "${EXP_DIR}"
fi

# -----------------------------------------------------------------------------
# 2. 信頼性評価 (Reliability & AURC) - Critical
# -----------------------------------------------------------------------------
# plot_al_reliability.py: --exp-dir (入力), --out-dir (出力) を使用
PLOT_REL="${ROOT}/scripts/utils/plot_al_reliability.py"

if [[ -f "${PLOT_REL}" ]]; then
    echo "[report_al] [2/4] Assessing Reliability (AURC, Reliability Diagram)..."
    python "${PLOT_REL}" --exp-dir "${EXP_DIR}" --out-dir "${PLOTS_DIR}" || {
        echo "[report_al] ERROR: Reliability evaluation failed! Stopping." >&2
        exit 1
    }
else
    # 万が一ファイル名変更を忘れていた場合のフォールバックは削除し、エラーにする
    echo "[report_al] ERROR: Script not found: ${PLOT_REL}" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 3. 基本性能 & 診断 (Performance & Diagnosis) - Important
# -----------------------------------------------------------------------------
# plot_al_confmat.py: 修正済み引数を使用 (--exp-dir, --out-dir)
PLOT_CM="${ROOT}/scripts/utils/plot_al_confmat.py"
if [[ -f "${PLOT_CM}" ]]; then
    echo "[report_al] [3/4] Generating Confusion Matrix..."
    python "${PLOT_CM}" --exp-dir "${EXP_DIR}" --out-dir "${PLOTS_DIR}" --split test || \
    echo "[report_al] ERROR: plot_al_confmat failed." >&2
fi

# plot_al_select.py: 修正済み引数を使用 (--exp-dir, --out-dir)
PLOT_SEL="${ROOT}/scripts/utils/plot_al_select.py"
if [[ -f "${PLOT_SEL}" ]]; then
    echo "[report_al] [3/4] Analyzing Selected Samples..."
    python "${PLOT_SEL}" --exp-dir "${EXP_DIR}" --out-dir "${PLOTS_DIR}" || \
    echo "[report_al] ERROR: plot_al_select failed." >&2
fi

# -----------------------------------------------------------------------------
# 4. 高負荷な可視化 (t-SNE) - Optional
# -----------------------------------------------------------------------------
# plot_al_embeddings.py: 従来の qid/root 方式を維持 (単一実行オプションがないため)
PLOT_EMB="${ROOT}/scripts/utils/plot_al_embeddings.py"
if [[ -f "${PLOT_EMB}" ]]; then
    echo "[report_al] [4/4] Generating Embeddings (t-SNE)..."
    python "${PLOT_EMB}" --qid "${QID}" --root "${ROOT}" --round 0 || {
        echo "[report_al] WARN: Embeddings visualization failed (Optional)." >&2
    }
fi

echo "[report_al] Done."
echo " - CSV Summaries: ${PLOTS_DIR}/aurc_summary.csv"
echo " - Analysis Plots: ${PLOTS_DIR}"