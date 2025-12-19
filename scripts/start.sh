#!/usr/bin/env bash
set -u

export TENSAKU_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "📍 Project Root: ${TENSAKU_ROOT}"

# --- 1. 自動で場所を決める ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# もし引数でベースを指定したければ変えられるようにしても良いが、今回は固定
LOG_BASE="logs/${TIMESTAMP}"

# --- 2. ディレクトリ作成 ---
mkdir -p "${LOG_BASE}"

echo "🚀 Launching experiment..."
echo "📂 Log Directory: ${LOG_BASE}"

# --- 3. 実行 (nohup) ---
# "$@" は、このスクリプトに渡された引数（--mode debug など）をそのまま run_all.sh に渡す魔法
nohup bash scripts/run_all.sh \
  --log-root "${LOG_BASE}" \
  --nohup-log "${LOG_BASE}/nohup.log" \
  --pid-file "${LOG_BASE}/run.pid" \
  "$@" \
  > /dev/null 2>&1 &

# --- 4. 確認用情報の表示 ---
PID=$!
echo "✅ Started with PID: ${PID}"
echo "---------------------------------------------------"
echo "👀 Monitor log:"
echo "   tail -f ${LOG_BASE}/nohup.log"
echo ""
echo "🛑 Stop command:"
echo "   kill ${PID}"
echo "---------------------------------------------------"