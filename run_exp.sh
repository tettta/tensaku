#!/usr/bin/env bash
# /home/esakit25/work/tensaku/run_exp.sh
#
# 目的:
# - Tensaku 実験（Hydra multirun）を nohup で起動し、1回の実行ログを logs/run_YYYYmmdd_HHMMSS/ にまとめる
# - QIDは短いエイリアス（y14_1..y14_6）で指定できる
# - （任意）monitor_tensaku.py も同時に起動し、同じ run ディレクトリに監視ログを保存する
#
# 使い方（例）:
#   ./run_exp.sh                       # 既定（Q_DEFAULT）で起動
#   Q=y14_1,y14_3 ./run_exp.sh         # エイリアス指定
#   QID=Y14_2-1_1_5 ./run_exp.sh       # 直指定（従来互換）
#   ./run_exp.sh --list-qid            # エイリアス一覧表示
#   MONITOR=0 ./run_exp.sh             # 監視なし
#   ./run_exp.sh al.rounds=200         # 追加のHydra override

set -euo pipefail

# =========================
# 設定（ここだけ触ればOK）
# =========================

# プロジェクトパス
TENSAKU_ROOT="${TENSAKU_ROOT:-/home/esakit25/work/tensaku}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-${TENSAKU_ROOT}/src}"

# 実行ログ保存先（1回の起動＝1ディレクトリ）
LOG_ROOT="${LOG_ROOT:-logs}"

# QIDエイリアス（追加はここ）
# 形式: ["alias"]="QID"
declare -A QID_MAP=(
  ["y14_1"]="Y14_1-2_1_3"
  ["y14_2"]="Y14_1-2_2_4"
  ["y14_3"]="Y14_2-1_1_5"
  ["y14_4"]="Y14_2-1_2_3"
  ["y14_5"]="Y14_2-2_1_4"
  ["y14_6"]="Y14_2-2_2_3"
)

# 既定で回す対象（必要なら変更）
#Q_DEFAULT="${Q_DEFAULT:-y14_1,y14_2,y14_3,y14_4,y14_5,y14_6}"
Q_DEFAULT="${Q_DEFAULT:-y14_4,y14_5,y14_6}"

# 実験パラメータ（環境変数で上書き可能）
# Q: エイリアス指定（カンマ区切り）例: Q=y14_1,y14_3
Q="${Q:-$Q_DEFAULT}"
# QID: 直指定（従来互換）。Qが指定されていれば最終的にQが優先される
QID="${QID:-}"

RECIPE="${RECIPE:-al_only}"
EPOCHS="${EPOCHS:-5}"
ROUNDS="${ROUNDS:-1000}"
BUDGET="${BUDGET:-50}"
SEEDS="${SEEDS:-42,43,44}"
ON_EXIST="${ON_EXIST:-overwrite}"
SAMPLERS="${SAMPLERS:-random,kmeans,unc_msp,unc_trust,kcenter,u2d_kmeans,d2u_kmeans}"
#SAMPLERS="${SAMPLERS:-random}"

# monitor（任意）
MONITOR="${MONITOR:-1}"                      # 1=起動 / 0=起動しない
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"        # du/dfの監視対象（Tensakuのoutputsルート想定）
MONITOR_PY="${MONITOR_PY:-tools/monitor_tensaku.py}"

# monitor ALERT（指定したものだけ有効）
ALERT_RSS_GB="${ALERT_RSS_GB:-}"                         # 例: 85
ALERT_DF_USED_PCT="${ALERT_DF_USED_PCT:-}"               # 例: 90
ALERT_RSS_GROWTH_MBPM="${ALERT_RSS_GROWTH_MBPM:-}"       # 例: 200
ALERT_PY_PROCS_MAX="${ALERT_PY_PROCS_MAX:-}"             # 例: 200
ALERT_GPU_MEM_USED_MIB="${ALERT_GPU_MEM_USED_MIB:-}"     # 例: 23000
ALERT_DU_GB="${ALERT_DU_GB:-}"                           # 例: 800
ON_ALERT_CMD="${ON_ALERT_CMD:-}"                         # 例: 'kill -STOP {pid}'
ALERT_COOLDOWN_SEC="${ALERT_COOLDOWN_SEC:-300}"

# =========================
# 関数
# =========================

list_qid() {
  echo "QID エイリアス一覧:"
  for k in "${!QID_MAP[@]}"; do
    printf "  %-6s -> %s\n" "$k" "${QID_MAP[$k]}"
  done | sort
}

resolve_qid_from_aliases() {
  # 入力: $1 = "y14_1,y14_3" のようなカンマ区切り
  # 出力: 標準出力に "Y14_... ,Y14_..." を出す（不正ならエラーで終了）
  local aliases="$1"
  local -a keys
  local -a resolved
  local key

  IFS=',' read -r -a keys <<< "$aliases"
  resolved=()

  for key in "${keys[@]}"; do
    key="${key//[[:space:]]/}"
    [[ -n "$key" ]] || continue

    if [[ -z "${QID_MAP[$key]+x}" ]]; then
      echo "[ERROR] 未定義のQIDエイリアスです: '$key'" >&2
      echo "        ./run_exp.sh --list-qid で一覧を確認してください" >&2
      exit 2
    fi
    resolved+=("${QID_MAP[$key]}")
  done

  if [[ "${#resolved[@]}" -eq 0 ]]; then
    echo "[ERROR] Q が指定されていますが、有効なエイリアスがありません: Q='$aliases'" >&2
    exit 2
  fi

  (IFS=','; echo "${resolved[*]}")
}

# =========================
# メイン
# =========================

if [[ "${1:-}" == "--list-qid" ]]; then
  list_qid
  exit 0
fi

cd "$TENSAKU_ROOT"
export PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}"

mkdir -p "$LOG_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/run_${TS}"
mkdir -p "$RUN_DIR"

# runディレクトリ配下に集約
MAIN_LOG="${RUN_DIR}/main.log"
MAIN_PID_FILE="${RUN_DIR}/pid_main.txt"
CMD_TXT="${RUN_DIR}/cmd.txt"
META_TXT="${RUN_DIR}/meta.txt"

MON_TSV="${RUN_DIR}/monitor.tsv"
MON_STDOUT="${RUN_DIR}/monitor.stdout.log"
MON_EVENTS="${RUN_DIR}/monitor.events.log"
ALERT_LOG="${RUN_DIR}/ALERT.log"
MON_PID_FILE="${RUN_DIR}/pid_monitor.txt"

# 直近runへのポインタ（便利）
echo "$RUN_DIR" > "${LOG_ROOT}/last_run_dir.txt"
echo "$MAIN_LOG" > "${LOG_ROOT}/last_log_path.txt"

# Q があれば（既定でも入る）、エイリアス→QIDへ展開して最終QIDを決める
# 直指定QIDも残すが、Qが有効ならQ優先
if [[ -n "${Q:-}" ]]; then
  QID="$(resolve_qid_from_aliases "$Q")"
fi

# 最終的にQIDは必須（フォールバックなし）
: "${QID:?QID が空です。Q=y14_1,... もしくは QID=Y14_... を指定してください。}"

# 追加のHydra override（run_exp.shの引数）を末尾に足す
EXTRA_OVERRIDES=("$@")

BASE_CMD=(python src/main.py -m)

DEFAULT_OVERRIDES=(
  "run.recipe=${RECIPE}"
  "data.qid=${QID}"
  "train.epochs=${EPOCHS}"
  "al.rounds=${ROUNDS}"
  "al.budget=${BUDGET}"
  "al/sampler=${SAMPLERS}"
  "run.seed=${SEEDS}"
  "run.on_exist=${ON_EXIST}"
)

CMD=("${BASE_CMD[@]}" "${DEFAULT_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}")

# メタ情報
{
  echo "ts=${TS}"
  echo "run_dir=${RUN_DIR}"
  echo "TENSAKU_ROOT=${TENSAKU_ROOT}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "Q=${Q}"
  echo "QID=${QID}"
  echo "RECIPE=${RECIPE}"
  echo "EPOCHS=${EPOCHS}"
  echo "ROUNDS=${ROUNDS}"
  echo "BUDGET=${BUDGET}"
  echo "SEEDS=${SEEDS}"
  echo "ON_EXIST=${ON_EXIST}"
  echo "SAMPLERS=${SAMPLERS}"
  echo "MONITOR=${MONITOR}"
  echo "MONITOR_INTERVAL=${MONITOR_INTERVAL}"
  echo "OUTPUTS_DIR=${OUTPUTS_DIR}"
  echo "ALERT_RSS_GB=${ALERT_RSS_GB}"
  echo "ALERT_DF_USED_PCT=${ALERT_DF_USED_PCT}"
  echo "ALERT_RSS_GROWTH_MBPM=${ALERT_RSS_GROWTH_MBPM}"
  echo "ALERT_PY_PROCS_MAX=${ALERT_PY_PROCS_MAX}"
  echo "ALERT_GPU_MEM_USED_MIB=${ALERT_GPU_MEM_USED_MIB}"
  echo "ALERT_DU_GB=${ALERT_DU_GB}"
  echo "ON_ALERT_CMD=${ON_ALERT_CMD}"
  echo "ALERT_COOLDOWN_SEC=${ALERT_COOLDOWN_SEC}"
} > "$META_TXT"

# 再現用コマンド
printf "%q " env HYDRA_FULL_ERROR=1 "${CMD[@]}" > "$CMD_TXT"
echo >> "$CMD_TXT"

# main 起動
nohup env HYDRA_FULL_ERROR=1 \
  "${CMD[@]}" \
  > "$MAIN_LOG" 2>&1 &

MAIN_PID="$!"
echo "$MAIN_PID" > "$MAIN_PID_FILE"
echo "$MAIN_PID" > "${LOG_ROOT}/last_pid.txt"

echo "起動しました"
echo "  run_dir: ${RUN_DIR}"
echo "  main:    pid = ${MAIN_PID} log = ${MAIN_LOG}"
echo "  qid:     ${QID}"

# monitor 起動（任意）
if [[ "$MONITOR" == "1" ]]; then
  if [[ ! -f "$MONITOR_PY" ]]; then
    echo "[ERROR] monitor を有効化していますが、見つかりません: ${MONITOR_PY}" >&2
    echo "        MONITOR=0 ./run_exp.sh で monitor無し起動にしてください。" >&2
    exit 2
  fi

  MON_CMD=(
    python "$MONITOR_PY"
    --pid "$MAIN_PID"
    --interval "$MONITOR_INTERVAL"
    --out-tsv "$MON_TSV"
    --outputs-dir "$OUTPUTS_DIR"
    --log "$MAIN_LOG"
    --event-out "$MON_EVENTS"
    --alert-log "$ALERT_LOG"
    --alert-cooldown-sec "$ALERT_COOLDOWN_SEC"
  )

  [[ -n "$ALERT_RSS_GB" ]] && MON_CMD+=(--alert-rss-gb "$ALERT_RSS_GB")
  [[ -n "$ALERT_DF_USED_PCT" ]] && MON_CMD+=(--alert-df-used-pct "$ALERT_DF_USED_PCT")
  [[ -n "$ALERT_RSS_GROWTH_MBPM" ]] && MON_CMD+=(--alert-rss-growth-mb-per-min "$ALERT_RSS_GROWTH_MBPM")
  [[ -n "$ALERT_PY_PROCS_MAX" ]] && MON_CMD+=(--alert-python-procs-max "$ALERT_PY_PROCS_MAX")
  [[ -n "$ALERT_GPU_MEM_USED_MIB" ]] && MON_CMD+=(--alert-gpu-mem-used-mib "$ALERT_GPU_MEM_USED_MIB")
  [[ -n "$ALERT_DU_GB" ]] && MON_CMD+=(--alert-du-gb "$ALERT_DU_GB")
  [[ -n "$ON_ALERT_CMD" ]] && MON_CMD+=(--on-alert-cmd "$ON_ALERT_CMD")

  nohup "${MON_CMD[@]}" > "$MON_STDOUT" 2>&1 &
  MON_PID="$!"
  echo "$MON_PID" > "$MON_PID_FILE"

  echo "$MON_PID" > "${LOG_ROOT}/last_monitor_pid.txt"
  echo "$MON_TSV" > "${LOG_ROOT}/last_monitor_tsv_path.txt"
  echo "$MON_STDOUT" > "${LOG_ROOT}/last_monitor_stdout_path.txt"
  echo "$MON_EVENTS" > "${LOG_ROOT}/last_monitor_events_path.txt"
  echo "$ALERT_LOG" > "${LOG_ROOT}/last_alert_log_path.txt"

  echo "  monitor: pid = ${MON_PID} tsv = ${MON_TSV}"
else
  echo "  monitor: 無効（MONITOR=0）"
fi
