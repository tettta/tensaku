#!/usr/bin/env bash
# /home/esakit25/work/tensaku/run_exp_seq.sh
# -*- coding: utf-8 -*-
# Tensaku 実験を「1ジョブ=1 Pythonプロセス」で直列実行するラッパ（Hydra -m を使わない）
#
# 目的:
# - Hydra multirun(-m) は同一プロセス内で複数ジョブを回すため、PyTorch/HF のキャッシュ等が残って
#   RSS が積み上がるケースがある。
# - このスクリプトは (qid, seed, sampler) を外側でループし、各ジョブを別プロセスで起動する。
#   => ジョブ終了ごとに OS がメモリを回収できる。
#
# 使い方:
#   # 例) y14_3 の seed=44 だけ、sampler 2種（環境変数）
#   QIDS="y14_3" SEEDS="44" SAMPLERS="d2u_kmeans ud2_kmeans" ./run_exp_seq.sh
#
#   # 例) CLI で指定（環境変数より優先）
#   ./run_exp_seq.sh --qid y14_3 --seeds 42,43,44 --samplers random,kmeans
#
#   # 例) Hydra group override が効かない構成のとき（field override を使う）
#   ./run_exp_seq.sh --qid y14_3 --seeds 42 --samplers random --sampler-override field
#
# 監視:
#   MONITOR=1 で tools/monitor_tensaku.py を各ジョブに付ける（PIDが死んだら監視も自動停止）。
#   アラートで止めたい場合は ON_ALERT_CMD を指定（例: "kill -STOP {pid}"）。

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_exp_seq.sh [options]

Options:
  --qid <alias|QID>            単一QID（例: y14_3 / Y14_2-1_1_5）
  --qids "<...>"               複数（スペース/カンマ区切り）
  --seeds "<...>"              seed（スペース/カンマ区切り。例: "42,43,44"）
  --samplers "<...>"           sampler（スペース/カンマ区切り）

  --recipe <name>              run.recipe（default: al_only）
  --epochs <int>               train.epochs（default: 5）
  --rounds <int>               al.rounds（default: 1000）
  --budget <int>               al.budget（default: 50）
  --on-exist <mode>            run.on_exist（default: overwrite）

  --sampler-override <group|field>
                               sampler の Hydra override 方法
                               group:  al/sampler=<name>（default）
                               field:  al.sampler.name=<name>

  --monitor <0|1>              monitor の有効/無効（default: 1）
  --monitor-interval <sec>     monitor 間隔（default: 60）

  --no-nohup                   controller を foreground 実行（default: nohup でBG）
  --dry-run                    コマンドを出力して実行しない

  --tensaku-root <path>        default: /home/esakit25/work/tensaku
  --python-bin <path>          default: python
  --log-root-rel <relpath>     default: logs

  -h, --help                   show help

Env (existing compatible):
  QIDS, SEEDS, SAMPLERS, RECIPE, EPOCHS, ROUNDS, BUDGET, ON_EXIST,
  MONITOR, MONITOR_INTERVAL, TENSAKU_ROOT, PYTHON_BIN, LOG_ROOT_REL,
  OUTPUTS_DIR_FOR_MONITOR, ON_ALERT_CMD, ALERT_* ...
USAGE
}

# ---------------------------
# helpers
# ---------------------------
_split_to_array() {
  # usage: _split_to_array "a,b c" out_array_name
  local s="$1"
  local out_name="$2"
  s="${s//,/ }"
  # collapse repeated spaces
  s="$(echo "$s" | tr -s ' ')"
  # shellcheck disable=SC2206
  local arr=( $s )
  # set -u safe assignment by name
  eval "$out_name=()"
  local x
  for x in "${arr[@]}"; do
    if [[ -n "$x" ]]; then
      eval "$out_name+=(\"$x\")"
    fi
  done
}

_now() { date '+%Y-%m-%d %H:%M:%S'; }

# ---------------------------
# ユーザーがいじる変数（env default）
# ---------------------------
TENSAKU_ROOT="${TENSAKU_ROOT:-/home/esakit25/work/tensaku}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT_REL="${LOG_ROOT_REL:-logs}"

RECIPE="${RECIPE:-al_only}"
EPOCHS="${EPOCHS:-5}"
ROUNDS="${ROUNDS:-1000}"
BUDGET="${BUDGET:-50}"
ON_EXIST="${ON_EXIST:-overwrite}"

SAMPLER_OVERRIDE_MODE="${SAMPLER_OVERRIDE_MODE:-group}"  # group|field

# 監視
MONITOR="${MONITOR:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
OUTPUTS_DIR_FOR_MONITOR="${OUTPUTS_DIR_FOR_MONITOR:-$TENSAKU_ROOT/outputs}"
ALERT_RSS_GB="${ALERT_RSS_GB:-}"
ALERT_RSS_GROWTH_MB_PER_MIN="${ALERT_RSS_GROWTH_MB_PER_MIN:-}"
ALERT_PYTHON_PROCS_MAX="${ALERT_PYTHON_PROCS_MAX:-}"
ALERT_DF_USED_PCT="${ALERT_DF_USED_PCT:-}"
ALERT_DU_GB="${ALERT_DU_GB:-}"
ALERT_GPU_MEM_USED_MIB="${ALERT_GPU_MEM_USED_MIB:-}"
ALERT_COOLDOWN_SEC="${ALERT_COOLDOWN_SEC:-300}"
ON_ALERT_CMD="${ON_ALERT_CMD:-}"

# 実行モード
USE_NOHUP=1
DRY_RUN=0

# QID エイリアス（追加はここ）
QID_MAP_LINES=(
  "y14_1=Y14_1-2_1_3"
  "y14_2=Y14_1-2_2_4"
  "y14_3=Y14_2-1_1_5"
  "y14_4=Y14_2-1_2_3"
  "y14_5=Y14_2-2_1_4"
  "y14_6=Y14_2-2_2_3"
)

# 実行対象デフォルト
QIDS_DEFAULT=(y14_3 y14_4 y14_5 y14_6)
SEEDS_DEFAULT=(42 43 44)
SAMPLERS_DEFAULT=(random kmeans unc_msp unc_trust kcenter u2d_kmeans d2u_kmeans)

# env による指定（スペース区切り）
QIDS_STR="${QIDS:-${QIDS_DEFAULT[*]}}"
SEEDS_STR="${SEEDS:-${SEEDS_DEFAULT[*]}}"
SAMPLERS_STR="${SAMPLERS:-${SAMPLERS_DEFAULT[*]}}"

# ---------------------------
# CLI parse（env より優先）
# ---------------------------
CLI_QIDS=""
CLI_SEEDS=""
CLI_SAMPLERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --qid)
      CLI_QIDS="$2"; shift 2
      ;;
    --qids)
      CLI_QIDS="$2"; shift 2
      ;;
    --seeds)
      CLI_SEEDS="$2"; shift 2
      ;;
    --samplers)
      CLI_SAMPLERS="$2"; shift 2
      ;;
    --recipe)
      RECIPE="$2"; shift 2
      ;;
    --epochs)
      EPOCHS="$2"; shift 2
      ;;
    --rounds)
      ROUNDS="$2"; shift 2
      ;;
    --budget)
      BUDGET="$2"; shift 2
      ;;
    --on-exist)
      ON_EXIST="$2"; shift 2
      ;;
    --sampler-override)
      SAMPLER_OVERRIDE_MODE="$2"; shift 2
      ;;
    --monitor)
      MONITOR="$2"; shift 2
      ;;
    --monitor-interval)
      MONITOR_INTERVAL="$2"; shift 2
      ;;
    --no-nohup)
      USE_NOHUP=0; shift
      ;;
    --dry-run)
      DRY_RUN=1; shift
      ;;
    --tensaku-root)
      TENSAKU_ROOT="$2"; shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"; shift 2
      ;;
    --log-root-rel)
      LOG_ROOT_REL="$2"; shift 2
      ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      echo "  try: ./run_exp_seq.sh --help" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$CLI_QIDS" ]]; then QIDS_STR="$CLI_QIDS"; fi
if [[ -n "$CLI_SEEDS" ]]; then SEEDS_STR="$CLI_SEEDS"; fi
if [[ -n "$CLI_SAMPLERS" ]]; then SAMPLERS_STR="$CLI_SAMPLERS"; fi

if [[ "$SAMPLER_OVERRIDE_MODE" != "group" && "$SAMPLER_OVERRIDE_MODE" != "field" ]]; then
  echo "[ERROR] --sampler-override must be 'group' or 'field' (got: $SAMPLER_OVERRIDE_MODE)" >&2
  exit 2
fi

# ---------------------------
# internal setup
# ---------------------------
cd "$TENSAKU_ROOT"
export PYTHONPATH="$TENSAKU_ROOT/src:${PYTHONPATH:-}"

LOG_ROOT="$TENSAKU_ROOT/$LOG_ROOT_REL"
mkdir -p "$LOG_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/run_${TS}"
mkdir -p "$RUN_DIR"

# QID map を associative array へ
declare -A QID_MAP
for kv in "${QID_MAP_LINES[@]}"; do
  key="${kv%%=*}"
  val="${kv#*=}"
  QID_MAP["$key"]="$val"
done

# 監視スクリプトの存在確認（MONITOR=1 のとき）
MONITOR_PY="$TENSAKU_ROOT/tools/monitor_tensaku.py"
if [[ "$MONITOR" == "1" ]]; then
  if [[ ! -f "$MONITOR_PY" ]]; then
    echo "[ERROR] monitor を有効化していますが、見つかりません: tools/monitor_tensaku.py" >&2
    echo "        MONITOR=0 ./run_exp_seq.sh で monitor無し起動にしてください。" >&2
    exit 1
  fi
fi

# targets を配列化
declare -a QIDS_ARR SEEDS_ARR SAMPLERS_ARR
_split_to_array "$QIDS_STR" QIDS_ARR
_split_to_array "$SEEDS_STR" SEEDS_ARR
_split_to_array "$SAMPLERS_STR" SAMPLERS_ARR

if [[ ${#QIDS_ARR[@]} -eq 0 || ${#SEEDS_ARR[@]} -eq 0 || ${#SAMPLERS_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] empty targets: QIDS='${QIDS_STR}' SEEDS='${SEEDS_STR}' SAMPLERS='${SAMPLERS_STR}'" >&2
  exit 2
fi

# controller を作って実行（= ターミナルを閉じても継続: デフォルト）
CTRL_SH="$RUN_DIR/controller.sh"
CTRL_LOG="$RUN_DIR/controller.log"

# controller テンプレ
cat > "$CTRL_SH" <<'CTRL'
#!/usr/bin/env bash
set -euo pipefail

_now() { date '+%Y-%m-%d %H:%M:%S'; }

resolve_qid() {
  local alias="$1"
  # alias が直接 QID っぽい場合はそのまま（明示的入力の扱い）
  if [[ "$alias" == Y*"-"*"-"*"-"* ]]; then
    echo "$alias"
    return 0
  fi
  case "$alias" in
__QID_CASE_LINES__
    *) echo "" ;;
  esac
}

echo "[$(_now)] controller started"

echo "[$(_now)] targets: QIDS=__QIDS__"
echo "[$(_now)] targets: SEEDS=__SEEDS__"
echo "[$(_now)] targets: SAMPLERS=__SAMPLERS__"

display_cmd() {
  local f="$1"
  echo "--- cmd ---"
  cat "$f"
  echo "-----------"
}

# 1ジョブずつ実行
for alias in __QIDS__; do
  qid="$(resolve_qid "$alias")"
  if [[ -z "$qid" ]]; then
    echo "[$(_now)] [ERROR] unknown qid alias: $alias" >&2
    exit 1
  fi

  for seed in __SEEDS__; do
    for sampler in __SAMPLERS__; do
      job_key="${alias}_seed${seed}_${sampler}"
      job_dir="__RUN_DIR__/jobs/${job_key}"
      mkdir -p "$job_dir"

      job_log="$job_dir/main.log"
      job_pid_file="$job_dir/pid.txt"
      job_cmd_file="$job_dir/cmd.txt"

      # main.py 実行コマンド（Hydra -m を使わない）
      cmd=(
        env HYDRA_FULL_ERROR=1
        __PYTHON_BIN__ __TENSAKU_ROOT__/src/main.py
        run.recipe=__RECIPE__
        data.qid="$qid"
        train.epochs=__EPOCHS__
        al.rounds=__ROUNDS__
        al.budget=__BUDGET__
        run.seed="$seed"
        run.on_exist=__ON_EXIST__
      )

      if [[ "__SAMPLER_MODE__" == "group" ]]; then
        cmd+=( "al/sampler=$sampler" )
      else
        cmd+=( "al.sampler.name=$sampler" )
      fi

      printf "%s\n" "${cmd[@]}" > "$job_cmd_file"
      echo "[$(_now)] start job: $job_key"
      display_cmd "$job_cmd_file"

      if [[ "__DRY_RUN__" == "1" ]]; then
        echo "[$(_now)] dry-run: skip execution"
        continue
      fi

      ("${cmd[@]}") > "$job_log" 2>&1 &
      pid=$!
      echo "$pid" > "$job_pid_file"

      # monitor
      mon_pid=""
      if [[ "__MONITOR__" == "1" ]]; then
        mon_tsv="$job_dir/monitor.tsv"
        mon_evt="$job_dir/monitor.events.log"
        mon_alert="$job_dir/monitor.alert.log"

        mon_args=(
          __PYTHON_BIN__ __TENSAKU_ROOT__/tools/monitor_tensaku.py
          --pid "$pid"
          --interval __MONITOR_INTERVAL__
          --out-tsv "$mon_tsv"
          --event-out "$mon_evt"
          --alert-log "$mon_alert"
          --alert-cooldown-sec __ALERT_COOLDOWN_SEC__
          --outputs-dir "__OUTPUTS_DIR_FOR_MONITOR__"
        )

        [[ -n "__ALERT_RSS_GB__" ]] && mon_args+=(--alert-rss-gb "__ALERT_RSS_GB__")
        [[ -n "__ALERT_RSS_GROWTH_MB_PER_MIN__" ]] && mon_args+=(--alert-rss-growth-mb-per-min "__ALERT_RSS_GROWTH_MB_PER_MIN__")
        [[ -n "__ALERT_PYTHON_PROCS_MAX__" ]] && mon_args+=(--alert-python-procs-max "__ALERT_PYTHON_PROCS_MAX__")
        [[ -n "__ALERT_DF_USED_PCT__" ]] && mon_args+=(--alert-df-used-pct "__ALERT_DF_USED_PCT__")
        [[ -n "__ALERT_DU_GB__" ]] && mon_args+=(--alert-du-gb "__ALERT_DU_GB__")
        [[ -n "__ALERT_GPU_MEM_USED_MIB__" ]] && mon_args+=(--alert-gpu-mem-used-mib "__ALERT_GPU_MEM_USED_MIB__")
        [[ -n "__ON_ALERT_CMD__" ]] && mon_args+=(--on-alert-cmd "__ON_ALERT_CMD__")

        ("${mon_args[@]}") > "$job_dir/monitor.stdout.log" 2>&1 &
        mon_pid=$!
        echo "$mon_pid" > "$job_dir/monitor_pid.txt"
      fi

      wait "$pid" || true
      echo "[$(_now)] end job: $job_key (pid=$pid)"

      if [[ -n "$mon_pid" ]]; then
        wait "$mon_pid" || true
      fi

      # 次ジョブ前に軽くスリープ（ログ追記やI/Oの落ち着き）
      sleep 2
    done
  done

done

echo "[$(_now)] controller finished"
CTRL

# QID case 行を生成
case_lines=()
for kv in "${QID_MAP_LINES[@]}"; do
  a="${kv%%=*}"; q="${kv#*=}"
  case_lines+=("    ${a}) echo '${q}' ;;")
done
case_block="$(printf "%s\n" "${case_lines[@]}")"

# targets を controller 用の文字列に
QIDS_ESC="${QIDS_ARR[*]}"
SEEDS_ESC="${SEEDS_ARR[*]}"
SAMPLERS_ESC="${SAMPLERS_ARR[*]}"

# sed 置換
sed -i \
  -e "s#__RUN_DIR__#${RUN_DIR}#g" \
  -e "s#__TENSAKU_ROOT__#${TENSAKU_ROOT}#g" \
  -e "s#__PYTHON_BIN__#${PYTHON_BIN}#g" \
  -e "s#__RECIPE__#${RECIPE}#g" \
  -e "s#__EPOCHS__#${EPOCHS}#g" \
  -e "s#__ROUNDS__#${ROUNDS}#g" \
  -e "s#__BUDGET__#${BUDGET}#g" \
  -e "s#__ON_EXIST__#${ON_EXIST}#g" \
  -e "s#__MONITOR__#${MONITOR}#g" \
  -e "s#__MONITOR_INTERVAL__#${MONITOR_INTERVAL}#g" \
  -e "s#__OUTPUTS_DIR_FOR_MONITOR__#${OUTPUTS_DIR_FOR_MONITOR}#g" \
  -e "s#__ALERT_COOLDOWN_SEC__#${ALERT_COOLDOWN_SEC}#g" \
  -e "s#__ALERT_RSS_GB__#${ALERT_RSS_GB}#g" \
  -e "s#__ALERT_RSS_GROWTH_MB_PER_MIN__#${ALERT_RSS_GROWTH_MB_PER_MIN}#g" \
  -e "s#__ALERT_PYTHON_PROCS_MAX__#${ALERT_PYTHON_PROCS_MAX}#g" \
  -e "s#__ALERT_DF_USED_PCT__#${ALERT_DF_USED_PCT}#g" \
  -e "s#__ALERT_DU_GB__#${ALERT_DU_GB}#g" \
  -e "s#__ALERT_GPU_MEM_USED_MIB__#${ALERT_GPU_MEM_USED_MIB}#g" \
  -e "s#__ON_ALERT_CMD__#${ON_ALERT_CMD}#g" \
  -e "s#__QIDS__#${QIDS_ESC}#g" \
  -e "s#__SEEDS__#${SEEDS_ESC}#g" \
  -e "s#__SAMPLERS__#${SAMPLERS_ESC}#g" \
  -e "s#__SAMPLER_MODE__#${SAMPLER_OVERRIDE_MODE}#g" \
  -e "s#__DRY_RUN__#${DRY_RUN}#g" \
  "$CTRL_SH"

# __QID_CASE_LINES__ を実際の case 行に置換（改行を含むので perl）
perl -0777 -i -pe "s#__QID_CASE_LINES__#${case_block//\n/\\n}#g" "$CTRL_SH"

chmod +x "$CTRL_SH"

# 起動
if [[ "$USE_NOHUP" == "1" ]]; then
  nohup "$CTRL_SH" > "$CTRL_LOG" 2>&1 &
  CTRL_PID=$!
else
  # foreground
  "$CTRL_SH" 2>&1 | tee "$CTRL_LOG"
  CTRL_PID=""
fi

# 親シェルから取り出せるように保存（nohup のときのみ PID がある）
printf "%s\n" "${CTRL_PID:-}" > "$LOG_ROOT/last_pid.txt"
printf "%s\n" "$CTRL_LOG" > "$LOG_ROOT/last_log_path.txt"
printf "%s\n" "$RUN_DIR" > "$LOG_ROOT/last_run_dir.txt"

# run_dir 側にも停止情報を保存（last_* は次回実行で上書きされ得るため）
printf "%s\n" "${CTRL_PID:-}" > "$RUN_DIR/controller.pid"
printf "%s\n" "${CTRL_PGID:-}" > "$RUN_DIR/controller.pgid"

# 停止手順をファイルに書き出す（「後から確実に止める」ため）
STOP_TXT="$RUN_DIR/STOP.txt"
STOP_SH="$RUN_DIR/stop.sh"
if [[ -n "${CTRL_PID:-}" ]]; then
  cat > "$STOP_TXT" <<STOPT
Stop this run (preferred):
  run_dir: $RUN_DIR
  controller pid : ${CTRL_PID}
  controller pgid: ${CTRL_PGID:-$CTRL_PID}

1) graceful stop (TERM):
  kill -TERM -- -${CTRL_PGID:-$CTRL_PID}

2) wait and force stop (KILL) if needed:
  sleep 2
  kill -KILL -- -${CTRL_PGID:-$CTRL_PID}

3) confirm:
  ps -o pid,pgid,ppid,stat,cmd -g ${CTRL_PGID:-$CTRL_PID}
STOPT

  cat > "$STOP_SH" <<'STOPSH'
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
PGID="$(cat "$RUN_DIR/controller.pgid" 2>/dev/null || true)"
PID="$(cat "$RUN_DIR/controller.pid" 2>/dev/null || true)"

if [[ -z "$PGID" ]]; then
  echo "[ERROR] PGID not found: $RUN_DIR/controller.pgid" >&2
  exit 1
fi

echo "[stop] run_dir=$RUN_DIR pid=${PID:-?} pgid=$PGID"
echo "[stop] sending TERM to process group..."
kill -TERM -- "-$PGID" 2>/dev/null || true
sleep 2
if ps -g "$PGID" >/dev/null 2>&1; then
  echo "[stop] still alive -> sending KILL to process group..."
  kill -KILL -- "-$PGID" 2>/dev/null || true
fi

echo "[stop] remaining (if any):"
ps -o pid,pgid,ppid,stat,cmd -g "$PGID" 2>/dev/null || echo "(none)"
STOPSH
  chmod +x "$STOP_SH"

  # 利便性: last_stop も残す
  printf "%s\n" "$STOP_SH" > "$LOG_ROOT/last_stop_sh.txt"
  printf "%s\n" "$STOP_TXT" > "$LOG_ROOT/last_stop_txt.txt"
else
  cat > "$STOP_TXT" <<STOPT
This run was started in foreground (no nohup/setsid).
Stop it from the terminal that started it (Ctrl-C), or kill the displayed PID if available.
STOPT
  printf "%s\n" "$STOP_TXT" > "$LOG_ROOT/last_stop_txt.txt"
fi

cat <<EOF
[$(_now)] started controller.
  run_dir: $RUN_DIR
  controller: ${CTRL_PID:+pid=$CTRL_PID }log=$CTRL_LOG
saved:
  $LOG_ROOT/last_pid.txt
  $LOG_ROOT/last_log_path.txt
  $LOG_ROOT/last_run_dir.txt

log follow (controller):
  tail -f "$CTRL_LOG"

jobs live check (nohup only):
  ps -fp "$(cat $LOG_ROOT/last_pid.txt)"

stop this run (nohup only):
  bash "$RUN_DIR/stop.sh"

(or view instructions):
  cat "$RUN_DIR/STOP.txt"

stop instructions saved:
  $RUN_DIR/STOP.txt
  $RUN_DIR/stop.sh
  $LOG_ROOT/last_stop_txt.txt
  $LOG_ROOT/last_stop_sh.txt
EOF
