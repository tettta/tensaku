# /home/esakit25/work/tensaku/tools/monitor_tensaku.py
# -*- coding: utf-8 -*-
"""Tensaku 実験プロセス監視（最小・堅牢版）

目的:
- /proc/<pid>/status から VmRSS/VmHWM/Threads/State を定期取得し、TSVで時系列保存する
- pythonプロセス数（Hydra増殖検知）と上位RSSを補助情報として記録する
- nvidia-smi があれば GPU util / VRAM / compute apps を補助情報として記録する（無ければ自動スキップ）
- df/du で outputs のディスク使用状況を記録する
- 閾値超過時は ALERT.log / events.log に記録し、必要なら任意コマンド（例: kill -STOP {pid}）を実行する

設計:
- 依存は標準ライブラリのみ（環境差で壊れにくい）
- 監視対象pidが消えたら終了（監視側が永遠に残らない）
- 監視に失敗しても例外で落とさず、events.log に記録して継続（運用優先）

出力:
- out_tsv: 1行/interval のTSV（ヘッダあり）
- event_out: INFO/ALERTなどのイベントログ（人間が読む用）
- alert_log: アラート時の詳細（ps上位/GPU一覧などを含む）
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: str, timeout: int = 10) -> Tuple[int, str]:
    """コマンドを実行し、(returncode, stdout+stderr) を返す。失敗しても例外は投げない。"""
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        return p.returncode, p.stdout
    except Exception as e:
        return 999, f"[monitor] command failed: {cmd}\n{e}\n"


def _read_proc_status(pid: int) -> Dict[str, str]:
    path = Path(f"/proc/{pid}/status")
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_kb_field(v: str) -> Optional[int]:
    # 例: "77545676 kB"
    m = re.search(r"(\d+)\s*kB", v)
    return int(m.group(1)) if m else None


def _safe_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _count_python_processes() -> Optional[int]:
    rc, out = _run("ps -C python -o pid= | wc -l")
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except Exception:
        return None


def _top_python_rss_lines() -> str:
    # rss降順で上位を出す（短め）
    rc, out = _run("ps -C python -o pid,ppid,rss,vsz,etime,cmd --sort=-rss | head -n 15")
    return out.strip()


def _gpu_summary() -> str:
    rc, _ = _run("command -v nvidia-smi")
    if rc != 0:
        return ""
    rc1, a = _run(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    )
    rc2, b = _run(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"
    )
    parts = []
    if rc1 == 0 and a.strip():
        parts.append("[gpu]\n" + a.strip())
    if rc2 == 0 and b.strip():
        parts.append("[gpu_apps]\n" + b.strip())
    return "\n".join(parts).strip()


def _df_used_percent(path: str) -> Optional[int]:
    rc, out = _run(f"df -P {shlex.quote(path)} | tail -n 1")
    if rc != 0:
        return None
    cols = out.split()
    if len(cols) < 5:
        return None
    m = re.match(r"(\d+)%", cols[4])
    return int(m.group(1)) if m else None


def _du_gb(path: str) -> Optional[float]:
    rc, out = _run(f"du -s {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'")
    if rc != 0:
        return None
    try:
        kb = float(out.strip())
        return kb / (1024.0 * 1024.0)
    except Exception:
        return None


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def _tail_file(path: str, n_lines: int = 80) -> str:
    """ファイル末尾を取得（存在しない/読めない場合は空文字）。"""
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.exists():
            return ""
        # 大きいログでも軽く読むため末尾だけ
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:]).strip()
    except Exception:
        return ""


def _format_on_alert(cmd_tpl: str, pid: int) -> str:
    return cmd_tpl.replace("{pid}", str(pid))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--log", default="", help="main.log など（任意）。アラート時に末尾を添付するために使用")
    ap.add_argument("--event-out", required=True)
    ap.add_argument("--alert-log", required=True)
    ap.add_argument("--alert-cooldown-sec", type=int, default=300)

    # 閾値（指定されたものだけ有効）
    ap.add_argument("--alert-rss-gb", type=float, default=None)
    ap.add_argument("--alert-df-used-pct", type=int, default=None)
    ap.add_argument("--alert-rss-growth-mb-per-min", type=float, default=None)
    ap.add_argument("--alert-python-procs-max", type=int, default=None)
    ap.add_argument("--alert-du-gb", type=float, default=None)
    ap.add_argument("--alert-gpu-mem-used-mib", type=int, default=None)
    ap.add_argument("--on-alert-cmd", type=str, default=None)

    args = ap.parse_args()

    pid = args.pid
    out_tsv = Path(args.out_tsv)
    event_out = Path(args.event_out)
    alert_log = Path(args.alert_log)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    is_new = not out_tsv.exists()
    with out_tsv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = [
            "ts",
            "pid_alive",
            "state",
            "threads",
            "VmRSS_kB",
            "VmHWM_kB",
            "python_proc_count",
            "df_used_pct",
            "du_gb",
        ]
        if is_new:
            writer.writerow(header)
            f.flush()

        last_rss_kb: Optional[int] = None
        last_ts: Optional[float] = None
        last_alert_ts: float = 0.0

        while True:
            t = time.time()
            ts = _now_iso()

            st = _read_proc_status(pid)
            alive = 1 if st else 0
            state = st.get("State", "") if st else ""
            threads = _safe_int(st.get("Threads")) if st else None
            rss_kb = _parse_kb_field(st.get("VmRSS", "")) if st else None
            hwm_kb = _parse_kb_field(st.get("VmHWM", "")) if st else None

            py_cnt = _count_python_processes()
            df_pct = _df_used_percent(args.outputs_dir)
            du_gb = _du_gb(args.outputs_dir)

            writer.writerow([ts, alive, state, threads, rss_kb, hwm_kb, py_cnt, df_pct, du_gb])
            f.flush()

            if alive == 0:
                _append_line(event_out, f"{ts}\tINFO\tpid {pid} is not alive -> stop monitor")
                break

            alerts = []

            if args.alert_rss_gb is not None and rss_kb is not None:
                thr_kb = int(args.alert_rss_gb * 1024 * 1024)
                if rss_kb >= thr_kb:
                    alerts.append(f"RSS_GB_EXCEEDED rss_gb={rss_kb/1024/1024:.2f} thr={args.alert_rss_gb}")

            if (
                args.alert_rss_growth_mb_per_min is not None
                and rss_kb is not None
                and last_rss_kb is not None
                and last_ts is not None
            ):
                dt_sec = max(1.0, t - last_ts)
                drss_kb = rss_kb - last_rss_kb
                mb_per_min = (drss_kb / 1024.0) / (dt_sec / 60.0)
                if mb_per_min >= args.alert_rss_growth_mb_per_min:
                    alerts.append(
                        f"RSS_GROWTH_HIGH mb_per_min={mb_per_min:.1f} thr={args.alert_rss_growth_mb_per_min}"
                    )

            if args.alert_python_procs_max is not None and py_cnt is not None:
                if py_cnt >= args.alert_python_procs_max:
                    alerts.append(f"PY_PROCS_TOO_MANY count={py_cnt} thr={args.alert_python_procs_max}")

            if args.alert_df_used_pct is not None and df_pct is not None:
                if df_pct >= args.alert_df_used_pct:
                    alerts.append(f"DF_USED_HIGH pct={df_pct} thr={args.alert_df_used_pct}")

            if args.alert_du_gb is not None and du_gb is not None:
                if du_gb >= args.alert_du_gb:
                    alerts.append(f"DU_TOO_LARGE gb={du_gb:.1f} thr={args.alert_du_gb}")

            if args.alert_gpu_mem_used_mib is not None:
                gpu_txt = _gpu_summary()
                if gpu_txt:
                    used = 0
                    for line in gpu_txt.splitlines():
                        if line.startswith("[") or not line.strip():
                            continue
                        cols = [c.strip() for c in line.split(",")]
                        if len(cols) >= 5:
                            try:
                                used += int(cols[3])
                            except Exception:
                                pass
                    if used >= args.alert_gpu_mem_used_mib:
                        alerts.append(f"GPU_MEM_USED_HIGH used_mib={used} thr={args.alert_gpu_mem_used_mib}")

            if alerts and (t - last_alert_ts) >= args.alert_cooldown_sec:
                last_alert_ts = t
                msg = f"{ts}\tALERT\tpid={pid}\t" + " | ".join(alerts)
                _append_line(event_out, msg)
                _append_line(alert_log, msg)
                _append_line(alert_log, _top_python_rss_lines())
                gpu_txt = _gpu_summary()
                if gpu_txt:
                    _append_line(alert_log, gpu_txt)

                # main.log 等の末尾（任意）
                log_tail = _tail_file(args.log, n_lines=120)
                if log_tail:
                    _append_line(alert_log, "[main_log_tail]\n" + log_tail)

                if args.on_alert_cmd:
                    cmd = _format_on_alert(args.on_alert_cmd, pid)
                    rc, out = _run(cmd, timeout=30)
                    _append_line(alert_log, f"{ts}\tON_ALERT_CMD\trc={rc}\tcmd={cmd}")
                    if out.strip():
                        _append_line(alert_log, out.strip())

            last_rss_kb = rss_kb
            last_ts = t
            time.sleep(max(1, args.interval))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
