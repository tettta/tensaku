# /home/esakit25/work/tensaku/tools/summarize_memlog.py
# -*- coding: utf-8 -*-
"""
Tensaku mem.log / monitor.tsv 集計スクリプト（調査用・split詳細対応版）

目的
- outputs 配下の mem.log（tensaku.mem）から、roundごとのRSS推移・増分（ΔRSS）、
  イベント別のRSS統計（min/median/max）を自動集計する。
- infer の split 単位イベント（infer_split_* / infer_df_*）があれば、
  「どの split / どの段階（predict/save/df_build）でRSSが増えたか」を分解して出力する。
- 同一フォルダに monitor.tsv があれば併せてサマリを作る（任意）。

使い方（例）
  python tools/summarize_memlog.py \
    --mem-log outputs/.../mem.log \
    --out-dir outputs/.../ops

出力
- round_end.tsv              : round_end の RSS/ΔRSS 一覧
- round_events_wide.tsv      : round_start/after_train/after_infer/round_end のRSSを横持ち
- infer_split_wide.tsv       : (round, split) ごとの infer_split_* RSS と増分
- infer_df_wide.tsv          : (round, split) ごとの infer_df_* RSS と増分
- event_stats.tsv            : event別のRSS統計（min/median/max）
- sizes.tsv                  : logits_mb / embs_mb のイベント別・split別の集計
- report.txt                 : 人間向けの短い所見（増え続け判定・どこで増えているか）
- monitor_summary.tsv        : （monitor.tsvがある場合）RSS/du/dfの要約

注意
- mem.log に本文が混入している場合でも落ちないよう、[tensaku.mem] 行以外は無視する。
- 解析はテキストストリーミングで行い、巨大ファイルでもメモリ使用を抑える。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------
# 低レベル: 1行パース
# -----------------------

_TS_RE = re.compile(r'^\[(?P<ts>[\d\-:, ]+)\]\[(?P<logger>[^\]]+)\]\[(?P<level>[A-Z]+)\]\s*-\s*(?P<msg>.*)$')

# key=value のゆるい抽出（"..." も対応）
_KV_RE = re.compile(r'(?P<k>[A-Za-z_][A-Za-z0-9_]*)=(?P<v>"[^"]*"|\S+)')


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _parse_kv(msg: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _KV_RE.finditer(msg):
        k = m.group("k")
        v = _strip_quotes(m.group("v"))
        out[k] = v
    return out


def _to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class MemRow:
    ts: str
    event: str
    r: Optional[int]
    rss_kb: Optional[int]
    hwm_kb: Optional[int]
    split: Optional[str]
    logits_mb: Optional[float]
    embs_mb: Optional[float]
    raw_msg: str


def parse_mem_line(line: str) -> Optional[MemRow]:
    """
    mem.log の1行をパースする。

    現行（推奨）: tensaku.mem が JSON 文字列を出力する形式
      [ts][tensaku.mem][DEBUG] - {"event": "...", "rss_kb": ..., "extra": {...}}

    旧形式（後方互換）: "[mem] event=... r=... rss_kb=..." の key=value 形式
      [ts][tensaku.mem][DEBUG] - [mem] event=round_end r=3 rss_kb=123 ...

    どちらにもマッチしない場合は None を返す。
    """
    m = _TS_RE.match(line.rstrip("\n"))
    if not m:
        return None
    ts = m.group("ts")
    logger = m.group("logger")
    msg = m.group("msg").strip()
    if logger != "tensaku.mem":
        return None

    # 1) JSON 形式（現行）
    if msg.startswith("{") and msg.endswith("}"):
        try:
            d = json.loads(msg)
        except Exception:
            d = None
        if isinstance(d, dict):
            event = str(d.get("event") or "")
            if not event:
                return None

            extra = d.get("extra") if isinstance(d.get("extra"), dict) else {}

            # round index はトップレベル r または extra.round/extra.r を優先
            r = _to_int(d.get("r"))
            if r is None:
                r = _to_int(extra.get("round"))
            if r is None:
                r = _to_int(extra.get("r"))

            rss_kb = _to_int(d.get("rss_kb"))
            hwm_kb = _to_int(d.get("hwm_kb"))

            # split はトップレベル split または extra.split / extra.split_name
            split = d.get("split")
            if split is None:
                split = extra.get("split")
            if split is None:
                split = extra.get("split_name")
            if split is not None:
                split = str(split)

            logits_mb = _to_float(d.get("logits_mb"))
            if logits_mb is None:
                logits_mb = _to_float(extra.get("logits_mb"))

            embs_mb = _to_float(d.get("embs_mb"))
            if embs_mb is None:
                embs_mb = _to_float(extra.get("embs_mb"))

            return MemRow(
                ts=str(d.get("ts") or ts),
                event=event,
                r=r,
                rss_kb=rss_kb,
                hwm_kb=hwm_kb,
                split=split,
                logits_mb=logits_mb,
                embs_mb=embs_mb,
                raw_msg=msg,
            )

    # 2) 旧 key=value 形式（後方互換）
    if "[mem]" not in msg:
        return None

    kv = _parse_kv(msg)
    event = kv.get("event") or ""
    if not event:
        if "tag=" in msg:
            event = kv.get("tag", "")
    if not event:
        return None

    r = _to_int(kv.get("r"))
    rss_kb = _to_int(kv.get("rss_kb"))
    hwm_kb = _to_int(kv.get("hwm_kb"))
    split = kv.get("split")
    logits_mb = _to_float(kv.get("logits_mb"))
    embs_mb = _to_float(kv.get("embs_mb"))

    return MemRow(
        ts=ts,
        event=event,
        r=r,
        rss_kb=rss_kb,
        hwm_kb=hwm_kb,
        split=split,
        logits_mb=logits_mb,
        embs_mb=embs_mb,
        raw_msg=msg,
    )


def kb_to_gb(kb: int) -> float:
    return kb / (1024.0 * 1024.0)


def safe_median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return statistics.median(xs)


def safe_min(xs: List[float]) -> float:
    return min(xs) if xs else float("nan")


def safe_max(xs: List[float]) -> float:
    return max(xs) if xs else float("nan")


def read_mem_rows(mem_log: Path) -> List[MemRow]:
    rows: List[MemRow] = []
    with mem_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            row = parse_mem_line(line)
            if row is None:
                continue
            rows.append(row)
    return rows


def write_tsv(path: Path, header: List[str], rows: Iterable[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")


# -----------------------
# 集計: round_end / wide / stats
# -----------------------

def summarize_round_end(rows: List[MemRow]) -> List[List[object]]:
    xs = [(r.r, r.rss_kb, r.ts) for r in rows if r.event == "round_end" and r.r is not None and r.rss_kb is not None]
    xs.sort(key=lambda t: t[0])  # by round
    out: List[List[object]] = []
    prev_kb: Optional[int] = None
    for rr, kb, ts in xs:
        gb = kb_to_gb(kb)
        d_gb = 0.0 if prev_kb is None else kb_to_gb(kb - prev_kb)
        out.append([rr, ts, f"{gb:.3f}", f"{d_gb:+.3f}"])
        prev_kb = kb
    return out


def summarize_round_events_wide(rows: List[MemRow], events: List[str]) -> List[List[object]]:
    # round -> event -> rss_gb
    d: Dict[int, Dict[str, float]] = {}
    t: Dict[int, Dict[str, str]] = {}
    for r in rows:
        if r.r is None or r.rss_kb is None:
            continue
        if r.event not in events:
            continue
        d.setdefault(r.r, {})[r.event] = kb_to_gb(r.rss_kb)
        t.setdefault(r.r, {})[r.event] = r.ts

    out: List[List[object]] = []
    for rr in sorted(d.keys()):
        row = [rr]
        ts = t.get(rr, {}).get("round_end") or next(iter(t.get(rr, {}).values()), "")
        row.append(ts)
        for ev in events:
            v = d.get(rr, {}).get(ev)
            row.append("" if v is None else f"{v:.3f}")
        out.append(row)
    return out


def summarize_event_stats(rows: List[MemRow]) -> List[List[object]]:
    by: Dict[str, List[float]] = {}
    for r in rows:
        if r.rss_kb is None:
            continue
        by.setdefault(r.event, []).append(kb_to_gb(r.rss_kb))
    out: List[List[object]] = []
    for ev in sorted(by.keys()):
        xs = by[ev]
        out.append([ev, len(xs), f"{safe_min(xs):.3f}", f"{safe_median(xs):.3f}", f"{safe_max(xs):.3f}"])
    return out


def summarize_sizes(rows: List[MemRow]) -> List[List[object]]:
    # event, split -> list of sizes
    by_logits: Dict[Tuple[str, str], List[float]] = {}
    by_embs: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        sp = r.split or ""
        if r.logits_mb is not None:
            by_logits.setdefault((r.event, sp), []).append(r.logits_mb)
        if r.embs_mb is not None:
            by_embs.setdefault((r.event, sp), []).append(r.embs_mb)

    keys = sorted(set(list(by_logits.keys()) + list(by_embs.keys())))
    out: List[List[object]] = []
    for (ev, sp) in keys:
        lg = by_logits.get((ev, sp), [])
        em = by_embs.get((ev, sp), [])
        out.append([
            ev, sp,
            len(lg), f"{safe_median(lg):.3f}", f"{safe_max(lg):.3f}",
            len(em), f"{safe_median(em):.3f}", f"{safe_max(em):.3f}",
        ])
    return out


# -----------------------
# 集計: infer split / df（詳細）
# -----------------------

def _collect_rss_by_round_split(rows: List[MemRow], events: List[str]) -> Tuple[Dict[Tuple[int, str], Dict[str, float]], Dict[Tuple[int, str], Dict[str, str]]]:
    d: Dict[Tuple[int, str], Dict[str, float]] = {}
    t: Dict[Tuple[int, str], Dict[str, str]] = {}
    for r in rows:
        if r.r is None or r.rss_kb is None:
            continue
        if r.event not in events:
            continue
        sp = r.split or ""
        key = (r.r, sp)
        d.setdefault(key, {})[r.event] = kb_to_gb(r.rss_kb)
        t.setdefault(key, {})[r.event] = r.ts
    return d, t


def summarize_infer_split_wide(rows: List[MemRow]) -> List[List[object]]:
    events = ["infer_split_start", "infer_split_after_predict", "infer_split_after_save"]
    d, t = _collect_rss_by_round_split(rows, events)
    out: List[List[object]] = []
    for (rr, sp) in sorted(d.keys(), key=lambda x: (x[0], x[1])):
        ts = t.get((rr, sp), {}).get("infer_split_after_save") or t.get((rr, sp), {}).get("infer_split_after_predict") or t.get((rr, sp), {}).get("infer_split_start") or ""
        v0 = d[(rr, sp)].get("infer_split_start")
        v1 = d[(rr, sp)].get("infer_split_after_predict")
        v2 = d[(rr, sp)].get("infer_split_after_save")
        dp = "" if (v0 is None or v1 is None) else f"{(v1 - v0):+.3f}"
        ds = "" if (v1 is None or v2 is None) else f"{(v2 - v1):+.3f}"
        dt = "" if (v0 is None or v2 is None) else f"{(v2 - v0):+.3f}"
        out.append([
            rr, ts, sp,
            "" if v0 is None else f"{v0:.3f}",
            "" if v1 is None else f"{v1:.3f}",
            "" if v2 is None else f"{v2:.3f}",
            dp, ds, dt,
        ])
    return out


def summarize_infer_df_wide(rows: List[MemRow]) -> List[List[object]]:
    events = ["infer_df_build_start", "infer_df_split_done", "infer_df_build_end"]
    d, t = _collect_rss_by_round_split(rows, events)
    out: List[List[object]] = []
    for (rr, sp) in sorted(d.keys(), key=lambda x: (x[0], x[1])):
        ts = t.get((rr, sp), {}).get("infer_df_build_end") or t.get((rr, sp), {}).get("infer_df_split_done") or t.get((rr, sp), {}).get("infer_df_build_start") or ""
        v0 = d[(rr, sp)].get("infer_df_build_start")
        v1 = d[(rr, sp)].get("infer_df_split_done")
        v2 = d[(rr, sp)].get("infer_df_build_end")
        dp = "" if (v0 is None or v1 is None) else f"{(v1 - v0):+.3f}"
        ds = "" if (v1 is None or v2 is None) else f"{(v2 - v1):+.3f}"
        dt = "" if (v0 is None or v2 is None) else f"{(v2 - v0):+.3f}"
        out.append([
            rr, ts, sp,
            "" if v0 is None else f"{v0:.3f}",
            "" if v1 is None else f"{v1:.3f}",
            "" if v2 is None else f"{v2:.3f}",
            dp, ds, dt,
        ])
    return out


def _median_from_col(rows: List[List[object]], idx: int) -> float:
    xs: List[float] = []
    for r in rows:
        try:
            s = r[idx]
            if s is None or s == "":
                continue
            xs.append(float(s))
        except Exception:
            continue
    return safe_median(xs)


def _topk_by_abs_delta(rows: List[List[object]], split_col: int, total_delta_col: int, k: int = 5) -> List[Tuple[str, float]]:
    """
    (split, median(|delta|)) の上位Kを返す。
    rows は (round, ts, split, ..., delta_total) の形式を想定。
    """
    by: Dict[str, List[float]] = {}
    for r in rows:
        sp = str(r[split_col])
        try:
            v = float(r[total_delta_col])
        except Exception:
            continue
        by.setdefault(sp, []).append(abs(v))
    scored = [(sp, safe_median(vs)) for sp, vs in by.items() if vs]
    scored.sort(key=lambda x: (x[1]), reverse=True)
    return scored[:k]


def write_report(out_path: Path, round_end_rows: List[List[object]], rows: List[MemRow], infer_split_wide: List[List[object]], infer_df_wide: List[List[object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # round_end の ΔRSS から簡易判定
    deltas = []
    for r in round_end_rows:
        try:
            deltas.append(float(r[3]))
        except Exception:
            pass

    last10 = deltas[-10:] if deltas else []
    avg10 = sum(last10) / len(last10) if last10 else 0.0
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)

    rss_all = [kb_to_gb(r.rss_kb) for r in rows if r.rss_kb is not None]
    max_rss = max(rss_all) if rss_all else float("nan")

    # infer split の増分（中央値）
    infer_pred_med = _median_from_col(infer_split_wide, idx=6)  # delta_pred_gb
    infer_save_med = _median_from_col(infer_split_wide, idx=7)  # delta_save_gb
    infer_total_med = _median_from_col(infer_split_wide, idx=8)  # delta_total_gb

    # infer df の増分（中央値）
    df_split_med = _median_from_col(infer_df_wide, idx=6)
    df_post_med = _median_from_col(infer_df_wide, idx=7)
    df_total_med = _median_from_col(infer_df_wide, idx=8)

    # split別の“増えやすさ”上位
    top_infer = _topk_by_abs_delta(infer_split_wide, split_col=2, total_delta_col=8, k=8)
    top_df = _topk_by_abs_delta(infer_df_wide, split_col=2, total_delta_col=8, k=8)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Tensaku mem.log summary (split detailed)\n")
        f.write(f"- max_rss_gb: {max_rss:.3f}\n")
        f.write(f"- round_end_deltas: n={len(deltas)} pos={pos} neg={neg}\n")
        f.write(f"- avg_delta_last10_gb: {avg10:+.3f}\n")
        if deltas:
            f.write("- last10_delta_gb: " + ", ".join(f"{d:+.3f}" for d in last10) + "\n")

        if infer_split_wide:
            f.write("\nInfer split deltas (median):\n")
            f.write(f"- infer_split_delta_pred_gb_median: {infer_pred_med:+.3f}\n")
            f.write(f"- infer_split_delta_save_gb_median: {infer_save_med:+.3f}\n")
            f.write(f"- infer_split_delta_total_gb_median: {infer_total_med:+.3f}\n")
            if top_infer:
                f.write("- infer_split_top_by_abs_total_delta_median:\n")
                for sp, v in top_infer:
                    f.write(f"  - split={sp or '(none)'} abs_delta_total_gb_median={v:.3f}\n")

        if infer_df_wide:
            f.write("\nInfer df_build deltas (median):\n")
            f.write(f"- infer_df_delta_split_done_gb_median: {df_split_med:+.3f}\n")
            f.write(f"- infer_df_delta_post_split_gb_median: {df_post_med:+.3f}\n")
            f.write(f"- infer_df_delta_total_gb_median: {df_total_med:+.3f}\n")
            if top_df:
                f.write("- infer_df_top_by_abs_total_delta_median:\n")
                for sp, v in top_df:
                    f.write(f"  - split={sp or '(none)'} abs_delta_total_gb_median={v:.3f}\n")

        f.write("\nInterpretation guide:\n")
        f.write("- infer_split_delta_pred が大きい: predict（モデル推論）がピークを更新している可能性。\n")
        f.write("- infer_split_delta_save が大きい: npy保存や後処理で一時領域が膨らんでいる可能性。\n")
        f.write("- infer_df_* が大きい: DataFrame生成・結合がメモリを押し上げている可能性。\n")
        f.write("- round_end delta が持続的に + なら「保持/蓄積」か「ピーク更新＋返却されない」を疑う。\n")


# -----------------------
# monitor.tsv（任意）
# -----------------------

def summarize_monitor_tsv(monitor_path: Path) -> List[List[object]]:
    """
    monitor.tsv の簡易サマリ。ヘッダ:
      ts pid_alive state threads VmRSS_kB VmHWM_kB python_proc_count df_used_pct du_gb
    """
    if not monitor_path.exists():
        return []

    rows = []
    with monitor_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split("\t")
        idx = {k: i for i, k in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            try:
                rss_kb = int(parts[idx["VmRSS_kB"]])
            except Exception:
                continue
            ts = parts[idx["ts"]]
            df_used = parts[idx.get("df_used_pct", -1)] if "df_used_pct" in idx else ""
            du_gb = parts[idx.get("du_gb", -1)] if "du_gb" in idx else ""
            rows.append((ts, rss_kb, df_used, du_gb))

    if not rows:
        return []

    rss_gb = [kb_to_gb(kb) for _, kb, _, _ in rows]
    out = [
        ["metric", "value"],
        ["n_rows", str(len(rows))],
        ["rss_gb_min", f"{min(rss_gb):.3f}"],
        ["rss_gb_median", f"{statistics.median(rss_gb):.3f}"],
        ["rss_gb_max", f"{max(rss_gb):.3f}"],
        ["rss_gb_last", f"{rss_gb[-1]:.3f}"],
        ["first_ts", rows[0][0]],
        ["last_ts", rows[-1][0]],
        ["last_df_used_pct", rows[-1][2]],
        ["last_du_gb", rows[-1][3]],
    ]
    return out


# -----------------------
# CLI
# -----------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-log", required=True, help="mem.log のパス（outputs配下）")
    ap.add_argument("--out-dir", default=None, help="出力先ディレクトリ（既定: mem.log と同じディレクトリ/ops）")
    ap.add_argument("--monitor-tsv", default=None, help="monitor.tsv のパス（任意）")
    args = ap.parse_args()

    mem_log = Path(args.mem_log)
    if not mem_log.exists():
        raise SystemExit(f"[error] mem.log not found: {mem_log}")

    out_dir = Path(args.out_dir) if args.out_dir else (mem_log.parent / "ops")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_mem_rows(mem_log)

    # round_end
    round_end = summarize_round_end(rows)
    write_tsv(out_dir / "round_end.tsv", ["round", "ts", "rss_gb", "delta_gb"], round_end)

    # wide events（従来）
    base_events = ["round_start", "round_after_train", "round_after_infer", "round_end"]
    wide = summarize_round_events_wide(rows, base_events)
    write_tsv(out_dir / "round_events_wide.tsv", ["round", "ts"] + [f"{e}_gb" for e in base_events], wide)

    # infer split / df（新規）
    infer_split_wide = summarize_infer_split_wide(rows)
    if infer_split_wide:
        write_tsv(
            out_dir / "infer_split_wide.tsv",
            ["round", "ts", "split",
             "infer_split_start_gb", "infer_split_after_predict_gb", "infer_split_after_save_gb",
             "delta_pred_gb", "delta_save_gb", "delta_total_gb"],
            infer_split_wide,
        )

    infer_df_wide = summarize_infer_df_wide(rows)
    if infer_df_wide:
        write_tsv(
            out_dir / "infer_df_wide.tsv",
            ["round", "ts", "split",
             "infer_df_build_start_gb", "infer_df_split_done_gb", "infer_df_build_end_gb",
             "delta_split_done_gb", "delta_post_split_gb", "delta_total_gb"],
            infer_df_wide,
        )

    # stats
    stats = summarize_event_stats(rows)
    write_tsv(out_dir / "event_stats.tsv", ["event", "n", "min_gb", "median_gb", "max_gb"], stats)

    # sizes
    sizes = summarize_sizes(rows)
    write_tsv(
        out_dir / "sizes.tsv",
        ["event", "split", "n_logits", "logits_mb_median", "logits_mb_max", "n_embs", "embs_mb_median", "embs_mb_max"],
        sizes,
    )

    # report
    write_report(out_dir / "report.txt", round_end, rows, infer_split_wide, infer_df_wide)

    # monitor
    monitor_path = Path(args.monitor_tsv) if args.monitor_tsv else (mem_log.parent / "monitor.tsv")
    mon = summarize_monitor_tsv(monitor_path)
    if mon:
        write_tsv(out_dir / "monitor_summary.tsv", mon[0], mon[1:])

    print(f"[ok] wrote summaries to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
