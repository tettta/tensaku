# /home/esakit25/work/tensaku/scripts/gen_memo_from_modules.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@module: gen_memo_from_modules
@role: src/tensaku/*.py の先頭docstringから @tags を抽出し、docs/MEMO_TENSAKU.md を自動生成
@inputs: src/tensaku 配下の Python モジュール（module-level docstring に @module, @role, ... を含む）
@outputs: docs/MEMO_TENSAKU.md（既存があれば JST タイムスタンプ付き .bak を作成してから上書き）
@api: CLI: python scripts/gen_memo_from_modules.py [--root ROOT] [--src REL] [--out REL]
@deps: 標準ライブラリのみ（ast, re, dataclasses, pathlib, datetime, textwrap）
@config: 既定 ROOT=/home/esakit25/work/tensaku, SRC=src/tensaku, OUT=docs/MEMO_TENSAKU.md
@contracts: 
  - docstringは**先頭**に配置し、行
  頭に `@key:` で開始。複数行値は次の `@` までの**インデント行**を連結。
  - 推奨タグ: @module, @role, @inputs, @outputs, @cli, @api, @deps, @config, @contracts, @errors, @notes, @tests
  - 未記載タグは空扱いで安全にスキップ。未知タグは「その他」に列挙。
@errors:
  - srcディレクトリ未検出 → 終了コード2
  - モジュールのdocstringがパース不能でも全体生成は継続（当該モジュールは空ブロック）
@notes:
  - 旧版からの変更点: ①複数行タグ対応、②拡張タグ群（@api/@deps/@config/@contracts/@errors/@tests）を出力、③"_"始まりを先頭へ並べ替え
"""
from __future__ import annotations
import ast
import datetime as dt
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- Config (defaults; CLIで上書き可) ----------
ROOT_DEFAULT = Path("/home/esakit25/work/tensaku")
SRC_REL = Path("src/tensaku")
OUT_REL = Path("docs/MEMO_TENSAKU.md")

# ---------- Tag parsing ----------
# 例: @role: 日本語BERTで… の行を起点に、次の@key行までの**インデントされた行**を値として連結
HEAD_PAT = re.compile(r"^@(?P<key>[\w\-]+)\s*:\s*(?P<val>.*)$")

@dataclass
class ModuleInfo:
    name: str                      # e.g., tensaku.confidence
    file_rel: str                  # e.g., confidence.py
    tags: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, str] = field(default_factory=dict)   # 未知タグなど

def _read_source(py_path: Path) -> Optional[str]:
    try:
        return py_path.read_text(encoding="utf-8")
    except Exception:
        return None

def extract_docstring(src: str) -> str:
    try:
        mod = ast.parse(src)
        return ast.get_docstring(mod) or ""
    except Exception:
        return ""

def read_doc_tags(py_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    複数行タグ対応のパーサ。
    値は改行で連結（末尾トリム）。未知タグは extra に格納。
    """
    src = _read_source(py_path)
    if not src:
        return {}, {}
    doc = extract_docstring(src)
    if not doc:
        return {}, {}

    lines = doc.splitlines()
    i, n = 0, len(lines)
    tags: Dict[str, str] = {}
    extra: Dict[str, str] = {}
    known_keys = {
        "module","role","inputs","outputs","cli","api","deps","config","contracts","errors","notes","tests"
    }

    while i < n:
        line = lines[i]
        m = HEAD_PAT.match(line.strip())
        if m:
            key = m.group("key").strip().lower()
            val_lines = [m.group("val").rstrip()]
            i += 1
            # 以降、次の @key が現れるまで、**先頭が空 or インデント**の行を値として取り込む
            while i < n:
                nxt = lines[i]
                if HEAD_PAT.match(nxt.strip()):
                    break
                if (nxt.strip() == "") or (nxt.startswith(" ") or nxt.startswith("\t")):
                    val_lines.append(nxt.rstrip())
                    i += 1
                else:
                    # 非インデントの通常文（タグ記法外）はスキップして次へ
                    i += 1
            val = "\n".join(val_lines).strip()
            if key in known_keys:
                tags[key] = val
            else:
                extra[key] = val
            continue
        i += 1

    return tags, extra

def discover_modules(src_dir: Path) -> List[Path]:
    return sorted([p for p in src_dir.glob("*.py") if p.name != "__init__.py"])

def guess_display_name(p: Path) -> str:
    return p.name

def _mk_section(title: str, body: str) -> str:
    return f"- **{title}**:\n{ textwrap.indent(body.strip() if body.strip() else '（未記載）', '  ') }"

def fmt_module_block(mi: ModuleInfo) -> str:
    # 主タグ
    role = mi.tags.get("role", "（未記載）")
    inputs = mi.tags.get("inputs", "")
    outputs = mi.tags.get("outputs", "")
    cli = mi.tags.get("cli", mi.tags.get("command", "（なし）"))
    api = mi.tags.get("api", "")
    deps = mi.tags.get("deps", "")
    config = mi.tags.get("config", "")
    contracts = mi.tags.get("contracts", "")
    errors = mi.tags.get("errors", "")
    notes = mi.tags.get("notes", "")
    tests = mi.tags.get("tests", "")

    parts: List[str] = []
    parts.append(f"### {mi.name}  (_{mi.file_rel}_)")
    parts.append(f"- **役割**: {role}")
    if inputs:   parts.append(_mk_section("入力", inputs))
    if outputs:  parts.append(_mk_section("出力", outputs))
    parts.append(f"- **CLI**: {cli}")
    if api:      parts.append(_mk_section("API", api))
    if deps:     parts.append(_mk_section("依存", deps))
    if config:   parts.append(_mk_section("設定", config))
    if contracts:parts.append(_mk_section("契約", contracts))
    if errors:   parts.append(_mk_section("エラー/例外", errors))
    if tests:    parts.append(_mk_section("テスト/スモーク", tests))
    if notes:    parts.append(_mk_section("補足", notes))

    # 未知タグを「その他」にまとめる
    if mi.extra:
        kv = "\n".join(f"- @{k}: {v}" for k, v in sorted(mi.extra.items()))
        parts.append(_mk_section("その他（未定義タグ）", kv))

    return "\n".join(parts)

def load_root_from_argv() -> Dict[str, Path]:
    # CLI: python gen_memo_from_modules.py [--root ROOT] [--src REL] [--out REL]
    root = ROOT_DEFAULT
    src_rel = SRC_REL
    out_rel = OUT_REL
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--root" and i + 1 < len(args):
            root = Path(args[i+1]).expanduser().resolve(); i += 2
        elif a == "--src" and i + 1 < len(args):
            src_rel = Path(args[i+1]); i += 2
        elif a == "--out" and i + 1 < len(args):
            out_rel = Path(args[i+1]); i += 2
        else:
            print(f"[warn] unknown arg ignored: {a}", file=sys.stderr); i += 1
    return {"root": root, "src_rel": src_rel, "out_rel": out_rel}

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def backup_if_exists(path: Path) -> Optional[Path]:
    if path.exists():
        ts = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d-%H%M%S")
        bak = path.with_suffix(path.suffix + f".{ts}.bak")
        path.replace(bak)
        return bak
    return None

def current_ts_jst() -> str:
    jst = dt.timezone(dt.timedelta(hours=9), name="JST")
    return dt.datetime.now(jst).strftime("%Y-%m-%d %H:%M JST")

def build_header(root: Path) -> str:
    return textwrap.dedent(f"""\
    # Tensaku 復元メモ（自動生成）

    *最終更新*: {current_ts_jst()}
    *生成元*: `scripts/gen_memo_from_modules.py`（各モジュール先頭docstringの @tags を収集）

    ## 方針
    - **「1レス1スクリプト」** を継続、メンテは **docstringの @tags 編集のみ**
    - 推奨タグ: `@module, @role, @inputs, @outputs, @cli, @api, @deps, @config, @contracts, @errors, @notes, @tests`
    - 複数行値は、`@key:` の直後の**インデント行**を次の `@key:` まで取り込む

    ## 主要パスと環境（固定）
    - ROOT: `{root}`
    - DATA: `{root}/data_sas`（train/dev/test/pool）
    - OUT : `{root}/outputs`
    - CKPT: `{root}/outputs/checkpoints_min/{{best.pt,last.pt}}`
    - 環境: VS Code + Jupyter / venv (Python 3.12) / PyTorch / Transformers≥4.44
    """)

def _sort_infos(infos: List[ModuleInfo]) -> List[ModuleInfo]:
    # "_" で始まる内部モジュールを先頭、その後は名前順
    def key(mi: ModuleInfo) -> Tuple[int, str]:
        return (0 if mi.file_rel.startswith("_") else 1, mi.name)
    return sorted(infos, key=key)

def main() -> int:
    cfg = load_root_from_argv()
    root: Path = cfg["root"]
    src_dir = (root / cfg["src_rel"]).resolve()
    out_path = (root / cfg["out_rel"]).resolve()

    if not src_dir.exists():
        print(f"[error] src dir not found: {src_dir}", file=sys.stderr)
        return 2

    mods = discover_modules(src_dir)
    if not mods:
        print(f"[warn] no modules found under: {src_dir}", file=sys.stderr)

    infos: List[ModuleInfo] = []
    for p in mods:
        tags, extra = read_doc_tags(p)
        dotted = f"tensaku.{p.stem}"
        info = ModuleInfo(name=dotted, file_rel=guess_display_name(p), tags=tags, extra=extra)
        infos.append(info)

    infos = _sort_infos(infos)

    # Assemble markdown
    parts: List[str] = [build_header(root), "", "## モジュール一覧", ""]
    for mi in infos:
        parts.append(fmt_module_block(mi))
        parts.append("")  # blank line
    md = "\n".join(parts).rstrip() + "\n"

    ensure_parent(out_path)
    bak = backup_if_exists(out_path)
    out_path.write_text(md, encoding="utf-8")

    if bak:
        print(f"[ok] wrote: {out_path} (backup: {bak.name})")
    else:
        print(f"[ok] wrote: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
