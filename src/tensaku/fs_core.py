# /home/esakit25/work/tensaku/src/tensaku/fs_core.py
# -*- coding: utf-8 -*-
"""
tensaku.fs_core
===============

宣言的なファイルシステム抽象化レイヤ（住所録）と、台帳（artifact ledger）を提供します。

設計原理
- **Smart Nodes**: Layoutから取得したファイルオブジェクトは、自分の親（LayoutContext）を知っています。
  これにより `file.save_checkpoint(model)` のような直感的な記述が可能です。
- **Atomic Operations**: チェックポイントなどの重要な保存は、一時ファイルへの書き込みとリネームによって安全に行われます。
- **Ledger Integration**: 保存操作を行うと、自動的に台帳（JSONL）にメタデータが記録されます。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Generic, Iterator, Iterable, Optional, Set, Type, TypeVar, Union
import json
import os
import datetime
import shutil


# ============================================================================
# Base Node
# ============================================================================

@dataclass(frozen=True)
class FSNode:
    """ファイル・ディレクトリ共通の基底クラス"""
    path: Path

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    @property
    def parent(self) -> Path:
        return self.path.parent

    @property
    def name(self) -> str:
        return self.path.name

    def exists(self) -> bool:
        return self.path.exists()

    def relative_to(self, other: Union[str, Path, "FSNode"]) -> Path:
        if isinstance(other, FSNode):
            other = other.path
        return self.path.relative_to(other)


# ============================================================================
# File / Dir Nodes (Base)
# ============================================================================

@dataclass(frozen=True)
class File(FSNode):
    """ファイルノード。pathlib.Path の薄いラッパ。"""

    @property
    def dir(self) -> Path:
        return self.path.parent

    def ensure_parent(self, *, parents: bool = True, exist_ok: bool = True) -> "File":
        if parents:
            self.path.parent.mkdir(parents=True, exist_ok=exist_ok)
        else:
            self.path.parent.mkdir(exist_ok=exist_ok)
        return self

    def touch(self, *, exist_ok: bool = True) -> "File":
        self.ensure_parent()
        self.path.touch(exist_ok=exist_ok)
        return self

    def open(self, *args: Any, **kwargs: Any):
        self.ensure_parent()
        return self.path.open(*args, **kwargs)

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.path.read_text(encoding=encoding)

    def write_text(self, text: str, encoding: str = "utf-8") -> "File":
        self.ensure_parent()
        self.path.write_text(text, encoding=encoding)
        return self

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def write_bytes(self, data: bytes) -> "File":
        self.ensure_parent()
        self.path.write_bytes(data)
        return self


@dataclass(frozen=True)
class Dir(FSNode):
    """ディレクトリノード。"""

    def mkdir(self, *, parents: bool = True, exist_ok: bool = True) -> "Dir":
        self.path.mkdir(parents=parents, exist_ok=exist_ok)
        return self

    def ensure(self) -> "Dir":
        return self.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Artifact-aware Nodes (Smart Nodes)
# ============================================================================

@dataclass(frozen=True)
class ArtifactFile(File):
    """
    メタ情報と保存機能を持つファイルノード。
    Layout経由でアクセスされた場合、_context に Layout インスタンスが注入されます。
    """
    kind: Optional[str] = None
    record: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Context injection (Layout instance)
    _context: Optional["Layout"] = field(default=None, repr=False, compare=False)

    def _require_context(self) -> "Layout":
        if self._context is None:
            raise RuntimeError(f"ArtifactFile '{self.path}' is detached from Layout context. Cannot save directly.")
        return self._context

    # --- Proxy Save Methods ---

    def save_checkpoint(self, state_dict: Any, *, atomic: bool = True, **kwargs) -> Path:
        """チェックポイントを保存 (Atomic推奨)"""
        return self._require_context().save_checkpoint(self, state_dict, atomic=atomic, **kwargs)

    def save_json(self, obj: Any, **kwargs) -> Path:
        return self._require_context().save_json(self, obj, **kwargs)

    def save_text(self, text: str, **kwargs) -> Path:
        return self._require_context().save_text(self, text, **kwargs)
    
    def save_csv(self, rows: Iterable[Iterable[Any]], **kwargs) -> Path:
        return self._require_context().save_csv(self, rows, **kwargs)

    def save_npy(self, arr: Any, **kwargs) -> Path:
        return self._require_context().save_npy(self, arr, **kwargs)

    def save_torch(self, obj: Any, **kwargs) -> Path:
        return self._require_context().save_torch(self, obj, **kwargs)


# tensaku/fs_core.py 内の ArtifactDir クラス

@dataclass(frozen=True)
class ArtifactDir(Dir):
    kind: Optional[str] = None
    record: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    _context: Optional["Layout"] = field(default=None, repr=False, compare=False)

    # 追加: / 演算子のオーバーロード
    def __truediv__(self, other: str) -> ArtifactFile:
        """
        dir / "filename" の構文で、コンテキストを引き継いだ子ファイル(Smart Node)を生成する。
        これにより train_core 内で自由にファイルを派生させられます。
        """
        new_path = self.path / other
        
        # 親(Dir)の設定を引き継ぐ
        # kindはファイルごとに異なる可能性が高いため、あえて None にして save_x 時に指定させるか、
        # save_checkpoint などの専用メソッドのデフォルトに任せるのが安全です。
        return ArtifactFile(
            path=new_path,
            kind=None,  # save時に決定 (checkpointならデフォルト "ckpt" になる)
            record=self.record, # ディレクトリが記録対象なら、中身も記録対象にするのが自然
            meta=self.meta.copy(),
            params=self.params.copy(),
            _context=self._context # 最重要: Layoutへの参照を引き継ぐ
        )


# ============================================================================
# Dynamic Family (Factory)
# ============================================================================

T = TypeVar("T", bound=FSNode)

@dataclass(frozen=True)
class Family(Generic[T]):
    root: Path
    pattern: str
    node_type: Type[T]
    kind: Optional[str] = None
    record: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    _context: Optional["Layout"] = field(default=None, repr=False, compare=False)

    def __call__(self, **kwargs: Any) -> T:
        rel_path = self.pattern.format(**kwargs)
        # 生成されるノードに context を伝播させる
        # type: ignore[call-arg]
        return self.node_type(
            self.root / rel_path,
            kind=self.kind,
            record=self.record,
            meta=dict(self.meta),
            params=dict(kwargs),
            _context=self._context
        )


# ============================================================================
# Descriptors (Context Injectors)
# ============================================================================

class PathRule:
    def __init__(
        self,
        rel_path: str,
        node_type: Type[Union[ArtifactFile, ArtifactDir]],
        *,
        kind: Optional[str] = None,
        record: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.rel_path = rel_path
        self.node_type = node_type
        self.kind = kind
        self.record = record
        self.meta = meta or {}

    def __get__(self, obj: Optional["Layout"], objtype=None) -> Any:
        if obj is None:
            return self
        # ここで Layout インスタンス (obj) を _context として注入
        # type: ignore[call-arg]
        return self.node_type(
            obj.root / self.rel_path,
            kind=self.kind,
            record=self.record,
            meta=dict(self.meta),
            params={},
            _context=obj
        )


class FamilyRule:
    def __init__(
        self,
        pattern: str,
        node_type: Type[Union[ArtifactFile, ArtifactDir]],
        *,
        kind: Optional[str] = None,
        record: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.pattern = pattern
        self.node_type = node_type
        self.kind = kind
        self.record = record
        self.meta = meta or {}

    def __get__(self, obj: Optional["Layout"], objtype=None) -> Any:
        if obj is None:
            return self
        # Family 自体にも context を持たせる
        return Family(
            obj.root,
            self.pattern,
            self.node_type,
            kind=self.kind,
            record=self.record,
            meta=dict(self.meta),
            _context=obj
        )


# --- Public Aliases ---

def define_file(path: str, *, kind: Optional[str] = None, record: bool = False, meta: Optional[Dict[str, Any]] = None) -> PathRule:
    return PathRule(path, ArtifactFile, kind=kind, record=record, meta=meta)

def define_dir(path: str, *, kind: Optional[str] = None, record: bool = False, meta: Optional[Dict[str, Any]] = None) -> PathRule:
    return PathRule(path, ArtifactDir, kind=kind, record=record, meta=meta)

def define_family(pattern: str, *, kind: Optional[str] = None, record: bool = False, meta: Optional[Dict[str, Any]] = None) -> FamilyRule:
    return FamilyRule(pattern, ArtifactFile, kind=kind, record=record, meta=meta)

def define_dir_family(pattern: str, node_type: Type[ArtifactDir] = ArtifactDir,*, kind: Optional[str] = None, record: bool = False, meta: Optional[Dict[str, Any]] = None) -> FamilyRule:
    return FamilyRule(pattern, node_type, kind=kind, record=record, meta=meta)


# ============================================================================
# Artifact Ledger
# ============================================================================

@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    path: str
    is_relative: bool
    round_index: Optional[int]
    meta: Dict[str, Any]
    size_bytes: Optional[int]
    created_at: str


class ArtifactLedger:
    def __init__(self, index_file: Union[str, Path, File]):
        if isinstance(index_file, File):
            self.index_file = index_file.path
        else:
            self.index_file = Path(index_file)

    def append(self, rec: ArtifactRecord) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec.__dict__, ensure_ascii=False, separators=(",", ":"))
        with self.index_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def iter_records(self) -> Iterator[ArtifactRecord]:
        if not self.index_file.exists():
            return
            yield
        with self.index_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    yield ArtifactRecord(
                        kind=str(obj["kind"]),
                        path=str(obj["path"]),
                        is_relative=bool(obj["is_relative"]),
                        round_index=obj.get("round_index"),
                        meta=dict(obj.get("meta") or {}),
                        size_bytes=obj.get("size_bytes"),
                        created_at=str(obj["created_at"]),
                    )
                except Exception:
                    # 破損行はスキップまたはログ出力（ここではスキップ）
                    continue


# ============================================================================
# Layout Base (I/O + Ledger)
# ============================================================================

class Layout:
    """住所録（パス定義）の基底。I/O と台帳は fs_core に集約する。"""

    # subclass can override
    artifact_index = define_file("artifacts/index.jsonl")

    # Optional schema validation
    ALLOWED_KINDS: Optional[Set[str]] = None

    def __init__(self, root: Union[str, Path], structure: Optional[Any] = None):
        self.root = Path(root).resolve()
        if structure is not None:
            self._load_structure(structure)
        self._ledger_cache: Optional[ArtifactLedger] = None

    def _load_structure(self, structure: Any) -> None:
        if isinstance(structure, dict):
            for k, v in structure.items():
                setattr(self, k, define_file(v))
        else:
            raise TypeError("structure must be dict[str,str]")

    @property
    def ledger(self) -> ArtifactLedger:
        if self._ledger_cache is None:
            idx = getattr(self, "artifact_index")
            if isinstance(idx, File):
                self._ledger_cache = ArtifactLedger(idx.path)
            else:
                self._ledger_cache = ArtifactLedger(Path(idx))
        return self._ledger_cache
    
    def list_artifacts(self, round_index: Optional[int] = None) -> Iterator[ArtifactRecord]:
        """
        Cleaner interface: iterate over recorded artifacts.
        Yields records with ABSOLUTE paths to ensure safe deletion.
        """
        for rec in self.ledger.iter_records():
            if round_index is not None:
                if rec.round_index != round_index:
                    continue
            
            # 相対パス(is_relative=True)の場合、Layout.root と結合して絶対パスにする
            if rec.is_relative:
                abs_path = self.root / rec.path
                # ArtifactRecord は frozen なので replace でコピーを作成
                rec = replace(rec, path=str(abs_path), is_relative=False)
            
            yield rec

    def _now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _resolve_is_relative(self, p: Path) -> tuple[str, bool]:
        try:
            rel = p.relative_to(self.root)
            return str(rel.as_posix()), True
        except Exception:
            return str(p), False

    def record_artifact(
        self,
        file: Union[File, Path, str],
        *,
        kind: str,
        round_index: Optional[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.ALLOWED_KINDS is not None and kind not in self.ALLOWED_KINDS:
            raise ValueError(f"Unknown kind '{kind}'. Allowed: {sorted(self.ALLOWED_KINDS)}")

        p = Path(file) if not isinstance(file, File) else file.path
        path_s, is_rel = self._resolve_is_relative(p)
        
        size = None
        try:
            if p.exists() and p.is_file():
                size = p.stat().st_size
        except Exception:
            pass

        rec = ArtifactRecord(
            kind=kind,
            path=path_s,
            is_relative=is_rel,
            round_index=round_index,
            meta=dict(meta or {}),
            size_bytes=size,
            created_at=self._now_iso(),
        )
        self.ledger.append(rec)

    # ---------------------------------------------------------------------
    # Save Helper Logic
    # ---------------------------------------------------------------------

    REQUIRED_ROUND_KINDS = {"ckpt", "emb", "prob", "logit"}

    def _coerce_file_and_defaults(
        self,
        target: Union[File, Path, str],
    ) -> tuple[Path, Optional[str], bool, Dict[str, Any], Dict[str, Any]]:
        if isinstance(target, File):
            p = target.path
            kind = getattr(target, "kind", None)
            record = bool(getattr(target, "record", False))
            meta = dict(getattr(target, "meta", {}) or {})
            params = dict(getattr(target, "params", {}) or {})
            return p, kind, record, meta, params
        p = Path(target)
        return p, None, False, {}, {}

    def _maybe_record_from_save(
        self,
        *,
        path: Path,
        kind0: Optional[str],
        record0: bool,
        meta0: Dict[str, Any],
        params: Dict[str, Any],
        record: Optional[bool],
        kind: Optional[str],
        round_index: Optional[int],
        meta: Optional[Dict[str, Any]],
    ) -> None:
        do_record = record0 if record is None else bool(record)
        if not do_record:
            return

        k = kind0 if kind is None else kind
        if not k:
            raise ValueError("record=True but kind is not specified/defined")

        def _infer_round(params_dict):
            if not params_dict: return None
            for key in ["round", "round_index"]:
                if key in params_dict:
                    try: return int(params_dict[key])
                    except: pass
            return None

        r = round_index if round_index is not None else _infer_round(params)
        if k in self.REQUIRED_ROUND_KINDS and r is None:
            # CheckpointなどはRound情報が必須
            raise ValueError(f"round_index is required for kind={k} (could not infer from file params)")

        m = {**meta0, **(meta or {})}
        self.record_artifact(path, kind=k, round_index=r, meta=m)

    # ---------------------------------------------------------------------
    # Save Methods
    # ---------------------------------------------------------------------

    def save_checkpoint(
        self,
        target: Union[File, Path, str],
        state_dict: Any,
        *,
        atomic: bool = True,
        record: Optional[bool] = None,
        kind: Optional[str] = None,
        round_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomic safe save for checkpoints."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch is required for save_checkpoint")

        p, k0, rec0, m0, params = self._coerce_file_and_defaults(target)
        final_kind = kind or k0 or "ckpt"  # default to ckpt
        
        p.parent.mkdir(parents=True, exist_ok=True)

        if atomic:
            # Atomic Save Strategy: Write to .tmp then rename
            tmp_p = p.with_suffix(p.suffix + ".tmp")
            try:
                torch.save(state_dict, tmp_p)
                if p.exists():
                    # Windows compatibility: rename cannot overwrite
                    try:
                        p.unlink()
                    except OSError:
                        pass
                tmp_p.rename(p)
            except Exception:
                if tmp_p.exists():
                    tmp_p.unlink()
                raise
        else:
            torch.save(state_dict, p)

        self._maybe_record_from_save(
            path=p, kind0=k0, record0=rec0, meta0=m0, params=params,
            record=record, kind=final_kind, round_index=round_index, meta=meta
        )
        return p

    def save_text(
        self,
        target: Union[File, Path, str],
        text: str,
        *,
        encoding: str = "utf-8",
        record: Optional[bool] = None,
        kind: Optional[str] = None,
        round_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        p, k0, rec0, m0, params = self._coerce_file_and_defaults(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        self._maybe_record_from_save(
            path=p, kind0=k0, record0=rec0, meta0=m0, params=params,
            record=record, kind=kind, round_index=round_index, meta=meta
        )
        return p

    def save_json(
        self,
        target: Union[File, Path, str],
        obj: Any,
        *,
        ensure_ascii: bool = False,
        indent: Optional[int] = 2,
        record: Optional[bool] = None,
        kind: Optional[str] = None,
        round_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        text = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
        return self.save_text(target, text, encoding="utf-8", record=record, kind=kind, round_index=round_index, meta=meta)

    def save_csv(
        self,
        target: Union[File, Path, str],
        rows: Iterable[Iterable[Any]],
        *,
        header: Optional[Iterable[str]] = None,
        encoding: str = "utf-8",
        record: Optional[bool] = None,
        kind: Optional[str] = None,
        round_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        p, k0, rec0, m0, params = self._coerce_file_and_defaults(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with p.open("w", encoding=encoding, newline="") as f:
            w = csv.writer(f)
            if header is not None:
                w.writerow(list(header))
            for r in rows:
                w.writerow(list(r))
            f.flush()
            os.fsync(f.fileno())

        self._maybe_record_from_save(
            path=p, kind0=k0, record0=rec0, meta0=m0, params=params,
            record=record, kind=kind, round_index=round_index, meta=meta
        )
        return p

    def save_npy(
        self,
        target: Union[File, Path, str],
        arr: Any,
        *,
        record: Optional[bool] = None,
        kind: Optional[str] = None,
        round_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError("NumPy is required for save_npy")

        p, k0, rec0, m0, params = self._coerce_file_and_defaults(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)

        self._maybe_record_from_save(
            path=p, kind0=k0, record0=rec0, meta0=m0, params=params,
            record=record, kind=kind, round_index=round_index, meta=meta
        )
        return p

    def save_torch(
        self,
        target: Union[File, Path, str],
        obj: Any,
        *,
        record: Optional[bool] = None,
        kind: Optional[str] = None,
        round_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch is required for save_torch")

        p, k0, rec0, m0, params = self._coerce_file_and_defaults(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, p)

        self._maybe_record_from_save(
            path=p, kind0=k0, record0=rec0, meta0=m0, params=params,
            record=record, kind=kind, round_index=round_index, meta=meta
        )
        return p

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------

    def ensure_all_dirs(self) -> None:
        """定義済みの全静的ディレクトリを作成"""
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, PathRule):
                node = getattr(self, name)
                if isinstance(node, File):
                    node.ensure_parent()
                elif isinstance(node, Dir):
                    node.mkdir()

    def tree(self) -> str:
        """定義ツリーの可視化"""
        entries = []
        # Class attributes
        for name in dir(self.__class__):
            if name.startswith("_"): continue
            attr = getattr(self.__class__, name)
            if isinstance(attr, PathRule):
                entries.append((Path(attr.rel_path).parts, name, "static", attr.node_type.__name__))
            elif isinstance(attr, FamilyRule):
                entries.append((Path(attr.pattern).parts, name, "family", attr.node_type.__name__))

        # Instance attributes
        for name, val in self.__dict__.items():
            if name.startswith("_"): continue
            if isinstance(val, FSNode): # File or Dir
                try:
                    rel = val.path.relative_to(self.root)
                    entries.append((rel.parts, name, "instance", type(val).__name__))
                except ValueError: pass
            elif isinstance(val, Family):
                entries.append((Path(val.pattern).parts, name, "family", val.node_type.__name__))

        tree_root: Dict[str, Any] = {}
        for parts, name, kind, type_name in entries:
            current = tree_root
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = f"{name} ({kind}/{type_name})"
                else:
                    if part not in current:
                        current[part] = {}
                    if not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]

        lines = [f"{self.root}/"]
        
        def _render(node: Dict[str, Any], prefix: str):
            keys = sorted(node.keys())
            for i, key in enumerate(keys):
                val = node[key]
                is_last = (i == len(keys) - 1)
                connector = "└── " if is_last else "├── "
                child_prefix = "    " if is_last else "│   "

                if isinstance(val, dict):
                    lines.append(f"{prefix}{connector}{key}/")
                    _render(val, prefix + child_prefix)
                else:
                    lines.append(f"{prefix}{connector}{key} -> {val}")

        _render(tree_root, "")
        return "\n".join(lines)