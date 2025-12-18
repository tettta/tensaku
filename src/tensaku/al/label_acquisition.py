# /home/esakit25/work/tensaku/src/tensaku/al/label_acquisition.py
# -*- coding: utf-8 -*-
"""tensaku.al.label_acquisition

@role: AL のサンプリング後に、必ず外部オラクルからラベルを取得する。

Why this module exists
- Loop から「外部ラベル付与」の処理を分離し、責務を明確化する。
- Pipeline/Task は「オラクルの中身」を知らず、Adapter 契約だけに依存する。

STRICT principles
- No fallback: oracle_labels が未実装なら即エラー。
- Missing labels: 取得できない ID があれば即エラー。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


class LabelAcquisitionError(RuntimeError):
    """Raised when oracle label acquisition fails in strict mode."""


def acquire_labels(adapter: Any, ids: Sequence[Any]) -> Dict[Any, Any]:
    """Acquire labels for the given ids from adapter.oracle_labels.

    Contract (STRICT)
    - adapter must implement `oracle_labels(ids) -> Mapping[id, label]`.
    - returned mapping must contain all requested ids.

    Args:
        adapter: Dataset adapter instance.
        ids: Selected ids (may contain duplicates).

    Returns:
        Dict[id, label]
    """

    if ids is None:
        raise LabelAcquisitionError("ids is None")

    unique_ids: List[Any] = list(dict.fromkeys(list(ids)))  # stable unique
    if len(unique_ids) == 0:
        return {}

    fn = getattr(adapter, "oracle_labels", None)
    if fn is None or not callable(fn):
        raise LabelAcquisitionError(
            "adapter.oracle_labels is required (contract). It is missing or not callable."
        )

    # Detect base (un-overridden) implementation if present.
    base_impl = getattr(getattr(adapter, "__class__", object), "oracle_labels", None)
    if base_impl is not None:
        # If adapter.oracle_labels is exactly the base implementation, treat as unimplemented.
        try:
            from tensaku.data.base import BaseDatasetAdapter  # local import to avoid cycles

            if getattr(adapter.__class__, "oracle_labels", None) is getattr(
                BaseDatasetAdapter, "oracle_labels", None
            ):
                raise LabelAcquisitionError(
                    "adapter.oracle_labels is not implemented (still BaseDatasetAdapter.oracle_labels)."
                )
        except Exception:
            # If import fails, we still rely on runtime errors below.
            pass

    out = fn(unique_ids)
    if not isinstance(out, Mapping):
        raise LabelAcquisitionError(
            f"adapter.oracle_labels must return Mapping, got: {type(out)}"
        )

    missing = [i for i in unique_ids if i not in out]
    if missing:
        # Strict: do not silently drop.
        preview = ", ".join(map(str, missing[:10]))
        more = "" if len(missing) <= 10 else f" (+{len(missing)-10} more)"
        raise LabelAcquisitionError(f"oracle labels missing for ids: {preview}{more}")

    return dict(out)
