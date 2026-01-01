# /home/esakit25/work/tensaku/src/tensaku/infer_pool.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.infer_pool
@role     : 推論実行 (Strict Mode; fs_core/ledger integrated)
@overview :
  - infer_core:
      * Config に従い推論を行い、preds_detail.csv と *.npy を保存する。
      * out_dir は Layout から取得した ArtifactDir を要求し、保存は fs_core 経由で台帳へ記録する。
      * デフォルト/フォールバックは置かず、必要キーが無ければ ConfigError を投げる。
  - infer_pool_core:
      * Task/Pipeline 用の薄いラッパ（df_detail, raw を返す）
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from tensaku.data.base import DatasetSplit
from tensaku.embed import predict_with_emb
from tensaku.fs_core import ArtifactDir
from tensaku.utils.strict_cfg import ConfigError, require_int, require_mapping, require_str

LOGGER = logging.getLogger(__name__)

MEM_LOGGER = logging.getLogger("tensaku.mem")


def _read_proc_status(pid: int) -> dict:
    """/proc/<pid>/status から主要メモリ値を読む（Linux専用）。"""
    rss_kb = None
    hwm_kb = None
    vms_kb = None
    threads = None
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm_kb = int(line.split()[1])
                elif line.startswith("VmSize:"):
                    vms_kb = int(line.split()[1])
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
    except Exception:
        pass
    return {"rss_kb": rss_kb, "hwm_kb": hwm_kb, "vms_kb": vms_kb, "threads": threads}


def _mem_event(event: str, extra: dict | None = None) -> None:
    """tensaku.mem ロガーへ、メモリ計測イベント（JSON）を出す。本文などの重い情報は絶対に入れない。"""
    if not MEM_LOGGER.isEnabledFor(logging.DEBUG):
        return
    pid = os.getpid()
    info = _read_proc_status(pid)
    payload = {
        "event": event,
        "ts": time.strftime("%F %T"),
        "pid": pid,
        **info,
        "extra": (extra or {}),
    }
    try:
        MEM_LOGGER.debug(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # ログが壊れて実験が止まるのを防ぐ（計測はベストエフォート）
        MEM_LOGGER.debug('{"event":"memlog_failed"}')



def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _select_device(device_name: str) -> torch.device:
    device_name = (device_name or "").strip().lower()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cfg.infer.device=cuda but CUDA is not available")
        return torch.device("cuda")
    if device_name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"cfg.infer.device={device_name} but CUDA is not available")
        return torch.device(device_name)
    raise ValueError(f"Unsupported device: {device_name} (expected: cpu/cuda/cuda:0...)")


def _require_artifact_dir(out_dir: Any) -> ArtifactDir:
    if not isinstance(out_dir, ArtifactDir):
        raise TypeError(
            "infer_core requires out_dir to be an ArtifactDir bound to Layout context. "
            "Pass e.g. layout.round_infer_dir(round=...) instead of a plain Path."
        )
    if getattr(out_dir, "_context", None) is None:
        raise TypeError(
            "infer_core requires out_dir to be an ArtifactDir bound to Layout context (detached). "
            "Make sure you obtained it from Layout, not by constructing ArtifactDir manually."
        )
    return out_dir


def _infer_round_index_from_dir(out_dir: ArtifactDir) -> Optional[int]:
    params = getattr(out_dir, "params", {}) or {}
    for key in ("round", "round_index"):
        if key in params:
            try:
                return int(params[key])
            except Exception:
                pass
    return None


def infer_core(
    split: DatasetSplit,
    out_dir: ArtifactDir,
    cfg: Mapping[str, Any],
    model: Optional[Any] = None,  # Task から渡される場合は in-memory model を使う
    return_df: bool = False,
    return_raw_outputs: bool = False,
) -> Tuple[int, Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Strict Mode:
      - 必須キー（最小）:
          data.id_key, data.label_key, data.text_key_primary, data.max_len
          model.name, model.num_labels
          infer.device, infer.batch_size
          infer.ckpt (model が None の場合のみ)
      - out_dir は ArtifactDir（Layout 由来）必須
      - すべての保存は fs_core を通し、台帳へ記録する（pred/logit/emb/meta）
    """
    out_dir = _require_artifact_dir(out_dir)
    out_dir.ensure()
    round_index = _infer_round_index_from_dir(out_dir)

    data_cfg = require_mapping(cfg, "data")
    model_cfg = require_mapping(cfg, "model")
    infer_cfg = require_mapping(cfg, "infer")

    # data keys
    id_key = require_str(data_cfg, "id_key")
    label_key = require_str(data_cfg, "label_key")
    text_key_primary = require_str(data_cfg, "text_key_primary")
    max_len = require_int(data_cfg, "max_len")

    # infer keys
    device_name = require_str(infer_cfg, "device")
    batch_size = require_int(infer_cfg, "batch_size")

    # model keys
    n_class = require_int(model_cfg, "num_labels")
    base_model_name = require_str(model_cfg, "name")

    # ---- 推論対象 ----
    splits_to_infer: Dict[str, List[dict]] = {
        "labeled": split.labeled,
        "dev": split.dev,
        "test": split.test,
        "pool": split.pool,
    }
    splits_to_infer = {k: v for k, v in splits_to_infer.items() if v}
    if not splits_to_infer:
        LOGGER.warning("All target splits are empty.")
        return 0, (pd.DataFrame() if return_df else None), ({} if return_raw_outputs else None)

    # ---- モデル準備 ----
    dev = _select_device(device_name)

    if model is None:
        ckpt_path_str = require_str(infer_cfg, "ckpt")
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        LOGGER.info(f"Loading model from {ckpt_path} (n_class={n_class})...")

        hf_cfg = AutoConfig.from_pretrained(base_model_name, num_labels=n_class)
        model0 = AutoModelForSequenceClassification.from_config(hf_cfg)

        try:
            bundle = torch.load(ckpt_path, map_location="cpu")
            state_dict = bundle
            if isinstance(bundle, dict):
                if "model" in bundle:
                    state_dict = bundle["model"]
                elif "state_dict" in bundle:
                    state_dict = bundle["state_dict"]
            if not isinstance(state_dict, dict):
                raise TypeError(f"Unsupported checkpoint format: {type(bundle)}")
            model0.load_state_dict(state_dict, strict=False)
        except Exception as e:
            LOGGER.exception(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}") from e

        model = model0.to(dev).eval()
        from tensaku.models import create_tokenizer  # local import to keep module light at import time
        tokenizer = create_tokenizer(cfg)
    else:
        LOGGER.info("Using in-memory model passed from Task.")
        from tensaku.models import create_tokenizer  # local import
        tokenizer = create_tokenizer(cfg)
        model = model.to(dev).eval()

    # ---- 推論ループ ----
    outputs: Dict[str, Dict[str, Any]] = {}

    for split_name, rows in splits_to_infer.items():
        LOGGER.info(f"Inferring split: {split_name} (n={len(rows)})")
        _mem_event("infer_split_start", {"round": round_index, "split": split_name, "n": len(rows)})

        want_logits = (split_name != "labeled")



        logits, labels, embs = predict_with_emb(


            model,


            rows,


            tokenizer=tokenizer,


            bs=batch_size,


            max_len=max_len,


            device=dev.type,              # existing implementation expects str


            label_key=label_key,


            text_key_primary=text_key_primary,


            return_logits=True,


        )
        _mem_event("infer_split_after_predict", {
            "round": round_index,
            "split": split_name,
            "n": len(rows),
            "logits_shape": list(getattr(logits, "shape", ())),
            "embs_shape": list(getattr(embs, "shape", ())),
        })

        y_pred = logits.argmax(axis=-1)  # logits は常に生成する（dtype object 混入を防止）
        outputs[split_name] = {
            "rows": rows,
            "logits": logits,
            "labels": labels,
            "embs": embs,
            "y_pred": y_pred,
        }

        # ---- fs_core 保存（台帳に記録）----
        if round_index is None:
            raise ConfigError(
                "[infer] out_dir does not contain round index in params; "
                "kind=logit/emb requires round_index. Use layout.round_infer_dir(round=...)."
            )

        if split_name != "labeled" and logits is not None:


            (out_dir / f"{split_name}_logits.npy").save_npy(


                logits, record=True, kind="logit", round_index=round_index, meta={"split": split_name}


            )
        (out_dir / f"{split_name}_embs.npy").save_npy(
            embs, record=True, kind="emb", round_index=round_index, meta={"split": split_name}
        )
        _mem_event("infer_split_after_save", {"round": round_index, "split": split_name, "n": len(rows)})

    # ---- DataFrame 構築 ----
    dfs: List[pd.DataFrame] = []
    raw_outputs: Dict[str, Any] = {}
    _mem_event("infer_df_build_start", {"round": round_index, "splits": list(outputs.keys())})

    for split_name, data in outputs.items():
        rows = data["rows"]

        ids_list = [r.get(id_key, str(i)) for i, r in enumerate(rows)]

        df_split = pd.DataFrame(
            {
                "id": [r.get(id_key, str(i)) for i, r in enumerate(rows)],
                "split": split_name,
                "y_pred": data["y_pred"],
            }
        )

        if data.get("labels") is not None:
            df_split["y_true"] = data["labels"]

        dfs.append(df_split)
        _mem_event("infer_df_split_done", {"round": round_index, "split": split_name, "n": len(rows)})

        if return_raw_outputs:
            raw_outputs[split_name] = {
                "ids": ids_list,
                "logits": data.get("logits"),
                "embs": data.get("embs"),
                "labels": data.get("labels"),
                "y_pred": data.get("y_pred"),
            }

    _mem_event("infer_df_build_end", {"round": round_index, "n_rows": int(sum(len(v.get("rows", [])) for v in outputs.values()))})
    final_df = pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()

    # ---- preds_detail.csv（台帳に記録）----
    preds_file = out_dir / "preds_detail.csv"
    preds_file.save_csv(
        final_df.itertuples(index=False, name=None),
        header=list(final_df.columns),
        record=True,
        kind="pred",
        round_index=round_index,
        meta={"name": "preds_detail"},
    )
    LOGGER.info(f"Saved predictions to {preds_file.path}")

    # ---- infer.meta.json（台帳に記録）----
    meta = {
        "generated_at": time.strftime("%F %T"),
        "ckpt": str(require_str(infer_cfg, "ckpt")) if model is None else "in-memory",
        "n_class": n_class,
        "splits": {k: len(v) for k, v in splits_to_infer.items()},
    }
    (out_dir / "infer.meta.json").save_json(meta, record=True, kind="meta", round_index=round_index)

    return 0, (final_df if return_df else None), (raw_outputs if return_raw_outputs else None)


def infer_pool_core(
    *,
    split: DatasetSplit,
    out_dir: ArtifactDir,
    cfg: Mapping[str, Any],
    model: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Task 側から呼ぶための薄いラッパ。
    - out_dir は layout.round_infer_dir(round=...) の ArtifactDir を渡すこと。
    """
    rc, df, raw = infer_core(
        split=split,
        out_dir=out_dir,
        cfg=cfg,
        model=model,
        return_df=True,
        return_raw_outputs=True,
    )
    if rc != 0:
        raise RuntimeError(f"infer_core failed with rc={rc}")
    return (df if df is not None else pd.DataFrame()), (raw if raw is not None else {})
