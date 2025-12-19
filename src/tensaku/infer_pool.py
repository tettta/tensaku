# /home/esakit25/work/tensaku/src/tensaku/infer_pool.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.infer_pool
@role     : 推論実行 (Strict Mode)
@overview :
    - infer_core: Configに従い、指定されたチェックポイントをロードして推論を行う。
                  デフォルト値を廃止し、CKPT不在時は即時エラーとする。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from tensaku.data.base import DatasetSplit
from tensaku.models import create_model, create_tokenizer
from tensaku.embed import predict_with_emb

LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[dict]:
    # (変更なし)
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _select_device(device_name: str) -> torch.device:
    """Strict: autoを許容しつつも、Configからの入力を必須とする"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


# =============================================================================
# infer_core (Strict Mode)
# =============================================================================

def infer_core(
    split: DatasetSplit,
    out_dir: Path,
    cfg: Mapping[str, Any],
    model: Optional[Any] = None, # タスクから渡される場合はメモリ上のモデルを使う
    return_df: bool = False,
    return_raw_outputs: bool = False,
) -> Tuple[int, Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Strict Mode:
      - 必須キー: run.save_predictions, run.save_logits, infer.ckpt, infer.device, infer.batch_size
      - ckptパスが存在しない場合は FileNotFoundError
    """
    # 必須セクション確認
    if "run" not in cfg: raise KeyError("cfg missing 'run' section")
    if "data" not in cfg: raise KeyError("cfg missing 'data' section")
    if "model" not in cfg: raise KeyError("cfg missing 'model' section")
    if "infer" not in cfg: raise KeyError("cfg missing 'infer' section")

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg["infer"]

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Strict: ID/Label key
    id_key = data_cfg["id_key"]
    label_key = data_cfg["label_key"] 

    # ---- データ取り出し ----
    splits_to_infer = {
        "labeled": split.labeled,
        "dev": split.dev,
        "test": split.test,
        "pool": split.pool,
    }
    splits_to_infer = {k: v for k, v in splits_to_infer.items() if v}

    if not splits_to_infer:
        LOGGER.warning("All target splits are empty.")
        return 0, None, None

    # ---- モデル準備 ----
    # Strict: device 必須
    device_name = infer_cfg["device"]
    dev = _select_device(device_name)
    
    # Strict: num_labels 必須 (ckptからの推測ロジック廃止)
    n_class = int(model_cfg["num_labels"])

    # モデル構築
    if model is None:
        # Strict: ckpt パス必須
        ckpt_path_str = infer_cfg["ckpt"]
        if not ckpt_path_str:
            raise ValueError("infer.ckpt must be specified in config (Strict Mode)")
        
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        LOGGER.info(f"Loading model from {ckpt_path} (n_class={n_class})...")
        
        base_model_name = str(model_cfg["name"])
        if not base_model_name:
            raise ValueError("cfg.model.name must be set (e.g., 'cl-tohoku/bert-base-japanese-v3')")

        hf_cfg = AutoConfig.from_pretrained(base_model_name, num_labels=n_class)
        model = AutoModelForSequenceClassification.from_config(hf_cfg)

        tokenizer = create_tokenizer(cfg)

        # 重みロード
        try:
            bundle = torch.load(ckpt_path, map_location="cpu")
            state_dict = bundle
            if isinstance(bundle, dict):
                if "model" in bundle:
                    state_dict = bundle["model"]
                elif "state_dict" in bundle:
                    state_dict = bundle["state_dict"]

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                # 警告は出すが、ヘッド入れ替えなどを許容するため例外にはしない
                pass
        except Exception as e:
            # [Strict修正] ロード失敗は即座に例外を投げる (return 2 による隠蔽をやめる)
            LOGGER.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}") from e

        model = model.to(dev).eval()

    else:
        LOGGER.info("Using in-memory model passed from Task.")
        tokenizer = create_tokenizer(cfg)
        model = model.to(dev).eval()

    # Strict: max_len, batch_size 必須
    max_len = int(data_cfg["max_len"])
    batch_size = int(infer_cfg["batch_size"])

    # ---- 推論ループ ----
    outputs: Dict[str, Dict[str, Any]] = {}

    for split_name, rows in splits_to_infer.items():
        LOGGER.info(f"Inferring split: {split_name} (n={len(rows)})")

        text_key = data_cfg["text_key_primary"]
        
        # predict_with_emb は既存実装を利用
        logits, labels, embs = predict_with_emb(
            model,
            rows,
            tokenizer=tokenizer,
            bs=batch_size,
            max_len=max_len,
            device=dev.type,
            label_key=label_key,       # <--- Added
            text_key_primary=text_key  # <--- Added
        )
        
        y_pred = logits.argmax(axis=-1)
        
        outputs[split_name] = {
            "logits": logits,
            "embs": embs,
            "labels": labels, 
            "y_pred": y_pred, 
            "rows": rows
        }

    # ---- 結果の保存と集約 ----
    do_save_predictions = bool(run_cfg["save_predictions"])
    do_save_logits = bool(run_cfg["save_logits"])

    # 1. npy 書き出し
    if do_save_logits:
        for split_name, data in outputs.items():
            if data.get("logits") is not None:
                np.save(out_dir / f"{split_name}_logits.npy", data["logits"])
            if data.get("embs") is not None:
                np.save(out_dir / f"{split_name}_embs.npy", data["embs"])

    # 2. DataFrame 構築
    dfs = []
    raw_outputs = {}

    for split_name, data in outputs.items():
        rows = data["rows"]
        
        # 基本列のみを持つ DataFrame
        df_split = pd.DataFrame({
            "id": [r.get(id_key, str(i)) for i, r in enumerate(rows)],
            "split": split_name,
            "y_pred": data["y_pred"],
        })

        # 正解ラベル (あれば)
        if data.get("labels") is not None:
            df_split["y_true"] = data["labels"]
        else:
            df_split["y_true"] = np.nan

        dfs.append(df_split)
        
        # Raw Outputs (Confidence計算用)
        raw_outputs[split_name] = {
            "logits": data["logits"],
            "embs": data["embs"],
            "labels": data["labels"],
            "y_pred": data["y_pred"],
            "ids": df_split["id"].tolist()
        }

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # 3. CSV 書き出し (Base)
    if do_save_predictions and not final_df.empty:
        save_path = out_dir / "preds_detail.csv"
        final_df.to_csv(save_path, index=False)
        LOGGER.info(f"Saved base predictions to {save_path}")

    # meta 書き出し
    meta = {
        "generated_at": time.strftime("%F %T"),
        "ckpt": str(infer_cfg.get("ckpt", "in-memory")),
    }
    with open(out_dir / "infer.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return 0, final_df if return_df else None, raw_outputs if return_raw_outputs else None

# =============================================================================
# Compatibility wrapper (Task/Pipeline convenience)
# =============================================================================

def infer_pool_core(
    *,
    split: Any,
    out_dir: Any,
    cfg: Mapping[str, Any],
    model: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compatibility wrapper.

    Returns:
      df_detail, raw_outputs

    Raises:
      RuntimeError if infer_core fails.
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
    if df is None or raw is None:
        # [Strict修正] 空の場合は None ではなく空のDataFrame/dictを返す
        return pd.DataFrame(), {}
    return df, raw