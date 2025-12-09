# /home/esakit25/work/tensaku/src/tensaku/infer_pool.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.infer_pool
@role     : dev / pool / test に対して ckpt で一括推論し、予測DataFrameと生データ(logits/embs)を返す。
@overview :
    - infer_core: メモリ上の DatasetSplit とモデルを受け取り、推論を行う。
                  return_df=True の場合、統合された DataFrame と、
                  信頼度計算用の生データ (raw_outputs) を返す。
    - run       : CLI 用ラッパー。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import pandas as pd
import torch

from tensaku.data.base import DatasetSplit
# Phase2 モデル定義
from tensaku.models import create_model, create_tokenizer
# 推論ユーティリティ (model_io への依存を削除)
from .embed import predict_with_emb

LOGGER = logging.getLogger(__name__)

DEF_BASE = "cl-tohoku/bert-base-japanese-v3"


# ===== I/O helpers ===============================================================================


def _read_jsonl(path: str) -> List[dict]:
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


def _extract_state_dict(bundle: Any) -> Optional[Dict[str, Any]]:
    if bundle is None:
        return None
    if isinstance(bundle, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "ema", "model_ema"]:
            v = bundle.get(k)
            if isinstance(v, dict):
                return v
        if any(hasattr(v, "shape") for v in bundle.values()):
            return bundle
    return None


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _normalize_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    sd = state_dict
    if any(k.startswith("module.") for k in sd.keys()):
        sd = _strip_prefix(sd, "module.")
    if any(k.startswith("model.") for k in sd.keys()):
        sd = _strip_prefix(sd, "model.")
    return sd


def _select_device(device_name: str = "auto") -> torch.device:
    """デバイス選択ロジック (model_io から移植)"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


# =============================================================================
# infer_core
# =============================================================================


def infer_core(
    split: DatasetSplit,
    out_dir: Path,
    cfg: Mapping[str, Any],
    model: Optional[Any] = None,
    return_df: bool = False,
) -> Tuple[int, Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    メモリ上の DatasetSplit を受け取り、推論を実行するコア関数。
    信頼度計算（TrustScore等）は行わず、logits/embs の生データと基本DataFrameを返す。
    """
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("infer", {})

    out_dir.mkdir(parents=True, exist_ok=True)
    id_key = data_cfg.get("id_key", "id")
    label_key = data_cfg.get("label_key", "score")

    # ---- データ取り出し ----
    # 信頼度計算（TrustScore等）のために labeled も推論しておくのが望ましい
    splits_to_infer = {
        "labeled": split.labeled,
        "dev": split.dev,
        "test": split.test,
        "pool": split.pool,
    }
    # 空のsplitは除外
    splits_to_infer = {k: v for k, v in splits_to_infer.items() if v}

    if not splits_to_infer:
        LOGGER.warning("All target splits are empty.")
        return 0, None, None

    # ---- モデル準備 ----
    device_name = infer_cfg.get("device", "auto")
    dev = _select_device(device_name)
    
    n_class: Optional[int] = model_cfg.get("num_labels")
    ckpt_path = infer_cfg.get("ckpt") or model_cfg.get("ckpt")
    
    # モデルが渡されていない場合のみロード処理
    if model is None:
        if not ckpt_path:
            base_out = run_cfg.get("out_dir", "./outputs")
            ckpt_path = os.path.join(base_out, "checkpoints_min", "best.pt")
        
        has_ckpt = os.path.exists(ckpt_path) if ckpt_path else False
        allow_random = bool(infer_cfg.get("allow_random", False))

        if not has_ckpt and not allow_random:
            LOGGER.error(f"ckpt not found: {ckpt_path}")
            return 2, None, None

        # ckptからn_class推定
        state_dict = None
        if n_class is None and has_ckpt:
            try:
                bundle = torch.load(ckpt_path, map_location="cpu")
                sd_raw = _extract_state_dict(bundle)
                if sd_raw is not None:
                    for key in ["classifier.weight", "score.weight", "head.weight"]:
                        if key in sd_raw and hasattr(sd_raw[key], "shape"):
                            n_class = int(sd_raw[key].shape[0])
                            break
                    state_dict = _normalize_keys(sd_raw)
            except Exception as e:
                LOGGER.warning(f"Failed to inspect ckpt for num_labels: {e}")

        if n_class is None:
            n_class = int(model_cfg.get("num_labels_fallback", 6))

        LOGGER.info(f"Creating model (n_class={n_class})...")
        tokenizer = create_tokenizer(cfg)
        model = create_model(cfg, n_class)

        if has_ckpt:
            if state_dict is None:
                try:
                    bundle = torch.load(ckpt_path, map_location="cpu")
                    sd_raw = _extract_state_dict(bundle)
                    state_dict = _normalize_keys(sd_raw) if sd_raw is not None else None
                except Exception as e:
                    LOGGER.error(f"Failed to load ckpt: {e}")
                    return 2, None, None
            
            if state_dict is not None:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    LOGGER.warning(f"load_state_dict mismatch: m={len(missing)}, u={len(unexpected)}")
        else:
            LOGGER.warning("Running with random initialization (no ckpt).")
            
        model = model.to(dev).eval()

    else:
        LOGGER.info("Using in-memory model.")
        # Tokenizerは軽量なので再作成
        tokenizer = create_tokenizer(cfg)
        model = model.to(dev).eval()

    max_len = int(infer_cfg.get("max_len", 128))
    batch_size = int(infer_cfg.get("batch_size", 32))

    # ---- 推論ループ ----
    outputs: Dict[str, Dict[str, Any]] = {}

    for split_name, rows in splits_to_infer.items():
        LOGGER.info(f"Inferring split: {split_name} (n={len(rows)})")
        
        # predict_with_emb は logits/labels/embs を返す
        logits, labels, embs = predict_with_emb(
            model,
            rows,
            tokenizer=tokenizer,
            bs=batch_size,
            max_len=max_len,
            device=dev.type,
        )
        
        # 予測クラス算出 (単純argmax)
        y_pred = logits.argmax(axis=-1)
        
        outputs[split_name] = {
            "logits": logits,
            "embs": embs,
            "labels": labels, # これは DatasetAdapter がパースした正解ラベル
            "y_pred": y_pred, 
            "rows": rows
        }

    # ---- 結果の保存と集約 ----
    do_save_csv = run_cfg.get("save_predictions", True)
    do_save_npy = run_cfg.get("save_logits", False)

    # 1. npy 書き出し
    if do_save_npy:
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
            # logits, embs は DataFrame には入れない
        })

        # 正解ラベル (あれば)
        if data.get("labels") is not None:
            df_split["y_true"] = data["labels"]
        else:
            # pool などで labels が無い場合は一応列確保（NaN）
            df_split["y_true"] = np.nan

        dfs.append(df_split)
        
        # Raw Outputs (Confidence計算用)
        raw_outputs[split_name] = {
            "logits": data["logits"],
            "embs": data["embs"],
            "labels": data["labels"],
            "y_pred": data["y_pred"], # step_confidenceで使用
            "ids": df_split["id"].tolist()
        }

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # 3. CSV 書き出し (Base)
    # ここでは Confidence 列がない状態だが、ステップ進行確認用に保存はしておく
    if do_save_csv and not final_df.empty:
        save_path = out_dir / "preds_detail.csv"
        final_df.to_csv(save_path, index=False)
        LOGGER.info(f"Saved base predictions to {save_path}")

    # meta 書き出し
    meta = {
        "generated_at": time.strftime("%F %T"),
        "ckpt": str(ckpt_path) if ckpt_path else "in-memory",
    }
    with open(out_dir / "infer.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if return_df:
        return 0, final_df, raw_outputs
    
    return 0, None, None


# =============================================================================
# CLI エントリポイント
# =============================================================================


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    """CLI ラッパー。"""
    if cfg is None and isinstance(argv, dict):
        cfg = argv
        argv = []
    if cfg is None:
        raise ValueError("tensaku.infer_pool.run requires cfg dict")

    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    
    data_dir = run_cfg.get("data_dir") or data_cfg.get("data_dir")
    out_dir_str = run_cfg.get("out_dir") or "./outputs"
    
    if not data_dir:
        LOGGER.error("run.data_dir is not set.")
        return 2

    files_cfg = data_cfg.get("files") or {}
    file_labeled = files_cfg.get("labeled", "labeled.jsonl")
    file_dev = files_cfg.get("dev", "dev.jsonl")
    file_pool = files_cfg.get("pool", "pool.jsonl")
    file_test = files_cfg.get("test", "test.jsonl")

    # JSONL 読み込み
    pool_rows = _read_jsonl(os.path.join(data_dir, file_pool))
    dev_rows = _read_jsonl(os.path.join(data_dir, file_dev))
    test_rows = _read_jsonl(os.path.join(data_dir, file_test))
    labeled_rows = _read_jsonl(os.path.join(data_dir, file_labeled))

    if not pool_rows and not dev_rows:
        LOGGER.error(f"pool or dev empty in {data_dir}")
        return 2

    split = DatasetSplit(
        labeled=labeled_rows,
        dev=dev_rows,
        test=test_rows,
        pool=pool_rows
    )

    out_dir = Path(out_dir_str)
    
    ret, _, _ = infer_core(split=split, out_dir=out_dir, cfg=cfg, return_df=False)
    return ret


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.info("Run via CLI: tensaku infer-pool -c <CFG.yaml>")
