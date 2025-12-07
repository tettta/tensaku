# /home/esakit25/work/tensaku/src/tensaku/infer_pool.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.infer_pool
@role     : dev / pool / test に対して ckpt で一括推論し、予測CSVと簡易メタ情報を out_dir に保存する。
@overview :
    - infer_core: DatasetSplit を受け取り、推論・TrustScore計算・CSV保存を行う。
    - run       : CLI 用ラッパー。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import torch

from tensaku.data.base import DatasetSplit
# Phase2 モデル定義
from tensaku.models import create_model, create_tokenizer
from .model_io import select_device
from .embed import labels_from_rows, predict_with_emb

try:
    from .trustscore import TrustScorer
except Exception:
    TrustScorer = None

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


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    s = np.sum(ex, axis=axis, keepdims=True)
    return ex / np.clip(s, 1e-12, None)


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


def _pred_and_conf(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prob = _softmax(logits, axis=-1)
    y_pred = prob.argmax(axis=-1).astype(int)
    idx = np.arange(prob.shape[0])
    conf = prob[idx, y_pred]
    return y_pred, conf


def _compute_trust_for_splits(
    trust: bool,
    labeled_rows: List[dict],
    dev_rows: List[dict],
    pool_rows: List[dict],
    test_rows: List[dict],
    outputs: Dict[str, Dict[str, Any]],
    label_key: str,
    trust_k: int,
    trust_metric: str,
    trust_version: str,
) -> Dict[str, Optional[np.ndarray]]:
    trust_dict: Dict[str, Optional[np.ndarray]] = {"dev": None, "pool": None, "test": None}
    if not trust:
        return trust_dict
    if TrustScorer is None:
        print("[infer_core] WARN: TrustScorer が利用できないため --trust を無視します。", file=sys.stderr)
        return trust_dict

    train_name = None
    train_rows: List[dict] = []
    
    # 学習データの埋め込みが必要
    if "labeled" in outputs and outputs["labeled"].get("embs") is not None and labeled_rows:
        train_name = "labeled"
        train_rows = labeled_rows
    elif "dev" in outputs and outputs["dev"].get("embs") is not None and dev_rows:
        train_name = "dev"
        train_rows = dev_rows
    else:
        print("[infer_core] WARN: TrustScore 学習用の埋め込みが無いため計算をスキップします。", file=sys.stderr)
        return trust_dict

    train_embs = outputs[train_name]["embs"]
    train_labels = labels_from_rows(train_rows, label_key=label_key)
    if train_labels is None:
        print(f"[infer_core] WARN: {train_name} にラベルが無いため TrustScore を計算できません。", file=sys.stderr)
        return trust_dict

    scorer = TrustScorer(
        version=trust_version,
        metric=trust_metric,
        k=int(trust_k),
        normalize="zscore",
        robust=(trust_version == "v2"),
    )
    scorer.fit(train_embs, train_labels)

    if "dev" in outputs and outputs["dev"].get("embs") is not None:
        trust_dict["dev"] = scorer.score(outputs["dev"]["embs"], outputs["dev"]["y_pred"])
    if "pool" in outputs and outputs["pool"].get("embs") is not None:
        trust_dict["pool"] = scorer.score(outputs["pool"]["embs"], outputs["pool"]["y_pred"])
    if "test" in outputs and outputs["test"].get("embs") is not None:
        trust_dict["test"] = scorer.score(outputs["test"]["embs"], outputs["test"]["y_pred"])

    return trust_dict


# =============================================================================
# infer_core
# =============================================================================


def infer_core(
    split: DatasetSplit,
    out_dir: Path,
    cfg: Mapping[str, Any],
) -> int:
    """
    メモリ上の DatasetSplit を受け取り、推論を実行するコア関数。
    """
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("infer", {})

    out_dir.mkdir(parents=True, exist_ok=True)

    label_key = data_cfg.get("label_key", "score")
    id_key = data_cfg.get("id_key", "id")

    # ---- データ取り出し ----
    labeled_rows = split.labeled
    dev_rows = split.dev
    pool_rows = split.pool
    test_rows = split.test

    if not pool_rows and not dev_rows and not test_rows:
        print("[infer_core] WARN: All target splits (dev/pool/test) are empty.")
        return 0

    # ---- モデル名・ckpt ----
    model_name = model_cfg.get("name") or DEF_BASE
    ckpt_path = infer_cfg.get("ckpt") or model_cfg.get("ckpt")
    
    if not ckpt_path:
        base_out = run_cfg.get("out_dir", "./outputs")
        ckpt_path = os.path.join(base_out, "checkpoints_min", "best.pt")

    allow_random = bool(infer_cfg.get("allow_random", False))
    has_ckpt = os.path.exists(ckpt_path) if ckpt_path else False

    if not has_ckpt and not allow_random:
        print(f"[infer_core] ERROR: ckpt が見つかりません: {ckpt_path}", file=sys.stderr)
        return 2

    # ---- n_class の決定 ----
    n_class: Optional[int] = model_cfg.get("num_labels")
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
            print(f"[infer_core] WARN: Failed to inspect ckpt for num_labels: {e}")

    if n_class is None:
        n_class = int(model_cfg.get("num_labels_fallback", 6))

    # ---- tokenizer / model 構築 ----
    print(f"[infer_core] Creating model (n_class={n_class})...")
    tokenizer = create_tokenizer(cfg)
    model = create_model(cfg, n_class)

    # ckpt ロード
    if has_ckpt:
        if state_dict is None:
            try:
                bundle = torch.load(ckpt_path, map_location="cpu")
                sd_raw = _extract_state_dict(bundle)
                state_dict = _normalize_keys(sd_raw) if sd_raw is not None else None
            except Exception as e:
                print(f"[infer_core] ERROR: Failed to load ckpt: {e}", file=sys.stderr)
                return 2
        
        if state_dict is not None:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[infer_core] WARN: load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}", file=sys.stderr)
        else:
            print("[infer_core] WARN: ckpt から有効な state_dict を取り出せませんでした。", file=sys.stderr)
    else:
        print("[infer_core] WARN: Running with random initialization (no ckpt).")

    device_name = infer_cfg.get("device", "auto")
    dev = select_device(device_name)
    model = model.to(dev).eval()

    max_len = int(infer_cfg.get("max_len", 128))
    batch_size = int(infer_cfg.get("batch_size", 32))

    trust_flag = bool(infer_cfg.get("trust", False))
    trust_k = int(infer_cfg.get("trust_k", 1))
    trust_metric = infer_cfg.get("trust_metric", "cosine")
    trust_version = infer_cfg.get("trust_version", "v2")

    # ---- 各 split で埋め込み＋logits を計算 ----
    outputs: Dict[str, Dict[str, Any]] = {}

    def _process_split(name: str, rows: List[dict]) -> None:
        if not rows:
            return
        logits, labels, embs = predict_with_emb(
            model,
            rows,
            tokenizer=tokenizer,
            bs=batch_size,
            max_len=max_len,
            device=dev.type,
        )
        y_pred, conf_msp = _pred_and_conf(logits)
        outputs[name] = {
            "rows": rows,
            "logits": logits,
            "labels": labels,
            "embs": embs,
            "y_pred": y_pred,
            "conf_msp": conf_msp,
        }

    # TrustScore学習用に labeled も計算
    if trust_flag and labeled_rows:
        _process_split("labeled", labeled_rows)

    _process_split("dev", dev_rows)
    _process_split("pool", pool_rows)
    _process_split("test", test_rows)

    # ---- Trust Score ----
    trust_dict = _compute_trust_for_splits(
        trust=trust_flag,
        labeled_rows=labeled_rows,
        dev_rows=dev_rows,
        pool_rows=pool_rows,
        test_rows=test_rows,
        outputs=outputs,
        label_key=label_key,
        trust_k=trust_k,
        trust_metric=trust_metric,
        trust_version=trust_version,
    )

    # ---- 書き出し ----
    for split_name in ("dev", "pool", "test"):
        if split_name in outputs:
            # logits / embs
            if outputs[split_name].get("logits") is not None:
                np.save(out_dir / f"{split_name}_logits.npy", outputs[split_name]["logits"])
            if outputs[split_name].get("embs") is not None:
                np.save(out_dir / f"{split_name}_embs.npy", outputs[split_name]["embs"])

    # CSV 書き出し
    def _write_csv(path_csv: Path, data: Dict[str, Any], trust_vals: Optional[np.ndarray], has_true: bool) -> None:
        rows = data["rows"]
        y_pred = data["y_pred"]
        conf_msp = data["conf_msp"]
        
        # ヘッダ作成 (y_true は必要な場合のみ)
        header = ["id", "y_pred", "conf_msp", "conf_trust"]
        if has_true:
            header.append("y_true")

        with open(path_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i, (r, yp, cm) in enumerate(zip(rows, y_pred, conf_msp)):
                rid = r.get(id_key, str(i))
                
                # スコア初期化
                tr_val = trust_vals[i] if trust_vals is not None else 0.0
                
                row_data = [rid, int(yp), float(cm), float(tr_val)]
                
                if has_true:
                    try: yt = int(r.get(label_key, -1))
                    except: yt = -1
                    row_data.append(yt)
                    
                writer.writerow(row_data)

    if "dev" in outputs:
        _write_csv(out_dir / "dev_preds.csv", outputs["dev"], trust_dict.get("dev"), True)
    if "pool" in outputs:
        _write_csv(out_dir / "pool_preds.csv", outputs["pool"], trust_dict.get("pool"), False)
    if "test" in outputs and outputs["test"]["rows"]:
        _write_csv(out_dir / "test_preds.csv", outputs["test"], trust_dict.get("test"), True)

    # meta 書き出し
    meta = {
        "model_name": model_name,
        "ckpt": ckpt_path if has_ckpt else None,
        "n_class": int(n_class),
        "generated_at": time.strftime("%F %T"),
    }
    with open(out_dir / "infer.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return 0


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
        print("[infer_pool] ERROR: run.data_dir is not set.")
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
        print(f"[infer_pool] ERROR: pool or dev empty in {data_dir}", file=sys.stderr)
        return 2

    split = DatasetSplit(
        labeled=labeled_rows,
        dev=dev_rows,
        test=test_rows,
        pool=pool_rows
    )

    out_dir = Path(out_dir_str)
    
    return infer_core(split=split, out_dir=out_dir, cfg=cfg)


if __name__ == "__main__":
    print("[infer-pool] Run via CLI: tensaku infer-pool -c <CFG.yaml>")