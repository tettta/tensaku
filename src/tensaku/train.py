# /home/esakit25/work/tensaku/src/tensaku/train.py
# -*- coding: utf-8 -*-
"""
@module     : tensaku.train
@role       : 分類モデルの学習実行（I/O 分離・統合版）
@overview   :
    - train_core: DatasetSplit (メモリ上データ) を受け取り、学習を実行するコアロジック
    - run       : CLI 用ラッパー。JSONL ファイルを読み込み train_core を呼ぶ
"""

from __future__ import annotations

import math
import os
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score

# Phase2 連携用
from tensaku.data.base import DatasetSplit
# モデル定義 (Phase2)
from tensaku.models import create_model, create_tokenizer


# =============================================================================
# ユーティリティ
# =============================================================================


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"[train] ERROR: file not found: {path}")
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


def _load_meta_if_exists(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[train] WARN: failed to read meta.json: {path} ({e})")
        return {}
    if not isinstance(meta, dict):
        print(f"[train] WARN: meta.json is not a JSON object: {path}")
        return {}
    return meta


def _normalize_labels(
    rows: List[Dict[str, Any]], label_key: str, split_name: str
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[int]]:
    clean: List[Dict[str, Any]] = []
    bad = 0
    min_y: Optional[int] = None
    max_y: Optional[int] = None

    for r in rows:
        v = r.get(label_key, None)
        try:
            y = int(v)
        except Exception:
            bad += 1
            continue
        if y < 0:
            bad += 1
            continue
        r2 = dict(r)
        r2[label_key] = y
        clean.append(r2)
        if min_y is None or y < min_y:
            min_y = y
        if max_y is None or y > max_y:
            max_y = y

    if bad > 0:
        print(f"[train] warn: skipped {bad} rows in {split_name} due to invalid '{label_key}'")

    return clean, min_y, max_y


def _analyze_token_lengths(rows: List[Dict[str, Any]], tok, text_key: str, split_name: str):
    """データセットのトークン長統計を表示する"""
    lengths = []
    for r in rows:
        text = r.get(text_key, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        ids = tok.encode(str(text), add_special_tokens=True)
        lengths.append(len(ids))
    
    if not lengths:
        return
    
    lengths = np.array(lengths)
    print(f"[train] Token Stats ({split_name}): "
          f"Mean={lengths.mean():.1f}, "
          f"Median={np.median(lengths):.1f}, "
          f"Max={lengths.max()}, "
          f"95%tile={np.percentile(lengths, 95):.1f}")


class EssayDS(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tok,
        text_key: str = "mecab",
        label_key: str = "score",
        max_len: int = 128,
        with_label: bool = True,
    ) -> None:
        self.rows = rows
        self.tok = tok
        self.text_key = text_key
        self.label_key = label_key
        self.max_len = max_len
        self.with_label = with_label

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        r = self.rows[i]
        text = r.get(self.text_key, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item: Dict[str, torch.Tensor] = {k: v.squeeze(0) for k, v in enc.items()}
        if self.with_label:
            item["labels"] = torch.tensor(int(r[self.label_key]), dtype=torch.long)
        return item


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, n_class: int) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


@torch.no_grad()
def _eval_qwk_rmse(model, loader: DataLoader, device: torch.device, n_class: int) -> Tuple[float, float]:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    for batch in loader:
        for k, v in list(batch.items()):
            if k != "labels" and hasattr(v, "to"):
                batch[k] = v.to(device, non_blocking=True)
        logits = model(**{k: batch[k] for k in batch if k != "labels"}).logits
        pred = logits.argmax(-1).cpu().numpy()
        ys.append(batch["labels"].cpu().numpy())
        ps.append(pred)
    
    if not ys: return 0.0, 0.0
    
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    qwk = _qwk(y, p, n_class)
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    return qwk, rmse


# =============================================================================
# コアロジック (train_core)
# =============================================================================


def train_core(
    split: DatasetSplit,
    out_dir: Path,
    cfg: Mapping[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """
    メモリ上のデータ (split) を用いて学習を実行するコア関数。
    """
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    # out_dir の確保
    out_dir.mkdir(parents=True, exist_ok=True)

    text_key = data_cfg.get("text_key_primary", "mecab")
    label_key = data_cfg.get("label_key", "score")
    max_len = int(data_cfg.get("max_len", 128))

    seed = int(run_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # データの取得 (DatasetSplit -> List[Dict])
    # train には labeled を使用する
    rows_tr_raw = split.labeled
    rows_dv_raw = split.dev

    if not rows_tr_raw:
        print(f"[train_core] ERROR: missing labeled data.")
        return 1
    
    # ラベル正規化
    rows_tr, min_tr, max_tr = _normalize_labels(rows_tr_raw, label_key, "train/labeled")
    if rows_dv_raw:
        rows_dv, min_dv, max_dv = _normalize_labels(rows_dv_raw, label_key, "dev")
    else:
        rows_dv, min_dv, max_dv = [], None, None

    if not rows_tr:
        print("[train_core] ERROR: no valid training data.")
        return 1

    # クラス数決定ロジック
    max_candidates = [v for v in (max_tr, max_dv) if v is not None]
    max_y = max(max_candidates) if max_candidates else 0
    n_class_data = int(max_y) + 1

    cfg_num_labels_raw = model_cfg.get("num_labels")
    cfg_num_labels: Optional[int] = None
    if cfg_num_labels_raw is not None:
        try:
            cfg_num_labels = int(cfg_num_labels_raw)
            if cfg_num_labels <= 0: cfg_num_labels = None
        except: cfg_num_labels = None

    meta_num_labels: Optional[int] = None
    if meta:
        meta_num_labels_raw = meta.get("num_labels")
        try:
            if meta_num_labels_raw is not None:
                meta_num_labels = int(meta_num_labels_raw)
                if meta_num_labels <= 0: meta_num_labels = None
        except: meta_num_labels = None

    n_class_candidates = [n_class_data]
    if meta_num_labels is not None: n_class_candidates.append(meta_num_labels)
    if cfg_num_labels is not None: n_class_candidates.append(cfg_num_labels)
    n_class = max(n_class_candidates)
    
    # トークナイザ・モデル構築 (models.py)
    tok = create_tokenizer(cfg)
    model = create_model(cfg, n_class)

    print(f"[train_core] Analyzing token lengths (max_len={max_len})...")
    _analyze_token_lengths(rows_tr, tok, text_key, "Train")
    if rows_dv:
        _analyze_token_lengths(rows_dv, tok, text_key, "Dev")

    speed = model_cfg.get("speed", "full")

    ds_tr = EssayDS(rows_tr, tok, text_key, label_key, max_len, with_label=True)
    ds_dv = EssayDS(rows_dv, tok, text_key, label_key, max_len, with_label=True)

    bs = int(train_cfg.get("batch_size", 16))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0)
    dl_dv = DataLoader(ds_dv, batch_size=bs, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer設定
    if speed == "full":
        opt = AdamW(model.parameters(), lr=float(train_cfg.get("lr_full", 2e-5)))
    else:
        # Frozen or FE_sklearn (model.freeze_base=True は models.py で処理済み)
        # requires_grad=True のパラメータのみ Optimizer に渡す
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = AdamW(params, lr=float(train_cfg.get("lr_frozen", 5e-4)))

    epochs = int(train_cfg.get("epochs", 5))
    total_steps = epochs * max(1, math.ceil(len(ds_tr) / bs))
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=total_steps
    )

    ckpt_dir = out_dir / "checkpoints_min"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_qwk = -1.0
    patience = int(train_cfg.get("patience", -1))
    no_improve_cnt = 0
    if patience > 0:
        print(f"[train_core] Early Stopping ENABLED: patience={patience}")

    AMP_ENABLED = device.type == "cuda"
    AMP_DTYPE = torch.bfloat16
    scaler = GradScaler(enabled=AMP_ENABLED)

    print(
        f"[train_core] out_dir={out_dir} n_class={n_class} speed={speed} "
        f"device={device.type} epochs={epochs} batch_size={bs} max_len={max_len}"
    )

    for ep in range(1, epochs + 1):
        model.train()
        for batch in dl_tr:
            for k, v in list(batch.items()):
                if k != "labels" and hasattr(v, "to"):
                    batch[k] = v.to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                out = model(**{k: batch[k] for k in batch if k != "labels"}, labels=batch["labels"])
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        # Validation
        if rows_dv:
            qwk, rmse = _eval_qwk_rmse(model, dl_dv, device, n_class)
            print(f"[train_core] Epoch {ep}/{epochs}: Dev QWK={qwk:.4f}, RMSE={rmse:.4f}")
            
            # Save Last
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "qwk": qwk, "n_class": n_class},
                ckpt_dir / "last.pt",
            )
            
            # Save Best & Early Stopping
            if qwk > best_qwk:
                best_qwk = qwk
                no_improve_cnt = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": ep, "qwk": qwk, "n_class": n_class},
                    ckpt_dir / "best.pt",
                )
            else:
                if patience > 0:
                    no_improve_cnt += 1
                    if no_improve_cnt >= patience:
                        print(f"[train_core] Early stopping triggered at epoch {ep}")
                        break
        else:
            print(f"[train_core] Epoch {ep}/{epochs}: (No dev data)")
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "n_class": n_class},
                ckpt_dir / "best.pt",
            )

    print(f"[train_core] Training finished. Best QWK={best_qwk:.4f}")
    return 0


# =============================================================================
# CLI エントリポイント (run)
# =============================================================================


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    CLI 用ラッパー。
    Config からパスを解決し、JSONL を読み込んで DatasetSplit を構築し、train_core を呼ぶ。
    """
    if cfg is None and isinstance(argv, dict):
        cfg = argv
        argv = []
    if cfg is None:
        raise ValueError("tensaku.train.run requires cfg dict")

    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    
    data_dir = run_cfg.get("data_dir")
    if not data_dir:
        print("[train] ERROR: run.data_dir is not set.")
        return 1

    files = data_cfg.get("files") or {}
    fn_labeled = files.get("labeled") or files.get("train") or "labeled.jsonl"
    path_tr = os.path.join(data_dir, fn_labeled)
    path_dv = os.path.join(data_dir, files.get("dev", "dev.jsonl"))

    rows_tr = _read_jsonl(path_tr)
    rows_dv = _read_jsonl(path_dv)

    if not rows_tr:
        print(f"[train] ERROR: failed to load labeled data from {path_tr}")
        return 1
    # dev は空でも許容 (train_core 側でハンドリング)

    meta = _load_meta_if_exists(data_dir)

    split = DatasetSplit(
        labeled=rows_tr,
        dev=rows_dv,
        test=[],
        pool=[]
    )

    out_dir_str = run_cfg.get("out_dir", "./outputs")
    out_dir = Path(out_dir_str)

    # コアロジックへ委譲
    return train_core(split=split, out_dir=out_dir, cfg=cfg, meta=meta)


if __name__ == "__main__":
    print("[train] Run via CLI: tensaku train -c <CFG.yaml>")