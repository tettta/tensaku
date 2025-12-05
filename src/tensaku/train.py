# /home/esakit25/work/tensaku/src/tensaku/train.py
# -*- coding: utf-8 -*-
"""
@module     : tensaku.train
@role       : 分類モデルの学習実行（Early Stopping / トークン統計 / モデル分離 対応版）
"""

from __future__ import annotations

import math
import os
import json
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score

# ★新規作成した models.py をインポート
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

    if min_y is not None and max_y is not None:
        print(
            f"[train] {split_name}: n={len(clean)}  "
            f"{label_key}_min={min_y}  {label_key}_max={max_y}"
        )
    else:
        print(f"[train] ERROR: no valid labels in {split_name} (key='{label_key}')")

    return clean, min_y, max_y


def _analyze_token_lengths(rows: List[Dict[str, Any]], tok, text_key: str, split_name: str):
    """★追加: データセットのトークン長統計を表示する"""
    lengths = []
    # 高速化のため、データが多すぎる場合はサンプリングしても良いが、
    # 今回は数千件程度想定なので全件チェックする
    for r in rows:
        text = r.get(text_key, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        # padding/truncationなしで実際の長さを計測
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
        ys.append(batch["labels"].numpy())
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    qwk = _qwk(y, p, n_class)
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    return qwk, rmse


# =============================================================================
# メイン
# =============================================================================


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    if cfg is None and isinstance(argv, dict):
        cfg = argv
        argv = []
    if cfg is None:
        raise ValueError("tensaku.train.run requires cfg dict")

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    data_dir = run_cfg.get("data_dir")
    files = data_cfg.get("files") or {}

    meta = _load_meta_if_exists(data_dir)

    text_key = data_cfg.get("text_key_primary", "mecab")
    label_key = data_cfg.get("label_key", "score")
    max_len = int(data_cfg.get("max_len", 128))

    seed = int(run_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fn_labeled = files.get("labeled") or files.get("train") or "labeled.jsonl"
    path_tr = os.path.join(data_dir, fn_labeled)
    path_dv = os.path.join(data_dir, files.get("dev", "dev.jsonl"))

    rows_tr_raw = _read_jsonl(path_tr)
    rows_dv_raw = _read_jsonl(path_dv)

    if not rows_tr_raw or not rows_dv_raw:
        print(f"[train] ERROR: missing or empty data.", file=sys.stderr)
        return 1

    rows_tr, min_tr, max_tr = _normalize_labels(rows_tr_raw, label_key, "train/labeled")
    rows_dv, min_dv, max_dv = _normalize_labels(rows_dv_raw, label_key, "dev")

    if not rows_tr or not rows_dv:
        print("[train] ERROR: no usable data after label normalization.", file=sys.stderr)
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

    if cfg_num_labels and cfg_num_labels < n_class_data:
        print(f"[train] WARN: cfg < data; using {n_class}")
    
    bad_values = sorted({int(r[label_key]) for r in (rows_tr + rows_dv) if int(r[label_key]) < 0 or int(r[label_key]) >= n_class})
    if bad_values:
        print(f"[train] ERROR: label out of range: {bad_values[:10]}")
        return 1

    # ---- トークナイザ・モデル (models.py への委譲) ----
    # ★変更点: create_tokenizer / create_model を使用
    tok = create_tokenizer(cfg)
    model = create_model(cfg, n_class)

    # ▼▼▼ 追加機能: トークン長統計の表示 ▼▼▼
    print(f"[train] Analyzing token lengths (max_len={max_len})...")
    _analyze_token_lengths(rows_tr, tok, text_key, "Train")
    _analyze_token_lengths(rows_dv, tok, text_key, "Dev")
    # ▲▲▲ 追加ここまで ▲▲▲

    # Speed 設定は models.py で処理済みなのでここでは不要だが、Optimizerのパラメータグループ分けのために参照は必要
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
        # Frozen時、classifier層のみ学習
        # ※ model.classifier が無いアーキテクチャへの対応が必要な場合は models.py 側で
        #    パラメータグループを返すように拡張するのがベストだが、今回は簡易対応
        if hasattr(model, "classifier"):
            opt = AdamW(model.classifier.parameters(), lr=float(train_cfg.get("lr_frozen", 5e-4)))
        elif hasattr(model, "fc"): # 一部のモデル対応
            opt = AdamW(model.fc.parameters(), lr=float(train_cfg.get("lr_frozen", 5e-4)))
        else:
            # 万が一 classifier が見つからない場合は全パラメータを渡す（Freeze自体は models.py でされているので安全）
            print("[train] WARN: specific classifier layer not found, using all params (gradients rely on requires_grad).")
            opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(train_cfg.get("lr_frozen", 5e-4)))

    epochs = int(train_cfg.get("epochs", 5))
    total_steps = epochs * max(1, math.ceil(len(ds_tr) / bs))
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=total_steps
    )

    out_root = run_cfg.get("out_dir", "./outputs")
    ckpt_dir = os.path.join(out_root, "checkpoints_min")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_qwk = -1.0
    
    # ▼▼▼ 追加機能: Early Stopping の設定 ▼▼▼
    patience = int(train_cfg.get("patience", -1)) # デフォルトは無効(-1)
    no_improve_cnt = 0
    if patience > 0:
        print(f"[train] Early Stopping ENABLED: patience={patience}")
    # ▲▲▲ 追加ここまで ▲▲▲

    AMP_ENABLED = device.type == "cuda"
    AMP_DTYPE = torch.bfloat16
    scaler = GradScaler(enabled=AMP_ENABLED)

    print(
        f"[train] data_dir={data_dir}  out_dir={out_root} "
        f"n_class={n_class}  speed={speed}  device={device.type}  "
        f"epochs={epochs}  batch_size={bs}  max_len={max_len}"
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

        qwk, rmse = _eval_qwk_rmse(model, dl_dv, device, n_class)
        print(f"[train] epoch {ep}/{epochs}  dev: QWK={qwk:.4f}  RMSE={rmse:.4f}")
        
        # 最終エポック保存
        torch.save(
            {"model": model.state_dict(), "epoch": ep, "qwk": qwk, "n_class": n_class},
            os.path.join(ckpt_dir, "last.pt"),
        )
        
        # ベスト更新判定
        if qwk > best_qwk:
            best_qwk = qwk
            no_improve_cnt = 0
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "qwk": qwk, "n_class": n_class},
                os.path.join(ckpt_dir, "best.pt"),
            )
        else:
            # Early Stopping 判定
            if patience > 0:
                no_improve_cnt += 1
                if no_improve_cnt >= patience:
                    print(f"[train] Early stopping triggered at epoch {ep} (no improve for {patience} epochs)")
                    break

    print(f"[train] done. best QWK={best_qwk:.4f}  ckpt={ckpt_dir}/best.pt")
    return 0


if __name__ == "__main__":
    print("[train] Run via: python -m tensaku.cli train -c /path/to/cfg.yaml")