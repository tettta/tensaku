# /home/esakit25/work/tensaku/src/tensaku/train.py
# -*- coding: utf-8 -*-
"""
@module     : tensaku.train
@role       : 分類モデルの学習実行（Strict Mode）
@overview   :
    - train_core: メモリ上の DatasetSplit を受け取り学習を実行。
                  Config不備は即座にエラーとする（サイレントなフォールバック禁止）。
    - 安定化: AdamW(weight_decay), warmup, grad clipping, bf16時はGradScaler無効化。
"""

from __future__ import annotations

import logging
import math
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score

from tensaku.fs_core import ArtifactDir
from tensaku.data.base import DatasetSplit
from tensaku.models import create_model, create_tokenizer
from tensaku.utils.memlog import snapshot as mem_snapshot

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Utilities
# =============================================================================

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        LOGGER.error("File not found: %s", path)
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


def _normalize_labels(
    rows: List[Dict[str, Any]], label_key: str, split_name: str
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[int]]:
    clean: List[Dict[str, Any]] = []
    min_y: Optional[int] = None
    max_y: Optional[int] = None

    for i, r in enumerate(rows):
        v = r.get(label_key, None)
        try:
            y = int(v)
        except Exception as e:
            # [Strict修正] 変換不可は即エラー (警告スキップ禁止)
            raise ValueError(f"[{split_name}] Row {i}: label '{label_key}' is not int-castable: {v!r}") from e
        
        if y < 0:
            # [Strict修正] 負ラベルは即エラー (警告スキップ禁止)
            raise ValueError(f"[{split_name}] Row {i}: negative label found: {y}")

        r2 = dict(r)
        r2[label_key] = y
        clean.append(r2)
        
        if min_y is None or y < min_y:
            min_y = y
        if max_y is None or y > max_y:
            max_y = y

    # bad > 0 チェックは不要になったため削除 (ループ内でraiseするため)
    return clean, min_y, max_y


def _analyze_token_lengths(rows: List[Dict[str, Any]], tok, text_key: str, split_name: str) -> None:
    lengths: List[int] = []
    for r in rows:
        text = r.get(text_key, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        ids = tok.encode(str(text), add_special_tokens=True)
        lengths.append(len(ids))

    if not lengths:
        return

    arr = np.array(lengths)
    LOGGER.info(
        "Token Stats (%s): Mean=%.1f, Median=%.1f, Max=%d, 95%%tile=%.1f",
        split_name,
        float(arr.mean()),
        float(np.median(arr)),
        int(arr.max()),
        float(np.percentile(arr, 95)),
    )


class EssayDS(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tok,
        text_key: str,
        label_key: str,
        max_len: int,
        with_label: bool = True,
    ) -> None:
        self.rows = rows
        self.tok = tok
        self.text_key = text_key
        self.label_key = label_key
        self.max_len = int(max_len)
        self.with_label = bool(with_label)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        r = self.rows[i]
        text = r.get(self.text_key, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        enc = self.tok(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item: Dict[str, torch.Tensor] = {k: v.squeeze(0) for k, v in enc.items()}
        if self.with_label:
            item["labels"] = torch.tensor(int(r[self.label_key]), dtype=torch.long)
        return item


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


@torch.no_grad()
def _eval_qwk_rmse(model, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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

    if not ys:
        return 0.0, 0.0

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    qwk = _qwk(y, p)
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    return qwk, rmse


# =============================================================================
# Core
# =============================================================================

def train_core(
    split: DatasetSplit,
    ckpt_dir: ArtifactDir,
    cfg: Mapping[str, Any],
    return_model: bool = False,
    save_checkpoints: bool = True,
) -> Tuple[int, Optional[Any]]:
    # ---- section presence ----
    if "run" not in cfg:
        raise KeyError("cfg missing 'run' section")
    if "data" not in cfg:
        raise KeyError("cfg missing 'data' section")
    if "model" not in cfg:
        raise KeyError("cfg missing 'model' section")
    if "train" not in cfg:
        raise KeyError("cfg missing 'train' section")

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    # ---- required keys ----
    text_key = data_cfg["text_key_primary"]
    label_key = data_cfg["label_key"]
    max_len = int(data_cfg["max_len"])

    speed = model_cfg["speed"]  # "full" or "frozen"
    if "num_labels" not in model_cfg:
        raise KeyError("Config 'model.num_labels' is required (Strict Mode).")
    cfg_num_labels = int(model_cfg["num_labels"])

    # optimization required keys
    if "weight_decay" not in train_cfg:
        raise KeyError("Config 'train.weight_decay' is required (e.g. 0.01).")
    if "warmup_ratio" not in train_cfg:
        raise KeyError("Config 'train.warmup_ratio' is required (e.g. 0.06).")
    if "max_grad_norm" not in train_cfg:
        raise KeyError("Config 'train.max_grad_norm' is required (e.g. 1.0).")

    weight_decay = float(train_cfg["weight_decay"])
    warmup_ratio = float(train_cfg["warmup_ratio"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(f"train.warmup_ratio must be in [0,1), got {warmup_ratio}")

    # ---- seed ----
    seed = int(run_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- data ----
    rows_tr_raw = split.labeled
    rows_dv_raw = split.dev

    # --- memlog ---
    mem_snapshot(event="train_start", extra={"ckpt_dir": str(ckpt_dir.path), "n_labeled": len(rows_tr_raw), "n_dev": len(rows_dv_raw)})

    if not rows_tr_raw:
        LOGGER.error("Missing labeled data.")
        return 1, None

    # [Strict修正] 不正データがあればここで落ちる
    rows_tr, min_tr, max_tr = _normalize_labels(rows_tr_raw, label_key, "train/labeled")
    
    if rows_dv_raw:
        rows_dv, min_dv, max_dv = _normalize_labels(rows_dv_raw, label_key, "dev")
    else:
        rows_dv, min_dv, max_dv = [], None, None

    LOGGER.info(
        "Label Stats (Train/Labeled): n=%d min=%s max=%s; (Dev): n=%d min=%s max=%s",
        len(rows_tr),
        str(min_tr),
        str(max_tr),
        len(rows_dv),
        str(min_dv),
        str(max_dv),
    )

    if not rows_tr:
        LOGGER.error("No valid training data after normalization.")
        return 1, None

    # ---- num_labels check ----
    # 負の値チェックは _normalize_labels で済んでいるが、最大値チェックはここで実施
    max_candidates = [v for v in (max_tr, max_dv) if v is not None]
    max_y = max(max_candidates) if max_candidates else 0
    n_class_data = int(max_y) + 1

    if cfg_num_labels < n_class_data:
        raise ValueError(
            f"Config num_labels={cfg_num_labels} is smaller than actual data max label "
            f"{n_class_data - 1} (requires {n_class_data})."
        )

    n_class = cfg_num_labels

    # ---- model ----
    tok = create_tokenizer(cfg)
    model = create_model(cfg, n_class)

    base_trainable = any(p.requires_grad for n, p in model.named_parameters() if n.startswith("bert."))
    if speed == "full" and not base_trainable:
        raise ValueError("speed=full but base model parameters are frozen.")
    if speed == "frozen" and base_trainable:
        raise ValueError("speed=frozen but base model parameters are trainable.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("Trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1))

    # Optional warm-start
    if "init_ckpt" in train_cfg and train_cfg["init_ckpt"]:
        init_path = str(train_cfg["init_ckpt"])
        LOGGER.info("Warm-start: loading init_ckpt=%s", init_path)
        state = torch.load(init_path, map_location="cpu")
        sd = state.get("model", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            LOGGER.warning("init_ckpt missing keys: %d (show up to 20): %s", len(missing), missing[:20])
        if unexpected:
            LOGGER.warning("init_ckpt unexpected keys: %d (show up to 20): %s", len(unexpected), unexpected[:20])

    # ---- stats ----
    LOGGER.info("Analyzing token lengths (max_len=%d)...", max_len)
    _analyze_token_lengths(rows_tr, tok, text_key, "Train")
    if rows_dv:
        _analyze_token_lengths(rows_dv, tok, text_key, "Dev")

    # ---- loaders ----
    ds_tr = EssayDS(rows_tr, tok, text_key, label_key, max_len, with_label=True)
    ds_dv = EssayDS(rows_dv, tok, text_key, label_key, max_len, with_label=True)

    bs = int(train_cfg["batch_size"])
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0)
    dl_dv = DataLoader(ds_dv, batch_size=bs, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- optimizer ----
    if speed == "full":
        lr = float(train_cfg["lr_full"])
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif speed == "frozen":
        lr = float(train_cfg["lr_frozen"])
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown speed mode: {speed}")

    # ---- schedule ----
    epochs = int(train_cfg["epochs"])
    total_steps = epochs * max(1, math.ceil(len(ds_tr) / bs))
    warmup_steps = int(round(total_steps * warmup_ratio))
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- early stopping ----
    patience = int(train_cfg["patience"])
    no_improve_cnt = 0
    early_stop_min_epochs = int(train_cfg.get("early_stop_min_epochs", 0))

    if patience > 0:
        LOGGER.info("Early Stopping ENABLED: patience=%d", patience)

    # ---- AMP ----
    AMP_ENABLED = device.type == "cuda"
    AMP_DTYPE = torch.bfloat16  # default: bf16
    scaler = GradScaler(enabled=(AMP_ENABLED and AMP_DTYPE == torch.float16))

    LOGGER.info(
        "Start training: out_file=%s n_class=%d speed=%s device=%s epochs=%d batch_size=%d max_len=%d",
        str(ckpt_dir),
        n_class,
        speed,
        device.type,
        epochs,
        bs,
        max_len,
    )
    LOGGER.info(
        "Optim: lr=%g weight_decay=%g warmup_steps=%d/%d max_grad_norm=%g AMP=%s dtype=%s scaler=%s",
        lr,
        weight_decay,
        warmup_steps,
        total_steps,
        max_grad_norm,
        AMP_ENABLED,
        str(AMP_DTYPE),
        scaler.is_enabled(),
    )

    best_qwk = -1.0

    for ep in range(1, epochs + 1):
        mem_snapshot(event="train_epoch_start", extra={"epoch": ep, "epochs": epochs})
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in dl_tr:
            for k, v in list(batch.items()):
                if k != "labels" and hasattr(v, "to"):
                    batch[k] = v.to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                out = model(**{k: batch[k] for k in batch if k != "labels"}, labels=batch["labels"])
                loss = out.loss

            train_loss_sum += float(loss.detach().cpu())
            n_batches += 1

            # backprop
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

            sched.step()

        train_loss_avg = train_loss_sum / max(n_batches, 1)

        # validation
        if rows_dv:
            qwk, rmse = _eval_qwk_rmse(model, dl_dv, device)
            LOGGER.info(
                "Epoch %d/%d: TrainLoss=%.4f  Dev QWK=%.4f, RMSE=%.4f",
                ep,
                epochs,
                train_loss_avg,
                qwk,
                rmse,
            )

            # save last (avoid duplicate ledger entries on overwrite)
            if save_checkpoints:
                record_last = not ckpt_dir.last.exists()
                ckpt_dir.last.save_checkpoint(
                    {"model": model.state_dict(), "epoch": ep, "qwk": qwk, "n_class": n_class},
                    record=record_last,
                )

            # save best / early stop
            if qwk > best_qwk:
                best_qwk = qwk
                no_improve_cnt = 0
                if save_checkpoints:
                    record_best = not ckpt_dir.best.exists()
                    ckpt_dir.best.save_checkpoint(
                        {"model": model.state_dict(), "epoch": ep, "qwk": qwk, "n_class": n_class},
                        record=record_best,
                    )
            else:
                if patience > 0 and ep >= early_stop_min_epochs:
                    no_improve_cnt += 1
                    if no_improve_cnt >= patience:
                        LOGGER.info("Early stopping triggered at epoch %d", ep)
                        break
        else:
            LOGGER.info("Epoch %d/%d: TrainLoss=%.4f  (No dev data)", ep, epochs, train_loss_avg)
            if save_checkpoints:
                record_best = not ckpt_dir.best.exists()
                ckpt_dir.best.save_checkpoint(
                    {"model": model.state_dict(), "epoch": ep, "n_class": n_class},
                    record=record_best,
                )

    mem_snapshot(event="train_end", extra={"best_qwk": best_qwk})
    LOGGER.info("Training finished. Best QWK=%.4f", best_qwk)

    if return_model:
        if hasattr(ckpt_dir, "best") and ckpt_dir.best.exists():
            try:
                state = torch.load(ckpt_dir.best.path, map_location=device)
                sd = state.get("model", state) if isinstance(state, dict) else state
                model.load_state_dict(sd, strict=False)
                model.eval()
                LOGGER.info("Reloaded best model state for in-memory return.")
                return 0, model
            except Exception as e:
                LOGGER.error("Failed to reload best model: %s", e)
                return 1, None
        model.eval()
        return 0, model

    return 0, None


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    # CLI wrapper (Strict)
    if cfg is None:
        raise ValueError("tensaku.train.run requires cfg dict (Strict Mode)")

    run_cfg = cfg["run"]
    data_dir = run_cfg["data_dir"]

    data_cfg = cfg["data"]
    # ここはプロジェクト運用上ほぼ使われないため、ファイル名は cfg 側で指定することを推奨。
    files = data_cfg.get("files", {})
    fn_labeled = files.get("labeled")
    fn_dev = files.get("dev")

    if not fn_labeled:
        raise KeyError("cfg.data.files.labeled is required for CLI run()")
    if not fn_dev:
        raise KeyError("cfg.data.files.dev is required for CLI run()")

    rows_tr = _read_jsonl(os.path.join(data_dir, fn_labeled))
    rows_dv = _read_jsonl(os.path.join(data_dir, fn_dev))

    if not rows_tr:
        LOGGER.error("Failed to load labeled data.")
        return 1

    split = DatasetSplit(labeled=rows_tr, dev=rows_dv, test=[], pool=[])

    out_dir_str = run_cfg["out_dir"]
    out_dir = Path(out_dir_str)

    try:
        ckpt_dir = ArtifactDir(out_dir)  # type: ignore[misc]
    except TypeError:
        try:
            ckpt_dir = ArtifactDir(path=out_dir)  # type: ignore[call-arg]
        except TypeError as e:
            raise TypeError(
                "Failed to construct ArtifactDir from run.out_dir. "
                "Use pipeline/task execution, or adjust ArtifactDir constructor handling in train.run()."
            ) from e

    rc, _ = train_core(split=split, ckpt_dir=ckpt_dir, cfg=cfg, return_model=False)
    return rc


if __name__ == "__main__":
    pass