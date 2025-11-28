# /home/esakit25/work/tensaku/src/tensaku/train.py
# -*- coding: utf-8 -*-
"""
@module     : tensaku.train
@role       : 日本語BERTで総得点の分類モデルを学習し、dev QWK ベースで best.pt を保存する。
@inputs     :
  - CFG(dict):
      run.data_dir : {labeled|train}.jsonl / dev.jsonl があるディレクトリ
      run.out_dir  : 出力ディレクトリ（checkpoints_min/ を作成）
      run.seed     : 乱数シード（任意）
      data.files   : {"labeled"|"train","dev"} のファイル名（省略時は labeled.jsonl, dev.jsonl）
      data.text_key_primary : テキストキー（既定 "mecab"）
      data.label_key        : ラベルキー（既定 "score"）
      data.max_len          : 最大トークン長（既定 128）
      model.name   : HuggingFace モデル名（既定 cl-tohoku/bert-base-japanese-v3）
      model.speed  : "full" または "frozen"
      train.batch_size      : バッチサイズ（既定 16）
      train.epochs          : エポック数（既定 5）
      train.lr_full         : full 時の LR（既定 2e-5）
      train.lr_frozen       : frozen 時の LR（既定 5e-4）
@outputs    :
  - {out_dir}/checkpoints_min/best.pt : dev QWK ベースのベストモデル
  - {out_dir}/checkpoints_min/last.pt : 最終エポックのモデル
@cli        : tensaku train -c CFG.yaml [--set KEY=VAL ...]
@api        : run(argv: list[str] | None, cfg: dict[str,Any]) -> int
@contracts  :
  - ラベルは 0..N の int として扱う（_normalize_labels で負値は 0 にクリップ）。
  - クラス数 n_class は次の優先順位で決定する:
      1) cfg.model.num_labels が正の整数ならそれを使用
      2) data_dir/meta.json.num_labels が存在すれば、その値と train+dev の max(label)+1 の最大値を使用
      3) 上記が無ければ、train+dev の max(label)+1 から自動推定する
  - 入力データのうち label_key を数値に出来ない行はスキップし、件数を warn ログに出す。
  - CUDA の device-side assert で落ちる前に、ラベル範囲の問題は Python 側で検出する。
  - AMP は CUDA 利用時のみ有効。
  - meta.json.is_classification が False の場合は警告を出しつつ分類として学習を続行する。
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import cohen_kappa_score


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
                # 壊れ行はスキップ
                continue
    return rows


def _load_meta_if_exists(data_dir: str) -> Dict[str, Any]:
    """split が生成した meta.json があれば読み込んで dict を返す（無ければ {}）。"""
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
    """label_key を int>=0 に正規化しつつ統計を出す。失敗行はスキップ。"""
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


class EssayDS(Dataset):
    """単一問題の答案データセット（テキスト→BERT 入力テンソル）"""

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
            # _normalize_labels で int>=0 にしている前提だが、念のため int(...) で明示
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
    """
    メイン学習ルーチン。
    - CLI からは run(argv, cfg) 形式で呼ばれる。
    - 後方互換のため run(cfg)（argv に dict を渡す）も受け付ける。
    """
    # 後方互換: run(cfg) スタイルを許容
    if cfg is None and isinstance(argv, dict):
        cfg = argv  # type: ignore[assignment]
        argv = []
    if cfg is None:
        raise ValueError("tensaku.train.run requires cfg dict")

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    data_dir = run_cfg["data_dir"]
    files = data_cfg["files"]

    # split で生成された meta.json（ラベル統計など）を読み込む（存在しなければ {}）
    meta = _load_meta_if_exists(data_dir)

    text_key = data_cfg.get("text_key_primary", "mecab")

    label_key = data_cfg.get("label_key", "score")
    max_len = int(data_cfg.get("max_len", 128))

    seed = int(run_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- 入力ファイル（'labeled' / 'train' 両対応） ----
    fn_labeled = files.get("labeled") or files.get("train") or "labeled.jsonl"
    path_tr = os.path.join(data_dir, fn_labeled)
    path_dv = os.path.join(data_dir, files.get("dev", "dev.jsonl"))

    rows_tr_raw = _read_jsonl(path_tr)
    rows_dv_raw = _read_jsonl(path_dv)

    # 【修正】データ不足時はエラー(1)を返す
    if not rows_tr_raw or not rows_dv_raw:
        print(
            f"[train] ERROR: missing or empty data. "
            f"labeled/train={len(rows_tr_raw)}, dev={len(rows_dv_raw)} at {data_dir}",
            file=sys.stderr
        )
        return 1

    # ---- ラベル正規化＋統計 ----
    rows_tr, min_tr, max_tr = _normalize_labels(rows_tr_raw, label_key, "train/labeled")
    rows_dv, min_dv, max_dv = _normalize_labels(rows_dv_raw, label_key, "dev")

    # 【修正】正規化後にデータが残らない場合もエラー(1)を返す
    if not rows_tr or not rows_dv:
        print("[train] ERROR: no usable data after label normalization.", file=sys.stderr)
        return 1

    # クラス数候補の決定: train+dev / meta.json / cfg.model.num_labels
    max_candidates = [v for v in (max_tr, max_dv) if v is not None]
    max_y = max(max_candidates) if max_candidates else 0
    n_class_data = int(max_y) + 1  # 観測ラベルからの自動推定値

    # YAML 明示指定
    cfg_num_labels_raw = model_cfg.get("num_labels")
    cfg_num_labels: Optional[int] = None
    if cfg_num_labels_raw is not None:
        try:
            cfg_num_labels = int(cfg_num_labels_raw)
            if cfg_num_labels <= 0:
                cfg_num_labels = None
        except Exception:
            cfg_num_labels = None

    # meta.json（存在する場合）
    meta_num_labels: Optional[int] = None
    meta_is_classification: Optional[bool] = None
    if meta:
        meta_is_classification = meta.get("is_classification")
        meta_num_labels_raw = meta.get("num_labels")
        try:
            if meta_num_labels_raw is not None:
                meta_num_labels = int(meta_num_labels_raw)
                if meta_num_labels <= 0:
                    meta_num_labels = None
        except Exception:
            meta_num_labels = None

    n_class_candidates: List[int] = [n_class_data]
    n_class_sources: List[str] = ["train+dev"]

    if meta_num_labels is not None:
        n_class_candidates.append(meta_num_labels)
        n_class_sources.append("meta.num_labels")

    if cfg_num_labels is not None:
        n_class_candidates.append(cfg_num_labels)
        n_class_sources.append("cfg.model.num_labels")

    n_class = max(n_class_candidates)
    chosen_src = n_class_sources[n_class_candidates.index(n_class)]

    print(
        "[train] class count resolution: "
        f"data={n_class_data}, meta={meta_num_labels}, cfg={cfg_num_labels} "
        f"-> n_class={n_class} (from {chosen_src})"
    )

    # 設定とデータの不整合があれば warn（致命的ではないが気づけるように）
    if cfg_num_labels is not None and cfg_num_labels < n_class_data:
        print(
            f"[train] WARN: cfg.model.num_labels={cfg_num_labels} < "
            f"max(label)+1={n_class_data}; using n_class={n_class}."
        )

    if meta_num_labels is not None and meta_num_labels < n_class_data:
        print(
            f"[train] WARN: meta.num_labels={meta_num_labels} < "
            f"max(label)+1={n_class_data}; using n_class={n_class}."
        )

    if meta_is_classification is False:
        print(
            "[train] WARN: meta.json indicates is_classification=False, "
            "but tensaku.train assumes classification; proceeding as classification."
        )

    # ラベル範囲の最終チェック（CUDA の前に問題を検出）
    bad_values = sorted(
        {
            int(r[label_key])
            for r in (rows_tr + rows_dv)
            if int(r[label_key]) < 0 or int(r[label_key]) >= n_class
        }
    )
    if bad_values:
        print(
            f"[train] ERROR: label values out of range [0, {n_class-1}]: "
            f"{bad_values[:10]} (total {len(bad_values)})"
        )
        print("[train]       check data.label_key / data.files / data_dir in config.")
        # 【修正】ラベル範囲エラーも return 1
        return 1


    # ---- トークナイザ・モデル ----
    name = model_cfg.get("name", "cl-tohoku/bert-base-japanese-v3")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_class)

    # SPEED モード：full / frozen
    speed = model_cfg.get("speed", "full")
    if speed in ("frozen", "fe_sklearn"):
        for p in model.bert.parameters():
            p.requires_grad = False

    # ---- DataLoader ----
    ds_tr = EssayDS(rows_tr, tok, text_key, label_key, max_len, with_label=True)
    ds_dv = EssayDS(rows_dv, tok, text_key, label_key, max_len, with_label=True)

    bs = int(train_cfg.get("batch_size", 16))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0)
    dl_dv = DataLoader(ds_dv, batch_size=bs, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- Optimizer / Scheduler ----
    if speed == "full":
        opt = AdamW(model.parameters(), lr=float(train_cfg.get("lr_full", 2e-5)))
    else:
        opt = AdamW(model.classifier.parameters(), lr=float(train_cfg.get("lr_frozen", 5e-4)))

    epochs = int(train_cfg.get("epochs", 5))
    total_steps = epochs * max(1, math.ceil(len(ds_tr) / bs))
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=total_steps
    )

    out_root = run_cfg.get("out_dir", "./outputs")
    ckpt_dir = os.path.join(out_root, "checkpoints_min")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_qwk = -1.0

    AMP_ENABLED = device.type == "cuda"
    AMP_DTYPE = torch.bfloat16
    scaler = GradScaler(enabled=AMP_ENABLED)

    # ログ（以前の実装と近いフォーマットに合わせる）
    print(
        f"[train] data_dir={data_dir}  out_dir={out_root}  model={name}  "
        f"n_class={n_class}  speed={speed}  device={device.type}  "
        f"epochs={epochs}  batch_size={bs}  max_len={max_len}"
    )

    # ---- 学習ループ ----
    for ep in range(1, epochs + 1):
        model.train()
        for batch in dl_tr:
            for k, v in list(batch.items()):
                if k != "labels" and hasattr(v, "to"):
                    batch[k] = v.to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                enabled=AMP_ENABLED,
                dtype=AMP_DTYPE,
            ):
                out = model(
                    **{k: batch[k] for k in batch if k != "labels"},
                    labels=batch["labels"],
                )
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        qwk, rmse = _eval_qwk_rmse(model, dl_dv, device, n_class)
        print(f"[train] epoch {ep}/{epochs}  dev: QWK={qwk:.4f}  RMSE={rmse:.4f}")
        torch.save(
            {"model": model.state_dict(), "epoch": ep, "qwk": qwk, "n_class": n_class},
            os.path.join(ckpt_dir, "last.pt"),
        )
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "qwk": qwk,
                    "n_class": n_class,
                },
                os.path.join(ckpt_dir, "best.pt"),
            )

    print(f"[train] done. best QWK={best_qwk:.4f}  ckpt={ckpt_dir}/best.pt")
    return 0


if __name__ == "__main__":
    print("[train] Run via: python -m tensaku.cli train -c /path/to/cfg.yaml")