# /home/esakit25/work/tensaku/src/tensaku/eval.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.eval
@role     : 学習済み ckpt を読み込み、dev/test を評価して指標（QWK/RMSE）と予測CSVを出力する
@inputs   :
  - cfg["run"].data_dir : dev/test JSONL があるディレクトリ
  - cfg["run"].out_dir  : 出力ディレクトリ
  - cfg["infer"].ckpt   : 任意の ckpt パス（省略時は {out_dir}/checkpoints_min/best.pt）
  - cfg["model"].name   : ベースモデル名（HF hub or ローカルパス）
  - cfg["data"].files   : dev/test のファイル名（dev.jsonl/test.jsonl が既定）
  - cfg["data"].text_key_primary : テキストキー（既定 "mecab"）
  - cfg["data"].label_key        : ラベルキー（既定 "score"）
  - cfg["data"].max_len          : 最大トークン長（既定 128）
@outputs  :
  - {out_dir}/dev_preds.csv  : id,y_true,y_pred,conf_msp
  - {out_dir}/test_preds.csv : id,y_true,y_pred,conf_msp（test.jsonl がある場合）
  - {out_dir}/eval.meta.json : 使用 ckpt / n_class / 指標などのメタ情報
@cli      : tensaku eval -c CFG.yaml [--target dev|test|both] [--batch 32]
@api      : run(argv: list[str] | None, cfg: dict[str,Any]) -> int
@contracts:
  - ラベルは 0..N の int として扱う（N+1 がクラス数）。
  - CUDA device-side assert を避けるため、Python 側でラベル範囲チェックを行う。
  - モデル読み込みは ckpt→pretrained の順で優先し、失敗しても安全に終了する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =============================================================================
# I/O ユーティリティ
# =============================================================================


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"[eval] WARN: file not found: {path}")
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


def _normalize_labels(
    rows: List[Dict[str, Any]], label_key: str
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
        print(f"[eval] warn: skipped {bad} rows due to invalid '{label_key}'")
    return clean, min_y, max_y


class EvalDataset(Dataset):
    """評価用 Dataset（dev/test 共通）。"""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer,
        text_key: str = "mecab",
        label_key: str = "score",
        max_len: int = 128,
        with_label: bool = True,
        id_key: str = "id",
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.label_key = label_key
        self.max_len = max_len
        self.with_label = with_label
        self.id_key = id_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        text = r.get(self.text_key, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item: Dict[str, Any] = {k: v.squeeze(0) for k, v in enc.items()}
        # ID はそのまま Python オブジェクトとして保持（Tensorにはしない）
        item["id"] = r.get(self.id_key, idx)
        if self.with_label:
            item["labels"] = torch.tensor(int(r[self.label_key]), dtype=torch.long)
        return item


# =============================================================================
# モデル / ckpt ロード
# =============================================================================


def _infer_num_labels_from_state(sd: Dict[str, Any]) -> Optional[int]:
    for key in ["classifier.weight", "score.weight", "head.weight"]:
        if key in sd and hasattr(sd[key], "shape"):
            return int(sd[key].shape[0])
    for k, v in sd.items():
        if k.endswith("classifier.weight") and hasattr(v, "shape"):
            return int(v.shape[0])  # type: ignore[arg-type]
    return None


def _extract_state_dict(bundle: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """多様な ckpt 形式から state_dict とメタを取り出す。"""
    meta: Dict[str, Any] = {}
    if isinstance(bundle, dict):
        # 典型パターン: {"model": sd, "epoch":..., "qwk":..., "n_class":...}
        for key in ["state_dict", "model", "model_state_dict", "net"]:
            v = bundle.get(key)
            if isinstance(v, dict):
                meta = {k: v2 for k, v2 in bundle.items() if k != key}
                return v, meta
        # 直接 state_dict の場合
        if all(hasattr(v, "shape") for v in bundle.values()):
            return bundle, meta
    return None, meta


def _load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    ckpt_path: Optional[str],
    device: torch.device,
) -> Tuple[Optional[Any], Optional[Any], Optional[int], Dict[str, Any]]:
    """Tokenizer + Model を初期化し、必要なら ckpt を適用。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(num_labels),
        ignore_mismatched_sizes=True,
    )
    meta: Dict[str, Any] = {"model_name": model_name, "num_labels": int(num_labels)}
    n_class_ckpt: Optional[int] = None

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            bundle = torch.load(ckpt_path, map_location="cpu")
            sd, ck_meta = _extract_state_dict(bundle)
            if ck_meta:
                meta.update(ck_meta)
            if sd is not None:
                n_class_ckpt = _infer_num_labels_from_state(sd) or ck_meta.get("n_class")
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing:
                    meta["missing_keys"] = list(missing)
                if unexpected:
                    meta["unexpected_keys"] = list(unexpected)
                print(f"[eval] loaded ckpt: {ckpt_path}")
            else:
                print(f"[eval] WARN: could not extract state_dict from ckpt: {ckpt_path}")
        except Exception as e:  # pragma: no cover - 例外ログのみ
            print(f"[eval] WARN: failed to load ckpt {ckpt_path}: {e}")
    else:
        if ckpt_path:
            print(f"[eval] WARN: ckpt not found, using pretrained only: {ckpt_path}")
        else:
            print("[eval] INFO: ckpt not specified, using pretrained only.")

    model.to(device)
    return model, tokenizer, n_class_ckpt, meta


# =============================================================================
# 評価ロジック
# =============================================================================


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_pred - y_true) ** 2).mean()))


@torch.no_grad()
def _predict(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[Any], List[int], List[int], List[float]]:
    """DataLoader から id / y_true / y_pred / conf_msp をまとめて取る。"""
    model.eval()
    all_ids: List[Any] = []
    all_true: List[Optional[int]] = []
    all_pred: List[int] = []
    all_conf: List[float] = []

    for batch in loader:
        ids = batch.pop("id")
        labels = batch.get("labels")
        # device 転送
        for k, v in list(batch.items()):
            if k != "labels" and hasattr(v, "to"):
                batch[k] = v.to(device, non_blocking=True)
        if labels is not None:
            labels = labels.to(device, non_blocking=True)

        logits = model(**{k: batch[k] for k in batch if k != "labels"}).logits
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        all_ids.extend(list(ids))
        if labels is not None:
            all_true.extend([int(x) for x in labels.cpu().numpy().tolist()])
        else:
            all_true.extend([None] * logits.size(0))
        all_pred.extend([int(x) for x in pred.cpu().numpy().tolist()])
        all_conf.extend([float(x) for x in probs.max(dim=-1).values.cpu().numpy().tolist()])

    return all_ids, all_true, all_pred, all_conf


def _write_preds_csv(
    path: str,
    ids: Sequence[Any],
    y_true: Sequence[Optional[int]],
    y_pred: Sequence[int],
    conf: Sequence[float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "y_true", "y_pred", "conf_msp"])
        for i, yt, yp, c in zip(ids, y_true, y_pred, conf):
            w.writerow([i, "" if yt is None else int(yt), int(yp), float(c)])


# =============================================================================
# メイン run
# =============================================================================


def run(argv: Optional[List[str]] = None, cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    メイン関数（cli.py から呼ばれる想定）。
    - argv: ["--target", "both", "--batch", "32"] のような引数列
    - cfg : tensaku.config.load_config(...) で読み込まれた dict
    """
    # 後方互換: run(cfg) 形式にも対応
    if cfg is None and isinstance(argv, dict):
        cfg = argv  # type: ignore[assignment]
        argv = []
    if cfg is None:
        raise ValueError("tensaku.eval.run requires cfg dict")

    if argv is None:
        argv = []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--batch", type=int, default=32)
    ns, _rest = parser.parse_known_args(argv)

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("infer", {})

    data_dir = run_cfg["data_dir"]
    out_dir = run_cfg["out_dir"]
    files = data_cfg.get("files", {})
    text_key = data_cfg.get("text_key_primary", "mecab")
    label_key = data_cfg.get("label_key", "score")
    max_len = int(data_cfg.get("max_len", 128))

    dev_name = files.get("dev", "dev.jsonl")
    test_name = files.get("test", "test.jsonl")
    path_dev = os.path.join(data_dir, dev_name)
    path_test = os.path.join(data_dir, test_name)

    rows_dev_raw = _read_jsonl(path_dev) if ns.target in ("dev", "both") else []
    rows_test_raw = _read_jsonl(path_test) if ns.target in ("test", "both") else []

    if not rows_dev_raw and not rows_test_raw:
        print(f"[eval] ERROR: no dev/test data found in {data_dir}")
        return 1

    # ラベル正規化（dev/test 合わせてクラス数を決める）
    rows_dev, min_d, max_d = _normalize_labels(rows_dev_raw, label_key) if rows_dev_raw else ([], None, None)
    rows_test, min_t, max_t = _normalize_labels(rows_test_raw, label_key) if rows_test_raw else ([], None, None)

    max_candidates = [v for v in (max_d, max_t) if v is not None]
    if max_candidates:
        max_y_data = int(max(max_candidates))
        n_class_data = max_y_data + 1
    else:
        n_class_data = None

    # モデル・ckpt・クラス数
    model_name = model_cfg.get("name", "cl-tohoku/bert-base-japanese-v3")
    out_ckpt_dir = os.path.join(out_dir, "checkpoints_min")
    ckpt_from_infer = infer_cfg.get("ckpt")
    ckpt_default = os.path.join(out_ckpt_dir, "best.pt")
    ckpt_path = ckpt_from_infer or ckpt_default

    # クラス数は data→cfg(model.num_labels)→fallback(6)
    n_class = n_class_data or int(model_cfg.get("num_labels", 6))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(ns.batch)

    model, tokenizer, n_class_ckpt, meta = _load_model_and_tokenizer(
        model_name=model_name,
        num_labels=n_class,
        ckpt_path=ckpt_path,
        device=device,
    )
    if model is None or tokenizer is None:
        print("[eval] ERROR: failed to initialize model/tokenizer.")
        return 1

    if n_class_ckpt is not None and n_class_ckpt != n_class:
        print(f"[eval] WARN: n_class from data/cfg={n_class}, from ckpt={n_class_ckpt}")

    # ===== dev =====
    results: Dict[str, Any] = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "ckpt_path": ckpt_path,
        "model_name": model_name,
        "num_labels_data": n_class_data,
        "num_labels_cfg": model_cfg.get("num_labels"),
        "num_labels_used": n_class,
        "generated_at": time.strftime("%F %T"),
    }
    results.update(meta)

    if rows_dev:
        ds_dev = EvalDataset(rows_dev, tokenizer, text_key, label_key, max_len, with_label=True)
        dl_dev = DataLoader(ds_dev, batch_size=batch_size, shuffle=False, num_workers=0)
        ids_d, yt_d, yp_d, cf_d = _predict(model, dl_dev, device)

        # None が混ざっていない前提（with_label=True）
        y_true_dev = np.array(yt_d, dtype=int)
        y_pred_dev = np.array(yp_d, dtype=int)
        qwk_dev = _qwk(y_true_dev, y_pred_dev)
        rmse_dev = _rmse(y_true_dev, y_pred_dev)
        print(f"[eval] dev: n={len(ids_d)}  QWK={qwk_dev:.4f}  RMSE={rmse_dev:.4f}")
        results["dev"] = {"n": len(ids_d), "qwk": qwk_dev, "rmse": rmse_dev}

        _write_preds_csv(os.path.join(out_dir, "dev_preds.csv"), ids_d, yt_d, yp_d, cf_d)
        print(f"[eval] wrote dev_preds.csv -> {out_dir}")

    # ===== test =====
    if rows_test:
        ds_te = EvalDataset(rows_test, tokenizer, text_key, label_key, max_len, with_label=True)
        dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)
        ids_t, yt_t, yp_t, cf_t = _predict(model, dl_te, device)

        y_true_te = np.array(yt_t, dtype=int)
        y_pred_te = np.array(yp_t, dtype=int)
        qwk_te = _qwk(y_true_te, y_pred_te)
        rmse_te = _rmse(y_true_te, y_pred_te)
        print(f"[eval] test: n={len(ids_t)}  QWK={qwk_te:.4f}  RMSE={rmse_te:.4f}")
        results["test"] = {"n": len(ids_t), "qwk": qwk_te, "rmse": rmse_te}

        _write_preds_csv(os.path.join(out_dir, "test_preds.csv"), ids_t, yt_t, yp_t, cf_t)
        print(f"[eval] wrote test_preds.csv -> {out_dir}")

    # メタ情報を書き出し
    meta_path = os.path.join(out_dir, "eval.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[eval] wrote meta -> {meta_path}")

    return 0


if __name__ == "__main__":  # 直叩き用（開発時向け）
    from tensaku.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--target", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--batch", type=int, default=32)
    args, _rest = parser.parse_known_args()

    cfg = load_config(args.config, [])
    sys.exit(run(["--target", args.target, "--batch", str(args.batch)], cfg))
