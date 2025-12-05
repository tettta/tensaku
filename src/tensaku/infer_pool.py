# /home/esakit25/work/tensaku/src/tensaku/infer_pool.py
# -*- coding: utf-8 -*-
"""
@module   : tensaku.infer_pool
@role     : dev / pool / test に対して ckpt で一括推論し、予測CSVと簡易メタ情報を out_dir に保存する。
@updated  : models.py を使用するようにリファクタリング
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ★変更点: models.py からファクトリ関数をインポート
from tensaku.models import create_model, create_tokenizer
from .model_io import select_device
from .embed import labels_from_rows, predict_with_emb

# 任意依存（trust は無ければスキップ）
try:
    from .trustscore import TrustScorer
except Exception:
    TrustScorer = None

DEF_BASE = "cl-tohoku/bert-base-japanese-v3"


# ===== I/O helpers ===============================================================================


def _read_jsonl(path: str) -> List[dict]:
    """シンプルな JSONL ローダー。存在しなければ空リスト。"""
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


def _infer_num_labels_from_rows(
    data_dir: str,
    files: Dict[str, str],
    label_key: str = "score",
) -> Optional[int]:
    """dev/test から max(label)+1 を推定する（失敗時 None）。"""
    max_y = -1
    for key in ("labeled", "train", "dev", "test"):
        fname = files.get(key)
        if not fname:
            if key in ("labeled", "train"):
                fname = f"{key}.jsonl"
            elif key in ("dev", "test"):
                fname = f"{key}.jsonl"
            else:
                continue
        path = os.path.join(data_dir, fname)
        rows = _read_jsonl(path)
        for r in rows:
            try:
                y = int(r[label_key])
            except Exception:
                continue
            if y > max_y:
                max_y = y
    return (max_y + 1) if max_y >= 0 else None


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    s = np.sum(ex, axis=axis, keepdims=True)
    return ex / np.clip(s, 1e-12, None)


# ===== ckpt helpers ==============================================================================


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


# ===== core inference ============================================================================


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
        print("[infer-pool] WARN: TrustScorer が利用できないため --trust を無視します。", file=sys.stderr)
        return trust_dict

    train_name = None
    train_rows: List[dict] = []
    if "labeled" in outputs and outputs["labeled"].get("embs") is not None and labeled_rows:
        train_name = "labeled"
        train_rows = labeled_rows
    elif "dev" in outputs and outputs["dev"].get("embs") is not None and dev_rows:
        train_name = "dev"
        train_rows = dev_rows
    else:
        print("[infer-pool] WARN: TrustScore 学習用の埋め込みが無いため計算をスキップします。", file=sys.stderr)
        return trust_dict

    train_embs = outputs[train_name]["embs"]
    train_labels = labels_from_rows(train_rows, label_key=label_key)
    if train_labels is None:
        print(f"[infer-pool] WARN: {train_name} にラベルが無いため TrustScore を計算できません。", file=sys.stderr)
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


def run(argv: Optional[List[str]], cfg: Dict[str, Any]) -> int:
    """研究モード用の dev/pool/test 一括推論エントリポイント。"""
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust", action="store_true", default=None)
    parser.add_argument("--trust-k", type=int, default=1)
    parser.add_argument("--trust-metric", choices=["cosine", "euclidean"], default="cosine")
    ns, _rest = parser.parse_known_args(argv or [])

    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    infer_cfg = cfg.get("infer") or {}

    data_dir = run_cfg.get("data_dir") or data_cfg.get("data_dir")
    out_dir = run_cfg.get("out_dir") or "./outputs"
    if not data_dir:
        print("[infer-pool] ERROR: run.data_dir または data.data_dir が未設定です。", file=sys.stderr)
        return 2

    files_cfg = data_cfg.get("files") or {}
    file_labeled = files_cfg.get("labeled", "labeled.jsonl")
    file_dev = files_cfg.get("dev", "dev.jsonl")
    file_pool = files_cfg.get("pool", "pool.jsonl")
    file_test = files_cfg.get("test", "test.jsonl")

    label_key = data_cfg.get("label_key", "score")
    id_key = data_cfg.get("id_key", "id")

    os.makedirs(out_dir, exist_ok=True)

    # ---- JSONL 読み込み ----
    path_pool = os.path.join(data_dir, file_pool)
    pool_rows = _read_jsonl(path_pool)
    if not pool_rows:
        print(f"[infer-pool] ERROR: pool が空 or 不存在: {path_pool}", file=sys.stderr)
        return 2

    path_dev = os.path.join(data_dir, file_dev)
    path_test = os.path.join(data_dir, file_test)
    dev_rows = _read_jsonl(path_dev)
    test_rows = _read_jsonl(path_test)
    
    path_labeled = os.path.join(data_dir, file_labeled)
    labeled_rows = _read_jsonl(path_labeled)

    # ---- モデル名・ckpt ----
    model_name = model_cfg.get("name") or DEF_BASE
    ckpt_default = os.path.join(out_dir, "checkpoints_min", "best.pt")
    ckpt_path = infer_cfg.get("ckpt") or model_cfg.get("ckpt") or ckpt_default

    allow_random = bool(infer_cfg.get("allow_random", False))
    has_ckpt = os.path.exists(ckpt_path)

    if not has_ckpt and not allow_random:
        print(f"[infer-pool] ERROR: ckpt が見つかりません: {ckpt_path}", file=sys.stderr)
        return 2

    # ---- n_class の決定 ----
    n_class: Optional[int] = model_cfg.get("num_labels")
    state_dict = None
    
    # ckptからラベル数推定を試みる
    if n_class is None and has_ckpt:
        try:
            bundle = torch.load(ckpt_path, map_location="cpu")
            sd_raw = _extract_state_dict(bundle)
            if sd_raw is not None:
                # 分類層のshapeから推定
                for key in ["classifier.weight", "score.weight", "head.weight"]:
                    if key in sd_raw and hasattr(sd_raw[key], "shape"):
                        n_class = int(sd_raw[key].shape[0])
                        break
                if n_class is None:
                    for k, v in sd_raw.items():
                        if k.endswith("classifier.weight") and hasattr(v, "shape"):
                            n_class = int(v.shape[0])
                            break
                state_dict = _normalize_keys(sd_raw)
        except Exception as e:
            print(f"[infer-pool] WARN: Failed to inspect ckpt for num_labels: {e}")

    if n_class is None:
        n_class = _infer_num_labels_from_rows(data_dir, files_cfg, label_key=label_key)
    if n_class is None:
        n_class = int(model_cfg.get("num_labels_fallback", 6))

    # ---- tokenizer / model 構築 (models.py 委譲) ----
    # ★変更点: models.py の create_tokenizer, create_model を使用
    print(f"[infer-pool] Creating model (n_class={n_class}) via models.py...")
    tokenizer = create_tokenizer(cfg)
    model = create_model(cfg, n_class)

    # ckpt ロード
    if has_ckpt:
        if state_dict is None:
            # まだロードしていなければここで読む
            try:
                bundle = torch.load(ckpt_path, map_location="cpu")
                sd_raw = _extract_state_dict(bundle)
                state_dict = _normalize_keys(sd_raw) if sd_raw is not None else None
            except Exception as e:
                print(f"[infer-pool] ERROR: Failed to load ckpt: {e}", file=sys.stderr)
                return 2
        
        if state_dict is not None:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[infer-pool] WARN: load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}", file=sys.stderr)
        else:
            print("[infer-pool] WARN: ckpt から有効な state_dict を取り出せませんでした（ランダム初期化のまま続行）。", file=sys.stderr)
    else:
        print("[infer-pool] WARN: Running with random initialization (no ckpt).")

    device_name = ns.device or infer_cfg.get("device", "auto")
    dev = select_device(device_name)
    model = model.to(dev).eval()

    max_len = int(ns.max_len if ns.max_len is not None else infer_cfg.get("max_len", 128))
    batch_size = int(ns.batch_size if ns.batch_size is not None else infer_cfg.get("batch_size", 32))

    trust_flag = bool(ns.trust if ns.trust is not None else infer_cfg.get("trust", False))
    trust_k = int(ns.trust_k)
    trust_metric = ns.trust_metric
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

    # labeled（train）も埋め込みを計算しておく（trust の学習集合用）
    _process_split("labeled", labeled_rows)

    _process_split("dev", dev_rows)
    _process_split("pool", pool_rows)
    _process_split("test", test_rows)

    # ---- Trust Score（任意） ----
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

    # ---- Logits & Embeddings 書き出し ----
    for split_name in ("dev", "pool", "test"):
        if split_name in outputs:
            if outputs[split_name].get("logits") is not None:
                path_logits = os.path.join(out_dir, f"{split_name}_logits.npy")
                np.save(path_logits, outputs[split_name]["logits"])
            
            if outputs[split_name].get("embs") is not None:
                path_embs = os.path.join(out_dir, f"{split_name}_embs.npy")
                np.save(path_embs, outputs[split_name]["embs"])

    # ---- CSV 書き出し ----
    def _write_csv(path_csv: str, data: Dict[str, Any], trust_vals: Optional[np.ndarray], has_true: bool) -> None:
        rows = data["rows"]
        y_pred = data["y_pred"]
        conf_msp = data["conf_msp"]
        with open(path_csv, "w", encoding="utf-8", newline="") as f:
            cols = ["id"]
            if has_true: cols.append("y_true")
            cols.extend(["y_pred", "conf_msp"])
            if trust_vals is not None: cols.append("conf_trust")
            writer = csv.writer(f)
            writer.writerow(cols)
            for i, (r, yp, cm) in enumerate(zip(rows, y_pred, conf_msp)):
                rid = r.get(id_key, str(i))
                row = [rid]
                if has_true:
                    try: yt = int(r.get(label_key, 0))
                    except: yt = 0
                    row.append(yt)
                row.extend([int(yp), float(cm)])
                if trust_vals is not None:
                    row.append(float(trust_vals[i]))
                writer.writerow(row)

    if "dev" in outputs:
        _write_csv(os.path.join(out_dir, "dev_preds.csv"), outputs["dev"], trust_dict.get("dev"), True)
    if "pool" in outputs:
        _write_csv(os.path.join(out_dir, "pool_preds.csv"), outputs["pool"], trust_dict.get("pool"), False)
    if "test" in outputs and outputs["test"]["rows"]:
        _write_csv(os.path.join(out_dir, "test_preds.csv"), outputs["test"], trust_dict.get("test"), True)

    # ---- meta 書き出し ----
    meta = {
        "model_name": model_name,
        "ckpt": ckpt_path if has_ckpt else None,
        "n_class": int(n_class),
        "data_dir": data_dir,
        "generated_at": time.strftime("%F %T"),
    }
    with open(os.path.join(out_dir, "pool_preds.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    print("Run via CLI: tensaku infer-pool -c <CFG.yaml>")