# -*- coding: utf-8 -*-
"""
@module: infer_pool
@role: 未採点プールに対して一括推論を行い、予測ラベルとMSP確信度を保存
@inputs: {data_dir}/pool.jsonl, {out_dir}/checkpoints_min/best.pt, model.name, data.max_len
@outputs: {out_dir}/pool_preds.csv（id,y_pred,conf_msp）, {out_dir}/pool_preds.meta.json
@cli: tensaku infer-pool
@notes:
  - クラス数は ①ckptのclassifier形状 → ②データ最大ラベル+1 → ③cfg → ④fallback(6) の順で推定。
  - ckpt未指定はエラー終了（cfg.infer.allow_random=True でのみ継続）。
  - 分類ヘッドが実際に適用されたかを 'loaded_classifier' で可視化。
"""
from __future__ import annotations
import os, json, sys, time
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

DEF_BASE = "cl-tohoku/bert-base-japanese-v3"

# -------- I/O --------
def _read_jsonl(path:str)->List[dict]:
    rows=[]
    if not os.path.exists(path):
        return rows
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

# -------- Dataset --------
class EssayDS(Dataset):
    def __init__(self, rows, tok, text_key="mecab", max_len=128):
        self.rows=rows; self.tok=tok; self.text_key=text_key; self.max_len=max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        text=r.get(self.text_key,"")
        if isinstance(text,list): text=" ".join(map(str,text))
        enc=self.tok(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item={k:v.squeeze(0) for k,v in enc.items()}
        item["id"]=r.get("id", str(i))
        return item

# -------- CKPT utils（頑健化） --------
def _extract_state_dict(bundle) -> Optional[dict]:
    """
    学習時の保存形式の差異に頑健に state_dict を取り出す。
    期待キー候補: 'state_dict','model','model_state_dict','ema','model_ema','net'
    直接Tensor辞書の場合も許容。
    """
    if not isinstance(bundle, dict):
        return None
    # 直接state_dictっぽい？
    if all(hasattr(v, "shape") for v in bundle.values()):
        return bundle
    for k in ["state_dict","model","model_state_dict","ema","model_ema","net"]:
        sd = bundle.get(k)
        if isinstance(sd, dict) and all(hasattr(v, "shape") for v in sd.values()):
            return sd
    return None

def _normalize_keys(sd: dict) -> dict:
    """module./model. 等のプレフィックス除去＋一般的な別名をclassifierへ正規化"""
    out={}
    for k,v in sd.items():
        if k.startswith("module."): k=k[len("module."):]
        if k.startswith("model."):  k=k[len("model."):]
        k=k.replace("classification_head.", "classifier.")
        out[k]=v
    return out

def _infer_num_labels_from_state(sd: dict) -> Optional[int]:
    """state_dict から classifier 出力次元を推定"""
    for key in ["classifier.weight", "score.weight", "head.weight"]:
        if key in sd and hasattr(sd[key], "shape"):
            return int(sd[key].shape[0])
    # サフィックス一致で拾う（bert系で前置詞がついた場合）
    for k in sd.keys():
        if k.endswith("classifier.weight") and hasattr(sd[k], "shape"):
            return int(sd[k].shape[0])
    return None

def _infer_num_labels_from_data(data_dir: str, files: Dict[str,str], label_key="score") -> Optional[int]:
    """labeled/dev/test を見て max(score)+1 を推定"""
    max_y = -1
    for key in ("labeled","dev","test"):
        path = os.path.join(data_dir, files.get(key, f"{key}.jsonl"))
        rows = _read_jsonl(path)
        for r in rows:
            try:
                y = int(r[label_key])
                if y > max_y: max_y = y
            except Exception:
                pass
    return (max_y + 1) if max_y >= 0 else None

def _save_meta(out_dir:str, meta:dict)->None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,"pool_preds.meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# -------- Main --------
@torch.no_grad()
def run(cfg: Dict[str,Any]) -> None:
    run_cfg = cfg["run"]; data = cfg["data"]; model_cfg = cfg["model"]; infer_cfg = cfg.get("infer", {})
    data_dir = run_cfg["data_dir"]; files = data["files"]
    text_key = data.get("text_key_primary","mecab"); max_len=data.get("max_len",128)
    out_dir = run_cfg.get("out_dir","./outputs")

    # 入力
    path_pool = os.path.join(data_dir, files["pool"])
    rows = _read_jsonl(path_pool)
    if not rows:
        print(f"[infer-pool] ERROR: empty pool: {path_pool}", file=sys.stderr)
        return

    # モデル名・ckpt
    name = model_cfg.get("name", DEF_BASE)
    # 優先度: infer.ckpt > model.ckpt > {out_dir}/checkpoints_min/best.pt
    ckpt_dir_default = os.path.join(out_dir, "checkpoints_min")
    ckpt_path = (infer_cfg.get("ckpt")
                 or model_cfg.get("ckpt")
                 or os.path.join(ckpt_dir_default, "best.pt"))

    allow_random = bool(infer_cfg.get("allow_random", False))
    if not os.path.exists(ckpt_path) and not allow_random:
        print(f"[infer-pool] ERROR: ckpt not found (set infer.ckpt or model.ckpt): {ckpt_path}", file=sys.stderr)
        return

    # ckpt ロード（あれば）
    sd_bundle = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else None
    state_dict = _normalize_keys(_extract_state_dict(sd_bundle)) if sd_bundle else None

    # --- n_class を堅牢に決定（優先: ckpt → data → cfg → fallback）---
    n_class: Optional[int] = None
    if state_dict is not None:
        n_class = _infer_num_labels_from_state(state_dict)
    if n_class is None:
        n_class = _infer_num_labels_from_data(data_dir, files, label_key=data.get("label_key","score"))
    if n_class is None:
        v = model_cfg.get("num_labels", None)
        n_class = int(v) if v is not None else None
    if n_class is None: n_class = 6

    tok = AutoTokenizer.from_pretrained(name)
    # HFの"未初期化ヘッド"警告は from_pretrained時に出るが、後で ckpt を上書きロードする。
    config = AutoModelForSequenceClassification.config = AutoConfig.from_pretrained(name, num_labels=int(n_class))
    model = AutoModelForSequenceClassification.from_config(config)

    loaded_classifier = False
    missing_keys: Tuple[List[str], List[str]] | None = None
    if state_dict is not None:
        try:
            ik = model.load_state_dict(state_dict, strict=False)  # IncompatibleKeys
            # PyTorch: ik has .missing_keys/.unexpected_keys
            missing = set(getattr(ik, "missing_keys", []))
            # “分類ヘッドが missing ではない”なら適用できた可能性が高い
            loaded_classifier = not any(k.endswith(("classifier.weight","classifier.bias")) for k in missing)
        except Exception as e:
            print(f"[infer-pool] WARNING: failed to load ckpt (continue with random head): {e}", file=sys.stderr)

    if not os.path.exists(ckpt_path):
        print("[infer-pool] continue with RANDOM head (allow_random=True)", file=sys.stderr)
    elif not loaded_classifier:
        print("[infer-pool] WARNING: ckpt loaded but classifier not applied (size/key mismatch?)", file=sys.stderr)

    # 推論
    ds = EssayDS(rows, tok, text_key, max_len)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()

    ids=[]; preds=[]; confs=[]
    for batch in dl:
        ids.extend(batch["id"])
        # labels 以外のテンソルを包括的にデバイスへ
        batch = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in batch.items() if k!="id"}
        logits = model(**batch).logits
        prob = torch.softmax(logits, dim=-1)
        pred = prob.argmax(-1)
        preds.extend(pred.cpu().numpy().tolist())
        confs.extend(prob.max(-1).values.cpu().numpy().tolist())

    # 出力
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "pool_preds.csv")
    with open(out_csv, "w", encoding="utf-8") as w:
        w.write("id,y_pred,conf_msp\n")
        for i,p,c in zip(ids,preds,confs):
            w.write(f"{i},{p},{c:.6f}\n")
    _save_meta(out_dir, {
        "base_name": name,
        "ckpt": ckpt_path if os.path.exists(ckpt_path) else None,
        "loaded_classifier": loaded_classifier,
        "n_class": int(n_class),
        "pool_path": path_pool,
        "generated_at": time.strftime("%F %T"),
    })

    print(f"[infer-pool] wrote {out_csv}  rows={len(ids)}  n_class={n_class}")
    if os.path.exists(ckpt_path) and not loaded_classifier:
        print("[infer-pool] NOTE: 分類ヘッドが未適用の可能性（num_labels不一致やキー名の齟齬）", file=sys.stderr)
