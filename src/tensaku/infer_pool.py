"""
@module: infer_pool
@role: 未採点プールに対して一括推論を行い、予測ラベルとMSP確信度を保存
@inputs: {data_dir}/pool.jsonl, {out_dir}/checkpoints_min/best.pt, model.name, data.max_len
@outputs: {out_dir}/pool_preds.csv（id,y_pred,conf_msp）
@cli: tensaku infer-pool
@notes: クラス数は ckpt→head形状→データ最大ラベル+1→cfg→fallback(6) の順で推定。
"""

# Path: /home/esakit25/work/tensaku/src/tensaku/infer_pool.py
from __future__ import annotations
import os, json
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

def _infer_num_labels_from_ckpt_state(sd: dict) -> Optional[int]:
    """ckpt の state_dict から classifier 出力次元を推定"""
    try:
        st = sd.get("model") if isinstance(sd, dict) else None
        if isinstance(st, dict):
            if "classifier.weight" in st and hasattr(st["classifier.weight"], "shape"):
                return int(st["classifier.weight"].shape[0])
            # 互換キー（念のため）
            for k in st.keys():
                if k.endswith("classifier.weight") and hasattr(st[k], "shape"):
                    return int(st[k].shape[0])
    except Exception:
        pass
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

@torch.no_grad()
def run(cfg: Dict[str,Any]) -> None:
    run = cfg["run"]; data = cfg["data"]; model_cfg = cfg["model"]; infer_cfg = cfg.get("infer", {})
    data_dir = run["data_dir"]; files = data["files"]
    text_key = data.get("text_key_primary","mecab"); max_len=data.get("max_len",128)

    path_pool = os.path.join(data_dir, files["pool"])
    rows = _read_jsonl(path_pool)
    if not rows:
        print(f"[infer-pool] empty pool: {path_pool}")
        return

    # モデルとトークナイザ
    name = model_cfg.get("name","cl-tohoku/bert-base-japanese-v3")
    tok = AutoTokenizer.from_pretrained(name)

    ckpt_dir = os.path.join(run.get("out_dir","./outputs"), "checkpoints_min")
    ckpt_path = infer_cfg.get("ckpt") or os.path.join(ckpt_dir, "best.pt")
    sd = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else {}

    # --- n_class を堅牢に決定 ---
    n_class: Optional[int] = None
    # 1) ckpt['n_class']
    try:
        v = sd.get("n_class", None)
        if v is not None:
            n_class = int(v)
    except Exception:
        n_class = None
    # 2) state_dict の classifier.weight 形状
    if n_class is None:
        n_class = _infer_num_labels_from_ckpt_state(sd)
    # 3) データの score から推定
    if n_class is None:
        n_class = _infer_num_labels_from_data(data_dir, files, label_key=data.get("label_key","score"))
    # 4) config の指定 or デフォルト
    if n_class is None:
        try:
            v = model_cfg.get("num_labels", None)
            if v is not None: n_class = int(v)
        except Exception:
            n_class = None
    if n_class is None:
        n_class = 6  # 最終フォールバック
    # ---------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=int(n_class))
    # ckpt があれば読み込み（strict=False で互換）
    if "model" in sd:
        model.load_state_dict(sd["model"], strict=False)

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
    out_dir = run.get("out_dir","./outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "pool_preds.csv")
    with open(out_csv, "w", encoding="utf-8") as w:
        w.write("id,y_pred,conf_msp\n")
        for i,p,c in zip(ids,preds,confs):
            w.write(f"{i},{p},{c:.6f}\n")
    print(f"[infer-pool] wrote {out_csv}  rows={len(ids)}  n_class={n_class}")
