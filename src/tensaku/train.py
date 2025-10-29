"""
@module: train
@role: 日本語BERTで総得点の分類モデルを学習し、dev QWK ベースで best.pt を保存
@inputs: {data_dir}/labeled.jsonl（id, mecab|text, score）, model.name, model/optim 設定
@outputs: {out_dir}/checkpoints_min/best.pt, {out_dir}/checkpoints_min/last.pt, 学習ログ
@cli: tensaku train
@notes: すべてのテンソルを同一 device に移送。AMP は CUDA 時のみ自動で有効化。
"""

# Path: /home/esakit25/work/tensaku/src/tensaku/train.py
from __future__ import annotations
import os, json, math, random
from typing import Dict, Any, List
from torch.amp import GradScaler, autocast
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import cohen_kappa_score
import numpy as np

def _read_jsonl(path:str)->List[Dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

class EssayDS(Dataset):
    def __init__(self, rows, tok, text_key="mecab", label_key="score", max_len=128, with_label=True):
        self.rows=rows; self.tok=tok
        self.text_key=text_key; self.label_key=label_key
        self.max_len=max_len; self.with_label=with_label
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        text=r.get(self.text_key,"")
        if isinstance(text,list): text=" ".join(map(str,text))
        enc=self.tok(text,truncation=True,padding="max_length",max_length=self.max_len,return_tensors="pt")
        item={k:v.squeeze(0) for k,v in enc.items()}
        if self.with_label:
            item["labels"]=torch.tensor(int(r[self.label_key]),dtype=torch.long)
        return item

def _qwk(y_true, y_pred, n_class):
    # y_true/pred は 0..N の整数
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

@torch.no_grad()
def _eval_qwk_rmse(model, loader, device, n_class):
    model.eval()
    ys=[]; ps=[]
    for batch in loader:
        for k, v in list(batch.items()):
            if k != "labels" and hasattr(v, "to"):
                batch[k] = v.to(device, non_blocking=True)

        logits = model(**{k:batch[k] for k in batch if k!="labels"}).logits
        pred = logits.argmax(-1).cpu().numpy()
        ys.append(batch["labels"].numpy())
        ps.append(pred)
    y=np.concatenate(ys); p=np.concatenate(ps)
    qwk=_qwk(y,p,n_class)
    rmse=float(np.sqrt(((p - y)**2).mean()))
    return qwk, rmse

def run(cfg: Dict[str,Any]) -> None:
    run = cfg["run"]; data = cfg["data"]; model_cfg = cfg["model"]; train_cfg = cfg["train"]
    data_dir = run["data_dir"]; files = data["files"]
    text_key = data.get("text_key_primary","mecab"); label_key=data.get("label_key","score"); max_len=data.get("max_len",128)
    seed = run.get("seed",42)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # 入力ファイル
    path_tr = os.path.join(data_dir, files["labeled"])
    path_dv = os.path.join(data_dir, files["dev"])
    rows_tr = _read_jsonl(path_tr)
    rows_dv = _read_jsonl(path_dv)
    if not rows_tr or not rows_dv:
        print(f"[train] missing or empty data. labeled={len(rows_tr)}, dev={len(rows_dv)}"); return

    # クラス数はデータから自動決定（max(score)+1）
    max_y = 0
    for r in rows_tr + rows_dv:
        try: max_y = max(max_y, int(r[label_key]))
        except Exception: pass
    n_class = int(max_y) + 1
    if model_cfg.get("num_labels") not in (None, n_class):
        # configに書いてあっても自動推定を優先（安全）
        pass

    # トークナイザ・モデル
    name = model_cfg.get("name","cl-tohoku/bert-base-japanese-v3")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_class)

    # SPEEDモード：full/frozen/fe_sklearn（今回は full と frozen のみ）
    speed = model_cfg.get("speed","full")
    if speed in ("frozen","fe_sklearn"):
        for p in model.bert.parameters(): p.requires_grad=False

    # Dataloader
    ds_tr = EssayDS(rows_tr, tok, text_key, label_key, max_len, with_label=True)
    ds_dv = EssayDS(rows_dv, tok, text_key, label_key, max_len, with_label=True)
    bs = int(train_cfg.get("batch_size",16))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0)
    dl_dv = DataLoader(ds_dv, batch_size=bs, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer/Scheduler
    if speed=="full":
        opt = AdamW(model.parameters(), lr=float(train_cfg.get("lr_full",2e-5)))
    else:
        opt = AdamW(model.classifier.parameters(), lr=float(train_cfg.get("lr_frozen",5e-4)))
    epochs = int(train_cfg.get("epochs",5))
    total_steps = epochs * math.ceil(len(ds_tr)/bs)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=total_steps)

    # 出力先
    out_root = run.get("out_dir","./outputs")
    ckpt_dir = os.path.join(out_root, "checkpoints_min")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_qwk = -1.0

    # 4090 なら bfloat16 が安定なので推奨（必要なら fp16 に変えてOK）
    AMP_ENABLED = (device.type == "cuda")
    AMP_DTYPE = torch.bfloat16  # 4090/AMPERE では bf16 推奨

    scaler = GradScaler(enabled=AMP_ENABLED)

    for ep in range(1, epochs+1):
        model.train()
        for batch in dl_tr:
            # ↓ labels 以外を一括転送（token_type_ids を含む全て）
            for k, v in list(batch.items()):
                if k != "labels" and hasattr(v, "to"):
                    batch[k] = v.to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                out = model(**{k:batch[k] for k in batch if k!="labels"}, labels=batch["labels"])
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            sched.step()

        qwk, rmse = _eval_qwk_rmse(model, dl_dv, device, n_class)
        print(f"[train] epoch {ep}/{epochs}  dev: QWK={qwk:.4f}  RMSE={rmse:.4f}")

        # best保存
        torch.save({"model":model.state_dict(),"epoch":ep,"qwk":qwk,"n_class":n_class}, os.path.join(ckpt_dir,"last.pt"))
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({"model":model.state_dict(),"epoch":ep,"qwk":qwk,"n_class":n_class}, os.path.join(ckpt_dir,"best.pt"))

    # 終了メッセージ
    print(f"[train] done. best QWK={best_qwk:.4f}  ckpt={ckpt_dir}/best.pt")
