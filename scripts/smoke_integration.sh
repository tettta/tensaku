#!/bin/bash
# smoke_integration_v2.sh
# 既存の train.py / infer_pool.py を温存し、
# train_core.py / infer_core.py を新規作成して統合テストを行う。

# エラー時に停止
set -e

# 作業用ディレクトリ
TEST_DIR="_smoke_test_integration_v2"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# ソースコード配置ディレクトリ（カレントからの相対）
SRC_DIR="../src/tensaku"
mkdir -p "$SRC_DIR"

echo "=== [1] Deploying Phase 2 Core Modules (Safe Mode) ==="

# ------------------------------------------------------------------
# 1. models.py (存在しない場合のみ作成、あるいは上書き)
# ------------------------------------------------------------------
cat << 'EOF' > "$SRC_DIR/models.py"
# /home/esakit25/work/tensaku/src/tensaku/models.py
from __future__ import annotations
import logging
from typing import Any, Mapping, Optional
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)
DEF_MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"

def create_tokenizer(cfg: Mapping[str, Any]) -> PreTrainedTokenizerBase:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", DEF_MODEL_NAME)
    return AutoTokenizer.from_pretrained(model_name)

def create_model(cfg: Mapping[str, Any], num_labels: int, state_dict: Optional[Mapping[str, Any]] = None) -> PreTrainedModel:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", DEF_MODEL_NAME)
    freeze_base = bool(model_cfg.get("freeze_base", False))
    
    hf_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    if "dropout" in model_cfg:
        hf_config.hidden_dropout_prob = float(model_cfg["dropout"])
        hf_config.attention_probs_dropout_prob = float(model_cfg["dropout"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=hf_config)
    
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    if freeze_base:
        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model
EOF

# ------------------------------------------------------------------
# 2. train_core.py (新規作成)
#    train.py のコアロジックを独立させる
# ------------------------------------------------------------------
cat << 'EOF' > "$SRC_DIR/train_core.py"
# /home/esakit25/work/tensaku/src/tensaku/train_core.py
from __future__ import annotations
import math
import os
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
from tensaku.data.base import DatasetSplit
from tensaku.models import create_model, create_tokenizer

def _normalize_labels(rows, label_key, split_name):
    clean = []
    min_y, max_y = None, None
    for r in rows:
        try:
            y = int(r.get(label_key, -1))
        except:
            continue
        if y < 0: continue
        r2 = dict(r)
        r2[label_key] = y
        clean.append(r2)
        if min_y is None or y < min_y: min_y = y
        if max_y is None or y > max_y: max_y = y
    return clean, min_y, max_y

class EssayDS(Dataset):
    def __init__(self, rows, tok, text_key="mecab", label_key="score", max_len=128):
        self.rows = rows
        self.tok = tok
        self.text_key = text_key
        self.label_key = label_key
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        text = str(r.get(self.text_key, ""))
        enc = self.tok(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(r[self.label_key]), dtype=torch.long)
        return item

@torch.no_grad()
def _eval_qwk(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        for k, v in batch.items():
            if k!="labels": batch[k] = v.to(device)
        logits = model(**{k: v for k, v in batch.items() if k!="labels"}).logits
        ps.append(logits.argmax(-1).cpu().numpy())
        ys.append(batch["labels"].numpy())
    return cohen_kappa_score(np.concatenate(ys), np.concatenate(ps), weights="quadratic")

def train_core(split: DatasetSplit, out_dir: Path, cfg: Mapping[str, Any], meta: Optional[Dict]=None) -> int:
    run_cfg = cfg.get("run", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(run_cfg.get("seed", 42))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    rows_tr, _, max_tr = _normalize_labels(split.labeled, data_cfg.get("label_key", "score"), "train")
    rows_dv, _, max_dv = _normalize_labels(split.dev, data_cfg.get("label_key", "score"), "dev")
    
    if not rows_tr or not rows_dv:
        print("Error: empty train/dev")
        return 1

    n_class = int(max(max_tr, max_dv)) + 1
    if model_cfg.get("num_labels"): n_class = int(model_cfg["num_labels"])

    tok = create_tokenizer(cfg)
    model = create_model(cfg, n_class)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    bs = int(train_cfg.get("batch_size", 4))
    ds_tr = EssayDS(rows_tr, tok, max_len=int(data_cfg.get("max_len", 128)))
    ds_dv = EssayDS(rows_dv, tok, max_len=int(data_cfg.get("max_len", 128)))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_dv = DataLoader(ds_dv, batch_size=bs, shuffle=False)

    opt = AdamW(model.parameters(), lr=float(train_cfg.get("lr_full", 1e-4)))
    epochs = int(train_cfg.get("epochs", 1))
    
    ckpt_dir = out_dir / "checkpoints_min"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_core] Start training: epochs={epochs}, n_class={n_class}, device={device}")

    for ep in range(1, epochs+1):
        model.train()
        for batch in dl_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        qwk = _eval_qwk(model, dl_dv, device)
        print(f"[train_core] Epoch {ep}: QWK={qwk:.3f}")
        torch.save(model.state_dict(), ckpt_dir / "best.pt") # 簡易的に毎回保存

    return 0
EOF

# ------------------------------------------------------------------
# 3. infer_core.py (新規作成)
#    infer_pool.py のコアロジックを独立させる
# ------------------------------------------------------------------
cat << 'EOF' > "$SRC_DIR/infer_core.py"
# /home/esakit25/work/tensaku/src/tensaku/infer_core.py
from __future__ import annotations
import os
import json
import csv
import sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from tensaku.data.base import DatasetSplit
from tensaku.models import create_model, create_tokenizer
from tensaku.model_io import select_device

# 必要なら TrustScore をインポート
try:
    from tensaku.trustscore import TrustScorer
except ImportError:
    TrustScorer = None

def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

@torch.no_grad()
def _predict(model, rows, tok, device, max_len=128):
    model.eval()
    logits_list, embs_list = [], []
    # バッチ処理は省略(スモークテスト用)
    for r in rows:
        text = str(r.get("text", ""))
        enc = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        logits_list.append(out.logits.cpu().numpy())
        # 簡易埋め込み: CLS token (last hidden state の 0番目)
        embs_list.append(out.hidden_states[-1][:, 0, :].cpu().numpy())
    
    if not logits_list: return None, None
    return np.concatenate(logits_list), np.concatenate(embs_list)

def infer_core(split: DatasetSplit, out_dir: Path, cfg: Mapping[str, Any]) -> int:
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("infer", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_class = int(model_cfg.get("num_labels", 2))
    tok = create_tokenizer(cfg)
    model = create_model(cfg, n_class)
    
    ckpt_path = infer_cfg.get("ckpt")
    if ckpt_path and os.path.exists(ckpt_path):
        # state_dict のキー整合性は簡易ロードで対応
        sd = torch.load(ckpt_path, map_location="cpu")
        if "model" in sd: sd = sd["model"] # train.py がネストして保存する場合
        model.load_state_dict(sd, strict=False)
    
    device = select_device(infer_cfg.get("device", "auto"))
    model.to(device)
    
    for name, rows in [("dev", split.dev), ("pool", split.pool), ("test", split.test)]:
        if not rows: continue
        logits, embs = _predict(model, rows, tok, device)
        if logits is None: continue
        
        probs = _softmax(logits)
        preds = probs.argmax(-1)
        conf_msp = probs.max(-1)
        
        # TrustScore (簡易: ランダム)
        conf_trust = np.random.rand(len(rows)) 

        # CSV 書き出し
        with open(out_dir / f"{name}_preds.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "y_pred", "conf_msp", "conf_trust"])
            for i, r in enumerate(rows):
                writer.writerow([r.get("id", i), preds[i], conf_msp[i], conf_trust[i]])

    return 0
EOF

# ------------------------------------------------------------------
# 4. SasTotalScoreTask の修正
#    train_core, infer_core をインポートするように書き換え
# ------------------------------------------------------------------
cat << 'EOF' > "$SRC_DIR/tasks/sas_total_score.py"
# /home/esakit25/work/tensaku/src/tensaku/tasks/sas_total_score.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import pandas as pd
import yaml

from tensaku.tasks.base import TrainInferHitlTask, TaskOutputs
from tensaku.data.base import DatasetSplit
# 【変更点】独立させた _core モジュールからインポート
from tensaku.train_core import train_core
from tensaku.infer_core import infer_core
from tensaku.pipelines.hitl import run_hitl_from_detail_df, HitlOutputs

LOGGER = logging.getLogger(__name__)

class SasTotalScoreTask(TrainInferHitlTask):
    task_name: str = "sas_total_score"

    def __init__(self, cfg: Mapping[str, Any], adapter: Any, layout: Any) -> None:
        super().__init__(cfg, adapter, layout)
        self._current_data_dir: Optional[Path] = None
        self._current_train_out_dir: Optional[Path] = None
        self._current_preds_path: Optional[Path] = None

    def step_train(self, round_index: int, split: DatasetSplit) -> None:
        LOGGER.info("[Task] step_train (round=%d)", round_index)
        data_dir = self.layout.root / "temp_data" / f"round_{round_index:03d}"
        if data_dir.exists(): shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._current_data_dir = data_dir

        self._write_jsonl(data_dir / "labeled.jsonl", split.labeled)
        self._write_jsonl(data_dir / "dev.jsonl", split.dev)
        self._write_jsonl(data_dir / "test.jsonl", split.test)
        self._write_jsonl(data_dir / "pool.jsonl", split.pool)

        train_out_dir = self.layout.root / "rounds" / f"round_{round_index:03d}" / "train"
        train_out_dir.mkdir(parents=True, exist_ok=True)
        self._current_train_out_dir = train_out_dir

        run_context_cfg = self._clone_and_update_cfg(new_run={"data_dir": str(data_dir), "out_dir": str(train_out_dir)})
        
        with open(train_out_dir / "config_final.yaml", "w") as f:
            yaml.safe_dump(run_context_cfg, f)

        ret = train_core(split=split, out_dir=train_out_dir, cfg=run_context_cfg)
        if ret != 0: raise RuntimeError(f"train_core failed: {ret}")

    def step_infer(self, round_index: int, split: DatasetSplit) -> None:
        LOGGER.info("[Task] step_infer (round=%d)", round_index)
        infer_out_dir = self.layout.root / "rounds" / f"round_{round_index:03d}" / "infer"
        infer_out_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = self._current_train_out_dir / "checkpoints_min" / "best.pt"
        
        infer_cfg = self._clone_and_update_cfg(
            new_run={"data_dir": str(self._current_data_dir), "out_dir": str(infer_out_dir)},
            new_infer={"ckpt": str(ckpt_path), "trust": True}
        )

        ret = infer_core(split=split, out_dir=infer_out_dir, cfg=infer_cfg)
        if ret != 0: raise RuntimeError(f"infer_core failed: {ret}")

        dfs = []
        for name in ["dev", "pool", "test"]:
            fpath = infer_out_dir / f"{name}_preds.csv"
            if fpath.exists():
                d = pd.read_csv(fpath)
                d["split"] = name
                dfs.append(d)
        
        if not dfs: raise RuntimeError("No prediction CSVs found")
        
        if hasattr(self.layout, "path_predictions_round_detail"):
            dest_path = self.layout.path_predictions_round_detail(round_index)
        else:
            dest_path = self.layout.predictions_rounds_dir / f"round_{round_index:03d}_preds_detail.csv"
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(dfs, ignore_index=True).to_csv(dest_path, index=False)
        self._current_preds_path = dest_path

    def step_confidence(self, r, s): pass

    def step_hitl(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        LOGGER.info("[Task] step_hitl (round=%d)", round_index)
        df = pd.read_csv(self._current_preds_path)
        hitl_out = run_hitl_from_detail_df(df, self.cfg)
        metrics = hitl_out.to_summary_dict()

        # Extract Pool Scores
        pool_scores = {}
        target_col = f"conf_{hitl_out.conf_key}" if f"conf_{hitl_out.conf_key}" in df.columns else hitl_out.conf_key
        if target_col in df.columns:
            pool_df = df[df["split"] == "pool"]
            if not pool_df.empty:
                pool_scores = dict(zip(pool_df.get("id", pool_df.index), pool_df[target_col]))

        return TaskOutputs(metrics=metrics, pool_scores=pool_scores)

    def _write_jsonl(self, path, records):
        with open(path, "w") as f:
            for r in records:
                d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _clone_and_update_cfg(self, new_run=None, new_model=None, new_infer=None):
        import copy
        cloned = copy.deepcopy(dict(self.cfg))
        if new_run: 
            if "run" not in cloned: cloned["run"] = {}
            cloned["run"].update(new_run)
        if new_infer:
            if "infer" not in cloned: cloned["infer"] = {}
            cloned["infer"].update(new_infer)
        return cloned
EOF


echo "=== [2] Creating Dummy Data & Config ==="

# 極小データ
DATA_DIR="data_sas/q-smoke_v2"
mkdir -p "$DATA_DIR"
cat << 'EOF' > "$DATA_DIR/labeled.jsonl"
{"id": "L1", "text": "A", "score": 1}
{"id": "L2", "text": "B", "score": 0}
EOF
cat << 'EOF' > "$DATA_DIR/dev.jsonl"
{"id": "D1", "text": "C", "score": 1}
EOF
cat << 'EOF' > "$DATA_DIR/pool.jsonl"
{"id": "P1", "text": "D", "score": 0}
{"id": "P2", "text": "E", "score": 1}
EOF
cat << 'EOF' > "$DATA_DIR/test.jsonl"
{"id": "T1", "text": "F", "score": 0}
EOF

# Config
cat << 'EOF' > config.yaml
run:
  out_dir: "output_integration_v2"
  data_dir: "data_sas/q-smoke_v2"
  run_id: "integration_test_v2"
  seed: 42
  task_name: "sas_total_score"

data:
  adapter: "sas_jsonl"
  id_key: "id"
  text_key_primary: "text"
  label_key: "score"
  files:
    labeled: "labeled.jsonl"
    dev: "dev.jsonl"
    pool: "pool.jsonl"
    test: "test.jsonl"

model:
  name: "prajjwal1/bert-tiny"
  num_labels: 2

train:
  epochs: 1
  batch_size: 2

infer:
  batch_size: 2
  trust: true
  trust_k: 1

gate:
  conf_key: "trust"

al:
  rounds: 2
  budget: 1
  sampler:
    name: "random"
EOF

echo "=== [3] Running Integration Driver ==="
cat << 'EOF' > run_driver.py
import sys, os, yaml
sys.path.insert(0, os.path.abspath("../src"))
from tensaku.pipelines.al import run_experiment

def main():
    with open("config.yaml", "r") as f: cfg = yaml.safe_load(f)
    print(">>> Starting run_experiment...")
    run_experiment(cfg)
    print(">>> Finished.")
    
    # 簡易チェック
    if os.path.exists("output_integration_v2/metrics/al_history.csv"):
        print("SUCCESS: al_history.csv created.")
    else:
        print("FAIL: al_history.csv missing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# 実行
export PYTHONPATH=$PYTHONPATH:$(pwd)/../src
# エラー詳細が見えるように標準出力のみ
python run_driver.py

exit $?