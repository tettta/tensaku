# /home/esakit25/work/tensaku/src/tensaku/embed.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)

class _SimpleDS(Dataset):
    def __init__(self, rows, tokenizer, max_len, text_key_primary):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_key_primary = text_key_primary # Strict: キーも受け取る
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        # Strict: Configで指定されたキーのみを見る
        text = row.get(self.text_key_primary, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        
        enc = self.tokenizer(
            str(text),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

@torch.no_grad()
def predict_with_emb(
    model: Any,
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    bs: int,
    max_len: int,
    device: str,
    label_key: str,           # ★ 追加: 必須引数
    text_key_primary: str,    # ★ 追加: 必須引数
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    label_key を infer_pool.py から受け取るように修正。
    """
    model.eval()
    
    # Dataset作成時に text_key も渡す
    ds = _SimpleDS(rows, tokenizer, max_len, text_key_primary)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    
    dev = torch.device(device)
    
    all_logits = []
    all_embs = []
    
    # ラベル回収 (Strict: label_key を使う)
    all_labels = []
    has_valid_label = False

    # 1件でもラベルキーが含まれていればラベルありとみなす
    if rows and any(label_key in r for r in rows):
        has_valid_label = True

    for row in rows:
        val = row.get(label_key)
        if val is not None:
            try:
                all_labels.append(int(val))
            except:
                all_labels.append(-1)
        else:
            all_labels.append(-1)

    for batch in dl:
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(**batch, output_hidden_states=True)
        
        logits = out.logits
        last_hidden = out.hidden_states[-1]
        cls_emb = last_hidden[:, 0, :]
        
        all_logits.append(logits.cpu().numpy())
        all_embs.append(cls_emb.cpu().numpy())

    if not all_logits:
        return np.array([]), None, np.array([])

    logits_concat = np.concatenate(all_logits, axis=0)
    embs_concat = np.concatenate(all_embs, axis=0)
    
    labels_concat = np.array(all_labels, dtype=int) if has_valid_label else None
    
    return logits_concat, labels_concat, embs_concat