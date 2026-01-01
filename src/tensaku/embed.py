# /home/esakit25/work/tensaku/src/tensaku/embed.py
# -*- coding: utf-8 -*-
"""tensaku.embed

@module : tensaku.embed
@role   : 推論時の logits と CLS 埋め込みの抽出

方針（正常動作優先）:
- logits は **学習時と同じ forward**（model(**batch)）が返す out.logits を使用する。
- CLS 埋め込みが必要なため output_hidden_states=True を有効化し、
  last_hidden_state の [CLS]（hidden_states[-1][:,0,:]）を保存する。
- メモリ最適化より「正しい挙動」を優先する（必要なら後で最適化を再設計）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


class _SimpleDS(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_len: int, text_key_primary: str) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.text_key_primary = str(text_key_primary)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        text = row.get(self.text_key_primary, "")
        if isinstance(text, list):
            text = " ".join(map(str, text))

        enc = self.tokenizer(
            str(text),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
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
    label_key: str,
    text_key_primary: str,
    return_logits: bool = True,
    enable_mem_release: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """推論して (logits|None, labels|None, CLS_embs) を返す（正常動作優先版）。

    - logits: out.logits（学習時と同じ forward の出力）
    - CLS: out.hidden_states[-1][:,0,:]
    """
    model.eval()

    ds = _SimpleDS(rows, tokenizer, max_len, text_key_primary)
    dl = DataLoader(ds, batch_size=int(bs), shuffle=False, num_workers=0)
    dev = torch.device(str(device))

    # ---- labels ----
    has_any_label_key = bool(rows) and any((label_key in r) for r in rows)
    all_labels: Optional[np.ndarray]
    if has_any_label_key:
        tmp: List[int] = []
        for r in rows:
            v = r.get(label_key, None)
            if v is None:
                tmp.append(-1)
            else:
                try:
                    tmp.append(int(v))
                except Exception:
                    tmp.append(-1)
        all_labels = np.asarray(tmp, dtype=np.int64)
    else:
        all_labels = None

    all_logits: List[np.ndarray] = []
    all_embs: List[np.ndarray] = []

    for batch in dl:
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}

        # ★旧挙動に戻す：学習時と同じ forward で logits を得る
        try:
            out = model(**batch, output_hidden_states=True, return_dict=True)
        except TypeError:
            # return_dict を受けないモデルでも動くように（ただしサイレントフォールバックはしない）
            out = model(**batch, output_hidden_states=True)

        if return_logits:
            logits = out.logits
            all_logits.append(logits.detach().cpu().to(torch.float32).numpy())

        # CLS embedding
        hs = getattr(out, "hidden_states", None)
        if hs is None or len(hs) == 0:
            raise TypeError(
                "predict_with_emb requires hidden_states (output_hidden_states=True) "
                "but model output has no hidden_states."
            )
        last_hidden = hs[-1]
        cls_emb = last_hidden[:, 0, :]
        all_embs.append(cls_emb.detach().cpu().to(torch.float32).numpy())

    if not all_embs:
        empty_embs = np.array([], dtype=np.float32)
        empty_labels = None if all_labels is None else all_labels[:0]
        return (None if not return_logits else np.array([], dtype=np.float32)), empty_labels, empty_embs

    embs_concat = np.concatenate(all_embs, axis=0)
    logits_concat: Optional[np.ndarray]
    if return_logits:
        logits_concat = np.concatenate(all_logits, axis=0) if all_logits else np.array([], dtype=np.float32)
    else:
        logits_concat = None

    return logits_concat, all_labels, embs_concat
