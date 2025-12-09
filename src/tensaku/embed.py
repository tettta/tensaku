# /home/esakit25/work/tensaku/src/tensaku/embed.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.embed
@role: BERT系モデルからのCLS埋め込み抽出とロジット推論のユーティリティ
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError("embed.py requires PyTorch") from e

try:
    from transformers import AutoTokenizer
except Exception as e:
    raise RuntimeError("embed.py requires transformers") from e


TEXT_KEYS = ("mecab", "text")


def texts_from_rows(rows: List[dict], text_key: str = "mecab") -> List[str]:
    out: List[str] = []
    for r in rows:
        t = r.get(text_key)
        if t is None:
            for k in TEXT_KEYS:
                if r.get(k) is not None:
                    t = r[k]
                    break
        if isinstance(t, list):
            t = " ".join(map(str, t))
        if isinstance(t, str):
            t = t.strip()
        out.append(t or "")
    return out


def labels_from_rows(rows: List[dict], label_key: str = "score") -> Optional[np.ndarray]:
    ys: List[int] = []
    has_any = False
    for r in rows:
        if label_key in r:
            has_any = True
            try:
                ys.append(int(r[label_key]))
            except Exception:
                ys.append(0)
        else:
            ys.append(0)
    return np.asarray(ys, dtype=int) if has_any else None


class _TextDS(Dataset):
    def __init__(self, texts: List[str], tok, max_len: int = 128):
        self.texts = texts
        self.tok = tok
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def _select_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_backbone(model: Any):
    if hasattr(model, "bert"):
        return model.bert
    if hasattr(model, "base_model"):
        return model.base_model
    return None


@torch.no_grad()
def predict_with_emb(
    model: Any,
    rows: List[dict],
    *,
    tokenizer: Optional[Any] = None,
    bs: int = 16,
    max_len: int = 128,
    device: str = "auto",
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Returns: (logits, labels, embs)
    """
    if not rows:
        raise RuntimeError("predict_with_emb: empty rows")

    dev = _select_device(device)
    model = model.to(dev).eval()

    if tokenizer is None:
        name = getattr(getattr(model, "config", None), "_name_or_path", None)
        if not name:
            name = "cl-tohoku/bert-base-japanese-v3"
        tokenizer = AutoTokenizer.from_pretrained(name)

    texts = texts_from_rows(rows)
    labels = labels_from_rows(rows)

    ds = _TextDS(texts, tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=int(bs), shuffle=False, num_workers=0)

    logits_list = []
    embs_list = []

    backbone = _get_backbone(model)

    for batch in dl:
        tb = {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}
        out = model(**tb)
        logits = out.logits if hasattr(out, "logits") else out[0]
        logits_list.append(logits.detach().cpu())

        if backbone is not None:
            out_b = backbone(**tb)
            cls = out_b.last_hidden_state[:, 0, :]
        else:
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                cls = out.hidden_states[-1][:, 0, :]
            elif hasattr(out, "last_hidden_state"):
                cls = out.last_hidden_state[:, 0, :]
            else:
                cls = logits
        embs_list.append(cls.detach().cpu())

    logits_np = torch.cat(logits_list, dim=0).numpy()
    embs_np = torch.cat(embs_list, dim=0).numpy()
    
    return logits_np, labels, embs_np