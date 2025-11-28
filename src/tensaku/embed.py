# -*- coding: utf-8 -*-
"""
@module: tensaku.embed
@role: BERT系モデルからのCLS埋め込み抽出とロジット推論のユーティリティ（dev/poolで共通利用）
@inputs:
  - model: transformers.AutoModelForSequenceClassification（.bert 互換を想定）
  - tokenizer: transformers.PreTrainedTokenizerBase | None（None時は model.config._name_or_path から推定）
  - rows: list[dict]  # {"mecab"| "text", "score"(任意), "id"(任意)}
  - max_len: int（既定128）, bs: int（既定16）
@outputs:
  - predict_with_emb(...) -> (logits: np.ndarray[N,C], labels: np.ndarray[N] or None, embs: np.ndarray[N,D])
@cli: 直接のCLIは持たない（infer_pool / gate から内部利用）
@api:
  - texts_from_rows(rows, text_key="mecab") -> list[str]
  - labels_from_rows(rows, label_key="score") -> np.ndarray[int] | None
  - predict_with_emb(model, rows, tokenizer=None, bs=16, max_len=128, device="auto")
@deps: torch, transformers, numpy
@config: なし（呼び出し側のCFGに従う）
@contracts:
  - score は 0..N の int を前提（無ければ None を返す）
  - model が .bert を持たない場合は base_model を試行
@errors:
  - 必須依存が無い/データが空の場合は RuntimeError
@notes:
  - DataLoader を使わずに最小限のGeneratorでも動くが、本実装は DataLoader を使用
  - AMPはここでは使わず、安定性重視でfp32推論
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Any, Iterable, Dict
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # pragma: no cover
    raise RuntimeError("embed.py requires PyTorch") from e

try:
    from transformers import AutoTokenizer
except Exception as e:  # pragma: no cover
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
    # HFの多くは base_model もしくは model.<arch> を持つ
    if hasattr(model, "base_model"):
        return model.base_model
    # 最後の手段：モデル自身に forward して last_hidden_state を期待
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
    if not rows:
        raise RuntimeError("predict_with_emb: empty rows")

    dev = _select_device(device)
    model = model.to(dev).eval()

    if tokenizer is None:
        # モデルから推定
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
            # ロバストフォールバック：model 出力に last_hidden_state がある前提
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                cls = out.hidden_states[-1][:, 0, :]
            elif hasattr(out, "last_hidden_state"):
                cls = out.last_hidden_state[:, 0, :]
            else:
                # logits からの簡易埋め込み（最終層手前を取得できない場合の形合わせ）
                cls = logits
        embs_list.append(cls.detach().cpu())

    logits_np = torch.cat(logits_list, dim=0).numpy()
    embs_np = torch.cat(embs_list, dim=0).numpy()
    return logits_np, labels, embs_np
