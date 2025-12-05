# /home/esakit25/work/tensaku/src/tensaku/models.py
# -*- coding: utf-8 -*-
"""
@module     : tensaku.models
@role       : モデルとトークナイザのファクトリ
@note       : 将来的に独自のアーキテクチャを追加する場合はここを拡張する。
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def create_tokenizer(cfg: Dict[str, Any]):
    """設定からトークナイザを生成して返す"""
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "cl-tohoku/bert-base-japanese-v3")
    
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"[models] ERROR: Failed to load tokenizer '{name}': {e}")
        raise e
        
    return tok

def create_model(cfg: Dict[str, Any], n_class: int):
    """設定とクラス数に基づいてモデルを生成して返す"""
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "cl-tohoku/bert-base-japanese-v3")
    
    # --- 将来の拡張ポイント ---
    # if name == "my_custom_brain_model":
    #     return MyCustomBrainModel(num_labels=n_class)
    # -----------------------

    # デフォルト: HuggingFace AutoModel
    print(f"[models] Loading AutoModel: {name} (num_labels={n_class})")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_class)
    except Exception as e:
        print(f"[models] ERROR: Failed to load model '{name}': {e}")
        raise e

    # Speed / Freeze 設定の適用
    # "frozen" 指定時は、分類ヘッド以外（バックボーン）を凍結する
    speed = model_cfg.get("speed", "full")
    if speed in ("frozen", "fe_sklearn"):
        print(f"[models] Freezing backbone layers (speed={speed})")
        # BERT系
        if hasattr(model, "bert"):
            for p in model.bert.parameters():
                p.requires_grad = False
        # RoBERTa系
        elif hasattr(model, "roberta"):
             for p in model.roberta.parameters():
                p.requires_grad = False
        # DeBERTa系など必要に応じて追加

    return model