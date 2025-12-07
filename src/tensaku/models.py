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
