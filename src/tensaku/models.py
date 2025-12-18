# /home/esakit25/work/tensaku/src/tensaku/models.py
from __future__ import annotations
import logging
from typing import Any, Mapping, Optional
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)

# Strict Mode: デフォルトモデル名の定数(DEF_MODEL_NAME)は廃止。
# Configから必ず指定させる。

def create_tokenizer(cfg: Mapping[str, Any]) -> PreTrainedTokenizerBase:
    """
    Configに基づいてTokenizerを作成する。
    
    Strict Mode:
      - cfg['model']['name'] が必須。
    """
    if "model" not in cfg:
        raise KeyError("Config missing 'model' section.")
    
    model_cfg = cfg["model"]
    # 必須: モデル名 (例: "cl-tohoku/bert-base-japanese-v3")
    model_name = model_cfg["name"]
    
    return AutoTokenizer.from_pretrained(model_name)


def create_model(cfg: Mapping[str, Any], num_labels: int, state_dict: Optional[Mapping[str, Any]] = None) -> PreTrainedModel:
    """
    Configとラベル数に基づいてモデルを作成する。
    
    Strict Mode:
      - cfg['model']['name'] 必須
      - cfg['model']['freeze_base'] 必須 (bool)
      - cfg['model']['dropout'] 必須 (float or null)
    """
    if "model" not in cfg:
        raise KeyError("Config missing 'model' section.")

    model_cfg = cfg["model"]
    
    # 1. 必須パラメータの取得
    model_name = model_cfg["name"]
    freeze_base = model_cfg["freeze_base"] # 必須: True/False をYAMLで明示
    dropout_rate = model_cfg["dropout"]    # 必須: 0.1 等 (使わないなら null などを許容するロジックにするか、0.0を強制)

    # Note: YAMLで dropout: null と書かれた場合は None になるためハンドリング
    if dropout_rate is None:
        dropout_rate = 0.0
    
    # 2. Config構築
    hf_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    
    # Dropout設定 (明示的に上書き)
    hf_config.hidden_dropout_prob = float(dropout_rate)
    hf_config.attention_probs_dropout_prob = float(dropout_rate)

    # 3. モデルロード
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=hf_config)
    
    # 4. 重みのロード (あれば)
    if state_dict is not None:
        # strict=False は「ヘッドの形状不一致」等を許容するため維持（転移学習用）
        model.load_state_dict(state_dict, strict=False)

    # 5. フリーズ処理
    if freeze_base:
        LOGGER.info("Freezing base model parameters.")
        for name, param in model.named_parameters():
            # classifier, score, head などの分類ヘッド以外を凍結
            if "classifier" in name or "score" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    return model