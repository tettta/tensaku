# src/tensaku/dataset/text_classification.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

class LdccDataset(Dataset):
    """Text Classification Dataset.
    
    Responsibility:
    - Load raw JSONL data.
    - Expose 'num_classes' for dynamic config injection.
    - Serve items for PyTorch DataLoader.
    """

    def __init__(
        self,
        input_file: str,
        label_key: str = "label",
        text_key: str = "text",
        id_key: str = "id",
        split: str = "train",  # split name (just for info/logging)
        **kwargs,
    ):
        super().__init__()
        self.input_file = Path(input_file)
        self.label_key = label_key
        self.text_key = text_key
        self.id_key = id_key
        self.split = split
        
        self.data: List[Dict[str, Any]] = []
        self.classes: List[int] = []
        
        # 初期化時にロード
        self._load_data()

    def _load_data(self):
        if not self.input_file.exists():
            # Probeフェーズでファイルが無いと困るのでエラーにする
            raise FileNotFoundError(f"Dataset file not found: {self.input_file}")

        labels_seen = set()
        
        with self.input_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                
                # データ保持
                self.data.append(obj)
                
                # ラベル収集（存在する場合）
                if self.label_key in obj:
                    val = obj[self.label_key]
                    # 数値であることを期待（あるいはエンコーディングが必要ならここでやる）
                    try:
                        labels_seen.add(int(val))
                    except (ValueError, TypeError):
                        pass # ラベルが無い/不正な行はスキップ（Active LearningのPool等）

        # クラス一覧をソートして保持
        if labels_seen:
            self.classes = sorted(list(labels_seen))
        else:
            self.classes = []

    @property
    def num_classes(self) -> int:
        """Dynamic config injection uses this property."""
        if not self.classes:
            return 0
        # 0始まりのインデックスと仮定して最大値+1、またはユニーク数
        # 通常は len(self.classes) で良いが、欠番がある場合は max(classes)+1 が安全
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data[index]
        return {
            "text": row.get(self.text_key, ""),
            "label": int(row.get(self.label_key, -1)),
            "id": row.get(self.id_key, ""),
        }