# /home/esakit25/work/tensaku/src/tensaku/data/base.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.data.base
@role  : データセットの抽象化レイヤ (Adapter & Split)
@overview:
    - DatasetSplit: {labeled, dev, test, pool} の4分割データを保持するコンテナ。
    - BaseDatasetAdapter: データの読み込みを担当する基底クラス。
    - SasJsonlAdapter: 'data_sas/q-xxx/' 以下のJSONL群を読み込む標準実装。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional

LOGGER = logging.getLogger(__name__)


# ======================================================================
# データコンテナ
# ======================================================================


@dataclass
class DatasetSplit:
    """Active Learning におけるデータの4分割状態を保持するコンテナ。

    Attributes:
        labeled: ラベル付きデータ (Train用)
        dev    : 開発データ (Validation/Model Selection用)
        test   : 評価データ (Test用)
        pool   : プールデータ (Unlabeled/Candidate用)
    """

    labeled: List[Mapping[str, Any]] = field(default_factory=list)
    dev: List[Mapping[str, Any]] = field(default_factory=list)
    test: List[Mapping[str, Any]] = field(default_factory=list)
    pool: List[Mapping[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, int]:
        """件数情報を返す（ログ用）。"""
        return {
            "n_labeled": len(self.labeled),
            "n_dev": len(self.dev),
            "n_test": len(self.test),
            "n_pool": len(self.pool),
        }


# ======================================================================
# アダプタ基底クラス
# ======================================================================


class BaseDatasetAdapter:
    """データセット読み込みの抽象基底クラス。"""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        # ALState等でIDを取り出す際に使うキー名
        self.id_key = str(cfg.get("data", {}).get("id_key", "id"))

    def make_initial_split(self) -> DatasetSplit:
        """初期状態の DatasetSplit を構築して返す。"""
        raise NotImplementedError


# ======================================================================
# 実装: SAS JSONL アダプタ (研究モード用)
# ======================================================================


class SasJsonlAdapter(BaseDatasetAdapter):
    """
    tensaku.split (Phase1) で生成された JSONL ファイル群を読み込むアダプタ。
    
    Expected Config:
        run.data_dir: JSONLファイルがあるディレクトリ
        data.files.labeled: (Optional) ファイル名
        data.files.dev: ...
    """

    def make_initial_split(self) -> DatasetSplit:
        run_cfg = self.cfg.get("run", {})
        data_cfg = self.cfg.get("data", {})
        
        data_dir_str = run_cfg.get("data_dir") or data_cfg.get("data_dir")
        if not data_dir_str:
            raise ValueError("SasJsonlAdapter requires 'run.data_dir' in config.")
        
        data_dir = Path(data_dir_str)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        files_cfg = data_cfg.get("files", {})
        
        # ファイル名解決 (デフォルトは split.py の出力仕様に準拠)
        f_labeled = files_cfg.get("labeled") or files_cfg.get("train") or "labeled.jsonl"
        f_dev = files_cfg.get("dev", "dev.jsonl")
        f_test = files_cfg.get("test", "test.jsonl")
        f_pool = files_cfg.get("pool", "pool.jsonl")

        LOGGER.info(f"Loading data from {data_dir} ...")

        labeled = self._load_jsonl(data_dir / f_labeled)
        dev = self._load_jsonl(data_dir / f_dev)
        test = self._load_jsonl(data_dir / f_test)
        pool = self._load_jsonl(data_dir / f_pool)

        # 読み込み件数チェック
        if not labeled and not pool:
            LOGGER.warning(f"Both labeled and pool are empty in {data_dir}.")
        
        return DatasetSplit(
            labeled=labeled,
            dev=dev,
            test=test,
            pool=pool,
        )

    def _load_jsonl(self, path: Path) -> List[Mapping[str, Any]]:
        rows = []
        if not path.exists():
            LOGGER.debug(f"File not found: {path} (treated as empty)")
            return rows
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    LOGGER.warning(f"Skipping invalid JSON line in {path}")
                    continue
        return rows


# ======================================================================
# Factory
# ======================================================================


def create_adapter(cfg: Mapping[str, Any]) -> BaseDatasetAdapter:
    """Config に基づいて DatasetAdapter を生成する。"""
    
    # 1. 明示的な指定があればそれを使う
    adapter_name = cfg.get("data", {}).get("adapter")
    
    # 2. 指定がなければデフォルト (sas_jsonl)
    if not adapter_name:
        adapter_name = "sas_jsonl"
    
    name = str(adapter_name).lower()
    
    if name == "sas_jsonl":
        return SasJsonlAdapter(cfg)
    
    # 将来の拡張用
    # if name == "huggingface":
    #     return HuggingFaceAdapter(cfg)

    LOGGER.warning(f"Unknown adapter '{name}'. Falling back to SasJsonlAdapter.")
    return SasJsonlAdapter(cfg)