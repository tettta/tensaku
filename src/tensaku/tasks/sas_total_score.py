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
from tensaku.train import train_core
from tensaku.infer_pool import infer_core
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
