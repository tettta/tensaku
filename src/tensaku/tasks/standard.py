# /home/esakit25/work/tensaku/src/tensaku/tasks/standard.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.tasks.standard
@role  : 標準的な教師あり学習 Active Learning タスクの汎用実装
"""

from __future__ import annotations

import json
import logging
import shutil
import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from tensaku import registry
from tensaku.tasks.base import TrainInferHitlTask, TaskOutputs
from tensaku.data.base import DatasetSplit

# コアロジック
from tensaku.train import train_core
from tensaku.infer_pool import infer_core
from tensaku.pipelines.hitl import run_hitl_from_detail_df

LOGGER = logging.getLogger(__name__)


class StandardSupervisedAlTask(TrainInferHitlTask):
    """
    標準的な教師あり学習 (Supervised Learning) を用いる AL タスクの基底実装。
    """
    name: str = "StandardSupervisedAlTask"
    
    def __init__(self, cfg: Mapping[str, Any], adapter: Any, layout: Any) -> None:
        super().__init__(cfg, adapter, layout)

        
        
        # ステップ間データ受け渡し用ステート
        self._current_data_dir: Optional[Path] = None
        self._current_train_out_dir: Optional[Path] = None
        # ★修正1: _cleanup_round_files 実行時のエラーに対応
        self._current_infer_out_dir: Optional[Path] = None 
        
        self._current_model: Optional[Any] = None
        self._current_preds_df: Optional[pd.DataFrame] = None

        self._current_raw_outputs: Optional[Any] = None
        # クリーンアップ対象の拡張属性（必要に応じてサブクラスが使う）
        self._extra_cleanup_dirs: List[Path] = []
        self._extra_cleanup_files: List[Path] = []

        # 現在のラウンド番号（クリーンアップ用）
        self._current_round_index: Optional[int] = None



    # =============================================================================
    # 抽象メソッドの実装 (Template Method)
    # =============================================================================

    def run_round(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        """
        クリーンアップ処理を追加したラッパー。
        """
        # ラウンドごとにクリーンアップ対象をリセットし、現在ラウンドを記録
        self._extra_cleanup_dirs = []
        self._extra_cleanup_files = []
        self._current_round_index = round_index

        try:
            return super().run_round(round_index, split)
        finally:
            self._cleanup_round_files()
            self._current_round_index = None



    def step_train(self, round_index: int, split: DatasetSplit) -> None:
        """学習を実行し、モデルを作成・保存するステップ。"""
        LOGGER.info(f"[{self.name}] step_train (round={round_index})")
        
        # 1. データの準備 (temp_data に書き出し)
        data_dir = self.layout.path_temp_round_dir(round_index)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        self._write_jsonl(data_dir / "labeled.jsonl", split.labeled)
        self._write_jsonl(data_dir / "dev.jsonl", split.dev)
        # self._write_jsonl(data_dir / "test.jsonl", split.test) # test は推論時に使う
        # self._write_jsonl(data_dir / "pool.jsonl", split.pool) # pool も推論時に使う

        self._current_data_dir = data_dir
        
        # 2. Config の準備
        train_cfg = self._clone_and_update_cfg(
            new_run={"data_dir": str(data_dir)},
            new_model={"output_dir": str(self.layout.path_models_round_dir(round_index))}
        )
        
        # 3. 学習コアロジック実行
        out_dir = self.layout.path_rounds_train_dir(round_index)
        out_dir.mkdir(parents=True, exist_ok=True)

        # モデルをメモリ上にロード
        ret, model = train_core(
            split=split,
            out_dir=out_dir,
            cfg=train_cfg,
            return_model=True
        )
        
        if ret != 0:
            raise RuntimeError(f"train_core failed with return code {ret}")

        # 4. ステートの更新
        self._current_train_out_dir = Path(out_dir)
        self._current_model = model


    def step_infer(self, round_index: int, split: DatasetSplit) -> None:
        """推論を実行し、予測結果を保持するステップ。"""
        LOGGER.info(f"[{self.name}] step_infer (round={round_index})")
        
        if self._current_model is None:
            raise RuntimeError("Model is missing in step_infer")
        if self._current_data_dir is None:
            raise RuntimeError("Data dir is missing in step_infer")
        
        # 1. Config の準備
        out_dir = self.layout.path_rounds_infer_dir(round_index)
        out_dir.mkdir(parents=True, exist_ok=True)

        infer_cfg = self._clone_and_update_cfg(
            new_run={
                "data_dir": str(self._current_data_dir),
                "out_dir": str(out_dir),
            },
            new_infer={"model_path": None}, # メモリ上のモデルを使うため None
        )

        # 2. 推論コアロジック実行
        ret, preds_df, raw_outputs = infer_core(
            split=split,
            out_dir=out_dir,
            cfg=infer_cfg,
            model=self._current_model, 
            return_df=True
        )
        
        # 3. ステートの更新
        self._current_infer_out_dir = Path(out_dir)
        self._current_preds_df = preds_df
        self._current_raw_outputs = raw_outputs
        

    def step_confidence(self, round_index: int, split: DatasetSplit) -> None:
        """推論結果に対して信頼度・不確実性を計算・付与するステップ。"""
        LOGGER.info(f"[{self.name}] step_confidence (round={round_index})")
        
        if self._current_preds_df is None:
            LOGGER.warning("preds_df not available, skipping step_confidence.")
            return

        df_all = self._current_preds_df.copy()
        raws = self._current_raw_outputs # Dict[split_name, Dict[str, np.ndarray]]

        if raws is None:
            LOGGER.warning("raw_outputs missing. Cannot run custom estimators.")
            return

        # 1. Config で指定された Estimator を取得
        est_cfg = self.cfg.get("confidence", {}).get("estimators", [])
        if not est_cfg:
            LOGGER.info("No confidence estimators configured.")
            return

        # ヘルパー: Softmax関数
        def _softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        # 2. 各 Estimator を実行
        for item in est_cfg:
            name = item.get("name")
            kwargs = item.get("params", {}) or {} # None対策
            col_name = f"conf_{name}"
            
            try:
                estimator = registry.create(name, **kwargs)
            except KeyError:
                LOGGER.warning(f"Unknown estimator '{name}', skipping.")
                continue

            LOGGER.info(f"  - Calculating: {name} -> col: {col_name}")

            # --- A. TrustScore の場合 (Fit & Score が必要) ---
            if name == "trust":
                # (1) 学習データで Fit
                if "labeled" in raws:
                    train_embs = raws["labeled"]["embs"]
                    train_labels = raws["labeled"]["labels"] # 正解ラベル
                    try:
                        estimator.fit(train_embs, train_labels)
                    except Exception as e:
                        LOGGER.error(f"TrustScore fit failed: {e}")
                        continue
                else:
                    LOGGER.warning("TrustScore requires 'labeled' split in raw_outputs for fitting.")
                    continue

                # (2) 各スプリットで Score
                df_all[col_name] = np.nan # 初期化
                for split_name in ["dev", "test", "pool"]:
                    if split_name not in raws:
                        continue
                    
                    target_embs = raws[split_name]["embs"]
                    target_pred = raws[split_name]["y_pred"]
                    
                    try:
                        # TrustScore.score(test_feats, test_pred)
                        scores = estimator.score(target_embs, target_pred)
                        
                        # 結果をDataFrameに埋め込む
                        mask = df_all["split"] == split_name
                        if mask.sum() == len(scores):
                            df_all.loc[mask, col_name] = scores
                        else:
                            LOGGER.error(f"Shape mismatch for {name} in {split_name}: df={mask.sum()}, score={len(scores)}")
                    except Exception as e:
                        LOGGER.error(f"TrustScore scoring failed on {split_name}: {e}")

            # --- B. その他の Estimator (Entropy, MSPなど) ---
            else:
                # 全データをまとめて計算（行順序は df_all と一致している前提だが、split毎にやるのが安全）
                # ここでは簡易的に split ごとに計算して埋める
                df_all[col_name] = np.nan
                for split_name, data in raws.items():
                    logits = data["logits"]
                    probs = data.get("probs")
                    
                    # probs が無ければ計算する
                    if probs is None and logits is not None:
                        probs = _softmax(logits)
                    
                    if probs is None and logits is None:
                        continue

                    try:
                        # estimator(probs=..., logits=...)
                        # ※ entropyなどは probs だけ、msp も probs だけで動くことが多い
                        scores = estimator(probs=probs, logits=logits)
                        
                        mask = df_all["split"] == split_name
                        if mask.sum() == len(scores):
                            df_all.loc[mask, col_name] = scores
                    except Exception as e:
                        LOGGER.warning(f"Estimator {name} failed on {split_name}: {e}")

        # 4. ステートの更新
        self._current_preds_df = df_all
        
        # 更新されたCSVを保存 (HITLやデバッグで確認するため)
        if self.cfg.get("run", {}).get("save_predictions", True):
            save_path = self.layout.path_predictions_round_detail(round_index)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(save_path, index=False)

    

    def step_hitl(self, round_index: int, split: DatasetSplit) -> TaskOutputs:
        """予測結果に基づき HITL 判定を行い、Metrics と Pool Scores を返すステップ。"""
        LOGGER.info(f"[{self.name}] step_hitl (round={round_index})")
        
        df = self._current_preds_df
        if df is None:
            raise RuntimeError("Predictions df is missing in step_hitl")

        # 1. HITL判定の実行 (計算のみ)
        hitl_out = run_hitl_from_detail_df(df, self.cfg)
        metrics = hitl_out.to_summary_dict()

        # ★修正2: gate_assign.csv の保存
        self._save_gate_assign(round_index, df, hitl_out)

        # 2. Poolスコア抽出（AL 用の conf_key を解決）
        pool_scores: Dict[Any, float] = {}

        conf_key_short = self._resolve_al_conf_key(hitl_out)
        target_col = f"conf_{conf_key_short}"

        # conf_key がそのまま列名の場合も考慮 (例: conf_trust, trust, etc.)
        if target_col not in df.columns:
            if conf_key_short in df.columns:
                target_col = conf_key_short
            else:
                LOGGER.warning(
                    f"[{self.name}] AL conf column not found "
                    f"(conf_key={conf_key_short}, target_col={target_col}). "
                    f"Available columns: {list(df.columns)}"
                )
                return TaskOutputs(metrics=metrics, pool_scores=pool_scores)

        pool_df = df[df["split"] == "pool"]
        if not pool_df.empty:
            pool_scores = dict(zip(pool_df["id"], pool_df[target_col]))

        return TaskOutputs(metrics=metrics, pool_scores=pool_scores)


    # =============================================================================
    # ヘルパーメソッド
    # =============================================================================

    # ★修正3: _write_jsonl をクラス内に移動 (AttributeError: '_write_jsonl' の修正)
    def _write_jsonl(self, path: Path, records: List[Any]) -> None:
        """リスト of dict/dataclass を JSONL 形式でファイルに書き出すヘルパー。"""
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                # to_dict メソッドがあればそれを使用し、なければ dict(r) を試みる
                d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


    def _clone_and_update_cfg(self, new_run=None, new_model=None, new_infer=None) -> Dict[str, Any]:
        """設定をディープコピーし、run/model/infer セクションを更新するヘルパー。"""
        cloned = copy.deepcopy(self.cfg)
        if new_run: cloned["run"].update(new_run)
        if new_model: cloned["model"].update(new_model)
        if new_infer: cloned["infer"].update(new_infer)
        return cloned


    def _cleanup_round_files(self) -> None:
        """
        ラウンド終了後に不要なファイル/ディレクトリを削除し、ディスク容量を節約する。
        run.cleanup_* の設定に応じて挙動を制御する。
        """
        run_cfg = self.cfg.get("run", {}) if isinstance(self.cfg.get("run", {}), Mapping) else {}

        cleanup_ckpt = bool(run_cfg.get("cleanup_round_checkpoints", False))
        cleanup_arrays = bool(run_cfg.get("cleanup_round_arrays", False))
        cleanup_temp = bool(run_cfg.get("cleanup_round_temp", False))
        cleanup_infer_npy = bool(run_cfg.get("cleanup_round_infer_npy", False))

        # 1. infer で生成された raw output (.npyファイル) を削除
        if cleanup_infer_npy and self._current_infer_out_dir and self._current_infer_out_dir.exists():
            for p in self._current_infer_out_dir.glob("*.npy"):
                try:
                    p.unlink()
                    LOGGER.debug(f"Cleanup: Removed infer npy file {p}")
                except OSError as e:
                    LOGGER.warning(f"Cleanup: Failed to remove infer npy {p}: {e}")

        # 2. train で生成されたモデルチェックポイント (.ptファイル) ディレクトリを削除
        if cleanup_ckpt and self._current_train_out_dir is not None:
            ckpt_dir = self._current_train_out_dir / "checkpoints_min"
            if ckpt_dir.exists():
                try:
                    shutil.rmtree(ckpt_dir)
                    LOGGER.debug(f"Cleanup: Removed checkpoint dir {ckpt_dir}")
                except OSError as e:
                    LOGGER.warning(f"Cleanup: Failed to remove checkpoint dir {ckpt_dir}: {e}")

        # 3. arrays/rounds/round_xxx_*.npy を削除
        if cleanup_arrays and self._current_round_index is not None:
            try:
                round_name = self.layout.round_name(self._current_round_index)
                arrays_dir = self.layout.arrays_rounds_dir
                if arrays_dir.exists():
                    prefix = f"{round_name}_"
                    for p in arrays_dir.glob(f"{prefix}*.npy"):
                        try:
                            p.unlink()
                            LOGGER.debug(f"Cleanup: Removed arrays npy file {p}")
                        except OSError as e:
                            LOGGER.warning(f"Cleanup: Failed to remove arrays npy {p}: {e}")
            except Exception as e:
                LOGGER.warning(f"Cleanup: arrays removal failed: {e}")

        # 4. temp_data/round_xxx/ を削除
        if cleanup_temp and self._current_round_index is not None:
            try:
                temp_dir = self.layout.path_temp_round_dir(self._current_round_index)
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    LOGGER.debug(f"Cleanup: Removed temp_data dir {temp_dir}")
            except OSError as e:
                LOGGER.warning(f"Cleanup: Failed to remove temp_data dir {temp_dir}: {e}")

        # 5. 追加で登録されたディレクトリも削除（サブクラス拡張用）
        for p in self._extra_cleanup_dirs:
            if p.exists():
                try:
                    shutil.rmtree(p)
                    LOGGER.debug(f"Cleanup: Removed extra dir {p}")
                except OSError as e:
                    LOGGER.warning(f"Cleanup: Failed to remove extra dir {p}: {e}")


    def _resolve_al_conf_key(self, hitl_out: Any) -> str:
        """
        AL（サンプリング）で使用する conf_key を決定する。

        優先順位:
          1) al.sampler.conf_key
          2) al.sampler.uncertainty_conf_key（旧名、後方互換）
          3) al.sampler.name（random/clustering/hybrid/uncertainty 以外ならそれを使用）
          4) hitl_out.conf_key (≒ gate.conf_key)
        """
        al_cfg = self.cfg.get("al", {})
        sampler_cfg = al_cfg.get("sampler", {}) if isinstance(al_cfg, Mapping) else {}

        sampler_name = str(sampler_cfg.get("name", "")).lower()
        conf_key = sampler_cfg.get("conf_key") or sampler_cfg.get("uncertainty_conf_key")

        special_names = {"", "random", "clustering", "hybrid", "uncertainty"}

        if not conf_key:
            if sampler_name and sampler_name not in special_names:
                # 例: name="trust" / "msp" / "entropy"
                conf_key = sampler_name
            else:
                # 最後の手段として HITL 側の conf_key を使う
                conf_key = getattr(hitl_out, "conf_key", "trust")

        return conf_key





    def _save_gate_assign(self, round_index: int, df: pd.DataFrame, hitl_out: Any) -> None:
        """
        hitl_out の結果 (mask_auto) をもとに、IDごとの判定結果 (auto/human) をCSVに保存する。
        devの結果 (GateDevResult) は mask_auto を持たないため、test/poolのみを保存する。
        """
        assignments = {}

        # devの結果は mask_auto を持たないため、testとpoolのみを処理
        for split_name in ["test", "pool"]: 
            
            # hitl_out は split 名の属性で GateApplyResult を持っている前提
            if not hasattr(hitl_out, split_name):
                continue
            
            res = getattr(hitl_out, split_name)
            if res is None:
                continue

            # df から該当 split の行を抽出（順序が hitl 実行時と同じであることを利用）
            sub_df = df[df["split"] == split_name]
            
            # 行数チェック (念のため)
            if len(sub_df) != len(res.mask_auto): 
                LOGGER.warning(f"Shape mismatch in {split_name}: df={len(sub_df)}, mask={len(res.mask_auto)}")
                continue
            
            # マスクに従って割り当てを決定 (True -> auto, False -> human)
            for pid, is_auto in zip(sub_df["id"], res.mask_auto):
                assignments[pid] = "auto" if is_auto else "human"

        # DataFrame 化して保存
        if assignments:
            assign_rows = [{"id": k, "gate_assignment": v} for k, v in assignments.items()]
            assign_df = pd.DataFrame(assign_rows)
            
            # 保存先パス (layout.py で path_predictions_round_gate_assign が定義されている前提)
            save_path = self.layout.path_predictions_round_gate_assign(round_index)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            assign_df.to_csv(save_path, index=False)
            LOGGER.info(f"Saved gate assignment to: {save_path}")
        else:
            LOGGER.warning("No gate assignments generated.")