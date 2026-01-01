# /home/esakit25/work/tensaku/src/tensaku/tasks/standard.py
# -*- coding: utf-8 -*-
"""tensaku.tasks.standard

@role: Task (Standard) = train -> infer(pool/dev/test) -> confidence -> HITL gate
@design:
  - Pipeline is a thin orchestrator; this Task owns "what to compute per round".
  - Sampler score key is decoupled from Gate score key(s).
  - Gate supports single metric or compare (multiple confidence keys) in one run_round.
  - No implicit fallback for required config keys (clear errors instead).
  - Supports simple runtime post-registration via .register().

Expected preds_detail columns (minimum):
  - id, split (one of labeled/dev/test/pool), y_true (optional for pool), y_pred
  - logits/probs/etc required by confidence estimators (depends on estimator)

Outputs:
  - predictions_round_detail (CSV)
  - predictions_round_gate_assign (CSV; long format: id, split, conf_key, tau, is_auto, is_human)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from pathlib import Path

import logging

import numpy as np
import pandas as pd

from tensaku.tasks.base import BaseTask, TaskOutputs
from tensaku.utils.strict_cfg import ConfigError, require_mapping, require_str, require_list, require_bool
from tensaku.utils.memlog import snapshot as mem_snapshot
from tensaku.contracts import validate_preds_detail, validate_gate_assign
import tensaku.registry as registory
from tensaku.pipelines.hitl import run_hitl_from_detail_df, build_gate_assign_df, create_empty_gate_assign_df

logger = logging.getLogger(__name__)


def _as_mapping(x: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise ConfigError(f"{where} must be a mapping/dict, got: {type(x)}")
    return x


def _resolve_sampler_conf_key(cfg: Mapping[str, Any]) -> Optional[str]:
    """Which confidence key the AL sampler should use (decoupled from gate).

    Rule (strict):
      - cfg['al']['sampler'] must be mapping and contain 'name'.
      - If sampler name is one of score-free samplers => None.
      - Else it must provide 'conf_key' (or 'by' as an alias).
    """
    al_cfg = require_mapping(cfg, "al")
    sampler_cfg = require_mapping(al_cfg, "sampler")

    name = require_str(sampler_cfg, "name")
    name_l = name.lower()

    # score-free samplers
    if name_l in {"random", "kmeans", "cluster", "clustering", "div_kmeans", "kcenter", "div_kcenter"}:
        return None

    # explicit score key
    if "conf_key" in sampler_cfg:
        return require_str(sampler_cfg, "conf_key")
    if "by" in sampler_cfg:
        # alias (explicit; not silent)
        return require_str(sampler_cfg, "by")

    # derived score key (explicit naming convention)
    # - legacy short names: msp/trust/entropy/margin/prob_margin/energy
    # - extended names: unc_<key>, u2d_* (uncertainty->diversity), d2u_* (diversity->uncertainty)
    #   NOTE: This is not a fallback; the key is encoded in the sampler *name* by contract.
    known = {"msp", "trust", "entropy", "margin", "prob_margin", "energy"}
    if name_l in known:
        return name_l
    if name_l.startswith("unc_"):
        key = name_l[len("unc_") :]
        if key in known:
            return key
    if name_l.startswith("u2d_"):
        # u2d_<unc>_<div>
        parts = name_l.split("_")
        if len(parts) >= 3 and parts[1] in known:
            return parts[1]
    if name_l.startswith("d2u_"):
        # d2u_<div>_<unc>
        parts = name_l.split("_")
        if len(parts) >= 3 and parts[-1] in known:
            return parts[-1]

    raise ConfigError(
        f"al.sampler.name='{name}' requires a score key. Set al.sampler.conf_key (e.g. 'msp', 'trust', ...), "
        f"or use a name that encodes the key (e.g. 'unc_msp'). (This is intentionally decoupled from gate.conf_key/conf_keys.)"
    )


def _resolve_conf_meta(cfg: Mapping[str, Any]) -> Dict[str, bool]:
    """
    confidenceセクションから {estimator_name: higher_is_better} のマップを作成する。
    Strict: higher_is_better は必須。
    """
    conf_cfg = require_mapping(cfg, "confidence")
    est_cfgs = require_list(conf_cfg, "estimators")
    
    meta = {}
    for item in est_cfgs:
        if not isinstance(item, Mapping):
             raise ConfigError("confidence.estimators must be a list of mappings.")
        name = require_str(item, "name")
        hib = require_bool(item, "higher_is_better")
        meta[name] = hib
    return meta


def _resolve_gate_targets(cfg: Mapping[str, Any]) -> Tuple[str, List[str]]:
    """
    Gateが使用するターゲット名リストを取得する。
    higher_is_better はここでは指定せず、confidence config 側の定義に従う。
    """
    gate_cfg = require_mapping(cfg, "gate")
    
    # 新しい形式: estimators (list of dict or list of str)
    if "estimators" in gate_cfg:
        raw_list = require_list(gate_cfg, "estimators")
        targets = []
        for item in raw_list:
            # - name: msp 形式の dict か、単なる文字列 "msp" を許容
            if isinstance(item, Mapping):
                name = require_str(item, "name")
                targets.append(name)
            elif isinstance(item, str):
                targets.append(item.strip())
            else:
                raise ConfigError("gate.estimators must be a list of mappings (with 'name') or strings.")
        
        mode = gate_cfg.get("mode", "compare")
        return mode, targets

    raise ConfigError("gate.estimators list is required.")


def _extract_pool_scores(df: pd.DataFrame, conf_key: str) -> Dict[Any, float]:
    col = f"conf_{conf_key}"
    if col not in df.columns:
        raise ConfigError(f"Required confidence column '{col}' not found in preds_detail.")
    pool_df = df[df["split"] == "pool"]
    if pool_df.empty:
        return {}
    # id can be int/str; keep as-is
    return {row["id"]: float(row[col]) for _, row in pool_df[["id", col]].iterrows()}


def _save_df_csv(layout: Any, file: Any, df: pd.DataFrame) -> None:
    """Save DataFrame using Layout.save_csv (ledger-aware) without assuming pandas I/O helpers."""
    header = list(df.columns)
    rows = df.itertuples(index=False, name=None)
    layout.save_csv(file, rows=rows, header=header)


class StandardSupervisedAlTask(BaseTask):
    """Standard AL task (supervised) with per-round train/infer/conf/gate.
    
    Adds simple in-memory registry for post-registration of hooks/callbacks.
    """

    def __init__(self, cfg: Mapping[str, Any], adapter: Any, layout: Any):
        self.cfg = cfg
        self.adapter = adapter
        self.layout = layout
        
        # --- Simple Registration Logic (In-Memory) ---
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, component: Any, overwrite: bool = False) -> None:
        """
        Dynamically register a component (e.g., a hook, callback, or extra logic).
        This allows 'post-registration' after the task is created.
        """
        if name in self._registry and not overwrite:
            raise ValueError(f"Component '{name}' is already registered. Use overwrite=True to replace.")
        
        self._registry[name] = component
        logger.info(f"[Task] Registered component: '{name}' (type: {type(component).__name__})")

    def get_component(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found in registry.")
        return self._registry[name]

    def _write_df(self, path: Any, df: pd.DataFrame) -> None:
        _save_df_csv(self.layout, path, df)

    def run_round(self, round_index: int, split: Any) -> TaskOutputs:
        logger.info("[Task] run_round=%s", round_index)
        mem_snapshot(event="round_start", extra={"round": round_index})

        # 0. Execute registered hooks (if any)
        # 登録されたコンポーネントに 'on_round_start' メソッドがあればここで実行します
        for name, comp in self._registry.items():
            if hasattr(comp, "on_round_start") and callable(comp.on_round_start):
                logger.info(f"[Task] Executing hook: {name}.on_round_start()")
                comp.on_round_start(round_index=round_index, split=split)

        # 1) train
        self.step_train(round_index=round_index, split=split)
        mem_snapshot(event="round_after_train", extra={"round": round_index})

        # 2) infer (pool/dev/test)
        df_detail, raw = self.step_infer(round_index=round_index, split=split)
        mem_snapshot(event="round_after_infer", extra={"round": round_index, "n_detail": (0 if df_detail is None else int(len(df_detail)))})
        validate_preds_detail(df_detail)

        # 3) confidence estimation
        df_detail = self.step_confidence(round_index=round_index, split=split, df=df_detail, raw=raw)
        mem_snapshot(event="round_after_conf", extra={"round": round_index})
        validate_preds_detail(df_detail)

        # save per-round preds_detail
        out_detail = self.layout.predictions_round_detail(round=round_index)
        _save_df_csv(self.layout, out_detail, df_detail)

        # 4) HITL gate
        out = self.step_hitl(round_index=round_index, split=split, df=df_detail)
        mem_snapshot(event="round_after_hitl", extra={"round": round_index})

        # 【追加】KMeans/Hybrid用にPoolの埋め込みを抽出してTaskOutputsに追加する
        # raw["pool"] が存在する場合のみ抽出
        pool_features = None
        pool_feature_ids = None
        if "pool" in raw:
            # rawの構造は {split: {embs: ..., ids: ...}}
            pool_raw = raw["pool"]
            if "embs" in pool_raw and "ids" in pool_raw:
                pool_features = pool_raw["embs"]
                pool_feature_ids = pool_raw["ids"]

        out = replace(out, pool_features=pool_features, pool_feature_ids=pool_feature_ids)

        mem_snapshot(event="round_end", extra={"round": round_index})

        return out

    # ----------------
    # Steps
    # ----------------

    def step_train(self, round_index: int, split: Any) -> None:
        from tensaku.train import train_core  # local import: keep task module light

        ckpt_dir = self.layout.round_ckpt_dir(round=round_index)

        rc, _ = train_core(split=split, ckpt_dir=ckpt_dir, cfg=self.cfg, return_model=False)
        if rc != 0:
            raise RuntimeError(f"train_core failed with rc={rc} (round={round_index})")

        if not ckpt_dir.exists():
            raise RuntimeError(f"Expected best checkpoint not found: {ckpt_dir.path}")

    def step_infer(self, round_index: int, split: Any):
        from tensaku.infer_pool import infer_pool_core  # local import

        ckpt_dir = self.layout.round_ckpt_dir(round=round_index)
        infer_dir = self.layout.round_infer_dir(round=round_index)

        # infer_core expects cfg['infer']['ckpt'] (Strict). Do not mutate shared cfg.
        infer_cfg = require_mapping(self.cfg, "infer")
        cfg_round = dict(self.cfg)
        cfg_round["infer"] = dict(infer_cfg)
        cfg_round["infer"]["ckpt"] = str(ckpt_dir.best.path)

        df_detail, raw = infer_pool_core(split=split, out_dir=infer_dir, cfg=cfg_round)
        return df_detail, raw
    def step_confidence(self, round_index: int, split: Any, df: pd.DataFrame, raw: Any) -> pd.DataFrame:
        # ここで単純に estimators を回すだけでなく、higher_is_better の定義チェックも兼ねる
        conf_cfg = require_mapping(self.cfg, "confidence")
        est_cfgs = require_list(conf_cfg, "estimators")
        
        if len(est_cfgs) == 0:
            raise ConfigError("confidence.estimators must be non-empty.")

        df2 = df.copy()

        for est_item in est_cfgs:
            est_item = _as_mapping(est_item, where="confidence.estimators[]")
            name = require_str(est_item, "name")
            
            # Strict: ここで higher_is_better があるかチェックしておく（使わないが、設定漏れを防ぐため）
            _ = require_bool(est_item, "higher_is_better")

            cfg_i = dict(est_item.get("cfg") or {})
            estimator = registory.create(name=name, cfg=cfg_i)

            df2 = self._apply_estimator(name=name, estimator=estimator, df=df2, raw=raw)

        return df2


    def step_hitl(self, round_index: int, split: Any, df: pd.DataFrame) -> TaskOutputs:
        gate_cfg = require_mapping(self.cfg, "gate")

        # 1) Confidence定義から {conf_key: higher_is_better} を取得
        conf_meta = _resolve_conf_meta(self.cfg)

        # 2) Gate設定からターゲット名を取得
        mode, gate_keys = _resolve_gate_targets(self.cfg)
        sampler_key = _resolve_sampler_conf_key(self.cfg)

        # eps_cse can be float or list[float] (sweep)
        eps_raw = gate_cfg.get("eps_cse", None)
        if isinstance(eps_raw, (list, tuple)):
            eps_list = [float(x) for x in eps_raw]
            if len(eps_list) == 0:
                raise ConfigError("gate.eps_cse list must be non-empty")
        else:
            eps_list = [float(eps_raw)]

        # 3) 必要なカラムの存在チェック
        required_cols = {"id", "split", "y_pred", "y_true"}  # y_true is required (pool may be NaN)
        required_cols |= {f"conf_{k}" for k in gate_keys}
        if sampler_key is not None:
            required_cols.add(f"conf_{sampler_key}")

        missing = [c for c in sorted(required_cols) if c not in df.columns]
        if missing:
            raise ConfigError(f"preds_detail missing required columns: {missing}")

        metrics: Dict[str, Any] = {}
        gate_assign_rows: List[pd.DataFrame] = []

        pool_ids = df.loc[df["split"] == "pool", "id"].to_list()

        def _eps_tag(eps: float) -> str:
            s = f"{eps:.6g}"
            return s.replace(".", "p")

        expanded_conf_keys: List[str] = []

        # 4) Gate 実行ループ
        first_written = False
        for conf_key in gate_keys:
            if conf_key not in conf_meta:
                raise ConfigError(f"Gate target '{conf_key}' is not defined in confidence.estimators.")

            # Confidence側で定義された higher_is_better を使う
            higher_is_better = bool(conf_meta[conf_key])
            col = f"conf_{conf_key}"

            for eps in eps_list:
                # keep gate_assign schema stable by encoding eps into conf_key
                conf_key_tag = conf_key if len(eps_list) == 1 else f"{conf_key}@eps{_eps_tag(eps)}"
                expanded_conf_keys.append(conf_key_tag)

                # GateConfig 用に辞書を構築
                gate_cfg_i = dict(gate_cfg)
                gate_cfg_i["higher_is_better"] = higher_is_better
                gate_cfg_i["eps_cse"] = float(eps)

                # Run HITL
                hitl_out = run_hitl_from_detail_df(df=df, gate_cfg=gate_cfg_i, conf_column_name=col)

                # metrics
                metrics[f"gate.{conf_key_tag}.eps_cse"] = float(eps)
                metrics.update(hitl_out.to_summary_dict(prefix=f"gate.{conf_key_tag}."))

                # Backward compat: first result also under gate.*
                if not first_written:
                    metrics.update(hitl_out.to_summary_dict(prefix="gate."))
                    first_written = True

                # gate assignment (pool)
                if pool_ids:
                    gate_assign_rows.append(
                        build_gate_assign_df(pool_ids=pool_ids, conf_key=conf_key_tag, hitl_out=hitl_out)
                    )
                else:
                    gate_assign_rows.append(create_empty_gate_assign_df())

        # gate meta
        metrics["gate.mode"] = mode
        metrics["gate.conf_keys"] = gate_keys
        metrics["gate.conf_keys_expanded"] = expanded_conf_keys
        metrics["gate.eps_cse"] = eps_list[0] if len(eps_list) == 1 else None
        metrics["gate.eps_cse_list"] = eps_list if len(eps_list) > 1 else None

        # write gate_assign
        if gate_assign_rows:
            gate_assign_df = pd.concat(gate_assign_rows, ignore_index=True)
            validate_gate_assign(gate_assign_df)
            self._write_df(path=self.layout.predictions_round_gate_assign(round=round_index), df=gate_assign_df)

        # sampler scores
        pool_scores = _extract_pool_scores(df=df, conf_key=sampler_key) if sampler_key is not None else {}



        return TaskOutputs(metrics=metrics, pool_scores=pool_scores, detail_df=df)
    def _apply_estimator(self, name: str, estimator: Any, df: pd.DataFrame, raw: Any) -> pd.DataFrame:
        """Apply an estimator and append one or more conf_* columns to df.

        Supported estimator return types:
          - np.ndarray / list / pd.Series: single score vector -> column conf_{name}
          - dict[str, vector]: multiple columns -> each saved as conf_{key} (unless already startswith 'conf_')
          - pd.DataFrame: merged by index; columns are added as-is (and also prefixed if needed)
        """
        # try common call patterns
        try:
            out = estimator.apply(df=df, raw=raw)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                out = estimator.compute(df=df, raw=raw)  # type: ignore[attr-defined]
            except AttributeError:
                try:
                    out = estimator(df=df, raw=raw)
                except TypeError:
                    out = estimator(df, raw)

        df2 = df.copy()

        def _add_col(col_name: str, vec: Any) -> None:
            arr = np.asarray(vec)
            if arr.ndim != 1 or len(arr) != len(df2):
                raise RuntimeError(f"Estimator '{name}' returned invalid shape for {col_name}: {arr.shape}")
            df2[col_name] = arr.astype(float)

        if isinstance(out, pd.DataFrame):
            for c in out.columns:
                # if it is a bare key (msp/trust) we still normalize to conf_*
                if c.startswith("conf_"):
                    df2[c] = out[c].to_numpy()
                else:
                    df2[f"conf_{c}"] = out[c].to_numpy()
            return df2

        if isinstance(out, Mapping):
            for k, v in out.items():
                k = str(k)
                col = k if k.startswith("conf_") else f"conf_{k}"
                _add_col(col, v)
            return df2

        # single vector
        _add_col(f"conf_{name}", out)
        return df2