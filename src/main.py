# /home/esakit25/work/tensaku/src/main.py
# -*- coding: utf-8 -*-
"""Hydra Entry Point (Strict Mode)

責務
- Hydra で Config を読み込み、トップレベル pipeline を呼び出す（薄いオーケストレータ）。
- 実行環境に依存するパス（out_dir/data_dir/input_all など）の絶対化をここで行う。
- Bootstrap（split）を保証する。
- 実験の FS ルート（ExperimentLayout）をここで確定し、pipeline に渡す。
- run_meta / index（実験台帳）をここで更新する。

境界
- 知ってよいこと: hydra 実行ディレクトリ/元CWD, cfg のパス注入, Bootstrapの呼び出し, 実験台帳更新
- 知らなくてよいこと: Split生成詳細, AL/HITLループ詳細, 学習/推論の中身
"""

from __future__ import annotations

import os
import json
import sys
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from tensaku.experiments.bootstrap import ensure_split_for_qid
from tensaku.experiments.index import append_experiment_index, utc_now_iso
from tensaku.experiments.layout import ExperimentLayout
from tensaku.pipelines.al import run_experiment
from tensaku.utils.strict_cfg import require_int, require_str


LOGGER = logging.getLogger(__name__)

# configs/ をHydraのconfig rootとして扱う（src/ からの相対）
CONFIG_PATH = "../configs"
CONFIG_NAME = "config"


def _abspath_from(base: str, p: str) -> str:
    if p is None:
        raise ValueError("path is None")
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base, p))



def _inject_model_num_labels_from_meta(cfg_py: Dict[str, Any], *, data_dir: Path, logger: logging.Logger) -> None:
    """Inject cfg_py['model']['num_labels'] from split meta.json (STRICT).

    - If cfg.model.num_labels is None/missing: set it from meta.label_stats.num_labels.
    - If cfg.model.num_labels is int: validate it matches meta.
    """
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"[main] meta.json not found under data_dir: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_stats = meta.get("label_stats")
    if not isinstance(label_stats, dict):
        raise RuntimeError("[main] meta.json missing 'label_stats' dict")

    n_raw = label_stats.get("num_labels")
    if n_raw is None:
        raise RuntimeError("[main] meta.json missing label_stats.num_labels")

    try:
        n = int(n_raw)
    except Exception as e:
        raise RuntimeError(f"[main] meta.label_stats.num_labels is not int-like: {n_raw!r}") from e

    if n <= 0:
        raise RuntimeError(f"[main] meta.label_stats.num_labels must be >0, got {n}")

    model_cfg = cfg_py.get("model")
    if not isinstance(model_cfg, dict):
        raise RuntimeError("[main] cfg.model must be a mapping (dict)")

    v = model_cfg.get("num_labels", None)
    if v is None:
        model_cfg["num_labels"] = n
        logger.info("[main] injected cfg.model.num_labels=%d from %s", n, meta_path)
        return

    try:
        v_int = int(v)
    except Exception as e:
        raise RuntimeError(f"[main] cfg.model.num_labels is not int-like: {v!r}") from e

    if v_int != n:
        raise RuntimeError(
            f"[main] cfg.model.num_labels={v_int} does not match split meta.label_stats.num_labels={n}. "
            "Fix config or regenerate split."
        )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()
    run_cwd = str(Path.cwd().resolve())

    # Strict: run/data は必須
    if "run" not in cfg or cfg.run is None:
        LOGGER.error("Config missing 'run' section.")
        sys.exit(1)
    if "data" not in cfg or cfg.data is None:
        LOGGER.error("Config missing 'data' section.")
        sys.exit(1)

    # --- out_dir/data_dir 注入 ---
    # Hydraは既に run dir に chdir 済み、という前提で
    # out_dir は run cwd を注入して良いが、data_dir は必ず明示してもらう。
    if getattr(cfg.run, "out_dir", None) in (None, ""):
        cfg.run.out_dir = run_cwd
    else:
        cfg.run.out_dir = _abspath_from(orig_cwd, cfg.run.out_dir)

    # Strict: data_dir は per-run ではなく、再利用したい split 位置を明示する。
    #   - cfg.run.data_dir があればそれを使う
    #   - 無い場合は cfg.data.base_dir を使う（base_dir が明示されている時のみ）
    if getattr(cfg.run, "data_dir", None) in (None, ""):
        if getattr(cfg.data, "base_dir", None) in (None, ""):
            LOGGER.error(
                "Config missing run.data_dir (Strict). "
                "Set run.data_dir to an existing split dir or a target dir."
            )
            sys.exit(1)
        cfg.run.data_dir = _abspath_from(orig_cwd, cfg.data.base_dir)
    else:
        cfg.run.data_dir = _abspath_from(orig_cwd, cfg.run.data_dir)

    # --- data 側のパス絶対化（project root基準） ---
    if getattr(cfg.data, "input_all", None) in (None, ""):
        LOGGER.error("Config missing data.input_all")
        sys.exit(1)
    cfg.data.input_all = _abspath_from(orig_cwd, cfg.data.input_all)

    # data.base_dir は optional（Strict: canonical split location は run.data_dir）
    if getattr(cfg.data, "base_dir", None) not in (None, ""):
        cfg.data.base_dir = _abspath_from(orig_cwd, cfg.data.base_dir)

    # Convert Hydra DictConfig -> plain dict (STRICT: throw_on_missing=True)
    cfg_py: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]

    # ----------------------------------------------------------------
    # Phase 1: Bootstrap Dataset Split (Strict)
    # ----------------------------------------------------------------
    LOGGER.info("--- [Phase 1] Bootstrapping Dataset Split ---")
    try:
        data_dir = ensure_split_for_qid(cfg_py, logger=LOGGER)
        cfg_py.setdefault("run", {})["data_dir"] = str(data_dir)
        _inject_model_num_labels_from_meta(cfg_py, data_dir=data_dir, logger=LOGGER)
    except Exception:
        LOGGER.exception("Bootstrap failed. Fix the config or check data consistency.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Phase 2: Create Layout + run_meta (Strict)
    # ----------------------------------------------------------------
    out_root = Path(require_str(cfg_py, ("run", "out_dir"), ctx="cfg.run"))
    out_root.mkdir(parents=True, exist_ok=True)
    layout = ExperimentLayout(root=out_root)

    started_at = utc_now_iso()
    status = "failed"
    exit_code = 1
    err_msg: Optional[str] = None

    qid = require_str(cfg_py, ("data", "qid"), ctx="cfg.data")
    tag = require_str(cfg_py, ("run", "tag"), ctx="cfg.run")
    sampler_name = require_str(cfg_py, ("al", "sampler", "name"), ctx="cfg.al.sampler")
    seed = require_int(cfg_py, ("run", "seed"), ctx="cfg.run")

    run_meta0: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "qid": qid,
        "tag": tag,
        "sampler": sampler_name,
        "seed": seed,
        "exp_dir": str(out_root.resolve()),
    }
    layout.save_json(layout.run_meta, run_meta0, record=True, kind="meta")

    # ----------------------------------------------------------------
    # Phase 3: Run Pipeline
    # ----------------------------------------------------------------
    LOGGER.info("Starting Tensaku Pipeline (Strict Mode)...")
    try:
        exit_code = int(run_experiment(cfg_py, layout=layout))
        status = "success" if exit_code == 0 else "failed"
        return_code = exit_code
    except Exception as e:
        status = "failed"
        return_code = 1
        err_msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        LOGGER.exception("Pipeline execution failed.")
    finally:
        ended_at = utc_now_iso()

        run_meta1 = dict(run_meta0)
        run_meta1.update({"status": status, "ended_at": ended_at, "exit_code": return_code})
        if err_msg is not None:
            run_meta1["error"] = err_msg

        # run_meta 更新
        try:
            layout.save_json(layout.run_meta, run_meta1, record=(not layout.run_meta.exists()), kind="meta")
        except Exception:
            LOGGER.exception("[main] failed to write run_meta")

        # index 更新（実験台帳）
        try:
            append_experiment_index(
                cfg=cfg_py,
                exp_dir=out_root,
                status=status,
                exit_code=return_code,
                started_at=started_at,
                ended_at=ended_at,
                error=err_msg,
                extra={"entrypoint": "main"},
            )
        except Exception:
            LOGGER.exception("[main] failed to append experiment index")

        if return_code != 0:
            sys.exit(return_code)


if __name__ == "__main__":
    main()
