# /home/esakit25/work/tensaku/src/main.py
# -*- coding: utf-8 -*-
"""Hydra Entry Point (Strict Mode)

このモジュールの責務
- Hydra で Config を読み込み、トップレベル pipeline を呼び出す。
- 相対パスの絶対化など「実行環境に依存する注入」をここで完結させる。
- [NEW] Bootstrap: データセットの物理的整合性（Split）を保証する。
※ Strict 方針: ここでの動的注入は行わない（サイレントな上書きはしない）。

境界
- 知ってよいこと: hydra 実行ディレクトリ/元CWD, cfg のパス注入, Bootstrapの呼び出し
- 知らなくてよいこと: Splitの詳細生成ロジック, AL/HITL のループ実装詳細
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

from omegaconf import OmegaConf

import hydra
from omegaconf import DictConfig

from tensaku.pipelines.al import run_experiment
# [NEW] Bootstrap用の関数をインポート
from tensaku.experiments.bootstrap import ensure_split_for_qid

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
            LOGGER.error("Config missing run.data_dir (Strict). Set run.data_dir to an existing split dir or a target dir.")
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

    # ----------------------------------------------------------------
    # [NEW] Phase 1: Bootstrap Dataset Split
    # パス解決が終わった状態で、データセット（Split）の物理的な整合性を保証する
    # ----------------------------------------------------------------
    LOGGER.info("--- [Phase 1] Bootstrapping Dataset Split ---")

    # Convert Hydra DictConfig -> plain dict (STRICT: throw_on_missing=True)
    cfg_py = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    try:
        ensure_split_for_qid(cfg, logger=LOGGER)
    except Exception:
        LOGGER.exception("Bootstrap failed. Fix the config or check data consistency.")
        sys.exit(1)



    # ----------------------------------------------------------------
    # 3. Run Pipeline
    # ----------------------------------------------------------------
    LOGGER.info("Starting Tensaku Pipeline (Strict Mode)...")
    try:
        # 修正された cfg が渡される
        exit_code = run_experiment(cfg_py)
    except Exception:
        LOGGER.exception("Pipeline execution failed.")
        sys.exit(1)

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()