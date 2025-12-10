import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
from pathlib import Path

# 既存のパイプラインをインポート
from tensaku.pipelines.al import run_experiment

LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Hydraによって起動されるメイン関数。
    """
    
    # ---------------------------------------------------------
    # 1. パスの解決 (Hydraのディレクトリ移動対策)
    # ---------------------------------------------------------
    orig_cwd = get_original_cwd()
    
    # YAML側で既に `${.qid}` が補間されている相対パスを取得
    # 例: "data_sas/splits/Y14_1-2_1_3"
    data_dir_relative = cfg.data.get("base_dir")

    # ★修正箇所: ここで qid を結合しない！
    abs_data_dir = Path(orig_cwd) / data_dir_relative
    
    LOGGER.info(f"Original CWD: {orig_cwd}")
    LOGGER.info(f"Resolved absolute data_dir: {abs_data_dir}")

    # ---------------------------------------------------------
    # 2. Configへの注入
    # ---------------------------------------------------------
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    if "run" not in cfg_dict:
        cfg_dict["run"] = {}
    
    # 絶対パスを渡す
    cfg_dict["run"]["data_dir"] = str(abs_data_dir)

    # ---------------------------------------------------------
    # 3. 実行
    # ---------------------------------------------------------
    LOGGER.info(f"Command executed in (Hydra output dir): {os.getcwd()}")

    try:
        # 修正済みの cfg_dict を渡す
        exit_code = run_experiment(cfg=cfg_dict, argv=None)
    except Exception as e:
        LOGGER.exception("Experiment failed with exception.")
        sys.exit(1)

    if exit_code != 0:
        sys.exit(exit_code)

if __name__ == "__main__":
    main()