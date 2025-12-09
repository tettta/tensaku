# /home/esakit25/work/tensaku/src/tensaku/tasks/sas_total_score.py (修正箇所のみ抜粋)
# -*- coding: utf-8 -*
from __future__ import annotations
import logging
from typing import Any, Mapping
# 必要なインポートはすべて base.py/standard.py で済ませる
# ↓基底クラスのインポートを変更
from tensaku.tasks.standard import StandardSupervisedAlTask
# 不要になったローカルインポートを削除 (train_core, infer_core, run_hitl_from_detail_df)

LOGGER = logging.getLogger(__name__)

# TrainInferHitlTask ではなく、StandardSupervisedAlTask を継承する
class SasTotalScoreTask(StandardSupervisedAlTask):
    """
    SASスコア予測タスクを実行するTaskクラス。
    標準的な教師あり学習ALフローは StandardSupervisedAlTask に委譲する。
    """
    
    name: str = "sas_total_score"
    
    # __init__ も不要 (StandardSupervisedAlTask のものが利用可能)
    # _current_preds_path の管理も StandardSupervisedAlTask に移動

    # step_* メソッドはすべて StandardSupervisedAlTask の実装をそのまま使うため、オーバーライドを削除

    # HACK: 現状は属性宣言のみ残す (将来的には不要になるはずだが、一旦保険)
    # def __init__(self, cfg: Mapping[str, Any], adapter: Any, layout: Any) -> None:
    #     super().__init__(cfg, adapter, layout)
    #     self._current_data_dir: Optional[Path] = None
    #     self._current_train_out_dir: Optional[Path] = None
    #     self._current_preds_path: Optional[Path] = None

    pass # 全てのロジックを StandardSupervisedAlTask に委譲