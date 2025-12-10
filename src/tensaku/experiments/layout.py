# /home/esakit25/work/tensaku/src/tensaku/experiments/layout.py
# -*- coding: utf-8 -*-
"""
tensaku.experiments.layout
==========================

@module: tensaku.experiments.layout
@role  : AL/HITL 実験 1 本分の「出力ディレクトリ構成」を一元管理するレイアウトクラス

@overview:
    - OUT_DIR 以下のサブディレクトリ・ファイルパスを API 化して提供する。
    - パイプライン（pipelines.al / pipelines.hitl）はこのクラス経由でパスを取得し、
      各モジュール（train / infer / confidence / gate / hitl_report / al）が
      「どこに何を書くか」を意識せずに済むようにする。

@inputs:
    - cfg["run"]["out_dir"]: 実験出力ディレクトリ（絶対 or 相対パス）

@outputs (代表例):
    - config/exp_config.yaml
    - config/run_meta.json
    - logs/pipeline.log, logs/round_000.log, ...
    - metrics/al_history.csv, metrics/al_learning_curve.csv
    - metrics/hitl_summary_final.csv, metrics/hitl_summary_rounds.csv
    - selection/al_samples_all.csv,
      selection/rounds/round_000_al_sample.csv,
      selection/rounds/round_000_al_sample_ids.txt, ...
    - predictions/final/preds_detail.csv, predictions/final/gate_assign.csv
    - predictions/rounds/round_000_preds_detail.csv,
      predictions/rounds/round_000_gate_assign.csv, ...
    - arrays/rounds/round_000_probs.npy,
      arrays/rounds/round_000_logits.npy,
      arrays/rounds/round_000_emb_cls.npy, ...
    - plots/*.png（ファイル名は呼び出し側で決定）

@notes:
    - このモジュールは「ファイルシステム構成」だけを責務とし、AL や学習ロジックには依存しない。
    - OUT_DIR の実体ディレクトリ作成は ensure_all_dirs() で明示的に行う。
    - パスの命名規則はフェーズ2パイプラインの“公式インターフェース”として扱う前提。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ExperimentLayout:
    """1 実験 (= 1 つの AL/HITL パイプライン) に対応する出力ディレクトリ構成を管理するクラス。

    NOTE:
        - `root` は OUT_DIR に相当する。
        - サブディレクトリはプロパティで参照し、ensure_all_dirs() でまとめて作成する。
        - ラウンド番号は 0-origin を前提とし、ファイル名では "round_000" 形式に変換する。
    """

    root: Path

    # ------------------------------------------------------------------
    # コンストラクタ系
    # ------------------------------------------------------------------
    @staticmethod
    def _get_config_value(cfg: Mapping[str, Any], keys: str, default: str) -> str:
        """ネストされた設定値を取得するヘルパー。例: 'data.qid'"""
        current = cfg
        for key in keys.split('.'):
            if not isinstance(current, Mapping) or key not in current:
                return default
            current = current[key]
        return str(current)
    

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "ExperimentLayout":
        """設定 dict から ExperimentLayout を構築するユーティリティ。

        Expected:
            cfg["run"]["out_dir"]: str or Path
        """
        run_cfg = cfg.get("run", {})
        out_dir = run_cfg.get("out_dir")
        if out_dir is None:
            raise KeyError("cfg['run']['out_dir'] が指定されていません。")

        root = Path(out_dir).expanduser()
        return cls(root=root)

    # ------------------------------------------------------------------
    # ベースディレクトリ
    # ------------------------------------------------------------------
    @property
    def config_dir(self) -> Path:
        """設定・メタ情報を格納するディレクトリ。"""
        return self.root / "config"

    @property
    def logs_dir(self) -> Path:
        """ログファイルを格納するディレクトリ。"""
        return self.root / "logs"

    @property
    def metrics_dir(self) -> Path:
        """評価指標や履歴を格納するディレクトリ。"""
        return self.root / "metrics"

    @property
    def selection_dir(self) -> Path:
        """AL におけるサンプル選択情報を格納するディレクトリ。"""
        return self.root / "selection"
    
    @property
    def rounds_dir(self) -> Path:
        """ラウンドごとの出力を格納するディレクトリ。"""
        return self.root / "rounds"

    @property
    def selection_rounds_dir(self) -> Path:
        """ラウンドごとのサンプル選択情報を格納するディレクトリ。"""
        return self.selection_dir / "rounds"

    @property
    def predictions_dir(self) -> Path:
        """予測結果 (preds_detail / gate_assign 等) を格納するディレクトリ。"""
        return self.root / "predictions"

    @property
    def predictions_final_dir(self) -> Path:
        """最終ラウンドの予測結果を格納するディレクトリ。"""
        return self.predictions_dir / "final"

    @property
    def predictions_rounds_dir(self) -> Path:
        """ラウンドごとの予測結果を格納するディレクトリ。"""
        return self.predictions_dir / "rounds"

    @property
    def arrays_dir(self) -> Path:
        """NumPy 配列などの中間表現を格納するディレクトリ。"""
        return self.root / "arrays"

    @property
    def arrays_rounds_dir(self) -> Path:
        """ラウンドごとの配列ファイルを格納するディレクトリ。"""
        return self.arrays_dir / "rounds"

    @property
    def plots_dir(self) -> Path:
        """可視化（図表）を格納するディレクトリ。"""
        return self.root / "plots"
    
    @property
    def temp_data_dir(self) -> Path:
        """一時データ格納ルートディレクトリ。 (e.g., OUT_DIR/temp_data)"""
        return self.root / "temp_data"
    


    # ------------------------------------------------------------------
    # ディレクトリの一括作成
    # ------------------------------------------------------------------
    def ensure_all_dirs(self) -> None:
        """必要なサブディレクトリをまとめて作成する。"""
        for d in self._all_dirs():
            d.mkdir(parents=True, exist_ok=True)

    def _all_dirs(self) -> Iterable[Path]:
        """作成対象となる全ディレクトリのイテレータ。"""
        return (
            self.root,
            self.config_dir,
            self.logs_dir,
            self.metrics_dir,
            self.selection_dir,
            self.selection_rounds_dir,
            self.predictions_dir,
            self.predictions_final_dir,
            self.predictions_rounds_dir,
            self.arrays_dir,
            self.arrays_rounds_dir,
            self.plots_dir,
        )

    # ------------------------------------------------------------------
    # ヘルパ: ラウンド名
    # ------------------------------------------------------------------
    @staticmethod
    def round_name(round_index: int) -> str:
        """ラウンド番号から "round_000" 形式の文字列を生成する。

        Args:
            round_index: 0-origin のラウンド番号。

        Returns:
            例: round_index=0 -> "round_000"
        """
        if round_index < 0:
            raise ValueError(f"round_index must be >= 0 (got {round_index})")
        return f"round_{round_index:03d}"

    # ------------------------------------------------------------------
    # config / meta
    # ------------------------------------------------------------------
    def path_exp_config(self) -> Path:
        """有効設定 cfg を YAML として保存するパス。"""
        return self.config_dir / "exp_config.yaml"

    def path_run_meta(self) -> Path:
        """実行時メタ情報 (JSON) を保存するパス。"""
        return self.config_dir / "run_meta.json"

    # ------------------------------------------------------------------
    # logs
    # ------------------------------------------------------------------
    def path_log_pipeline(self) -> Path:
        """AL/HITL パイプライン全体のログファイルのパス。"""
        return self.logs_dir / "pipeline.log"

    def path_log_round(self, round_index: int) -> Path:
        """各ラウンド単位のログファイルのパス。"""
        return self.logs_dir / f"{self.round_name(round_index)}.log"
    

    # ------------------------------------------------------------------
    # rounds (学習・推論・ログのラウンド別出力)
    # ------------------------------------------------------------------
    
    def path_rounds_round_dir(self, round_index: int) -> Path:
        """ラウンドごとのルートディレクトリ。
        例: rounds/round_000/
        """
        return self.rounds_dir / self.round_name(round_index)

    def path_rounds_train_dir(self, round_index: int) -> Path:
        """学習結果の出力ディレクトリ。
        例: rounds/round_000/train/
        """
        return self.path_rounds_round_dir(round_index) / "train"

    def path_rounds_infer_dir(self, round_index: int) -> Path:
        """推論結果の出力ディレクトリ。
        例: rounds/round_000/infer/
        """
        return self.path_rounds_round_dir(round_index) / "infer"
    
    # ------------------------------------------------------------------
    # models
    def path_models_round_dir(self, round_index: int) -> Path:
        return self.path_rounds_train_dir(round_index)

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def path_metrics_al_history(self) -> Path:
        """AL ラウンドごとの履歴 (coverage など) を保存する CSV のパス。"""
        return self.metrics_dir / "al_history.csv"

    def path_metrics_al_learning_curve(self) -> Path:
        """学習曲線 (ラウンドごとの指標) を保存する CSV のパス。"""
        return self.metrics_dir / "al_learning_curve.csv"

    def path_metrics_hitl_summary_final(self) -> Path:
        """最終ラウンドの HITL 要約指標を保存する CSV のパス。"""
        return self.metrics_dir / "hitl_summary_final.csv"

    def path_metrics_hitl_summary_rounds(self) -> Path:
        """ラウンドごとの HITL 要約指標を保存する CSV のパス。"""
        return self.metrics_dir / "hitl_summary_rounds.csv"

    # ------------------------------------------------------------------
    # selection (Active Learning で選ばれたサンプル情報)
    # ------------------------------------------------------------------
    def path_selection_all_samples(self) -> Path:
        """全ラウンド分の AL サンプル選択履歴を集約した CSV のパス。"""
        return self.selection_dir / "al_samples_all.csv"

    def path_selection_round_samples(self, round_index: int) -> Path:
        """各ラウンドの AL サンプル選択 CSV のパス。"""
        return self.selection_rounds_dir / f"{self.round_name(round_index)}_al_sample.csv"

    def path_selection_round_sample_ids(self, round_index: int) -> Path:
        """各ラウンドの AL サンプル ID 一覧 (TXT) のパス。"""
        return self.selection_rounds_dir / f"{self.round_name(round_index)}_al_sample_ids.txt"

    # ------------------------------------------------------------------
    # predictions (preds_detail / gate_assign)
    # ------------------------------------------------------------------
    def path_predictions_final_detail(self) -> Path:
        """最終ラウンドの preds_detail（dev/pool/test を含む）CSV のパス。"""
        return self.predictions_final_dir / "preds_detail.csv"

    def path_predictions_final_gate_assign(self) -> Path:
        """最終ラウンドの gate_assign（auto/manual 割当）CSV のパス。"""
        return self.predictions_final_dir / "gate_assign.csv"

    def path_predictions_round_detail(self, round_index: int) -> Path:
        """各ラウンドの preds_detail CSV のパス。

        例: predictions/rounds/round_000_preds_detail.csv
        """
        return self.predictions_rounds_dir / f"{self.round_name(round_index)}_preds_detail.csv"

    def path_predictions_round_gate_assign(self, round_index: int) -> Path:
        """各ラウンドの gate_assign CSV のパス。

        例: predictions/rounds/round_000_gate_assign.csv
        """
        return self.predictions_rounds_dir / f"{self.round_name(round_index)}_gate_assign.csv"

    # ------------------------------------------------------------------
    # arrays (NumPy 配列など)
    # ------------------------------------------------------------------
    def path_arrays_round_probs(self, round_index: int) -> Path:
        """各ラウンドの予測確率 (probs) の NumPy 配列パス。"""
        return self.arrays_rounds_dir / f"{self.round_name(round_index)}_probs.npy"

    def path_arrays_round_logits(self, round_index: int) -> Path:
        """各ラウンドのロジット (logits) の NumPy 配列パス。"""
        return self.arrays_rounds_dir / f"{self.round_name(round_index)}_logits.npy"

    def path_arrays_round_emb_cls(self, round_index: int) -> Path:
        """各ラウンドの CLS 埋め込み (emb_cls) の NumPy 配列パス。"""
        return self.arrays_rounds_dir / f"{self.round_name(round_index)}_emb_cls.npy"

    def path_arrays_round_file(self, round_index: int, name: str) -> Path:
        """任意の名前を付けたいラウンド別 NumPy 配列ファイルのパス。

        例:
            path_arrays_round_file(0, "entropy.npy")
            -> arrays/rounds/round_000_entropy.npy
        """
        safe_name = name.lstrip("_")
        return self.arrays_rounds_dir / f"{self.round_name(round_index)}_{safe_name}"
    

    def path_arrays_round_pool_embs(self, round_index: int) -> Path:
        infer_dir = self.path_rounds_infer_dir(round_index)
        return infer_dir / "pool_embs.npy"

    def path_arrays_round_pool_preds(self, round_index: int) -> Path:
        infer_dir = self.path_rounds_infer_dir(round_index)
        return infer_dir / "pool_preds.csv"

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------
    def path_plot(self, filename: str) -> Path:
        """任意の可視化ファイル (PNG 等) を保存するパス。

        Args:
            filename: 例 "curve_coverage_rmse.png"
        """
        return self.plots_dir / filename
    
        # ------------------------------------------------------------------
    # temp data
    # ------------------------------------------------------------------
    def path_temp_round_dir(self, round_index: int) -> Path:
        """各ラウンドで利用するデータセットファイルを格納する一時ディレクトリ。

        例: temp_data/round_000/
        """
        return self.temp_data_dir / self.round_name(round_index)